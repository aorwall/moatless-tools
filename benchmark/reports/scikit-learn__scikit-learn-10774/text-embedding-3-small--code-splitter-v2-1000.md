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
1 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
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
2 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
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
3 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
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


def load_sample_images():
    """Load sample images for image manipulation.

    Loads both, ``china`` and ``flower``.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes : 'images', the
        two sample images, 'filenames', the file names for the images, and
        'DESCR' the full description of the dataset.

    Examples
    --------
    To load the data and visualize the images:

    >>> from sklearn.datasets import load_sample_images
    >>> dataset = load_sample_images()     #doctest: +SKIP
    >>> len(dataset.images)                #doctest: +SKIP
    2
    >>> first_img_data = dataset.images[0] #doctest: +SKIP
    >>> first_img_data.shape               #doctest: +SKIP
    (427, 640, 3)
    >>> first_img_data.dtype               #doctest: +SKIP
    dtype('uint8')
    """
    # import PIL only when needed
    from ..externals._pilutil import imread

    module_path = join(dirname(__file__), "images")
    with open(join(module_path, 'README.txt')) as f:
        descr = f.read()
    filenames = [join(module_path, filename)
                 for filename in os.listdir(module_path)
                 if filename.endswith(".jpg")]
    # Load image data for each image in the source folder.
    images = [imread(filename) for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames,
                 DESCR=descr)



```
4 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
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
5 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
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
6 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
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
7 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
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
    """
    data_home = get_data_home(data_home=data_home)
    data_home = join(data_home, 'openml')
    if cache is False:
        # no caching will be applied
        data_home = None

    # check valid function arguments. data_id XOR (name, version) should be
    # provided
    if name is not None:
        # OpenML is case-insensitive, but the caching mechanism is not
        # convert all data names (str) to lower case
        name = name.lower()
        if data_id is not None:
            raise ValueError(
                "Dataset data_id={} and name={} passed, but you can only "
                "specify a numeric data_id or a name, not "
                "both.".format(data_id, name))
        data_info = _get_data_info_by_name(name, version, data_home)
        data_id = data_info['did']
    elif data_id is not None:
        # from the previous if statement, it is given that name is None
        if version != "active":
            raise ValueError(
                "Dataset data_id={} and version={} passed, but you can only "
                "specify a numeric data_id or a version, not "
                "both.".format(data_id, name))
    else:
        raise ValueError(
            "Neither name nor data_id are provided. Please provide name or "
            "data_id.")

    data_description = _get_data_description_by_id(data_id, data_home)
    if data_description['status'] != "active":
        warn("Version {} of dataset {} is inactive, meaning that issues have "
             "been found in the dataset. Try using a newer version from "
             "this URL: {}".format(
                data_description['version'],
                data_description['name'],
                data_description['url']))
    if 'error' in data_description:
        warn("OpenML registered a problem with the dataset. It might be "
             "unusable. Error: {}".format(data_description['error']))
    if 'warning' in data_description:
        warn("OpenML raised a warning on the dataset. It might be "
             "unusable. Warning: {}".format(data_description['warning']))

    # download data features, meta-info about column types
    features_list = _get_data_features(data_id, data_home)

    for feature in features_list:
        if 'true' in (feature['is_ignore'], feature['is_row_identifier']):
            continue
        if feature['data_type'] == 'string':
            raise ValueError('STRING attributes are not yet supported')

    if target_column == "default-target":
        # determines the default target based on the data feature results
        # (which is currently more reliable than the data description;
        # see issue: https://github.com/openml/OpenML/issues/768)
        target_column = [feature['name'] for feature in features_list
                         if feature['is_target'] == 'true']
    elif isinstance(target_column, str):
        # for code-simplicity, make target_column by default a list
        target_column = [target_column]
    elif target_column is None:
        target_column = []
    elif not isinstance(target_column, list):
        raise TypeError("Did not recognize type of target_column"
                        "Should be str, list or None. Got: "
                        "{}".format(type(target_column)))
    data_columns = _valid_data_column_names(features_list,
                                            target_column)

    # prepare which columns and data types should be returned for the X and y
    features_dict = {feature['name']: feature for feature in features_list}

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
    
```
**8 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
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
**9 - /tmp/repos/scikit-learn/sklearn/datasets/covtype.py**:
```python
def fetch_covtype(data_home=None, download_if_missing=True,
                  random_state=None, shuffle=False, return_X_y=False):
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
10 - /tmp/repos/scikit-learn/sklearn/feature_extraction/dict_vectorizer.py
```python
def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        self
        """
        feature_names = []
        vocab = {}

        for x in X:
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                if f not in vocab:
                    feature_names.append(f)
                    vocab[f] = len(vocab)

        if self.sort:
            feature_names.sort()
            vocab = dict((f, i) for i, f in enumerate(feature_names))

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        return self

    def _transform(self, X, fitting):
        # Sanity check: Python's array has no way of explicitly requesting the
        # signed 32-bit integers that scipy.sparse needs, so we use the next
        # best thing: typecode "i" (int). However, if that gives larger or
        # smaller integers than 32-bit ones, np.frombuffer screws up.
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug report")

        dtype = self.dtype
        if fitting:
            feature_names = []
            vocab = {}
        else:
            feature_names = self.feature_names_
            vocab = self.vocabulary_

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        # collect all the possible feature names and build sparse matrix at
        # same time
        for x in X:
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                if f in vocab:
                    indices.append(vocab[f])
                    values.append(dtype(v))
                else:
                    if fitting:
                        feature_names.append(f)
                        vocab[f] = len(vocab)
                        indices.append(vocab[f])
                        values.append(dtype(v))

            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        # Sort everything if asked
        if fitting and self.sort:
            feature_names.sort()
            map_index = np.empty(len(feature_names), dtype=np.int32)
            for new_val, f in enumerate(feature_names):
                map_index[new_val] = vocab[f]
                vocab[f] = new_val
            result_matrix = result_matrix[:, map_index]

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        if fitting:
            self.feature_names_ = feature_names
            self.vocabulary_ = vocab

        return result_matrix

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
11 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
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


def dump_svmlight_file(X, y, f,  zero_based=True, comment=None, query_id=None,
                       multilabel=False):
    
```
12 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
def _get_small_trainset(self, X_binned_train, y_train, seed):
        """Compute the indices of the subsample set and return this set.

        For efficiency, we need to subsample the training set to compute scores
        with scorers.
        """
        subsample_size = 10000
        rng = check_random_state(seed)
        indices = np.arange(X_binned_train.shape[0])
        if X_binned_train.shape[0] > subsample_size:
            # TODO: not critical but stratify using resample()
            indices = rng.choice(indices, subsample_size, replace=False)
        X_binned_small_train = X_binned_train[indices]
        y_small_train = y_train[indices]
        X_binned_small_train = np.ascontiguousarray(X_binned_small_train)
        return X_binned_small_train, y_small_train

    def _check_early_stopping_scorer(self, X_binned_small_train, y_small_train,
                                     X_binned_val, y_val):
        """Check if fitting should be early-stopped based on scorer.

        Scores are computed on validation data or on training data.
        """
        self.train_score_.append(
            self.scorer_(self, X_binned_small_train, y_small_train)
        )

        if self._use_validation_data:
            self.validation_score_.append(
                self.scorer_(self, X_binned_val, y_val)
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)

    def _check_early_stopping_loss(self,
                                   raw_predictions,
                                   y_train,
                                   raw_predictions_val,
                                   y_val):
        """Check if fitting should be early-stopped based on loss.

        Scores are computed on validation data or on training data.
        """

        self.train_score_.append(
            -self.loss_(y_train, raw_predictions)
        )

        if self._use_validation_data:
            self.validation_score_.append(
                -self.loss_(y_val, raw_predictions_val)
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)

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
13 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
def cross_val_predict(estimator, X, y=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """Generate cross-validated estimates for each input data point

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :func:`sklearn.model_selection.cross_val_predict` instead.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

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

    Returns
    -------
    preds : ndarray
        This is the result of calling 'predict'

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.cross_validation import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y)
    """
    X, y = indexable(X, y)

    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    preds_blocks = parallel(delayed(_fit_and_predict)(clone(estimator), X, y,
                                                      train, test, verbose,
                                                      fit_params)
                            for train, test in cv)

    preds = [p for p, _ in preds_blocks]
    locs = np.concatenate([loc for _, loc in preds_blocks])
    if not _check_is_partition(locs, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')
    inv_locs = np.empty(len(locs), dtype=int)
    inv_locs[locs] = np.arange(len(locs))

    # Check for sparse predictions
    if sp.issparse(preds[0]):
        preds = sp.vstack(preds, format=preds[0].format)
    else:
        preds = np.concatenate(preds)
    return preds[inv_locs]



```
14 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
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

    
```
15 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
"""This module implements a loader and dumper for the svmlight format

This format is a text-based format, with one sample per line. It does
not store zero valued features hence is suitable for sparse dataset.

The first element of each line can be used to store a target variable to
predict.

This format is used as the default format for both svmlight and the
libsvm command line programs.
"""

# Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck
#          Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from contextlib import closing
import io
import os.path

import numpy as np
import scipy.sparse as sp

from ._svmlight_format import _load_svmlight_file
from .. import __version__
from ..externals import six
from ..externals.six import u, b
from ..externals.six.moves import range, zip
from ..utils import check_array


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
16 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array


def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    
```
17 - /tmp/repos/scikit-learn/sklearn/manifold/mds.py
```python
def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from X, and returns the embedded coordinates

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y: Ignored

        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        X = check_array(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_, metric=self.metric,
            n_components=self.n_components, init=init, n_init=self.n_init,
            n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
            eps=self.eps, random_state=self.random_state,
            return_n_iter=True)

        return self.embedding_

```
18 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
                                                         n_samples)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state,
                                                self.presort)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth,
                                            self.min_impurity_decrease,
                                            min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

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

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # Classification
        if is_classifier(self):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    
```
**19 - /tmp/repos/scikit-learn/sklearn/datasets/rcv1.py**:
```python
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
    """inverse permutation p"""
    n = p.size
    s = np.zeros(n, dtype=np.int32)
    i = np.arange(n, dtype=np.int32)
    np.put(s, p, i)  # s[p] = i
    return s


def _find_permutation(a, b):
    """find the permutation from a to b"""
    t = np.argsort(a)
    u = np.argsort(b)
    u_ = _inverse_permutation(u)
    return t[u_]

```
20 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
def _make_test_folds(self, X, y=None):
        rng = self.random_state
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        n_samples = y.shape[0]
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of members in any class cannot"
                           " be less than n_splits=%d."
                           % (min_groups, self.n_splits)), Warning)

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each class so as to respect the balance of
        # classes
        # NOTE: Passing the data corresponding to ith class say X[y==class_i]
        # will break when the data is not 100% stratifiable for all classes.
        # So we pass np.zeroes(max(c, n_splits)) as data to the KFold
        per_cls_cvs = [
            KFold(self.n_splits, shuffle=self.shuffle,
                  random_state=rng).split(np.zeros(max(count, self.n_splits)))
            for count in y_counts]

        test_folds = np.zeros(n_samples, dtype=np.int)
        for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
            for cls, (_, test_split) in zip(unique_y, per_cls_splits):
                cls_test_folds = test_folds[y == cls]
                # the test split can be too big because we used
                # KFold(...).split(X[:max(c, n_splits)]) when data is not 100%
                # stratifiable for all the classes
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(cls_test_folds)]
                cls_test_folds[test_split] = test_fold_indices
                test_folds[y == cls] = cls_test_folds

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedKFold, self).split(X, y, groups)



```
21 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
# -*- coding: utf-8 -*-

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
22 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels."""
    if hasattr(estimator, 'kernel') and callable(estimator.kernel) \
       and not isinstance(estimator.kernel, GPKernel):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_subset = [X[idx] for idx in indices]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            if train_indices is None:
                X_subset = X[np.ix_(indices, indices)]
            else:
                X_subset = X[np.ix_(indices, train_indices)]
        else:
            X_subset = safe_indexing(X, indices)

    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


def _score(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set."""
    if y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)
    if hasattr(score, 'item'):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    if not isinstance(score, numbers.Number):
        raise ValueError("scoring must return a number, got %s (%s) instead."
                         % (str(score), type(score)))
    return score


def _permutation_test_score(estimator, X, y, cv, scorer):
    """Auxiliary function for permutation_test_score"""
    avg_score = []
    for train, test in cv:
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        estimator.fit(X_train, y_train)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)


def _shuffle(y, labels, random_state):
    """Return a shuffled copy of y eventually shuffle among same labels."""
    if labels is None:
        ind = random_state.permutation(len(y))
    else:
        ind = np.arange(len(labels))
        for label in np.unique(labels):
            this_mask = (labels == label)
            ind[this_mask] = random_state.permutation(ind[this_mask])
    return safe_indexing(y, ind)



```
23 - /tmp/repos/scikit-learn/sklearn/gaussian_process/kernels.py
```python
def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='euclidean')
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='euclidean')

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2 \
                    / (length_scale ** 2)
            else:
                D = squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                K_gradient = K[..., np.newaxis] * D \
                    / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K_gradient[~np.isfinite(K_gradient)] = 0
            elif self.nu == 1.5:
                K_gradient = \
                    3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    
```
24 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
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



```
25 - /tmp/repos/scikit-learn/sklearn/manifold/spectral_embedding_.py
```python
def _get_affinity_matrix(self, X, Y=None):
        """Calculate the affinity matrix from data
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Y: Ignored

        Returns
        -------
        affinity_matrix, shape (n_samples, n_samples)
        """
        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
            return self.affinity_matrix_
        if self.affinity == 'nearest_neighbors':
            if sparse.issparse(X):
                warnings.warn("Nearest neighbors affinity currently does "
                              "not support sparse input, falling back to "
                              "rbf affinity")
                self.affinity = "rbf"
            else:
                self.n_neighbors_ = (self.n_neighbors
                                     if self.n_neighbors is not None
                                     else max(int(X.shape[0] / 10), 1))
                self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_,
                                                         include_self=True,
                                                         n_jobs=self.n_jobs)
                # currently only symmetric affinity_matrix supported
                self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ +
                                               self.affinity_matrix_.T)
                return self.affinity_matrix_
        if self.affinity == 'rbf':
            self.gamma_ = (self.gamma
                           if self.gamma is not None else 1.0 / X.shape[1])
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Y: Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = check_array(X, ensure_min_samples=2, estimator=self)

        random_state = check_random_state(self.random_state)
        if isinstance(self.affinity, six.string_types):
            if self.affinity not in set(("nearest_neighbors", "rbf",
                                         "precomputed")):
                raise ValueError(("%s is not a valid affinity. Expected "
                                  "'precomputed', 'rbf', 'nearest_neighbors' "
                                  "or a callable.") % self.affinity)
        elif not callable(self.affinity):
            raise ValueError(("'affinity' is expected to be an affinity "
                              "name or a callable. Got: %s") % self.affinity)

        affinity_matrix = self._get_affinity_matrix(X)
        self.embedding_ = spectral_embedding(affinity_matrix,
                                             n_components=self.n_components,
                                             eigen_solver=self.eigen_solver,
                                             random_state=random_state)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Y: Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_

```
26 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size,
                                                  self.train_size)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([' '.join(row.astype('str')) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of groups for any class cannot"
                             " be less than 2.")

        if n_train < n_classes:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, n_classes))
        if n_test < n_classes:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_test, n_classes))

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                 np.cumsum(class_counts)[:-1])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation,
                                                             mode='clip')

                train.extend(perm_indices_class_i[:n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedShuffleSplit, self).split(X, y, groups)



```
27 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
if hasattr(f, "write"):
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        with open(f, "wb") as f:
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)

```
28 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def _transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape

        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (indices.shape[0] - 1, n_features))

        # We use only those categorical features of X that are known using fit.
        # i.e lesser than n_values_ using mask.
        # This means, if self.handle_unknown is "ignore", the row_indices and
        # col_indices corresponding to the unknown categorical feature are
        # ignored.
        mask = (X < self.n_values_).ravel()
        if np.any(~mask):
            if self.handle_unknown not in ['error', 'ignore']:
                raise ValueError("handle_unknown should be either error or "
                                 "unknown got %s" % self.handle_unknown)
            if self.handle_unknown == 'error':
                raise ValueError("unknown categorical feature present %s "
                                 "during transform." % X.ravel()[~mask])

        column_indices = (X + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(np.sum(mask))
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if (isinstance(self.n_values, six.string_types) and
                self.n_values == 'auto'):
            out = out[:, self.active_features_]

        return out if self.sparse else out.toarray()

    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input array of type int.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)


class QuantileTransformer(BaseEstimator, TransformerMixin):
    
```
29 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py
```python
def fit_transform(self, X, y=None):
        """Fit the model and recover the sources from X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return self._fit(X, compute_sources=True)

    def fit(self, X, y=None):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self
        """
        self._fit(X, compute_sources=False)
        return self

    def transform(self, X, y='deprecated', copy=True):
        """Recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform, where n_samples is the number of samples
            and n_features is the number of features.
        y : (ignored)
            .. deprecated:: 0.19
               This parameter will be removed in 0.21.
        copy : bool (optional)
            If False, data passed to fit are overwritten. Defaults to True.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        check_is_fitted(self, 'mixing_')

        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        if self.whiten:
            X -= self.mean_

        return np.dot(X, self.components_.T)

    def inverse_transform(self, X, copy=True):
        """Transform the sources back to the mixed data (apply mixing matrix).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Sources, where n_samples is the number of samples
            and n_components is the number of components.
        copy : bool (optional)
            If False, data passed to fit are overwritten. Defaults to True.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
        """
        check_is_fitted(self, 'mixing_')

        X = check_array(X, copy=(copy and self.whiten), dtype=FLOAT_DTYPES)
        X = np.dot(X, self.mixing_.T)
        if self.whiten:
            X += self.mean_

        return X

```
30 - /tmp/repos/scikit-learn/sklearn/gaussian_process/kernels.py
```python


    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_periodicity(self):
        return Hyperparameter(
            "periodicity", "numeric", self.periodicity_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric='euclidean'))
            arg = np.pi * dists / self.periodicity
            sin_of_arg = np.sin(arg)
            K = np.exp(- 2 * (sin_of_arg / self.length_scale) ** 2)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='euclidean')
            K = np.exp(- 2 * (np.sin(np.pi / self.periodicity * dists)
                              / self.length_scale) ** 2)

        if eval_gradient:
            cos_of_arg = np.cos(arg)
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = \
                    4 / self.length_scale**2 * sin_of_arg**2 * K
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # length_scale is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
            # gradient with respect to p
            if not self.hyperparameter_periodicity.fixed:
                periodicity_gradient = \
                    4 * arg / self.length_scale**2 * cos_of_arg \
                    * sin_of_arg * K
                periodicity_gradient = periodicity_gradient[:, :, np.newaxis]
            else:  # p is kept fixed
                periodicity_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((length_scale_gradient, periodicity_gradient))
        else:
            return K

    def __repr__(self):
        return "{0}(length_scale={1:.3g}, periodicity={2:.3g})".format(
            self.__class__.__name__, self.length_scale, self.periodicity)



```
31 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
"""Load dataset from multiple files in SVMlight format

    This function is equivalent to mapping load_svmlight_file over a list of
    files, except that the results are concatenated into a single, flat list
    and the samples vectors are constrained to all have the same number of
    features.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    Parameters
    ----------
    files : iterable over {str, file-like, int}
        (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. File-likes and file descriptors will not be
        closed by this function. File-like objects must be opened in binary
        mode.

    n_features : int or None
        The number of features to use. If None, it will be inferred from the
        maximum column index occurring in any of the files.

        This can be set to a higher value than the actual number of features
        in any of the input files, but setting it to a lower value will cause
        an exception to be raised.

    dtype : numpy data type, default np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : boolean, optional
        Samples may have several labels each (see
        http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

    zero_based : boolean or "auto", optional
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no offset or length is passed.
        If offset or length are passed, the "auto" mode falls back
        to zero_based=True to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : boolean, defaults to False
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
    [X1, y1, ..., Xn, yn]
    where each (Xi, yi) pair is the result from load_svmlight_file(files[i]).

    If query_id is set to True, this will return instead [X1, y1, q1,
    ..., Xn, yn, qn] where (Xi, yi, qi) is the result from
    load_svmlight_file(files[i])

    Notes
    -----
    When fitting a model to a matrix X_train and evaluating it against a
    matrix X_test, it is essential that X_train and X_test have the same
    number of features (X_train.shape[1] == X_test.shape[1]). This may not
    be the case if you load the files individually with load_svmlight_file.

    See also
    --------
    load_svmlight_file
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

    
```
32 - /tmp/repos/scikit-learn/sklearn/decomposition/incremental_pca.py
```python
def partial_fit(self, X, y=None, check_input=True):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        check_input : bool
            Run check_array on X.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if check_input:
            X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to "
                             "the batch number of samples "
                             "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0] !=
                                               self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self.components_.shape[0], self.n_components_))

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.mean_ = .0
            self.var_ = .0

        # Update stats - they are 0 if this is the fisrt step
        col_mean, col_var, n_total_samples = \
            _incremental_mean_and_var(X, last_mean=self.mean_,
                                      last_variance=self.var_,
                                      last_sample_count=self.n_samples_seen_)

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = np.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                np.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self.mean_ - col_batch_mean)
            X = np.vstack((self.singular_values_.reshape((-1, 1)) *
                          self.components_, X, mean_correction))

        U, S, V = linalg.svd(X, full_matrices=False)
        U, V = svd_flip(U, V, u_based_decision=False)
        explained_variance = S ** 2 / (n_total_samples - 1)
        explained_variance_ratio = S ** 2 / np.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = V[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = \
            explained_variance_ratio[:self.n_components_]
        if self.n_components_ < n_features:
            self.noise_variance_ = \
                explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = 0.
        return self

```
33 - /tmp/repos/scikit-learn/sklearn/utils/__init__.py
```python
def safe_indexing(X, indices):
    """Return items or rows from X using indices.

    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]



```
34 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
class SupervisedIntegerMixin:
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        if not isinstance(X, (KDTree, BallTree)):
            X, y = check_X_y(X, y, "csr", multi_output=True)

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn("A column-vector y was passed when a 1d array "
                              "was expected. Please change the shape of y to "
                              "(n_samples, ), for example using ravel().",
                              DataConversionWarning, stacklevel=2)

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        check_classification_targets(y)
        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        return self._fit(X)


class UnsupervisedMixin:
    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        """
        return self._fit(X)

```
35 - /tmp/repos/scikit-learn/sklearn/dummy.py
```python
def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_outputs]
            Predicted target values for X.
        """
        check_is_fitted(self, 'classes_')

        # numpy random_state expects Python int and not long as size argument
        # under Windows
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1:
            # Get same type even for self.n_outputs_ == 1
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]
        # Compute probability only once
        if self.strategy == "stratified":
            proba = self.predict_proba(X)
            if self.n_outputs_ == 1:
                proba = [proba]

        if self.sparse_output_:
            class_prob = None
            if self.strategy in ("most_frequent", "prior"):
                classes_ = [np.array([cp.argmax()]) for cp in class_prior_]

            elif self.strategy == "stratified":
                class_prob = class_prior_

            elif self.strategy == "uniform":
                raise ValueError("Sparse target prediction is not "
                                 "supported with the uniform strategy")

            elif self.strategy == "constant":
                classes_ = [np.array([c]) for c in constant]

            y = random_choice_csc(n_samples, classes_, class_prob,
                                  self.random_state)
        else:
            if self.strategy in ("most_frequent", "prior"):
                y = np.tile([classes_[k][class_prior_[k].argmax()] for
                             k in range(self.n_outputs_)], [n_samples, 1])

            elif self.strategy == "stratified":
                y = np.vstack(classes_[k][proba[k].argmax(axis=1)] for
                              k in range(self.n_outputs_)).T

            elif self.strategy == "uniform":
                ret = [classes_[k][rs.randint(n_classes_[k], size=n_samples)]
                       for k in range(self.n_outputs_)]
                y = np.vstack(ret).T

            elif self.strategy == "constant":
                y = np.tile(self.constant, (n_samples, 1))

            if self.n_outputs_ == 1 and not self.output_2d_:
                y = np.ravel(y)

        return y

    
```
36 - /tmp/repos/scikit-learn/sklearn/datasets/__init__.py
```python
"""
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


__all__ = ['clear_data_home',
           'dump_svmlight_file',
           'fetch_20newsgroups',
           'fetch_20newsgroups_vectorized',
           'fetch_lfw_pairs',
           'fetch_lfw_people',
           'fetch_mldata',
           'fetch_olivetti_faces',
           'fetch_species_distributions',
           'fetch_california_housing',
           'fetch_covtype',
           'fetch_rcv1',
           'fetch_kddcup99',
           'get_data_home',
           'load_boston',
           'load_diabetes',
           'load_digits',
           'load_files',
           'load_iris',
           'load_breast_cancer',
           'load_linnerud',
           'load_mlcomp',
           'load_sample_image',
           'load_sample_images',
           'load_svmlight_file',
           'load_svmlight_files',
           'load_wine',
           'make_biclusters',
           'make_blobs',
           'make_circles',
           'make_classification',
           'make_checkerboard',
           'make_friedman1',
           'make_friedman2',
           'make_friedman3',
           'make_gaussian_quantiles',
           'make_hastie_10_2',
           'make_low_rank_matrix',
           'make_moons',
           'make_multilabel_classification',
           'make_regression',
           'make_s_curve',
           'make_sparse_coded_signal',
           'make_sparse_spd_matrix',
           'make_sparse_uncorrelated',
           'make_spd_matrix',
           'make_swiss_roll',
           'mldata_filename']
```
37 - /tmp/repos/scikit-learn/sklearn/utils/__init__.py
```python
"""
The :mod:`sklearn.utils` module includes various utilities.
"""
from collections import Sequence

import numpy as np
from scipy.sparse import issparse
import warnings

from .murmurhash import murmurhash3_32
from .validation import (as_float_array,
                         assert_all_finite,
                         check_random_state, column_or_1d, check_array,
                         check_consistent_length, check_X_y, indexable,
                         check_symmetric)
from .class_weight import compute_class_weight, compute_sample_weight
from ..externals.joblib import cpu_count
from ..exceptions import DataConversionWarning
from .deprecation import deprecated


__all__ = ["murmurhash3_32", "as_float_array",
           "assert_all_finite", "check_array",
           "check_random_state",
           "compute_class_weight", "compute_sample_weight",
           "column_or_1d", "safe_indexing",
           "check_consistent_length", "check_X_y", 'indexable',
           "check_symmetric", "indices_to_mask", "deprecated"]


class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass


def safe_mask(X, mask):
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : array
        Mask to be used on X.

    Returns
    -------
        mask
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.signedinteger):
        return mask

    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask


def axis0_safe_slice(X, mask, len_mask):
    """
    This mask is safer than safe_mask since it returns an
    empty array, when a sparse matrix is sliced with a boolean mask
    with all False, instead of raising an unhelpful error in older
    versions of SciPy.

    See: https://github.com/scipy/scipy/issues/5361

    Also note that we can avoid doing the dot product by checking if
    the len_mask is not zero in _huber_loss_and_gradient but this
    is not going to be the bottleneck, since the number of outliers
    and non_outliers are typically non-zero and it makes the code
    tougher to follow.
    """
    if len_mask != 0:
        return X[safe_mask(X, mask), :]
    return np.zeros(shape=(0, X.shape[1]))



```
38 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation, and
    the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)

    squared : boolean, optional
        Return squared Euclidean distances.

    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)

    Returns
    -------
    distances : {array, sparse matrix}, shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[ 0.,  1.],
           [ 1.,  0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[ 1.        ],
           [ 1.41421356]])

    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    X, Y = check_pairwise_arrays(X, Y)

    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)


def pairwise_distances_argmin_min(X, Y, axis=1, metric="euclidean",
                                  batch_size=500, metric_kwargs=None):
    
```
39 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
\
        optional (default=None)
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled. See :term:`random_state`.

    Attributes
    ----------
    n_iter_ : int
        The number of iterations as selected by early stopping (if
        n_iter_no_change is not None). Otherwise it corresponds to max_iter.
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. For regressors,
        this is always 1.
    train_score_ : ndarray, shape (n_iter_ + 1,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    validation_score_ : ndarray, shape (n_iter_ + 1,)
        The scores at each iteration on the held-out validation data. The
        first entry is the score of the ensemble before the first iteration.
        Scores are computed according to the ``scoring`` parameter. Empty if
        no early stopping or if ``validation_fraction`` is None.

    Examples
    --------
    >>> # To use this experimental feature, we need to explicitly ask for it:
    >>> from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    >>> from sklearn.ensemble import HistGradientBoostingRegressor
    >>> from sklearn.datasets import load_boston
    >>> X, y = load_boston(return_X_y=True)
    >>> est = HistGradientBoostingRegressor().fit(X, y)
    >>> est.score(X, y)
    0.98...
    """

    _VALID_LOSSES = ('least_squares',)

    def __init__(self, loss='least_squares', learning_rate=0.1,
                 max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0., max_bins=256,
                 warm_start=False, scoring=None, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-7, verbose=0,
                 random_state=None):
        super(HistGradientBoostingRegressor, self).__init__(
            loss=loss, learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, max_bins=max_bins,
            warm_start=warm_start, scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, verbose=verbose,
            random_state=random_state)

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

    def _encode_y(self, y):
        # Just convert y to the expected dtype
        self.n_trees_per_iteration_ = 1
        y = y.astype(Y_DTYPE, copy=False)
        return y

    def _get_loss(self):
        return _LOSSES[self.loss]()


class HistGradientBoostingClassifier(BaseHistGradientBoosting,
                                     ClassifierMixin):
    
```
40 - /tmp/repos/scikit-learn/examples/datasets/plot_iris_dataset.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
The Iris Dataset
=========================================================
This data sets consists of 3 different types of irises'
(Setosa, Versicolour, and Virginica) petal and sepal
length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being:
Sepal Length, Sepal Width, Petal Length and Petal Width.

The below plot uses the first two features.
See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.
"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
```
41 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
def _sparse_predict(self, X):
        # Precondition: X is a csr_matrix of dtype np.float64.
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        C = 0.0  # C is not useful here

        return libsvm_sparse.libsvm_sparse_predict(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            LIBSVM_IMPL.index(self._impl), kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            # in the case of precomputed kernel given as a function, we
            # have to compute explicitly the kernel matrix
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')
        return X

    def _decision_function(self, X):
        """Distance of the samples X to the separating hyperplane.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X : array-like, shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        # NOTE: _validate_for_predict contains check for is_fitted
        # hence must be placed before any other attributes are used.
        X = self._validate_for_predict(X)
        X = self._compute_kernel(X)

        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function.
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            return -dec_func.ravel()

        return dec_func

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order="C")

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        return libsvm.decision_function(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_,
            svm_type=LIBSVM_IMPL.index(self._impl),
            kernel=kernel, degree=self.degree, cache_size=self.cache_size,
            coef0=self.coef0, gamma=self._gamma)

    def _sparse_decision_function(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')

        kernel = self.kernel
        if hasattr(kernel, '__call__'):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_decision_function(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            LIBSVM_IMPL.index(self._impl), kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            self.C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)

    
```
**42 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    return data


def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
                                  download_if_missing=True):
    """Load the 20 newsgroups dataset and transform it into tf-idf vectors.

    This is a convenience function; the tf-idf transformation is done using the
    default settings for `sklearn.feature_extraction.text.Vectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_20newsgroups with a custom `Vectorizer` or `CountVectorizer`.

    Read more in the :ref:`User Guide <20newsgroups>`.

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

    Returns
    -------
    bunch : Bunch object
        bunch.data: sparse matrix, shape [n_samples, n_features]
        bunch.target: array, shape [n_samples]
        bunch.target_names: list, length [n_classes]
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

    return Bunch(data=data, target=target, target_names=target_names)

```
43 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
class SupervisedIntegerMixin(object):
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        if not isinstance(X, (KDTree, BallTree)):
            X, y = check_X_y(X, y, "csr", multi_output=True)

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn("A column-vector y was passed when a 1d array "
                              "was expected. Please change the shape of y to "
                              "(n_samples, ), for example using ravel().",
                              DataConversionWarning, stacklevel=2)

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        check_classification_targets(y)
        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        return self._fit(X)


class UnsupervisedMixin(object):
    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        """
        return self._fit(X)

```
44 - /tmp/repos/scikit-learn/sklearn/datasets/species_distributions.py
```python
if not exists(archive_path):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")
        logger.info('Downloading species data from %s to %s' % (
            SAMPLES.url, data_home))
        samples_path = _fetch_remote(SAMPLES, dirname=data_home)
        with np.load(samples_path) as X:  # samples.zip is a valid npz
            for f in X.files:
                fhandle = BytesIO(X[f])
                if 'train' in f:
                    train = _load_csv(fhandle)
                if 'test' in f:
                    test = _load_csv(fhandle)
        remove(samples_path)

        logger.info('Downloading coverage data from %s to %s' % (
            COVERAGES.url, data_home))
        coverages_path = _fetch_remote(COVERAGES, dirname=data_home)
        with np.load(coverages_path) as X:  # coverages.zip is a valid npz
            coverages = []
            for f in X.files:
                fhandle = BytesIO(X[f])
                logger.debug(' - converting {}'.format(f))
                coverages.append(_load_coverage(fhandle))
            coverages = np.asarray(coverages, dtype=dtype)
        remove(coverages_path)

        bunch = Bunch(coverages=coverages,
                      test=test,
                      train=train,
                      **extra_params)
        joblib.dump(bunch, archive_path, compress=9)
    else:
        bunch = joblib.load(archive_path)

    return bunch

```
45 - /tmp/repos/scikit-learn/sklearn/impute/_base.py
```python
def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        check_is_fitted(self, "features_")
        X = self._validate_input(X)

        if X.shape[1] != self._n_features:
            raise ValueError("X has a different number of features "
                             "than during fitting.")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if (self.error_on_new and features_diff_fit_trans.size > 0):
                raise ValueError("The features {} have missing values "
                                 "in transform but have no missing values "
                                 "in fit.".format(features_diff_fit_trans))

            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {'allow_nan': True,
                'X_types': ['2darray', 'string']}

```
46 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
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
    """
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    
```
47 - /tmp/repos/scikit-learn/examples/model_selection/plot_cv_indices.py
```python



###############################################################################
# Let's see how it looks for the `KFold` cross-validation object:

fig, ax = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)

###############################################################################
# As you can see, by default the KFold cross-validation iterator does not
# take either datapoint class or group into consideration. We can change this
# by using the ``StratifiedKFold`` like so.

fig, ax = plt.subplots()
cv = StratifiedKFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)

###############################################################################
# In this case, the cross-validation retained the same ratio of classes across
# each CV split. Next we'll visualize this behavior for a number of CV
# iterators.
#
# Visualize cross-validation indices for many CV objects
# ------------------------------------------------------
#
# Let's visually compare the cross validation behavior for many
# scikit-learn cross-validation objects. Below we will loop through several
# common cross-validation objects, visualizing the behavior of each.
#
# Note how some use the group/class information while others do not.

cvs = [KFold, GroupKFold, ShuffleSplit, StratifiedKFold,
       GroupShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit]


for cv in cvs:
    this_cv = cv(n_splits=n_splits)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_cv_indices(this_cv, X, y, groups, ax, n_splits)

    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=.7)
plt.show()

```
**48 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
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

    return Bunch(data=data, target=target)


def _fetch_brute_kddcup99(data_home=None,
                          download_if_missing=True, percent10=True):

    """Load the kddcup99 dataset, downloading it if necessary.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    Returns
    -------
    dataset : dict-like object with the following attributes:
        dataset.data : numpy array of shape (494021, 41)
            Each row corresponds to the 41 features in the dataset.
        dataset.target : numpy array of shape (494021,)
            Each value corresponds to one of the 21 attack types or to the
            label 'normal.'.
        dataset.DESCR : string
            Description of the kddcup99 dataset.

    """

    data_home = get_data_home(data_home=data_home)
    if sys.version_info[0] == 3:
        # The zlib compression format use by joblib is not compatible when
        # switching from Python 2 to Python 3, let us use a separate folder
        # under Python 3:
        dir_suffix = "-py3"
    else:
        # Backward compat for Python 2 users
        dir_suffix = ""

    if percent10:
        kddcup_dir = join(data_home, "kddcup99_10" + dir_suffix)
        archive = ARCHIVE_10_PERCENT
    else:
        kddcup_dir = join(data_home, "kddcup99" + dir_suffix)
        archive = ARCHIVE

    samples_path = join(kddcup_dir, "samples")
    targets_path = join(kddcup_dir, "targets")
    available = exists(samples_path)

    
```
49 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
"""Stratified K-Folds cross validation iterator

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :class:`sklearn.model_selection.StratifiedKFold` instead.

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a variation of KFold that
    returns stratified folds. The folds are made by preserving
    the percentage of samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    y : array-like, [n_samples]
        Samples to split in K folds.

    n_folds : int, default=3
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``shuffle`` == True.

    Examples
    --------
    >>> from sklearn.cross_validation import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(y, n_folds=2)
    >>> len(skf)
    2
    >>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
    sklearn.cross_validation.StratifiedKFold(labels=[0 0 1 1], n_folds=2,
                                             shuffle=False, random_state=None)
    >>> for train_index, test_index in skf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]

    Notes
    -----
    Train and test sizes may be different in each fold, with a difference of at
    most ``n_classes``.

    See also
    --------
    LabelKFold: K-fold iterator variant with non-overlapping labels.
    """

    
```
50 - /tmp/repos/scikit-learn/sklearn/multiclass.py
```python


    @if_delegate_has_method(delegate='estimator')
    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators

        Should be used when memory is inefficient to train all data. Chunks
        of data can be passed in several iteration, where the first call
        should have an array of all target variables.


        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : array-like, shape = [n_samples]
            Multi-class targets.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self
        """
        if _check_partial_fit_first_call(self, classes):
            self.estimators_ = [clone(self.estimator) for i in
                                range(self.n_classes_ *
                                      (self.n_classes_ - 1) // 2)]

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError("Mini-batch contains {0} while it "
                             "must be subset of {1}".format(np.unique(y),
                                                            self.classes_))

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)
        combinations = itertools.combinations(range(self.n_classes_), 2)
        self.estimators_ = Parallel(
            n_jobs=self.n_jobs)(
                delayed(_partial_fit_ovo_binary)(
                    estimator, X, y, self.classes_[i], self.classes_[j])
                for estimator, (i, j) in izip(self.estimators_,
                                              (combinations)))

        self.pairwise_indices_ = None

        return self

    def predict(self, X):
        """Estimate the best class label for each sample in X.

        This is implemented as ``argmax(decision_function(X), axis=1)`` which
        will return the label of the class with most votes by estimators
        predicting the outcome of a decision for each possible class pair.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        Y = self.decision_function(X)
        if self.n_classes_ == 2:
            return self.classes_[(Y > 0).astype(np.int)]
        return self.classes_[Y.argmax(axis=1)]

    def decision_function(self, X):
        """Decision function for the OneVsOneClassifier.

        The decision values for the samples are computed by adding the
        normalized sum of pair-wise classification confidence levels to the
        votes in order to disambiguate between the decision values when the
        votes for all the classes are equal leading to a tie.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        Y : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'estimators_')

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack([est.predict(Xi)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        confidences = np.vstack([_predict_binary(est, Xi)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        Y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        if self.n_classes_ == 2:
            return Y[:, 1]
        return Y

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)


class OutputCodeClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    
```
51 - /tmp/repos/scikit-learn/sklearn/learning_curve.py
```python
X, y = indexable(X, y)
    # Make a list since we will be iterating multiple times over the folds
    cv = list(check_cv(cv, X, y, classifier=is_classifier(estimator)))
    scorer = check_scoring(estimator, scoring=scoring)

    # HACK as long as boolean indices are allowed in cv generators
    if cv[0][0].dtype == bool:
        new_cv = []
        for i in range(len(cv)):
            new_cv.append((np.nonzero(cv[i][0])[0], np.nonzero(cv[i][1])[0]))
        cv = new_cv

    n_max_training_samples = len(cv[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes,
                                             n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)
    if exploit_incremental_learning:
        classes = np.unique(y) if is_classifier(estimator) else None
        out = parallel(delayed(_incremental_fit_estimator)(
            clone(estimator), X, y, classes, train, test, train_sizes_abs,
            scorer, verbose) for train, test in cv)
    else:
        out = parallel(delayed(_fit_and_score)(
            clone(estimator), X, y, scorer, train[:n_train_samples], test,
            verbose, parameters=None, fit_params=None, return_train_score=True,
            error_score=error_score)
            for train, test in cv for n_train_samples in train_sizes_abs)
        out = np.array(out)[:, :2]
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 2)

    out = np.asarray(out).transpose((2, 1, 0))

    return train_sizes_abs, out[0], out[1]



```
52 - /tmp/repos/scikit-learn/sklearn/impute.py
```python
def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        check_is_fitted(self, "features_")

        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = check_array(X, accept_sparse=('csc', 'csr'),
                        force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)

        if X.shape[1] != self._n_features:
            raise ValueError("X has a different number of features "
                             "than during fitting.")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if (self.error_on_new and features_diff_fit_trans.size > 0):
                raise ValueError("The features {} have missing values "
                                 "in transform but have no missing values "
                                 "in fit.".format(features_diff_fit_trans))

            if (self.features_.size > 0 and
                    self.features_.size < self._n_features):
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        return self.fit(X, y).transform(X)

```
53 - /tmp/repos/scikit-learn/sklearn/feature_extraction/dict_vectorizer.py
```python
def inverse_transform(self, X, dict_type=dict):
        """Transform array or sparse matrix X back to feature mappings.

        X must have been produced by this DictVectorizer's transform or
        fit_transform method; it may only have passed through transformers
        that preserve the number of features and their order.

        In the case of one-hot/one-of-K coding, the constructed feature
        names and values are returned rather than the original ones.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Sample matrix.
        dict_type : callable, optional
            Constructor for feature mappings. Must conform to the
            collections.Mapping API.

        Returns
        -------
        D : list of dict_type objects, length = n_samples
            Feature mappings for the samples in X.
        """
        # COO matrix is not subscriptable
        X = check_array(X, accept_sparse=['csr', 'csc'])
        n_samples = X.shape[0]

        names = self.feature_names_
        dicts = [dict_type() for _ in xrange(n_samples)]

        if sp.issparse(X):
            for i, j in zip(*X.nonzero()):
                dicts[i][names[j]] = X[i, j]
        else:
            for i, d in enumerate(dicts):
                for j, v in enumerate(X[i, :]):
                    if v != 0:
                        d[names[j]] = X[i, j]

        return dicts

    def transform(self, X):
        """Transform feature->value dicts to array or sparse matrix.

        Named features not encountered during fit or fit_transform will be
        silently ignored.

        Parameters
        ----------
        X : Mapping or iterable over Mappings, length = n_samples
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        if self.sparse:
            return self._transform(X, fitting=False)

        else:
            dtype = self.dtype
            vocab = self.vocabulary_
            X = _tosequence(X)
            Xa = np.zeros((len(X), len(vocab)), dtype=dtype)

            for i, x in enumerate(X):
                for f, v in six.iteritems(x):
                    if isinstance(v, six.string_types):
                        f = "%s%s%s" % (f, self.separator, v)
                        v = 1
                    try:
                        Xa[i, vocab[f]] = dtype(v)
                    except KeyError:
                        pass

            return Xa

    def get_feature_names(self):
        """Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        """
        return self.feature_names_

    def restrict(self, support, indices=False):
        """Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : boolean, optional
            Whether support is a list of indices.

        Returns
        -------
        self

        Examples
        --------
        >>> from sklearn.feature_extraction import DictVectorizer
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> v = DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
        >>> v.get_feature_names()
        ['bar', 'baz', 'foo']
        >>> v.restrict(support.get_support()) # doctest: +ELLIPSIS
        DictVectorizer(dtype=..., separator='=', sort=True,
                sparse=True)
        >>> v.get_feature_names()
        ['bar', 'foo']
        """
        if not indices:
            support = np.where(support)[0]

        names = self.feature_names_
        new_vocab = {}
        for i in support:
            new_vocab[names[i]] = len(new_vocab)

        self.vocabulary_ = new_vocab
        self.feature_names_ = [f for f, i in sorted(six.iteritems(new_vocab),
                                                    key=itemgetter(1))]

        return self

```
54 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            X -= X.min()
        X = pairwise_estimator_convert_X(X, classifier_orig)
        set_random_state(classifier)
        # raises error on malformed input for fit
        with assert_raises(ValueError, msg="The classifier {} does not"
                           " raise an error when incorrect/malformed input "
                           "data for fit is passed. The number of training "
                           "examples is not the same as the number of labels."
                           " Perhaps use check_X_y in fit.".format(name)):
            classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert_true(hasattr(classifier, "classes_"))
        y_pred = classifier.predict(X)
        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        if name not in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            assert_greater(accuracy_score(y, y_pred), 0.83)

        # raises error on malformed input for predict
        if _is_pairwise(classifier):
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when shape of X"
                               "in predict is not equal to (n_test_samples,"
                               "n_training_samples)".format(name)):
                classifier.predict(X.reshape(-1, 1))
        else:
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when the number of features "
                               "in predict is different from the number of"
                               " features in fit.".format(name)):
                classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    assert_equal(decision.shape, (n_samples,))
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                if (n_classes == 3 and
                        # 1on1 of LibSVM works differently
                        not isinstance(classifier, BaseLibSVM)):
                    assert_equal(decision.shape, (n_samples, n_classes))
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if _is_pairwise(classifier):
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the  "
                                       "shape of X in decision_function is "
                                       "not equal to (n_test_samples, "
                                       "n_training_samples) in fit."
                                       .format(name)):
                        classifier.decision_function(X.reshape(-1, 1))
                else:
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the number "
                                       "of features in decision_function is "
                                       "different from the number of features"
                                       " in fit.".format(name)):
                        classifier.decision_function(X.T)
            except NotImplementedError:
                pass
        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert_equal(y_prob.shape, (n_samples, n_classes))
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_allclose(np.sum(y_prob, axis=1), np.ones(n_samples))
            # raises error on malformed input for predict_proba
            if _is_pairwise(classifier_orig):
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the shape of X"
                                   "in predict_proba is not equal to "
                                   "(n_test_samples, n_training_samples)."
                                   .format(name)):
                    classifier.predict_proba(X.reshape(-1, 1))
            else:
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the number of "
                                   "features in predict_proba is different "
                                   "from the number of features in fit."
                                   .format(name)):
                    classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))



```
55 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
def cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      method='predict'):
    
```
56 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if exploit_incremental_learning and not hasattr(estimator, "partial_fit"):
        raise ValueError("An estimator must support the partial_fit interface "
                         "to exploit incremental learning")
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # Store it as list as we will be iterating over the list multiple times
    cv_iter = list(cv.split(X, y, groups))

    scorer = check_scoring(estimator, scoring=scoring)

    n_max_training_samples = len(cv_iter[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes,
                                             n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)

    if shuffle:
        rng = check_random_state(random_state)
        cv_iter = ((rng.permutation(train), test) for train, test in cv_iter)

    if exploit_incremental_learning:
        classes = np.unique(y) if is_classifier(estimator) else None
        out = parallel(delayed(_incremental_fit_estimator)(
            clone(estimator), X, y, classes, train, test, train_sizes_abs,
            scorer, verbose) for train, test in cv_iter)
    else:
        train_test_proportions = []
        for train, test in cv_iter:
            for n_train_samples in train_sizes_abs:
                train_test_proportions.append((train[:n_train_samples], test))

        out = parallel(delayed(_fit_and_score)(
            clone(estimator), X, y, scorer, train, test,
            verbose, parameters=None, fit_params=None, return_train_score=True)
            for train, test in train_test_proportions)
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 2)

    out = np.asarray(out).transpose((2, 1, 0))

    return train_sizes_abs, out[0], out[1]



```
**57 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
"""Load the 20 newsgroups dataset and vectorize it into token counts \
(classification).

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

    
```
**58 - /tmp/repos/scikit-learn/sklearn/datasets/lfw.py**:
```python
def _fetch_lfw_pairs(index_file_path, data_folder_path, slice_=None,
                     color=False, resize=None):
    """Perform the actual data loading for the LFW pairs dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # parse the index file to find the number of pairs to be able to allocate
    # the right amount of memory before starting to decode the jpeg files
    with open(index_file_path, 'rb') as index_file:
        split_lines = [ln.strip().split(b('\t')) for ln in index_file]
    pair_specs = [sl for sl in split_lines if len(sl) > 2]
    n_pairs = len(pair_specs)

    # iterating over the metadata lines for each pair to find the filename to
    # decode and load in memory
    target = np.zeros(n_pairs, dtype=np.int)
    file_paths = list()
    for i, components in enumerate(pair_specs):
        if len(components) == 3:
            target[i] = 1
            pair = (
                (components[0], int(components[1]) - 1),
                (components[0], int(components[2]) - 1),
            )
        elif len(components) == 4:
            target[i] = 0
            pair = (
                (components[0], int(components[1]) - 1),
                (components[2], int(components[3]) - 1),
            )
        else:
            raise ValueError("invalid line %d: %r" % (i + 1, components))
        for j, (name, idx) in enumerate(pair):
            try:
                person_folder = join(data_folder_path, name)
            except TypeError:
                person_folder = join(data_folder_path, str(name, 'UTF-8'))
            filenames = list(sorted(listdir(person_folder)))
            file_path = join(person_folder, filenames[idx])
            file_paths.append(file_path)

    pairs = _load_imgs(file_paths, slice_, color, resize)
    shape = list(pairs.shape)
    n_faces = shape.pop(0)
    shape.insert(0, 2)
    shape.insert(0, n_faces // 2)
    pairs.shape = shape

    return pairs, target, np.array(['Different persons', 'Same person'])


def fetch_lfw_pairs(subset='train', data_home=None, funneled=True, resize=0.5,
                    color=False, slice_=(slice(70, 195), slice(78, 172)),
                    download_if_missing=True):
    
```
59 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
"""Fits the GraphicalLasso covariance model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)
        """
        # Covariance does not make sense for a single feature
        X = check_array(X, ensure_min_features=2, estimator=self)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        emp_cov = empirical_covariance(
            X, assume_centered=self.assume_centered)

        cv = check_cv(self.cv, y, classifier=False)

        # List of (alpha, scores, covs)
        path = list()
        n_alphas = self.alphas
        inner_verbose = max(0, self.verbose - 1)

        if isinstance(n_alphas, collections.Sequence):
            alphas = self.alphas
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            alpha_1 = alpha_max(emp_cov)
            alpha_0 = 1e-2 * alpha_1
            alphas = np.logspace(np.log10(alpha_0), np.log10(alpha_1),
                                 n_alphas)[::-1]

        t0 = time.time()
        
```
60 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

        .. versionadded:: 0.18

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    X, Y, dtype_float = _return_float_dtype(X, Y)

    warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype,
                            warn_on_dtype=warn_on_dtype, estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y


def check_paired_arrays(X, Y):
    """ Set X and Y appropriately and checks inputs for paired distances

    All paired distance metrics should use this function first to assert that
    the given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the dimensions of the two arrays are equal.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    X, Y = check_pairwise_arrays(X, Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y should be of same shape. They were "
                         "respectively %r and %r long." % (X.shape, Y.shape))
    return X, Y


# Pairwise distances

```
61 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def additive_chi2_kernel(X, Y=None):
    """Computes the additive chi-squared kernel between observations in X and Y

    The chi-squared kernel is computed between each pair of rows in X and Y.  X
    and Y have to be non-negative. This kernel is most commonly applied to
    histograms.

    The chi-squared kernel is given by::

        k(x, y) = -Sum [(x - y)^2 / (x + y)]

    It can be interpreted as a weighted difference per entry.

    Read more in the :ref:`User Guide <chi2_kernel>`.

    Notes
    -----
    As the negative of a distance, this kernel is only conditionally positive
    definite.


    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)

    References
    ----------
    * Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
      Local features and kernels for classification of texture and object
      categories: A comprehensive study
      International Journal of Computer Vision 2007
      http://research.microsoft.com/en-us/um/people/manik/projects/trade-off/papers/ZhangIJCV06.pdf


    See also
    --------
    chi2_kernel : The exponentiated version of the kernel, which is usually
        preferable.

    sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
        to this kernel.
    """
    if issparse(X) or issparse(Y):
        raise ValueError("additive_chi2 does not support sparse matrices.")
    X, Y = check_pairwise_arrays(X, Y)
    if (X < 0).any():
        raise ValueError("X contains negative values.")
    if Y is not X and (Y < 0).any():
        raise ValueError("Y contains negative values.")

    result = np.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)
    _chi2_kernel_fast(X, Y, result)
    return result


def chi2_kernel(X, Y=None, gamma=1.):
    """Computes the exponential chi-squared kernel X and Y.

    The chi-squared kernel is computed between each pair of rows in X and Y.  X
    and Y have to be non-negative. This kernel is most commonly applied to
    histograms.

    The chi-squared kernel is given by::

        k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])

    It can be interpreted as a weighted difference per entry.

    Read more in the :ref:`User Guide <chi2_kernel>`.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float, default=1.
        Scaling parameter of the chi2 kernel.

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)

    References
    ----------
    * Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
      Local features and kernels for classification of texture and object
      categories: A comprehensive study
      International Journal of Computer Vision 2007
      http://research.microsoft.com/en-us/um/people/manik/projects/trade-off/papers/ZhangIJCV06.pdf

    See also
    --------
    additive_chi2_kernel : The additive version of this kernel

    sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
        to the additive version of this kernel.
    """
    K = additive_chi2_kernel(X, Y)
    K *= gamma
    return np.exp(K, K)


# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cityblock': manhattan_distances,
    'cosine': cosine_distances,
    'euclidean': euclidean_distances,
    'l2': euclidean_distances,
    'l1': manhattan_distances,
    'manhattan': manhattan_distances,
    'precomputed': None,  # HACK: precomputed is always allowed, never called
}



```
62 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class ShuffleSplit(BaseShuffleSplit):
    """Random permutation cross-validator

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, default=0.1
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default (the parameter is
        unspecified), the value is set to 0.1.
        The default will change in version 0.21. It will remain 0.1 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> from sklearn.model_selection import ShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> rs = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
    >>> rs.get_n_splits(X)
    3
    >>> print(rs)
    ShuffleSplit(n_splits=3, random_state=0, test_size=0.25, train_size=None)
    >>> for train_index, test_index in rs.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...  # doctest: +ELLIPSIS
    TRAIN: [3 1 0] TEST: [2]
    TRAIN: [2 1 3] TEST: [0]
    TRAIN: [0 2 1] TEST: [3]
    >>> rs = ShuffleSplit(n_splits=3, train_size=0.5, test_size=.25,
    ...                   random_state=0)
    >>> for train_index, test_index in rs.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...  # doctest: +ELLIPSIS
    TRAIN: [3 1] TEST: [2]
    TRAIN: [2 1] TEST: [0]
    TRAIN: [0 2] TEST: [3]
    """

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(n_samples,
                                                  self.test_size,
                                                  self.train_size)
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test:(n_test + n_train)]
            yield ind_train, ind_test



```
63 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = _boston_subset(n_samples=50)
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_data_not_an_array(name, estimator_orig, X, y):
    if name in CROSS_DECOMPOSITION:
        raise SkipTest("Skipping check_estimators_data_not_an_array "
                       "for cross decomposition module as estimators "
                       "are not deterministic.")
    # separate estimators to control random seeds
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    set_random_state(estimator_1)
    set_random_state(estimator_2)

    y_ = NotAnArray(np.asarray(y))
    X_ = NotAnArray(np.asarray(X))

    # fit
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


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


def multioutput_estimator_convert_y_2d(estimator, y):
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if "MultiTask" in estimator.__class__.__name__:
        return np.reshape(y, (-1, 1))
    return y
```
**64 - /tmp/repos/scikit-learn/sklearn/datasets/rcv1.py**:
```python
"""RCV1 dataset.
"""

# Author: Tom Dupre la Tour
# License: BSD 3 clause

import logging

from os import remove
from os.path import exists, join
from gzip import GzipFile

import numpy as np
import scipy.sparse as sp

from .base import get_data_home
from .base import _pkl_filepath
from .base import _fetch_remote
from .base import RemoteFileMetadata
from ..utils.fixes import makedirs
from ..externals import joblib
from .svmlight_format import load_svmlight_files
from ..utils import shuffle as shuffle_
from ..utils import Bunch


# The original data can be found at:
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt0.dat.gz
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt1.dat.gz
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt2.dat.gz
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt3.dat.gz
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz
XY_METADATA = (
    RemoteFileMetadata(
        url='https://ndownloader.figshare.com/files/5976069',
        checksum=('ed40f7e418d10484091b059703eeb95a'
                  'e3199fe042891dcec4be6696b9968374'),
        filename='lyrl2004_vectors_test_pt0.dat.gz'),
    RemoteFileMetadata(
        url='https://ndownloader.figshare.com/files/5976066',
        checksum=('87700668ae45d45d5ca1ef6ae9bd81ab'
                  '0f5ec88cc95dcef9ae7838f727a13aa6'),
        filename='lyrl2004_vectors_test_pt1.dat.gz'),
    RemoteFileMetadata(
        url='https://ndownloader.figshare.com/files/5976063',
        checksum=('48143ac703cbe33299f7ae9f4995db4'
                  '9a258690f60e5debbff8995c34841c7f5'),
        filename='lyrl2004_vectors_test_pt2.dat.gz'),
    RemoteFileMetadata(
        url='https://ndownloader.figshare.com/files/5976060',
        checksum=('dfcb0d658311481523c6e6ca0c3f5a3'
                  'e1d3d12cde5d7a8ce629a9006ec7dbb39'),
        filename='lyrl2004_vectors_test_pt3.dat.gz'),
    RemoteFileMetadata(
        url='https://ndownloader.figshare.com/files/5976057',
        checksum=('5468f656d0ba7a83afc7ad44841cf9a5'
                  '3048a5c083eedc005dcdb5cc768924ae'),
        filename='lyrl2004_vectors_train.dat.gz')
)

# The original data can be found at:
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
TOPICS_METADATA = RemoteFileMetadata(
    url='https://ndownloader.figshare.com/files/5976048',
    checksum=('2a98e5e5d8b770bded93afc8930d882'
              '99474317fe14181aee1466cc754d0d1c1'),
    filename='rcv1v2.topics.qrels.gz')

logger = logging.getLogger(__name__)


def fetch_rcv1(data_home=None, subset='all', download_if_missing=True,
               random_state=None, shuffle=False):
    
```
65 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings
def check_classifiers_predictions(X, y, name, classifier_orig):
    classes = np.unique(y)
    classifier = clone(classifier_orig)
    if name == 'BernoulliNB':
        X = X > X.mean()
    set_random_state(classifier)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X)
        n_samples, n_features = X.shape
        assert isinstance(decision, np.ndarray)
        if len(classes) == 2:
            dec_pred = (decision.ravel() > 0).astype(np.int)
            dec_exp = classifier.classes_[dec_pred]
            assert_array_equal(dec_exp, y_pred,
                               err_msg="decision_function does not match "
                               "classifier for %r: expected '%s', got '%s'" %
                               (classifier, ", ".join(map(str, dec_exp)),
                                ", ".join(map(str, y_pred))))
        elif getattr(classifier, 'decision_function_shape', 'ovr') == 'ovr':
            decision_y = np.argmax(decision, axis=1).astype(int)
            y_exp = classifier.classes_[decision_y]
            assert_array_equal(y_exp, y_pred,
                               err_msg="decision_function does not match "
                               "classifier for %r: expected '%s', got '%s'" %
                               (classifier, ", ".join(map(str, y_exp)),
                                ", ".join(map(str, y_pred))))

    # training set performance
    if name != "ComplementNB":
        # This is a pathological data set for ComplementNB.
        # For some specific cases 'ComplementNB' predicts less classes
        # than expected
        assert_array_equal(np.unique(y), np.unique(y_pred))
    assert_array_equal(classes, classifier.classes_,
                       err_msg="Unexpected classes_ attribute for %r: "
                       "expected '%s', got '%s'" %
                       (classifier, ", ".join(map(str, classes)),
                        ", ".join(map(str, classifier.classes_))))


def choose_check_classifiers_labels(name, y, y_names):
    return y if name in ["LabelPropagation", "LabelSpreading"] else y_names

def check_classifiers_classes(name, classifier_orig):
    X_multiclass, y_multiclass = make_blobs(n_samples=30, random_state=0,
                                            cluster_std=0.1)
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass,
                                         random_state=7)
    X_multiclass = StandardScaler().fit_transform(X_multiclass)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X_multiclass -= X_multiclass.min() - .1

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_multiclass = pairwise_estimator_convert_X(X_multiclass, classifier_orig)
    X_binary = pairwise_estimator_convert_X(X_binary, classifier_orig)

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    for X, y, y_names in [(X_multiclass, y_multiclass, y_names_multiclass),
                          (X_binary, y_binary, y_names_binary)]:
        for y_names_i in [y_names, y_names.astype('O')]:
            y_ = choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)
```
66 - /tmp/repos/scikit-learn/sklearn/discriminant_analysis.py
```python
def predict(self, X):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
        """
        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred

    def predict_proba(self, X):
        """Return posterior probabilities of classification.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples/test vectors.

        Returns
        -------
        C : array, shape = [n_samples, n_classes]
            Posterior probabilities of classification per class.
        """
        values = self._decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        """Return posterior probabilities of classification.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples/test vectors.

        Returns
        -------
        C : array, shape = [n_samples, n_classes]
            Posterior log-probabilities of classification per class.
        """
        # XXX : can do better to avoid precision overflows
        probas_ = self.predict_proba(X)
        return np.log(probas_)

```
67 - /tmp/repos/scikit-learn/sklearn/discriminant_analysis.py
```python
def fit(self, X, y):
        """Fit LinearDiscriminantAnalysis model according to the given
           training data and parameters.

           .. versionchanged:: 0.19
              *store_covariance* has been moved to main constructor.

           .. versionchanged:: 0.19
              *tol* has been moved to main constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)
        self.classes_ = unique_labels(y)

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if self.priors_.sum() != 1:
            warnings.warn("The priors do not sum to 1. Renormalizing",
                          UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Get the maximum number of components
        if self.n_components is None:
            self._max_components = len(self.classes_) - 1
        else:
            self._max_components = min(len(self.classes_) - 1,
                                       self.n_components)

        if self.solver == 'svd':
            if self.shrinkage is not None:
                raise NotImplementedError('shrinkage not supported')
            self._solve_svd(X, y)
        elif self.solver == 'lsqr':
            self._solve_lsqr(X, y, shrinkage=self.shrinkage)
        elif self.solver == 'eigen':
            self._solve_eigen(X, y, shrinkage=self.shrinkage)
        else:
            raise ValueError("unknown solver {} (valid solvers are 'svd', "
                             "'lsqr', and 'eigen').".format(self.solver))
        if self.classes_.size == 2:  # treat binary case as a special case
            self.coef_ = np.array(self.coef_[1, :] - self.coef_[0, :], ndmin=2)
            self.intercept_ = np.array(self.intercept_[1] - self.intercept_[0],
                                       ndmin=1)
        return self

    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        if self.solver == 'lsqr':
            raise NotImplementedError("transform not implemented for 'lsqr' "
                                      "solver (use 'svd' or 'eigen').")
        check_is_fitted(self, ['xbar_', 'scalings_'], all_or_any=any)

        X = check_array(X)
        if self.solver == 'svd':
            X_new = np.dot(X - self.xbar_, self.scalings_)
        elif self.solver == 'eigen':
            X_new = np.dot(X, self.scalings_)

        return X_new[:, :self._max_components]

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if len(self.classes_) == 2:  # binary case
            return np.column_stack([1 - prob, prob])
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    
```
68 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class BaseShuffleSplit(with_metaclass(ABCMeta)):
    """Base class for ShuffleSplit and StratifiedShuffleSplit"""

    def __init__(self, n_splits=10, test_size="default", train_size=None,
                 random_state=None):
        _validate_shuffle_split_init(test_size, train_size)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test

    @abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

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
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def __repr__(self):
        return _build_repr(self)



```
69 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
def _dense_fit(self, X, y, sample_weight, solver_type, kernel,
                   random_seed):
        if callable(self.kernel):
            # you must store a reference to X to compute the kernel in predict
            # TODO: add keyword copy to copy on demand
            self.__Xfit = X
            X = self._compute_kernel(X)

            if X.shape[0] != X.shape[1]:
                raise ValueError("X.shape[0] should be equal to X.shape[1]")

        libsvm.set_verbosity_wrap(self.verbose)

        # we don't pass **self.get_params() to allow subclasses to
        # add other parameters to __init__
        self.support_, self.support_vectors_, self.n_support_, \
            self.dual_coef_, self.intercept_, self.probA_, \
            self.probB_, self.fit_status_ = libsvm.fit(
                X, y,
                svm_type=solver_type, sample_weight=sample_weight,
                class_weight=self.class_weight_, kernel=kernel, C=self.C,
                nu=self.nu, probability=self.probability, degree=self.degree,
                shrinking=self.shrinking, tol=self.tol,
                cache_size=self.cache_size, coef0=self.coef0,
                gamma=self._gamma, epsilon=self.epsilon,
                max_iter=self.max_iter, random_seed=random_seed)

        self._warn_from_fit_status()

    def _sparse_fit(self, X, y, sample_weight, solver_type, kernel,
                    random_seed):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')
        X.sort_indices()

        kernel_type = self._sparse_kernels.index(kernel)

        libsvm_sparse.set_verbosity_wrap(self.verbose)

        self.support_, self.support_vectors_, dual_coef_data, \
            self.intercept_, self.n_support_, \
            self.probA_, self.probB_, self.fit_status_ = \
            libsvm_sparse.libsvm_sparse_train(
                X.shape[1], X.data, X.indices, X.indptr, y, solver_type,
                kernel_type, self.degree, self._gamma, self.coef0, self.tol,
                self.C, self.class_weight_,
                sample_weight, self.nu, self.cache_size, self.epsilon,
                int(self.shrinking), int(self.probability), self.max_iter,
                random_seed)

        self._warn_from_fit_status()

        if hasattr(self, "classes_"):
            n_class = len(self.classes_) - 1
        else:   # regression
            n_class = 1
        n_SV = self.support_vectors_.shape[0]

        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1,
                                     dual_coef_indices.size / n_class)
        self.dual_coef_ = sp.csr_matrix(
            (dual_coef_data, dual_coef_indices, dual_coef_indptr),
            (n_class, n_SV))

    def predict(self, X):
        """Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)

    def _dense_predict(self, X):
        n_samples, n_features = X.shape
        X = self._compute_kernel(X)
        if X.ndim == 1:
            X = check_array(X, order='C')

        kernel = self.kernel
        if callable(self.kernel):
            kernel = 'precomputed'
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))

        svm_type = LIBSVM_IMPL.index(self._impl)

        return libsvm.predict(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_, svm_type=svm_type, kernel=kernel,
            degree=self.degree, coef0=self.coef0, gamma=self._gamma,
            cache_size=self.cache_size)

    
```
70 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        
```
71 - /tmp/repos/scikit-learn/sklearn/linear_model/ridge.py
```python
"""
    if return_intercept and sparse.issparse(X) and solver != 'sag':
        if solver != 'auto':
            warnings.warn("In Ridge, only 'sag' solver can currently fit the "
                          "intercept when X is sparse. Solver has been "
                          "automatically changed into 'sag'.")
        solver = 'sag'

    _dtype = [np.float64, np.float32]

    # SAG needs X and y columns to be C-contiguous and np.float64
    if solver in ['sag', 'saga']:
        X = check_array(X, accept_sparse=['csr'],
                        dtype=np.float64, order='C')
        y = check_array(y, dtype=np.float64, ensure_2d=False, order='F')
    else:
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=_dtype)
        y = check_array(y, dtype=X.dtype, ensure_2d=False)
    check_consistent_length(X, y)

    n_samples, n_features = X.shape

    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True

    n_samples_, n_targets = y.shape

    if n_samples != n_samples_:
        raise ValueError("Number of samples in X and y does not correspond:"
                         " %d != %d" % (n_samples, n_samples_))

    has_sw = sample_weight is not None

    if solver == 'auto':
        # cholesky if it's a dense array and cg in any other case
        if not sparse.issparse(X) or has_sw:
            solver = 'cholesky'
        else:
            solver = 'sparse_cg'

    if has_sw:
        if np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if solver not in ['sag', 'saga']:
            # SAG supports sample_weight directly. For other solvers,
            # we implement sample_weight via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

    # There should be either 1 or n_targets penalties
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    if alpha.size not in [1, n_targets]:
        raise ValueError("Number of targets and number of penalties "
                         "do not correspond: %d != %d"
                         % (alpha.size, n_targets))

    if alpha.size == 1 and n_targets > 1:
        alpha = np.repeat(alpha, n_targets)

    if solver not in ('sparse_cg', 'cholesky', 'svd', 'lsqr', 'sag', 'saga'):
        raise ValueError('Solver %s not understood' % solver)

    n_iter = None
    
```
72 - /tmp/repos/scikit-learn/sklearn/datasets/species_distributions.py
```python
def construct_grids(batch):
    """Construct the map grid from the batch object

    Parameters
    ----------
    batch : Batch object
        The object returned by :func:`fetch_species_distributions`

    Returns
    -------
    (xgrid, ygrid) : 1-D arrays
        The grid corresponding to the values in batch.coverages
    """
    # x,y coordinates for corner cells
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + (batch.Nx * batch.grid_size)
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + (batch.Ny * batch.grid_size)

    # x coordinates of the grid cells
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    # y coordinates of the grid cells
    ygrid = np.arange(ymin, ymax, batch.grid_size)

    return (xgrid, ygrid)


def fetch_species_distributions(data_home=None,
                                download_if_missing=True):
    """Loader for species distribution dataset from Phillips et. al. (2006)

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    --------
    The data is returned as a Bunch object with the following attributes:

    coverages : array, shape = [14, 1592, 1212]
        These represent the 14 features measured at each point of the map grid.
        The latitude/longitude values for the grid are discussed below.
        Missing data is represented by the value -9999.

    train : record array, shape = (1623,)
        The training points for the data.  Each point has three fields:

        - train['species'] is the species name
        - train['dd long'] is the longitude, in degrees
        - train['dd lat'] is the latitude, in degrees

    test : record array, shape = (619,)
        The test points for the data.  Same format as the training data.

    Nx, Ny : integers
        The number of longitudes (x) and latitudes (y) in the grid

    x_left_lower_corner, y_left_lower_corner : floats
        The (x,y) position of the lower-left corner, in degrees

    grid_size : float
        The spacing between points of the grid, in degrees

    References
    ----------

    * `"Maximum entropy modeling of species geographic distributions"
      <http://rob.schapire.net/papers/ecolmod.pdf>`_
      S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
      190:231-259, 2006.

    Notes
    -----

    This dataset represents the geographic distribution of species.
    The dataset is provided by Phillips et. al. (2006).

    The two species are:

    - `"Bradypus variegatus"
      <http://www.iucnredlist.org/details/3038/0>`_ ,
      the Brown-throated Sloth.

    - `"Microryzomys minutus"
      <http://www.iucnredlist.org/details/13408/0>`_ ,
      also known as the Forest Small Rice Rat, a rodent that lives in Peru,
      Colombia, Ecuador, Peru, and Venezuela.

    - For an example of using this dataset with scikit-learn, see
      :ref:`examples/applications/plot_species_distribution_modeling.py
      <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
    """
    data_home = get_data_home(data_home)
    if not exists(data_home):
        makedirs(data_home)

    # Define parameters for the data files.  These should not be changed
    # unless the data model changes.  They will be saved in the npz file
    # with the downloaded data.
    extra_params = dict(x_left_lower_corner=-94.8,
                        Nx=1212,
                        y_left_lower_corner=-56.05,
                        Ny=1592,
                        grid_size=0.05)
    dtype = np.int16

    archive_path = _pkl_filepath(data_home, DATA_ARCHIVE_NAME)

    
```
73 - /tmp/repos/scikit-learn/sklearn/covariance/elliptic_envelope.py
```python
def decision_function(self, X, raw_values=None):
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        raw_values : bool, optional
            Whether or not to consider raw Mahalanobis distances as the
            decision function. Must be False (default) for compatibility
            with the others outlier detection tools.

        .. deprecated:: 0.20
            ``raw_values`` has been deprecated in 0.20 and will be removed
            in 0.22.

        Returns
        -------

        decision : array-like, shape (n_samples, )
            Decision function of the samples.
            It is equal to the shifted Mahalanobis distances.
            The threshold for being an outlier is 0, which ensures a
            compatibility with other outlier detection algorithms.

        """
        check_is_fitted(self, 'offset_')
        negative_mahal_dist = self.score_samples(X)

        # raw_values deprecation:
        if raw_values is not None:
            warnings.warn("raw_values parameter is deprecated in 0.20 and will"
                          " be removed in 0.22.", DeprecationWarning)

            if not raw_values:
                return (-self.offset_) ** 0.33 - (-negative_mahal_dist) ** 0.33

        return negative_mahal_dist - self.offset_

    def score_samples(self, X):
        """Compute the negative Mahalanobis distances.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        negative_mahal_distances : array-like, shape (n_samples, )
            Opposite of the Mahalanobis distances.
        """
        check_is_fitted(self, 'offset_')
        X = check_array(X)
        return -self.mahalanobis(X)

    def predict(self, X):
        """
        Predict the labels (1 inlier, -1 outlier) of X according to the
        fitted model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        is_inlier : array, shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        X = check_array(X)
        is_inlier = -np.ones(X.shape[0], dtype=int)
        values = self.decision_function(X)
        is_inlier[values >= 0] = 1

        return is_inlier

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    @property
    def threshold_(self):
        warnings.warn("threshold_ attribute is deprecated in 0.20 and will"
                      " be removed in 0.22.", DeprecationWarning)
        return self.offset_

```
74 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------

    X : {array, sparse matrix}, shape [n_samples, n_features + 1]
        Same data with dummy feature added as first column.

    Examples
    --------

    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[ 1.,  0.,  1.],
           [ 1.,  1.,  0.]])
    """
    X = check_array(X, accept_sparse=['csc', 'csr', 'coo'], dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    if sparse.issparse(X):
        if sparse.isspmatrix_coo(X):
            # Shift columns to the right.
            col = X.col + 1
            # Column indices of dummy feature are 0 everywhere.
            col = np.concatenate((np.zeros(n_samples), col))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            row = np.concatenate((np.arange(n_samples), X.row))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.ones(n_samples) * value, X.data))
            return sparse.coo_matrix((data, (row, col)), shape)
        elif sparse.isspmatrix_csc(X):
            # Shift index pointers since we need to add n_samples elements.
            indptr = X.indptr + n_samples
            # indptr[0] must be 0.
            indptr = np.concatenate((np.array([0]), indptr))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            indices = np.concatenate((np.arange(n_samples), X.indices))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.ones(n_samples) * value, X.data))
            return sparse.csc_matrix((data, indices, indptr), shape)
        else:
            klass = X.__class__
            return klass(add_dummy_feature(X.tocoo(), value))
    else:
        return np.hstack((np.ones((n_samples, 1)) * value, X))


def _transform_selected(X, transform, selected="all", copy=True):
    """Apply a transform function to portion of selected features

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Dense array or sparse matrix.

    transform : callable
        A callable transform(X) -> X_transformed

    copy : boolean, optional
        Copy X even if it could be avoided.

    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.

    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    X = check_array(X, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES)

    if isinstance(selected, six.string_types) and selected == "all":
        return transform(X)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))


class OneHotEncoder(BaseEstimator, TransformerMixin):
    
```
75 - /tmp/repos/scikit-learn/sklearn/linear_model/base.py
```python
"""
Generalized Linear models.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Fabian Pedregosa <fabian.pedregosa@inria.fr>
# Olivier Grisel <olivier.grisel@ensta.org>
#         Vincent Michel <vincent.michel@inria.fr>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
#         Maryan Morel <maryan.morel@polytechnique.edu>
#         Giorgio Patrini <giorgio.patrini@anu.edu.au>
# License: BSD 3 clause

from __future__ import division
from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse

from ..externals import six
from ..externals.joblib import Parallel, delayed
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils import check_array, check_X_y, deprecated, as_float_array
from ..utils.validation import FLOAT_DTYPES
from ..utils import check_random_state
from ..utils.extmath import safe_sparse_dot
from ..utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from ..utils.fixes import sparse_lsqr
from ..utils.seq_dataset import ArrayDataset, CSRDataset
from ..utils.validation import check_is_fitted
from ..exceptions import NotFittedError
from ..preprocessing.data import normalize as f_normalize

# TODO: bayesian_ridge_regression and bayesian_regression_ard
# should be squashed into its respective objects.

SPARSE_INTERCEPT_DECAY = 0.01
# For sparse data intercept updates are scaled by this decay factor to avoid
# intercept oscillation.


def make_dataset(X, y, sample_weight, random_state=None):
    """Create ``Dataset`` abstraction for sparse and dense inputs.

    This also returns the ``intercept_decay`` which is different
    for sparse datasets.
    """

    rng = check_random_state(random_state)
    # seed should never be 0 in SequentialDataset
    seed = rng.randint(1, np.iinfo(np.int32).max)

    if sp.issparse(X):
        dataset = CSRDataset(X.data, X.indptr, X.indices, y, sample_weight,
                             seed=seed)
        intercept_decay = SPARSE_INTERCEPT_DECAY
    else:
        dataset = ArrayDataset(X, y, sample_weight, seed=seed)
        intercept_decay = 1.0

    return dataset, intercept_decay
```
76 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
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

        super(GradientBoostingClassifier, self).__init__(
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

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        score = self._decision_function(X)
        if score.shape[1] == 1:
            return score.ravel()
        return score

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        for dec in self._staged_decision_function(X):
            # no yield from in Python2.X
            yield dec

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        score = self.decision_function(X)
        decisions = self.loss_._score_to_decision(score)
        return self.classes_.take(decisions, axis=0)

    
```
77 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class _CVIterableWrapper(BaseCrossValidator):
    """Wrapper class for old style cv objects and iterables."""
    def __init__(self, cv):
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

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
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.cv)

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


def check_cv(cv=3, y=None, classifier=False):
    """Input checker utility for building a cross-validator

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    y : array-like, optional
        The target variable for supervised learning problems.

    classifier : boolean, optional, default False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """
    if cv is None:
        cv = 3

    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return StratifiedKFold(cv)
        else:
            return KFold(cv)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


def train_test_split(*arrays, **options):
    
```
**78 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
"""KDDCUP 99 dataset.

A classic dataset for anomaly detection.

The dataset page is available from UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz

"""

import sys
import errno
from gzip import GzipFile
import logging
import os
from os.path import exists, join

import numpy as np


from .base import _fetch_remote
from .base import get_data_home
from .base import RemoteFileMetadata
from ..utils import Bunch
from ..externals import joblib, six
from ..utils import check_random_state
from ..utils import shuffle as shuffle_method

# The original data can be found at:
# http://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz
ARCHIVE = RemoteFileMetadata(
    filename='kddcup99_data',
    url='https://ndownloader.figshare.com/files/5976045',
    checksum=('3b6c942aa0356c0ca35b7b595a26c89d'
              '343652c9db428893e7494f837b274292'))

# The original data can be found at:
# http://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz
ARCHIVE_10_PERCENT = RemoteFileMetadata(
    filename='kddcup99_10_data',
    url='https://ndownloader.figshare.com/files/5976042',
    checksum=('8045aca0d84e70e622d1148d7df78249'
              '6f6333bf6eb979a1b0837c42a9fd9561'))

logger = logging.getLogger(__name__)


def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
                   random_state=None,
                   percent10=True, download_if_missing=True):
    """Load and return the kddcup 99 dataset (classification).

    The KDD Cup '99 dataset was created by processing the tcpdump portions
    of the 1998 DARPA Intrusion Detection System (IDS) Evaluation dataset,
    created by MIT Lincoln Lab [1]. The artificial data was generated using
    a closed network and hand-injected attacks to produce a large number of
    different types of attack with normal activity in the background.
    As the initial goal was to produce a large training set for supervised
    learning algorithms, there is a large proportion (80.1%) of abnormal
    data which is unrealistic in real world, and inappropriate for unsupervised
    anomaly detection which aims at detecting 'abnormal' data, ie

    1) qualitatively different from normal data.

    2) in large minority among the observations.

    We thus transform the KDD Data set into two different data sets: SA and SF.

    - SA is obtained by simply selecting all the normal data, and a small
      proportion of abnormal data to gives an anomaly proportion of 1%.

    - SF is obtained as in [2]
      by simply picking up the data whose attribute logged_in is positive, thus
      focusing on the intrusion attack, which gives a proportion of 0.3% of
      attack.

    - http and smtp are two subsets of SF corresponding with third feature
      equal to 'http' (resp. to 'smtp')


    General KDD structure :

    ================      ==========================================
    Samples total         4898431
    Dimensionality        41
    Features              discrete (int) or continuous (float)
    Targets               str, 'normal.' or name of the anomaly type
    ================      ==========================================

    SA structure :

    ================      ==========================================
    Samples total         976158
    Dimensionality        41
    Features              discrete (int) or continuous (float)
    Targets               str, 'normal.' or name of the anomaly type
    ================      ==========================================

    SF structure :

    ================      ==========================================
    Samples total         699691
    Dimensionality        4
    Features              discrete (int) or continuous (float)
    Targets               str, 'normal.' or name of the anomaly type
    ================      ==========================================

    http structure :

    ================      ==========================================
    Samples total         619052
    Dimensionality        3
    Features              discrete (int) or continuous (float)
    Targets               str, 'normal.' or name of the anomaly type
    ================      ==========================================

    smtp structure :

    ================      ==========================================
    Samples total         95373
    Dimensionality        3
    Features              discrete (int) or continuous (float)
    Targets               str, 'normal.' or name of the anomaly type
    ================      ==========================================

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

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset. If subset='SA', this random
        state is also used to randomly select the small proportion of abnormal
        samples.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn and 'target', the regression target for each
        sample.


    References
    ----------
    .. [1] Analysis and Results of the 1999 DARPA Off-Line Intrusion
           Detection Evaluation Richard Lippmann, Joshua W. Haines,
           David J. Fried, Jonathan Korba, Kumar Das

    .. [2] K. Yamanishi, J.-I. Takeuchi, G. Williams, and P. Milne. Online
           unsupervised outlier detection using finite mixtures with
           discounting learning algorithms. In Proceedings of the sixth
           ACM SIGKDD international conference on Knowledge discovery
           and data mining, pages 320-324. ACM Press, 2000.

    
```
79 - /tmp/repos/scikit-learn/sklearn/neighbors/classification.py
```python
def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_samples = X.shape[0]
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_samples = X.shape[0]

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(X.shape[0])
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_samples, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities


class RadiusNeighborsClassifier(NeighborsBase, RadiusNeighborsMixin,
                                SupervisedIntegerMixin, ClassifierMixin):
    
```
80 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
if self._fit_method == 'brute':
            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == 'euclidean':
                dist = pairwise_distances(X, self._fit_X, 'euclidean',
                                          n_jobs=self.n_jobs, squared=True)
                radius *= radius
            else:
                dist = pairwise_distances(X, self._fit_X,
                                          self.effective_metric_,
                                          n_jobs=self.n_jobs,
                                          **self.effective_metric_params_)

            neigh_ind_list = [np.where(d <= radius)[0] for d in dist]

            # See https://github.com/numpy/numpy/issues/5456
            # if you want to understand why this is initialized this way.
            neigh_ind = np.empty(n_samples, dtype='object')
            neigh_ind[:] = neigh_ind_list

            if return_distance:
                dist_array = np.empty(n_samples, dtype='object')
                if self.effective_metric_ == 'euclidean':
                    dist_list = [np.sqrt(d[neigh_ind[i]])
                                 for i, d in enumerate(dist)]
                else:
                    dist_list = [d[neigh_ind[i]]
                                 for i, d in enumerate(dist)]
                dist_array[:] = dist_list

                results = dist_array, neigh_ind
            else:
                results = neigh_ind

        elif self._fit_method in ['ball_tree', 'kd_tree']:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'" % self._fit_method)
            results = self._tree.query_radius(X, radius,
                                              return_distance=return_distance)
            if return_distance:
                results = results[::-1]
        else:
            raise ValueError("internal: _fit_method not recognized")

        if not query_is_train:
            return results
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = results
            else:
                neigh_ind = results

            for ind, ind_neighbor in enumerate(neigh_ind):
                mask = ind_neighbor != ind

                neigh_ind[ind] = ind_neighbor[mask]
                if return_distance:
                    dist[ind] = dist[ind][mask]

            if return_distance:
                return dist, neigh_ind
            return neigh_ind

    
```
81 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params):
    """Fit estimator and predict values for a given dataset split.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    Returns
    -------
    preds : sequence
        Result of calling 'estimator.predict'

    test : array-like
        This is the value of the test parameter
    """
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    preds = estimator.predict(X_test)
    return preds, test


def _check_is_partition(locs, n):
    """Check whether locs is a reordering of the array np.arange(n)

    Parameters
    ----------
    locs : ndarray
        integer array to test
    n : int
        number of expected elements

    Returns
    -------
    is_partition : bool
        True iff sorted(locs) is range(n)
    """
    if len(locs) != n:
        return False
    hit = np.zeros(n, bool)
    hit[locs] = True
    if not np.all(hit):
        return False
    return True



```
82 - /tmp/repos/scikit-learn/sklearn/isotonic.py
```python
def _build_f(self, X, y):
        """Build the f_ interp1d function."""

        # Handle the out_of_bounds argument by setting bounds_error
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError("The argument ``out_of_bounds`` must be in "
                             "'nan', 'clip', 'raise'; got {0}"
                             .format(self.out_of_bounds))

        bounds_error = self.out_of_bounds == "raise"
        if len(y) == 1:
            # single y, constant prediction
            self.f_ = lambda x: y.repeat(x.shape)
        else:
            self.f_ = interpolate.interp1d(X, y, kind='linear',
                                           bounds_error=bounds_error)

    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        """Build the y_ IsotonicRegression."""
        check_consistent_length(X, y, sample_weight)
        X, y = [check_array(x, ensure_2d=False) for x in [X, y]]

        y = as_float_array(y)
        self._check_fit_data(X, y, sample_weight)

        # Determine increasing if auto-determination requested
        if self.increasing == 'auto':
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing

        # If sample_weights is passed, removed zero-weight values and clean
        # order
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            mask = sample_weight > 0
            X, y, sample_weight = X[mask], y[mask], sample_weight[mask]
        else:
            sample_weight = np.ones(len(y))

        order = np.lexsort((y, X))
        X, y, sample_weight = [array[order].astype(np.float64, copy=False)
                               for array in [X, y, sample_weight]]
        unique_X, unique_y, unique_sample_weight = _make_unique(
            X, y, sample_weight)

        # Store _X_ and _y_ to maintain backward compat during the deprecation
        # period of X_ and y_
        self._X_ = X = unique_X
        self._y_ = y = isotonic_regression(unique_y, unique_sample_weight,
                                           self.y_min, self.y_max,
                                           increasing=self.increasing_)

        # Handle the left and right bounds on X
        self.X_min_, self.X_max_ = np.min(X), np.max(X)

        if trim_duplicates:
            # Remove unnecessary points for faster prediction
            keep_data = np.ones((len(y),), dtype=bool)
            # Aside from the 1st and last point, remove points whose y values
            # are equal to both the point before and the point after it.
            keep_data[1:-1] = np.logical_or(
                np.not_equal(y[1:-1], y[:-2]),
                np.not_equal(y[1:-1], y[2:])
            )
            return X[keep_data], y[keep_data]
        else:
            # The ability to turn off trim_duplicates is only used to it make
            # easier to unit test that removing duplicates in y does not have
            # any impact the resulting interpolation function (besides
            # prediction speed).
            return X, y

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

    
```
83 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py
```python
def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like, shape = [n_samples]
            Labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        z : float
        """
        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis])


class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    
```
84 - /tmp/repos/scikit-learn/sklearn/discriminant_analysis.py
```python
def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

            .. versionchanged:: 0.19
               ``store_covariances`` has been moved to main constructor as
               ``store_covariance``

            .. versionchanged:: 0.19
               ``tol`` has been moved to main constructor.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('The number of classes has to be greater than'
                             ' one; got %d class' % (n_classes))
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        cov = None
        store_covariance = self.store_covariance or self.store_covariances
        if self.store_covariances:
            warnings.warn("'store_covariances' was renamed to store_covariance"
                          " in version 0.19 and will be removed in 0.21.",
                          DeprecationWarning)
        if store_covariance:
            cov = []
        means = []
        scalings = []
        rotations = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance '
                                 'is ill defined.' % str(self.classes_[ind]))
            Xgc = Xg - meang
            # Xgc = U * S * V.T
            U, S, Vt = np.linalg.svd(Xgc, full_matrices=False)
            rank = np.sum(S > self.tol)
            if rank < n_features:
                warnings.warn("Variables are collinear")
            S2 = (S ** 2) / (len(Xg) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append(np.dot(S2 * Vt.T, Vt))
            scalings.append(S2)
            rotations.append(Vt.T)
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2 ** 2, 1))
        norm2 = np.array(norm2).T   # shape = [len(X), n_classes]
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        return (-0.5 * (norm2 + u) + np.log(self.priors_))

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples (test vectors).

        Returns
        -------
        C : array, shape = [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,], giving the
            log likelihood ratio of the positive class.
        """
        dec_func = self._decision_function(X)
        # handle special case of two classes
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    
```
85 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
def _validate_for_predict(self, X):
        check_is_fitted(self, 'support_')

        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)
        n_samples, n_features = X.shape

        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))
        elif n_features != self.shape_fit_[1]:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time" %
                             (n_features, self.shape_fit_[1]))
        return X

    @property
    def coef_(self):
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a '
                                 'linear kernel')

        coef = self._get_coef()

        # coef_ being a read-only property, it's better to mark the value as
        # immutable to avoid hiding potential bugs for the unsuspecting user.
        if sp.issparse(coef):
            # sparse matrix do not have global flags
            coef.data.flags.writeable = False
        else:
            # regular dense array
            coef.flags.writeable = False
        return coef

    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)


class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):
    """ABC for LibSVM-based classifiers."""
    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, C, nu,
                 shrinking, probability, cache_size, class_weight, verbose,
                 max_iter, decision_function_shape, random_state):
        self.decision_function_shape = decision_function_shape
        super(BaseSVC, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=nu, epsilon=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            random_state=random_state)

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                " class" % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')

    def decision_function(self, X):
        """Distance of the samples X to the separating hyperplane.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X : array-like, shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes)
        """
        dec = self._decision_function(X)
        if self.decision_function_shape == 'ovr' and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec

    def predict(self, X):
        """Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        y = super(BaseSVC, self).predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp))
```
86 - /tmp/repos/scikit-learn/sklearn/decomposition/kernel_pca.py
```python
def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, accept_sparse='csr', copy=self.copy_X)
        self._centerer = KernelCenterer()
        K = self._get_kernel(X)
        self._fit_transform(K)

        if self.fit_inverse_transform:
            sqrt_lambdas = np.diag(np.sqrt(self.lambdas_))
            X_transformed = np.dot(self.alphas_, sqrt_lambdas)
            self._fit_inverse_transform(X_transformed, X)

        self.X_fit_ = X
        return self

    def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X, **params)

        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)

        if self.fit_inverse_transform:
            self._fit_inverse_transform(X_transformed, X)

        return X_transformed

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'X_fit_')

        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))
        return np.dot(K, self.alphas_ / np.sqrt(self.lambdas_))

    def inverse_transform(self, X):
        """Transform X back to original space.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)

        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        if not self.fit_inverse_transform:
            raise NotFittedError("The fit_inverse_transform parameter was not"
                                 " set to True when instantiating and hence "
                                 "the inverse transform is not available.")

        K = self._get_kernel(X, self.X_transformed_fit_)

        return np.dot(K, self.dual_coef_)

```
87 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py
```python
def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted regression values.
        """
        check_is_fitted(self, "estimator_weights_")
        X = self._validate_X_predict(X)

        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted regression values.
        """
        check_is_fitted(self, "estimator_weights_")
        X = self._validate_X_predict(X)

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)

```
88 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
if self._fit_method == 'brute':
            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == 'euclidean':
                dist = pairwise_distances(X, self._fit_X, 'euclidean',
                                          n_jobs=n_jobs, squared=True)
            else:
                dist = pairwise_distances(
                    X, self._fit_X, self.effective_metric_, n_jobs=n_jobs,
                    **self.effective_metric_params_)

            neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
            neigh_ind = neigh_ind[:, :n_neighbors]
            # argpartition doesn't guarantee sorted order, so we sort again
            neigh_ind = neigh_ind[
                sample_range, np.argsort(dist[sample_range, neigh_ind])]

            if return_distance:
                if self.effective_metric_ == 'euclidean':
                    result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
                else:
                    result = dist[sample_range, neigh_ind], neigh_ind
            else:
                result = neigh_ind

        elif self._fit_method in ['ball_tree', 'kd_tree']:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'" % self._fit_method)
            result = Parallel(n_jobs, backend='threading')(
                delayed(self._tree.query, check_pickle=False)(
                    X[s], n_neighbors, return_distance)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
            if return_distance:
                dist, neigh_ind = tuple(zip(*result))
                result = np.vstack(dist), np.vstack(neigh_ind)
            else:
                result = np.vstack(result)
        else:
            raise ValueError("internal: _fit_method not recognized")

        if not query_is_train:
            return result
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = result
            else:
                neigh_ind = result

            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False

            neigh_ind = np.reshape(
                neigh_ind[sample_mask], (n_samples, n_neighbors - 1))

            if return_distance:
                dist = np.reshape(
                    dist[sample_mask], (n_samples, n_neighbors - 1))
                return dist, neigh_ind
            return neigh_ind

    
```
89 - /tmp/repos/scikit-learn/sklearn/decomposition/dict_learning.py
```python
def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)

        U, (A, B), self.n_iter_ = dict_learning_online(
            X, self.n_components, self.alpha,
            n_iter=self.n_iter, return_code=False,
            method=self.fit_algorithm,
            n_jobs=self.n_jobs, dict_init=self.dict_init,
            batch_size=self.batch_size, shuffle=self.shuffle,
            verbose=self.verbose, random_state=random_state,
            return_inner_stats=True,
            return_n_iter=True,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code)
        self.components_ = U
        # Keep track of the state of the algorithm to be able to do
        # some online fitting (partial_fit)
        self.inner_stats_ = (A, B)
        self.iter_offset_ = self.n_iter
        return self

    def partial_fit(self, X, y=None, iter_offset=None):
        """Updates the model using the data in X as a mini-batch.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        iter_offset : integer, optional
            The number of iteration on data batches that has been
            performed before this call to partial_fit. This is optional:
            if no number is passed, the memory of the object is
            used.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        X = check_array(X)
        if hasattr(self, 'components_'):
            dict_init = self.components_
        else:
            dict_init = self.dict_init
        inner_stats = getattr(self, 'inner_stats_', None)
        if iter_offset is None:
            iter_offset = getattr(self, 'iter_offset_', 0)
        U, (A, B) = dict_learning_online(
            X, self.n_components, self.alpha,
            n_iter=self.n_iter, method=self.fit_algorithm,
            n_jobs=self.n_jobs, dict_init=dict_init,
            batch_size=len(X), shuffle=False,
            verbose=self.verbose, return_code=False,
            iter_offset=iter_offset, random_state=self.random_state_,
            return_inner_stats=True, inner_stats=inner_stats,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code)
        self.components_ = U

        # Keep track of the state of the algorithm to be able to do
        # some online fitting (partial_fit)
        self.inner_stats_ = (A, B)
        self.iter_offset_ = iter_offset + self.n_iter
        return self

```
**90 - /tmp/repos/scikit-learn/sklearn/datasets/rcv1.py**:
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
    
```
91 - /tmp/repos/scikit-learn/sklearn/multioutput.py
```python
class MultiOutputEstimator(six.with_metaclass(ABCMeta, BaseEstimator,
                                              MetaEstimatorMixin)):
    @abstractmethod
    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets.

        classes : list of numpy arrays, shape (n_outputs)
            Each array is unique classes for one output in str/int
            Can be obtained by via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where y is the
            target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        first_time = not hasattr(self, 'estimators_')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else self.estimator,
                X, y[:, i],
                classes[i] if classes is not None else None,
                sample_weight, first_time) for i in range(y.shape[1]))
        return self

    def fit(self, X, y, sample_weight=None):
        """ Fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], sample_weight)
            for i in range(y.shape[1]))
        return self

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(parallel_helper)(e, 'predict', X)
            for e in self.estimators_)

        return np.asarray(y).T



```
92 - /tmp/repos/scikit-learn/sklearn/feature_extraction/dict_vectorizer.py
```python
# Authors: Lars Buitinck
#          Dan Blanchard <dblanchard@ets.org>
# License: BSD 3 clause

from array import array
from collections import Mapping
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six.moves import xrange
from ..utils import check_array, tosequence


def _tosequence(X):
    """Turn X into a sequence or ndarray, avoiding a copy if possible."""
    if isinstance(X, Mapping):  # single sample
        return [X]
    else:
        return tosequence(X)


class DictVectorizer(BaseEstimator, TransformerMixin):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int, the DictVectorizer can be
    followed by :class:`sklearn.preprocessing.CategoricalEncoder` to complete
    binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse : boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort : boolean, optional.
        Whether ``feature_names_`` and ``vocabulary_`` should be sorted when fitting.
        True by default.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[ 2.,  0.,  1.],
           [ 0.,  1.,  3.]])
    >>> v.inverse_transform(X) == \
        [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[ 0.,  0.,  4.]])

    See also
    --------
    FeatureHasher : performs vectorization using only a hash function.
    sklearn.preprocessing.CategoricalEncoder : handles nominal/categorical
      features encoded as columns of arbitrary data types.
    """

    def __init__(self, dtype=np.float64, separator="=", sparse=True,
                 sort=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort

    
```
93 - /tmp/repos/scikit-learn/sklearn/feature_selection/mutual_info_.py
```python
def _compute_mi_cd(c, d, n_neighbors):
    """Compute mutual information between continuous and discrete variables.

    Parameters
    ----------
    c : ndarray, shape (n_samples,)
        Samples of a continuous random variable.

    d : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    n_samples = c.shape[0]
    c = c.reshape((-1, 1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    nn.set_params(algorithm='kd_tree')
    nn.fit(c)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    m_all = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) -
          np.mean(digamma(label_counts)) -
          np.mean(digamma(m_all + 1)))

    return max(0, mi)


def _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3):
    """Compute mutual information between two variables.

    This is a simple wrapper which selects a proper function to call based on
    whether `x` and `y` are discrete or not.
    """
    if x_discrete and y_discrete:
        return mutual_info_score(x, y)
    elif x_discrete and not y_discrete:
        return _compute_mi_cd(y, x, n_neighbors)
    elif not x_discrete and y_discrete:
        return _compute_mi_cd(x, y, n_neighbors)
    else:
        return _compute_mi_cc(x, y, n_neighbors)


def _iterate_columns(X, columns=None):
    """Iterate over columns of a matrix.

    Parameters
    ----------
    X : ndarray or csc_matrix, shape (n_samples, n_features)
        Matrix over which to iterate.

    columns : iterable or None, default None
        Indices of columns to iterate over. If None, iterate over all columns.

    Yields
    ------
    x : ndarray, shape (n_samples,)
        Columns of `X` in dense format.
    """
    if columns is None:
        columns = range(X.shape[1])

    if issparse(X):
        for i in columns:
            x = np.zeros(X.shape[0])
            start_ptr, end_ptr = X.indptr[i], X.indptr[i + 1]
            x[X.indices[start_ptr:end_ptr]] = X.data[start_ptr:end_ptr]
            yield x
    else:
        for i in columns:
            yield X[:, i]



```
94 - /tmp/repos/scikit-learn/sklearn/neural_network/rbm.py
```python
def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, "components_")

        v = check_array(X, accept_sparse='csr')
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)

    
```
95 - /tmp/repos/scikit-learn/sklearn/decomposition/dict_learning.py
```python
def __init__(self, n_components=None, alpha=1, max_iter=1000, tol=1e-8,
                 fit_algorithm='lars', transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 n_jobs=1, code_init=None, dict_init=None, verbose=False,
                 split_sign=False, random_state=None):

        self._set_sparse_coding_params(n_components, transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.code_init = code_init
        self.dict_init = dict_init
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the object itself
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        V, U, E, self.n_iter_ = dict_learning(
            X, n_components, self.alpha,
            tol=self.tol, max_iter=self.max_iter,
            method=self.fit_algorithm,
            n_jobs=self.n_jobs,
            code_init=self.code_init,
            dict_init=self.dict_init,
            verbose=self.verbose,
            random_state=random_state,
            return_n_iter=True)
        self.components_ = U
        self.error_ = E
        return self


class MiniBatchDictionaryLearning(BaseEstimator, SparseCodingMixin):
    
```
**96 - /tmp/repos/scikit-learn/sklearn/datasets/lfw.py**:
```python
index_file_path = join(lfw_home, label_filenames[subset])

    # load and memoize the pairs as np arrays
    pairs, target, target_names = load_func(
        index_file_path, data_folder_path, resize=resize, color=color,
        slice_=slice_)

    # pack the results as a Bunch instance
    return Bunch(data=pairs.reshape(len(pairs), -1), pairs=pairs,
                 target=target, target_names=target_names,
                 DESCR="'%s' segment of the LFW pairs dataset" % subset)

```
97 - /tmp/repos/scikit-learn/sklearn/preprocessing/_function_transformer.py
```python
def transform(self, X, y='deprecated'):
        """Transform X using the forward function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : (ignored)
            .. deprecated::0.19

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        return self._transform(X, y=y, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, X, y='deprecated'):
        """Transform X using the inverse function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : (ignored)
            .. deprecated::0.19

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on inverse_transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)
        return self._transform(X, y=y, func=self.inverse_func,
                               kw_args=self.inv_kw_args)

    def _transform(self, X, y=None, func=None, kw_args=None):
        if self.validate:
            X = check_array(X, self.accept_sparse)

        if func is None:
            func = _identity

        if (not isinstance(self.pass_y, string_types) or
                self.pass_y != 'deprecated'):
            # We do this to know if pass_y was set to False / True
            pass_y = self.pass_y
            warnings.warn("The parameter pass_y is deprecated since 0.19 and "
                          "will be removed in 0.21", DeprecationWarning)
        else:
            pass_y = False

        return func(X, *((y,) if pass_y else ()),
                    **(kw_args if kw_args else {}))

```
98 - /tmp/repos/scikit-learn/sklearn/neighbors/regression.py
```python
def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        if issparse(X) and self.metric == 'precomputed':
            raise ValueError(
                "Sparse matrices not supported for prediction with "
                "precomputed kernels. Densify your matrix."
            )
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


class RadiusNeighborsRegressor(NeighborsBase, RadiusNeighborsMixin,
                               SupervisedFloatMixin,
                               RegressorMixin):
    
```
99 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""Generate cross-validated estimates for each input data point

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

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

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

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

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y)
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

    
```
100 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
""" Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix inputs.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    See also
    --------
    paired_distances : Computes the distances between corresponding
                       elements of two arrays
    """
    if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s, or 'precomputed', or a "
                         "callable" % (metric, _VALID_METRICS))

    
```
101 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
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
102 - /tmp/repos/scikit-learn/sklearn/compose/_column_transformer.py
```python
def _get_column_indices(X, key):
    """
    Get feature column indices for input data X and key.

    For accepted values of `key`, see the docstring of _get_column

    """
    n_columns = X.shape[1]

    if (_check_key_type(key, int)
            or hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_)):
        # Convert key into positive indexes
        idx = np.arange(n_columns)[key]
        return np.atleast_1d(idx).tolist()
    elif _check_key_type(key, str):
        try:
            all_columns = list(X.columns)
        except AttributeError:
            raise ValueError("Specifying the columns using strings is only "
                             "supported for pandas DataFrames")
        if isinstance(key, str):
            columns = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is not None:
                start = all_columns.index(start)
            if stop is not None:
                # pandas indexing with strings is endpoint included
                stop = all_columns.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(range(n_columns)[slice(start, stop)])
        else:
            columns = list(key)

        return [all_columns.index(col) for col in columns]
    else:
        raise ValueError("No valid specification of the columns. Only a "
                         "scalar, list or slice of all integers or all "
                         "strings, or boolean mask is allowed")


def _is_empty_column_selection(column):
    """
    Return True if the column selection is empty (empty list or all-False
    boolean array).

    """
    if hasattr(column, 'dtype') and np.issubdtype(column.dtype, np.bool_):
        return not column.any()
    elif hasattr(column, '__len__'):
        return len(column) == 0
    else:
        return False


def _get_transformer_list(estimators):
    """
    Construct (name, trans, column) tuples from list

    """
    transformers, columns = zip(*estimators)
    names, _ = zip(*_name_estimators(transformers))

    transformer_list = list(zip(names, transformers, columns))
    return transformer_list



```
**103 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
return Bunch(data=X, target=y)


def _mkdirp(d):
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

```
104 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
""" Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix inputs.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    See also
    --------
    pairwise_distances_chunked : performs the same calculation as this
        function, but returns a generator of chunks of the distance matrix, in
        order to limit memory usage.
    paired_distances : Computes the distances between corresponding
                       elements of two arrays
    """
    if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s, or 'precomputed', or a "
                         "callable" % (metric, _VALID_METRICS))

    
```
105 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class _BaseKFold(with_metaclass(ABCMeta, BaseCrossValidator)):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

        for train, test in super(_BaseKFold, self).split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

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
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits



```
106 - /tmp/repos/scikit-learn/sklearn/multioutput.py
```python
class _BaseChain(six.with_metaclass(ABCMeta, BaseEstimator)):
    def __init__(self, base_estimator, order=None, cv=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, Y):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples, n_classes)
            The target values.

        Returns
        -------
        self : object
        """
        X, Y = check_X_y(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        check_array(X, accept_sparse=True)
        self.order_ = self.order
        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == 'random':
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
                raise ValueError("invalid order")

        self.estimators_ = [clone(self.base_estimator)
                            for _ in range(Y.shape[1])]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format='lil')
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format='lil')

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(X_aug[:, :(X.shape[1] + chain_idx)], y)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx],
                    y=y, cv=self.cv)
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result

        return self

    def predict(self, X):
        """Predict on the data matrix X using the ClassifierChain model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_pred : array-like, shape (n_samples, n_classes)
            The predicted values.

        """
        X = check_array(X, accept_sparse=True)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                if chain_idx == 0:
                    X_aug = X
                else:
                    X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]

        return Y_pred


class ClassifierChain(_BaseChain, ClassifierMixin, MetaEstimatorMixin):
    
```
107 - /tmp/repos/scikit-learn/sklearn/naive_bayes.py
```python
def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    # XXX The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)



```
108 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
"""
Generate samples of synthetic data sets.
"""

# Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Nothman
# License: BSD 3 clause

import numbers
import array
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from collections import Iterable

from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils.random import sample_without_replacement
from ..externals import six
map = six.moves.map
zip = six.moves.zip


def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions
    """
    if dimensions > 30:
        return np.hstack([rng.randint(2, size=(samples, dimensions - 30)),
                          _generate_hypercube(samples, 30, rng)])
    out = sample_without_replacement(2 ** dimensions, samples,
                                     random_state=rng).astype(dtype='>u4',
                                                              copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out


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
109 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def linear_kernel(X, Y=None):
    """
    Compute the linear kernel between X and Y.

    Read more in the :ref:`User Guide <linear_kernel>`.

    Parameters
    ----------
    X : array of shape (n_samples_1, n_features)

    Y : array of shape (n_samples_2, n_features)

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    return safe_sparse_dot(X, Y.T, dense_output=True)


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y::

        K(X, Y) = (gamma <X, Y> + coef0)^degree

    Read more in the :ref:`User Guide <polynomial_kernel>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 3

    gamma : float, default None
        if None, defaults to 1.0 / n_features

    coef0 : int, default 1

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
    return K


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """
    Compute the sigmoid kernel between X and Y::

        K(X, Y) = tanh(gamma <X, Y> + coef0)

    Read more in the :ref:`User Guide <sigmoid_kernel>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    gamma : float, default None
        If None, defaults to 1.0 / n_features

    coef0 : int, default 1

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    np.tanh(K, K)   # compute tanh in-place
    return K


def rbf_kernel(X, Y=None, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.

    Read more in the :ref:`User Guide <rbf_kernel>`.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float, default None
        If None, defaults to 1.0 / n_features

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K



```
110 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exageration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                          **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] Error after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : Ignored
        """
        self.fit_transform(X)
        return self

```
**111 - /tmp/repos/scikit-learn/sklearn/datasets/covtype.py**:
```python
"""Forest covertype dataset.

A classic dataset for classification benchmarks, featuring categorical and
real-valued features.

The dataset page is available from UCI Machine Learning Repository

    http://archive.ics.uci.edu/ml/datasets/Covertype

Courtesy of Jock A. Blackard and Colorado State University.
"""

# Author: Lars Buitinck
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
# License: BSD 3 clause

from gzip import GzipFile
import logging
from os.path import exists, join
from os import remove

import numpy as np

from .base import get_data_home
from .base import _fetch_remote
from .base import RemoteFileMetadata
from ..utils import Bunch
from .base import _pkl_filepath
from ..utils.fixes import makedirs
from ..externals import joblib
from ..utils import check_random_state

# The original data can be found in:
# http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
ARCHIVE = RemoteFileMetadata(
    filename='covtype.data.gz',
    url='https://ndownloader.figshare.com/files/5976039',
    checksum=('614360d0257557dd1792834a85a1cdeb'
              'fadc3c4f30b011d56afee7ffb5b15771'))

logger = logging.getLogger(__name__)


def fetch_covtype(data_home=None, download_if_missing=True,
                  random_state=None, shuffle=False):
    """Load the covertype dataset, downloading it if necessary.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

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

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest covertypes with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest covertype dataset.

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
        y = Xy[:, -1].astype(np.int32)

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

    return Bunch(data=X, target=y, DESCR=__doc__)

```
112 - /tmp/repos/scikit-learn/sklearn/base.py
```python
class BiclusterMixin(object):
    """Mixin class for all bicluster estimators in scikit-learn"""

    @property
    def biclusters_(self):
        """Convenient way to get row and column indicators together.

        Returns the ``rows_`` and ``columns_`` members.
        """
        return self.rows_, self.columns_

    def get_indices(self, i):
        """Row and column indices of the i'th bicluster.

        Only works if ``rows_`` and ``columns_`` attributes exist.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        row_ind : np.array, dtype=np.intp
            Indices of rows in the dataset that belong to the bicluster.
        col_ind : np.array, dtype=np.intp
            Indices of columns in the dataset that belong to the bicluster.

        """
        rows = self.rows_[i]
        columns = self.columns_[i]
        return np.nonzero(rows)[0], np.nonzero(columns)[0]

    def get_shape(self, i):
        """Shape of the i'th bicluster.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        shape : (int, int)
            Number of rows and columns (resp.) in the bicluster.
        """
        indices = self.get_indices(i)
        return tuple(len(i) for i in indices)

    def get_submatrix(self, i, data):
        """Returns the submatrix corresponding to bicluster `i`.

        Parameters
        ----------
        i : int
            The index of the cluster.
        data : array
            The data.

        Returns
        -------
        submatrix : array
            The submatrix corresponding to bicluster i.

        Notes
        -----
        Works with sparse matrices. Only works if ``rows_`` and
        ``columns_`` attributes exist.
        """
        from .utils.validation import check_array
        data = check_array(data, accept_sparse='csr')
        row_ind, col_ind = self.get_indices(i)
        return data[row_ind[:, np.newaxis], col_ind]


###############################################################################
class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class DensityMixin(object):
    """Mixin class for all density estimators in scikit-learn."""
    _estimator_type = "DensityEstimator"

    def score(self, X, y=None):
        """Returns the score of the model on the data X

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        score : float
        """
        pass


class OutlierMixin(object):
    """Mixin class for all outlier detection estimators in scikit-learn."""
    _estimator_type = "outlier_detector"

    def fit_predict(self, X, y=None):
        """Performs outlier detection on X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X).predict(X)


###############################################################################
class MetaEstimatorMixin(object):
    """Mixin class for all meta estimators in scikit-learn."""
    # this is just a tag for the moment


###############################################################################


```
113 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
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

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        check_is_fitted(self, 'estimator_')
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_
```
114 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python
def fit(self, X, y, Xy=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Xy : array-like, shape (n_samples,) or (n_samples, n_targets), \
                optional
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.

        Returns
        -------
        self : object
            returns an instance of self.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)

        alpha = getattr(self, 'alpha', 0.)
        if hasattr(self, 'n_nonzero_coefs'):
            alpha = 0.  # n_nonzero_coefs parametrization takes priority
            max_iter = self.n_nonzero_coefs
        else:
            max_iter = self.max_iter

        self._fit(X, y, max_iter=max_iter, alpha=alpha, fit_path=self.fit_path,
                  Xy=Xy)

        return self


class LassoLars(Lars):
    """Lasso model fit with Least Angle Regression a.k.a. Lars

    It is a Linear Model trained with an L1 prior as regularizer.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`. For numerical reasons, using
        ``alpha = 0`` with the LassoLars object is not advised and you
        should prefer the LinearRegression object.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : integer, optional
        Maximum number of iterations to perform.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    fit_path : boolean
        If ``True`` the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients will not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.

    Attributes
    ----------
    alphas_ : array, shape (n_alphas + 1,) | list of n_targets such arrays
        Maximum of covariances (in absolute value) at each iteration. 
```
115 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
def _get_json_content_from_openml_api(url, error_message, raise_if_error,
                                      data_home):
    """
    Loads json data from the openml api

    Parameters
    ----------
    url : str
        The URL to load from. Should be an official OpenML endpoint

    error_message : str or None
        The error message to raise if an acceptable OpenML error is thrown
        (acceptable error is, e.g., data id not found. Other errors, like 404's
        will throw the native error message)

    raise_if_error : bool
        Whether to raise an error if OpenML returns an acceptable error (e.g.,
        date not found). If this argument is set to False, a None is returned
        in case of acceptable errors. Note that all other errors (e.g., 404)
        will still be raised as normal.

    data_home : str or None
        Location to cache the response. None if no cache is required.

    Returns
    -------
    json_data : json or None
        the json result from the OpenML server if the call was successful;
        None otherwise iff raise_if_error was set to False and the error was
        ``acceptable``
    """

    @_retry_with_clean_cache(url, data_home)
    def _load_json():
        with closing(_open_openml_url(url, data_home)) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return _load_json()
    except HTTPError as error:
        # 412 is an OpenML specific error code, indicating a generic error
        # (e.g., data not found)
        if error.code != 412:
            raise error

    # 412 error, not in except for nicer traceback
    if raise_if_error:
        raise ValueError(error_message)
    return None


def _split_sparse_columns(arff_data, include_columns):
    """
    obtains several columns from sparse arff representation. Additionally, the
    column indices are re-labelled, given the columns that are not included.
    (e.g., when including [1, 2, 3], the columns will be relabelled to
    [0, 1, 2])

    Parameters
    ----------
    arff_data : tuple
        A tuple of three lists of equal size; first list indicating the value,
        second the x coordinate and the third the y coordinate.

    include_columns : list
        A list of columns to include.

    Returns
    -------
    arff_data_new : tuple
        Subset of arff data with only the include columns indicated by the
        include_columns argument.
    """
    arff_data_new = (list(), list(), list())
    reindexed_columns = {column_idx: array_idx for array_idx, column_idx
                         in enumerate(include_columns)}
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            arff_data_new[0].append(val)
            arff_data_new[1].append(row_idx)
            arff_data_new[2].append(reindexed_columns[col_idx])
    return arff_data_new


def _sparse_data_to_array(arff_data, include_columns):
    # turns the sparse data back into an array (can't use toarray() function,
    # as this does only work on numeric data)
    num_obs = max(arff_data[1]) + 1
    y_shape = (num_obs, len(include_columns))
    reindexed_columns = {column_idx: array_idx for array_idx, column_idx
                         in enumerate(include_columns)}
    # TODO: improve for efficiency
    y = np.empty(y_shape, dtype=np.float64)
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            y[row_idx, reindexed_columns[col_idx]] = val
    return y



```
116 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """

        rnd = check_random_state(self.random_state)

        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)

        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        y = self._validate_targets(y)

        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)
        solver_type = LIBSVM_IMPL.index(self._impl)

        # input validation
        if solver_type != 2 and X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != X.shape[0]:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                             % (sample_weight.shape, X.shape))

        if self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:  # pragma: no cover
            print('[LibSVM]', end='')

        seed = rnd.randint(np.iinfo('i').max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
        # see comment on the other call to np.iinfo in this file

        self.shape_fit_ = X.shape

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_ internally.
        self._intercept_ = self.intercept_.copy()
        self._dual_coef_ = self.dual_coef_
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            self.intercept_ *= -1
            self.dual_coef_ = -self.dual_coef_

        return self

    def _validate_targets(self, y):
        """Validation of y and class_weight.

        Default implementation for SVR and one-class; overridden in BaseSVC.
        """
        # XXX this is ugly.
        # Regression models should not have a class_weight_ attribute.
        self.class_weight_ = np.empty(0)
        return column_or_1d(y, warn=True).astype(np.float64)

    def _warn_from_fit_status(self):
        assert self.fit_status_ in (0, 1)
        if self.fit_status_ == 1:
            warnings.warn('Solver terminated early (max_iter=%i).'
                          '  Consider pre-processing your data with'
                          ' StandardScaler or MinMaxScaler.'
                          % self.max_iter, ConvergenceWarning)

    
```
117 - /tmp/repos/scikit-learn/sklearn/neighbors/classification.py
```python
def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.

        """
        X = check_array(X, accept_sparse='csr')
        n_samples = X.shape[0]

        neigh_dist, neigh_ind = self.radius_neighbors(X)
        inliers = [i for i, nind in enumerate(neigh_ind) if len(nind) != 0]
        outliers = [i for i, nind in enumerate(neigh_ind) if len(nind) == 0]

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]
        n_outputs = len(classes_)

        if self.outlier_label is not None:
            neigh_dist[outliers] = 1e-6
        elif outliers:
            raise ValueError('No neighbors found for test samples %r, '
                             'you can try using larger radius, '
                             'give a label for outliers, '
                             'or consider removing them from your dataset.'
                             % outliers)

        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            pred_labels = np.zeros(len(neigh_ind), dtype=object)
            pred_labels[:] = [_y[ind, k] for ind in neigh_ind]
            if weights is None:
                mode = np.array([stats.mode(pl)[0]
                                 for pl in pred_labels[inliers]], dtype=np.int)
            else:
                mode = np.array([weighted_mode(pl, w)[0]
                                 for (pl, w)
                                 in zip(pred_labels[inliers], weights[inliers])],
                                dtype=np.int)

            mode = mode.ravel()

            y_pred[inliers, k] = classes_k.take(mode)

        if outliers:
            y_pred[outliers, :] = self.outlier_label

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

```
118 - /tmp/repos/scikit-learn/sklearn/compose/_column_transformer.py
```python
def _hstack(self, Xs):
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : List of numpy arrays, sparse arrays, or DataFrames
        """
        if self.sparse_output_:
            try:
                # since all columns should be numeric before stacking them
                # in a sparse matrix, `check_array` is used for the
                # dtype conversion if necessary.
                converted_Xs = [check_array(X,
                                            accept_sparse=True,
                                            force_all_finite=False)
                                for X in Xs]
            except ValueError:
                raise ValueError("For a sparse output, all columns should"
                                 " be a numeric or convertible to a numeric.")

            return sparse.hstack(converted_Xs).tocsr()
        else:
            Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
            return np.hstack(Xs)


def _check_X(X):
    """Use check_array only on lists and other non-array-likes / sparse"""
    if hasattr(X, '__array__') or sparse.issparse(X):
        return X
    return check_array(X, force_all_finite='allow-nan', dtype=np.object)


def _check_key_type(key, superclass):
    """
    Check that scalar, list or slice is of a certain type.

    This is only used in _get_column and _get_column_indices to check
    if the `key` (column specification) is fully integer or fully string-like.

    Parameters
    ----------
    key : scalar, list, slice, array-like
        The column specification to check
    superclass : int or str
        The type for which to check the `key`

    """
    if isinstance(key, superclass):
        return True
    if isinstance(key, slice):
        return (isinstance(key.start, (superclass, type(None))) and
                isinstance(key.stop, (superclass, type(None))))
    if isinstance(key, list):
        return all(isinstance(x, superclass) for x in key)
    if hasattr(key, 'dtype'):
        if superclass is int:
            return key.dtype.kind == 'i'
        else:
            # superclass = str
            return key.dtype.kind in ('O', 'U', 'S')
    return False


def _get_column(X, key):
    """
    Get feature column(s) from input data X.

    Supported input types (X): numpy arrays, sparse arrays and DataFrames

    Supported key types (key):
    - scalar: output is 1D
    - lists, slices, boolean masks: output is 2D
    - callable that returns any of the above

    Supported key data types:

    - integer or boolean mask (positional):
        - supported for arrays, sparse matrices and dataframes
    - string (key-based):
        - only supported for dataframes
        - So no keys other than strings are allowed (while in principle you
          can use any hashable object as key).

    """
    # check whether we have string column names or integers
    if _check_key_type(key, int):
        column_names = False
    elif _check_key_type(key, str):
        column_names = True
    elif hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_):
        # boolean mask
        column_names = False
        if hasattr(X, 'loc'):
            # pandas boolean masks don't work with iloc, so take loc path
            column_names = True
    else:
        raise ValueError("No valid specification of the columns. Only a "
                         "scalar, list or slice of all integers or all "
                         "strings, or boolean mask is allowed")

    if column_names:
        if hasattr(X, 'loc'):
            # pandas dataframes
            return X.loc[:, key]
        else:
            raise ValueError("Specifying the columns using strings is only "
                             "supported for pandas DataFrames")
    else:
        if hasattr(X, 'iloc'):
            # pandas dataframes
            return X.iloc[:, key]
        else:
            # numpy arrays, sparse arrays
            return X[:, key]



```
119 - /tmp/repos/scikit-learn/sklearn/linear_model/base.py
```python
def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True,
                     sample_weight=None, return_mean=False):
    """
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output

        X = (X - X_offset) / X_scale

    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).

    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    """

    if isinstance(sample_weight, numbers.Number):
        sample_weight = None

    X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'],
                    dtype=FLOAT_DTYPES)
    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0)
            if not return_mean:
                X_offset[:] = X.dtype.type(0)

            if normalize:

                # TODO: f_normalize could be used here as well but the function
                # inplace_csr_row_normalize_l2 must be changed such that it
                # can return also the norms computed internally

                # transform variance to norm in-place
                X_var *= X.shape[0]
                X_scale = np.sqrt(X_var, X_var)
                del X_var
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)

        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False,
                                         return_norm=True)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


# TODO: _rescale_data should be factored into _preprocess_data.
# Currently, the fact that sag implements its own way to deal with
# sample_weight makes the refactoring tricky.

def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = sample_weight * np.ones(n_samples)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                  shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y



```
120 - /tmp/repos/scikit-learn/sklearn/decomposition/pca.py
```python
def fit(self, X, y=None):
        """Fit the model with X by extracting the first principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(check_array(X))
        return self

    def _fit(self, X):
        """Fit the model to the data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            The input data, copied, centered and whitened when requested.
        """
        random_state = check_random_state(self.random_state)
        X = np.atleast_2d(as_float_array(X, copy=self.copy))

        n_samples = X.shape[0]

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        U, S, V = randomized_svd(X, n_components,
                                 n_iter=self.iterated_power,
                                 random_state=random_state)

        self.explained_variance_ = exp_var = (S ** 2) / (n_samples - 1)
        full_var = np.var(X, ddof=1, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = S  # Store the singular values.

        if self.whiten:
            self.components_ = V / S[:, np.newaxis] * sqrt(n_samples)
        else:
            self.components_ = V

        return X

    def transform(self, X):
        """Apply dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_

        X = np.dot(X, self.components_.T)
        return X

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        X = check_array(X)
        X = self._fit(X)
        return np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples in the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        If whitening is enabled, inverse_transform does not compute the
        exact inverse operation of transform.
        """
        check_is_fitted(self, 'mean_')

        X_original = np.dot(X, self.components_)
        if self.mean_ is not None:
            X_original = X_original + self.mean_
        return X_original

```
121 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
def _valid_data_column_names(features_list, target_columns):
    # logic for determining on which columns can be learned. Note that from the
    # OpenML guide follows that columns that have the `is_row_identifier` or
    # `is_ignore` flag, these can not be learned on. Also target columns are
    # excluded.
    valid_data_column_names = []
    for feature in features_list:
        if (feature['name'] not in target_columns
                and feature['is_ignore'] != 'true'
                and feature['is_row_identifier'] != 'true'):
            valid_data_column_names.append(feature['name'])
    return valid_data_column_names


def fetch_openml(name=None, version='active', data_id=None, data_home=None,
                 target_column='default-target', cache=True, return_X_y=False):
    
```
122 - /tmp/repos/scikit-learn/sklearn/gaussian_process/kernels.py
```python


    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("kernel__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.kernel == b.kernel and self.exponent == b.exponent)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K, K_gradient = self.kernel(X, Y, eval_gradient=True)
            K_gradient *= \
                self.exponent * K[:, :, np.newaxis] ** (self.exponent - 1)
            return K ** self.exponent, K_gradient
        else:
            K = self.kernel(X, Y, eval_gradient=False)
            return K ** self.exponent

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.kernel.diag(X) ** self.exponent

    def __repr__(self):
        return "{0} ** {1}".format(self.kernel, self.exponent)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.kernel.is_stationary()



```
123 - /tmp/repos/scikit-learn/sklearn/kernel_approximation.py
```python
def fit(self, X, y=None):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Training data.
        """
        X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        basis_kernel = pairwise_kernels(basis, metric=self.kernel,
                                        filter_params=True,
                                        **self._get_kernel_params())

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = inds
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array, shape=(n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, 'components_')
        X = check_array(X, accept_sparse='csr')

        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(X, self.components_,
                                    metric=self.kernel,
                                    filter_params=True,
                                    **kernel_params)
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel):
            for param in (KERNEL_PARAMS[self.kernel]):
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        else:
            if (self.gamma is not None or
                    self.coef0 is not None or
                    self.degree is not None):
                warnings.warn(
                    "Passing gamma, coef0 or degree to Nystroem when using a"
                    " callable kernel is deprecated in version 0.19 and will"
                    " raise an error in 0.21, as they are ignored. Use "
                    "kernel_params instead.", DeprecationWarning)

        return params

```
124 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py
```python
def predict(self, X):
        """
        Predict data using the ``centroids_`` of subclusters.

        Avoid computation of the row norms of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : ndarray, shape(n_samples)
            Labelled data.
        """
        X = check_array(X, accept_sparse='csr')
        self._check_fit(X)
        reduced_distance = safe_sparse_dot(X, self.subcluster_centers_.T)
        reduced_distance *= -2
        reduced_distance += self._subcluster_norms
        return self.subcluster_labels_[np.argmin(reduced_distance, axis=1)]

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

    def _global_clustering(self, X=None):
        """
        Global clustering for the subclusters obtained after fitting
        """
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = (X is not None) and self.compute_labels

        # Preprocessing for the global clustering.
        not_enough_centroids = False
        if isinstance(clusterer, int):
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters)
            # There is no need to perform the global clustering step.
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        elif (clusterer is not None and not
              hasattr(clusterer, 'fit_predict')):
            raise ValueError("n_clusters should be an instance of "
                             "ClusterMixin or an int")

        # To use in predict to avoid recalculation.
        self._subcluster_norms = row_norms(
            self.subcluster_centers_, squared=True)

        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                warnings.warn(
                    "Number of subclusters found (%d) by Birch is less "
                    "than (%d). Decrease the threshold."
                    % (len(centroids), self.n_clusters), ConvergenceWarning)
        else:
            # The global clustering step that clusters the subclusters of
            # the leaves. It assumes the centroids of the subclusters as
            # samples and finds the final centroids.
            self.subcluster_labels_ = clusterer.fit_predict(
                self.subcluster_centers_)

        if compute_labels:
            self.labels_ = self.predict(X)

```
125 - /tmp/repos/scikit-learn/sklearn/isotonic.py
```python
# Authors: Fabian Pedregosa <fabian@fseoane.net>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from .base import BaseEstimator, TransformerMixin, RegressorMixin
from .utils import as_float_array, check_array, check_consistent_length
from .utils import deprecated
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
import warnings
import math


__all__ = ['check_increasing', 'isotonic_regression',
           'IsotonicRegression']


def check_increasing(x, y):
    """Determine whether y is monotonically correlated with x.

    y is found increasing or decreasing with respect to x based on a Spearman
    correlation test.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
            Training data.

    y : array-like, shape=(n_samples,)
        Training target.

    Returns
    -------
    increasing_bool : boolean
        Whether the relationship is increasing or decreasing.

    Notes
    -----
    The Spearman correlation coefficient is estimated from the data, and the
    sign of the resulting estimate is used as the result.

    In the event that the 95% confidence interval based on Fisher transform
    spans zero, a warning is raised.

    References
    ----------
    Fisher transformation. Wikipedia.
    https://en.wikipedia.org/wiki/Fisher_transformation
    """

    # Calculate Spearman rho estimate and set return accordingly.
    rho, _ = spearmanr(x, y)
    increasing_bool = rho >= 0

    # Run Fisher transform to get the rho CI, but handle rho=+/-1
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1. + rho) / (1. - rho))
        F_se = 1 / math.sqrt(len(x) - 3)

        # Use a 95% CI, i.e., +/-1.96 S.E.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)

        # Warn if the CI spans zero.
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn("Confidence interval of the Spearman "
                          "correlation coefficient spans zero. "
                          "Determination of ``increasing`` may be "
                          "suspect.")

    return increasing_bool



```
126 - /tmp/repos/scikit-learn/sklearn/utils/multiclass.py
```python
# Author: Arnaud Joly, Joel Nothman, Hamzeh Alsalhi
#
# License: BSD 3 clause
"""
Multi-class / multi-label utility function
==========================================

"""
from __future__ import division
from collections import Sequence
from itertools import chain

from scipy.sparse import issparse
from scipy.sparse.base import spmatrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

import numpy as np

from ..externals.six import string_types
from .validation import check_array



def _unique_multiclass(y):
    if hasattr(y, '__array__'):
        return np.unique(np.asarray(y))
    else:
        return set(y)


def _unique_indicator(y):
    return np.arange(check_array(y, ['csr', 'csc', 'coo']).shape[1])


_FN_UNIQUE_LABELS = {
    'binary': _unique_multiclass,
    'multiclass': _unique_multiclass,
    'multilabel-indicator': _unique_indicator,
}


def unique_labels(*ys):
    """Extract an ordered array of unique labels

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    if not ys:
        raise ValueError('No argument has been passed.')
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == set(["binary", "multiclass"]):
        ys_types = set(["multiclass"])

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Check consistency for the indicator format
    if (label_type == "multilabel-indicator" and
            len(set(check_array(y, ['csr', 'csc', 'coo']).shape[1]
                    for y in ys)) > 1):
        raise ValueError("Multi-label binary indicator input with "
                         "different numbers of labels")

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

    # Check that we don't mix string type with number type
    if (len(set(isinstance(label, string_types) for label in ys_labels)) > 1):
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))


def _is_integral_float(y):
    return y.dtype.kind == 'f' and np.all(y.astype(int) == y)



```
127 - /tmp/repos/scikit-learn/sklearn/pipeline.py
```python
def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : FeatureUnion
            This estimator
        """
        transformers = self._parallel_func(X, y, {}, _fit_one)
        if not transformers:
            # All transformers are None
            return self

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(step %d of %d) Processing %s' % (idx, total, name)

    def _parallel_func(self, X, y, fit_params, func):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = list(self._iter())

        return Parallel(n_jobs=self.n_jobs)(delayed(func)(
            transformer, X, y, weight,
            message_clsname='FeatureUnion',
            message=self._log_message(name, idx, len(transformers)),
            **fit_params) for idx, (name, transformer,
                                    weight) in enumerate(transformers, 1))

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [(name, old if old is None or old == 'drop'
                                     else next(transformers))
                                    for name, old in self.transformer_list]



```
128 - /tmp/repos/scikit-learn/sklearn/impute.py
```python
def transform(self, X):
        """Imputes all missing values in X.

        Note that this is stochastic, and that if random_state is not fixed,
        repeated calls, or permuted input, will yield different results.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        check_is_fitted(self, 'initial_imputer_')

        X, X_filled, mask_missing_values = self._initial_imputation(X)

        # edge case: in case the user specifies 0 for n_imputations,
        # then there is no need to do burn in and the result should be
        # just the initial imputation (before clipping)
        if self.n_imputations < 1:
            return X_filled

        X_filled = np.clip(X_filled, self._min_value, self._max_value)

        n_rounds = self.n_burn_in + self.n_imputations
        n_imputations = len(self.imputation_sequence_)
        imputations_per_round = n_imputations // n_rounds
        i_rnd = 0
        Xt = np.zeros(X.shape, dtype=X.dtype)
        if self.verbose > 0:
            print("[ChainedImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        for it, predictor_triplet in enumerate(self.imputation_sequence_):
            X_filled, _ = self._impute_one_feature(
                X_filled,
                mask_missing_values,
                predictor_triplet.feat_idx,
                predictor_triplet.neighbor_feat_idx,
                predictor=predictor_triplet.predictor,
                fit_mode=False
            )
            if not (it + 1) % imputations_per_round:
                if i_rnd >= self.n_burn_in:
                    Xt += X_filled
                if self.verbose > 1:
                    print('[ChainedImputer] Ending imputation round '
                          '%d/%d, elapsed time %0.2f'
                          % (i_rnd + 1, n_rounds, time() - start_t))
                i_rnd += 1

        Xt /= self.n_imputations
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt

    def fit(self, X, y=None):
        """Fits the imputer on X and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        self.fit_transform(X)
        return self


class MissingIndicator(BaseEstimator, TransformerMixin):
    
```
129 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def pairwise_kernels(X, Y=None, metric="linear", filter_params=False,
                     n_jobs=1, **kwds):
    """Compute the kernel between arrays X and optional array Y.

    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for metric are::
        ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise kernels between samples, or a feature array.

    Y : array [n_samples_b, n_features]
        A second feature array only if X has shape [n_samples_a, n_features].

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    filter_params : boolean
        Whether to filter invalid parameters or not.

    n_jobs : int
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.

    """
    # import GPKernel locally to prevent circular imports
    from ..gaussian_process.kernels import Kernel as GPKernel

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        func = metric.__call__
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = dict((k, kwds[k]) for k in kwds
                        if k in KERNEL_PARAMS[metric])
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)

```
130 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        """Computes the (weighted) graph of Neighbors for points in X

        Neighborhoods are restricted the points at a distance lower than
        radius.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features], optional
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float
            Radius of neighborhoods.
            (default is the value passed to the constructor).

        mode : {'connectivity', 'distance'}, optional
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are Euclidean distance between points.

        Returns
        -------
        A : sparse matrix in CSR format, shape = [n_samples, n_samples]
            A[i, j] is assigned the weight of edge that connects i to j.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(radius=1.5)
        >>> neigh.fit(X) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> A = neigh.radius_neighbors_graph(X)
        >>> A.toarray()
        array([[ 1.,  0.,  1.],
               [ 0.,  1.,  0.],
               [ 1.,  0.,  1.]])

        See also
        --------
        kneighbors_graph
        """
        if X is not None:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        n_samples2 = self._fit_X.shape[0]
        if radius is None:
            radius = self.radius

        # construct CSR matrix representation of the NN graph
        if mode == 'connectivity':
            A_ind = self.radius_neighbors(X, radius,
                                          return_distance=False)
            A_data = None
        elif mode == 'distance':
            dist, A_ind = self.radius_neighbors(X, radius,
                                                return_distance=True)
            A_data = np.concatenate(list(dist))
        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                'or "distance" but got %s instead' % mode)

        n_samples1 = A_ind.shape[0]
        n_neighbors = np.array([len(a) for a in A_ind])
        A_ind = np.concatenate(list(A_ind))
        if A_data is None:
            A_data = np.ones(len(A_ind))
        A_indptr = np.concatenate((np.zeros(1, dtype=int),
                                   np.cumsum(n_neighbors)))

        return csr_matrix((A_data, A_ind, A_indptr),
                          shape=(n_samples1, n_samples2))


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
131 - /tmp/repos/scikit-learn/sklearn/manifold/locally_linear.py
```python
def __init__(self, n_neighbors=5, n_components=2, reg=1E-3,
                 eigen_solver='auto', tol=1E-6, max_iter=100,
                 method='standard', hessian_tol=1E-4, modified_tol=1E-12,
                 neighbors_algorithm='auto', random_state=None, n_jobs=1):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def _fit_transform(self, X):
        self.nbrs_ = NearestNeighbors(self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      n_jobs=self.n_jobs)

        random_state = check_random_state(self.random_state)
        X = check_array(X, dtype=float)
        self.nbrs_.fit(X)
        self.embedding_, self.reconstruction_error_ = \
            locally_linear_embedding(
                self.nbrs_, self.n_neighbors, self.n_components,
                eigen_solver=self.eigen_solver, tol=self.tol,
                max_iter=self.max_iter, method=self.method,
                hessian_tol=self.hessian_tol, modified_tol=self.modified_tol,
                random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)

    def fit(self, X, y=None):
        """Compute the embedding vectors for data X

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.

        y: Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Compute the embedding vectors for data X and transform X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.

        y: Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """
        Transform new points into embedding space.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array, shape = [n_samples, n_components]

        Notes
        -----
        Because of scaling performed by this method, it is discouraged to use
        it together with methods that are not scale-invariant (like SVMs)
        """
        check_is_fitted(self, "nbrs_")

        X = check_array(X)
        ind = self.nbrs_.kneighbors(X, n_neighbors=self.n_neighbors,
                                    return_distance=False)
        weights = barycenter_weights(X, self.nbrs_._fit_X[ind],
                                     reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new

```
132 - /tmp/repos/scikit-learn/sklearn/linear_model/ridge.py
```python
def _solve_cholesky(X, y, alpha):
    # w = inv(X^t X + alpha*Id) * X.T y
    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    A = safe_sparse_dot(X.T, X, dense_output=True)
    Xy = safe_sparse_dot(X.T, y, dense_output=True)

    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])

    if one_alpha:
        A.flat[::n_features + 1] += alpha[0]
        return linalg.solve(A, Xy, sym_pos=True,
                            overwrite_a=True).T
    else:
        coefs = np.empty([n_targets, n_features], dtype=X.dtype)
        for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
            A.flat[::n_features + 1] += current_alpha
            coef[:] = linalg.solve(A, target, sym_pos=True,
                                   overwrite_a=False).ravel()
            A.flat[::n_features + 1] -= current_alpha
        return coefs


def _solve_cholesky_kernel(K, y, alpha, sample_weight=None, copy=False):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    if copy:
        K = K.copy()

    alpha = np.atleast_1d(alpha)
    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = np.sqrt(np.atleast_1d(sample_weight))
        y = y * sw[:, np.newaxis]
        K *= np.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[::n_samples + 1] += alpha[0]

        try:
            # Note: we must use overwrite_a=False in order to be able to
            #       use the fall-back solution below in case a LinAlgError
            #       is raised
            dual_coef = linalg.solve(K, y, sym_pos=True,
                                     overwrite_a=False)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in solving dual problem. Using "
                          "least-squares solution instead.")
            dual_coef = linalg.lstsq(K, y)[0]

        # K is expensive to compute and store in memory so change it back in
        # case it was user-given.
        K.flat[::n_samples + 1] -= alpha[0]

        if has_sw:
            dual_coef *= sw[:, np.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = np.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[::n_samples + 1] += current_alpha

            dual_coef[:] = linalg.solve(K, target, sym_pos=True,
                                        overwrite_a=False).ravel()

            K.flat[::n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[np.newaxis, :]

        return dual_coefs.T


def _solve_svd(X, y, alpha):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, np.newaxis]
    UTy = np.dot(U.T, y)
    d = np.zeros((s.size, alpha.size), dtype=X.dtype)
    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    d_UT_y = d * UTy
    return np.dot(Vt.T, d_UT_y).T


def ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                     max_iter=None, tol=1e-3, verbose=0, random_state=None,
                     return_n_iter=False, return_intercept=False):
    """Solve the ridge equation by the method of normal equations.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix, LinearOperator},
        shape = [n_samples, n_features]
        Training data

    y : array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values

    alpha : {float, array-like},
        shape = [n_targets] if array-like
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number.

    sample_weight : float or numpy array of shape [n_samples]
        Individual weights for each sample. If sample_weight is not None and
        solver='auto', the solver will be set to 'cholesky'.

        .. versionadded:: 0.17

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than
          'cholesky'.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution via a Cholesky decomposition of
          dot(X.T, X)

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.


        All last five solvers support both dense and sparse data. However, only
        'sag' and 'saga' supports sparse input when`fit_intercept` is True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        For the 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' and saga solver, the default value is
        1000.

    tol : float
        Precision of the solution.

    verbose : int
        Verbosity level. Setting verbose > 0 will display additional
        information depending on the solver used.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag'.

    return_n_iter : boolean, default False
        If True, the method also returns `n_iter`, the actual number of
        iteration performed by the solver.

        .. versionadded:: 0.17

    return_intercept : boolean, default False
        If True and if X is sparse, the method also returns the intercept,
        and the solver is automatically changed to 'sag'. This is only a
        temporary fix for fitting the intercept with sparse data. For dense
        data, use sklearn.linear_model._preprocess_data before your regression.

        .. versionadded:: 0.17

    Returns
    -------
    coef : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).

    n_iter : int, optional
        The actual number of iteration performed by the solver.
        Only returned if `return_n_iter` is True.

    intercept : float or array, shape = [n_targets]
        The intercept of the model. Only returned if `return_intercept`
        is True and if X is a scipy sparse array.

    Notes
    -----
    This function won't compute the intercept.
    
```
133 - /tmp/repos/scikit-learn/sklearn/dummy.py
```python
def predict(self, X, return_std=False):
        """
        Perform classification on test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        return_std : boolean, optional
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_outputs]
            Predicted target values for X.

        y_std : array, shape = [n_samples] or [n_samples, n_outputs]
            Standard deviation of predictive distribution of query points.
        """
        check_is_fitted(self, "constant_")
        n_samples = _num_samples(X)

        y = np.full((n_samples, self.n_outputs_), self.constant_,
                    dtype=np.array(self.constant_).dtype)
        y_std = np.zeros((n_samples, self.n_outputs_))

        if self.n_outputs_ == 1 and not self.output_2d_:
            y = np.ravel(y)
            y_std = np.ravel(y_std)

        return (y, y_std) if return_std else y

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : {array-like, None}
            Test samples with shape = (n_samples, n_features) or None.
            For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
            Passing None as test samples gives the same result
            as passing real test samples, since DummyRegressor
            operates independently of the sampled observations.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)

```
134 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
def _get_data_description_by_id(data_id, data_home):
    # OpenML API function: https://www.openml.org/api_docs#!/data/get_data_id
    url = _DATA_INFO.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, True,
                                                  data_home)
    return json_data['data_set_description']


def _get_data_features(data_id, data_home):
    # OpenML function:
    # https://www.openml.org/api_docs#!/data/get_data_features_id
    url = _DATA_FEATURES.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, True,
                                                  data_home)
    return json_data['data_features']['feature']


def _get_data_qualities(data_id, data_home):
    # OpenML API function:
    # https://www.openml.org/api_docs#!/data/get_data_qualities_id
    url = _DATA_QUALITIES.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, True,
                                                  data_home)
    try:
        return json_data['data_qualities']['quality']
    except KeyError:
        # the qualities might not be available, but we still try to process
        # the data
        return None


def _get_data_shape(data_qualities):
    # Using the data_info dictionary from _get_data_info_by_name to extract
    # the number of samples / features
    if data_qualities is None:
        return None
    qualities = {d['name']: d['value'] for d in data_qualities}
    try:
        return (int(float(qualities['NumberOfInstances'])),
                int(float(qualities['NumberOfFeatures'])))
    except AttributeError:
        return None


def _download_data_arff(file_id, sparse, data_home, encode_nominal=True):
    # Accesses an ARFF file on the OpenML server. Documentation:
    # https://www.openml.org/api_data_docs#!/data/get_download_id
    # encode_nominal argument is to ensure unit testing, do not alter in
    # production!
    url = _DATA_FILE.format(file_id)

    @_retry_with_clean_cache(url, data_home)
    def _arff_load():
        with closing(_open_openml_url(url, data_home)) as response:
            if sparse is True:
                return_type = _arff.COO
            else:
                return_type = _arff.DENSE_GEN

            arff_file = _arff.loads(response.read().decode('utf-8'),
                                    encode_nominal=encode_nominal,
                                    return_type=return_type)
        return arff_file

    return _arff_load()


def _verify_target_data_type(features_dict, target_columns):
    # verifies the data type of the y array in case there are multiple targets
    # (throws an error if these targets do not comply with sklearn support)
    if not isinstance(target_columns, list):
        raise ValueError('target_column should be list, '
                         'got: %s' % type(target_columns))
    found_types = set()
    for target_column in target_columns:
        if target_column not in features_dict:
            raise KeyError('Could not find target_column={}')
        if features_dict[target_column]['data_type'] == "numeric":
            found_types.add(np.float64)
        else:
            found_types.add(object)

        # note: we compare to a string, not boolean
        if features_dict[target_column]['is_ignore'] == 'true':
            warn('target_column={} has flag is_ignore.'.format(
                target_column))
        if features_dict[target_column]['is_row_identifier'] == 'true':
            warn('target_column={} has flag is_row_identifier.'.format(
                target_column))
    if len(found_types) > 1:
        raise ValueError('Can only handle homogeneous multi-target datasets, '
                         'i.e., all targets are either numeric or '
                         'categorical.')



```
135 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
"""
The :mod:`sklearn.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>,
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause


from __future__ import print_function
from __future__ import division

import warnings
from itertools import chain, combinations
from collections import Iterable
from math import ceil, floor
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np

from ..utils import indexable, check_random_state, safe_indexing
from ..utils.validation import _num_samples, column_or_1d
from ..utils.validation import check_array
from ..utils.multiclass import type_of_target
from ..externals.six import with_metaclass
from ..externals.six.moves import zip
from ..utils.fixes import signature, comb
from ..base import _pprint

__all__ = ['BaseCrossValidator',
           'KFold',
           'GroupKFold',
           'LeaveOneGroupOut',
           'LeaveOneOut',
           'LeavePGroupsOut',
           'LeavePOut',
           'RepeatedStratifiedKFold',
           'RepeatedKFold',
           'ShuffleSplit',
           'GroupShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'PredefinedSplit',
           'train_test_split',
           'check_cv']


class BaseCrossValidator(with_metaclass(ABCMeta)):
    """Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def __init__(self):
        # We need this for the build_repr to work properly in py2.7
        # see #6304
        pass

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, of length n_samples
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""

    def __repr__(self):
        return _build_repr(self)



```
136 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self

        """
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[i])

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.

        """
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)

        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

    
```
137 - /tmp/repos/scikit-learn/sklearn/utils/metaestimators.py
```python
def if_delegate_has_method(delegate):
    """Create a decorator for methods that are delegated to a sub-estimator

    This enables ducktyping by hasattr returning True according to the
    sub-estimator.

    Parameters
    ----------
    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object will be used.

    """
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)

    return lambda fn: _IffHasAttrDescriptor(fn, delegate,
                                            attribute_name=fn.__name__)


def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    Labels y will always be sliced only along the last axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be sliced. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be sliced.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.

    Returns
    -------
    X_sliced : array-like, sparse matrix or list
        Sliced data.

    y_sliced : array-like, sparse matrix or list
        Sliced targets.

    """
    if getattr(estimator, "_pairwise", False):
        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = safe_indexing(X, indices)

    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset

```
138 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def pairwise_kernels(X, Y=None, metric="linear", filter_params=False,
                     n_jobs=None, **kwds):
    """Compute the kernel between arrays X and optional array Y.

    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for metric are::
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
         'laplacian', 'sigmoid', 'cosine']

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise kernels between samples, or a feature array.

    Y : array [n_samples_b, n_features]
        A second feature array only if X has shape [n_samples_a, n_features].

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    filter_params : boolean
        Whether to filter invalid parameters or not.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.

    """
    # import GPKernel locally to prevent circular imports
    from ..gaussian_process.kernels import Kernel as GPKernel

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        func = metric.__call__
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = {k: kwds[k] for k in kwds
                    if k in KERNEL_PARAMS[metric]}
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)

```
139 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]

    def score(self, X, y=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.

        y : Ignored

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return -_labels_inertia(X, x_squared_norms, self.cluster_centers_)[1]


def _mini_batch_step(X, x_squared_norms, centers, counts,
                     old_center_buffer, compute_squared_diff,
                     distances, random_reassign=False,
                     random_state=None, reassignment_ratio=.01,
                     verbose=False):
    
```
140 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        if self.min_impurity_split is not None:
            warnings.warn("The min_impurity_split parameter is deprecated and"
                          " will be removed in version 0.21. "
                          "Use the min_impurity_decrease parameter instead.",
                          DeprecationWarning)
            min_impurity_split = self.min_impurity_split
        else:
            min_impurity_split = 1e-7

        if min_impurity_split < 0.:
            raise ValueError("min_impurity_split must be greater than "
                             "or equal to 0")

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        allowed_presort = ('auto', True, False)
        if self.presort not in allowed_presort:
            raise ValueError("'presort' should be in {}. Got {!r} instead."
                             .format(allowed_presort, self.presort))

        if self.presort is True and issparse(X):
            raise ValueError("Presorting is not supported for sparse "
                             "matrices.")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if self.presort == 'auto':
            presort = not issparse(X)

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        
```
141 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
"""

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            presort=presort, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like, shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

```
142 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
"""Stratified ShuffleSplit cross validation iterator

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :class:`sklearn.model_selection.StratifiedShuffleSplit` instead.

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.

    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    y : array, [n_samples]
        Labels of samples.

    n_iter : int (default 10)
        Number of re-shuffling & splitting iterations.

    test_size : float (default 0.1), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> from sklearn.cross_validation import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
    >>> len(sss)
    3
    >>> print(sss)       # doctest: +ELLIPSIS
    StratifiedShuffleSplit(labels=[0 0 1 1], n_iter=3, ...)
    >>> for train_index, test_index in sss:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 2] TEST: [3 0]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [0 2] TEST: [3 1]
    """

    def __init__(self, y, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):

        super(StratifiedShuffleSplit, self).__init__(
            len(y), n_iter, test_size, train_size, random_state)

        self.y = np.array(y)
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        n_cls = self.classes.shape[0]

        if np.min(np.bincount(self.y_indices)) < 2:
            raise ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of labels for any class cannot"
                             " be less than 2.")

        if self.n_train < n_cls:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (self.n_train, n_cls))
        if self.n_test < n_cls:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (self.n_test, n_cls))

    
```
143 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
"""

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        super(GradientBoostingRegressor, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            presort=presort, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        return self._decision_function(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for y in self._staged_decision_function(X):
            yield y.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super(GradientBoostingRegressor, self).apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves

```
144 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, optional (default=False)
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            Projected data
        """

        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                column_slice = slice(X.indptr[feature_idx],
                                     X.indptr[feature_idx + 1])
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice], self.quantiles_[:, feature_idx],
                    inverse)
        else:
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx],
                    inverse)

        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        X = self._check_inputs(X)
        self._check_is_fitted(X)

        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        X = self._check_inputs(X, accept_sparse_negative=True)
        self._check_is_fitted(X)

        return self._transform(X, inverse=True)



```
145 - /tmp/repos/scikit-learn/sklearn/decomposition/truncated_svd.py
```python
def __init__(self, n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y=None):
        """Fit LSI model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = check_array(X, accept_sparse=['csr', 'csc'])
        random_state = check_random_state(self.random_state)

        if self.algorithm == "arpack":
            U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            Sigma = Sigma[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])

        elif self.algorithm == "randomized":
            k = self.n_components
            n_features = X.shape[1]
            if k >= n_features:
                raise ValueError("n_components must be < n_features;"
                                 " got %d >= %d" % (k, n_features))
            U, Sigma, VT = randomized_svd(X, self.n_components,
                                          n_iter=self.n_iter,
                                          random_state=random_state)
        else:
            raise ValueError("unknown algorithm %r" % self.algorithm)

        self.components_ = VT

        # Calculate explained variance & explained variance ratio
        X_transformed = U * Sigma
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        X = check_array(X)
        return np.dot(X, self.components_)

```
146 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
"""

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored

        """
        random_state = check_random_state(self.random_state)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                precompute_distances=self.precompute_distances,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True)
        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        y : Ignored

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        y : Ignored

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        return self.fit(X)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    
```
147 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def inverse_transform(self, X):
        """Convert back the data to the original representation.

        In case unknown categories are encountered (all zero's in the
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
        n_transformed_features = sum([len(cats) for cats in self.categories_])

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if self.encoding == 'ordinal' and X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        elif (self.encoding.startswith('onehot')
                and X.shape[1] != n_transformed_features):
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        if self.encoding == 'ordinal':
            for i in range(n_features):
                labels = X[:, i].astype('int64')
                X_tr[:, i] = self.categories_[i][labels]

        else:  # encoding == 'onehot' / 'onehot-dense'
            j = 0
            found_unknown = {}

            for i in range(n_features):
                n_categories = len(self.categories_[i])
                sub = X[:, j:j + n_categories]

                # for sparse X argmax returns 2D matrix, ensure 1D array
                labels = np.asarray(_argmax(sub, axis=1)).flatten()
                X_tr[:, i] = self.categories_[i][labels]

                if self.handle_unknown == 'ignore':
                    # ignored unknown categories: we have a row of all zero's
                    unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                    if unknown.any():
                        found_unknown[i] = unknown

                j += n_categories

            # if ignored are found: potentially need to upcast result to
            # insert None values
            if found_unknown:
                if X_tr.dtype != object:
                    X_tr = X_tr.astype(object)

                for idx, mask in found_unknown.items():
                    X_tr[mask, idx] = None

        return X_tr

```
148 - /tmp/repos/scikit-learn/sklearn/externals/_arff.py
```python
# -*- coding: utf-8 -*-
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2011 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
The liac-arff module implements functions to read and write ARFF files in
Python. It was created in the Connectionist Artificial Intelligence Laboratory
(LIAC), which takes place at the Federal University of Rio Grande do Sul
(UFRGS), in Brazil.

ARFF (Attribute-Relation File Format) is an file format specially created for
describe datasets which are commonly used for machine learning experiments and
softwares. This file format was created to be used in Weka, the best
representative software for machine learning automated experiments.

An ARFF file can be divided into two sections: header and data. The Header
describes the metadata of the dataset, including a general description of the
dataset, its name and its attributes. The source below is an example of a
header section in a XOR dataset::

    %
    % XOR Dataset
    %
    % Created by Renato Pereira
    %            rppereira@inf.ufrgs.br
    %            http://inf.ufrgs.br/~rppereira
    %
    %
    @RELATION XOR

    @ATTRIBUTE input1 REAL
    @ATTRIBUTE input2 REAL
    @ATTRIBUTE y REAL

The Data section of an ARFF file describes the observations of the dataset, in
the case of XOR dataset::

    @DATA
    0.0,0.0,0.0
    0.0,1.0,1.0
    1.0,0.0,1.0
    1.0,1.0,0.0
    %
    %
    %

Notice that several lines are starting with an ``%`` symbol, denoting a
comment, thus, lines with ``%`` at the beginning will be ignored, except by the
description part at the beginning of the file. The declarations ``@RELATION``,
``@ATTRIBUTE``, and ``@DATA`` are all case insensitive and obligatory.

For more information and details about the ARFF file description, consult
http://www.cs.waikato.ac.nz/~ml/weka/arff.html


ARFF Files in Python
~~~~~~~~~~~~~~~~~~~~

This module uses built-ins python objects to represent a deserialized ARFF
file. A dictionary is used as the container of the data and metadata of ARFF,
and have the following keys:

- **description**: (OPTIONAL) a string with the description of the dataset.
- **relation**: (OBLIGATORY) a string with the name of the dataset.
- **attributes**: (OBLIGATORY) a list of attributes with the following
  template::

    (attribute_name, attribute_type)

  the attribute_name is a string, and attribute_type must be an string
  or a list of strings.
- **data**: (OBLIGATORY) a list of data instances. Each data instance must be
  a list with values, depending on the attributes.

The above keys must follow the case which were described, i.e., the keys are
case sensitive. The attribute type ``attribute_type`` must be one of these
strings (they are not case sensitive): ``NUMERIC``, ``INTEGER``, ``REAL`` or
``STRING``. For nominal attributes, the ``atribute_type`` must be a list of
strings.

In this format, the XOR dataset presented above can be represented as a python
object as::

    xor_dataset = {
        'description': 'XOR Dataset',
        'relation': 'XOR',
        'attributes': [
            ('input1', 'REAL'),
            ('input2', 'REAL'),
            ('y', 'REAL'),
        ],
        'data': [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ]
    }


Features
~~~~~~~~

This module provides several features, including:

- Read and write ARFF files using python built-in structures, such dictionaries
  and lists;
- Supports `scipy.sparse.coo <http://docs.scipy
  .org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix>`_
  and lists of dictionaries as used by SVMLight
- Supports the following attribute types: NUMERIC, REAL, INTEGER, STRING, and
  NOMINAL;
- Has an interface similar to other built-in modules such as ``json``, or
  ``zipfile``;
- Supports read and write the descriptions of files;
- Supports missing values and names with spaces;
- Supports unicode values and names;
- Fully compatible with Python 2.7+, Python 3.3+, pypy and pypy3;
- Under `MIT License <http://opensource.org/licenses/MIT>`_


```
149 - /tmp/repos/scikit-learn/sklearn/neighbors/regression.py
```python
def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of float, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.radius_neighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        empty_obs = np.full_like(_y[0], np.nan)

        if weights is None:
            y_pred = np.array([np.mean(_y[ind, :], axis=0)
                               if len(ind) else empty_obs
                               for (i, ind) in enumerate(neigh_ind)])

        else:
            y_pred = np.array([np.average(_y[ind, :], axis=0,
                               weights=weights[i])
                               if len(ind) else empty_obs
                               for (i, ind) in enumerate(neigh_ind)])

        if np.max(np.isnan(y_pred)):
            empty_warning_msg = ("One or more samples have no neighbors "
                                 "within specified radius; predicting NaN.")
            warnings.warn(empty_warning_msg)


        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

```
150 - /tmp/repos/scikit-learn/sklearn/impute/_base.py
```python
def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self, 'statistics_')

        X = self._validate_input(X)

        statistics = self.statistics_

        if X.shape[1] != statistics.shape[0]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[0]))

        if self.add_indicator:
            X_trans_indicator = self.indicator_.transform(X)

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
        else:
            # same as np.isnan but also works for object dtypes
            invalid_mask = _get_mask(statistics, np.nan)
            valid_mask = np.logical_not(invalid_mask)
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = np.flatnonzero(valid_mask)

            if invalid_mask.any():
                missing = np.arange(X.shape[1])[invalid_mask]
                if self.verbose:
                    warnings.warn("Deleting features without "
                                  "observed values: %s" % missing)
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sparse.issparse(X):
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                mask = _get_mask(X.data, self.missing_values)
                indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
                                    np.diff(X.indptr))[mask]

                X.data[mask] = valid_statistics[indexes].astype(X.dtype,
                                                                copy=False)
        else:
            mask = _get_mask(X, self.missing_values)
            n_missing = np.sum(mask, axis=0)
            values = np.repeat(valid_statistics, n_missing)
            coordinates = np.where(mask.transpose())[::-1]

            X[coordinates] = values

        if self.add_indicator:
            hstack = sparse.hstack if sparse.issparse(X) else np.hstack
            X = hstack((X, X_trans_indicator))

        return X

    def _more_tags(self):
        return {'allow_nan': True}


class MissingIndicator(BaseEstimator, TransformerMixin):
    
```
151 - /tmp/repos/scikit-learn/sklearn/linear_model/ridge.py
```python
if solver == 'sparse_cg':
        coef = _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose)

    elif solver == 'lsqr':
        coef, n_iter = _solve_lsqr(X, y, alpha, max_iter, tol)

    elif solver == 'cholesky':
        if n_features > n_samples:
            K = safe_sparse_dot(X, X.T, dense_output=True)
            try:
                dual_coef = _solve_cholesky_kernel(K, y, alpha)

                coef = safe_sparse_dot(X.T, dual_coef, dense_output=True).T
            except linalg.LinAlgError:
                # use SVD solver if matrix is singular
                solver = 'svd'

        else:
            try:
                coef = _solve_cholesky(X, y, alpha)
            except linalg.LinAlgError:
                # use SVD solver if matrix is singular
                solver = 'svd'

    elif solver in ['sag', 'saga']:
        # precompute max_squared_sum for all targets
        max_squared_sum = row_norms(X, squared=True).max()

        coef = np.empty((y.shape[1], n_features))
        n_iter = np.empty(y.shape[1], dtype=np.int32)
        intercept = np.zeros((y.shape[1], ))
        for i, (alpha_i, target) in enumerate(zip(alpha, y.T)):
            init = {'coef': np.zeros((n_features + int(return_intercept), 1))}
            coef_, n_iter_, _ = sag_solver(
                X, target.ravel(), sample_weight, 'squared', alpha_i, 0,
                max_iter, tol, verbose, random_state, False, max_squared_sum,
                init,
                is_saga=solver == 'saga')
            if return_intercept:
                coef[i] = coef_[:-1]
                intercept[i] = coef_[-1]
            else:
                coef[i] = coef_
            n_iter[i] = n_iter_

        if intercept.shape[0] == 1:
            intercept = intercept[0]
        coef = np.asarray(coef)

    if solver == 'svd':
        if sparse.issparse(X):
            raise TypeError('SVD solver does not support sparse'
                            ' inputs currently')
        coef = _solve_svd(X, y, alpha)

    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        coef = coef.ravel()

    if return_n_iter and return_intercept:
        return coef, n_iter, intercept
    elif return_intercept:
        return coef, intercept
    elif return_n_iter:
        return coef, n_iter
    else:
        return coef



```
152 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/binning.py
```python
class _BinMapper(BaseEstimator, TransformerMixin):
    """Transformer that maps a dataset into integer-valued bins.

    The bins are created in a feature-wise fashion, using quantiles so that
    each bins contains approximately the same number of samples.

    For large datasets, quantiles are computed on a subset of the data to
    speed-up the binning, but the quantiles should remain stable.

    If the number of unique values for a given feature is less than
    ``max_bins``, then the unique values of this feature are used instead of
    the quantiles.

    Parameters
    ----------
    max_bins : int, optional (default=256)
        The maximum number of bins to use. If for a given feature the number of
        unique values is less than ``max_bins``, then those unique values
        will be used to compute the bin thresholds, instead of the quantiles.
    subsample : int or None, optional (default=2e5)
        If ``n_samples > subsample``, then ``sub_samples`` samples will be
        randomly choosen to compute the quantiles. If ``None``, the whole data
        is used.
    random_state: int or numpy.random.RandomState or None, \
        optional (default=None)
        Pseudo-random number generator to control the random sub-sampling.
        See :term:`random_state`.
    """
    def __init__(self, max_bins=256, subsample=int(2e5), random_state=None):
        self.max_bins = max_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit data X by computing the binning thresholds.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to bin.
        y: None
            Ignored.

        Returns
        -------
        self : object
        """
        X = check_array(X, dtype=[X_DTYPE])
        self.bin_thresholds_ = _find_binning_thresholds(
            X, self.max_bins, subsample=self.subsample,
            random_state=self.random_state)

        self.actual_n_bins_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32)

        return self

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
153 - /tmp/repos/scikit-learn/sklearn/utils/sparsefuncs.py
```python
def count_nonzero(X, axis=None, sample_weight=None):
    """A variant of X.getnnz() with extension to weighting on axis 0

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : CSR sparse matrix, shape = (n_samples, n_labels)
        Input data.

    axis : None, 0 or 1
        The axis on which the data is aggregated.

    sample_weight : array, shape = (n_samples,), optional
        Weight for each row of X.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif X.format != 'csr':
        raise TypeError('Expected CSR sparse format, got {0}'.format(X.format))

    # We rely here on the fact that np.diff(Y.indptr) for a CSR
    # will return the number of nonzero entries in each row.
    # A bincount over Y.indices will return the number of nonzeros
    # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
    if axis is None:
        if sample_weight is None:
            return X.nnz
        else:
            return np.dot(np.diff(X.indptr), sample_weight)
    elif axis == 1:
        out = np.diff(X.indptr)
        if sample_weight is None:
            return out
        return out * sample_weight
    elif axis == 0:
        if sample_weight is None:
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            weights = np.repeat(sample_weight, np.diff(X.indptr))
            return np.bincount(X.indices, minlength=X.shape[1],
                            weights=weights)
    else:
        raise ValueError('Unsupported axis: {0}'.format(axis))


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.

    This function is used to support sparse matrices; it modifies data in-place
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (_get_elem_at_rank(middle - 1, data, n_negative, n_zeros) +
            _get_elem_at_rank(middle, data, n_negative, n_zeros)) / 2.


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]


def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.
    It is equivalent to doing np.median(X, axis=0).

    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    Returns
    -------
    median : ndarray, shape (n_features,)
        Median.

    """
    if not isinstance(X, sp.csc_matrix):
        raise TypeError("Expected matrix of CSC format, got %s" % X.format)

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):

        # Prevent modifying X in place
        data = np.copy(X.data[start: end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median

```
154 - /tmp/repos/scikit-learn/sklearn/kernel_approximation.py
```python
def _transform_dense(self, X):
        non_zero = (X != 0.0)
        X_nz = X[non_zero]

        X_step = np.zeros_like(X)
        X_step[non_zero] = np.sqrt(X_nz * self.sample_interval_)

        X_new = [X_step]

        log_step_nz = self.sample_interval_ * np.log(X_nz)
        step_nz = 2 * X_nz * self.sample_interval_

        for j in range(1, self.sample_steps):
            factor_nz = np.sqrt(step_nz /
                                np.cosh(np.pi * j * self.sample_interval_))

            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.cos(j * log_step_nz)
            X_new.append(X_step)

            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.sin(j * log_step_nz)
            X_new.append(X_step)

        return np.hstack(X_new)

    def _transform_sparse(self, X):
        indices = X.indices.copy()
        indptr = X.indptr.copy()

        data_step = np.sqrt(X.data * self.sample_interval_)
        X_step = sp.csr_matrix((data_step, indices, indptr),
                               shape=X.shape, dtype=X.dtype, copy=False)
        X_new = [X_step]

        log_step_nz = self.sample_interval_ * np.log(X.data)
        step_nz = 2 * X.data * self.sample_interval_

        for j in range(1, self.sample_steps):
            factor_nz = np.sqrt(step_nz /
                                np.cosh(np.pi * j * self.sample_interval_))

            data_step = factor_nz * np.cos(j * log_step_nz)
            X_step = sp.csr_matrix((data_step, indices, indptr),
                                   shape=X.shape, dtype=X.dtype, copy=False)
            X_new.append(X_step)

            data_step = factor_nz * np.sin(j * log_step_nz)
            X_step = sp.csr_matrix((data_step, indices, indptr),
                                   shape=X.shape, dtype=X.dtype, copy=False)
            X_new.append(X_step)

        return sp.hstack(X_new)


class Nystroem(BaseEstimator, TransformerMixin):
    
```
155 - /tmp/repos/scikit-learn/sklearn/decomposition/sparse_pca.py
```python
def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)

        _check_normalize_components(
            self.normalize_components, self.__class__.__name__
        )

        self.mean_ = X.mean(axis=0)
        X = X - self.mean_

        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
        Vt, _, self.n_iter_ = dict_learning_online(
            X.T, n_components, alpha=self.alpha,
            n_iter=self.n_iter, return_code=True,
            dict_init=None, verbose=self.verbose,
            callback=self.callback,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            n_jobs=self.n_jobs, method=self.method,
            random_state=random_state,
            return_n_iter=True)
        self.components_ = Vt.T

        components_norm = np.linalg.norm(
            self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm

        return self

```
**156 - /tmp/repos/scikit-learn/sklearn/datasets/lfw.py**:
```python
def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
                     min_faces_per_person=0, color=False,
                     slice_=(slice(70, 195), slice(78, 172)),
                     download_if_missing=True):
    """Loader for the Labeled Faces in the Wild (LFW) people dataset

    This dataset is a collection of JPEG pictures of famous people
    collected on the internet, all details are available on the
    official website:

        http://vis-www.cs.umass.edu/lfw/

    Each picture is centered on a single face. Each pixel of each channel
    (color in RGB) is encoded by a float in range 0.0 - 1.0.

    The task is called Face Recognition (or Identification): given the
    picture of a face, find the name of the person given a training set
    (gallery).

    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize : float, optional, default 0.5
        Ratio used to resize the each face picture.

    min_faces_per_person : int, optional, default None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (13233, 2914)
        Each row corresponds to a ravelled face image of original size 62 x 47
        pixels. Changing the ``slice_`` or resize parameters will change the
        shape of the output.

    dataset.images : numpy array of shape (13233, 62, 47)
        Each row is a face image corresponding to one of the 5749 people in
        the dataset. Changing the ``slice_`` or resize parameters will change
        the shape of the output.

    dataset.target : numpy array of shape (13233,)
        Labels associated to each face image. Those labels range from 0-5748
        and correspond to the person IDs.

    dataset.DESCR : string
        Description of the Labeled Faces in the Wild (LFW) dataset.
    """
    lfw_home, data_folder_path = check_fetch_lfw(
        data_home=data_home, funneled=funneled,
        download_if_missing=download_if_missing)
    logger.debug('Loading LFW people faces from %s', lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    m = Memory(cachedir=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    # load and memoize the pairs as np arrays
    faces, target, target_names = load_func(
        data_folder_path, resize=resize,
        min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)

    # pack the results as a Bunch instance
    return Bunch(data=faces.reshape(len(faces), -1), images=faces,
                 target=target, target_names=target_names,
                 DESCR="LFW faces dataset")


#
# Task #2:  Face Verification on pairs of face pictures
#



```
157 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class GroupShuffleSplit(ShuffleSplit):
    '''Shuffle-Group(s)-Out cross-validation iterator

    Provides randomized train/test indices to split data according to a
    third-party provided group. This group information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    The difference between LeavePGroupsOut and GroupShuffleSplit is that
    the former generates splits using all subsets of size ``p`` unique groups,
    whereas GroupShuffleSplit generates a user-determined number of random
    test splits, each with a user-determined fraction of unique groups.

    For example, a less computationally intensive alternative to
    ``LeavePGroupsOut(p=10)`` would be
    ``GroupShuffleSplit(test_size=10, n_splits=100)``.

    Note: The parameters ``test_size`` and ``train_size`` refer to groups, and
    not to samples, as in ShuffleSplit.


    Parameters
    ----------
    n_splits : int (default 5)
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.2.
        The default will change in version 0.21. It will remain 0.2 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float, int, or None, default is None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    '''

    def __init__(self, n_splits=5, test_size="default", train_size=None,
                 random_state=None):
        if test_size == "default":
            if train_size is not None:
                warnings.warn("From version 0.21, test_size will always "
                              "complement train_size unless both "
                              "are specified.",
                              FutureWarning)
            test_size = 0.2

        super(GroupShuffleSplit, self).__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)

    def _iter_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        classes, group_indices = np.unique(groups, return_inverse=True)
        for group_train, group_test in super(
                GroupShuffleSplit, self)._iter_indices(X=classes):
            # these are the indices of classes in the partition
            # invert them into data indices

            train = np.flatnonzero(np.in1d(group_indices, group_train))
            test = np.flatnonzero(np.in1d(group_indices, group_test))

            yield train, test



```
158 - /tmp/repos/scikit-learn/sklearn/impute.py
```python
def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))

        if self.predictor is None:
            from .linear_model import BayesianRidge
            self._predictor = BayesianRidge()
        else:
            self._predictor = clone(self.predictor)

        self._min_value = np.nan if self.min_value is None else self.min_value
        self._max_value = np.nan if self.max_value is None else self.max_value

        self.initial_imputer_ = None
        X, X_filled, mask_missing_values = self._initial_imputation(X)

        # edge case: in case the user specifies 0 for n_imputations,
        # then there is no need to do burn in and the result should be
        # just the initial imputation (before clipping)
        if self.n_imputations < 1:
            return X_filled

        X_filled = np.clip(X_filled, self._min_value, self._max_value)

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)

        abs_corr_mat = self._get_abs_corr_mat(X_filled)

        # impute data
        n_rounds = self.n_burn_in + self.n_imputations
        n_samples, n_features = X_filled.shape
        Xt = np.zeros((n_samples, n_features), dtype=X.dtype)
        self.imputation_sequence_ = []
        if self.verbose > 0:
            print("[MICE] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        for i_rnd in range(n_rounds):
            if self.imputation_order == 'random':
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(n_features,
                                                                feat_idx,
                                                                abs_corr_mat)
                X_filled, predictor = self._impute_one_feature(
                    X_filled, mask_missing_values, feat_idx, neighbor_feat_idx,
                    predictor=None, fit_mode=True)
                predictor_triplet = MICETriplet(feat_idx,
                                                neighbor_feat_idx,
                                                predictor)
                self.imputation_sequence_.append(predictor_triplet)

            if i_rnd >= self.n_burn_in:
                Xt += X_filled
            if self.verbose > 0:
                print('[MICE] Ending imputation round '
                      '%d/%d, elapsed time %0.2f'
                      % (i_rnd + 1, n_rounds, time() - start_t))

        Xt /= self.n_imputations
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt

    
```
159 - /tmp/repos/scikit-learn/sklearn/utils/sparsefuncs.py
```python
def count_nonzero(X, axis=None, sample_weight=None):
    """A variant of X.getnnz() with extension to weighting on axis 0

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : CSR sparse matrix, shape = (n_samples, n_labels)
        Input data.

    axis : None, 0 or 1
        The axis on which the data is aggregated.

    sample_weight : array, shape = (n_samples,), optional
        Weight for each row of X.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif X.format != 'csr':
        raise TypeError('Expected CSR sparse format, got {0}'.format(X.format))

    # We rely here on the fact that np.diff(Y.indptr) for a CSR
    # will return the number of nonzero entries in each row.
    # A bincount over Y.indices will return the number of nonzeros
    # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
    if axis is None:
        if sample_weight is None:
            return X.nnz
        else:
            return np.dot(np.diff(X.indptr), sample_weight)
    elif axis == 1:
        out = np.diff(X.indptr)
        if sample_weight is None:
            # astype here is for consistency with axis=0 dtype
            return out.astype('intp')
        return out * sample_weight
    elif axis == 0:
        if sample_weight is None:
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            weights = np.repeat(sample_weight, np.diff(X.indptr))
            return np.bincount(X.indices, minlength=X.shape[1],
                            weights=weights)
    else:
        raise ValueError('Unsupported axis: {0}'.format(axis))


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.

    This function is used to support sparse matrices; it modifies data in-place
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (_get_elem_at_rank(middle - 1, data, n_negative, n_zeros) +
            _get_elem_at_rank(middle, data, n_negative, n_zeros)) / 2.


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]


def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.
    It is equivalent to doing np.median(X, axis=0).

    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    Returns
    -------
    median : ndarray, shape (n_features,)
        Median.

    """
    if not isinstance(X, sp.csc_matrix):
        raise TypeError("Expected matrix of CSC format, got %s" % X.format)

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):

        # Prevent modifying X in place
        data = np.copy(X.data[start: end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median

```
160 - /tmp/repos/scikit-learn/sklearn/cross_decomposition/pls_.py
```python
for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.algorithm == "nipals":
                x_weights, y_weights, n_iter_ = \
                    _nipals_twoblocks_inner_loop(
                        X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                        tol=self.tol, norm_y_weights=self.norm_y_weights)
                self.n_iter_.append(n_iter_)
            elif self.algorithm == "svd":
                x_weights, y_weights = _svd_cross_product(X=Xk, Y=Yk)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # compute scores
            x_scores = np.dot(Xk, x_weights)
            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
                warnings.warn('X scores are null at iteration %s' % k)
                break
            # 2) Deflation (in place)
            # ----------------------
            # Possible memory footprint reduction may done here: in order to
            # avoid the allocation of a data chunk for the rank-one
            # approximations matrix which is then subtracted to Xk, we suggest
            # to perform a column-wise deflation.
            #
            # - regress Xk's on x_score
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - subtract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)
            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, y_scores)
                              / np.dot(y_scores.T, y_scores))
                Yk -= np.dot(y_scores, y_loadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, x_scores)
                              / np.dot(x_scores.T, x_scores))
                Yk -= np.dot(x_scores, y_loadings.T)
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_),
                  check_finite=False))
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_),
                      check_finite=False))
        else:
            self.y_rotations_ = np.ones(1)

        
```
161 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
"""
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    if 2 ** n_informative < n_classes * n_clusters_per_class:
        raise ValueError("n_classes * n_clusters_per_class must"
                         " be smaller or equal 2 ** n_informative")
    if weights and len(weights) not in [n_classes, n_classes - 1]:
        raise ValueError("Weights specified but incompatible with number "
                         "of classes.")

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    if weights and len(weights) == (n_classes - 1):
        weights = weights + [1.0 - sum(weights)]

    if weights is None:
        weights = [1.0 / n_classes] * n_classes
        weights[-1] = 1.0 - sum(weights[:-1])

    # Distribute samples among clusters by weight
    n_samples_per_cluster = []
    for k in range(n_clusters):
        n_samples_per_cluster.append(int(n_samples * weights[k % n_classes]
                                     / n_clusters_per_class))
    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative,
                                    generator).astype(float)
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.rand(n_clusters, 1)
        centroids *= generator.rand(1, n_informative)

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.randn(n_samples, n_informative)

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster

        A = 2 * generator.rand(n_informative, n_informative) - 1
        X_k[...] = np.dot(X_k, A)  # introduce random covariance

        X_k += centroid  # shift the cluster to a vertex

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = generator.randn(n_samples, n_useless)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.rand(n_features)
    X *= scale

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y



```
162 - /tmp/repos/scikit-learn/sklearn/linear_model/base.py
```python



@deprecated("sparse_center_data was deprecated in version 0.18 and will be "
            "removed in 0.20. Use utilities in preprocessing.data instead")
def sparse_center_data(X, y, fit_intercept, normalize=False):
    """
    Compute information needed to center data to have mean zero along
    axis 0. Be aware that X will not be centered since it would break
    the sparsity, but will be normalized if asked so.
    """
    if fit_intercept:
        # we might require not to change the csr matrix sometimes
        # store a copy if normalize is True.
        # Change dtype to float64 since mean_variance_axis accepts
        # it that way.
        if sp.isspmatrix(X) and X.getformat() == 'csr':
            X = sp.csr_matrix(X, copy=normalize, dtype=np.float64)
        else:
            X = sp.csc_matrix(X, copy=normalize, dtype=np.float64)

        X_offset, X_var = mean_variance_axis(X, axis=0)
        if normalize:
            # transform variance to std in-place
            X_var *= X.shape[0]
            X_std = np.sqrt(X_var, X_var)
            del X_var
            X_std[X_std == 0] = 1
            inplace_column_scale(X, 1. / X_std)
        else:
            X_std = np.ones(X.shape[1])
        y_offset = y.mean(axis=0)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_offset = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_std


@deprecated("center_data was deprecated in version 0.18 and will be removed "
            "in 0.20. Use utilities in preprocessing.data instead")
def center_data(X, y, fit_intercept, normalize=False, copy=True,
                sample_weight=None):
    """
    Centers data to have mean zero along axis 0. This is here because
    nearly all linear models will want their data to be centered.
    If sample_weight is not None, then the weighted mean of X and y
    is zero, and not the mean itself
    """
    X = as_float_array(X, copy)
    if fit_intercept:
        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sp.issparse(X):
            X_offset = np.zeros(X.shape[1])
            X_std = np.ones(X.shape[1])
        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            # XXX: currently scaled to variance=n_samples
            if normalize:
                X_std = np.sqrt(np.sum(X ** 2, axis=0))
                X_std[X_std == 0] = 1
                X /= X_std
            else:
                X_std = np.ones(X.shape[1])
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_offset = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
    return X, y, X_offset, y_offset, X_std



```
**163 - /tmp/repos/scikit-learn/sklearn/datasets/lfw.py**:
```python
def _fetch_lfw_people(data_folder_path, slice_=None, color=False, resize=None,
                      min_faces_per_person=0):
    """Perform the actual data loading for the lfw people dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %
                         min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, color, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    return faces, target, target_names



```
164 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def __init__(self, n_values="auto", categorical_features="all",
                 dtype=np.float64, sparse=True, handle_unknown='error'):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            Input array of type int.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def _fit_transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape
        if (isinstance(self.n_values, six.string_types) and
                self.n_values == 'auto'):
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self.n_values, numbers.Integral):
            if (np.max(X, axis=0) >= self.n_values).any():
                raise ValueError("Feature out of bounds for n_values=%d"
                                 % self.n_values)
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self.n_values)
        else:
            try:
                n_values = np.asarray(self.n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected"
                                " 'auto', int or array of ints, got %r"
                                % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError("Shape mismatch: if n_values is an array,"
                                 " it has to be of shape (n_features,).")

        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices

        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()

        if (isinstance(self.n_values, six.string_types) and
                self.n_values == 'auto'):
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self.active_features_ = active_features

        return out if self.sparse else out.toarray()

    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            Input array of type int.
        """
        return _transform_selected(X, self._fit_transform,
                                   self.categorical_features, copy=True)

    
```
165 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py
```python
# Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Joel Nothman <joel.nothman@gmail.com>
# License: BSD 3 clause
from __future__ import division

import warnings
import numpy as np
from scipy import sparse
from math import sqrt

from ..metrics.pairwise import euclidean_distances
from ..base import TransformerMixin, ClusterMixin, BaseEstimator
from ..externals.six.moves import xrange
from ..utils import check_array
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..exceptions import NotFittedError, ConvergenceWarning
from .hierarchical import AgglomerativeClustering


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


def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_node2 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    dist = euclidean_distances(
        node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    n_clusters = dist.shape[0]

    farthest_idx = np.unravel_index(
        dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[[farthest_idx]]

    node1_closer = node1_dist < node2_dist
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _CFNode(object):
    
```
166 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class PredefinedSplit(BaseCrossValidator):
    """Predefined split cross-validator

    Provides train/test indices to split data into train/test sets using a
    predefined scheme specified by the user with the ``test_fold`` parameter.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    test_fold : array-like, shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set (i.e. include sample ``i`` in every training set) by
        setting ``test_fold[i]`` equal to -1.

    Examples
    --------
    >>> from sklearn.model_selection import PredefinedSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> test_fold = [0, 1, -1, 1]
    >>> ps = PredefinedSplit(test_fold)
    >>> ps.get_n_splits()
    2
    >>> print(ps)       # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))
    >>> for train_index, test_index in ps.split():
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 2 3] TEST: [0]
    TRAIN: [0 2] TEST: [1 3]
    """

    def __init__(self, test_fold):
        self.test_fold = np.array(test_fold, dtype=np.int)
        self.test_fold = column_or_1d(self.test_fold)
        self.unique_folds = np.unique(self.test_fold)
        self.unique_folds = self.unique_folds[self.unique_folds != -1]

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
        ind = np.arange(len(self.test_fold))
        for test_index in self._iter_test_masks():
            train_index = ind[np.logical_not(test_index)]
            test_index = ind[test_index]
            yield train_index, test_index

    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

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
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.unique_folds)



```
167 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
def _convert_arff_data(arff_data, col_slice_x, col_slice_y, shape=None):
    """
    converts the arff object into the appropriate matrix type (np.array or
    scipy.sparse.csr_matrix) based on the 'data part' (i.e., in the
    liac-arff dict, the object from the 'data' key)

    Parameters
    ----------
    arff_data : list or dict
        as obtained from liac-arff object

    col_slice_x : list
        The column indices that are sliced from the original array to return
        as X data

    col_slice_y : list
        The column indices that are sliced from the original array to return
        as y data

    Returns
    -------
    X : np.array or scipy.sparse.csr_matrix
    y : np.array
    """
    if isinstance(arff_data, Generator):
        if shape[0] == -1:
            count = -1
        else:
            count = shape[0] * shape[1]
        data = np.fromiter(itertools.chain.from_iterable(arff_data),
                           dtype='float64', count=count)
        data = data.reshape(*shape)
        X = data[:, col_slice_x]
        y = data[:, col_slice_y]
        return X, y
    elif isinstance(arff_data, tuple):
        arff_data_X = _split_sparse_columns(arff_data, col_slice_x)
        num_obs = max(arff_data[1]) + 1
        X_shape = (num_obs, len(col_slice_x))
        X = scipy.sparse.coo_matrix(
            (arff_data_X[0], (arff_data_X[1], arff_data_X[2])),
            shape=X_shape, dtype=np.float64)
        X = X.tocsr()
        y = _sparse_data_to_array(arff_data, col_slice_y)
        return X, y
    else:
        # This should never happen
        raise ValueError('Unexpected Data Type obtained from arff.')


def _get_data_info_by_name(name, version, data_home):
    """
    Utilizes the openml dataset listing api to find a dataset by
    name/version
    OpenML api function:
    https://www.openml.org/api_docs#!/data/get_data_list_data_name_data_name

    Parameters
    ----------
    name : str
        name of the dataset

    version : int or str
        If version is an integer, the exact name/version will be obtained from
        OpenML. If version is a string (value: "active") it will take the first
        version from OpenML that is annotated as active. Any other string
        values except "active" are treated as integer.

    data_home : str or None
        Location to cache the response. None if no cache is required.

    Returns
    -------
    first_dataset : json
        json representation of the first dataset object that adhired to the
        search criteria

    """
    if version == "active":
        # situation in which we return the oldest active version
        url = _SEARCH_NAME.format(name) + "/status/active/"
        error_msg = "No active dataset {} found.".format(name)
        json_data = _get_json_content_from_openml_api(url, error_msg, True,
                                                      data_home)
        res = json_data['data']['dataset']
        if len(res) > 1:
            warn("Multiple active versions of the dataset matching the name"
                 " {name} exist. Versions may be fundamentally different, "
                 "returning version"
                 " {version}.".format(name=name, version=res[0]['version']))
        return res[0]

    # an integer version has been provided
    url = (_SEARCH_NAME + "/data_version/{}").format(name, version)
    json_data = _get_json_content_from_openml_api(url, None, False,
                                                  data_home)
    if json_data is None:
        # we can do this in 1 function call if OpenML does not require the
        # specification of the dataset status (i.e., return datasets with a
        # given name / version regardless of active, deactivated, etc. )
        # TODO: feature request OpenML.
        url += "/status/deactivated"
        error_msg = "Dataset {} with version {} not found.".format(name,
                                                                   version)
        json_data = _get_json_content_from_openml_api(url, error_msg, True,
                                                      data_home)

    return json_data['data']['dataset'][0]



```
168 - /tmp/repos/scikit-learn/sklearn/cluster/_feature_agglomeration.py
```python
"""
Feature agglomeration. Base classes and functions for performing feature
agglomeration.
"""
# Author: V. Michel, A. Gramfort
# License: BSD 3 clause

import numpy as np

from ..base import TransformerMixin
from ..utils import check_array
from ..utils.validation import check_is_fitted
from scipy.sparse import issparse

###############################################################################
# Mixin class for feature agglomeration.


class AgglomerationTransform(TransformerMixin):
    """
    A class for feature agglomeration via the transform interface
    """

    pooling_func = np.mean

    def transform(self, X):
        """
        Transform a new matrix using the built clustering

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features] or [n_features]
            A M by N array of M observations in N dimensions or a length
            M array of M one-dimensional observations.

        Returns
        -------
        Y : array, shape = [n_samples, n_clusters] or [n_clusters]
            The pooled values for each feature cluster.
        """
        check_is_fitted(self, "labels_")

        pooling_func = self.pooling_func
        X = check_array(X)
        if len(self.labels_) != X.shape[1]:
            raise ValueError("X has a different number of features than "
                             "during fitting.")
        if pooling_func == np.mean and not issparse(X):
            size = np.bincount(self.labels_)
            n_samples = X.shape[0]
            # a fast way to compute the mean of grouped features
            nX = np.array([np.bincount(self.labels_, X[i, :]) / size
                          for i in range(n_samples)])
        else:
            nX = []
            for l in np.unique(self.labels_):
                nX.append(pooling_func(X[:, self.labels_ == l], axis=1))
            nX = np.array(nX).T
        return nX

    def inverse_transform(self, Xred):
        """
        Inverse the transformation.
        Return a vector of size nb_features with the values of Xred assigned
        to each group of features

        Parameters
        ----------
        Xred : array-like, shape=[n_samples, n_clusters] or [n_clusters,]
            The values to be assigned to each cluster of samples

        Returns
        -------
        X : array, shape=[n_samples, n_features] or [n_features]
            A vector of size n_samples with the values of Xred assigned to
            each of the cluster of samples.
        """
        check_is_fitted(self, "labels_")

        unil, inverse = np.unique(self.labels_, return_inverse=True)
        return Xred[..., inverse]

```
169 - /tmp/repos/scikit-learn/sklearn/decomposition/base.py
```python
"""Principal Component Analysis Base Classes"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <denis-alexander.engemann@inria.fr>
#         Kyle Kastner <kastnerkyle@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import linalg

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..externals import six
from abc import ABCMeta, abstractmethod


class _BasePCA(six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin)):
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    def get_covariance(self):
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where  S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        cov = np.dot(components_.T * exp_var_diff, components_)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov

    def get_precision(self):
        """Compute data precision matrix with the generative model.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return np.eye(n_features) / self.noise_variance_
        if self.n_components_ == n_features:
            return linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        precision = np.dot(components_, components_.T) / self.noise_variance_
        precision.flat[::len(precision) + 1] += 1. / exp_var_diff
        precision = np.dot(components_.T,
                           np.dot(linalg.inv(precision), components_))
        precision /= -(self.noise_variance_ ** 2)
        precision.flat[::len(precision) + 1] += 1. / self.noise_variance_
        return precision

    @abstractmethod
    def fit(X, y=None):
        """Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    
```
170 - /tmp/repos/scikit-learn/sklearn/ensemble/voting.py
```python
def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                 flatten_transform=True):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        return super().fit(X, transformed_y, sample_weight)

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like, shape (n_samples,)
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators_')
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self._weights_not_none)),
                axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        avg : array-like, shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns array-like of shape (n_classifiers, n_samples *
                n_classes), being class probabilities calculated by each
                classifier.
            If `voting='soft' and `flatten_transform=False`:
                array-like of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                array-like of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators_')

        if self.voting == 'soft':
            probas = self._collect_probas(X)
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)

        else:
            return self._predict(X)



```
171 - /tmp/repos/scikit-learn/sklearn/linear_model/ridge.py
```python
def _pre_compute(self, X, y, centered_kernel=True):
        # even if X is very sparse, K is usually very dense
        K = safe_sparse_dot(X, X.T, dense_output=True)
        # the following emulates an additional constant regressor
        # corresponding to fit_intercept=True
        # but this is done only when the features have been centered
        if centered_kernel:
            K += np.ones_like(K)
        v, Q = linalg.eigh(K)
        QT_y = np.dot(Q.T, y)
        return v, Q, QT_y

    def _decomp_diag(self, v_prime, Q):
        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        return (v_prime * Q ** 2).sum(axis=-1)

    def _diag_dot(self, D, B):
        # compute dot(diag(D), B)
        if len(B.shape) > 1:
            # handle case where B is > 1-d
            D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
        return D * B

    def _errors_and_values_helper(self, alpha, y, v, Q, QT_y):
        """Helper function to avoid code duplication between self._errors and
        self._values.

        Notes
        -----
        We don't construct matrix G, instead compute action on y & diagonal.
        """
        w = 1. / (v + alpha)
        constant_column = np.var(Q, 0) < 1.e-12
        # detect constant columns
        w[constant_column] = 0  # cancel the regularization for the intercept

        c = np.dot(Q, self._diag_dot(w, QT_y))
        G_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return (c / G_diag) ** 2, c

    def _values(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return y - (c / G_diag), c

    def _pre_compute_svd(self, X, y, centered_kernel=True):
        if sparse.issparse(X):
            raise TypeError("SVD not supported for sparse matrices")
        if centered_kernel:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        # to emulate fit_intercept=True situation, add a column on ones
        # Note that by centering, the other columns are orthogonal to that one
        U, s, _ = linalg.svd(X, full_matrices=0)
        v = s ** 2
        UT_y = np.dot(U.T, y)
        return v, U, UT_y

    def _errors_and_values_svd_helper(self, alpha, y, v, U, UT_y):
        """Helper function to avoid code duplication between self._errors_svd
        and self._values_svd.
        """
        constant_column = np.var(U, 0) < 1.e-12
        # detect columns colinear to ones
        w = ((v + alpha) ** -1) - (alpha ** -1)
        w[constant_column] = - (alpha ** -1)
        # cancel the regularization for the intercept
        c = np.dot(U, self._diag_dot(w, UT_y)) + (alpha ** -1) * y
        G_diag = self._decomp_diag(w, U) + (alpha ** -1)
        if len(y.shape) != 1:
            # handle case where y is 2-d
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return (c / G_diag) ** 2, c

    def _values_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return y - (c / G_diag), c

    
```
172 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def partial_fit(self, X, y=None):
        """Online computation of max absolute value of X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            mins, maxs = min_max_axis(X, axis=0)
            max_abs = np.maximum(np.abs(mins), np.abs(maxs))
        else:
            max_abs = np.abs(X).max(axis=0)

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next passes
        else:
            max_abs = np.maximum(self.max_abs_, max_abs)
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs
        self.scale_ = _handle_zeros_in_scale(max_abs)
        return self

    def transform(self, X):
        """Scale the data

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be scaled.
        """
        check_is_fitted(self, 'scale_')
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            inplace_column_scale(X, 1.0 / self.scale_)
        else:
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be transformed back.
        """
        check_is_fitted(self, 'scale_')
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            inplace_column_scale(X, self.scale_)
        else:
            X *= self.scale_
        return X


def maxabs_scale(X, axis=0, copy=True):
    """Scale each feature to the [-1, 1] range without breaking the sparsity.

    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    See also
    --------
    MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.

    # If copy is required, it will be done inside the scaler object.
    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
                    ensure_2d=False, dtype=FLOAT_DTYPES)
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MaxAbsScaler(copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X


class RobustScaler(BaseEstimator, TransformerMixin):
    
```
173 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimtor."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                       dtype=np.float64)
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(
                X, self.init_).astype(np.float64)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate,
                       raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        raw_predictions : generator of array, shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate,
                          raw_predictions)
            yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        self._check_initialized()

        relevant_trees = [tree
                          for stage in self.estimators_ for tree in stage
                          if tree.tree_.node_count > 1]
        if not relevant_trees:
            # degenerate case where all trees have only one node
            return np.zeros(shape=self.n_features_, dtype=np.float64)

        relevant_feature_importances = [
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in relevant_trees
        ]
        avg_feature_importances = np.mean(relevant_feature_importances,
                                          axis=0, dtype=np.float64)
        return avg_feature_importances / np.sum(avg_feature_importances)

    def _validate_y(self, y, sample_weight):
        # 'sample_weight' is not utilised but is used for
        # consistency with similar method _validate_y of GBC
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(DOUBLE)
        # Default implementation
        return y

    
```
174 - /tmp/repos/scikit-learn/sklearn/manifold/isomap.py
```python
def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.

        Returns
        -------
        reconstruction_error : float

        Notes
        -------
        The cost function of an isomap embedding is

        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:

        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        G = -0.5 * self.dist_matrix_ ** 2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.lambdas_
        return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]

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

    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = check_array(X)
        distances, indices = self.nbrs_.kneighbors(X, return_distance=True)

        # Create the graph of shortest distances from X to self.training_data_
        # via the nearest neighbors of X.
        # This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:
        G_X = np.zeros((X.shape[0], self.training_data_.shape[0]))
        for i in range(X.shape[0]):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] +
                            distances[i][:, None], 0)

        G_X **= 2
        G_X *= -0.5

        return self.kernel_pca_.transform(G_X)

```
175 - /tmp/repos/scikit-learn/sklearn/cluster/mean_shift_.py
```python
def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

        return pairwise_distances_argmin(X, self.cluster_centers_)

```
176 - /tmp/repos/scikit-learn/sklearn/pipeline.py
```python
def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and "
                                "transform. '%s' (type %s) doesn't" %
                                (t, type(t)))

    def _iter(self):
        """Generate (name, est, weight) tuples excluding None transformers
        """
        get_weight = (self.transformer_weights or {}).get
        return ((name, trans, get_weight(name))
                for name, trans in self.transformer_list
                if trans is not None)

    def get_feature_names(self):
        """Get feature names from all transformers.

        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not "
                                     "provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            feature_names.extend([name + "__" + f for f in
                                  trans.get_feature_names()])
        return feature_names

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : FeatureUnion
            This estimator
        """
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X, y)
            for _, trans, _ in self._iter())
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, None if old is None else next(transformers))
            for name, old in self.transformer_list
        ]



```
177 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        raw_predictions = self.decision_function(X)
        encoded_labels = \
            self.loss_._raw_prediction_to_decision(raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = \
                self.loss_._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        raw_predictions = self.decision_function(X)
        try:
            return self.loss_._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    
```
178 - /tmp/repos/scikit-learn/sklearn/datasets/mldata.py
```python
"""Fetch an mldata.org data set

    If the file does not exist yet, it is downloaded from mldata.org .

    mldata.org does not have an enforced convention for storing data or
    naming the columns in a data set. The default behavior of this function
    works well with the most common cases:

      1) data values are stored in the column 'data', and target values in the
         column 'label'
      2) alternatively, the first column stores target values, and the second
         data values
      3) the data array is stored as `n_features x n_samples` , and thus needs
         to be transposed to match the `sklearn` standard

    Keyword arguments allow to adapt these defaults to specific data sets
    (see parameters `target_name`, `data_name`, `transpose_data`, and
    the examples below).

    mldata.org data sets may have multiple columns, which are stored in the
    Bunch object with their original name.

    Parameters
    ----------

    dataname : str
        Name of the data set on mldata.org,
        e.g.: "leukemia", "Whistler Daily Snowfall", etc.
        The raw name is automatically converted to a mldata.org URL .

    target_name : optional, default: 'label'
        Name or index of the column containing the target values.

    data_name : optional, default: 'data'
        Name or index of the column containing the data.

    transpose_data : optional, default: True
        If True, transpose the downloaded data array.

    data_home : optional, default: None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    Returns
    -------

    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'DESCR', the full description of the dataset, and
        'COL_NAMES', the original names of the dataset columns.

    Examples
    --------
    Load the 'iris' dataset from mldata.org:

    >>> from sklearn.datasets.mldata import fetch_mldata
    >>> import tempfile
    >>> test_data_home = tempfile.mkdtemp()

    >>> iris = fetch_mldata('iris', data_home=test_data_home)
    >>> iris.target.shape
    (150,)
    >>> iris.data.shape
    (150, 4)

    Load the 'leukemia' dataset from mldata.org, which needs to be transposed
    to respects the scikit-learn axes convention:

    >>> leuk = fetch_mldata('leukemia', transpose_data=True,
    ...                     data_home=test_data_home)
    >>> leuk.data.shape
    (72, 7129)

    Load an alternative 'iris' dataset, which has different names for the
    columns:

    >>> iris2 = fetch_mldata('datasets-UCI iris', target_name=1,
    ...                      data_name=0, data_home=test_data_home)
    >>> iris3 = fetch_mldata('datasets-UCI iris',
    ...                      target_name='class', data_name='double0',
    ...                      data_home=test_data_home)

    >>> import shutil
    >>> shutil.rmtree(test_data_home)
    """

    # normalize dataset name
    dataname = mldata_filename(dataname)

    # check if this data set has been already downloaded
    data_home = get_data_home(data_home=data_home)
    data_home = join(data_home, 'mldata')
    if not exists(data_home):
        os.makedirs(data_home)

    matlab_name = dataname + '.mat'
    filename = join(data_home, matlab_name)

    # if the file does not exist, download it
    if not exists(filename):
        urlname = MLDATA_BASE_URL % quote(dataname)
        try:
            mldata_url = urlopen(urlname)
        except HTTPError as e:
            if e.code == 404:
                e.msg = "Dataset '%s' not found on mldata.org." % dataname
            raise
        # store Matlab file
        try:
            with open(filename, 'w+b') as matlab_file:
                copyfileobj(mldata_url, matlab_file)
        except:
            os.remove(filename)
            raise
        mldata_url.close()

    # load dataset matlab file
    with open(filename, 'rb') as matlab_file:
        matlab_dict = io.loadmat(matlab_file, struct_as_record=True)

    # -- extract data from matlab_dict

    # flatten column names
    col_names = [str(descr[0])
                 for descr in matlab_dict['mldata_descr_ordering'][0]]

    
```
179 - /tmp/repos/scikit-learn/examples/exercises/plot_iris_exercise.py
```python
"""
================================
SVM Exercise
================================

A tutorial exercise for using different SVM kernels.

This exercise is used in the :ref:`using_kernels_tut` part of the
:ref:`supervised_learning_tut` section of the :ref:`stat_learn_tut_index`.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()
```
180 - /tmp/repos/scikit-learn/sklearn/impute/_iterative.py
```python
def transform(self, X):
        """Imputes all missing values in X.

        Note that this is stochastic, and that if random_state is not fixed,
        repeated calls, or permuted input, will yield different results.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        check_is_fitted(self, 'initial_imputer_')

        if self.add_indicator:
            X_trans_indicator = self.indicator_.transform(X)

        X, Xt, mask_missing_values = self._initial_imputation(X)

        if self.n_iter_ == 0 or np.all(mask_missing_values):
            return Xt

        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        i_rnd = 0
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        for it, estimator_triplet in enumerate(self.imputation_sequence_):
            Xt, _ = self._impute_one_feature(
                Xt,
                mask_missing_values,
                estimator_triplet.feat_idx,
                estimator_triplet.neighbor_feat_idx,
                estimator=estimator_triplet.estimator,
                fit_mode=False
            )
            if not (it + 1) % imputations_per_round:
                if self.verbose > 1:
                    print('[IterativeImputer] Ending imputation round '
                          '%d/%d, elapsed time %0.2f'
                          % (i_rnd + 1, self.n_iter_, time() - start_t))
                i_rnd += 1

        Xt[~mask_missing_values] = X[~mask_missing_values]

        if self.add_indicator:
            Xt = np.hstack((Xt, X_trans_indicator))
        return Xt

    def fit(self, X, y=None):
        """Fits the imputer on X and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        self.fit_transform(X)
        return self

    def _more_tags(self):
        return {'allow_nan': True}

```
181 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
def check_cv(cv, X=None, y=None, classifier=False):
    """Input checker utility for building a CV in a user friendly way.

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :func:`sklearn.model_selection.check_cv` instead.

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if classifier is True and ``y`` is binary or
        multiclass, :class:`StratifiedKFold` is used. In all other cases,
        :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    X : array-like
        The data the cross-val object will be applied on.

    y : array-like
        The target variable for a supervised learning problem.

    classifier : boolean optional
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validation generator instance.
        The return value is guaranteed to be a cv generator instance, whatever
        the input type.
    """
    is_sparse = sp.issparse(X)
    if cv is None:
        cv = 3
    if isinstance(cv, numbers.Integral):
        if classifier:
            if type_of_target(y) in ['binary', 'multiclass']:
                cv = StratifiedKFold(y, cv)
            else:
                cv = KFold(_num_samples(y), cv)
        else:
            if not is_sparse:
                n_samples = len(X)
            else:
                n_samples = X.shape[0]
            cv = KFold(n_samples, cv)
    return cv



```
182 - /tmp/repos/scikit-learn/examples/cross_decomposition/plot_compare_cross_decomposition.py
```python
plt.scatter(Y_test_r[:, 0], Y_test_r[:, 1], label="test",
            marker="*", c="r", s=50)
plt.xlabel("Y comp. 1")
plt.ylabel("Y comp. 2")
plt.title('Y comp. 1 vs Y comp. 2 , (test corr = %.2f)'
          % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1])
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.show()

# #############################################################################
# PLS regression, with multivariate response, a.k.a. PLS2

n = 1000
q = 3
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
B = np.array([[1, 2] + [0] * (p - 2)] * q).T
# each Yj = 1*X1 + 2*X2 + noize
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5

pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
# compare pls2.coef_ with B
print("Estimated B")
print(np.round(pls2.coef_, 1))
pls2.predict(X)

# PLS regression, with univariate response, a.k.a. PLS1

n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5
pls1 = PLSRegression(n_components=3)
pls1.fit(X, y)
# note that the number of components exceeds 1 (the dimension of y)
print("Estimated betas")
print(np.round(pls1.coef_, 1))

# #############################################################################
# CCA (PLS mode B with symmetric deflation)

cca = CCA(n_components=2)
cca.fit(X_train, Y_train)
X_train_r, Y_train_r = cca.transform(X_train, Y_train)
X_test_r, Y_test_r = cca.transform(X_test, Y_test)

```
183 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_fit_returns_self(name, estimator_orig):
    """Check if self is returned when calling fit"""
    X, y = make_blobs(random_state=0, n_samples=9, n_features=4)
    # some want non-negative input
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    assert_true(estimator.fit(X, y) is estimator)


@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise either AttributeError or ValueError.
    The specific exception type NotFittedError inherits from both and can
    therefore be adequately raised for that purpose.
    """

    # Common test for Regressors, Classifiers and Outlier detection estimators
    X, y = _boston_subset()

    est = clone(estimator_orig)

    msg = "fit"
    if hasattr(est, 'predict'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict, X)

    if hasattr(est, 'decision_function'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.decision_function, X)

    if hasattr(est, 'predict_proba'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict_proba, X)

    if hasattr(est, 'predict_log_proba'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict_log_proba, X)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_2d(name, estimator_orig):
    if "MultiTask" in name:
        # These only work on 2d, so this test makes no sense
        return
    rnd = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)), estimator_orig)
    y = np.arange(10) % 3
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % (
        ", ".join([str(w_x) for w_x in w]))
    if name not in MULTI_OUTPUT:
        # check that we warned if we don't support multi-output
        assert_greater(len(w), 0, msg)
        assert_true("DataConversionWarning('A column-vector y"
                    " was passed when a 1d array was expected" in msg)
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())



```
184 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
def _fit(self, X):
        self._check_algorithm_metric()
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        effective_p = self.effective_metric_params_.get('p', self.p)
        if self.metric in ['wminkowski', 'minkowski']:
            self.effective_metric_params_['p'] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == 'minkowski':
            p = self.effective_metric_params_.pop('p', 2)
            if p < 1:
                raise ValueError("p must be greater than one "
                                 "for minkowski metric")
            elif p == 1:
                self.effective_metric_ = 'manhattan'
            elif p == 2:
                self.effective_metric_ = 'euclidean'
            elif p == np.inf:
                self.effective_metric_ = 'chebyshev'
            else:
                self.effective_metric_params_['p'] = p

        if isinstance(X, NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            return self

        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'ball_tree'
            return self

        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'kd_tree'
            return self

        X = check_array(X, accept_sparse='csr')

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("n_samples must be greater than 0")

        if issparse(X):
            if self.algorithm not in ('auto', 'brute'):
                warnings.warn("cannot use tree with sparse input: "
                              "using brute force")
            if self.effective_metric_ not in VALID_METRICS_SPARSE['brute'] \
                    and not callable(self.effective_metric_):

                raise ValueError("metric '%s' not valid for sparse input"
                                 % self.effective_metric_)
            self._fit_X = X.copy()
            self._tree = None
            self._fit_method = 'brute'
            return self

        self._fit_method = self.algorithm
        self._fit_X = X

        if self._fit_method == 'auto':
            # A tree approach is better for small number of neighbors,
            # and KDTree is generally faster when available
            if ((self.n_neighbors is None or
                 self.n_neighbors < self._fit_X.shape[0] // 2) and
                    self.metric != 'precomputed'):
                if self.effective_metric_ in VALID_METRICS['kd_tree']:
                    self._fit_method = 'kd_tree'
                elif (callable(self.effective_metric_) or
                        self.effective_metric_ in VALID_METRICS['ball_tree']):
                    self._fit_method = 'ball_tree'
                else:
                    self._fit_method = 'brute'
            else:
                self._fit_method = 'brute'

        if self._fit_method == 'ball_tree':
            self._tree = BallTree(X, self.leaf_size,
                                  metric=self.effective_metric_,
                                  **self.effective_metric_params_)
        elif self._fit_method == 'kd_tree':
            self._tree = KDTree(X, self.leaf_size,
                                metric=self.effective_metric_,
                                **self.effective_metric_params_)
        elif self._fit_method == 'brute':
            self._tree = None
        else:
            raise ValueError("algorithm = '%s' not recognized"
                             % self.algorithm)

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    "Expected n_neighbors > 0. Got %d" %
                    self.n_neighbors
                )

        return self

    @property
    def _pairwise(self):
        # For cross-validation routines to split data correctly
        return self.metric == 'precomputed'


class KNeighborsMixin(object):
    """Mixin for k-neighbors searches"""

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        
```
185 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _apply_func(func, X):
    # apply function on the whole set and on mini batches
    result_full = func(X)
    n_features = X.shape[1]
    result_by_batch = [func(batch.reshape(1, n_features))
                       for batch in X]
    # func can output tuple (e.g. score_samples)
    if type(result_full) == tuple:
        result_full = result_full[0]
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    return np.ravel(result_full), np.ravel(result_by_batch)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini bathes or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "score_samples", "predict_proba"]:

        msg = ("{method} of {name} is not invariant when applied "
               "to a subset.").format(method=method, name=name)
        # TODO remove cases when corrected
        if (name, method) in [('SVC', 'decision_function'),
                              ('SparsePCA', 'transform'),
                              ('MiniBatchSparsePCA', 'transform'),
                              ('BernoulliRBM', 'score_samples')]:
            raise SkipTest(msg)

        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_func(
                getattr(estimator, method), X)
            assert_allclose(result_full, result_by_batch,
                            atol=1e-7, err_msg=msg)


@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(1, 10))
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)

    msgs = ["1 sample", "n_samples = 1", "n_samples=1", "one sample",
            "1 class", "one class"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e


@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # ensure two labels in subsample for RandomizedLogisticRegression
    if name == 'RandomizedLogisticRegression':
        estimator.sample_fraction = 1
    # ensure non skipped trials for RANSACRegressor
    if name == 'RANSACRegressor':
        estimator.residual_threshold = 0.5

    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator, 1)

    msgs = ["1 feature(s)", "n_features = 1", "n_features=1"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e



```
186 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py
```python
def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X)
                       for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    
```
187 - /tmp/repos/scikit-learn/examples/svm/plot_iris.py
```python
"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

```
188 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py
```python
def fit(self, X, y=None):
        """ Create affinity matrix from negative euclidean distances, then
        apply affinity propagation clustering.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.

        y : Ignored

        """
        X = check_array(X, accept_sparse='csr')
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        self.cluster_centers_indices_, self.labels_, self.n_iter_ = \
            affinity_propagation(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_indices_")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Predict method is not supported when "
                             "affinity='precomputed'.")

        if self.cluster_centers_.size > 0:
            return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn("This model does not have any cluster centers "
                          "because affinity propagation did not converge. "
                          "Labeling every sample as '-1'.", ConvergenceWarning)
            return np.array([-1] * X.shape[0])

```
189 - /tmp/repos/scikit-learn/sklearn/gaussian_process/kernels.py
```python
class CompoundKernel(Kernel):
    """Kernel which is composed of a set of other kernels.

    .. versionadded:: 0.18
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return dict(kernels=self.kernels)

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k_dims = self.k1.n_dims
        for i, kernel in enumerate(self.kernels):
            kernel.theta = theta[i * k_dims:(i + 1) * k_dims]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return np.vstack([kernel.bounds for kernel in self.kernels])

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y, n_kernels)
            Kernel k(X, Y)

        K_gradient : array, shape (n_samples_X, n_samples_X, n_dims, n_kernels)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K = []
            K_grad = []
            for kernel in self.kernels:
                K_single, K_grad_single = kernel(X, Y, eval_gradient)
                K.append(K_single)
                K_grad.append(K_grad_single[..., np.newaxis])
            return np.dstack(K), np.concatenate(K_grad, 3)
        else:
            return np.dstack([kernel(X, Y, eval_gradient)
                              for kernel in self.kernels])

    def __eq__(self, b):
        if type(self) != type(b) or len(self.kernels) != len(b.kernels):
            return False
        return np.all([self.kernels[i] == b.kernels[i]
                       for i in range(len(self.kernels))])

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return np.all([kernel.is_stationary() for kernel in self.kernels])

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X, n_kernels)
            Diagonal of kernel k(X, X)
        """
        return np.vstack([kernel.diag(X) for kernel in self.kernels]).T



```
190 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
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

        super(BaggingClassifier, self).__init__(
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
        super(BaggingClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_
        classes_ = self.classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~samples

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
191 - /tmp/repos/scikit-learn/sklearn/linear_model/omp.py
```python
def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.

        y : array-like, shape [n_samples]
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : object
            returns an instance of self.
        """
        X, y = check_X_y(X, y, y_numeric=True, ensure_min_features=2,
                         estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)
        cv = check_cv(self.cv, classifier=False)
        max_iter = (min(max(int(0.1 * X.shape[1]), 5), X.shape[1])
                    if not self.max_iter
                    else self.max_iter)
        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_omp_path_residues)(
                X[train], y[train], X[test], y[test], self.copy,
                self.fit_intercept, self.normalize, max_iter)
            for train, test in cv.split(X))

        min_early_stop = min(fold.shape[0] for fold in cv_paths)
        mse_folds = np.array([(fold[:min_early_stop] ** 2).mean(axis=1)
                              for fold in cv_paths])
        best_n_nonzero_coefs = np.argmin(mse_folds.mean(axis=0)) + 1
        self.n_nonzero_coefs_ = best_n_nonzero_coefs
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=best_n_nonzero_coefs,
                                        fit_intercept=self.fit_intercept,
                                        normalize=self.normalize)
        omp.fit(X, y)
        self.coef_ = omp.coef_
        self.intercept_ = omp.intercept_
        self.n_iter_ = omp.n_iter_
        return self

```
192 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def paired_euclidean_distances(X, Y):
    """
    Computes the paired euclidean distances between X and Y

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray (n_samples, )
    """
    X, Y = check_paired_arrays(X, Y)
    return row_norms(X - Y)


def paired_manhattan_distances(X, Y):
    """Compute the L1 distances between the vectors in X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray (n_samples, )
    """
    X, Y = check_paired_arrays(X, Y)
    diff = X - Y
    if issparse(diff):
        diff.data = np.abs(diff.data)
        return np.squeeze(np.array(diff.sum(axis=1)))
    else:
        return np.abs(diff).sum(axis=-1)


def paired_cosine_distances(X, Y):
    """
    Computes the paired cosine distances between X and Y

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray, shape (n_samples, )

    Notes
    ------
    The cosine distance is equivalent to the half the squared
    euclidean distance if each sample is normalized to unit norm
    """
    X, Y = check_paired_arrays(X, Y)
    return .5 * row_norms(normalize(X) - normalize(Y), squared=True)


PAIRED_DISTANCES = {
    'cosine': paired_cosine_distances,
    'euclidean': paired_euclidean_distances,
    'l2': paired_euclidean_distances,
    'l1': paired_manhattan_distances,
    'manhattan': paired_manhattan_distances,
    'cityblock': paired_manhattan_distances}


def paired_distances(X, Y, metric="euclidean", **kwds):
    """
    Computes the paired distances between X and Y.

    Computes the distances between (X[0], Y[0]), (X[1], Y[1]), etc...

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Array 1 for distance computation.

    Y : ndarray (n_samples, n_features)
        Array 2 for distance computation.

    metric : string or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        specified in PAIRED_DISTANCES, including "euclidean",
        "manhattan", or "cosine".
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    Returns
    -------
    distances : ndarray (n_samples, )

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_distances
    >>> X = [[0, 1], [1, 1]]
    >>> Y = [[0, 1], [2, 1]]
    >>> paired_distances(X, Y)
    array([ 0.,  1.])

    See also
    --------
    pairwise_distances : Computes the distance between every pair of samples
    """

    if metric in PAIRED_DISTANCES:
        func = PAIRED_DISTANCES[metric]
        return func(X, Y)
    elif callable(metric):
        # Check the matrix first (it is usually done by the metric)
        X, Y = check_paired_arrays(X, Y)
        distances = np.zeros(len(X))
        for i in range(len(X)):
            distances[i] = metric(X[i], Y[i])
        return distances
    else:
        raise ValueError('Unknown distance %s' % metric)


# Kernels

```
193 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class LeaveOneOut(BaseCrossValidator):
    """Leave-One-Out cross-validator

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.

    Note: ``LeaveOneOut()`` is equivalent to ``KFold(n_splits=n)`` and
    ``LeavePOut(p=1)`` where ``n`` is the number of samples.

    Due to the high number of test sets (which is the same as the
    number of samples) this cross-validation method can be very costly.
    For large datasets one should favor :class:`KFold`, :class:`ShuffleSplit`
    or :class:`StratifiedKFold`.

    Read more in the :ref:`User Guide <cross_validation>`.

    Examples
    --------
    >>> from sklearn.model_selection import LeaveOneOut
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([1, 2])
    >>> loo = LeaveOneOut()
    >>> loo.get_n_splits(X)
    2
    >>> print(loo)
    LeaveOneOut()
    >>> for train_index, test_index in loo.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [1] TEST: [0]
    [[3 4]] [[1 2]] [2] [1]
    TRAIN: [0] TEST: [1]
    [[1 2]] [[3 4]] [1] [2]

    See also
    --------
    LeaveOneGroupOut
        For splitting the data according to explicit, domain-specific
        stratification of the dataset.

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_indices(self, X, y=None, groups=None):
        return range(_num_samples(X))

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)



```
194 - /tmp/repos/scikit-learn/sklearn/decomposition/dict_learning.py
```python
def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the object itself
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        V, U, E, self.n_iter_ = dict_learning(
            X, n_components, self.alpha,
            tol=self.tol, max_iter=self.max_iter,
            method=self.fit_algorithm,
            n_jobs=self.n_jobs,
            code_init=self.code_init,
            dict_init=self.dict_init,
            verbose=self.verbose,
            random_state=random_state,
            return_n_iter=True,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code)
        self.components_ = U
        self.error_ = E
        return self


class MiniBatchDictionaryLearning(BaseEstimator, SparseCodingMixin):
    """Mini-batch dictionary learning

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || Y - U V ||_2^2 + alpha * || U ||_1
                    (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int,
        number of dictionary elements to extract

    alpha : float,
        sparsity controlling parameter

    n_iter : int,
        total number of iterations to perform

    fit_algorithm : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int or None, optional (default=None)
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    batch_size : int,
        number of samples in each mini-batch

    shuffle : bool,
        whether to shuffle the samples before forming batches

    dict_init : array of shape (n_components, n_features),
        initial value of the dictionary for warm restart scenarios

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 
```
195 - /tmp/repos/scikit-learn/sklearn/preprocessing/_target.py
```python
def _fit_transformer(self, y):
        if (self.transformer is not None and
                (self.func is not None or self.inverse_func is not None)):
            raise ValueError("'transformer' and functions 'func'/"
                             "'inverse_func' cannot both be set.")
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError("When 'func' is provided, 'inverse_func' must"
                                 " also be provided")
            self.transformer_ = FunctionTransformer(
                func=self.func, inverse_func=self.inverse_func, validate=True,
                check_inverse=self.check_inverse)
        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code should be modified accordingly. At the time to consider the
        # sample_prop feature, it is also a good use case to be considered.
        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            if not np.allclose(y_sel,
                               self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn("The provided functions or transformer are"
                              " not strictly inverse of each other. If"
                              " you are sure you want to proceed regardless"
                              ", set 'check_inverse=False'", UserWarning)

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        if self.regressor is None:
            from ..linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.fit_transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)
        if sample_weight is None:
            self.regressor_.fit(X, y_trans)
        else:
            self.regressor_.fit(X, y_trans, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Predict using the base regressor, applying inverse.

        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.

        """
        check_is_fitted(self, "regressor_")
        pred = self.regressor_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(
                pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (self._training_dim == 1 and
                pred_trans.ndim == 2 and pred_trans.shape[1] == 1):
            pred_trans = pred_trans.squeeze(axis=1)

        return pred_trans

```
196 - /tmp/repos/scikit-learn/sklearn/covariance/outlier_detection.py
```python
def decision_function(self, X, raw_values=False):
        """Compute the decision function of the given observations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        raw_values : bool
            Whether or not to consider raw Mahalanobis distances as the
            decision function. Must be False (default) for compatibility
            with the others outlier detection tools.

        Returns
        -------
        decision : array-like, shape (n_samples, )
            Decision function of the samples.
            It is equal to the Mahalanobis distances if `raw_values`
            is True. By default (``raw_values=False``), it is equal
            to the cubic root of the shifted Mahalanobis distances.
            In that case, the threshold for being an outlier is 0, which
            ensures a compatibility with other outlier detection tools
            such as the One-Class SVM.

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        mahal_dist = self.mahalanobis(X)
        if raw_values:
            decision = mahal_dist
        else:
            transformed_mahal_dist = mahal_dist ** 0.33
            decision = self.threshold_ ** 0.33 - transformed_mahal_dist

        return decision

    def predict(self, X):
        """Outlyingness of observations in X according to the fitted model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        is_outliers : array, shape = (n_samples, ), dtype = bool
            For each observation, tells whether or not it should be considered
            as an outlier according to the fitted model.

        threshold : float,
            The values of the less outlying point's decision function.

        """
        check_is_fitted(self, 'threshold_')
        X = check_array(X)
        is_inlier = -np.ones(X.shape[0], dtype=int)
        if self.contamination is not None:
            values = self.decision_function(X, raw_values=True)
            is_inlier[values <= self.threshold_] = 1
        else:
            raise NotImplementedError("You must provide a contamination rate.")

        return is_inlier

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = (n_samples,), optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

```
197 - /tmp/repos/scikit-learn/examples/svm/plot_iris_svc.py
```python
"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

```
198 - /tmp/repos/scikit-learn/sklearn/preprocessing/_discretization.py
```python
def fit(self, X, y=None):
        """Fits the estimator.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.

        y : ignored

        Returns
        -------
        self
        """
        X = check_array(X, dtype='numeric')

        valid_encode = ('onehot', 'onehot-dense', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. "
                             "Got encode={!r} instead."
                             .format(valid_encode, self.encode))
        valid_strategy = ('uniform', 'quantile', 'kmeans')
        if self.strategy not in valid_strategy:
            raise ValueError("Valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, self.strategy))

        n_features = X.shape[1]
        n_bins = self._validate_n_bins(n_features)

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            column = X[:, jj]
            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                warnings.warn("Feature %d is constant and will be "
                              "replaced with 0." % jj)
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == 'uniform':
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)

            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            elif self.strategy == 'kmeans':
                from ..cluster import KMeans  # fixes import loops

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(col_min, col_max, n_bins[jj] + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

                # 1D k-means procedure
                km = KMeans(n_clusters=n_bins[jj], init=init, n_init=1)
                centers = km.fit(column[:, None]).cluster_centers_[:, 0]
                # Must sort, centers may be unsorted even with sorted init
                centers.sort()
                bin_edges[jj] = (centers[1:] + centers[:-1]) * 0.5
                bin_edges[jj] = np.r_[col_min, bin_edges[jj], col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in ('quantile', 'kmeans'):
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warnings.warn('Bins whose width are too small (i.e., <= '
                                  '1e-8) in feature %d are removed. Consider '
                                  'decreasing the number of bins.' % jj)
                    n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        if 'onehot' in self.encode:
            self._encoder = OneHotEncoder(
                categories=[np.arange(i) for i in self.n_bins_],
                sparse=self.encode == 'onehot')
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_)), dtype=int))

        return self

    
```
199 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
def sample_example():
        _, n_classes = p_w_c.shape

        # pick a nonzero number of labels per document by rejection sampling
        y_size = n_classes + 1
        while (not allow_unlabeled and y_size == 0) or y_size > n_classes:
            y_size = generator.poisson(n_labels)

        # pick n classes
        y = set()
        while len(y) != y_size:
            # pick a class with probability P(c)
            c = np.searchsorted(cumulative_p_c,
                                generator.rand(y_size - len(y)))
            y.update(c)
        y = list(y)

        # pick a non-zero document length by rejection sampling
        n_words = 0
        while n_words == 0:
            n_words = generator.poisson(length)

        # generate a document of length n_words
        if len(y) == 0:
            # if sample does not belong to any class, generate noise word
            words = generator.randint(n_features, size=n_words)
            return words, y

        # sample words with replacement from selected classes
        cumulative_p_w_sample = p_w_c.take(y, axis=1).sum(axis=1).cumsum()
        cumulative_p_w_sample /= cumulative_p_w_sample[-1]
        words = np.searchsorted(cumulative_p_w_sample, generator.rand(n_words))
        return words, y

    X_indices = array.array('i')
    X_indptr = array.array('i', [0])
    Y = []
    for i in range(n_samples):
        words, y = sample_example()
        X_indices.extend(words)
        X_indptr.append(len(X_indices))
        Y.append(y)
    X_data = np.ones(len(X_indices), dtype=np.float64)
    X = sp.csr_matrix((X_data, X_indices, X_indptr),
                      shape=(n_samples, n_features))
    X.sum_duplicates()
    if not sparse:
        X = X.toarray()

    # return_indicator can be True due to backward compatibility
    if return_indicator in (True, 'sparse', 'dense'):
        lb = MultiLabelBinarizer(sparse_output=(return_indicator == 'sparse'))
        Y = lb.fit([range(n_classes)]).transform(Y)
    elif return_indicator is not False:
        raise ValueError("return_indicator must be either 'sparse', 'dense' "
                         'or False.')
    if return_distributions:
        return X, Y, p_c, p_w_c
    return X, Y


def make_hastie_10_2(n_samples=12000, random_state=None):
    """Generates data for binary classification used in
    Hastie et al. 2009, Example 10.2.

    The ten features are standard independent Gaussian and
    the target ``y`` is defined by::

      y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=12000)
        The number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, 10]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
           Learning Ed. 2", Springer, 2009.

    See also
    --------
    make_gaussian_quantiles: a generalization of this dataset approach
    """
    rs = check_random_state(random_state)

    shape = (n_samples, 10)
    X = rs.normal(size=shape).reshape(shape)
    y = ((X ** 2.0).sum(axis=1) > 9.34).astype(np.float64)
    y[y == 0.0] = -1.0

    return X, y


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    
```
200 - /tmp/repos/scikit-learn/sklearn/feature_selection/mutual_info_.py
```python
def _estimate_mi(X, y, discrete_features='auto', discrete_target=False,
                 n_neighbors=3, copy=True, random_state=None):
    """Estimate mutual information between the features and the target.

    Parameters
    ----------
    X : array_like or sparse matrix, shape (n_samples, n_features)
        Feature matrix.

    y : array_like, shape (n_samples,)
        Target vector.

    discrete_features : {'auto', bool, array_like}, default 'auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    discrete_target : bool, default False
        Whether to consider `y` as a discrete variable.

    n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables,
        see [1]_ and [2]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`.

    Returns
    -------
    mi : ndarray, shape (n_features,)
        Estimated mutual information between each feature and the target.
        A negative value will be replaced by 0.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    """
    X, y = check_X_y(X, y, accept_sparse='csc', y_numeric=not discrete_target)
    n_samples, n_features = X.shape

    if discrete_features == 'auto':
        discrete_features = issparse(X)

    if isinstance(discrete_features, bool):
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = np.asarray(discrete_features)
        if discrete_features.dtype != 'bool':
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features

    continuous_mask = ~discrete_mask
    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")

    rng = check_random_state(random_state)
    if np.any(continuous_mask):
        if copy:
            X = X.copy()

        if not discrete_target:
            X[:, continuous_mask] = scale(X[:, continuous_mask],
                                          with_mean=False, copy=False)

        # Add small noise to continuous features as advised in Kraskov et. al.
        X = X.astype(float)
        means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
        X[:, continuous_mask] += 1e-10 * means * rng.randn(
                n_samples, np.sum(continuous_mask))

    if not discrete_target:
        y = scale(y, with_mean=False)
        y += 1e-10 * np.maximum(1, np.mean(np.abs(y))) * rng.randn(n_samples)

    mi = [_compute_mi(x, y, discrete_feature, discrete_target, n_neighbors) for
          x, discrete_feature in moves.zip(_iterate_columns(X), discrete_mask)]

    return np.array(mi)



```
201 - /tmp/repos/scikit-learn/sklearn/multiclass.py
```python


    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iteration.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self
        """
        if _check_partial_fit_first_call(self, classes):
            if not hasattr(self.estimator, "partial_fit"):
                raise ValueError(("Base estimator {0}, doesn't have "
                                 "partial_fit method").format(self.estimator))
            self.estimators_ = [clone(self.estimator) for _ in range
                                (self.n_classes_)]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(("Mini-batch contains {0} while classes " +
                             "must be subset of {1}").format(np.unique(y),
                                                             self.classes_))

        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(estimator, X, column)
            for estimator, column in izip(self.estimators_, columns))

        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.
        """
        check_is_fitted(self, 'estimators_')
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            thresh = 0
        else:
            thresh = .5

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)
```
202 - /tmp/repos/scikit-learn/sklearn/learning_curve.py
```python
"""Utilities to evaluate models with respect to a variable
"""
# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import warnings

import numpy as np

from .base import is_classifier, clone
from .cross_validation import check_cv
from .externals.joblib import Parallel, delayed
from .cross_validation import _safe_split, _score, _fit_and_score
from .metrics.scorer import check_scoring
from .utils import indexable


warnings.warn("This module was deprecated in version 0.18 in favor of the "
              "model_selection module into which all the functions are moved."
              " This module will be removed in 0.20",
              DeprecationWarning)


__all__ = ['learning_curve', 'validation_curve']


def learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
                   cv=None, scoring=None, exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0,
                   error_score='raise'):
    
```
203 - /tmp/repos/scikit-learn/sklearn/svm/classes.py
```python
def fit(self, X, y=None, sample_weight=None, **params):
        """
        Detects the soft boundary of the set of samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        self : object

        Notes
        -----
        If X is not a C-ordered contiguous array it is copied.

        """
        super().fit(X, np.ones(_num_samples(X)),
                    sample_weight=sample_weight, **params)
        self.offset_ = -self._intercept_
        return self

    def decision_function(self, X):
        """Signed distance to the separating hyperplane.

        Signed distance is positive for an inlier and negative for an outlier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        dec : array-like, shape (n_samples,)
            Returns the decision function of the samples.
        """
        dec = self._decision_function(X).ravel()
        return dec

    def score_samples(self, X):
        """Raw scoring function of the samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        score_samples : array-like, shape (n_samples,)
            Returns the (unshifted) scoring function of the samples.
        """
        return self.decision_function(X) + self.offset_

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        y = super().predict(X)
        return np.asarray(y, dtype=np.intp)

```
204 - /tmp/repos/scikit-learn/benchmarks/bench_covertype.py
```python
"""
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

"""
from __future__ import division, print_function

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Arnaud Joly <arnaud.v.joly@gmail.com>
# License: BSD 3 clause

import os
from time import time
import argparse
import numpy as np

from sklearn.datasets import fetch_covtype, get_data_home
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import zero_one_loss
from sklearn.externals.joblib import Memory
from sklearn.utils import check_array

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'covertype_benchmark_data'),
                mmap_mode='r')


@memory.cache
def load_data(dtype=np.float32, order='C', random_state=13):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_covtype(download_if_missing=True, shuffle=True,
                         random_state=random_state)
    X = check_array(data['data'], dtype=dtype, order=order)
    y = (data['target'] != 1).astype(np.int)

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 522911
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Standardize first 10 features (the numerical ones)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    mean[10:] = 0.0
    std[10:] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test



```
205 - /tmp/repos/scikit-learn/sklearn/linear_model/base.py
```python
def _pre_fit(X, y, Xy, precompute, normalize, fit_intercept, copy):
    """Aux function used at beginning of fit in linear models"""
    n_samples, n_features = X.shape

    if sparse.isspmatrix(X):
        # copy is not needed here as X is not modified inplace when X is sparse
        precompute = False
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=fit_intercept, normalize=normalize,
            copy=False, return_mean=True)
    else:
        # copy was done in fit if necessary
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=fit_intercept, normalize=normalize, copy=copy)
    if hasattr(precompute, '__array__') and (
            fit_intercept and not np.allclose(X_offset, np.zeros(n_features)) or
            normalize and not np.allclose(X_scale, np.ones(n_features))):
        warnings.warn("Gram matrix was provided but X was centered"
                      " to fit intercept, "
                      "or X was normalized : recomputing Gram matrix.",
                      UserWarning)
        # recompute Gram
        precompute = 'auto'
        Xy = None

    # precompute if n_samples > n_features
    if isinstance(precompute, six.string_types) and precompute == 'auto':
        precompute = (n_samples > n_features)

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype,
                              order='C')
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, '__array__'):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, '__array__') and Xy is None:
        common_dtype = np.find_common_type([X.dtype, y.dtype], [])
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order='C')
            np.dot(X.T, y, out=Xy)
        else:
            # Make sure that Xy is always F contiguous even if X or y are not
            # contiguous: the goal is to make it fast to extract the data for a
            # specific target.
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype,
                          order='F')
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy

```
206 - /tmp/repos/scikit-learn/sklearn/cluster/bicluster.py
```python
def _fit(self, X):
        normalized_data, row_diag, col_diag = _scale_normalize(X)
        n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        u, v = self._svd(normalized_data, n_sv, n_discard=1)
        z = np.vstack((row_diag[:, np.newaxis] * u,
                       col_diag[:, np.newaxis] * v))

        _, labels = self._k_means(z, self.n_clusters)

        n_rows = X.shape[0]
        self.row_labels_ = labels[:n_rows]
        self.column_labels_ = labels[n_rows:]

        self.rows_ = np.vstack(self.row_labels_ == c
                               for c in range(self.n_clusters))
        self.columns_ = np.vstack(self.column_labels_ == c
                                  for c in range(self.n_clusters))


class SpectralBiclustering(BaseSpectral):
    
```
207 - /tmp/repos/scikit-learn/sklearn/feature_selection/univariate_selection.py
```python
def f_regression(X, y, center=True):
    """Univariate linear regression tests.

    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature seletion procedure, not
    a free standing feature selection procedure.

    This is done in 2 steps:

    1. The correlation between each regressor and the target is computed,
       that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
       std(y)).
    2. It is converted to an F score then to a p-value.

    For more on usage see the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will be tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    center : True, bool,
        If true, X and y will be centered.

    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.


    See also
    --------
    mutual_info_regression: Mutual information for a continuous target.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    SelectPercentile: Select features based on percentile of the highest
        scores.
    """
    X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64)
    n_samples = X.shape[0]

    # compute centered values
    # note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X
    if center:
        y = y - np.mean(y)
        if issparse(X):
            X_means = X.mean(axis=0).getA1()
        else:
            X_means = X.mean(axis=0)
        # compute the scaled standard deviations via moments
        X_norms = np.sqrt(row_norms(X.T, squared=True) -
                          n_samples * X_means ** 2)
    else:
        X_norms = row_norms(X.T)

    # compute the correlation
    corr = safe_sparse_dot(y, X)
    corr /= X_norms
    corr /= np.linalg.norm(y)

    # convert to p-value
    degrees_of_freedom = y.size - (2 if center else 1)
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)
    return F, pv


######################################################################
# Base classes


```
208 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.with_std:
                # First pass
                if not hasattr(self, 'n_samples_seen_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                    self.n_samples_seen_ = X.shape[0]
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_)
            else:
                self.mean_ = None
                self.var_ = None
        else:
            # First pass
            if not hasattr(self, 'n_samples_seen_'):
                self.mean_ = .0
                self.n_samples_seen_ = 0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, y='deprecated', copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        y : (ignored)
            .. deprecated:: 0.19
               This parameter will be removed in 0.21.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    
```
209 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
if sparse.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives.")
        if axis != 0:
            raise ValueError("Can only scale sparse matrix on axis=0, "
                             " got axis=%d" % axis)
        if with_std:
            _, var = mean_variance_axis(X, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        if with_mean:
            mean_ = np.mean(X, axis)
        if with_std:
            scale_ = np.std(X, axis)
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = Xr.mean(axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1, 0):
                warnings.warn("Numerical issues were encountered "
                              "when centering the data "
                              "and might not be solved. Dataset may "
                              "contain too large values. You may need "
                              "to prescale your features.")
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = Xr.mean(axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
                if not np.allclose(mean_2, 0):
                    warnings.warn("Numerical issues were encountered "
                                  "when scaling the data "
                                  "and might not be solved. The standard "
                                  "deviation of the data is probably "
                                  "very close to 0. ")
                    Xr -= mean_2
    return X


class MinMaxScaler(BaseEstimator, TransformerMixin):
    
```
210 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.

    max_train_size : int, optional
        Maximum size for a single training set.

    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(max_train_size=None, n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples.
    """
    def __init__(self, n_splits=3, max_train_size=None):
        super(TimeSeriesSplit, self).__init__(n_splits,
                                              shuffle=False,
                                              random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (indices[test_start - self.max_train_size:test_start],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])



```
**211 - /tmp/repos/scikit-learn/sklearn/datasets/california_housing.py**:
```python
def fetch_california_housing(data_home=None, download_if_missing=True):
    """Loader for the California housing dataset from StatLib.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : ndarray, shape [20640, 8]
        Each row corresponding to the 8 feature values in order.

    dataset.target : numpy array of shape (20640,)
        Each value corresponds to the average house value in units of 100,000.

    dataset.feature_names : array of length 8
        Array of ordered feature names used in the dataset.

    dataset.DESCR : string
        Description of the California housing dataset.

    Notes
    ------

    This dataset consists of 20,640 samples and 9 features.
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    filepath = _pkl_filepath(data_home, 'cal_housing.pkz')
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info('Downloading Cal. housing from {} to {}'.format(
            ARCHIVE.url, data_home))

        archive_path = _fetch_remote(ARCHIVE, dirname=data_home)

        with tarfile.open(mode="r:gz", name=archive_path) as f:
            cal_housing = np.loadtxt(
                f.extractfile('CaliforniaHousing/cal_housing.data'),
                delimiter=',')
            # Columns are not in the same order compared to the previous
            # URL resource on lib.stat.cmu.edu
            columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
            cal_housing = cal_housing[:, columns_index]

            joblib.dump(cal_housing, filepath, compress=6)
        remove(archive_path)

    else:
        cal_housing = joblib.load(filepath)

    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]

    target, data = cal_housing[:, 0], cal_housing[:, 1:]

    # avg rooms = total rooms / households
    data[:, 2] /= data[:, 5]

    # avg bed rooms = total bed rooms / households
    data[:, 3] /= data[:, 5]

    # avg occupancy = population / households
    data[:, 5] = data[:, 4] / data[:, 5]

    # target in units of 100,000
    target = target / 100000.0

    return Bunch(data=data,
                 target=target,
                 feature_names=feature_names,
                 DESCR=MODULE_DOCS)

```
