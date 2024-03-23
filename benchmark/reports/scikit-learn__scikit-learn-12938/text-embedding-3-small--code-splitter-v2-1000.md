0 - /tmp/repos/scikit-learn/examples/plot_compare_reduction.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.
"""

###############################################################################
# Illustration of ``Pipeline`` and ``GridSearchCV``
###############################################################################
# This section illustrates the use of a ``Pipeline`` with
# ``GridSearchCV``

# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

print(__doc__)

pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LinearSVC())
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
digits = load_digits()
grid.fit(digits.data, digits.target)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()

###############################################################################
# Caching transformers within a ``Pipeline``
###############################################################################
# It is sometimes worthwhile storing the state of a specific transformer
# since it could be used again. Using a pipeline in ``GridSearchCV`` triggers
# such situations. Therefore, we use the argument ``memory`` to enable caching.
#
# .. warning::
#     Note that this example is, however, only an illustration since for this
#     specific case fitting PCA is not necessarily slower than loading the
#     cache. Hence, use the ``memory`` constructor parameter when the fitting
#     of a transformer is costly.

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory

# Create a temporary folder to store the transformers of the pipeline
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)
cached_pipe = Pipeline([('reduce_dim', PCA()),
                        ('classify', LinearSVC())],
                       memory=memory)

# This time, a cached pipeline will be used within the grid search
grid = GridSearchCV(cached_pipe, cv=3, n_jobs=1, param_grid=param_grid)
digits = load_digits()
grid.fit(digits.data, digits.target)

# Delete the temporary cache before exiting
rmtree(cachedir)

###############################################################################
# The ``PCA`` fitting is only computed at the evaluation of the first

```
1 - /tmp/repos/scikit-learn/examples/compose/plot_compare_reduction.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.
"""

###############################################################################
# Illustration of ``Pipeline`` and ``GridSearchCV``
###############################################################################
# This section illustrates the use of a ``Pipeline`` with
# ``GridSearchCV``

# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

print(__doc__)

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', 'passthrough'),
    ('classify', LinearSVC(dual=False, max_iter=10000))
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)
digits = load_digits()
grid.fit(digits.data, digits.target)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()

###############################################################################
# Caching transformers within a ``Pipeline``
###############################################################################
# It is sometimes worthwhile storing the state of a specific transformer
# since it could be used again. Using a pipeline in ``GridSearchCV`` triggers
# such situations. Therefore, we use the argument ``memory`` to enable caching.
#
# .. warning::
#     Note that this example is, however, only an illustration since for this
#     specific case fitting PCA is not necessarily slower than loading the
#     cache. Hence, use the ``memory`` constructor parameter when the fitting
#     of a transformer is costly.

from joblib import Memory
from shutil import rmtree

# Create a temporary folder to store the transformers of the pipeline
location = 'cachedir'
memory = Memory(location=location, verbose=10)
cached_pipe = Pipeline([('reduce_dim', PCA()),
                        ('classify', LinearSVC(dual=False, max_iter=10000))],
                       memory=memory)

# This time, a cached pipeline will be used within the grid search
grid = GridSearchCV(cached_pipe, cv=5, n_jobs=1, param_grid=param_grid)
digits = load_digits()
grid.fit(digits.data, digits.target)

# Delete the temporary cache before exiting
memory.clear(warn=False)
rmtree(location)

###############################################################################
# The ``PCA`` fitting is only computed at the evaluation of the first

```
2 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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


def _yield_non_meta_checks(name, estimator):
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    yield check_dtype_object
    yield check_sample_weights_pandas_series
    yield check_sample_weights_list
    yield check_estimators_fit_returns_self
    yield check_complex_data

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION + ['SpectralEmbedding']:
        # SpectralEmbedding is non-deterministic,
        # see issue #4236
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    if name not in ['SimpleImputer', 'Imputer']:
        # Test that all estimators check their input for NaN's and infs
        yield check_estimators_nan_inf

    if name not in ['GaussianProcess']:
        # FIXME!
        # in particular GaussianProcess!
        yield check_estimators_overwrite_params
    if hasattr(estimator, 'sparsify'):
        yield check_sparsify_coefficients

    yield check_estimator_sparse_data

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle



```
3 - /tmp/repos/scikit-learn/examples/model_selection/grid_search_text_feature_extraction.py
```python
"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached and reused for the document
classification example.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.

Here is a sample output of a run on a quad-core machine::

  Loading 20 newsgroups dataset for categories:
  ['alt.atheism', 'talk.religion.misc']
  1427 documents
  2 categories

  Performing grid search...
  pipeline: ['vect', 'tfidf', 'clf']
  parameters:
  {'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
   'clf__n_iter': (10, 50, 80),
   'clf__penalty': ('l2', 'elasticnet'),
   'tfidf__use_idf': (True, False),
   'vect__max_n': (1, 2),
   'vect__max_df': (0.5, 0.75, 1.0),
   'vect__max_features': (None, 5000, 10000, 50000)}
  done in 1737.030s

  Best score: 0.940
  Best parameters set:
      clf__alpha: 9.9999999999999995e-07
      clf__n_iter: 50
      clf__penalty: 'elasticnet'
      tfidf__use_idf: True
      vect__max_n: 2
      vect__max_df: 0.75
      vect__max_features: 50000

"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

data = fetch_20newsgroups(subset='train', categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (5,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}


```
4 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if name in ('CCA', 'LocallyLinearEmbedding', 'KernelPCA') and _is_32bit():
        # Those transformers yield non-deterministic output when executed on
        # a 32bit Python. The same transformers are stable on 64bit Python.
        # FIXME: try to isolate a minimalistic reproduction case only depending
        # scipy and/or maybe generate a test dataset that does not
        # cause such unstable behaviors.
        msg = name + ' is non deterministic on 32bit Python'
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = np.arange(10) % 3
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert_true(args[1] in ["y", "Y"],
                        "Expected y or Y as second argument for method "
                        "%s of %s. Got arguments: %r."
                        % (func_name, type(estimator).__name__, args))


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
5 - /tmp/repos/scikit-learn/examples/preprocessing/plot_discretization_classification.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
Feature discretization
======================

A demonstration of feature discretization on synthetic classification datasets.
Feature discretization decomposes each feature into a set of bins, here equally
distributed in width. The discrete values are then one-hot encoded, and given
to a linear classifier. This preprocessing enables a non-linear behavior even
though the classifier is linear.

On this example, the first two rows represent linearly non-separable datasets
(moons and concentric circles) while the third is approximately linearly
separable. On the two linearly non-separable datasets, feature discretization
largely increases the performance of linear classifiers. On the linearly
separable dataset, feature discretization decreases the performance of linear
classifiers. Two non-linear classifiers are also shown for comparison.

This example should be taken with a grain of salt, as the intuition conveyed
does not necessarily carry over to real datasets. Particularly in
high-dimensional spaces, data can more easily be separated linearly. Moreover,
using feature discretization and one-hot encoding increases the number of
features, which easily lead to overfitting when the number of samples is small.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
# Code source: Tom Dupré la Tour
# Adapted from plot_classifier_comparison by Gaël Varoquaux and Andreas Müller
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

print(__doc__)

h = .02  # step size in the mesh


def get_name(estimator):
    name = estimator.__class__.__name__
    if name == 'Pipeline':
        name = [get_name(est[1]) for est in estimator.steps]
        name = ' + '.join(name)
    return name


# list of (estimator, param_grid), where param_grid is used in GridSearchCV
classifiers = [
    (LogisticRegression(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (LinearSVC(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'),
        LogisticRegression(random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'logisticregression__C': np.logspace(-2, 7, 10),
        }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'), LinearSVC(random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'linearsvc__C': np.logspace(-2, 7, 10),
        }),
    (GradientBoostingClassifier(n_estimators=50, random_state=0), {
        'learning_rate': np.logspace(-4, 0, 10)
    }),
    (SVC(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
]

names = [get_name(e) for e, g in classifiers]

n_samples = 100
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                        n_informative=2, random_state=2,
                        n_clusters_per_class=1)
]

fig, axes = plt.subplots(nrows=len(datasets), ncols=len(classifiers) + 1,
                         figsize=(21, 9))

cm = plt.cm.PiYG
cm_bright = ListedColormap(['#b30065', '#178000'])

# iterate over datasets

```
6 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting_higgsboson.py
```python
from urllib.request import urlretrieve
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


```
7 - /tmp/repos/scikit-learn/sklearn/model_selection/__init__.py
```python
from ._split import BaseCrossValidator
from ._split import KFold
from ._split import GroupKFold
from ._split import StratifiedKFold
from ._split import TimeSeriesSplit
from ._split import LeaveOneGroupOut
from ._split import LeaveOneOut
from ._split import LeavePGroupsOut
from ._split import LeavePOut
from ._split import RepeatedKFold
from ._split import RepeatedStratifiedKFold
from ._split import ShuffleSplit
from ._split import GroupShuffleSplit
from ._split import StratifiedShuffleSplit
from ._split import PredefinedSplit
from ._split import train_test_split
from ._split import check_cv

from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import cross_validate
from ._validation import learning_curve
from ._validation import permutation_test_score
from ._validation import validation_curve

from ._search import GridSearchCV
from ._search import RandomizedSearchCV
from ._search import ParameterGrid
from ._search import ParameterSampler
from ._search import fit_grid_point

__all__ = ('BaseCrossValidator',
           'GridSearchCV',
           'TimeSeriesSplit',
           'KFold',
           'GroupKFold',
           'GroupShuffleSplit',
           'LeaveOneGroupOut',
           'LeaveOneOut',
           'LeavePGroupsOut',
           'LeavePOut',
           'RepeatedKFold',
           'RepeatedStratifiedKFold',
           'ParameterGrid',
           'ParameterSampler',
           'PredefinedSplit',
           'RandomizedSearchCV',
           'ShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'check_cv',
           'cross_val_predict',
           'cross_val_score',
           'cross_validate',
           'fit_grid_point',
           'learning_curve',
           'permutation_test_score',
           'train_test_split',
           'validation_curve')
```
8 - /tmp/repos/scikit-learn/examples/preprocessing/plot_scaling_importance.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=========================================================
Importance of Feature Scaling
=========================================================

Feature scaling through standardization (or Z-score normalization)
can be an important preprocessing step for many machine learning
algorithms. Standardization involves rescaling the features such
that they have the properties of a standard normal distribution
with a mean of zero and a standard deviation of one.

While many algorithms (such as SVM, K-nearest neighbors, and logistic
regression) require features to be normalized, intuitively we can
think of Principle Component Analysis (PCA) as being a prime example
of when normalization is important. In PCA we are interested in the
components that maximize the variance. If one component (e.g. human
height) varies less than another (e.g. weight) because of their
respective scales (meters vs. kilos), PCA might determine that the
direction of maximal variance more closely corresponds with the
'weight' axis, if those features are not scaled. As a change in
height of one meter can be considered much more important than the
change in weight of one kilogram, this is clearly incorrect.

To illustrate this, PCA is performed comparing the use of data with
:class:`StandardScaler <sklearn.preprocessing.StandardScaler>` applied,
to unscaled data. The results are visualized and a clear difference noted.
The 1st principal component in the unscaled set can be seen. It can be seen
that feature #13 dominates the direction, being a whole two orders of
magnitude above the other features. This is contrasted when observing
the principal component for the scaled version of the data. In the scaled
version, the orders of magnitude are roughly the same across all the features.

The dataset used is the Wine Dataset available at UCI. This dataset
has continuous features that are heterogeneous in scale due to differing
properties that they measure (i.e alcohol content, and malic acid).

The transformed data is then used to train a naive Bayes classifier, and a
clear difference in prediction accuracies is observed wherein the dataset
which is scaled before PCA vastly outperforms the unscaled version.

"""
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)

# Code source: Tyler Lanigan <tylerlanigan@gmail.com>
#              Sebastian Raschka <mail@sebastianraschka.com>

# License: BSD 3 clause

RANDOM_STATE = 42
FIG_SIZE = (10, 7)


features, target = load_wine(return_X_y=True)

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

# Fit to data and predict using pipelined GNB and PCA.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

# Show prediction accuracies in scaled and unscaled data.
print('\nPrediction accuracy for the normal test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal componenets
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Scale and use PCA on X_train data for visualization.
scaler = std_clf.named_steps['standardscaler']
X_train_std = pca_std.transform(scaler.transform(X_train))

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)



```
9 - /tmp/repos/scikit-learn/examples/preprocessing/plot_discretization_classification.py
```python
for ds_cnt, (X, y) in enumerate(datasets):
    print('\ndataset %d\n---------' % ds_cnt)

    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.5, random_state=42)

    # create the grid for background colors
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot the dataset first
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    # plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    # iterate over classifiers
    for est_idx, (name, (estimator, param_grid)) in \
            enumerate(zip(names, classifiers)):
        ax = axes[ds_cnt, est_idx + 1]

        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('%s: %.2f' % (name, score))

        # plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]*[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(name.replace(' + ', '\n'))
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0'), size=15,
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
                transform=ax.transAxes, horizontalalignment='right')


plt.tight_layout()

# Add suptitles above the figure
plt.subplots_adjust(top=0.90)
suptitles = [
    'Linear classifiers',
    'Feature discretization and linear classifiers',
    'Non-linear classifiers',
]
for i, suptitle in zip([1, 3, 5], suptitles):
    ax = axes[0, i]
    ax.text(1.05, 1.25, suptitle, transform=ax.transAxes,
            horizontalalignment='center', size='x-large')
plt.show()

```
10 - /tmp/repos/scikit-learn/examples/impute/plot_iterative_imputer_variants_comparison.py
```python
"""
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
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

N_SPLITS = 5

rng = np.random.RandomState(0)

X_full, y_full = fetch_california_housing(return_X_y=True)
# ~2k samples is enough for the purpose of the example.
# Remove the following two lines for a slower run with different error bars.
X_full = X_full[::10]
y_full = y_full[::10]
n_samples, n_features = X_full.shape

# Estimate the score on the entire dataset, with no missing values
br_estimator = BayesianRidge()
score_full_data = pd.DataFrame(
    cross_val_score(
        br_estimator, X_full, y_full, scoring='neg_mean_squared_error',
        cv=N_SPLITS
    ),
    columns=['Full Data']
)

# Add a single missing value to each row
X_missing = X_full.copy()
y_missing = y_full
missing_samples = np.arange(n_samples)
missing_features = rng.choice(n_features, n_samples, replace=True)
X_missing[missing_samples, missing_features] = np.nan

# Estimate the score after imputation (mean and median strategies)
score_simple_imputer = pd.DataFrame()
for strategy in ('mean', 'median'):
    estimator = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=strategy),
        br_estimator
    )
    score_simple_imputer[strategy] = cross_val_score(
        estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
        cv=N_SPLITS
    )

# Estimate the score after iterative imputation of the missing values
# with different estimators
estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15)
]
score_iterative_imputer = pd.DataFrame()

```
11 - /tmp/repos/scikit-learn/examples/compose/plot_column_transformer_mixed_types.py
```python
"""
===================================
Column Transformer with Mixed Types
===================================

This example illustrates how to apply different preprocessing and
feature extraction pipelines to different subsets of features,
using :class:`sklearn.compose.ColumnTransformer`.
This is particularly handy for the case of datasets that contain
heterogeneous data types, since we may want to scale the
numeric features and one-hot encode the categorical ones.

In this example, the numeric data is standard-scaled after
mean-imputation, while the categorical data is one-hot
encoded after imputing missing values with a new category
(``'missing'``).

Finally, the preprocessing pipeline is integrated in a
full prediction pipeline using :class:`sklearn.pipeline.Pipeline`,
together with a simple classification model.
"""

# Author: Pedro Morales <part.morales@gmail.com>
#
# License: BSD 3 clause

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# Read data from Titanic dataset.
titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)

# We will train our classifier with the following features:
# Numeric Features:
# - age: float.
# - fare: float.
# Categorical Features:
# - embarked: categories encoded as strings {'C', 'S', 'Q'}.
# - sex: categories encoded as strings {'female', 'male'}.
# - pclass: ordinal integers {1, 2, 3}.

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))


###############################################################################
# Using the prediction pipeline in a grid search
###############################################################################
# Grid search can also be performed on the different preprocessing steps
# defined in the ``ColumnTransformer`` object, together with the classifier's
# hyperparameters as part of the ``Pipeline``.
# We will search for both the imputer strategy of the numeric preprocessing
# and the regularization parameter of the logistic regression using
# :class:`sklearn.model_selection.GridSearchCV`.


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))
```
12 - /tmp/repos/scikit-learn/benchmarks/bench_mnist.py
```python
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
13 - /tmp/repos/scikit-learn/examples/plot_digits_pipe.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Pipelining: chaining a PCA and a logistic regression
=========================================================

The PCA does an unsupervised dimensionality reduction, while the logistic
regression does the prediction.

We use a GridSearchCV to set the dimensionality of the PCA

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Plot the PCA spectrum
pca.fit(X_digits)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_digits, y_digits)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()
```
14 - /tmp/repos/scikit-learn/benchmarks/bench_mnist.py
```python
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



```
15 - /tmp/repos/scikit-learn/examples/model_selection/plot_grid_search_refit_callable.py
```python
"""
==================================================
Balance model complexity and cross-validated score
==================================================

This example balances model complexity and cross-validated score by
finding a decent accuracy within 1 standard deviation of the best accuracy
score while minimising the number of PCA components [1].

The figure shows the trade-off between cross-validated score and the number
of PCA components. The balanced case is when n_components=6 and accuracy=0.80,
which falls into the range within 1 standard deviation of the best accuracy
score.

[1] Hastie, T., Tibshirani, R.,, Friedman, J. (2001). Model Assessment and
Selection. The Elements of Statistical Learning (pp. 219-260). New York,
NY, USA: Springer New York Inc..
"""
# Author: Wenhao Zhang <wenhaoz@ucla.edu>

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def lower_bound(cv_results):
    """
    Calculate the lower bound within 1 standard deviation
    of the best `mean_test_scores`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`

    Returns
    -------
    float
        Lower bound within 1 standard deviation of the
        best `mean_test_score`.
    """
    best_score_idx = np.argmax(cv_results['mean_test_score'])

    return (cv_results['mean_test_score'][best_score_idx]
            - cv_results['std_test_score'][best_score_idx])


def best_low_complexity(cv_results):
    """
    Balance model complexity with cross-validated score.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.

    Return
    ------
    int
        Index of a model that has the fewest PCA components
        while has its test score within 1 standard deviation of the best
        `mean_test_score`.
    """
    threshold = lower_bound(cv_results)
    candidate_idx = np.flatnonzero(cv_results['mean_test_score'] >= threshold)
    best_idx = candidate_idx[cv_results['param_reduce_dim__n_components']
                             [candidate_idx].argmin()]
    return best_idx


pipe = Pipeline([
        ('reduce_dim', PCA(random_state=42)),
        ('classify', LinearSVC(random_state=42)),
])

param_grid = {
    'reduce_dim__n_components': [2, 4, 6, 8]
}

grid = GridSearchCV(pipe, cv=10, n_jobs=1, param_grid=param_grid,
                    scoring='accuracy', refit=best_low_complexity)
digits = load_digits()
grid.fit(digits.data, digits.target)

n_components = grid.cv_results_['param_reduce_dim__n_components']
test_scores = grid.cv_results_['mean_test_score']

plt.figure()
plt.bar(n_components, test_scores, width=1.3, color='b')

lower = lower_bound(grid.cv_results_)
plt.axhline(np.max(test_scores), linestyle='--', color='y',
            label='Best score')
plt.axhline(lower, linestyle='--', color='.5', label='Best score - 1 std')

plt.title("Balance model complexity and cross-validated score")
plt.xlabel('Number of PCA components used')
plt.ylabel('Digit classification accuracy')
plt.xticks(n_components.tolist())
plt.ylim((0, 1.0))
plt.legend(loc='upper left')

best_index_ = grid.best_index_

print("The best_index_ is %d" % best_index_)
print("The n_components selected is %d" % n_components[best_index_])
print("The corresponding accuracy score is %.2f"
      % grid.cv_results_['mean_test_score'][best_index_])
plt.show()

```
16 - /tmp/repos/scikit-learn/examples/text/plot_document_classification_20newsgroups.py
```python



results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50, tol=1e-3),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty,
                                           max_iter=5)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet",
                                       max_iter=5)))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

```
17 - /tmp/repos/scikit-learn/examples/model_selection/plot_grid_search_digits.py
```python
"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
```
18 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
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
19 - /tmp/repos/scikit-learn/examples/compose/column_transformer_mixed_types.py
```python
"""
===================================
Column Transformer with Mixed Types
===================================

This example illustrates how to apply different preprocessing and
feature extraction pipelines to different subsets of features,
using :class:`sklearn.compose.ColumnTransformer`.
This is particularly handy for the case of datasets that contain
heterogeneous data types, since we may want to scale the
numeric features and one-hot encode the categorical ones.

In this example, the numeric data is standard-scaled after
mean-imputation, while the categorical data is one-hot
encoded after imputing missing values with a new category
(``'missing'``).

Finally, the preprocessing pipeline is integrated in a
full prediction pipeline using :class:`sklearn.pipeline.Pipeline`,
together with a simple classification model.
"""

# Author: Pedro Morales <part.morales@gmail.com>
#
# License: BSD 3 clause

from __future__ import print_function

import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, CategoricalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


# Read data from Titanic dataset.
titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)

# We will train our classifier with the following features:
# Numeric Features:
# - age: float.
# - fare: float.
# Categorical Features:
# - embarked: categories encoded as strings {'C', 'S', 'Q'}.
# - sex: categories encoded as strings {'female', 'male'}.
# - pclass: ordinal integers {1, 2, 3}.
numeric_features = ['age', 'fare']
categorical_features = ['embarked', 'sex', 'pclass']

# Provisionally, use pd.fillna() to impute missing values for categorical
# features; SimpleImputer will eventually support strategy="constant".
data[categorical_features] = data[categorical_features].fillna(value='missing')

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
categorical_transformer = CategoricalEncoder('onehot-dense',
                                             handle_unknown='ignore')

preprocessing_pl = make_column_transformer(
    (numeric_features, numeric_transformer),
    (categorical_features, categorical_transformer),
    remainder='drop'
)

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = make_pipeline(preprocessing_pl, LogisticRegression())

X = data.drop('survived', axis=1)
y = data.survived.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True)

clf.fit(X_train, y_train)
print("model score: %f" % clf.score(X_test, y_test))


###############################################################################
# Using the prediction pipeline in a grid search
###############################################################################
# Grid search can also be performed on the different preprocessing steps
# defined in the ``ColumnTransformer`` object, together with the classifier's
# hyperparameters as part of the ``Pipeline``.
# We will search for both the imputer strategy of the numeric preprocessing
# and the regularization parameter of the logistic regression using
# :class:`sklearn.model_selection.GridSearchCV`.


param_grid = {
    'columntransformer__pipeline__simpleimputer__strategy': ['mean', 'median'],
    'logisticregression__C': [0.1, 1.0, 1.0],
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %f"
       % grid_search.score(X_test, y_test)))
```
20 - /tmp/repos/scikit-learn/benchmarks/bench_covertype.py
```python
ESTIMATORS = {
    'GBRT': GradientBoostingClassifier(n_estimators=250),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=20),
    'RandomForest': RandomForestClassifier(n_estimators=20),
    'CART': DecisionTreeClassifier(min_samples_split=5),
    'SGD': SGDClassifier(alpha=0.001, max_iter=1000, tol=1e-3),
    'GaussianNB': GaussianNB(),
    'liblinear': LinearSVC(loss="l2", penalty="l2", C=1000, dual=False,
                           tol=1e-3),
    'SAG': LogisticRegression(solver='sag', max_iter=2, C=1000)
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifiers', nargs="+",
                        choices=ESTIMATORS, type=str,
                        default=['liblinear', 'GaussianNB', 'SGD', 'CART'],
                        help="list of classifiers to benchmark.")
    parser.add_argument('--n-jobs', nargs="?", default=1, type=int,
                        help="Number of concurrently running workers for "
                             "models that support parallelism.")
    parser.add_argument('--order', nargs="?", default="C", type=str,
                        choices=["F", "C"],
                        help="Allow to choose between fortran and C ordered "
                             "data")
    parser.add_argument('--random-seed', nargs="?", default=13, type=int,
                        help="Common seed used by random number generator.")
    args = vars(parser.parse_args())

    print(__doc__)

    X_train, X_test, y_train, y_test = load_data(
        order=args["order"], random_state=args["random_seed"])

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of train samples:".ljust(25),
             X_train.shape[0], np.sum(y_train == 1),
             np.sum(y_train == 0), int(X_train.nbytes / 1e6)))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of test samples:".ljust(25),
             X_test.shape[0], np.sum(y_test == 1),
             np.sum(y_test == 0), int(X_test.nbytes / 1e6)))

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
    print("%s %s %s %s"
          % ("Classifier  ", "train-time", "test-time", "error-rate"))
    print("-" * 44)
    for name in sorted(args["classifiers"], key=error.get):
        print("%s %s %s %s" % (name.ljust(12),
                               ("%.4fs" % train_time[name]).center(10),
                               ("%.4fs" % test_time[name]).center(10),
                               ("%.4f" % error[name]).center(10)))

    print()

```
21 - /tmp/repos/scikit-learn/sklearn/grid_search.py
```python
def fit_grid_point(X, y, estimator, parameters, train, test, scorer,
                   verbose, error_score='raise', **fit_params):
    """Run fit on one set of parameters.

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :func:`sklearn.model_selection.fit_grid_point` instead.

    Parameters
    ----------
    X : array-like, sparse matrix or list
        Input data.

    y : array-like or None
        Targets for input data.

    estimator : estimator object
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    parameters : dict
        Parameters to be set on estimator for this grid point.

    train : ndarray, dtype int or bool
        Boolean mask or indices for training set.

    test : ndarray, dtype int or bool
        Boolean mask or indices for test set.

    scorer : callable or None.
        If provided must be a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int
        Verbosity level.

    **fit_params : kwargs
        Additional parameter passed to the fit function of the estimator.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Returns
    -------
    score : float
        Score of this parameter setting on given training / test split.

    parameters : dict
        The parameters that have been evaluated.

    n_samples_test : int
        Number of test samples in this split.
    """
    score, n_samples_test, _ = _fit_and_score(estimator, X, y, scorer, train,
                                              test, verbose, parameters,
                                              fit_params, error_score)
    return score, parameters, n_samples_test


def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            check = [isinstance(v, k) for k in (list, tuple, np.ndarray)]
            if True not in check:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))


class _CVScoreTuple (namedtuple('_CVScoreTuple',
                                ('parameters',
                                 'mean_validation_score',
                                 'cv_validation_scores'))):
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __repr__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __repr__(self):
        """Simple custom repr to summarize the main info"""
        return "mean: {0:.5f}, std: {1:.5f}, params: {2}".format(
            self.mean_validation_score,
            np.std(self.cv_validation_scores),
            self.parameters)


class BaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    
```
22 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    assert_raises(ValueError, estimator.fit, X, y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_general(name, transformer):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    _check_transformer(name, transformer, X, y)
    _check_transformer(name, transformer, X.tolist(), y.tolist())


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_data_not_an_array(name, transformer):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - .1
    this_X = NotAnArray(X)
    this_y = NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformers_unfitted(name, transformer):
    X, y = _boston_subset()

    transformer = clone(transformer)
    with assert_raises((AttributeError, ValueError), msg="The unfitted "
                       "transformer {} does not raise an error when "
                       "transform is called. Perhaps use "
                       "check_is_fitted in transform.".format(name)):
        transformer.transform(X)



```
23 - /tmp/repos/scikit-learn/sklearn/inspection/partial_dependence.py
```python


    # convert features into a seq of int tuples
    tmp_features = []
    for fxs in features:
        if isinstance(fxs, (numbers.Integral, str)):
            fxs = (fxs,)
        try:
            fxs = [convert_feature(fx) for fx in fxs]
        except TypeError:
            raise ValueError('Each entry in features must be either an int, '
                             'a string, or an iterable of size at most 2.')
        if not (1 <= np.size(fxs) <= 2):
            raise ValueError('Each entry in features must be either an int, '
                             'a string, or an iterable of size at most 2.')

        tmp_features.append(fxs)

    features = tmp_features

    names = []
    try:
        for fxs in features:
            names_ = []
            # explicit loop so "i" is bound for exception below
            for i in fxs:
                names_.append(feature_names[i])
            names.append(names_)
    except IndexError:
        raise ValueError('All entries of features must be less than '
                         'len(feature_names) = {0}, got {1}.'
                         .format(len(feature_names), i))

    # compute averaged predictions
    pd_result = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(partial_dependence)(estimator, X, fxs,
                                    response_method=response_method,
                                    method=method,
                                    grid_resolution=grid_resolution,
                                    percentiles=percentiles)
        for fxs in features)

    # For multioutput regression, we can only check the validity of target
    # now that we have the predictions.
    # Also note: as multiclass-multioutput classifiers are not supported,
    # multiclass and multioutput scenario are mutually exclusive. So there is
    # no risk of overwriting target_idx here.
    avg_preds, _ = pd_result[0]  # checking the first result is enough
    if is_regressor(estimator) and avg_preds.shape[0] > 1:
        if target is None:
            raise ValueError(
                'target must be specified for multi-output regressors')
        if not 0 <= target <= avg_preds.shape[0]:
            raise ValueError(
                'target must be in [0, n_tasks], got {}.'.format(target))
        target_idx = target
    else:
        target_idx = 0

    # get global min and max values of PD grouped by plot type
    pdp_lim = {}
    for avg_preds, values in pd_result:
        min_pd = avg_preds[target_idx].min()
        max_pd = avg_preds[target_idx].max()
        n_fx = len(values)
        old_min_pd, old_max_pd = pdp_lim.get(n_fx, (min_pd, max_pd))
        min_pd = min(min_pd, old_min_pd)
        max_pd = max(max_pd, old_max_pd)
        pdp_lim[n_fx] = (min_pd, max_pd)

    # create contour levels for two-way plots
    if 2 in pdp_lim:
        Z_level = np.linspace(*pdp_lim[2], num=8)

    if fig is None:
        fig = plt.figure()
    else:
        fig.clear()

    if line_kw is None:
        line_kw = {'color': 'green'}
    if contour_kw is None:
        contour_kw = {}

    n_cols = min(n_cols, len(features))
    n_rows = int(np.ceil(len(features) / float(n_cols)))
    axs = []
    
```
24 - /tmp/repos/scikit-learn/examples/impute/plot_missing_values.py
```python
"""
====================================================
Imputing missing values before building an estimator
====================================================

Missing values can be replaced by the mean, the median or the most frequent
value using the basic :class:`sklearn.impute.SimpleImputer`.
The median is a more robust estimator for data with high magnitude variables
which could dominate results (otherwise known as a 'long tail').

Another option is the :class:`sklearn.impute.IterativeImputer`. This uses
round-robin linear regression, treating every variable as an output in
turn. The version implemented assumes Gaussian (output) variables. If your
features are obviously non-Normal, consider transforming them to look more
Normal so as to potentially improve performance.

In addition of using an imputing method, we can also keep an indication of the
missing information using :func:`sklearn.impute.MissingIndicator` which might
carry some information.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

N_SPLITS = 5
REGRESSOR = RandomForestRegressor(random_state=0)


def get_scores_for_imputer(imputer, X_missing, y_missing):
    estimator = make_pipeline(
        make_union(imputer, MissingIndicator(missing_values=0)),
        REGRESSOR)
    impute_scores = cross_val_score(estimator, X_missing, y_missing,
                                    scoring='neg_mean_squared_error',
                                    cv=N_SPLITS)
    return impute_scores


def get_results(dataset):
    X_full, y_full = dataset.data, dataset.target
    n_samples = X_full.shape[0]
    n_features = X_full.shape[1]

    # Estimate the score on the entire dataset, with no missing values
    full_scores = cross_val_score(REGRESSOR, X_full, y_full,
                                  scoring='neg_mean_squared_error',
                                  cv=N_SPLITS)

    # Add missing values in 75% of the lines
    missing_rate = 0.75
    n_missing_samples = int(np.floor(n_samples * missing_rate))
    missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                          dtype=np.bool),
                                 np.ones(n_missing_samples,
                                         dtype=np.bool)))
    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[np.where(missing_samples)[0], missing_features] = 0
    y_missing = y_full.copy()

    # Estimate the score after replacing missing values by 0
    imputer = SimpleImputer(missing_values=0,
                            strategy='constant',
                            fill_value=0)
    zero_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)

    # Estimate the score after imputation (mean strategy) of the missing values
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    mean_impute_scores = get_scores_for_imputer(imputer, X_missing, y_missing)

    # Estimate the score after iterative imputation of the missing values
    imputer = IterativeImputer(missing_values=0,
                               random_state=0,
                               n_nearest_features=5)
    iterative_impute_scores = get_scores_for_imputer(imputer,
                                                     X_missing,
                                                     y_missing)

    return ((full_scores.mean(), full_scores.std()),
            (zero_impute_scores.mean(), zero_impute_scores.std()),
            (mean_impute_scores.mean(), mean_impute_scores.std()),
            (iterative_impute_scores.mean(), iterative_impute_scores.std()))
```
25 - /tmp/repos/scikit-learn/sklearn/grid_search.py
```python
def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        pre_dispatch = self.pre_dispatch

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(
            delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                    train, test, self.verbose, parameters,
                                    self.fit_params, return_parameters=True,
                                    error_score=self.error_score)
                for parameters in parameter_iterable
                for train, test in cv)

        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, _, parameters in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :class:`sklearn.model_selection.GridSearchCV` instead.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

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

        .. versionchanged:: 0.17
           Upgraded to joblib 0.9.3.

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
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.


    Examples
    --------
    >>> from sklearn import svm, grid_search, datasets
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = grid_search.GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape='ovr', degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params={}, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=...,
           scoring=..., verbose=...)


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

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.

    :func:`sklearn.cross_validation.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    
```
26 - /tmp/repos/scikit-learn/examples/impute/plot_iterative_imputer_variants_comparison.py
```python
for impute_estimator in estimators:
    estimator = make_pipeline(
        IterativeImputer(random_state=0, estimator=impute_estimator),
        br_estimator
    )
    score_iterative_imputer[impute_estimator.__class__.__name__] = \
        cross_val_score(
            estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
            cv=N_SPLITS
        )

scores = pd.concat(
    [score_full_data, score_simple_imputer, score_iterative_imputer],
    keys=['Original', 'SimpleImputer', 'IterativeImputer'], axis=1
)

# plot boston results
fig, ax = plt.subplots(figsize=(13, 6))
means = -scores.mean()
errors = scores.std()
means.plot.barh(xerr=errors, ax=ax)
ax.set_title('California Housing Regression with Different Imputation Methods')
ax.set_xlabel('MSE (smaller is better)')
ax.set_yticks(np.arange(means.shape[0]))
ax.set_yticklabels([" w/ ".join(label) for label in means.index.get_values()])
plt.tight_layout(pad=1)
plt.show()

```
27 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
28 - /tmp/repos/scikit-learn/examples/model_selection/grid_search_text_feature_extraction.py
```python
if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

```
29 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_empty_data_messages(name, estimator_orig):
    e = clone(estimator_orig)
    set_random_state(e, 1)

    X_zero_samples = np.empty(0).reshape(0, 3)
    # The precise message can change depending on whether X or y is
    # validated first. Let us test the type of exception only:
    with assert_raises(ValueError, msg="The estimator {} does not"
                       " raise an error when an empty data is used "
                       "to train. Perhaps use "
                       "check_array in train.".format(name)):
        e.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape(3, 0)
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = multioutput_estimator_convert_y_2d(e, np.array([1, 0, 1]))
    msg = (r"0 feature\(s\) \(shape=\(3, 0\)\) while a minimum of \d* "
           "is required.")
    assert_raises_regex(ValueError, msg, e.fit, X_zero_features, y)


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
30 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python


        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        iid = self.iid
        if self.iid == 'warn':
            if len(np.unique(test_sample_counts)) > 1:
                warnings.warn("The default of the `iid` parameter will change "
                              "from True to False in version 0.22 and will be"
                              " removed in 0.24. This will change numeric"
                              " results when test-set sizes are unequal.",
                              DeprecationWarning)
            iid = True

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if iid else None)
            if self.return_train_score:
                prev_keys = set(results.keys())
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

                if self.return_train_score == 'warn':
                    for key in set(results.keys()) - prev_keys:
                        message = (
                            'You are accessing a training score ({!r}), '
                            'which will not be available by default '
                            'any more in 0.21. If you need training scores, '
                            'please set return_train_score=True').format(key)
                        # warn on key access
                        results.add_warning(key, message, FutureWarning)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = candidate_params[self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @property
    def grid_scores_(self):
        check_is_fitted(self, 'cv_results_')
        if self.multimetric_:
            raise AttributeError("grid_scores_ attribute is not available for"
                                 " multi-metric evaluation.")
        warnings.warn(
            "The grid_scores_ attribute was deprecated in version 0.18"
            " in favor of the more elaborate cv_results_ attribute."
            " The grid_scores_ attribute will not be available from 0.20",
            DeprecationWarning)

        grid_scores = list()

        for i, (params, mean, std) in enumerate(zip(
                self.cv_results_['params'],
                self.cv_results_['mean_test_score'],
                self.cv_results_['std_test_score'])):
            scores = np.array(list(self.cv_results_['split%d_test_score'
                                                    % s][i]
                                   for s in range(self.n_splits_)),
                              dtype=np.float64)
            grid_scores.append(_CVScoreTuple(params, mean, scores))

        return grid_scores


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

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

    refit : boolean, or string, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer is used to find the best parameters for refitting the estimator
        at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

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


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape='ovr', degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params=None, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
           scoring=..., verbose=...)
    >>> sorted(clf.cv_results_.keys())
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
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

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.

    :func:`sklearn.model_selection.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    
```
31 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
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
32 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
```python
def one_run(n_samples):
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
    if args.catboost:
        print("Fitting a CatBoost model...")
        # get_cat does not accept loss='auto'
        if args.problem == 'classification':
            loss = 'binary_crossentropy' if args.n_classes == 2 else \
                'categorical_crossentropy'
            est.set_params(loss=loss)
        cat_est = get_equivalent_estimator(est, lib='catboost')

        tic = time()
        cat_est.fit(X_train, y_train)
        cat_fit_duration = time() - tic
        tic = time()
        cat_score = cat_est.score(X_test, y_test)
        cat_score_duration = time() - tic
        print("score: {:.4f}".format(cat_score))
        print("fit duration: {:.3f}s,".format(cat_fit_duration))
        print("score duration: {:.3f}s,".format(cat_score_duration))

    return (sklearn_score, sklearn_fit_duration, sklearn_score_duration,
            lightgbm_score, lightgbm_fit_duration, lightgbm_score_duration,
            xgb_score, xgb_fit_duration, xgb_score_duration,
            cat_score, cat_fit_duration, cat_score_duration)
```
33 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def plot_results(results_df, plot_name):
    if results_df is None:
        return None

    plt.figure(figsize=(16, 6))
    colors = 'bgr'
    markers = 'ovs'
    ax = plt.subplot(1, 3, 1)
    for i, init in enumerate(np.unique(results_df['init'])):
        plt.subplot(1, 3, i + 1, sharex=ax, sharey=ax)
        for j, method in enumerate(np.unique(results_df['method'])):
            mask = np.logical_and(results_df['init'] == init,
                                  results_df['method'] == method)
            selected_items = results_df[mask]

            plt.plot(selected_items['time'], selected_items['loss'],
                     color=colors[j % len(colors)], ls='-',
                     marker=markers[j % len(markers)],
                     label=method)

        plt.legend(loc=0, fontsize='x-small')
        plt.xlabel("Time (s)")
        plt.ylabel("loss")
        plt.title("%s" % init)
    plt.suptitle(plot_name, fontsize=16)


@ignore_warnings(category=ConvergenceWarning)
# use joblib to cache the results.
# X_shape is specified in arguments for avoiding hashing X
@mem.cache(ignore=['X', 'W0', 'H0'])
def bench_one(name, X, W0, H0, X_shape, clf_type, clf_params, init,
              n_components, random_state):
    W = W0.copy()
    H = H0.copy()

    clf = clf_type(**clf_params)
    st = time()
    W = clf.fit_transform(X, W=W, H=H)
    end = time()
    H = clf.components_

    this_loss = _beta_divergence(X, W, H, 2.0, True)
    duration = end - st
    return this_loss, duration


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


def load_20news():
    print("Loading 20 newsgroups dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(dataset.data)
    return tfidf


def load_faces():
    print("Loading Olivetti face dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=True)
    return faces.data



```
34 - /tmp/repos/scikit-learn/sklearn/grid_search.py
```python
"""
The :mod:`sklearn.grid_search` includes utilities to fine-tune the parameters
of an estimator.
"""
from __future__ import print_function

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
from collections import Mapping, namedtuple, Sized
from functools import partial, reduce
from itertools import product
import operator
import warnings

import numpy as np

from .base import BaseEstimator, is_classifier, clone
from .base import MetaEstimatorMixin
from .cross_validation import check_cv
from .cross_validation import _fit_and_score
from .externals.joblib import Parallel, delayed
from .externals import six
from .utils import check_random_state
from .utils.random import sample_without_replacement
from .utils.validation import _num_samples, indexable
from .utils.metaestimators import if_delegate_has_method
from .metrics.scorer import check_scoring


__all__ = ['GridSearchCV', 'ParameterGrid', 'fit_grid_point',
           'ParameterSampler', 'RandomizedSearchCV']


warnings.warn("This module was deprecated in version 0.18 in favor of the "
              "model_selection module into which all the refactored classes "
              "and functions are moved. This module will be removed in 0.20.",
              DeprecationWarning)



```
35 - /tmp/repos/scikit-learn/examples/model_selection/plot_precision_recall.py
```python
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,
                                                    random_state=random_state)

# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)

###############################################################################
# Compute the average precision score
# ...................................
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

###############################################################################
# Plot the Precision-Recall curve
# ................................
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.externals.funcsigs import signature

precision, recall, _ = precision_recall_curve(y_test, y_score)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

###############################################################################
# In multi-label settings
# ------------------------
#
# Create multi-label data, fit, and predict
# ...........................................
#
# We create a multi-label dataset, to illustrate the precision-recall in
# multi-label settings

from sklearn.preprocessing import label_binarize

# Use label_binarize to be multi-label like settings
Y = label_binarize(y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)


###############################################################################
# The average precision score in multi-label settings
# ....................................................
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

###############################################################################
# Plot the micro-averaged Precision-Recall curve
# ...............................................
#

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

###############################################################################
# Plot Precision-Recall curve for each class and iso-f1 curves
# .............................................................
#
from itertools import cycle
# setup plot details

```
36 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig):
    clusterer = clone(clusterer_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    n_samples, n_features = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)
    if name == 'AffinityPropagation':
        clusterer.set_params(preference=-100)
        clusterer.set_params(max_iter=100)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert_equal(pred.shape, (n_samples,))
    assert_greater(adjusted_rand_score(pred, y), 0.4)
    # fit another time with ``fit_predict`` and compare results
    if name == 'SpectralClustering':
        # there is no way to make Spectral clustering deterministic :(
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert_in(pred.dtype, [np.dtype('int32'), np.dtype('int64')])
    assert_in(pred2.dtype, [np.dtype('int32'), np.dtype('int64')])

    # Add noise to X to test the possible values of the labels
    rng = np.random.RandomState(7)
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(labels_sorted, np.arange(labels_sorted[0],
                                                labels_sorted[-1] + 1))

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert_true(labels_sorted[0] in [0, -1])
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, 'n_clusters'):
        n_clusters = getattr(clusterer, 'n_clusters')
        assert_greater_equal(n_clusters - 1, labels_sorted[-1])
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=DeprecationWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels"""
    X, y = make_blobs(n_samples=20, random_state=0)
    clusterer = clone(clusterer_orig)

    if hasattr(clusterer, "compute_labels"):
        # MiniBatchKMeans
        if hasattr(clusterer, "random_state"):
            clusterer.set_params(random_state=0)

        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)


@ignore_warnings(category=DeprecationWarning)
def check_classifiers_one_label(name, classifier_orig):
    error_string_fit = "Classifier can't train when only one class is present."
    error_string_predict = ("Classifier can't predict when only one class is "
                            "present.")
    rnd = np.random.RandomState(0)
    X_train = rnd.uniform(size=(10, 3))
    X_test = rnd.uniform(size=(10, 3))
    y = np.ones(10)
    # catch deprecation warnings
    with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
        classifier = clone(classifier_orig)
        # try to fit
        try:
            classifier.fit(X_train, y)
        except ValueError as e:
            if 'class' not in repr(e):
                print(error_string_fit, classifier, e)
                traceback.print_exc(file=sys.stdout)
                raise e
            else:
                return
        except Exception as exc:
            print(error_string_fit, classifier, exc)
            traceback.print_exc(file=sys.stdout)
            raise exc
        # predict
        try:
            assert_array_equal(classifier.predict(X_test), y)
        except Exception as exc:
            print(error_string_predict, classifier, exc)
            raise exc


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    
```
37 - /tmp/repos/scikit-learn/examples/plot_missing_values.py
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
38 - /tmp/repos/scikit-learn/examples/compose/plot_digits_pipe.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Pipelining: chaining a PCA and a logistic regression
=========================================================

The PCA does an unsupervised dimensionality reduction, while the logistic
regression does the prediction.

We use a GridSearchCV to set the dimensionality of the PCA

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
                         max_iter=10000, tol=1e-5, random_state=0)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [5, 20, 30, 40, 50, 64],
    'logistic__alpha': np.logspace(-4, 4, 5),
}
search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Plot the PCA spectrum
pca.fit(X_digits)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.tight_layout()
plt.show()
```
39 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [2., 5., 4.]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                               random_state=0, n_features=2, cluster_std=0.1)
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert_greater_equal(iter_, 1)
        else:
            assert_greater_equal(estimator.n_iter_, 1)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    class T(BaseEstimator):
        """Mock classifier
        """

        def __init__(self):
            pass

        def fit(self, X, y):
            return self

        def transform(self, X):
            return X

    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert_true(all(item in deep_params.items() for item in
                    shallow_params.items()))


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    boston = load_boston()
    X, y = boston.data, boston.target
    e = clone(estimator_orig)
    msg = 'Unknown label type: '
    assert_raises_regex(ValueError, msg, e.fit, X, y)



```
40 - /tmp/repos/scikit-learn/examples/preprocessing/plot_all_scaling.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================
Compare the effect of different scalers on data with outliers
=============================================================

Feature 0 (median income in a block) and feature 5 (number of households) of
the `California housing dataset
<http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>`_ have very
different scales and contain some very large outliers. These two
characteristics lead to difficulties to visualize the data and, more
importantly, they can degrade the predictive performance of many machine
learning algorithms. Unscaled data can also slow down or even prevent the
convergence of many gradient-based estimators.

Indeed many estimators are designed with the assumption that each feature takes
values close to zero or more importantly that all features vary on comparable
scales. In particular, metric-based and gradient-based estimators often assume
approximately standardized data (centered features with unit variances). A
notable exception are decision tree-based estimators that are robust to
arbitrary scaling of the data.

This example uses different scalers, transformers, and normalizers to bring the
data within a pre-defined range.

Scalers are linear (or more precisely affine) transformers and differ from each
other in the way to estimate the parameters used to shift and scale each
feature.

``QuantileTransformer`` provides non-linear transformations in which distances
between marginal outliers and inliers are shrunk. ``PowerTransformer`` provides
non-linear transformations in which data is mapped to a normal distribution to
stabilize variance and minimize skewness.

Unlike the previous transformations, normalization refers to a per sample
transformation instead of a per feature transformation.

The following code is a bit verbose, feel free to jump directly to the analysis
of the results_.

"""

# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause

from __future__ import print_function

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.datasets import fetch_california_housing

print(__doc__)

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target

# Take only 2 features to make visualization easier
# Feature of 0 has a long tail distribution.
# Feature 5 has a few but very large outliers.

X = X_full[:, [0, 5]]

distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer(method='box-cox').fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(X)),
]

# scale the output between 0 and 1 for the colorbar
y = minmax_scale(y_full)

# plasma does not exist in matplotlib < 1.5
cmap = getattr(cm, 'plasma_r', cm.hot_r)


```
41 - /tmp/repos/scikit-learn/examples/model_selection/plot_multi_metric_evaluation.py
```python
"""
============================================================================
Demonstration of multi-metric evaluation on cross_val_score and GridSearchCV
============================================================================

Multiple metric parameter search can be done by setting the ``scoring``
parameter to a list of metric scorer names or a dict mapping the scorer names
to the scorer callables.

The scores of all the scorers are available in the ``cv_results_`` dict at keys
ending in ``'_<scorer_name>'`` (``'mean_test_precision'``,
``'rank_test_precision'``, etc...)

The ``best_estimator_``, ``best_index_``, ``best_score_`` and ``best_params_``
correspond to the scorer (key) that is set to the ``refit`` attribute.
"""

# Author: Raghav RV <rvraghav93@gmail.com>
# License: BSD

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

print(__doc__)

###############################################################################
# Running ``GridSearchCV`` using multiple evaluation metrics
# ----------------------------------------------------------
#

X, y = make_hastie_10_2(n_samples=8000, random_state=42)

# The scorers can be either be one of the predefined metric strings or a scorer
# callable, like the one returned by make_scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# Setting refit='AUC', refits an estimator on the whole dataset with the
# parameter setting that has the best cross-validated AUC score.
# That estimator is made available at ``gs.best_estimator_`` along with
# parameters like ``gs.best_score_``, ``gs.best_parameters_`` and
# ``gs.best_index_``
gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid={'min_samples_split': range(2, 403, 10)},
                  scoring=scoring, cv=5, refit='AUC')
gs.fit(X, y)
results = gs.cv_results_

###############################################################################
# Plotting the result
# -------------------

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, 402)
ax.set_ylim(0.73, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()
```
42 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
43 - /tmp/repos/scikit-learn/examples/model_selection/plot_cv_indices.py
```python
"""
Visualizing cross-validation behavior in scikit-learn
=====================================================

Choosing the right cross-validation object is a crucial part of fitting a
model properly. There are many ways to split data into training and test
sets in order to avoid model overfitting, to standardize the number of
groups in test sets, etc.

This example visualizes the behavior of several common scikit-learn objects
for comparison.
"""

from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4

###############################################################################
# Visualize our data
# ------------------
#
# First, we must understand the structure of our data. It has 100 randomly
# generated input datapoints, 3 classes split unevenly across datapoints,
# and 10 "groups" split evenly across datapoints.
#
# As we'll see, some cross-validation objects do specific things with
# labeled data, others behave differently with grouped data, and others
# do not use this information.
#
# To begin, we'll visualize our data.

# Generate the class/group data
n_points = 100
X = np.random.randn(100, 10)

percentiles_classes = [.1, .3, .6]
y = np.hstack([[ii] * int(100 * perc)
               for ii, perc in enumerate(percentiles_classes)])

# Evenly spaced groups repeated once
groups = np.hstack([[ii] * 10 for ii in range(10)])


def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")


visualize_groups(y, groups, 'no groups')

###############################################################################
# Define a function to visualize cross-validation behavior
# --------------------------------------------------------
#
# We'll define a function that lets us visualize the behavior of each
# cross-validation object. We'll perform 4 splits of the data. On each
# split, we'll visualize the indices chosen for the training set
# (in blue) and the test set (in red).


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
```
44 - /tmp/repos/scikit-learn/examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py
```python
"""
==============================================
Feature agglomeration vs. univariate selection
==============================================

This example compares 2 dimensionality reduction strategies:

- univariate feature selection with Anova

- feature agglomeration with Ward hierarchical clustering

Both methods are compared in a regression problem using
a BayesianRidge as supervised estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

print(__doc__)

import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# #############################################################################
# Generate data
n_samples = 200
size = 40  # image size
roi_size = 15
snr = 5.
np.random.seed(0)
mask = np.ones([size, size], dtype=np.bool)

coef = np.zeros((size, size))
coef[0:roi_size, 0:roi_size] = -1.
coef[-roi_size:, -roi_size:] = 1.

X = np.random.randn(n_samples, size ** 2)
for x in X:  # smooth data
    x[:] = ndimage.gaussian_filter(x.reshape(size, size), sigma=1.0).ravel()
X -= X.mean(axis=0)
X /= X.std(axis=0)

y = np.dot(X, coef.ravel())
noise = np.random.randn(y.shape[0])
noise_coef = (linalg.norm(y, 2) / np.exp(snr / 20.)) / linalg.norm(noise, 2)
y += noise_coef * noise  # add noise

# #############################################################################
# Compute the coefs of a Bayesian Ridge with GridSearch
cv = KFold(2)  # cross-validation generator for model selection
ridge = BayesianRidge()
cachedir = tempfile.mkdtemp()
mem = Memory(cachedir=cachedir, verbose=1)

# Ward agglomeration followed by BayesianRidge
connectivity = grid_to_graph(n_x=size, n_y=size)
ward = FeatureAgglomeration(n_clusters=10, connectivity=connectivity,
                            memory=mem)
clf = Pipeline([('ward', ward), ('ridge', ridge)])
# Select the optimal number of parcels with grid search
clf = GridSearchCV(clf, {'ward__n_clusters': [10, 20, 30]}, n_jobs=1, cv=cv)
clf.fit(X, y)  # set the best parameters
coef_ = clf.best_estimator_.steps[-1][1].coef_
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_)
coef_agglomeration_ = coef_.reshape(size, size)

# Anova univariate feature selection followed by BayesianRidge
f_regression = mem.cache(feature_selection.f_regression)  # caching function
anova = feature_selection.SelectPercentile(f_regression)
clf = Pipeline([('anova', anova), ('ridge', ridge)])
# Select the optimal percentage of features with grid search
clf = GridSearchCV(clf, {'anova__percentile': [5, 10, 20]}, cv=cv)
clf.fit(X, y)  # set the best parameters
coef_ = clf.best_estimator_.steps[-1][1].coef_
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_.reshape(1, -1))
coef_selection_ = coef_.reshape(size, size)

# #############################################################################
# Inverse the transformation to plot the results on an image
plt.close('all')
plt.figure(figsize=(7.3, 2.7))
plt.subplot(1, 3, 1)
plt.imshow(coef, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("True weights")
plt.subplot(1, 3, 2)
plt.imshow(coef_selection_, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("Feature Selection")
plt.subplot(1, 3, 3)
plt.imshow(coef_agglomeration_, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("Feature Agglomeration")
plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.16, 0.26)
plt.show()

# Attempt to remove the temporary cachedir, but don't worry if it fails
shutil.rmtree(cachedir, ignore_errors=True)
```
45 - /tmp/repos/scikit-learn/benchmarks/bench_20newsgroups.py
```python
from __future__ import print_function, division
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
46 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_no_attributes_set_in_init(name, estimator):
    """Check setting during init. """

    if hasattr(type(estimator).__init__, "deprecated_original"):
        return

    init_params = _get_args(type(estimator).__init__)
    parents_init_params = [param for params_parent in
                           (_get_args(parent) for parent in
                            type(estimator).__mro__)
                           for param in params_parent]

    # Test for no setting apart from parameters during init
    invalid_attr = (set(vars(estimator)) - set(init_params)
                    - set(parents_init_params))
    assert_false(invalid_attr,
                 "Estimator %s should not set any attribute apart"
                 " from parameters during init. Found attributes %s."
                 % (name, sorted(invalid_attr)))
    # Ensure that each parameter is set in init
    invalid_attr = (set(init_params) - set(vars(estimator))
                    - set(["self"]))
    assert_false(invalid_attr,
                 "Estimator %s should store all parameters"
                 " as an attribute during init. Did not find "
                 "attributes %s." % (name, sorted(invalid_attr)))


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sparsify_coefficients(name, estimator_orig):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
                  [-1, -2], [2, 2], [-2, -2]])
    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    est = clone(estimator_orig)

    est.fit(X, y)
    pred_orig = est.predict(X)

    # test sparsify with dense inputs
    est.sparsify()
    assert_true(sparse.issparse(est.coef_))
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)

    # pickle and unpickle with sparse coef_
    est = pickle.loads(pickle.dumps(est))
    assert_true(sparse.issparse(est.coef_))
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)


@ignore_warnings(category=DeprecationWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X = np.array([[3, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 1]])
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = [1, 1, 1, 2, 2, 2]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)



```
47 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
def fit_grid_point(X, y, estimator, parameters, train, test, scorer,
                   verbose, error_score='raise', **fit_params):
    """Run fit on one set of parameters.

    Parameters
    ----------
    X : array-like, sparse matrix or list
        Input data.

    y : array-like or None
        Targets for input data.

    estimator : estimator object
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    parameters : dict
        Parameters to be set on estimator for this grid point.

    train : ndarray, dtype int or bool
        Boolean mask or indices for training set.

    test : ndarray, dtype int or bool
        Boolean mask or indices for test set.

    scorer : callable or None
        The scorer callable object / function must have its signature as
        ``scorer(estimator, X, y)``.

        If ``None`` the estimator's default scorer is used.

    verbose : int
        Verbosity level.

    **fit_params : kwargs
        Additional parameter passed to the fit function of the estimator.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Returns
    -------
    score : float
         Score of this parameter setting on given training / test split.

    parameters : dict
        The parameters that have been evaluated.

    n_samples_test : int
        Number of test samples in this split.
    """
    # NOTE we are not using the return value as the scorer by itself should be
    # validated before. We use check_scoring only to reject multimetric scorer
    check_scoring(estimator, scorer)
    scores, n_samples_test = _fit_and_score(estimator, X, y,
                                            scorer, train,
                                            test, verbose, parameters,
                                            fit_params=fit_params,
                                            return_n_test_samples=True,
                                            error_score=error_score)
    return scores, parameters, n_samples_test


def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, six.string_types) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))


# XXX Remove in 0.20
class _CVScoreTuple (namedtuple('_CVScoreTuple',
                                ('parameters',
                                 'mean_validation_score',
                                 'cv_validation_scores'))):
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __repr__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __repr__(self):
        """Simple custom repr to summarize the main info"""
        return "mean: {0:.5f}, std: {1:.5f}, params: {2}".format(
            self.mean_validation_score,
            np.std(self.cv_validation_scores),
            self.parameters)


class BaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    
```
48 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
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
49 - /tmp/repos/scikit-learn/benchmarks/bench_random_projections.py
```python
print("")

    ###########################################################################
    # Print results
    ###########################################################################
    print("Script arguments")
    print("===========================")
    arguments = vars(opts)
    print("%s \t | %s " % ("Arguments".ljust(16),
                           "Value".center(12),))
    print(25 * "-" + ("|" + "-" * 14) * 1)
    for key, value in arguments.items():
        print("%s \t | %s " % (str(key).ljust(16),
                               str(value).strip().center(12)))
    print("")

    print("Transformer performance:")
    print("===========================")
    print("Results are averaged over %s repetition(s)." % opts.n_times)
    print("")
    print("%s | %s | %s" % ("Transformer".ljust(30),
                            "fit".center(12),
                            "transform".center(12)))
    print(31 * "-" + ("|" + "-" * 14) * 2)

    for name in sorted(selected_transformers):
        print_row(name,
                  np.mean(time_fit[name]),
                  np.mean(time_transform[name]))

    print("")
    print("")

```
50 - /tmp/repos/scikit-learn/examples/model_selection/plot_nested_cross_validation_iris.py
```python
"""
=========================================
Nested versus non-nested cross-validation
=========================================

This example compares non-nested and nested cross-validation strategies on a
classifier of the iris data set. Nested cross-validation (CV) is often used to
train a model in which hyperparameters also need to be optimized. Nested CV
estimates the generalization error of the underlying model and its
(hyper)parameter search. Choosing the parameters that maximize non-nested CV
biases the model to the dataset, yielding an overly-optimistic score.

Model selection without nested CV uses the same data to tune model parameters
and evaluate model performance. Information may thus "leak" into the model
and overfit the data. The magnitude of this effect is primarily dependent on
the size of the dataset and the stability of the model. See Cawley and Talbot
[1]_ for an analysis of these issues.

To avoid this problem, nested CV effectively uses a series of
train/validation/test set splits. In the inner loop (here executed by
:class:`GridSearchCV <sklearn.model_selection.GridSearchCV>`), the score is
approximately maximized by fitting a model to each training set, and then
directly maximized in selecting (hyper)parameters over the validation set. In
the outer loop (here in :func:`cross_val_score
<sklearn.model_selection.cross_val_score>`), generalization error is estimated
by averaging test set scores over several dataset splits.

The example below uses a support vector classifier with a non-linear kernel to
build a model with optimized hyperparameters by grid search. We compare the
performance of non-nested and nested CV strategies by taking the difference
between their scores.

.. topic:: See Also:

    - :ref:`cross_validation`
    - :ref:`grid_search`

.. topic:: References:

    .. [1] `Cawley, G.C.; Talbot, N.L.C. On over-fitting in model selection and
     subsequent selection bias in performance evaluation.
     J. Mach. Learn. Res 2010,11, 2079-2107.
     <http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf>`_

"""
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np

print(__doc__)

# Number of random trials
NUM_TRIALS = 30

# Load the dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf")

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
    clf.fit(X_iris, y_iris)
    non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
    nested_scores[i] = nested_score.mean()

score_difference = non_nested_scores - nested_scores

print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
nested_line, = plt.plot(nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
          x=.5, y=1.1, fontsize="15")


```
51 - /tmp/repos/scikit-learn/examples/model_selection/plot_cv_indices.py
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
52 - /tmp/repos/scikit-learn/examples/ensemble/plot_gradient_boosting_early_stopping.py
```python
"""
===================================
Early stopping of Gradient Boosting
===================================

Gradient boosting is an ensembling technique where several weak learners
(regression trees) are combined to yield a powerful single model, in an
iterative fashion.

Early stopping support in Gradient Boosting enables us to find the least number
of iterations which is sufficient to build a model that generalizes well to
unseen data.

The concept of early stopping is simple. We specify a ``validation_fraction``
which denotes the fraction of the whole dataset that will be kept aside from
training to assess the validation loss of the model. The gradient boosting
model is trained using the training set and evaluated using the validation set.
When each additional stage of regression tree is added, the validation set is
used to score the model.  This is continued until the scores of the model in
the last ``n_iter_no_change`` stages do not improve by atleast `tol`. After
that the model is considered to have converged and further addition of stages
is "stopped early".

The number of stages of the final model is available at the attribute
``n_estimators_``.

This example illustrates how the early stopping can used in the
:class:`sklearn.ensemble.GradientBoostingClassifier` model to achieve
almost the same accuracy as compared to a model built without early stopping
using many fewer estimators. This can significantly reduce training time,
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
```
53 - /tmp/repos/scikit-learn/examples/inspection/plot_partial_dependence.py
```python
def main():
    cal_housing = fetch_california_housing()

    X, y = cal_housing.data, cal_housing.target
    names = cal_housing.feature_names

    # Center target to avoid gradient boosting init bias: gradient boosting
    # with the 'recursion' method does not account for the initial estimator
    # (here the average target, by default)
    y -= y.mean()

    print("Training MLPRegressor...")
    est = MLPRegressor(activation='logistic')
    est.fit(X, y)
    print('Computing partial dependence plots...')
    # We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
    # with the brute method.
    features = [0, 5, 1, 2]
    plot_partial_dependence(est, X, features, feature_names=names,
                            n_jobs=3, grid_resolution=50)
    fig = plt.gcf()
    fig.suptitle('Partial dependence of house value on non-location features\n'
                 'for the California housing dataset, with MLPRegressor')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    print("Training GradientBoostingRegressor...")
    est = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    est.fit(X, y)
    print('Computing partial dependence plots...')
    features = [0, 5, 1, 2, (5, 1)]
    plot_partial_dependence(est, X, features, feature_names=names,
                            n_jobs=3, grid_resolution=50)
    fig = plt.gcf()
    fig.suptitle('Partial dependence of house value on non-location features\n'
                 'for the California housing dataset, with Gradient Boosting')
    plt.subplots_adjust(top=0.9)

    print('Custom 3d plot via ``partial_dependence``')
    fig = plt.figure()

    target_feature = (1, 5)
    pdp, axes = partial_dependence(est, X, target_feature,
                                   grid_resolution=50)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of house value on median\n'
                 'age and average occupancy, with Gradient Boosting')
    plt.subplots_adjust(top=0.9)

    plt.show()


# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()

```
54 - /tmp/repos/scikit-learn/examples/text/plot_document_classification_20newsgroups.py
```python



data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
```
55 - /tmp/repos/scikit-learn/examples/model_selection/plot_learning_curve.py
```python
axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

digits = load_digits()
X, y = digits.data, digits.target


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()

```
56 - /tmp/repos/scikit-learn/examples/preprocessing/plot_function_transformer.py
```python
"""
=========================================================
Using FunctionTransformer to select columns
=========================================================

Shows how to use a function transformer in a pipeline. If you know your
dataset's first principle component is irrelevant for a classification task,
you can use the FunctionTransformer to select all but the first column of the
PCA transformed data.
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def _generate_vector(shift=0.5, noise=15):
    return np.arange(1000) + (np.random.rand(1000) - shift) * noise


def generate_dataset():
    """
    This dataset is two lines with a slope ~ 1, where one has
    a y offset of ~100
    """
    return np.vstack((
        np.vstack((
            _generate_vector(),
            _generate_vector() + 100,
        )).T,
        np.vstack((
            _generate_vector(),
            _generate_vector(),
        )).T,
    )), np.hstack((np.zeros(1000), np.ones(1000)))


def all_but_first_column(X):
    return X[:, 1:]


def drop_first_component(X, y):
    """
    Create a pipeline with PCA and the column selector and use it to
    transform the dataset.
    """
    pipeline = make_pipeline(
        PCA(), FunctionTransformer(all_but_first_column),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    return pipeline.transform(X_test), y_test


if __name__ == '__main__':
    X, y = generate_dataset()
    lw = 0
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, lw=lw)
    plt.figure()
    X_transformed, y_transformed = drop_first_component(*generate_dataset())
    plt.scatter(
        X_transformed[:, 0],
        np.zeros(len(X_transformed)),
        c=y_transformed,
        lw=lw,
        s=60
    )
    plt.show()

```
57 - /tmp/repos/scikit-learn/examples/preprocessing/plot_transformed_target.py
```python
f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

###############################################################################
# The effect of the transformer is weaker than on the synthetic data. However,
# the transform induces a decrease of the MAE.

f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

regr = RidgeCV()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

ax0.scatter(y_test, y_pred)
ax0.plot([0, 10], [0, 10], '--k')
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('Ridge regression \n without target transformation')
ax0.text(1, 9, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
ax0.set_xlim([0, 10])
ax0.set_ylim([0, 10])

regr_trans = TransformedTargetRegressor(
    regressor=RidgeCV(),
    transformer=QuantileTransformer(output_distribution='normal'))
regr_trans.fit(X_train, y_train)
y_pred = regr_trans.predict(X_test)

ax1.scatter(y_test, y_pred)
ax1.plot([0, 10], [0, 10], '--k')
ax1.set_ylabel('Target predicted')
ax1.set_xlabel('True Target')
ax1.set_title('Ridge regression \n with target transformation')
ax1.text(1, 9, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 10])

f.suptitle("Boston housing data: distance to employment centers", y=0.035)
f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

plt.show()

```
58 - /tmp/repos/scikit-learn/examples/text/plot_document_classification_20newsgroups.py
```python
"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


# #############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
```
59 - /tmp/repos/scikit-learn/sklearn/preprocessing/__init__.py
```python
"""
The :mod:`sklearn.preprocessing` module includes scaling, centering,
normalization, binarization and imputation methods.
"""

from ._function_transformer import FunctionTransformer

from .data import Binarizer
from .data import KernelCenterer
from .data import MinMaxScaler
from .data import MaxAbsScaler
from .data import Normalizer
from .data import RobustScaler
from .data import StandardScaler
from .data import QuantileTransformer
from .data import add_dummy_feature
from .data import binarize
from .data import normalize
from .data import scale
from .data import robust_scale
from .data import maxabs_scale
from .data import minmax_scale
from .data import quantile_transform
from .data import power_transform
from .data import OneHotEncoder
from .data import PowerTransformer
from .data import CategoricalEncoder
from .data import PolynomialFeatures

from .label import label_binarize
from .label import LabelBinarizer
from .label import LabelEncoder
from .label import MultiLabelBinarizer

from ._target import TransformedTargetRegressor
from .imputation import Imputer


__all__ = [
    'Binarizer',
    'FunctionTransformer',
    'Imputer',
    'KernelCenterer',
    'LabelBinarizer',
    'LabelEncoder',
    'MultiLabelBinarizer',
    'MinMaxScaler',
    'MaxAbsScaler',
    'QuantileTransformer',
    'Normalizer',
    'OneHotEncoder',
    'CategoricalEncoder',
    'PowerTransformer',
    'RobustScaler',
    'StandardScaler',
    'TransformedTargetRegressor',
    'add_dummy_feature',
    'PolynomialFeatures',
    'binarize',
    'normalize',
    'scale',
    'robust_scale',
    'maxabs_scale',
    'minmax_scale',
    'label_binarize',
    'quantile_transform',
    'power_transform',
]
```
60 - /tmp/repos/scikit-learn/doc/tutorial/machine_learning_map/pyparsing.py
```python
def dump(self, indent='', depth=0, full=True):
        """
        Diagnostic method for listing out the contents of a C{ParseResults}.
        Accepts an optional C{indent} argument so that this string can be embedded
        in a nested display of other data.

        Example::
            integer = Word(nums)
            date_str = integer("year") + '/' + integer("month") + '/' + integer("day")
            
            result = date_str.parseString('12/31/1999')
            print(result.dump())
        prints::
            ['12', '/', '31', '/', '1999']
            - day: 1999
            - month: 31
            - year: 12
        """
        out = []
        NL = '\n'
        out.append( indent+_ustr(self.asList()) )
        if full:
            if self.haskeys():
                items = sorted((str(k), v) for k,v in self.items())
                for k,v in items:
                    if out:
                        out.append(NL)
                    out.append( "%s%s- %s: " % (indent,('  '*depth), k) )
                    if isinstance(v,ParseResults):
                        if v:
                            out.append( v.dump(indent,depth+1) )
                        else:
                            out.append(_ustr(v))
                    else:
                        out.append(repr(v))
            elif any(isinstance(vv,ParseResults) for vv in self):
                v = self
                for i,vv in enumerate(v):
                    if isinstance(vv,ParseResults):
                        out.append("\n%s%s[%d]:\n%s%s%s" % (indent,('  '*(depth)),i,indent,('  '*(depth+1)),vv.dump(indent,depth+1) ))
                    else:
                        out.append("\n%s%s[%d]:\n%s%s%s" % (indent,('  '*(depth)),i,indent,('  '*(depth+1)),_ustr(vv)))
            
        return "".join(out)

    def pprint(self, *args, **kwargs):
        """
        Pretty-printer for parsed results as a list, using the C{pprint} module.
        Accepts additional positional or keyword args as defined for the 
        C{pprint.pprint} method. (U{http://docs.python.org/3/library/pprint.html#pprint.pprint})

        Example::
            ident = Word(alphas, alphanums)
            num = Word(nums)
            func = Forward()
            term = ident | num | Group('(' + func + ')')
            func <<= ident + Group(Optional(delimitedList(term)))
            result = func.parseString("fna a,b,(fnb c,d,200),100")
            result.pprint(width=40)
        prints::
            ['fna',
             ['a',
              'b',
              ['(', 'fnb', ['c', 'd', '200'], ')'],
              '100']]
        """
        pprint.pprint(self.asList(), *args, **kwargs)

    # add support for pickle protocol
    def __getstate__(self):
        return ( self.__toklist,
                 ( self.__tokdict.copy(),
                   self.__parent is not None and self.__parent() or None,
                   self.__accumNames,
                   self.__name ) )

    def __setstate__(self,state):
        self.__toklist = state[0]
        (self.__tokdict,
         par,
         inAccumNames,
         self.__name) = state[1]
        self.__accumNames = {}
        self.__accumNames.update(inAccumNames)
        if par is not None:
            self.__parent = wkref(par)
        else:
            self.__parent = None

    def __getnewargs__(self):
        return self.__toklist, self.__name, self.__asList, self.__modal

    def __dir__(self):
        return (dir(type(self)) + list(self.keys()))

collections.MutableMapping.register(ParseResults)


```
61 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _yield_classifier_checks(name, classifier):
    # test classifiers can handle non-array data
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    # basic consistency testing
    yield check_classifiers_train
    yield check_classifiers_regression_target
    if (name not in ["MultinomialNB", "ComplementNB", "LabelPropagation",
                     "LabelSpreading"] and
        # TODO some complication with -1 label
            name not in ["DecisionTreeClassifier", "ExtraTreeClassifier"]):
        # We don't raise a warning in these classifiers, as
        # the column y interface is used by the forests.

        yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    # test if NotFittedError is raised
    yield check_estimators_unfitted
    if 'class_weight' in classifier.get_params().keys():
        yield check_class_weight_classifiers

    yield check_non_transformer_estimators_n_iter
    # test if predict_proba is a monotonic transformation of decision_function
    yield check_decision_proba_consistency


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_no_nan(name, estimator_orig):
    # Checks that the Estimator targets are not NaN.
    estimator = clone(estimator_orig)
    rng = np.random.RandomState(888)
    X = rng.randn(10, 5)
    y = np.ones(10) * np.inf
    y = multioutput_estimator_convert_y_2d(estimator, y)

    errmsg = "Input contains NaN, infinity or a value too large for " \
             "dtype('float64')."
    try:
        estimator.fit(X, y)
    except ValueError as e:
        if str(e) != errmsg:
            raise ValueError("Estimator {0} raised error as expected, but "
                             "does not match expected error message"
                             .format(name))
    else:
        raise ValueError("Estimator {0} should have raised error on fitting "
                         "array y with NaN value.".format(name))


def _yield_regressor_checks(name, regressor):
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing
    yield check_regressors_train
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    yield check_regressors_no_decision_function
    yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    if name != 'CCA':
        # check that the regressor handles int input
        yield check_regressors_int
    if name != "GaussianProcessRegressor":
        # Test if NotFittedError is raised
        yield check_estimators_unfitted
    yield check_non_transformer_estimators_n_iter


def _yield_transformer_checks(name, transformer):
    # All transformers should either deal with sparse data or raise an
    # exception with type TypeError and an intelligible error message
    if name not in ['AdditiveChi2Sampler', 'Binarizer', 'Normalizer',
                    'PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']:
        yield check_transformer_data_not_an_array
    # these don't actually fit the data, so don't raise errors
    if name not in ['AdditiveChi2Sampler', 'Binarizer',
                    'FunctionTransformer', 'Normalizer']:
        # basic tests
        yield check_transformer_general
        yield check_transformers_unfitted
    # Dependent on external solvers and hence accessing the iter
    # param is non-trivial.
    external_solver = ['Isomap', 'KernelPCA', 'LocallyLinearEmbedding',
                       'RandomizedLasso', 'LogisticRegressionCV']
    if name not in external_solver:
        yield check_transformer_n_iter


def _yield_clustering_checks(name, clusterer):
    yield check_clusterer_compute_labels_predict
    if name not in ('WardAgglomeration', "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield check_estimators_partial_fit_n_features
    yield check_non_transformer_estimators_n_iter



```
62 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _check_transformer(name, transformer_orig, X, y):
    if name in ('CCA', 'LocallyLinearEmbedding', 'KernelPCA') and _is_32bit():
        # Those transformers yield non-deterministic output when executed on
        # a 32bit Python. The same transformers are stable on 64bit Python.
        # FIXME: try to isolate a minimalistic reproduction case only depending
        # on numpy & scipy and/or maybe generate a test dataset that does not
        # cause such unstable behaviors.
        msg = name + ' is non deterministic on 32bit Python'
        raise SkipTest(msg)
    n_samples, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    # fit

    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[y, y]
        y_[::2, 1] *= 2
    else:
        y_ = y

    transformer.fit(X, y_)
    # fit_transform method should work on non fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y_)

    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert_equal(x_pred.shape[0], n_samples)
    else:
        # check for consistent n_samples
        assert_equal(X_pred.shape[0], n_samples)

    if hasattr(transformer, 'transform'):
        if name in CROSS_DECOMPOSITION:
            X_pred2 = transformer.transform(X, y_)
            X_pred3 = transformer.fit_transform(X, y=y_)
        else:
            X_pred2 = transformer.transform(X)
            X_pred3 = transformer.fit_transform(X, y=y_)
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                assert_allclose_dense_sparse(
                    x_pred, x_pred2, atol=1e-2,
                    err_msg="fit_transform and transform outcomes "
                            "not consistent in %s"
                    % transformer)
                assert_allclose_dense_sparse(
                    x_pred, x_pred3, atol=1e-2,
                    err_msg="consecutive fit_transform outcomes "
                            "not consistent in %s"
                    % transformer)
        else:
            assert_allclose_dense_sparse(
                X_pred, X_pred2,
                err_msg="fit_transform and transform outcomes "
                        "not consistent in %s"
                % transformer, atol=1e-2)
            assert_allclose_dense_sparse(
                X_pred, X_pred3, atol=1e-2,
                err_msg="consecutive fit_transform outcomes "
                        "not consistent in %s"
                % transformer)
            assert_equal(_num_samples(X_pred2), n_samples)
            assert_equal(_num_samples(X_pred3), n_samples)

        # raises error on malformed input for transform
        if hasattr(X, 'T'):
            # If it's not an array, it does not have a 'T' property
            with assert_raises(ValueError, msg="The transformer {} does "
                               "not raise an error when the number of "
                               "features in transform is different from"
                               " the number of features in "
                               "fit.".format(name)):
                transformer.transform(X.T)
```
63 - /tmp/repos/scikit-learn/sklearn/inspection/partial_dependence.py
```python

    check_matplotlib_support('plot_partial_dependence')  # noqa
    import matplotlib.pyplot as plt  # noqa
    from matplotlib import transforms  # noqa
    from matplotlib.ticker import MaxNLocator  # noqa
    from matplotlib.ticker import ScalarFormatter  # noqa

    # set target_idx for multi-class estimators
    if hasattr(estimator, 'classes_') and np.size(estimator.classes_) > 2:
        if target is None:
            raise ValueError('target must be specified for multi-class')
        target_idx = np.searchsorted(estimator.classes_, target)
        if (not (0 <= target_idx < len(estimator.classes_)) or
                estimator.classes_[target_idx] != target):
            raise ValueError('target not in est.classes_, got {}'.format(
                target))
    else:
        # regression and binary classification
        target_idx = 0

    X = check_array(X)
    n_features = X.shape[1]

    # convert feature_names to list
    if feature_names is None:
        # if feature_names is None, use feature indices as name
        feature_names = [str(i) for i in range(n_features)]
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('feature_names should not contain duplicates.')

    def convert_feature(fx):
        if isinstance(fx, str):
            try:
                fx = feature_names.index(fx)
            except ValueError:
                raise ValueError('Feature %s not in feature_names' % fx)
        return int(fx)
```
64 - /tmp/repos/scikit-learn/examples/svm/plot_rbf_parameters.py
```python


# #############################################################################
# Load and prepare data set
#
# dataset for grid search

iris = load_iris()
X = iris.data
y = iris.target

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

```
65 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
        y_ = enforce_estimator_tags_y(estimator, y_)

        set_random_state(estimator, 0)
        if name == 'AffinityPropagation':
            estimator.fit(X)
        else:
            estimator.fit(X, y_)

        assert estimator.n_iter_ >= 1


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [2., 5., 4.]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                               random_state=0, n_features=2, cluster_std=0.1)
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert iter_ >= 1
        else:
            assert estimator.n_iter_ >= 1


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert all(item in deep_params.items() for item in
               shallow_params.items())



```
66 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
**67 - /tmp/repos/scikit-learn/sklearn/utils/_pprint.py**:
```python
"""This module contains the _EstimatorPrettyPrinter class used in
BaseEstimator.__repr__ for pretty-printing estimators"""

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 Python Software Foundation;
# All Rights Reserved

# Authors: Fred L. Drake, Jr. <fdrake@acm.org> (built-in CPython pprint module)
#          Nicolas Hug (scikit-learn specific changes)

# License: PSF License version 2 (see below)

# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# --------------------------------------------

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
# and the Individual or Organization ("Licensee") accessing and otherwise
# using this software ("Python") in source or binary form and its associated
# documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
# grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use Python alone or in any
# derivative version, provided, however, that PSF's License Agreement and
# PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
# 2017, 2018 Python Software Foundation; All Rights Reserved" are retained in
# Python alone or in any derivative version prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates Python or any part thereof, and wants to make the derivative
# work available to others as provided herein, then Licensee hereby agrees to
# include in any such work a brief summary of the changes made to Python.

# 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES
# NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT
# NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
# PYTHON WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY
# INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE
# THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between PSF and
# Licensee. This License Agreement does not grant permission to use PSF
# trademarks or trade name in a trademark sense to endorse or promote products
# or services of Licensee, or any third party.

# 8. By copying, installing or otherwise using Python, Licensee agrees to be
# bound by the terms and conditions of this License Agreement.


# Brief summary of changes to original code:
# - "compact" parameter is supported for dicts, not just lists or tuples
# - estimators have a custom handler, they're not just treated as objects
# - long sequences (lists, tuples, dict items) with more than N elements are
#   shortened using ellipsis (', ...') at the end.

from inspect import signature
import pprint
from collections import OrderedDict

from ..base import BaseEstimator
from .._config import get_config
from . import is_scalar_nan


class KeyValTuple(tuple):
    """Dummy class for correctly rendering key-value tuples from dicts."""
    def __repr__(self):
        # needed for _dispatch[tuple.__repr__] not to be overridden
        return super().__repr__()


class KeyValTupleParam(KeyValTuple):
    """Dummy class for correctly rendering key-value tuples from parameters."""
    pass



```
