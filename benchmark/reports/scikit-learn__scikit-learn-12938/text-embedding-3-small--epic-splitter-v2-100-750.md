0 - /tmp/repos/scikit-learn/examples/plot_compare_reduction.py
```python
# ...#!/usr/bin/env python
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

```
1 - /tmp/repos/scikit-learn/examples/compose/plot_compare_reduction.py
```python
# ...#!/usr/bin/env python
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

```
2 - /tmp/repos/scikit-learn/examples/compose/plot_column_transformer_mixed_types.py
```python
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
3 - /tmp/repos/scikit-learn/examples/impute/plot_iterative_imputer_variants_comparison.py
```python
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
4 - /tmp/repos/scikit-learn/examples/compose/column_transformer_mixed_types.py
```python
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
5 - /tmp/repos/scikit-learn/examples/preprocessing/plot_discretization_classification.py
```python



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
for ds_cnt, (X, y) in enumerate(datasets):
    
```
6 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
**8 - /tmp/repos/scikit-learn/sklearn/utils/_pprint.py**:
```python
# ..."""This module contains the _EstimatorPrettyPrinter class used in
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
    
```
