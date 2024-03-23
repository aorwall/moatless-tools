# scikit-learn__scikit-learn-10774

* repo: scikit-learn/scikit-learn
* base_commit: `ccbf9975fcf1676f6ac4f311e388529d3a3c4d3f`

## Problem statement

return_X_y should be available on more dataset loaders/fetchers
Version 0.18 added a `return_X_y` option to `load_iris` et al., but not to, for example, `fetch_kddcup99`.

All dataset loaders that currently return Bunches should also be able to return (X, y).


## Patch

```diff
diff --git a/sklearn/datasets/california_housing.py b/sklearn/datasets/california_housing.py
--- a/sklearn/datasets/california_housing.py
+++ b/sklearn/datasets/california_housing.py
@@ -50,7 +50,8 @@
 logger = logging.getLogger(__name__)
 
 
-def fetch_california_housing(data_home=None, download_if_missing=True):
+def fetch_california_housing(data_home=None, download_if_missing=True,
+                             return_X_y=False):
     """Loader for the California housing dataset from StatLib.
 
     Read more in the :ref:`User Guide <datasets>`.
@@ -65,6 +66,12 @@ def fetch_california_housing(data_home=None, download_if_missing=True):
         If False, raise a IOError if the data is not locally available
         instead of trying to download the data from the source site.
 
+
+    return_X_y : boolean, default=False. If True, returns ``(data.data,
+    data.target)`` instead of a Bunch object.
+
+        .. versionadded:: 0.20
+
     Returns
     -------
     dataset : dict-like object with the following attributes:
@@ -81,6 +88,10 @@ def fetch_california_housing(data_home=None, download_if_missing=True):
     dataset.DESCR : string
         Description of the California housing dataset.
 
+    (data, target) : tuple if ``return_X_y`` is True
+
+        .. versionadded:: 0.20
+
     Notes
     ------
 
@@ -132,6 +143,9 @@ def fetch_california_housing(data_home=None, download_if_missing=True):
     # target in units of 100,000
     target = target / 100000.0
 
+    if return_X_y:
+        return data, target
+
     return Bunch(data=data,
                  target=target,
                  feature_names=feature_names,
diff --git a/sklearn/datasets/covtype.py b/sklearn/datasets/covtype.py
--- a/sklearn/datasets/covtype.py
+++ b/sklearn/datasets/covtype.py
@@ -42,7 +42,7 @@
 
 
 def fetch_covtype(data_home=None, download_if_missing=True,
-                  random_state=None, shuffle=False):
+                  random_state=None, shuffle=False, return_X_y=False):
     """Load the covertype dataset, downloading it if necessary.
 
     Read more in the :ref:`User Guide <datasets>`.
@@ -67,6 +67,11 @@ def fetch_covtype(data_home=None, download_if_missing=True,
     shuffle : bool, default=False
         Whether to shuffle dataset.
 
+    return_X_y : boolean, default=False. If True, returns ``(data.data,
+    data.target)`` instead of a Bunch object.
+
+        .. versionadded:: 0.20
+
     Returns
     -------
     dataset : dict-like object with the following attributes:
@@ -81,6 +86,9 @@ def fetch_covtype(data_home=None, download_if_missing=True,
     dataset.DESCR : string
         Description of the forest covertype dataset.
 
+    (data, target) : tuple if ``return_X_y`` is True
+
+        .. versionadded:: 0.20
     """
 
     data_home = get_data_home(data_home=data_home)
@@ -120,4 +128,7 @@ def fetch_covtype(data_home=None, download_if_missing=True,
         X = X[ind]
         y = y[ind]
 
+    if return_X_y:
+        return X, y
+
     return Bunch(data=X, target=y, DESCR=__doc__)
diff --git a/sklearn/datasets/kddcup99.py b/sklearn/datasets/kddcup99.py
--- a/sklearn/datasets/kddcup99.py
+++ b/sklearn/datasets/kddcup99.py
@@ -47,7 +47,7 @@
 
 def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
                    random_state=None,
-                   percent10=True, download_if_missing=True):
+                   percent10=True, download_if_missing=True, return_X_y=False):
     """Load and return the kddcup 99 dataset (classification).
 
     The KDD Cup '99 dataset was created by processing the tcpdump portions
@@ -155,6 +155,12 @@ def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
         If False, raise a IOError if the data is not locally available
         instead of trying to download the data from the source site.
 
+    return_X_y : boolean, default=False.
+        If True, returns ``(data, target)`` instead of a Bunch object. See
+        below for more information about the `data` and `target` object.
+
+        .. versionadded:: 0.20
+
     Returns
     -------
     data : Bunch
@@ -162,6 +168,9 @@ def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
         'data', the data to learn and 'target', the regression target for each
         sample.
 
+    (data, target) : tuple if ``return_X_y`` is True
+
+        .. versionadded:: 0.20
 
     References
     ----------
@@ -230,6 +239,9 @@ def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
     if shuffle:
         data, target = shuffle_method(data, target, random_state=random_state)
 
+    if return_X_y:
+        return data, target
+
     return Bunch(data=data, target=target)
 
 
diff --git a/sklearn/datasets/lfw.py b/sklearn/datasets/lfw.py
--- a/sklearn/datasets/lfw.py
+++ b/sklearn/datasets/lfw.py
@@ -238,7 +238,7 @@ def _fetch_lfw_people(data_folder_path, slice_=None, color=False, resize=None,
 def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
                      min_faces_per_person=0, color=False,
                      slice_=(slice(70, 195), slice(78, 172)),
-                     download_if_missing=True):
+                     download_if_missing=True, return_X_y=False):
     """Loader for the Labeled Faces in the Wild (LFW) people dataset
 
     This dataset is a collection of JPEG pictures of famous people
@@ -287,6 +287,12 @@ def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
         If False, raise a IOError if the data is not locally available
         instead of trying to download the data from the source site.
 
+    return_X_y : boolean, default=False. If True, returns ``(dataset.data,
+    dataset.target)`` instead of a Bunch object. See below for more
+    information about the `dataset.data` and `dataset.target` object.
+
+        .. versionadded:: 0.20
+
     Returns
     -------
     dataset : dict-like object with the following attributes:
@@ -307,6 +313,11 @@ def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
 
     dataset.DESCR : string
         Description of the Labeled Faces in the Wild (LFW) dataset.
+
+    (data, target) : tuple if ``return_X_y`` is True
+
+        .. versionadded:: 0.20
+
     """
     lfw_home, data_folder_path = check_fetch_lfw(
         data_home=data_home, funneled=funneled,
@@ -323,8 +334,13 @@ def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
         data_folder_path, resize=resize,
         min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)
 
+    X = faces.reshape(len(faces), -1)
+
+    if return_X_y:
+        return X, target
+
     # pack the results as a Bunch instance
-    return Bunch(data=faces.reshape(len(faces), -1), images=faces,
+    return Bunch(data=X, images=faces,
                  target=target, target_names=target_names,
                  DESCR="LFW faces dataset")
 
diff --git a/sklearn/datasets/rcv1.py b/sklearn/datasets/rcv1.py
--- a/sklearn/datasets/rcv1.py
+++ b/sklearn/datasets/rcv1.py
@@ -70,7 +70,7 @@
 
 
 def fetch_rcv1(data_home=None, subset='all', download_if_missing=True,
-               random_state=None, shuffle=False):
+               random_state=None, shuffle=False, return_X_y=False):
     """Load the RCV1 multilabel dataset, downloading it if necessary.
 
     Version: RCV1-v2, vectors, full sets, topics multilabels.
@@ -112,6 +112,12 @@ def fetch_rcv1(data_home=None, subset='all', download_if_missing=True,
     shuffle : bool, default=False
         Whether to shuffle dataset.
 
+    return_X_y : boolean, default=False. If True, returns ``(dataset.data,
+    dataset.target)`` instead of a Bunch object. See below for more
+    information about the `dataset.data` and `dataset.target` object.
+
+        .. versionadded:: 0.20
+
     Returns
     -------
     dataset : dict-like object with the following attributes:
@@ -132,6 +138,10 @@ def fetch_rcv1(data_home=None, subset='all', download_if_missing=True,
     dataset.DESCR : string
         Description of the RCV1 dataset.
 
+    (data, target) : tuple if ``return_X_y`` is True
+
+        .. versionadded:: 0.20
+
     References
     ----------
     Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A new
@@ -254,6 +264,9 @@ def fetch_rcv1(data_home=None, subset='all', download_if_missing=True,
     if shuffle:
         X, y, sample_id = shuffle_(X, y, sample_id, random_state=random_state)
 
+    if return_X_y:
+        return X, y
+
     return Bunch(data=X, target=y, sample_id=sample_id,
                  target_names=categories, DESCR=__doc__)
 
diff --git a/sklearn/datasets/twenty_newsgroups.py b/sklearn/datasets/twenty_newsgroups.py
--- a/sklearn/datasets/twenty_newsgroups.py
+++ b/sklearn/datasets/twenty_newsgroups.py
@@ -275,7 +275,7 @@ def fetch_20newsgroups(data_home=None, subset='train', categories=None,
 
 
 def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
-                                  download_if_missing=True):
+                                  download_if_missing=True, return_X_y=False):
     """Load the 20 newsgroups dataset and transform it into tf-idf vectors.
 
     This is a convenience function; the tf-idf transformation is done using the
@@ -309,12 +309,21 @@ def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
         If False, raise an IOError if the data is not locally available
         instead of trying to download the data from the source site.
 
+    return_X_y : boolean, default=False. If True, returns ``(data.data,
+    data.target)`` instead of a Bunch object.
+
+        .. versionadded:: 0.20
+
     Returns
     -------
     bunch : Bunch object
         bunch.data: sparse matrix, shape [n_samples, n_features]
         bunch.target: array, shape [n_samples]
         bunch.target_names: list, length [n_classes]
+
+    (data, target) : tuple if ``return_X_y`` is True
+
+        .. versionadded:: 0.20
     """
     data_home = get_data_home(data_home=data_home)
     filebase = '20newsgroup_vectorized'
@@ -369,4 +378,7 @@ def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
         raise ValueError("%r is not a valid subset: should be one of "
                          "['train', 'test', 'all']" % subset)
 
+    if return_X_y:
+        return data, target
+
     return Bunch(data=data, target=target, target_names=target_names)

```

## Test Patch

```diff
diff --git a/sklearn/datasets/tests/test_20news.py b/sklearn/datasets/tests/test_20news.py
--- a/sklearn/datasets/tests/test_20news.py
+++ b/sklearn/datasets/tests/test_20news.py
@@ -5,6 +5,8 @@
 from sklearn.utils.testing import assert_equal
 from sklearn.utils.testing import assert_true
 from sklearn.utils.testing import SkipTest
+from sklearn.datasets.tests.test_common import check_return_X_y
+from functools import partial
 
 from sklearn import datasets
 
@@ -77,6 +79,10 @@ def test_20news_vectorized():
     assert_equal(bunch.target.shape[0], 7532)
     assert_equal(bunch.data.dtype, np.float64)
 
+    # test return_X_y option
+    fetch_func = partial(datasets.fetch_20newsgroups_vectorized, subset='test')
+    check_return_X_y(bunch, fetch_func)
+
     # test subset = all
     bunch = datasets.fetch_20newsgroups_vectorized(subset='all')
     assert_true(sp.isspmatrix_csr(bunch.data))
diff --git a/sklearn/datasets/tests/test_base.py b/sklearn/datasets/tests/test_base.py
--- a/sklearn/datasets/tests/test_base.py
+++ b/sklearn/datasets/tests/test_base.py
@@ -5,6 +5,7 @@
 import numpy
 from pickle import loads
 from pickle import dumps
+from functools import partial
 
 from sklearn.datasets import get_data_home
 from sklearn.datasets import clear_data_home
@@ -19,6 +20,7 @@
 from sklearn.datasets import load_boston
 from sklearn.datasets import load_wine
 from sklearn.datasets.base import Bunch
+from sklearn.datasets.tests.test_common import check_return_X_y
 
 from sklearn.externals.six import b, u
 from sklearn.externals._pilutil import pillow_installed
@@ -27,7 +29,6 @@
 from sklearn.utils.testing import assert_true
 from sklearn.utils.testing import assert_equal
 from sklearn.utils.testing import assert_raises
-from sklearn.utils.testing import assert_array_equal
 
 
 DATA_HOME = tempfile.mkdtemp(prefix="scikit_learn_data_home_test_")
@@ -139,11 +140,7 @@ def test_load_digits():
     assert_equal(numpy.unique(digits.target).size, 10)
 
     # test return_X_y option
-    X_y_tuple = load_digits(return_X_y=True)
-    bunch = load_digits()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(digits, partial(load_digits))
 
 
 def test_load_digits_n_class_lt_10():
@@ -177,11 +174,7 @@ def test_load_diabetes():
     assert_true(res.DESCR)
 
     # test return_X_y option
-    X_y_tuple = load_diabetes(return_X_y=True)
-    bunch = load_diabetes()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(res, partial(load_diabetes))
 
 
 def test_load_linnerud():
@@ -194,11 +187,7 @@ def test_load_linnerud():
     assert_true(os.path.exists(res.target_filename))
 
     # test return_X_y option
-    X_y_tuple = load_linnerud(return_X_y=True)
-    bunch = load_linnerud()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(res, partial(load_linnerud))
 
 
 def test_load_iris():
@@ -210,11 +199,7 @@ def test_load_iris():
     assert_true(os.path.exists(res.filename))
 
     # test return_X_y option
-    X_y_tuple = load_iris(return_X_y=True)
-    bunch = load_iris()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(res, partial(load_iris))
 
 
 def test_load_wine():
@@ -225,11 +210,7 @@ def test_load_wine():
     assert_true(res.DESCR)
 
     # test return_X_y option
-    X_y_tuple = load_wine(return_X_y=True)
-    bunch = load_wine()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(res, partial(load_wine))
 
 
 def test_load_breast_cancer():
@@ -241,11 +222,7 @@ def test_load_breast_cancer():
     assert_true(os.path.exists(res.filename))
 
     # test return_X_y option
-    X_y_tuple = load_breast_cancer(return_X_y=True)
-    bunch = load_breast_cancer()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(res, partial(load_breast_cancer))
 
 
 def test_load_boston():
@@ -257,11 +234,7 @@ def test_load_boston():
     assert_true(os.path.exists(res.filename))
 
     # test return_X_y option
-    X_y_tuple = load_boston(return_X_y=True)
-    bunch = load_boston()
-    assert_true(isinstance(X_y_tuple, tuple))
-    assert_array_equal(X_y_tuple[0], bunch.data)
-    assert_array_equal(X_y_tuple[1], bunch.target)
+    check_return_X_y(res, partial(load_boston))
 
 
 def test_loads_dumps_bunch():
diff --git a/sklearn/datasets/tests/test_california_housing.py b/sklearn/datasets/tests/test_california_housing.py
new file mode 100644
--- /dev/null
+++ b/sklearn/datasets/tests/test_california_housing.py
@@ -0,0 +1,26 @@
+"""Test the california_housing loader.
+
+Skipped if california_housing is not already downloaded to data_home.
+"""
+
+from sklearn.datasets import fetch_california_housing
+from sklearn.utils.testing import SkipTest
+from sklearn.datasets.tests.test_common import check_return_X_y
+from functools import partial
+
+
+def fetch(*args, **kwargs):
+    return fetch_california_housing(*args, download_if_missing=False, **kwargs)
+
+
+def test_fetch():
+    try:
+        data = fetch()
+    except IOError:
+        raise SkipTest("California housing dataset can not be loaded.")
+    assert((20640, 8) == data.data.shape)
+    assert((20640, ) == data.target.shape)
+
+    # test return_X_y option
+    fetch_func = partial(fetch)
+    check_return_X_y(data, fetch_func)
diff --git a/sklearn/datasets/tests/test_common.py b/sklearn/datasets/tests/test_common.py
new file mode 100644
--- /dev/null
+++ b/sklearn/datasets/tests/test_common.py
@@ -0,0 +1,9 @@
+"""Test loaders for common functionality.
+"""
+
+
+def check_return_X_y(bunch, fetch_func_partial):
+    X_y_tuple = fetch_func_partial(return_X_y=True)
+    assert(isinstance(X_y_tuple, tuple))
+    assert(X_y_tuple[0].shape == bunch.data.shape)
+    assert(X_y_tuple[1].shape == bunch.target.shape)
diff --git a/sklearn/datasets/tests/test_covtype.py b/sklearn/datasets/tests/test_covtype.py
--- a/sklearn/datasets/tests/test_covtype.py
+++ b/sklearn/datasets/tests/test_covtype.py
@@ -5,6 +5,8 @@
 
 from sklearn.datasets import fetch_covtype
 from sklearn.utils.testing import assert_equal, SkipTest
+from sklearn.datasets.tests.test_common import check_return_X_y
+from functools import partial
 
 
 def fetch(*args, **kwargs):
@@ -28,3 +30,7 @@ def test_fetch():
     y1, y2 = data1['target'], data2['target']
     assert_equal((X1.shape[0],), y1.shape)
     assert_equal((X1.shape[0],), y2.shape)
+
+    # test return_X_y option
+    fetch_func = partial(fetch)
+    check_return_X_y(data1, fetch_func)
diff --git a/sklearn/datasets/tests/test_kddcup99.py b/sklearn/datasets/tests/test_kddcup99.py
--- a/sklearn/datasets/tests/test_kddcup99.py
+++ b/sklearn/datasets/tests/test_kddcup99.py
@@ -6,7 +6,10 @@
 """
 
 from sklearn.datasets import fetch_kddcup99
+from sklearn.datasets.tests.test_common import check_return_X_y
 from sklearn.utils.testing import assert_equal, SkipTest
+from functools import partial
+
 
 
 def test_percent10():
@@ -38,6 +41,9 @@ def test_percent10():
     assert_equal(data.data.shape, (9571, 3))
     assert_equal(data.target.shape, (9571,))
 
+    fetch_func = partial(fetch_kddcup99, 'smtp')
+    check_return_X_y(data, fetch_func)
+
 
 def test_shuffle():
     try:
diff --git a/sklearn/datasets/tests/test_lfw.py b/sklearn/datasets/tests/test_lfw.py
--- a/sklearn/datasets/tests/test_lfw.py
+++ b/sklearn/datasets/tests/test_lfw.py
@@ -13,6 +13,7 @@
 import shutil
 import tempfile
 import numpy as np
+from functools import partial
 from sklearn.externals import six
 from sklearn.externals._pilutil import pillow_installed, imsave
 from sklearn.datasets import fetch_lfw_pairs
@@ -22,6 +23,7 @@
 from sklearn.utils.testing import assert_equal
 from sklearn.utils.testing import SkipTest
 from sklearn.utils.testing import assert_raises
+from sklearn.datasets.tests.test_common import check_return_X_y
 
 
 SCIKIT_LEARN_DATA = tempfile.mkdtemp(prefix="scikit_learn_lfw_test_")
@@ -139,6 +141,13 @@ def test_load_fake_lfw_people():
                        ['Abdelatif Smith', 'Abhati Kepler', 'Camara Alvaro',
                         'Chen Dupont', 'John Lee', 'Lin Bauman', 'Onur Lopez'])
 
+    # test return_X_y option
+    fetch_func = partial(fetch_lfw_people, data_home=SCIKIT_LEARN_DATA,
+                         resize=None,
+                         slice_=None, color=True,
+                         download_if_missing=False)
+    check_return_X_y(lfw_people, fetch_func)
+
 
 def test_load_fake_lfw_people_too_restrictive():
     assert_raises(ValueError, fetch_lfw_people, data_home=SCIKIT_LEARN_DATA,
diff --git a/sklearn/datasets/tests/test_rcv1.py b/sklearn/datasets/tests/test_rcv1.py
--- a/sklearn/datasets/tests/test_rcv1.py
+++ b/sklearn/datasets/tests/test_rcv1.py
@@ -6,7 +6,9 @@
 import errno
 import scipy.sparse as sp
 import numpy as np
+from functools import partial
 from sklearn.datasets import fetch_rcv1
+from sklearn.datasets.tests.test_common import check_return_X_y
 from sklearn.utils.testing import assert_almost_equal
 from sklearn.utils.testing import assert_array_equal
 from sklearn.utils.testing import assert_equal
@@ -53,6 +55,11 @@ def test_fetch_rcv1():
     X2, Y2 = data2.data, data2.target
     s2 = data2.sample_id
 
+    # test return_X_y option
+    fetch_func = partial(fetch_rcv1, shuffle=False, subset='train',
+                         download_if_missing=False)
+    check_return_X_y(data2, fetch_func)
+
     # The first 23149 samples are the training samples
     assert_array_equal(np.sort(s1[:23149]), np.sort(s2))
 

```
