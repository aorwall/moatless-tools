# scikit-learn__scikit-learn-10774

| **scikit-learn/scikit-learn** | `ccbf9975fcf1676f6ac4f311e388529d3a3c4d3f` |
| ---- | ---- |
| **No of patches** | 6 |
| **All found context length** | 16486 |
| **Any found context length** | 7231 |
| **Avg pos** | 38.666666666666664 |
| **Min pos** | 16 |
| **Max pos** | 44 |
| **Top file pos** | 3 |
| **Missing snippets** | 23 |
| **Missing patch files** | 3 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/datasets/california_housing.py | 53 | 53 | - | - | -
| sklearn/datasets/california_housing.py | 68 | 68 | - | - | -
| sklearn/datasets/california_housing.py | 84 | 84 | - | - | -
| sklearn/datasets/california_housing.py | 135 | 135 | - | - | -
| sklearn/datasets/covtype.py | 45 | 45 | - | - | -
| sklearn/datasets/covtype.py | 70 | 70 | - | - | -
| sklearn/datasets/covtype.py | 84 | 84 | - | - | -
| sklearn/datasets/covtype.py | 123 | 123 | - | - | -
| sklearn/datasets/kddcup99.py | 50 | 50 | 30 | 17 | 12856
| sklearn/datasets/kddcup99.py | 158 | 158 | 30 | 17 | 12856
| sklearn/datasets/kddcup99.py | 165 | 165 | 30 | 17 | 12856
| sklearn/datasets/kddcup99.py | 233 | 233 | 38 | 17 | 16486
| sklearn/datasets/lfw.py | 241 | 241 | - | - | -
| sklearn/datasets/lfw.py | 290 | 290 | - | - | -
| sklearn/datasets/lfw.py | 310 | 310 | - | - | -
| sklearn/datasets/lfw.py | 326 | 326 | - | - | -
| sklearn/datasets/rcv1.py | 73 | 73 | - | 3 | -
| sklearn/datasets/rcv1.py | 115 | 115 | - | 3 | -
| sklearn/datasets/rcv1.py | 135 | 135 | - | 3 | -
| sklearn/datasets/rcv1.py | 257 | 257 | - | 3 | -
| sklearn/datasets/twenty_newsgroups.py | 278 | 278 | 44 | 9 | 18836
| sklearn/datasets/twenty_newsgroups.py | 312 | 312 | 44 | 9 | 18836
| sklearn/datasets/twenty_newsgroups.py | 372 | 372 | 16 | 9 | 7231


## Problem Statement

```
return_X_y should be available on more dataset loaders/fetchers
Version 0.18 added a `return_X_y` option to `load_iris` et al., but not to, for example, `fetch_kddcup99`.

All dataset loaders that currently return Bunches should also be able to return (X, y).

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/datasets/base.py | 326 | 391| 523 | 523 | 6979 | 
| 2 | 1 sklearn/datasets/base.py | 612 | 671| 495 | 1018 | 6979 | 
| 3 | 2 sklearn/utils/validation.py | 663 | 677| 205 | 1223 | 14341 | 
| 4 | 2 sklearn/datasets/base.py | 674 | 741| 516 | 1739 | 14341 | 
| 5 | 2 sklearn/datasets/base.py | 394 | 476| 709 | 2448 | 14341 | 
| 6 | 2 sklearn/datasets/base.py | 559 | 609| 431 | 2879 | 14341 | 
| 7 | 2 sklearn/datasets/base.py | 479 | 556| 614 | 3493 | 14341 | 
| 8 | 2 sklearn/datasets/base.py | 249 | 323| 579 | 4072 | 14341 | 
| 9 | **3 sklearn/datasets/rcv1.py** | 155 | 238| 797 | 4869 | 16999 | 
| 10 | 4 sklearn/model_selection/_split.py | 1836 | 1859| 129 | 4998 | 34995 | 
| 11 | 5 sklearn/feature_extraction/dict_vectorizer.py | 213 | 231| 144 | 5142 | 37723 | 
| 12 | 6 sklearn/datasets/svmlight_format.py | 452 | 482| 313 | 5455 | 42234 | 
| 13 | 7 sklearn/datasets/__init__.py | 1 | 52| 516 | 5971 | 43092 | 
| 14 | 7 sklearn/utils/validation.py | 562 | 662| 979 | 6950 | 43092 | 
| 15 | 8 sklearn/manifold/isomap.py | 150 | 167| 122 | 7072 | 44776 | 
| **-> 16 <-** | **9 sklearn/datasets/twenty_newsgroups.py** | 359 | 373| 159 | 7231 | 47929 | 
| 17 | 9 sklearn/manifold/isomap.py | 169 | 185| 118 | 7349 | 47929 | 
| 18 | 10 sklearn/neighbors/base.py | 728 | 745| 167 | 7516 | 54609 | 
| 19 | 11 sklearn/model_selection/_validation.py | 699 | 731| 348 | 7864 | 66583 | 
| 20 | 12 sklearn/isotonic.py | 234 | 250| 145 | 8009 | 69961 | 
| 21 | 13 sklearn/cluster/birch.py | 22 | 37| 142 | 8151 | 75252 | 
| 22 | 14 sklearn/cross_validation.py | 1381 | 1398| 214 | 8365 | 92744 | 
| 23 | 14 sklearn/datasets/svmlight_format.py | 33 | 147| 1125 | 9490 | 92744 | 
| 24 | 14 sklearn/datasets/svmlight_format.py | 319 | 373| 474 | 9964 | 92744 | 
| 25 | 15 sklearn/metrics/pairwise.py | 33 | 55| 157 | 10121 | 104276 | 
| 26 | 15 sklearn/datasets/svmlight_format.py | 376 | 450| 772 | 10893 | 104276 | 
| 27 | 16 sklearn/gaussian_process/kernels.py | 1289 | 1345| 566 | 11459 | 119528 | 
| 28 | 16 sklearn/model_selection/_split.py | 1203 | 1229| 161 | 11620 | 119528 | 
| 29 | 16 sklearn/cluster/birch.py | 578 | 596| 130 | 11750 | 119528 | 
| **-> 30 <-** | **17 sklearn/datasets/kddcup99.py** | 48 | 178| 1106 | 12856 | 122704 | 
| 31 | 17 sklearn/gaussian_process/kernels.py | 1176 | 1242| 644 | 13500 | 122704 | 
| 32 | 18 sklearn/cluster/k_means_.py | 912 | 930| 126 | 13626 | 136429 | 
| 33 | 18 sklearn/cross_validation.py | 1713 | 1743| 279 | 13905 | 136429 | 
| 34 | 19 sklearn/manifold/mds.py | 392 | 432| 400 | 14305 | 140256 | 
| 35 | 20 benchmarks/bench_plot_randomized_svd.py | 131 | 178| 522 | 14827 | 144640 | 
| 36 | 21 sklearn/utils/__init__.py | 122 | 164| 307 | 15134 | 148546 | 
| 37 | 21 sklearn/cross_validation.py | 1283 | 1380| 807 | 15941 | 148546 | 
| **-> 38 <-** | **21 sklearn/datasets/kddcup99.py** | 179 | 233| 545 | 16486 | 148546 | 
| 39 | 22 sklearn/dummy.py | 162 | 235| 595 | 17081 | 152295 | 
| 40 | 23 examples/neural_networks/plot_rbm_logistic_classification.py | 45 | 76| 271 | 17352 | 153386 | 
| 41 | 24 sklearn/manifold/t_sne.py | 862 | 881| 146 | 17498 | 161695 | 
| 42 | 24 sklearn/cluster/k_means_.py | 932 | 953| 175 | 17673 | 161695 | 
| 43 | 24 sklearn/gaussian_process/kernels.py | 1791 | 1836| 418 | 18091 | 161695 | 
| **-> 44 <-** | **24 sklearn/datasets/twenty_newsgroups.py** | 277 | 357| 745 | 18836 | 161695 | 
| 45 | 25 sklearn/datasets/species_distributions.py | 235 | 273| 361 | 19197 | 163880 | 
| 46 | 25 sklearn/cluster/birch.py | 555 | 576| 164 | 19361 | 163880 | 
| 47 | 26 sklearn/utils/metaestimators.py | 144 | 208| 524 | 19885 | 165534 | 
| 48 | 26 sklearn/model_selection/_split.py | 413 | 426| 137 | 20022 | 165534 | 
| 49 | 26 sklearn/model_selection/_split.py | 1752 | 1778| 161 | 20183 | 165534 | 
| 50 | 26 sklearn/neighbors/base.py | 748 | 802| 457 | 20640 | 165534 | 
| 51 | 26 sklearn/datasets/svmlight_format.py | 279 | 316| 378 | 21018 | 165534 | 
| 52 | 27 examples/datasets/plot_iris_dataset.py | 1 | 69| 538 | 21556 | 166080 | 
| 53 | 28 sklearn/decomposition/fastica_.py | 541 | 570| 257 | 21813 | 170793 | 
| 54 | 28 sklearn/gaussian_process/kernels.py | 459 | 498| 346 | 22159 | 170793 | 
| 55 | 28 sklearn/decomposition/fastica_.py | 506 | 539| 203 | 22362 | 170793 | 
| 56 | 28 sklearn/model_selection/_split.py | 288 | 321| 274 | 22636 | 170793 | 
| 57 | 28 sklearn/model_selection/_split.py | 1586 | 1621| 274 | 22910 | 170793 | 
| 58 | 29 sklearn/manifold/spectral_embedding_.py | 518 | 540| 149 | 23059 | 175718 | 
| 59 | 30 sklearn/svm/base.py | 282 | 299| 156 | 23215 | 183476 | 
| 60 | 31 sklearn/kernel_approximation.py | 153 | 179| 208 | 23423 | 187600 | 
| 61 | 32 sklearn/decomposition/incremental_pca.py | 191 | 283| 851 | 24274 | 190149 | 


## Missing Patch Files

 * 1: sklearn/datasets/california_housing.py
 * 2: sklearn/datasets/covtype.py
 * 3: sklearn/datasets/kddcup99.py
 * 4: sklearn/datasets/lfw.py
 * 5: sklearn/datasets/rcv1.py
 * 6: sklearn/datasets/twenty_newsgroups.py

### Hint

```
Looks like a doable first issue - may I take it on?
Sure.

On 1 March 2018 at 12:59, Chris Catalfo <notifications@github.com> wrote:

> Looks like a doable first issue - may I take it on?
>
> —
> You are receiving this because you authored the thread.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/10734#issuecomment-369448829>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz6wB-I558FOQMikXvOGJLH12xAcD7ks5tZ1XygaJpZM4SXlSa>
> .
>

Please refer to the implementation and testing of load_iris's similar
feature.

Thanks - will do.
```

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


## Code snippets

### 1 - sklearn/datasets/base.py:

Start line: 326, End line: 391

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
### 2 - sklearn/datasets/base.py:

Start line: 612, End line: 671

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
### 3 - sklearn/utils/validation.py:

Start line: 663, End line: 677

```python
def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
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
```
### 4 - sklearn/datasets/base.py:

Start line: 674, End line: 741

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
### 5 - sklearn/datasets/base.py:

Start line: 394, End line: 476

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
### 6 - sklearn/datasets/base.py:

Start line: 559, End line: 609

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
### 7 - sklearn/datasets/base.py:

Start line: 479, End line: 556

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
### 8 - sklearn/datasets/base.py:

Start line: 249, End line: 323

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
### 9 - sklearn/datasets/rcv1.py:

Start line: 155, End line: 238

```python
def fetch_rcv1(data_home=None, subset='all', download_if_missing=True,
               random_state=None, shuffle=False):
    # ... other code
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
    # ... other code
```
### 10 - sklearn/model_selection/_split.py:

Start line: 1836, End line: 1859

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
### 16 - sklearn/datasets/twenty_newsgroups.py:

Start line: 359, End line: 373

```python
def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
                                  download_if_missing=True):
    # ... other code

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
### 30 - sklearn/datasets/kddcup99.py:

Start line: 48, End line: 178

```python
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

    """
    # ... other code
```
### 38 - sklearn/datasets/kddcup99.py:

Start line: 179, End line: 233

```python
def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
                   random_state=None,
                   percent10=True, download_if_missing=True):
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
```
### 44 - sklearn/datasets/twenty_newsgroups.py:

Start line: 277, End line: 357

```python
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
    # ... other code
```
