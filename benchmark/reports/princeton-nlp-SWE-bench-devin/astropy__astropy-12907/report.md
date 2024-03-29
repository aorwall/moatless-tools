# astropy__astropy-12907

| **astropy/astropy** | `d16bfe05a744909de4b27f5875fe0d4ed41ce607` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
         cright = _coord_matrix(right, 'right', noutp)
     else:
         cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
 
     return np.hstack([cleft, cright])
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| astropy/modeling/separable.py | 245 | 245 | - | 1 | -


## Problem Statement

```
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

\`\`\`python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
\`\`\`

It's separability matrix as you might expect is a diagonal:

\`\`\`python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
\`\`\`

If I make the model more complex:
\`\`\`python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True, False],
       [False, False, False,  True]])
\`\`\`

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
\`\`\`python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True,  True],
       [False, False,  True,  True]])
\`\`\`
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 astropy/modeling/separable.py** | 18 | 63| 403 | 403 | 2411 | 
| 2 | **1 astropy/modeling/separable.py** | 66 | 102| 383 | 786 | 2411 | 
| 3 | **1 astropy/modeling/separable.py** | 290 | 318| 247 | 1033 | 2411 | 
| 4 | **1 astropy/modeling/separable.py** | 171 | 216| 361 | 1394 | 2411 | 
| 5 | 2 astropy/modeling/core.py | 4071 | 4474| 3343 | 4737 | 38799 | 
| 6 | **2 astropy/modeling/separable.py** | 105 | 127| 154 | 4891 | 38799 | 
| 7 | 2 astropy/modeling/core.py | 2466 | 3284| 6615 | 11506 | 38799 | 
| 8 | 3 astropy/modeling/mappings.py | 1 | 98| 796 | 12302 | 41368 | 
| 9 | 4 astropy/io/misc/asdf/tags/transform/compound.py | 63 | 93| 209 | 12511 | 42342 | 
| 10 | 4 astropy/modeling/core.py | 3286 | 4068| 6324 | 18835 | 42342 | 
| 11 | 5 astropy/modeling/convolution.py | 86 | 116| 234 | 19069 | 43191 | 
| 12 | 6 astropy/modeling/bounding_box.py | 1508 | 1544| 265 | 19334 | 53305 | 
| 13 | 6 astropy/modeling/bounding_box.py | 1267 | 1312| 353 | 19687 | 53305 | 
| 14 | 6 astropy/io/misc/asdf/tags/transform/compound.py | 37 | 61| 250 | 19937 | 53305 | 
| 15 | 6 astropy/modeling/bounding_box.py | 1386 | 1420| 297 | 20234 | 53305 | 
| 16 | 6 astropy/io/misc/asdf/tags/transform/compound.py | 3 | 34| 211 | 20445 | 53305 | 
| 17 | 6 astropy/modeling/bounding_box.py | 1334 | 1384| 365 | 20810 | 53305 | 
| 18 | 6 astropy/modeling/bounding_box.py | 1468 | 1488| 181 | 20991 | 53305 | 
| 19 | 6 astropy/modeling/core.py | 88 | 883| 6529 | 27520 | 53305 | 
| 20 | 7 astropy/modeling/fitting.py | 718 | 792| 880 | 28400 | 68479 | 
| 21 | 8 astropy/io/misc/asdf/tags/transform/functional_models.py | 305 | 313| 118 | 28518 | 75939 | 
| 22 | 8 astropy/modeling/fitting.py | 79 | 92| 160 | 28678 | 75939 | 
| 23 | 8 astropy/io/misc/asdf/tags/transform/functional_models.py | 271 | 282| 145 | 28823 | 75939 | 
| 24 | 8 astropy/io/misc/asdf/tags/transform/functional_models.py | 425 | 435| 134 | 28957 | 75939 | 
| 25 | 9 astropy/modeling/tabular.py | 240 | 260| 211 | 29168 | 78829 | 
| 26 | 9 astropy/modeling/fitting.py | 570 | 656| 847 | 30015 | 78829 | 
| 27 | 9 astropy/modeling/bounding_box.py | 1490 | 1506| 165 | 30180 | 78829 | 
| 28 | 10 astropy/modeling/models.py | 9 | 73| 667 | 30847 | 79543 | 
| 29 | 10 astropy/io/misc/asdf/tags/transform/functional_models.py | 391 | 400| 120 | 30967 | 79543 | 
| 30 | 10 astropy/modeling/core.py | 1703 | 2464| 6325 | 37292 | 79543 | 
| 31 | 10 astropy/io/misc/asdf/tags/transform/functional_models.py | 456 | 464| 113 | 37405 | 79543 | 
| 32 | 10 astropy/io/misc/asdf/tags/transform/functional_models.py | 753 | 762| 125 | 37530 | 79543 | 
| 33 | 11 astropy/modeling/__init__.py | 1 | 15| 90 | 37620 | 79633 | 
| 34 | 12 astropy/modeling/polynomial.py | 1080 | 1119| 276 | 37896 | 93854 | 
| 35 | 12 astropy/io/misc/asdf/tags/transform/functional_models.py | 177 | 188| 144 | 38040 | 93854 | 
| 36 | 12 astropy/modeling/bounding_box.py | 763 | 800| 252 | 38292 | 93854 | 
| 37 | 12 astropy/io/misc/asdf/tags/transform/compound.py | 96 | 136| 285 | 38577 | 93854 | 
| 38 | 12 astropy/io/misc/asdf/tags/transform/functional_models.py | 43 | 52| 125 | 38702 | 93854 | 
| 39 | 13 astropy/modeling/functional_models.py | 2945 | 2962| 205 | 38907 | 121581 | 
| 40 | **13 astropy/modeling/separable.py** | 250 | 287| 230 | 39137 | 121581 | 
| 41 | 13 astropy/modeling/mappings.py | 128 | 178| 380 | 39517 | 121581 | 
| 42 | 13 astropy/io/misc/asdf/tags/transform/functional_models.py | 643 | 655| 161 | 39678 | 121581 | 
| 43 | 13 astropy/modeling/bounding_box.py | 1422 | 1442| 170 | 39848 | 121581 | 
| 44 | 14 astropy/io/misc/asdf/tags/transform/tabular.py | 4 | 46| 346 | 40194 | 122323 | 
| 45 | 14 astropy/modeling/fitting.py | 658 | 716| 670 | 40864 | 122323 | 
| 46 | 15 astropy/io/misc/asdf/tags/transform/basic.py | 3 | 67| 476 | 41340 | 124545 | 
| 47 | 15 astropy/modeling/bounding_box.py | 1041 | 1106| 330 | 41670 | 124545 | 
| 48 | 16 astropy/coordinates/baseframe.py | 1720 | 1762| 328 | 41998 | 140711 | 
| 49 | 16 astropy/io/misc/asdf/tags/transform/functional_models.py | 787 | 797| 146 | 42144 | 140711 | 
| 50 | 16 astropy/io/misc/asdf/tags/transform/tabular.py | 48 | 90| 377 | 42521 | 140711 | 
| 51 | 17 astropy/coordinates/sky_coordinate.py | 1150 | 1190| 329 | 42850 | 160319 | 
| 52 | 17 astropy/modeling/fitting.py | 1488 | 1497| 137 | 42987 | 160319 | 
| 53 | 17 astropy/modeling/convolution.py | 68 | 84| 125 | 43112 | 160319 | 
| 54 | 17 astropy/modeling/fitting.py | 381 | 447| 739 | 43851 | 160319 | 
| 55 | 17 astropy/modeling/mappings.py | 243 | 327| 549 | 44400 | 160319 | 
| 56 | 17 astropy/modeling/functional_models.py | 431 | 449| 225 | 44625 | 160319 | 
| 57 | 17 astropy/modeling/core.py | 16 | 86| 545 | 45170 | 160319 | 
| 58 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 106 | 116| 133 | 45303 | 160319 | 
| 59 | **17 astropy/modeling/separable.py** | 130 | 168| 286 | 45589 | 160319 | 
| 60 | 17 astropy/io/misc/asdf/tags/transform/functional_models.py | 217 | 243| 241 | 45830 | 160319 | 
| 61 | 17 astropy/modeling/tabular.py | 139 | 170| 226 | 46056 | 160319 | 
| 62 | 17 astropy/modeling/bounding_box.py | 1314 | 1332| 173 | 46229 | 160319 | 
| 63 | 17 astropy/modeling/bounding_box.py | 1108 | 1140| 210 | 46439 | 160319 | 
| 64 | 17 astropy/modeling/bounding_box.py | 934 | 952| 130 | 46569 | 160319 | 
| 65 | 18 astropy/convolution/utils.py | 162 | 196| 402 | 46971 | 163268 | 
| 66 | 19 astropy/utils/masked/function_helpers.py | 156 | 230| 630 | 47601 | 172041 | 
| 67 | 19 astropy/modeling/fitting.py | 1499 | 1542| 378 | 47979 | 172041 | 
| 68 | 19 astropy/io/misc/asdf/tags/transform/functional_models.py | 586 | 612| 249 | 48228 | 172041 | 
| 69 | 19 astropy/convolution/utils.py | 2 | 31| 146 | 48374 | 172041 | 
| 70 | 19 astropy/io/misc/asdf/tags/transform/functional_models.py | 573 | 583| 133 | 48507 | 172041 | 
| 71 | 20 astropy/io/misc/asdf/tags/transform/physical_models.py | 4 | 38| 249 | 48756 | 172799 | 
| 72 | 20 astropy/modeling/functional_models.py | 1960 | 1971| 170 | 48926 | 172799 | 
| 73 | 20 astropy/io/misc/asdf/tags/transform/functional_models.py | 4 | 20| 263 | 49189 | 172799 | 
| 74 | 20 astropy/io/misc/asdf/tags/transform/functional_models.py | 140 | 149| 125 | 49314 | 172799 | 
| 75 | 20 astropy/coordinates/baseframe.py | 1678 | 1718| 350 | 49664 | 172799 | 
| 76 | 20 astropy/coordinates/sky_coordinate.py | 1097 | 1148| 424 | 50088 | 172799 | 
| 77 | 20 astropy/modeling/fitting.py | 135 | 162| 154 | 50242 | 172799 | 
| 78 | 20 astropy/modeling/functional_models.py | 3064 | 3081| 211 | 50453 | 172799 | 
| 79 | 20 astropy/io/misc/asdf/tags/transform/functional_models.py | 360 | 368| 115 | 50568 | 172799 | 
| 80 | 21 astropy/io/misc/asdf/tags/transform/projections.py | 35 | 46| 113 | 50681 | 175240 | 
| 81 | 21 astropy/io/misc/asdf/tags/transform/functional_models.py | 316 | 339| 219 | 50900 | 175240 | 
| 82 | 21 astropy/io/misc/asdf/tags/transform/functional_models.py | 403 | 423| 194 | 51094 | 175240 | 
| 83 | 21 astropy/io/misc/asdf/tags/transform/functional_models.py | 658 | 682| 200 | 51294 | 175240 | 
| 84 | 22 astropy/io/misc/asdf/tags/transform/polynomial.py | 174 | 192| 204 | 51498 | 178041 | 
| 85 | 23 astropy/convolution/convolve.py | 930 | 961| 274 | 51772 | 189363 | 
| 86 | 23 astropy/modeling/polynomial.py | 1544 | 1557| 185 | 51957 | 189363 | 
| 87 | 23 astropy/io/misc/asdf/tags/transform/basic.py | 206 | 233| 167 | 52124 | 189363 | 
| 88 | 23 astropy/io/misc/asdf/tags/transform/projections.py | 4 | 33| 211 | 52335 | 189363 | 
| 89 | 23 astropy/modeling/fitting.py | 487 | 568| 712 | 53047 | 189363 | 
| 90 | 23 astropy/modeling/fitting.py | 1724 | 1742| 145 | 53192 | 189363 | 
| 91 | 24 astropy/wcs/wcsapi/fitswcs.py | 294 | 319| 273 | 53465 | 195195 | 
| 92 | 24 astropy/modeling/functional_models.py | 2158 | 2168| 163 | 53628 | 195195 | 


## Patch

```diff
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
         cright = _coord_matrix(right, 'right', noutp)
     else:
         cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
 
     return np.hstack([cleft, cright])
 

```

## Test Patch

```diff
diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py
--- a/astropy/modeling/tests/test_separable.py
+++ b/astropy/modeling/tests/test_separable.py
@@ -28,6 +28,13 @@
 p1 = models.Polynomial1D(1, name='p1')
 
 
+cm_4d_expected = (np.array([False, False, True, True]),
+                  np.array([[True,  True,  False, False],
+                            [True,  True,  False, False],
+                            [False, False, True,  False],
+                            [False, False, False, True]]))
+
+
 compound_models = {
     'cm1': (map3 & sh1 | rot & sh1 | sh1 & sh2 & sh1,
             (np.array([False, False, True]),
@@ -52,7 +59,17 @@
     'cm7': (map2 | p2 & sh1,
             (np.array([False, True]),
              np.array([[True, False], [False, True]]))
-            )
+            ),
+    'cm8': (rot & (sh1 & sh2), cm_4d_expected),
+    'cm9': (rot & sh1 & sh2, cm_4d_expected),
+    'cm10': ((rot & sh1) & sh2, cm_4d_expected),
+    'cm11': (rot & sh1 & (scl1 & scl2),
+             (np.array([False, False, True, True, True]),
+              np.array([[True,  True,  False, False, False],
+                        [True,  True,  False, False, False],
+                        [False, False, True,  False, False],
+                        [False, False, False, True,  False],
+                        [False, False, False, False, True]]))),
 }
 
 

```


## Code snippets

### 1 - astropy/modeling/separable.py:

Start line: 18, End line: 63

```python
import numpy as np

from .core import Model, ModelDefinitionError, CompoundModel
from .mappings import Mapping


__all__ = ["is_separable", "separability_matrix"]


def is_separable(transform):
    """
    A separability test for the outputs of a transform.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    is_separable : ndarray
        A boolean array with size ``transform.n_outputs`` where
        each element indicates whether the output is independent
        and the result of a separable transform.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> is_separable(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([ True,  True]...)
    >>> is_separable(Shift(1) & Shift(2) | Rotation2D(2))
        array([False, False]...)
    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([False, False]...)
    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([ True,  True,  True,  True]...)

    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        is_separable = np.array([False] * transform.n_outputs).T
        return is_separable
    separable_matrix = _separable(transform)
    is_separable = separable_matrix.sum(1)
    is_separable = np.where(is_separable != 1, False, True)
    return is_separable
```
### 2 - astropy/modeling/separable.py:

Start line: 66, End line: 102

```python
def separability_matrix(transform):
    """
    Compute the correlation between outputs and inputs.

    Parameters
    ----------
    transform : `~astropy.modeling.core.Model`
        A (compound) model.

    Returns
    -------
    separable_matrix : ndarray
        A boolean correlation matrix of shape (n_outputs, n_inputs).
        Indicates the dependence of outputs on inputs. For completely
        independent outputs, the diagonal elements are True and
        off-diagonal elements are False.

    Examples
    --------
    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D
    >>> separability_matrix(Shift(1) & Shift(2) | Scale(1) & Scale(2))
        array([[ True, False], [False,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Rotation2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) | \
        Polynomial2D(1) & Polynomial2D(2))
        array([[ True,  True], [ True,  True]]...)
    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))
        array([[ True, False], [False,  True], [ True, False], [False,  True]]...)

    """
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        return np.ones((transform.n_outputs, transform.n_inputs),
                       dtype=np.bool_)
    separable_matrix = _separable(transform)
    separable_matrix = np.where(separable_matrix != 0, True, False)
    return separable_matrix
```
### 3 - astropy/modeling/separable.py:

Start line: 290, End line: 318

```python
def _separable(transform):
    """
    Calculate the separability of outputs.

    Parameters
    ----------
    transform : `astropy.modeling.Model`
        A transform (usually a compound model).

    Returns :
    is_separable : ndarray of dtype np.bool
        An array of shape (transform.n_outputs,) of boolean type
        Each element represents the separablity of the corresponding output.
    """
    if (transform_matrix := transform._calculate_separability_matrix()) is not NotImplemented:
        return transform_matrix
    elif isinstance(transform, CompoundModel):
        sepleft = _separable(transform.left)
        sepright = _separable(transform.right)
        return _operators[transform.op](sepleft, sepright)
    elif isinstance(transform, Model):
        return _coord_matrix(transform, 'left', transform.n_outputs)


# Maps modeling operators to a function computing and represents the
# relationship of axes as an array of 0-es and 1-s
_operators = {'&': _cstack, '|': _cdot, '+': _arith_oper, '-': _arith_oper,
              '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}
```
### 4 - astropy/modeling/separable.py:

Start line: 171, End line: 216

```python
def _coord_matrix(model, pos, noutp):
    """
    Create an array representing inputs and outputs of a simple model.

    The array has a shape (noutp, model.n_inputs).

    Parameters
    ----------
    model : `astropy.modeling.Model`
        model
    pos : str
        Position of this model in the expression tree.
        One of ['left', 'right'].
    noutp : int
        Number of outputs of the compound model of which the input model
        is a left or right child.

    """
    if isinstance(model, Mapping):
        axes = []
        for i in model.mapping:
            axis = np.zeros((model.n_inputs,))
            axis[i] = 1
            axes.append(axis)
        m = np.vstack(axes)
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[: model.n_outputs, :model.n_inputs] = m
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = m
        return mat
    if not model.separable:
        # this does not work for more than 2 coordinates
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[:model.n_outputs, : model.n_inputs] = 1
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = 1
    else:
        mat = np.zeros((noutp, model.n_inputs))

        for i in range(model.n_inputs):
            mat[i, i] = 1
        if pos == 'right':
            mat = np.roll(mat, (noutp - model.n_outputs))
    return mat
```
### 5 - astropy/modeling/core.py:

Start line: 4071, End line: 4474

```python
def fix_inputs(modelinstance, values, bounding_boxes=None, selector_args=None):
    """
    This function creates a compound model with one or more of the input
    values of the input model assigned fixed values (scalar or array).

    Parameters
    ----------
    modelinstance : `~astropy.modeling.Model` instance
        This is the model that one or more of the
        model input values will be fixed to some constant value.
    values : dict
        A dictionary where the key identifies which input to fix
        and its value is the value to fix it at. The key may either be the
        name of the input or a number reflecting its order in the inputs.

    Examples
    --------

    >>> from astropy.modeling.models import Gaussian2D
    >>> g = Gaussian2D(1, 2, 3, 4, 5)
    >>> gv = fix_inputs(g, {0: 2.5})

    Results in a 1D function equivalent to Gaussian2D(1, 2, 3, 4, 5)(x=2.5, y)
    """
    model = CompoundModel('fix_inputs', modelinstance, values)
    if bounding_boxes is not None:
        if selector_args is None:
            selector_args = tuple([(key, True) for key in values.keys()])
        bbox = CompoundBoundingBox.validate(modelinstance, bounding_boxes, selector_args)
        _selector = bbox.selector_args.get_fixed_values(modelinstance, values)

        new_bbox = bbox[_selector]
        new_bbox = new_bbox.__class__.validate(model, new_bbox)

        model.bounding_box = new_bbox
    return model


def bind_bounding_box(modelinstance, bounding_box, ignored=None, order='C'):
    """
    Set a validated bounding box to a model instance.

    Parameters
    ----------
    modelinstance : `~astropy.modeling.Model` instance
        This is the model that the validated bounding box will be set on.
    bounding_box : tuple
        A bounding box tuple, see :ref:`astropy:bounding-boxes` for details
    ignored : list
        List of the inputs to be ignored by the bounding box.
    order : str, optional
        The ordering of the bounding box tuple, can be either ``'C'`` or
        ``'F'``.
    """
    modelinstance.bounding_box = ModelBoundingBox.validate(modelinstance,
                                                           bounding_box,
                                                           ignored=ignored,
                                                           order=order)


def bind_compound_bounding_box(modelinstance, bounding_boxes, selector_args,
                               create_selector=None, ignored=None, order='C'):
    """
    Add a validated compound bounding box to a model instance.

    Parameters
    ----------
    modelinstance : `~astropy.modeling.Model` instance
        This is the model that the validated compound bounding box will be set on.
    bounding_boxes : dict
        A dictionary of bounding box tuples, see :ref:`astropy:bounding-boxes`
        for details.
    selector_args : list
        List of selector argument tuples to define selection for compound
        bounding box, see :ref:`astropy:bounding-boxes` for details.
    create_selector : callable, optional
        An optional callable with interface (selector_value, model) which
        can generate a bounding box based on a selector value and model if
        there is no bounding box in the compound bounding box listed under
        that selector value. Default is ``None``, meaning new bounding
        box entries will not be automatically generated.
    ignored : list
        List of the inputs to be ignored by the bounding box.
    order : str, optional
        The ordering of the bounding box tuple, can be either ``'C'`` or
        ``'F'``.
    """
    modelinstance.bounding_box = CompoundBoundingBox.validate(modelinstance,
                                                              bounding_boxes, selector_args,
                                                              create_selector=create_selector,
                                                              ignored=ignored,
                                                              order=order)


def custom_model(*args, fit_deriv=None):
    """
    Create a model from a user defined function. The inputs and parameters of
    the model will be inferred from the arguments of the function.

    This can be used either as a function or as a decorator.  See below for
    examples of both usages.

    The model is separable only if there is a single input.

    .. note::

        All model parameters have to be defined as keyword arguments with
        default values in the model function.  Use `None` as a default argument
        value if you do not want to have a default value for that parameter.

        The standard settable model properties can be configured by default
        using keyword arguments matching the name of the property; however,
        these values are not set as model "parameters". Moreover, users
        cannot use keyword arguments matching non-settable model properties,
        with the exception of ``n_outputs`` which should be set to the number of
        outputs of your function.

    Parameters
    ----------
    func : function
        Function which defines the model.  It should take N positional
        arguments where ``N`` is dimensions of the model (the number of
        independent variable in the model), and any number of keyword arguments
        (the parameters).  It must return the value of the model (typically as
        an array, but can also be a scalar for scalar inputs).  This
        corresponds to the `~astropy.modeling.Model.evaluate` method.
    fit_deriv : function, optional
        Function which defines the Jacobian derivative of the model. I.e., the
        derivative with respect to the *parameters* of the model.  It should
        have the same argument signature as ``func``, but should return a
        sequence where each element of the sequence is the derivative
        with respect to the corresponding argument. This corresponds to the
        :meth:`~astropy.modeling.FittableModel.fit_deriv` method.

    Examples
    --------
    Define a sinusoidal model function as a custom 1D model::

        >>> from astropy.modeling.models import custom_model
        >>> import numpy as np
        >>> def sine_model(x, amplitude=1., frequency=1.):
        ...     return amplitude * np.sin(2 * np.pi * frequency * x)
        >>> def sine_deriv(x, amplitude=1., frequency=1.):
        ...     return 2 * np.pi * amplitude * np.cos(2 * np.pi * frequency * x)
        >>> SineModel = custom_model(sine_model, fit_deriv=sine_deriv)

    Create an instance of the custom model and evaluate it::

        >>> model = SineModel()
        >>> model(0.25)
        1.0

    This model instance can now be used like a usual astropy model.

    The next example demonstrates a 2D Moffat function model, and also
    demonstrates the support for docstrings (this example could also include
    a derivative, but it has been omitted for simplicity)::

        >>> @custom_model
        ... def Moffat2D(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, gamma=1.0,
        ...            alpha=1.0):
        ...     \"\"\"Two dimensional Moffat function.\"\"\"
        ...     rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2
        ...     return amplitude * (1 + rr_gg) ** (-alpha)
        ...
        >>> print(Moffat2D.__doc__)
        Two dimensional Moffat function.
        >>> model = Moffat2D()
        >>> model(1, 1)  # doctest: +FLOAT_CMP
        0.3333333333333333
    """

    if len(args) == 1 and callable(args[0]):
        return _custom_model_wrapper(args[0], fit_deriv=fit_deriv)
    elif not args:
        return functools.partial(_custom_model_wrapper, fit_deriv=fit_deriv)
    else:
        raise TypeError(
            "{0} takes at most one positional argument (the callable/"
            "function to be turned into a model.  When used as a decorator "
            "it should be passed keyword arguments only (if "
            "any).".format(__name__))


def _custom_model_inputs(func):
    """
    Processes the inputs to the `custom_model`'s function into the appropriate
    categories.

    Parameters
    ----------
    func : callable

    Returns
    -------
    inputs : list
        list of evaluation inputs
    special_params : dict
        dictionary of model properties which require special treatment
    settable_params : dict
        dictionary of defaults for settable model properties
    params : dict
        dictionary of model parameters set by `custom_model`'s function
    """
    inputs, parameters = get_inputs_and_params(func)

    special = ['n_outputs']
    settable = [attr for attr, value in vars(Model).items()
                if isinstance(value, property) and value.fset is not None]
    properties = [attr for attr, value in vars(Model).items()
                  if isinstance(value, property) and value.fset is None and attr not in special]

    special_params = {}
    settable_params = {}
    params = {}
    for param in parameters:
        if param.name in special:
            special_params[param.name] = param.default
        elif param.name in settable:
            settable_params[param.name] = param.default
        elif param.name in properties:
            raise ValueError(f"Parameter '{param.name}' cannot be a model property: {properties}.")
        else:
            params[param.name] = param.default

    return inputs, special_params, settable_params, params


def _custom_model_wrapper(func, fit_deriv=None):
    """
    Internal implementation `custom_model`.

    When `custom_model` is called as a function its arguments are passed to
    this function, and the result of this function is returned.

    When `custom_model` is used as a decorator a partial evaluation of this
    function is returned by `custom_model`.
    """

    if not callable(func):
        raise ModelDefinitionError(
            "func is not callable; it must be a function or other callable "
            "object")

    if fit_deriv is not None and not callable(fit_deriv):
        raise ModelDefinitionError(
            "fit_deriv not callable; it must be a function or other "
            "callable object")

    model_name = func.__name__

    inputs, special_params, settable_params, params = _custom_model_inputs(func)

    if (fit_deriv is not None and
            len(fit_deriv.__defaults__) != len(params)):
        raise ModelDefinitionError("derivative function should accept "
                                   "same number of parameters as func.")

    params = {param: Parameter(param, default=default)
              for param, default in params.items()}

    mod = find_current_module(2)
    if mod:
        modname = mod.__name__
    else:
        modname = '__main__'

    members = {
        '__module__': str(modname),
        '__doc__': func.__doc__,
        'n_inputs': len(inputs),
        'n_outputs': special_params.pop('n_outputs', 1),
        'evaluate': staticmethod(func),
        '_settable_properties': settable_params
    }

    if fit_deriv is not None:
        members['fit_deriv'] = staticmethod(fit_deriv)

    members.update(params)

    cls = type(model_name, (FittableModel,), members)
    cls._separable = True if (len(inputs) == 1) else False
    return cls


def render_model(model, arr=None, coords=None):
    """
    Evaluates a model on an input array. Evaluation is limited to
    a bounding box if the `Model.bounding_box` attribute is set.

    Parameters
    ----------
    model : `Model`
        Model to be evaluated.
    arr : `numpy.ndarray`, optional
        Array on which the model is evaluated.
    coords : array-like, optional
        Coordinate arrays mapping to ``arr``, such that
        ``arr[coords] == arr``.

    Returns
    -------
    array : `numpy.ndarray`
        The model evaluated on the input ``arr`` or a new array from
        ``coords``.
        If ``arr`` and ``coords`` are both `None`, the returned array is
        limited to the `Model.bounding_box` limits. If
        `Model.bounding_box` is `None`, ``arr`` or ``coords`` must be passed.

    Examples
    --------
    :ref:`astropy:bounding-boxes`
    """

    bbox = model.bounding_box

    if (coords is None) & (arr is None) & (bbox is None):
        raise ValueError('If no bounding_box is set,'
                         'coords or arr must be input.')

    # for consistent indexing
    if model.n_inputs == 1:
        if coords is not None:
            coords = [coords]
        if bbox is not None:
            bbox = [bbox]

    if arr is not None:
        arr = arr.copy()
        # Check dimensions match model
        if arr.ndim != model.n_inputs:
            raise ValueError('number of array dimensions inconsistent with '
                             'number of model inputs.')
    if coords is not None:
        # Check dimensions match arr and model
        coords = np.array(coords)
        if len(coords) != model.n_inputs:
            raise ValueError('coordinate length inconsistent with the number '
                             'of model inputs.')
        if arr is not None:
            if coords[0].shape != arr.shape:
                raise ValueError('coordinate shape inconsistent with the '
                                 'array shape.')
        else:
            arr = np.zeros(coords[0].shape)

    if bbox is not None:
        # assures position is at center pixel, important when using add_array
        pd = pos, delta = np.array([(np.mean(bb), np.ceil((bb[1] - bb[0]) / 2))
                                    for bb in bbox]).astype(int).T

        if coords is not None:
            sub_shape = tuple(delta * 2 + 1)
            sub_coords = np.array([extract_array(c, sub_shape, pos)
                                   for c in coords])
        else:
            limits = [slice(p - d, p + d + 1, 1) for p, d in pd.T]
            sub_coords = np.mgrid[limits]

        sub_coords = sub_coords[::-1]

        if arr is None:
            arr = model(*sub_coords)
        else:
            try:
                arr = add_array(arr, model(*sub_coords), pos)
            except ValueError:
                raise ValueError('The `bounding_box` is larger than the input'
                                 ' arr in one or more dimensions. Set '
                                 '`model.bounding_box = None`.')
    else:

        if coords is None:
            im_shape = arr.shape
            limits = [slice(i) for i in im_shape]
            coords = np.mgrid[limits]

        arr += model(*coords[::-1])

    return arr


def hide_inverse(model):
    """
    This is a convenience function intended to disable automatic generation
    of the inverse in compound models by disabling one of the constituent
    model's inverse. This is to handle cases where user provided inverse
    functions are not compatible within an expression.

    Example:
        compound_model.inverse = hide_inverse(m1) + m2 + m3

    This will insure that the defined inverse itself won't attempt to
    build its own inverse, which would otherwise fail in this example
    (e.g., m = m1 + m2 + m3 happens to raises an exception for this
    reason.)

    Note that this permanently disables it. To prevent that either copy
    the model or restore the inverse later.
    """
    del model.inverse
    return model
```
### 6 - astropy/modeling/separable.py:

Start line: 105, End line: 127

```python
def _compute_n_outputs(left, right):
    """
    Compute the number of outputs of two models.

    The two models are the left and right model to an operation in
    the expression tree of a compound model.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    """
    if isinstance(left, Model):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if isinstance(right, Model):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    noutp = lnout + rnout
    return noutp
```
### 7 - astropy/modeling/core.py:

Start line: 2466, End line: 3284

```python
class Model(metaclass=_ModelMeta):

    def _initialize_parameters(self, args, kwargs):
        # ... other code

        if kwargs:
            # If any keyword arguments were left over at this point they are
            # invalid--the base class should only be passed the parameter
            # values, constraints, and param_dim
            for kwarg in kwargs:
                # Just raise an error on the first unrecognized argument
                raise TypeError(
                    '{0}.__init__() got an unrecognized parameter '
                    '{1!r}'.format(self.__class__.__name__, kwarg))

        # Determine the number of model sets: If the model_set_axis is
        # None then there is just one parameter set; otherwise it is determined
        # by the size of that axis on the first parameter--if the other
        # parameters don't have the right number of axes or the sizes of their
        # model_set_axis don't match an error is raised
        if model_set_axis is not False and n_models != 1 and params:
            max_ndim = 0
            if model_set_axis < 0:
                min_ndim = abs(model_set_axis)
            else:
                min_ndim = model_set_axis + 1

            for name in self.param_names:
                value = getattr(self, name)
                param_ndim = np.ndim(value)
                if param_ndim < min_ndim:
                    raise InputParameterError(
                        "All parameter values must be arrays of dimension "
                        "at least {0} for model_set_axis={1} (the value "
                        "given for {2!r} is only {3}-dimensional)".format(
                            min_ndim, model_set_axis, name, param_ndim))

                max_ndim = max(max_ndim, param_ndim)

                if n_models is None:
                    # Use the dimensions of the first parameter to determine
                    # the number of model sets
                    n_models = value.shape[model_set_axis]
                elif value.shape[model_set_axis] != n_models:
                    raise InputParameterError(
                        "Inconsistent dimensions for parameter {0!r} for "
                        "{1} model sets.  The length of axis {2} must be the "
                        "same for all input parameter values".format(
                            name, n_models, model_set_axis))

            self._check_param_broadcast(max_ndim)
        else:
            if n_models is None:
                n_models = 1

            self._check_param_broadcast(None)

        self._n_models = n_models
        # now validate parameters
        for name in params:
            param = getattr(self, name)
            if param._validator is not None:
                param._validator(self, param.value)

    def _initialize_parameter_value(self, param_name, value):
        """Mostly deals with consistency checks and determining unit issues."""
        if isinstance(value, Parameter):
            self.__dict__[param_name] = value
            return
        param = getattr(self, param_name)
        # Use default if value is not provided
        if value is None:
            default = param.default
            if default is None:
                # No value was supplied for the parameter and the
                # parameter does not have a default, therefore the model
                # is underspecified
                raise TypeError("{0}.__init__() requires a value for parameter "
                                "{1!r}".format(self.__class__.__name__, param_name))
            value = default
            unit = param.unit
        else:
            if isinstance(value, Quantity):
                unit = value.unit
                value = value.value
            else:
                unit = None
        if unit is None and param.unit is not None:
            raise InputParameterError(
                "{0}.__init__() requires a Quantity for parameter "
                "{1!r}".format(self.__class__.__name__, param_name))
        param._unit = unit
        param.internal_unit = None
        if param._setter is not None:
            if unit is not None:
                _val = param._setter(value * unit)
            else:
                _val = param._setter(value)
            if isinstance(_val, Quantity):
                param.internal_unit = _val.unit
                param._internal_value = np.array(_val.value)
            else:
                param.internal_unit = None
                param._internal_value = np.array(_val)
        else:
            param._value = np.array(value)

    def _initialize_slices(self):

        param_metrics = self._param_metrics
        total_size = 0

        for name in self.param_names:
            param = getattr(self, name)
            value = param.value
            param_size = np.size(value)
            param_shape = np.shape(value)
            param_slice = slice(total_size, total_size + param_size)
            param_metrics[name]['slice'] = param_slice
            param_metrics[name]['shape'] = param_shape
            param_metrics[name]['size'] = param_size
            total_size += param_size
        self._parameters = np.empty(total_size, dtype=np.float64)

    def _parameters_to_array(self):
        # Now set the parameter values (this will also fill
        # self._parameters)
        param_metrics = self._param_metrics
        for name in self.param_names:
            param = getattr(self, name)
            value = param.value
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            self._parameters[param_metrics[name]['slice']] = value.ravel()

        # Finally validate all the parameters; we do this last so that
        # validators that depend on one of the other parameters' values will
        # work

    def _array_to_parameters(self):
        param_metrics = self._param_metrics
        for name in self.param_names:
            param = getattr(self, name)
            value = self._parameters[param_metrics[name]['slice']]
            value.shape = param_metrics[name]['shape']
            param.value = value

    def _check_param_broadcast(self, max_ndim):
        """
        This subroutine checks that all parameter arrays can be broadcast
        against each other, and determines the shapes parameters must have in
        order to broadcast correctly.

        If model_set_axis is None this merely checks that the parameters
        broadcast and returns an empty dict if so.  This mode is only used for
        single model sets.
        """
        all_shapes = []
        model_set_axis = self._model_set_axis

        for name in self.param_names:
            param = getattr(self, name)
            value = param.value
            param_shape = np.shape(value)
            param_ndim = len(param_shape)
            if max_ndim is not None and param_ndim < max_ndim:
                # All arrays have the same number of dimensions up to the
                # model_set_axis dimension, but after that they may have a
                # different number of trailing axes.  The number of trailing
                # axes must be extended for mutual compatibility.  For example
                # if max_ndim = 3 and model_set_axis = 0, an array with the
                # shape (2, 2) must be extended to (2, 1, 2).  However, an
                # array with shape (2,) is extended to (2, 1).
                new_axes = (1,) * (max_ndim - param_ndim)

                if model_set_axis < 0:
                    # Just need to prepend axes to make up the difference
                    broadcast_shape = new_axes + param_shape
                else:
                    broadcast_shape = (param_shape[:model_set_axis + 1] +
                                       new_axes +
                                       param_shape[model_set_axis + 1:])
                self._param_metrics[name]['broadcast_shape'] = broadcast_shape
                all_shapes.append(broadcast_shape)
            else:
                all_shapes.append(param_shape)

        # Now check mutual broadcastability of all shapes
        try:
            check_broadcast(*all_shapes)
        except IncompatibleShapeError as exc:
            shape_a, shape_a_idx, shape_b, shape_b_idx = exc.args
            param_a = self.param_names[shape_a_idx]
            param_b = self.param_names[shape_b_idx]

            raise InputParameterError(
                "Parameter {0!r} of shape {1!r} cannot be broadcast with "
                "parameter {2!r} of shape {3!r}.  All parameter arrays "
                "must have shapes that are mutually compatible according "
                "to the broadcasting rules.".format(param_a, shape_a,
                                                    param_b, shape_b))

    def _param_sets(self, raw=False, units=False):
        """
        Implementation of the Model.param_sets property.

        This internal implementation has a ``raw`` argument which controls
        whether or not to return the raw parameter values (i.e. the values that
        are actually stored in the ._parameters array, as opposed to the values
        displayed to users.  In most cases these are one in the same but there
        are currently a few exceptions.

        Note: This is notably an overcomplicated device and may be removed
        entirely in the near future.
        """

        values = []
        shapes = []
        for name in self.param_names:
            param = getattr(self, name)

            if raw and param._setter:
                value = param._internal_value
            else:
                value = param.value

            broadcast_shape = self._param_metrics[name].get('broadcast_shape')
            if broadcast_shape is not None:
                value = value.reshape(broadcast_shape)

            shapes.append(np.shape(value))

            if len(self) == 1:
                # Add a single param set axis to the parameter's value (thus
                # converting scalars to shape (1,) array values) for
                # consistency
                value = np.array([value])

            if units:
                if raw and param.internal_unit is not None:
                    unit = param.internal_unit
                else:
                    unit = param.unit
                if unit is not None:
                    value = Quantity(value, unit)

            values.append(value)

        if len(set(shapes)) != 1 or units:
            # If the parameters are not all the same shape, converting to an
            # array is going to produce an object array
            # However the way Numpy creates object arrays is tricky in that it
            # will recurse into array objects in the list and break them up
            # into separate objects.  Doing things this way ensures a 1-D
            # object array the elements of which are the individual parameter
            # arrays.  There's not much reason to do this over returning a list
            # except for consistency
            psets = np.empty(len(values), dtype=object)
            psets[:] = values
            return psets

        return np.array(values)

    def _format_repr(self, args=[], kwargs={}, defaults={}):
        """
        Internal implementation of ``__repr__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__repr__`` while keeping the same basic
        formatting.
        """

        parts = [repr(a) for a in args]

        parts.extend(
            f"{name}={param_repr_oneline(getattr(self, name))}"
            for name in self.param_names)

        if self.name is not None:
            parts.append(f'name={self.name!r}')

        for kwarg, value in kwargs.items():
            if kwarg in defaults and defaults[kwarg] == value:
                continue
            parts.append(f'{kwarg}={value!r}')

        if len(self) > 1:
            parts.append(f"n_models={len(self)}")

        return f"<{self.__class__.__name__}({', '.join(parts)})>"

    def _format_str(self, keywords=[], defaults={}):
        """
        Internal implementation of ``__str__``.

        This is separated out for ease of use by subclasses that wish to
        override the default ``__str__`` while keeping the same basic
        formatting.
        """

        default_keywords = [
            ('Model', self.__class__.__name__),
            ('Name', self.name),
            ('Inputs', self.inputs),
            ('Outputs', self.outputs),
            ('Model set size', len(self))
        ]

        parts = [f'{keyword}: {value}'
                 for keyword, value in default_keywords
                 if value is not None]

        for keyword, value in keywords:
            if keyword.lower() in defaults and defaults[keyword.lower()] == value:
                continue
            parts.append(f'{keyword}: {value}')
        parts.append('Parameters:')

        if len(self) == 1:
            columns = [[getattr(self, name).value]
                       for name in self.param_names]
        else:
            columns = [getattr(self, name).value
                       for name in self.param_names]

        if columns:
            param_table = Table(columns, names=self.param_names)
            # Set units on the columns
            for name in self.param_names:
                param_table[name].unit = getattr(self, name).unit
            parts.append(indent(str(param_table), width=4))

        return '\n'.join(parts)


class FittableModel(Model):
    """
    Base class for models that can be fitted using the built-in fitting
    algorithms.
    """

    linear = False
    # derivative with respect to parameters
    fit_deriv = None
    """
    Function (similar to the model's `~Model.evaluate`) to compute the
    derivatives of the model with respect to its parameters, for use by fitting
    algorithms.  In other words, this computes the Jacobian matrix with respect
    to the model's parameters.
    """
    # Flag that indicates if the model derivatives with respect to parameters
    # are given in columns or rows
    col_fit_deriv = True
    fittable = True


class Fittable1DModel(FittableModel):
    """
    Base class for one-dimensional fittable models.

    This class provides an easier interface to defining new models.
    Examples can be found in `astropy.modeling.functional_models`.
    """
    n_inputs = 1
    n_outputs = 1
    _separable = True


class Fittable2DModel(FittableModel):
    """
    Base class for two-dimensional fittable models.

    This class provides an easier interface to defining new models.
    Examples can be found in `astropy.modeling.functional_models`.
    """

    n_inputs = 2
    n_outputs = 1


def _make_arithmetic_operator(oper):
    # We don't bother with tuple unpacking here for efficiency's sake, but for
    # documentation purposes:
    #
    #     f_eval, f_n_inputs, f_n_outputs = f
    #
    # and similarly for g
    def op(f, g):
        return (make_binary_operator_eval(oper, f[0], g[0]), f[1], f[2])

    return op


def _composition_operator(f, g):
    # We don't bother with tuple unpacking here for efficiency's sake, but for
    # documentation purposes:
    #
    #     f_eval, f_n_inputs, f_n_outputs = f
    #
    # and similarly for g
    return (lambda inputs, params: g[0](f[0](inputs, params), params),
            f[1], g[2])


def _join_operator(f, g):
    # We don't bother with tuple unpacking here for efficiency's sake, but for
    # documentation purposes:
    #
    #     f_eval, f_n_inputs, f_n_outputs = f
    #
    # and similarly for g
    return (lambda inputs, params: (f[0](inputs[:f[1]], params) +
                                    g[0](inputs[f[1]:], params)),
            f[1] + g[1], f[2] + g[2])


BINARY_OPERATORS = {
    '+': _make_arithmetic_operator(operator.add),
    '-': _make_arithmetic_operator(operator.sub),
    '*': _make_arithmetic_operator(operator.mul),
    '/': _make_arithmetic_operator(operator.truediv),
    '**': _make_arithmetic_operator(operator.pow),
    '|': _composition_operator,
    '&': _join_operator
}

SPECIAL_OPERATORS = _SpecialOperatorsDict()


def _add_special_operator(sop_name, sop):
    return SPECIAL_OPERATORS.add(sop_name, sop)


class CompoundModel(Model):
    '''
    Base class for compound models.

    While it can be used directly, the recommended way
    to combine models is through the model operators.
    '''

    def __init__(self, op, left, right, name=None):
        self.__dict__['_param_names'] = None
        self._n_submodels = None
        self.op = op
        self.left = left
        self.right = right
        self._bounding_box = None
        self._user_bounding_box = None
        self._leaflist = None
        self._tdict = None
        self._parameters = None
        self._parameters_ = None
        self._param_metrics = None

        if op != 'fix_inputs' and len(left) != len(right):
            raise ValueError(
                'Both operands must have equal values for n_models')
        self._n_models = len(left)

        if op != 'fix_inputs' and ((left.model_set_axis != right.model_set_axis)
                                   or left.model_set_axis):  # not False and not 0
            raise ValueError("model_set_axis must be False or 0 and consistent for operands")
        self._model_set_axis = left.model_set_axis

        if op in ['+', '-', '*', '/', '**'] or op in SPECIAL_OPERATORS:
            if (left.n_inputs != right.n_inputs) or \
               (left.n_outputs != right.n_outputs):
                raise ModelDefinitionError(
                    'Both operands must match numbers of inputs and outputs')
            self.n_inputs = left.n_inputs
            self.n_outputs = left.n_outputs
            self.inputs = left.inputs
            self.outputs = left.outputs
        elif op == '&':
            self.n_inputs = left.n_inputs + right.n_inputs
            self.n_outputs = left.n_outputs + right.n_outputs
            self.inputs = combine_labels(left.inputs, right.inputs)
            self.outputs = combine_labels(left.outputs, right.outputs)
        elif op == '|':
            if left.n_outputs != right.n_inputs:
                raise ModelDefinitionError(
                    "Unsupported operands for |: {0} (n_inputs={1}, "
                    "n_outputs={2}) and {3} (n_inputs={4}, n_outputs={5}); "
                    "n_outputs for the left-hand model must match n_inputs "
                    "for the right-hand model.".format(
                        left.name, left.n_inputs, left.n_outputs, right.name,
                        right.n_inputs, right.n_outputs))

            self.n_inputs = left.n_inputs
            self.n_outputs = right.n_outputs
            self.inputs = left.inputs
            self.outputs = right.outputs
        elif op == 'fix_inputs':
            if not isinstance(left, Model):
                raise ValueError('First argument to "fix_inputs" must be an instance of an astropy Model.')
            if not isinstance(right, dict):
                raise ValueError('Expected a dictionary for second argument of "fix_inputs".')

            # Dict keys must match either possible indices
            # for model on left side, or names for inputs.
            self.n_inputs = left.n_inputs - len(right)
            # Assign directly to the private attribute (instead of using the setter)
            # to avoid asserting the new number of outputs matches the old one.
            self._outputs = left.outputs
            self.n_outputs = left.n_outputs
            newinputs = list(left.inputs)
            keys = right.keys()
            input_ind = []
            for key in keys:
                if np.issubdtype(type(key), np.integer):
                    if key >= left.n_inputs or key < 0:
                        raise ValueError(
                            'Substitution key integer value '
                            'not among possible input choices.')
                    if key in input_ind:
                        raise ValueError("Duplicate specification of "
                                         "same input (index/name).")
                    input_ind.append(key)
                elif isinstance(key, str):
                    if key not in left.inputs:
                        raise ValueError(
                            'Substitution key string not among possible '
                            'input choices.')
                    # Check to see it doesn't match positional
                    # specification.
                    ind = left.inputs.index(key)
                    if ind in input_ind:
                        raise ValueError("Duplicate specification of "
                                         "same input (index/name).")
                    input_ind.append(ind)
            # Remove substituted inputs
            input_ind.sort()
            input_ind.reverse()
            for ind in input_ind:
                del newinputs[ind]
            self.inputs = tuple(newinputs)
            # Now check to see if the input model has bounding_box defined.
            # If so, remove the appropriate dimensions and set it for this
            # instance.
            try:
                self.bounding_box = \
                    self.left.bounding_box.fix_inputs(self, right)
            except NotImplementedError:
                pass

        else:
            raise ModelDefinitionError('Illegal operator: ', self.op)
        self.name = name
        self._fittable = None
        self.fit_deriv = None
        self.col_fit_deriv = None
        if op in ('|', '+', '-'):
            self.linear = left.linear and right.linear
        else:
            self.linear = False
        self.eqcons = []
        self.ineqcons = []
        self.n_left_params = len(self.left.parameters)
        self._map_parameters()

    def _get_left_inputs_from_args(self, args):
        return args[:self.left.n_inputs]

    def _get_right_inputs_from_args(self, args):
        op = self.op
        if op == '&':
            # Args expected to look like (*left inputs, *right inputs, *left params, *right params)
            return args[self.left.n_inputs: self.left.n_inputs + self.right.n_inputs]
        elif op == '|' or  op == 'fix_inputs':
            return None
        else:
            return args[:self.left.n_inputs]

    def _get_left_params_from_args(self, args):
        op = self.op
        if op == '&':
            # Args expected to look like (*left inputs, *right inputs, *left params, *right params)
            n_inputs = self.left.n_inputs + self.right.n_inputs
            return args[n_inputs: n_inputs + self.n_left_params]
        else:
            return args[self.left.n_inputs: self.left.n_inputs + self.n_left_params]

    def _get_right_params_from_args(self, args):
        op = self.op
        if op == 'fix_inputs':
            return None
        if op == '&':
            # Args expected to look like (*left inputs, *right inputs, *left params, *right params)
            return args[self.left.n_inputs + self.right.n_inputs + self.n_left_params:]
        else:
            return args[self.left.n_inputs + self.n_left_params:]

    def _get_kwarg_model_parameters_as_positional(self, args, kwargs):
        # could do it with inserts but rebuilding seems like simpilist way

        #TODO: Check if any param names are in kwargs maybe as an intersection of sets?
        if self.op == "&":
            new_args = list(args[:self.left.n_inputs + self.right.n_inputs])
            args_pos = self.left.n_inputs + self.right.n_inputs
        else:
            new_args = list(args[:self.left.n_inputs])
            args_pos = self.left.n_inputs

        for param_name in self.param_names:
            kw_value = kwargs.pop(param_name, None)
            if kw_value is not None:
                value = kw_value
            else:
                try:
                    value = args[args_pos]
                except IndexError:
                    raise IndexError("Missing parameter or input")

                args_pos += 1
            new_args.append(value)

        return new_args, kwargs

    def _apply_operators_to_value_lists(self, leftval, rightval, **kw):
        op = self.op
        if op == '+':
            return binary_operation(operator.add, leftval, rightval)
        elif op == '-':
            return binary_operation(operator.sub, leftval, rightval)
        elif op == '*':
            return binary_operation(operator.mul, leftval, rightval)
        elif op == '/':
            return binary_operation(operator.truediv, leftval, rightval)
        elif op == '**':
            return binary_operation(operator.pow, leftval, rightval)
        elif op == '&':
            if not isinstance(leftval, tuple):
                leftval = (leftval,)
            if not isinstance(rightval, tuple):
                rightval = (rightval,)
            return leftval + rightval
        elif op in SPECIAL_OPERATORS:
            return binary_operation(SPECIAL_OPERATORS[op], leftval, rightval)
        else:
            raise ModelDefinitionError('Unrecognized operator {op}')

    def evaluate(self, *args, **kw):
        op = self.op
        args, kw = self._get_kwarg_model_parameters_as_positional(args, kw)
        left_inputs = self._get_left_inputs_from_args(args)
        left_params = self._get_left_params_from_args(args)

        if op == 'fix_inputs':
            pos_index = dict(zip(self.left.inputs, range(self.left.n_inputs)))
            fixed_inputs = {
                key if np.issubdtype(type(key), np.integer) else pos_index[key]: value
                for key, value in self.right.items()
            }
            left_inputs = [
                fixed_inputs[ind] if ind in fixed_inputs.keys() else inp
                for ind, inp in enumerate(left_inputs)
            ]

        leftval = self.left.evaluate(*itertools.chain(left_inputs, left_params))

        if op == 'fix_inputs':
            return leftval

        right_inputs = self._get_right_inputs_from_args(args)
        right_params = self._get_right_params_from_args(args)

        if op == "|":
            if isinstance(leftval, tuple):
                return self.right.evaluate(*itertools.chain(leftval, right_params))
            else:
                return self.right.evaluate(leftval, *right_params)
        else:
            rightval = self.right.evaluate(*itertools.chain(right_inputs, right_params))

        return self._apply_operators_to_value_lists(leftval, rightval, **kw)

    @property
    def n_submodels(self):
        if self._leaflist is None:
            self._make_leaflist()
        return len(self._leaflist)

    @property
    def submodel_names(self):
        """ Return the names of submodels in a ``CompoundModel``."""
        if self._leaflist is None:
            self._make_leaflist()
        names = [item.name for item in self._leaflist]
        nonecount = 0
        newnames = []
        for item in names:
            if item is None:
                newnames.append(f'None_{nonecount}')
                nonecount += 1
            else:
                newnames.append(item)
        return tuple(newnames)

    def both_inverses_exist(self):
        '''
        if both members of this compound model have inverses return True
        '''
        warnings.warn(
            "CompoundModel.both_inverses_exist is deprecated. "
            "Use has_inverse instead.",
            AstropyDeprecationWarning
        )

        try:
            linv = self.left.inverse
            rinv = self.right.inverse
        except NotImplementedError:
            return False

        return True

    def _pre_evaluate(self, *args, **kwargs):
        """
        CompoundModel specific input setup that needs to occur prior to
            model evaluation.

        Note
        ----
            All of the _pre_evaluate for each component model will be
            performed at the time that the individual model is evaluated.
        """

        # If equivalencies are provided, necessary to map parameters and pass
        # the leaflist as a keyword input for use by model evaluation so that
        # the compound model input names can be matched to the model input
        # names.
        if 'equivalencies' in kwargs:
            # Restructure to be useful for the individual model lookup
            kwargs['inputs_map'] = [(value[0], (value[1], key)) for
                                    key, value in self.inputs_map().items()]

        # Setup actual model evaluation method
        def evaluate(_inputs):
            return self._evaluate(*_inputs, **kwargs)

        return evaluate, args, None, kwargs

    @property
    def _argnames(self):
        """No inputs should be used to determine input_shape when handling compound models"""
        return ()

    def _post_evaluate(self, inputs, outputs, broadcasted_shapes, with_bbox, **kwargs):
        """
        CompoundModel specific post evaluation processing of outputs

        Note
        ----
            All of the _post_evaluate for each component model will be
            performed at the time that the individual model is evaluated.
        """
        if self.get_bounding_box(with_bbox) is not None and self.n_outputs == 1:
            return outputs[0]
        return outputs

    def _evaluate(self, *args, **kw):
        op = self.op
        if op != 'fix_inputs':
            if op != '&':
                leftval = self.left(*args, **kw)
                if op != '|':
                    rightval = self.right(*args, **kw)
                else:
                    rightval = None

            else:
                leftval = self.left(*(args[:self.left.n_inputs]), **kw)
                rightval = self.right(*(args[self.left.n_inputs:]), **kw)

            if op != "|":
                return self._apply_operators_to_value_lists(leftval, rightval, **kw)

            elif op == '|':
                if isinstance(leftval, tuple):
                    return self.right(*leftval, **kw)
                else:
                    return self.right(leftval, **kw)

        else:
            subs = self.right
            newargs = list(args)
            subinds = []
            subvals = []
            for key in subs.keys():
                if np.issubdtype(type(key), np.integer):
                    subinds.append(key)
                elif isinstance(key, str):
                    ind = self.left.inputs.index(key)
                    subinds.append(ind)
                subvals.append(subs[key])
            # Turn inputs specified in kw into positional indices.
            # Names for compound inputs do not propagate to sub models.
            kwind = []
            kwval = []
            for kwkey in list(kw.keys()):
                if kwkey in self.inputs:
                    ind = self.inputs.index(kwkey)
                    if ind < len(args):
                        raise ValueError("Keyword argument duplicates "
                                         "positional value supplied.")
                    kwind.append(ind)
                    kwval.append(kw[kwkey])
                    del kw[kwkey]
            # Build new argument list
            # Append keyword specified args first
            if kwind:
                kwargs = list(zip(kwind, kwval))
                kwargs.sort()
                kwindsorted, kwvalsorted = list(zip(*kwargs))
                newargs = newargs + list(kwvalsorted)
            if subinds:
                subargs = list(zip(subinds, subvals))
                subargs.sort()
                # subindsorted, subvalsorted = list(zip(*subargs))
                # The substitutions must be inserted in order
                for ind, val in subargs:
                    newargs.insert(ind, val)
            return self.left(*newargs, **kw)
```
### 8 - astropy/modeling/mappings.py:

Start line: 1, End line: 98

```python
"""
Special models useful for complex compound models where control is needed over
which outputs from a source model are mapped to which inputs of a target model.
"""
# pylint: disable=invalid-name

from .core import FittableModel, Model
from astropy.units import Quantity


__all__ = ['Mapping', 'Identity', 'UnitsMapping']


class Mapping(FittableModel):
    """
    Allows inputs to be reordered, duplicated or dropped.

    Parameters
    ----------
    mapping : tuple
        A tuple of integers representing indices of the inputs to this model
        to return and in what order to return them.  See
        :ref:`astropy:compound-model-mappings` for more details.
    n_inputs : int
        Number of inputs; if `None` (default) then ``max(mapping) + 1`` is
        used (i.e. the highest input index used in the mapping).
    name : str, optional
        A human-friendly name associated with this model instance
        (particularly useful for identifying the individual components of a
        compound model).
    meta : dict-like
        Free-form metadata to associate with this model.

    Raises
    ------
    TypeError
        Raised when number of inputs is less that ``max(mapping)``.

    Examples
    --------

    >>> from astropy.modeling.models import Polynomial2D, Shift, Mapping
    >>> poly1 = Polynomial2D(1, c0_0=1, c1_0=2, c0_1=3)
    >>> poly2 = Polynomial2D(1, c0_0=1, c1_0=2.4, c0_1=2.1)
    >>> model = (Shift(1) & Shift(2)) | Mapping((0, 1, 0, 1)) | (poly1 & poly2)
    >>> model(1, 2)  # doctest: +FLOAT_CMP
    (17.0, 14.2)
    """
    linear = True  # FittableModel is non-linear by default

    def __init__(self, mapping, n_inputs=None, name=None, meta=None):
        self._inputs = ()
        self._outputs = ()
        if n_inputs is None:
            self._n_inputs = max(mapping) + 1
        else:
            self._n_inputs = n_inputs

        self._n_outputs = len(mapping)
        super().__init__(name=name, meta=meta)

        self.inputs = tuple('x' + str(idx) for idx in range(self._n_inputs))
        self.outputs = tuple('x' + str(idx) for idx in range(self._n_outputs))

        self._mapping = mapping
        self._input_units_strict = {key: False for key in self._inputs}
        self._input_units_allow_dimensionless = {key: False for key in self._inputs}

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def mapping(self):
        """Integers representing indices of the inputs."""
        return self._mapping

    def __repr__(self):
        if self.name is None:
            return f'<Mapping({self.mapping})>'
        return f'<Mapping({self.mapping}, name={self.name!r})>'

    def evaluate(self, *args):
        if len(args) != self.n_inputs:
            name = self.name if self.name is not None else "Mapping"

            raise TypeError(f'{name} expects {self.n_inputs} inputs; got {len(args)}')

        result = tuple(args[idx] for idx in self._mapping)

        if self.n_outputs == 1:
            return result[0]

        return result
```
### 9 - astropy/io/misc/asdf/tags/transform/compound.py:

Start line: 63, End line: 93

```python
class CompoundType(TransformType):

    @classmethod
    def to_tree_tagged(cls, model, ctx):
        left = model.left

        if isinstance(model.right, dict):
            right = {
                'keys': list(model.right.keys()),
                'values': list(model.right.values())
            }
        else:
            right = model.right

        node = {
            'forward': [left, right]
        }

        try:
            tag_name = 'transform/' + _operator_to_tag_mapping[model.op]
        except KeyError:
            raise ValueError(f"Unknown operator '{model.op}'")

        node = tagged.tag_object(cls.make_yaml_tag(tag_name), node, ctx=ctx)

        return cls._to_tree_base_transform_members(model, node, ctx)

    @classmethod
    def assert_equal(cls, a, b):
        # TODO: If models become comparable themselves, remove this.
        TransformType.assert_equal(a, b)
        assert_tree_match(a.left, b.left)
        assert_tree_match(a.right, b.right)
```
### 10 - astropy/modeling/core.py:

Start line: 3286, End line: 4068

```python
class CompoundModel(Model):

    @property
    def param_names(self):
        """ An ordered list of parameter names."""
        return self._param_names

    def _make_leaflist(self):
        tdict = {}
        leaflist = []
        make_subtree_dict(self, '', tdict, leaflist)
        self._leaflist = leaflist
        self._tdict = tdict

    def __getattr__(self, name):
        """
        If someone accesses an attribute not already defined, map the
        parameters, and then see if the requested attribute is one of
        the parameters
        """
        # The following test is needed to avoid infinite recursion
        # caused by deepcopy. There may be other such cases discovered.
        if name == '__setstate__':
            raise AttributeError
        if name in self._param_names:
            return self.__dict__[name]
        else:
            raise AttributeError(f'Attribute "{name}" not found')

    def __getitem__(self, index):
        if self._leaflist is None:
            self._make_leaflist()
        leaflist = self._leaflist
        tdict = self._tdict
        if isinstance(index, slice):
            if index.step:
                raise ValueError('Steps in slices not supported '
                                 'for compound models')
            if index.start is not None:
                if isinstance(index.start, str):
                    start = self._str_index_to_int(index.start)
                else:
                    start = index.start
            else:
                start = 0
            if index.stop is not None:
                if isinstance(index.stop, str):
                    stop = self._str_index_to_int(index.stop)
                else:
                    stop = index.stop - 1
            else:
                stop = len(leaflist) - 1
            if index.stop == 0:
                raise ValueError("Slice endpoint cannot be 0")
            if start < 0:
                start = len(leaflist) + start
            if stop < 0:
                stop = len(leaflist) + stop
            # now search for matching node:
            if stop == start:  # only single value, get leaf instead in code below
                index = start
            else:
                for key in tdict:
                    node, leftind, rightind = tdict[key]
                    if leftind == start and rightind == stop:
                        return node
                raise IndexError("No appropriate subtree matches slice")
        if isinstance(index, type(0)):
            return leaflist[index]
        elif isinstance(index, type('')):
            return leaflist[self._str_index_to_int(index)]
        else:
            raise TypeError('index must be integer, slice, or model name string')

    def _str_index_to_int(self, str_index):
        # Search through leaflist for item with that name
        found = []
        for nleaf, leaf in enumerate(self._leaflist):
            if getattr(leaf, 'name', None) == str_index:
                found.append(nleaf)
        if len(found) == 0:
            raise IndexError(f"No component with name '{str_index}' found")
        if len(found) > 1:
            raise IndexError("Multiple components found using '{}' as name\n"
                             "at indices {}".format(str_index, found))
        return found[0]

    @property
    def n_inputs(self):
        """ The number of inputs of a model."""
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self._n_inputs = value

    @property
    def n_outputs(self):
        """ The number of outputs of a model."""
        return self._n_outputs

    @n_outputs.setter
    def n_outputs(self, value):
        self._n_outputs = value

    @property
    def eqcons(self):
        return self._eqcons

    @eqcons.setter
    def eqcons(self, value):
        self._eqcons = value

    @property
    def ineqcons(self):
        return self._eqcons

    @ineqcons.setter
    def ineqcons(self, value):
        self._eqcons = value

    def traverse_postorder(self, include_operator=False):
        """ Postorder traversal of the CompoundModel tree."""
        res = []
        if isinstance(self.left, CompoundModel):
            res = res + self.left.traverse_postorder(include_operator)
        else:
            res = res + [self.left]
        if isinstance(self.right, CompoundModel):
            res = res + self.right.traverse_postorder(include_operator)
        else:
            res = res + [self.right]
        if include_operator:
            res.append(self.op)
        else:
            res.append(self)
        return res

    def _format_expression(self, format_leaf=None):
        leaf_idx = 0
        operands = deque()

        if format_leaf is None:
            format_leaf = lambda i, l: f'[{i}]'

        for node in self.traverse_postorder():
            if not isinstance(node, CompoundModel):
                operands.append(format_leaf(leaf_idx, node))
                leaf_idx += 1
                continue

            right = operands.pop()
            left = operands.pop()
            if node.op in OPERATOR_PRECEDENCE:
                oper_order = OPERATOR_PRECEDENCE[node.op]

                if isinstance(node, CompoundModel):
                    if (isinstance(node.left, CompoundModel) and
                            OPERATOR_PRECEDENCE[node.left.op] < oper_order):
                        left = f'({left})'
                    if (isinstance(node.right, CompoundModel) and
                            OPERATOR_PRECEDENCE[node.right.op] < oper_order):
                        right = f'({right})'

                operands.append(' '.join((left, node.op, right)))
            else:
                left = f'(({left}),'
                right = f'({right}))'
                operands.append(' '.join((node.op[0], left, right)))

        return ''.join(operands)

    def _format_components(self):
        if self._parameters_ is None:
            self._map_parameters()
        return '\n\n'.join('[{0}]: {1!r}'.format(idx, m)
                           for idx, m in enumerate(self._leaflist))

    def __str__(self):
        expression = self._format_expression()
        components = self._format_components()
        keywords = [
            ('Expression', expression),
            ('Components', '\n' + indent(components))
        ]
        return super()._format_str(keywords=keywords)

    def rename(self, name):
        self.name = name
        return self

    @property
    def isleaf(self):
        return False

    @property
    def inverse(self):
        if self.op == '|':
            return self.right.inverse | self.left.inverse
        elif self.op == '&':
            return self.left.inverse & self.right.inverse
        else:
            return NotImplemented

    @property
    def fittable(self):
        """ Set the fittable attribute on a compound model."""
        if self._fittable is None:
            if self._leaflist is None:
                self._map_parameters()
            self._fittable = all(m.fittable for m in self._leaflist)
        return self._fittable

    __add__ = _model_oper('+')
    __sub__ = _model_oper('-')
    __mul__ = _model_oper('*')
    __truediv__ = _model_oper('/')
    __pow__ = _model_oper('**')
    __or__ = _model_oper('|')
    __and__ = _model_oper('&')

    def _map_parameters(self):
        """
        Map all the constituent model parameters to the compound object,
        renaming as necessary by appending a suffix number.

        This can be an expensive operation, particularly for a complex
        expression tree.

        All the corresponding parameter attributes are created that one
        expects for the Model class.

        The parameter objects that the attributes point to are the same
        objects as in the constiutent models. Changes made to parameter
        values to either are seen by both.

        Prior to calling this, none of the associated attributes will
        exist. This method must be called to make the model usable by
        fitting engines.

        If oldnames=True, then parameters are named as in the original
        implementation of compound models.
        """
        if self._parameters is not None:
            # do nothing
            return
        if self._leaflist is None:
            self._make_leaflist()
        self._parameters_ = {}
        param_map = {}
        self._param_names = []
        for lindex, leaf in enumerate(self._leaflist):
            if not isinstance(leaf, dict):
                for param_name in leaf.param_names:
                    param = getattr(leaf, param_name)
                    new_param_name = f"{param_name}_{lindex}"
                    self.__dict__[new_param_name] = param
                    self._parameters_[new_param_name] = param
                    self._param_names.append(new_param_name)
                    param_map[new_param_name] = (lindex, param_name)
        self._param_metrics = {}
        self._param_map = param_map
        self._param_map_inverse = dict((v, k) for k, v in param_map.items())
        self._initialize_slices()
        self._param_names = tuple(self._param_names)

    def _initialize_slices(self):
        param_metrics = self._param_metrics
        total_size = 0

        for name in self.param_names:
            param = getattr(self, name)
            value = param.value
            param_size = np.size(value)
            param_shape = np.shape(value)
            param_slice = slice(total_size, total_size + param_size)
            param_metrics[name] = {}
            param_metrics[name]['slice'] = param_slice
            param_metrics[name]['shape'] = param_shape
            param_metrics[name]['size'] = param_size
            total_size += param_size
        self._parameters = np.empty(total_size, dtype=np.float64)

    @staticmethod
    def _recursive_lookup(branch, adict, key):
        if isinstance(branch, CompoundModel):
            return adict[key]
        return branch, key

    def inputs_map(self):
        """
        Map the names of the inputs to this ExpressionTree to the inputs to the leaf models.
        """
        inputs_map = {}
        if not isinstance(self.op, str):  # If we don't have an operator the mapping is trivial
            return {inp: (self, inp) for inp in self.inputs}

        elif self.op == '|':
            if isinstance(self.left, CompoundModel):
                l_inputs_map = self.left.inputs_map()
            for inp in self.inputs:
                if isinstance(self.left, CompoundModel):
                    inputs_map[inp] = l_inputs_map[inp]
                else:
                    inputs_map[inp] = self.left, inp
        elif self.op == '&':
            if isinstance(self.left, CompoundModel):
                l_inputs_map = self.left.inputs_map()
            if isinstance(self.right, CompoundModel):
                r_inputs_map = self.right.inputs_map()
            for i, inp in enumerate(self.inputs):
                if i < len(self.left.inputs):  # Get from left
                    if isinstance(self.left, CompoundModel):
                        inputs_map[inp] = l_inputs_map[self.left.inputs[i]]
                    else:
                        inputs_map[inp] = self.left, self.left.inputs[i]
                else:  # Get from right
                    if isinstance(self.right, CompoundModel):
                        inputs_map[inp] = r_inputs_map[self.right.inputs[i - len(self.left.inputs)]]
                    else:
                        inputs_map[inp] = self.right, self.right.inputs[i - len(self.left.inputs)]
        elif self.op == 'fix_inputs':
            fixed_ind = list(self.right.keys())
            ind = [list(self.left.inputs).index(i) if isinstance(i, str) else i for i in fixed_ind]
            inp_ind = list(range(self.left.n_inputs))
            for i in ind:
                inp_ind.remove(i)
            for i in inp_ind:
                inputs_map[self.left.inputs[i]] = self.left, self.left.inputs[i]
        else:
            if isinstance(self.left, CompoundModel):
                l_inputs_map = self.left.inputs_map()
            for inp in self.left.inputs:
                if isinstance(self.left, CompoundModel):
                    inputs_map[inp] = l_inputs_map[inp]
                else:
                    inputs_map[inp] = self.left, inp
        return inputs_map

    def _parameter_units_for_data_units(self, input_units, output_units):
        if self._leaflist is None:
            self._map_parameters()
        units_for_data = {}
        for imodel, model in enumerate(self._leaflist):
            units_for_data_leaf = model._parameter_units_for_data_units(input_units, output_units)
            for param_leaf in units_for_data_leaf:
                param = self._param_map_inverse[(imodel, param_leaf)]
                units_for_data[param] = units_for_data_leaf[param_leaf]
        return units_for_data

    @property
    def input_units(self):
        inputs_map = self.inputs_map()
        input_units_dict = {key: inputs_map[key][0].input_units[orig_key]
                            for key, (mod, orig_key) in inputs_map.items()
                            if inputs_map[key][0].input_units is not None}
        if input_units_dict:
            return input_units_dict
        return None

    @property
    def input_units_equivalencies(self):
        inputs_map = self.inputs_map()
        input_units_equivalencies_dict = {
            key: inputs_map[key][0].input_units_equivalencies[orig_key]
            for key, (mod, orig_key) in inputs_map.items()
            if inputs_map[key][0].input_units_equivalencies is not None
        }
        if not input_units_equivalencies_dict:
            return None

        return input_units_equivalencies_dict

    @property
    def input_units_allow_dimensionless(self):
        inputs_map = self.inputs_map()
        return {key: inputs_map[key][0].input_units_allow_dimensionless[orig_key]
                for key, (mod, orig_key) in inputs_map.items()}

    @property
    def input_units_strict(self):
        inputs_map = self.inputs_map()
        return {key: inputs_map[key][0].input_units_strict[orig_key]
                for key, (mod, orig_key) in inputs_map.items()}

    @property
    def return_units(self):
        outputs_map = self.outputs_map()
        return {key: outputs_map[key][0].return_units[orig_key]
                for key, (mod, orig_key) in outputs_map.items()
                if outputs_map[key][0].return_units is not None}

    def outputs_map(self):
        """
        Map the names of the outputs to this ExpressionTree to the outputs to the leaf models.
        """
        outputs_map = {}
        if not isinstance(self.op, str):  # If we don't have an operator the mapping is trivial
            return {out: (self, out) for out in self.outputs}

        elif self.op == '|':
            if isinstance(self.right, CompoundModel):
                r_outputs_map = self.right.outputs_map()
            for out in self.outputs:
                if isinstance(self.right, CompoundModel):
                    outputs_map[out] = r_outputs_map[out]
                else:
                    outputs_map[out] = self.right, out

        elif self.op == '&':
            if isinstance(self.left, CompoundModel):
                l_outputs_map = self.left.outputs_map()
            if isinstance(self.right, CompoundModel):
                r_outputs_map = self.right.outputs_map()
            for i, out in enumerate(self.outputs):
                if i < len(self.left.outputs):  # Get from left
                    if isinstance(self.left, CompoundModel):
                        outputs_map[out] = l_outputs_map[self.left.outputs[i]]
                    else:
                        outputs_map[out] = self.left, self.left.outputs[i]
                else:  # Get from right
                    if isinstance(self.right, CompoundModel):
                        outputs_map[out] = r_outputs_map[self.right.outputs[i - len(self.left.outputs)]]
                    else:
                        outputs_map[out] = self.right, self.right.outputs[i - len(self.left.outputs)]
        elif self.op == 'fix_inputs':
            return self.left.outputs_map()
        else:
            if isinstance(self.left, CompoundModel):
                l_outputs_map = self.left.outputs_map()
            for out in self.left.outputs:
                if isinstance(self.left, CompoundModel):
                    outputs_map[out] = l_outputs_map()[out]
                else:
                    outputs_map[out] = self.left, out
        return outputs_map

    @property
    def has_user_bounding_box(self):
        """
        A flag indicating whether or not a custom bounding_box has been
        assigned to this model by a user, via assignment to
        ``model.bounding_box``.
        """

        return self._user_bounding_box is not None

    def render(self, out=None, coords=None):
        """
        Evaluate a model at fixed positions, respecting the ``bounding_box``.

        The key difference relative to evaluating the model directly is that
        this method is limited to a bounding box if the `Model.bounding_box`
        attribute is set.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            An array that the evaluated model will be added to.  If this is not
            given (or given as ``None``), a new array will be created.
        coords : array-like, optional
            An array to be used to translate from the model's input coordinates
            to the ``out`` array. It should have the property that
            ``self(coords)`` yields the same shape as ``out``.  If ``out`` is
            not specified, ``coords`` will be used to determine the shape of
            the returned array. If this is not provided (or None), the model
            will be evaluated on a grid determined by `Model.bounding_box`.

        Returns
        -------
        out : `numpy.ndarray`
            The model added to ``out`` if  ``out`` is not ``None``, or else a
            new array from evaluating the model over ``coords``.
            If ``out`` and ``coords`` are both `None`, the returned array is
            limited to the `Model.bounding_box` limits. If
            `Model.bounding_box` is `None`, ``arr`` or ``coords`` must be
            passed.

        Raises
        ------
        ValueError
            If ``coords`` are not given and the the `Model.bounding_box` of
            this model is not set.

        Examples
        --------
        :ref:`astropy:bounding-boxes`
        """

        bbox = self.get_bounding_box()

        ndim = self.n_inputs

        if (coords is None) and (out is None) and (bbox is None):
            raise ValueError('If no bounding_box is set, '
                             'coords or out must be input.')

        # for consistent indexing
        if ndim == 1:
            if coords is not None:
                coords = [coords]
            if bbox is not None:
                bbox = [bbox]

        if coords is not None:
            coords = np.asanyarray(coords, dtype=float)
            # Check dimensions match out and model
            assert len(coords) == ndim
            if out is not None:
                if coords[0].shape != out.shape:
                    raise ValueError('inconsistent shape of the output.')
            else:
                out = np.zeros(coords[0].shape)

        if out is not None:
            out = np.asanyarray(out)
            if out.ndim != ndim:
                raise ValueError('the array and model must have the same '
                                 'number of dimensions.')

        if bbox is not None:
            # Assures position is at center pixel, important when using
            # add_array.
            pd = np.array([(np.mean(bb), np.ceil((bb[1] - bb[0]) / 2))
                           for bb in bbox]).astype(int).T
            pos, delta = pd

            if coords is not None:
                sub_shape = tuple(delta * 2 + 1)
                sub_coords = np.array([extract_array(c, sub_shape, pos)
                                       for c in coords])
            else:
                limits = [slice(p - d, p + d + 1, 1) for p, d in pd.T]
                sub_coords = np.mgrid[limits]

            sub_coords = sub_coords[::-1]

            if out is None:
                out = self(*sub_coords)
            else:
                try:
                    out = add_array(out, self(*sub_coords), pos)
                except ValueError:
                    raise ValueError(
                        'The `bounding_box` is larger than the input out in '
                        'one or more dimensions. Set '
                        '`model.bounding_box = None`.')
        else:
            if coords is None:
                im_shape = out.shape
                limits = [slice(i) for i in im_shape]
                coords = np.mgrid[limits]

            coords = coords[::-1]

            out += self(*coords)

        return out

    def replace_submodel(self, name, model):
        """
        Construct a new `~astropy.modeling.CompoundModel` instance from an
        existing CompoundModel, replacing the named submodel with a new model.

        In order to ensure that inverses and names are kept/reconstructed, it's
        necessary to rebuild the CompoundModel from the replaced node all the
        way back to the base. The original CompoundModel is left untouched.

        Parameters
        ----------
        name : str
            name of submodel to be replaced
        model : `~astropy.modeling.Model`
            replacement model
        """
        submodels = [m for m in self.traverse_postorder()
                     if getattr(m, 'name', None) == name]
        if submodels:
            if len(submodels) > 1:
                raise ValueError(f"More than one submodel named {name}")

            old_model = submodels.pop()
            if len(old_model) != len(model):
                raise ValueError("New and old models must have equal values "
                                 "for n_models")

            # Do this check first in order to raise a more helpful Exception,
            # although it would fail trying to construct the new CompoundModel
            if (old_model.n_inputs != model.n_inputs or
                        old_model.n_outputs != model.n_outputs):
                raise ValueError("New model must match numbers of inputs and "
                                 "outputs of existing model")

            tree = _get_submodel_path(self, name)
            while tree:
                branch = self.copy()
                for node in tree[:-1]:
                    branch = getattr(branch, node)
                setattr(branch, tree[-1], model)
                model = CompoundModel(branch.op, branch.left, branch.right,
                                      name=branch.name)
                tree = tree[:-1]
            return model

        else:
            raise ValueError(f"No submodels found named {name}")

    def _set_sub_models_and_parameter_units(self, left, right):
        """
        Provides a work-around to properly set the sub models and respective
        parameters's units/values when using ``without_units_for_data``
        or ``without_units_for_data`` methods.
        """
        model = CompoundModel(self.op, left, right)

        self.left = left
        self.right = right

        for name in model.param_names:
            model_parameter = getattr(model, name)
            parameter = getattr(self, name)

            parameter.value = model_parameter.value
            parameter._set_unit(model_parameter.unit, force=True)

    def without_units_for_data(self, **kwargs):
        """
        See `~astropy.modeling.Model.without_units_for_data` for overview
        of this method.

        Notes
        -----
        This modifies the behavior of the base method to account for the
        case where the sub-models of a compound model have different output
        units. This is only valid for compound * and / compound models as
        in that case it is reasonable to mix the output units. It does this
        by modifying the output units of each sub model by using the output
        units of the other sub model so that we can apply the original function
        and get the desired result.

        Additional data has to be output in the mixed output unit case
        so that the units can be properly rebuilt by
        `~astropy.modeling.CompoundModel.with_units_from_data`.

        Outside the mixed output units, this method is identical to the
        base method.
        """
        if self.op in ['*', '/']:
            model = self.copy()
            inputs = {inp: kwargs[inp] for inp in self.inputs}

            left_units = self.left.output_units(**kwargs)
            right_units = self.right.output_units(**kwargs)

            if self.op == '*':
                left_kwargs = {out: kwargs[out] / right_units[out]
                               for out in self.left.outputs if kwargs[out] is not None}
                right_kwargs = {out: kwargs[out] / left_units[out]
                                for out in self.right.outputs if kwargs[out] is not None}
            else:
                left_kwargs = {out: kwargs[out] * right_units[out]
                               for out in self.left.outputs if kwargs[out] is not None}
                right_kwargs = {out: 1 / kwargs[out] * left_units[out]
                                for out in self.right.outputs if kwargs[out] is not None}

            left_kwargs.update(inputs.copy())
            right_kwargs.update(inputs.copy())

            left = self.left.without_units_for_data(**left_kwargs)
            if isinstance(left, tuple):
                left_kwargs['_left_kwargs'] = left[1]
                left_kwargs['_right_kwargs'] = left[2]
                left = left[0]

            right = self.right.without_units_for_data(**right_kwargs)
            if isinstance(right, tuple):
                right_kwargs['_left_kwargs'] = right[1]
                right_kwargs['_right_kwargs'] = right[2]
                right = right[0]

            model._set_sub_models_and_parameter_units(left, right)

            return model, left_kwargs, right_kwargs
        else:
            return super().without_units_for_data(**kwargs)

    def with_units_from_data(self, **kwargs):
        """
        See `~astropy.modeling.Model.with_units_from_data` for overview
        of this method.

        Notes
        -----
        This modifies the behavior of the base method to account for the
        case where the sub-models of a compound model have different output
        units. This is only valid for compound * and / compound models as
        in that case it is reasonable to mix the output units. In order to
        do this it requires some additional information output by
        `~astropy.modeling.CompoundModel.without_units_for_data` passed as
        keyword arguments under the keywords ``_left_kwargs`` and ``_right_kwargs``.

        Outside the mixed output units, this method is identical to the
        base method.
        """

        if self.op in ['*', '/']:
            left_kwargs = kwargs.pop('_left_kwargs')
            right_kwargs = kwargs.pop('_right_kwargs')

            left = self.left.with_units_from_data(**left_kwargs)
            right = self.right.with_units_from_data(**right_kwargs)

            model = self.copy()
            model._set_sub_models_and_parameter_units(left, right)

            return model
        else:
            return super().with_units_from_data(**kwargs)


def _get_submodel_path(model, name):
    """Find the route down a CompoundModel's tree to the model with the
    specified name (whether it's a leaf or not)"""
    if getattr(model, 'name', None) == name:
        return []
    try:
        return ['left'] + _get_submodel_path(model.left, name)
    except (AttributeError, TypeError):
        pass
    try:
        return ['right'] + _get_submodel_path(model.right, name)
    except (AttributeError, TypeError):
        pass


def binary_operation(binoperator, left, right):
    '''
    Perform binary operation. Operands may be matching tuples of operands.
    '''
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple([binoperator(item[0], item[1])
                      for item in zip(left, right)])
    return binoperator(left, right)


def get_ops(tree, opset):
    """
    Recursive function to collect operators used.
    """
    if isinstance(tree, CompoundModel):
        opset.add(tree.op)
        get_ops(tree.left, opset)
        get_ops(tree.right, opset)
    else:
        return


def make_subtree_dict(tree, nodepath, tdict, leaflist):
    '''
    Traverse a tree noting each node by a key that indicates all the
    left/right choices necessary to reach that node. Each key will
    reference a tuple that contains:

    - reference to the compound model for that node.
    - left most index contained within that subtree
       (relative to all indices for the whole tree)
    - right most index contained within that subtree
    '''
    # if this is a leaf, just append it to the leaflist
    if not hasattr(tree, 'isleaf'):
        leaflist.append(tree)
    else:
        leftmostind = len(leaflist)
        make_subtree_dict(tree.left, nodepath+'l', tdict, leaflist)
        make_subtree_dict(tree.right, nodepath+'r', tdict, leaflist)
        rightmostind = len(leaflist)-1
        tdict[nodepath] = (tree, leftmostind, rightmostind)


_ORDER_OF_OPERATORS = [('fix_inputs',), ('|',), ('&',), ('+', '-'), ('*', '/'), ('**',)]
OPERATOR_PRECEDENCE = {}
for idx, ops in enumerate(_ORDER_OF_OPERATORS):
    for op in ops:
        OPERATOR_PRECEDENCE[op] = idx
del idx, op, ops
```
### 40 - astropy/modeling/separable.py:

Start line: 250, End line: 287

```python
def _cdot(left, right):
    """
    Function corresponding to "|" operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """

    left, right = right, left

    def _n_inputs_outputs(input, position):
        """
        Return ``n_inputs``, ``n_outputs`` for a model or coord_matrix.
        """
        if isinstance(input, Model):
            coords = _coord_matrix(input, position, input.n_outputs)
        else:
            coords = input
        return coords

    cleft = _n_inputs_outputs(left, 'left')
    cright = _n_inputs_outputs(right, 'right')

    try:
        result = np.dot(cleft, cright)
    except ValueError:
        raise ModelDefinitionError(
            'Models cannot be combined with the "|" operator; '
            'left coord_matrix is {}, right coord_matrix is {}'.format(
                cright, cleft))
    return result
```
### 59 - astropy/modeling/separable.py:

Start line: 130, End line: 168

```python
def _arith_oper(left, right):
    """
    Function corresponding to one of the arithmetic operators
    ['+', '-'. '*', '/', '**'].

    This always returns a nonseparable output.


    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.
    """
    # models have the same number of inputs and outputs
    def _n_inputs_outputs(input):
        if isinstance(input, Model):
            n_outputs, n_inputs = input.n_outputs, input.n_inputs
        else:
            n_outputs, n_inputs = input.shape
        return n_inputs, n_outputs

    left_inputs, left_outputs = _n_inputs_outputs(left)
    right_inputs, right_outputs = _n_inputs_outputs(right)

    if left_inputs != right_inputs or left_outputs != right_outputs:
        raise ModelDefinitionError(
            "Unsupported operands for arithmetic operator: left (n_inputs={}, "
            "n_outputs={}) and right (n_inputs={}, n_outputs={}); "
            "models must have the same n_inputs and the same "
            "n_outputs for this operator.".format(
                left_inputs, left_outputs, right_inputs, right_outputs))

    result = np.ones((left_outputs, left_inputs))
    return result
```
