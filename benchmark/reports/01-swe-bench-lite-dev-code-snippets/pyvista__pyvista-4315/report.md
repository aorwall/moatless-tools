# pyvista__pyvista-4315

| **pyvista/pyvista** | `db6ee8dd4a747b8864caae36c5d05883976a3ae5` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 7205 |
| **Any found context length** | 7205 |
| **Avg pos** | 15.0 |
| **Min pos** | 15 |
| **Max pos** | 15 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pyvista/core/grid.py b/pyvista/core/grid.py
--- a/pyvista/core/grid.py
+++ b/pyvista/core/grid.py
@@ -135,23 +135,30 @@ def __init__(self, *args, check_duplicates=False, deep=False, **kwargs):
                     self.shallow_copy(args[0])
             elif isinstance(args[0], (str, pathlib.Path)):
                 self._from_file(args[0], **kwargs)
-            elif isinstance(args[0], np.ndarray):
-                self._from_arrays(args[0], None, None, check_duplicates)
+            elif isinstance(args[0], (np.ndarray, Sequence)):
+                self._from_arrays(np.asanyarray(args[0]), None, None, check_duplicates)
             else:
                 raise TypeError(f'Type ({type(args[0])}) not understood by `RectilinearGrid`')
 
         elif len(args) == 3 or len(args) == 2:
-            arg0_is_arr = isinstance(args[0], np.ndarray)
-            arg1_is_arr = isinstance(args[1], np.ndarray)
+            arg0_is_arr = isinstance(args[0], (np.ndarray, Sequence))
+            arg1_is_arr = isinstance(args[1], (np.ndarray, Sequence))
             if len(args) == 3:
-                arg2_is_arr = isinstance(args[2], np.ndarray)
+                arg2_is_arr = isinstance(args[2], (np.ndarray, Sequence))
             else:
                 arg2_is_arr = False
 
             if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
-                self._from_arrays(args[0], args[1], args[2], check_duplicates)
+                self._from_arrays(
+                    np.asanyarray(args[0]),
+                    np.asanyarray(args[1]),
+                    np.asanyarray(args[2]),
+                    check_duplicates,
+                )
             elif all([arg0_is_arr, arg1_is_arr]):
-                self._from_arrays(args[0], args[1], None, check_duplicates)
+                self._from_arrays(
+                    np.asanyarray(args[0]), np.asanyarray(args[1]), None, check_duplicates
+                )
             else:
                 raise TypeError("Arguments not understood by `RectilinearGrid`.")
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pyvista/core/grid.py | 138 | 150 | 15 | 1 | 7205


## Problem Statement

```
Rectilinear grid does not allow Sequences as inputs
### Describe the bug, what's wrong, and what you expected.

Rectilinear grid gives an error when `Sequence`s are passed in, but `ndarray` are ok.

### Steps to reproduce the bug.

This doesn't work
\`\`\`python
import pyvista as pv
pv.RectilinearGrid([0, 1], [0, 1], [0, 1])
\`\`\`

This works
\`\`\`py
import pyvista as pv
import numpy as np
pv.RectilinearGrid(np.ndarray([0, 1]), np.ndarray([0, 1]), np.ndarray([0, 1]))
\`\`\`
### System Information

\`\`\`shell
--------------------------------------------------------------------------------
  Date: Wed Apr 19 20:15:10 2023 UTC

                OS : Linux
            CPU(s) : 2
           Machine : x86_64
      Architecture : 64bit
       Environment : IPython
        GPU Vendor : Mesa/X.org
      GPU Renderer : llvmpipe (LLVM 11.0.1, 256 bits)
       GPU Version : 4.5 (Core Profile) Mesa 20.3.5

  Python 3.11.2 (main, Mar 23 2023, 17:12:29) [GCC 10.2.1 20210110]

           pyvista : 0.38.5
               vtk : 9.2.6
             numpy : 1.24.2
           imageio : 2.27.0
            scooby : 0.7.1
             pooch : v1.7.0
        matplotlib : 3.7.1
           IPython : 8.12.0
--------------------------------------------------------------------------------
\`\`\`


### Screenshots

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | **1 pyvista/core/grid.py** | 0 | 14| 124 | 124 | 
| 2 | 2 pyvista/core/filters/rectilinear_grid.py | 83 | 110| 382 | 506 | 
| 3 | 2 pyvista/core/filters/rectilinear_grid.py | 11 | 8| 705 | 1211 | 
| 4 | 3 examples/00-load/create-explicit-structured-grid.py | 0 | 41| 321 | 1532 | 
| 5 | 4 examples/00-load/create-uniform-grid.py | 0 | 70| 589 | 2121 | 
| 6 | **4 pyvista/core/grid.py** | 778 | 803| 200 | 2321 | 
| 7 | 5 examples/00-load/create-unstructured-surface.py | 135 | 202| 691 | 3012 | 
| 8 | 6 pyvista/utilities/reader.py | 684 | 681| 204 | 3216 | 
| 9 | 7 pyvista/core/filters/structured_grid.py | 164 | 196| 345 | 3561 | 
| 10 | 7 examples/00-load/create-unstructured-surface.py | 0 | 79| 700 | 4261 | 
| 11 | 8 examples/00-load/linear-cells.py | 0 | 105| 810 | 5071 | 
| 12 | **8 pyvista/core/grid.py** | 365 | 392| 211 | 5282 | 
| 13 | 9 pyvista/core/pointset.py | 1884 | 1960| 765 | 6047 | 
| 14 | 10 examples/00-load/create-polyhedron.py | 0 | 60| 738 | 6785 | 
| **-> 15 <-** | **10 pyvista/core/grid.py** | 125 | 167| 420 | 7205 | 
| 16 | 11 pyvista/examples/examples.py | 347 | 393| 406 | 7611 | 
| 17 | **11 pyvista/core/grid.py** | 273 | 271| 319 | 7930 | 
| 18 | **11 pyvista/core/grid.py** | 478 | 560| 769 | 8699 | 
| 19 | 11 pyvista/core/pointset.py | 2346 | 2378| 404 | 9103 | 
| 20 | 11 examples/00-load/linear-cells.py | 153 | 199| 625 | 9728 | 
| 21 | 12 pyvista/core/filters/uniform_grid.py | 394 | 417| 276 | 10004 | 
| 22 | 13 examples/02-plot/point-cell-scalars.py | 0 | 79| 561 | 10565 | 
| 23 | 13 pyvista/core/pointset.py | 2254 | 2305| 505 | 11070 | 
| 24 | **13 pyvista/core/grid.py** | 749 | 776| 221 | 11291 | 
| 25 | **13 pyvista/core/grid.py** | 169 | 226| 457 | 11748 | 
| 26 | **13 pyvista/core/grid.py** | 63 | 123| 537 | 12285 | 
| 27 | 13 pyvista/core/pointset.py | 2564 | 2572| 132 | 12417 | 
| 28 | 13 examples/00-load/linear-cells.py | 107 | 116| 127 | 12544 | 
| 29 | **13 pyvista/core/grid.py** | 588 | 624| 419 | 12963 | 


## Patch

```diff
diff --git a/pyvista/core/grid.py b/pyvista/core/grid.py
--- a/pyvista/core/grid.py
+++ b/pyvista/core/grid.py
@@ -135,23 +135,30 @@ def __init__(self, *args, check_duplicates=False, deep=False, **kwargs):
                     self.shallow_copy(args[0])
             elif isinstance(args[0], (str, pathlib.Path)):
                 self._from_file(args[0], **kwargs)
-            elif isinstance(args[0], np.ndarray):
-                self._from_arrays(args[0], None, None, check_duplicates)
+            elif isinstance(args[0], (np.ndarray, Sequence)):
+                self._from_arrays(np.asanyarray(args[0]), None, None, check_duplicates)
             else:
                 raise TypeError(f'Type ({type(args[0])}) not understood by `RectilinearGrid`')
 
         elif len(args) == 3 or len(args) == 2:
-            arg0_is_arr = isinstance(args[0], np.ndarray)
-            arg1_is_arr = isinstance(args[1], np.ndarray)
+            arg0_is_arr = isinstance(args[0], (np.ndarray, Sequence))
+            arg1_is_arr = isinstance(args[1], (np.ndarray, Sequence))
             if len(args) == 3:
-                arg2_is_arr = isinstance(args[2], np.ndarray)
+                arg2_is_arr = isinstance(args[2], (np.ndarray, Sequence))
             else:
                 arg2_is_arr = False
 
             if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
-                self._from_arrays(args[0], args[1], args[2], check_duplicates)
+                self._from_arrays(
+                    np.asanyarray(args[0]),
+                    np.asanyarray(args[1]),
+                    np.asanyarray(args[2]),
+                    check_duplicates,
+                )
             elif all([arg0_is_arr, arg1_is_arr]):
-                self._from_arrays(args[0], args[1], None, check_duplicates)
+                self._from_arrays(
+                    np.asanyarray(args[0]), np.asanyarray(args[1]), None, check_duplicates
+                )
             else:
                 raise TypeError("Arguments not understood by `RectilinearGrid`.")
 

```

## Test Patch

```diff
diff --git a/tests/test_grid.py b/tests/test_grid.py
--- a/tests/test_grid.py
+++ b/tests/test_grid.py
@@ -735,6 +735,21 @@ def test_create_rectilinear_grid_from_specs():
     assert grid.n_cells == 9 * 3 * 19
     assert grid.n_points == 10 * 4 * 20
     assert grid.bounds == (-10.0, 8.0, -10.0, 5.0, -10.0, 9.0)
+
+    # with Sequence
+    xrng = [0, 1]
+    yrng = [0, 1, 2]
+    zrng = [0, 1, 2, 3]
+    grid = pyvista.RectilinearGrid(xrng)
+    assert grid.n_cells == 1
+    assert grid.n_points == 2
+    grid = pyvista.RectilinearGrid(xrng, yrng)
+    assert grid.n_cells == 2
+    assert grid.n_points == 6
+    grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
+    assert grid.n_cells == 6
+    assert grid.n_points == 24
+
     # 2D example
     cell_spacings = np.array([1.0, 1.0, 2.0, 2.0, 5.0, 10.0])
     x_coordinates = np.cumsum(cell_spacings)

```


## Code snippets

### 1 - pyvista/core/grid.py:

```python
"""Sub-classes for vtk.vtkRectilinearGrid and vtk.vtkImageData."""
from functools import wraps
import pathlib
from typing import Sequence, Tuple, Union
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.core.dataset import DataSet
from pyvista.core.filters import RectilinearGridFilters, UniformGridFilters, _get_output
from pyvista.utilities import abstract_class, assert_empty_kwargs
import pyvista.utilities.helpers as helpers
from pyvista.utilities.misc import PyVistaDeprecationWarning, raise_has_duplicates
```
### 2 - pyvista/core/filters/rectilinear_grid.py:

Start line: 83, End line: 110

```python
@abstract_class
class RectilinearGridFilters:

    def to_tetrahedra(
        self,
        tetra_per_cell: int = 5,
        mixed: Union[Sequence[int], bool] = False,
        pass_cell_ids: bool = False,
        progress_bar: bool = False,
    ):
        # ... other code
        if mixed is not False:
            if isinstance(mixed, str):
                self.cell_data.active_scalars_name = mixed
            elif isinstance(mixed, (np.ndarray, collections.abc.Sequence)):
                self.cell_data['_MIXED_CELLS_'] = mixed  # type: ignore
            elif not isinstance(mixed, bool):
                raise TypeError('`mixed` must be either a sequence of ints or bool')
            alg.SetTetraPerCellTo5And12()
        else:
            if tetra_per_cell not in [5, 6, 12]:
                raise ValueError(
                    f'`tetra_per_cell` should be either 5, 6, or 12, not {tetra_per_cell}'
                )

            # Edge case causing a seg-fault where grid is flat in one dimension
            # See: https://gitlab.kitware.com/vtk/vtk/-/issues/18650
            if 1 in self.dimensions and tetra_per_cell == 12:  # type: ignore
                raise RuntimeError(
                    'Cannot split cells into 12 tetrahedrals when at least '  # type: ignore
                    f'one dimension is 1. Dimensions are {self.dimensions}.'
                )

            alg.SetTetraPerCell(tetra_per_cell)

        alg.SetInputData(self)
        _update_alg(alg, progress_bar, 'Converting to tetrahedra')
        return _get_output(alg)
```
### 3 - pyvista/core/filters/rectilinear_grid.py:

Start line: 11, End line: 8

```python
"""Filters module with the class to manage filters/algorithms for rectilinear grid datasets."""

import collections
from typing import Sequence, Union

import numpy as np

from pyvista import _vtk, abstract_class
from pyvista.core.filters import _get_output, _update_alg


@abstract_class
class RectilinearGridFilters:
    """An internal class to manage filters/algorithms for rectilinear grid datasets."""

    def to_tetrahedra(
        self,
        tetra_per_cell: int = 5,
        mixed: Union[Sequence[int], bool] = False,
        pass_cell_ids: bool = False,
        progress_bar: bool = False,
    ):
        """Create a tetrahedral mesh structured grid.

        Parameters
        ----------
        tetra_per_cell : int, default: 5
            The number of tetrahedrons to divide each cell into. Can be
            either ``5``, ``6``, or ``12``. If ``mixed=True``, this value is
            overridden.

        mixed : str, bool, sequence, default: False
            When set, subdivides some cells into 5 and some cells into 12. Set
            to ``True`` to use the active cell scalars of the
            :class:`pyvista.RectilinearGrid` to be either 5 or 12 to
            determining the number of tetrahedra to generate per cell.

            When a sequence, uses these values to subdivide the cells. When a
            string uses a cell array rather than the active array to determine
            the number of tetrahedra to generate per cell.

        pass_cell_ids : bool, default: False
            Set to ``True`` to make the tetrahedra have scalar data indicating
            which cell they came from in the original
            :class:`pyvista.RectilinearGrid`.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the tetrahedral cells.

        Examples
        --------
        Divide a rectangular grid into tetrahedrons. Each cell contains by
        default 5 tetrahedrons.

        First, create and plot the grid.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.linspace(0, 1, 2)
        >>> yrng = np.linspace(0, 1, 2)
        >>> zrng = np.linspace(0, 2, 3)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.plot()

        Now, generate the tetrahedra plot in the exploded view of the cell.

        >>> tet_grid = grid.to_tetrahedra()
        >>> tet_grid.explode(factor=0.5).plot(show_edges=True)

        Take the same grid but divide the first cell into 5 cells and the other
        cell into 12 tetrahedrons per cell.

        >>> tet_grid = grid.to_tetrahedra(mixed=[5, 12])
        >>> tet_grid.explode(factor=0.5).plot(show_edges=True)

        """
        alg = _vtk.vtkRectilinearGridToTetrahedra()
        alg.SetRememberVoxelId(pass_cell_ids)
        # ... other code
```
### 4 - examples/00-load/create-explicit-structured-grid.py:

```python
"""
.. _ref_create_explicit_structured_grid:

Creating an Explicit Structured Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an explicit structured grid from NumPy arrays.

"""

import numpy as np

import pyvista as pv

ni, nj, nk = 4, 5, 6
si, sj, sk = 20, 10, 1

xcorn = np.arange(0, (ni + 1) * si, si)
xcorn = np.repeat(xcorn, 2)
xcorn = xcorn[1:-1]
xcorn = np.tile(xcorn, 4 * nj * nk)

ycorn = np.arange(0, (nj + 1) * sj, sj)
ycorn = np.repeat(ycorn, 2)
ycorn = ycorn[1:-1]
ycorn = np.tile(ycorn, (2 * ni, 2 * nk))
ycorn = np.transpose(ycorn)
ycorn = ycorn.flatten()

zcorn = np.arange(0, (nk + 1) * sk, sk)
zcorn = np.repeat(zcorn, 2)
zcorn = zcorn[1:-1]
zcorn = np.repeat(zcorn, (4 * ni * nj))

corners = np.stack((xcorn, ycorn, zcorn))
corners = corners.transpose()

dims = np.asarray((ni, nj, nk)) + 1
grid = pv.ExplicitStructuredGrid(dims, corners)
grid = grid.compute_connectivity()
grid.plot(show_edges=True)
```
### 5 - examples/00-load/create-uniform-grid.py:

```python
"""
Creating a Uniform Grid
~~~~~~~~~~~~~~~~~~~~~~~

Create a simple uniform grid from a 3D NumPy array of values.

"""

import numpy as np

import pyvista as pv

###############################################################################
# Take a 3D NumPy array of data values that holds some spatial data where each
# axis corresponds to the XYZ cartesian axes. This example will create a
# :class:`pyvista.UniformGrid` object that will hold the spatial reference for
# a 3D grid which a 3D NumPy array of values can be plotted against.

###############################################################################
# Create the 3D NumPy array of spatially referenced data.
# This is spatially referenced such that the grid is 20 by 5 by 10
# (nx by ny by nz)
values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
values.shape

# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(values.shape) + 1

# Edit the spatial reference
grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array

# Now plot the grid
grid.plot(show_edges=True)


###############################################################################
# Don't like cell data? You could also add the NumPy array to the point data of
# a :class:`pyvista.UniformGrid`. Take note of the subtle difference when
# setting the grid dimensions upon initialization.

# Create the 3D NumPy array of spatially referenced data
# This is spatially referenced such that the grid is 20 by 5 by 10
#   (nx by ny by nz)
values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
values.shape

# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape because we want to inject our values on the
#   POINT data
grid.dimensions = values.shape

# Edit the spatial reference
grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.point_data["values"] = values.flatten(order="F")  # Flatten the array

# Now plot the grid
grid.plot(show_edges=True)
```
### 6 - pyvista/core/grid.py:

Start line: 778, End line: 803

```python
class UniformGrid(_vtk.vtkImageData, Grid, UniformGridFilters):

    def cast_to_rectilinear_grid(self) -> 'RectilinearGrid':
        """Cast this uniform grid to a rectilinear grid.

        Returns
        -------
        pyvista.RectilinearGrid
            This uniform grid as a rectilinear grid.

        """

        def gen_coords(i):
            coords = (
                np.cumsum(np.insert(np.full(self.dimensions[i] - 1, self.spacing[i]), 0, 0))
                + self.origin[i]
            )
            return coords

        xcoords = gen_coords(0)
        ycoords = gen_coords(1)
        zcoords = gen_coords(2)
        grid = pyvista.RectilinearGrid(xcoords, ycoords, zcoords)
        grid.point_data.update(self.point_data)
        grid.cell_data.update(self.cell_data)
        grid.field_data.update(self.field_data)
        grid.copy_meta_from(self, deep=True)
        return grid
```
### 7 - examples/00-load/create-unstructured-surface.py:

Start line: 135, End line: 202

```python
grid = pv.UnstructuredGrid(cells, celltypes, points)

# Alternate versions:
grid = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells.reshape([-1, 9])[:, 1:]}, points)
grid = pv.UnstructuredGrid(
    {CellType.HEXAHEDRON: np.delete(cells, np.arange(0, cells.size, 9))}, points
)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)


###############################################################################
# Tetrahedral Grid
# ~~~~~~~~~~~~~~~~
# Here is how we can create an unstructured tetrahedral grid.

# There are 10 cells here, each cell is [4, INDEX0, INDEX1, INDEX2, INDEX3]
# where INDEX is one of the corners of the tetrahedron.
#
# Note that the array does not need to be shaped like this, we could have a
# flat array, but it's easier to make out the structure of the array this way.
cells = np.array(
    [
        [4, 6, 5, 8, 7],
        [4, 7, 3, 8, 9],
        [4, 7, 3, 1, 5],
        [4, 9, 3, 1, 7],
        [4, 2, 6, 5, 8],
        [4, 2, 6, 0, 4],
        [4, 6, 2, 0, 8],
        [4, 5, 2, 8, 3],
        [4, 5, 3, 8, 7],
        [4, 2, 6, 4, 5],
    ]
)

celltypes = np.full(10, fill_value=CellType.TETRA, dtype=np.uint8)

# These are the 10 points. The number of cells does not need to match the
# number of points, they just happen to in this example
points = np.array(
    [
        [-0.0, 0.0, -0.5],
        [0.0, 0.0, 0.5],
        [-0.43, 0.0, -0.25],
        [-0.43, 0.0, 0.25],
        [-0.0, 0.43, -0.25],
        [0.0, 0.43, 0.25],
        [0.43, 0.0, -0.25],
        [0.43, 0.0, 0.25],
        [0.0, -0.43, -0.25],
        [0.0, -0.43, 0.25],
    ]
)

# Create and plot the unstructured grid
grid = pv.UnstructuredGrid(cells, celltypes, points)
grid.plot(show_edges=True)


###############################################################################
# For fun, let's separate all the cells and plot out the individual cells. Shift
# them a little bit from the center to create an "exploded view".

split_cells = grid.explode(0.5)
split_cells.plot(show_edges=True, ssao=True)
```
### 8 - pyvista/utilities/reader.py:

Start line: 684, End line: 681

```python
class XMLPRectilinearGridReader(BaseReader, PointCellDataSelection):
    """Parallel XML RectilinearGrid Reader for .pvtr files."""

    _class_reader = _vtk.vtkXMLPRectilinearGridReader


class XMLUnstructuredGridReader(BaseReader, PointCellDataSelection):
    """XML UnstructuredGrid Reader for .vtu files.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> filename = examples.download_notch_displacement(load=False)
    >>> filename.split("/")[-1]  # omit the path
    'notch_disp.vtu'
    >>> reader = pyvista.get_reader(filename)
    >>> mesh = reader.read()
    >>> mesh.plot(
    ...     scalars="Nodal Displacement",
    ...     component=0,
    ...     cpos='xy',
    ...     show_scalar_bar=False,
    ... )

    """

    _class_reader = _vtk.vtkXMLUnstructuredGridReader
```
### 9 - pyvista/core/filters/structured_grid.py:

Start line: 164, End line: 196

```python
@abstract_class
class StructuredGridFilters(DataSetFilters):

    def concatenate(self, other, axis, tolerance=0.0):
        # ... other code
        for name, point_array in self.point_data.items():
            arr_1 = self._reshape_point_array(point_array)
            arr_2 = other._reshape_point_array(other.point_data[name])
            if not np.array_equal(
                np.take(arr_1, indices=-1, axis=axis), np.take(arr_2, indices=0, axis=axis)
            ):
                raise RuntimeError(
                    f'Grids cannot be joined along axis {axis}, as field '
                    '`{name}` is not identical along the seam.'
                )
            new_point_data[name] = np.concatenate((arr_1[slice_spec], arr_2), axis=axis).ravel(
                order='F'
            )

        new_dims = np.array(self.dimensions)
        new_dims[axis] += other.dimensions[axis] - 1

        # concatenate cell arrays
        new_cell_data = {}
        for name, cell_array in self.cell_data.items():
            arr_1 = self._reshape_cell_array(cell_array)
            arr_2 = other._reshape_cell_array(other.cell_data[name])
            new_cell_data[name] = np.concatenate((arr_1, arr_2), axis=axis).ravel(order='F')

        # assemble output
        joined = pyvista.StructuredGrid()
        joined.dimensions = list(new_dims)
        joined.points = new_points.reshape((-1, 3), order='F')
        joined.point_data.update(new_point_data)
        joined.cell_data.update(new_cell_data)

        return joined
```
### 10 - examples/00-load/create-unstructured-surface.py:

```python
"""
.. _create_unstructured_example:

Creating an Unstructured Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an irregular, unstructured grid from NumPy arrays.
"""

import numpy as np

import pyvista as pv
from pyvista import CellType

###############################################################################
# An unstructured grid can be created directly from NumPy arrays.
# This is useful when creating a grid from scratch or copying it from another
# format.  See `vtkUnstructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html>`_
# for available cell types and their descriptions.

# Contains information on the points composing each cell.
# Each cell begins with the number of points in the cell and then the points
# composing the cell
cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

# cell type array. Contains the cell type of each cell
cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON])

# in this example, each cell uses separate points
cell1 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

cell2 = np.array(
    [
        [0, 0, 2],
        [1, 0, 2],
        [1, 1, 2],
        [0, 1, 2],
        [0, 0, 3],
        [1, 0, 3],
        [1, 1, 3],
        [0, 1, 3],
    ]
)

# points of the cell array
points = np.vstack((cell1, cell2)).astype(float)

# create the unstructured grid directly from the numpy arrays
grid = pv.UnstructuredGrid(cells, cell_type, points)

# For cells of fixed sizes (like the mentioned Hexahedra), it is also possible to use the
# simplified dictionary interface. This automatically calculates the cell array.
# Note that for mixing with additional cell types, just the appropriate key needs to be
# added to the dictionary.
cells_hex = np.arange(16).reshape([2, 8])
# = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
grid = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells_hex}, points)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)

###############################################################################
# UnstructuredGrid with Shared Points
# -----------------------------------
#
# The next example again creates an unstructured grid containing
# hexahedral cells, but using common points between the cells.

# these points will all be shared between the cells
```
### 12 - pyvista/core/grid.py:

Start line: 365, End line: 392

```python
class RectilinearGrid(_vtk.vtkRectilinearGrid, Grid, RectilinearGridFilters):

    @z.setter
    def z(self, coords: Sequence):
        """Set the coordinates along the Z-direction."""
        self.SetZCoordinates(helpers.convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @Grid.dimensions.setter  # type: ignore
    def dimensions(self, dims):
        """Do not let the dimensions of the RectilinearGrid be set."""
        raise AttributeError(
            "The dimensions of a `RectilinearGrid` are implicitly "
            "defined and thus cannot be set."
        )

    def cast_to_structured_grid(self) -> 'pyvista.StructuredGrid':
        """Cast this rectilinear grid to a structured grid.

        Returns
        -------
        pyvista.StructuredGrid
            This grid as a structured grid.

        """
        alg = _vtk.vtkRectilinearGridToPointSet()
        alg.SetInputData(self)
        alg.Update()
        return _get_output(alg)
```
### 15 - pyvista/core/grid.py:

Start line: 125, End line: 167

```python
class RectilinearGrid(_vtk.vtkRectilinearGrid, Grid, RectilinearGridFilters):

    def __init__(self, *args, check_duplicates=False, deep=False, **kwargs):
        """Initialize the rectilinear grid."""
        super().__init__()

        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkRectilinearGrid):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0], **kwargs)
            elif isinstance(args[0], np.ndarray):
                self._from_arrays(args[0], None, None, check_duplicates)
            else:
                raise TypeError(f'Type ({type(args[0])}) not understood by `RectilinearGrid`')

        elif len(args) == 3 or len(args) == 2:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            if len(args) == 3:
                arg2_is_arr = isinstance(args[2], np.ndarray)
            else:
                arg2_is_arr = False

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(args[0], args[1], args[2], check_duplicates)
            elif all([arg0_is_arr, arg1_is_arr]):
                self._from_arrays(args[0], args[1], None, check_duplicates)
            else:
                raise TypeError("Arguments not understood by `RectilinearGrid`.")

    def __repr__(self):
        """Return the default representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the str representation."""
        return DataSet.__str__(self)

    def _update_dimensions(self):
        """Update the dimensions if coordinates have changed."""
        return self.SetDimensions(len(self.x), len(self.y), len(self.z))
```
### 17 - pyvista/core/grid.py:

Start line: 273, End line: 271

```python
class RectilinearGrid(_vtk.vtkRectilinearGrid, Grid, RectilinearGridFilters):

    @points.setter
    def points(self, points):
        """Raise an AttributeError.

        This setter overrides the base class's setter to ensure a user
        does not attempt to set them.
        """
        raise AttributeError(
            "The points cannot be set. The points of "
            "`RectilinearGrid` are defined in each axial direction. Please "
            "use the `x`, `y`, and `z` setters individually."
        )

    @property
    def x(self) -> np.ndarray:
        """Return or set the coordinates along the X-direction.

        Examples
        --------
        Return the x coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.x
        array([-10.,   0.])

        Set the x coordinates of a RectilinearGrid.

        >>> grid.x = [-10.0, 0.0, 10.0]
        >>> grid.x
        array([-10.,   0.,  10.])

        """
        return helpers.convert_array(self.GetXCoordinates())
```
### 18 - pyvista/core/grid.py:

Start line: 478, End line: 560

```python
class UniformGrid(_vtk.vtkImageData, Grid, UniformGridFilters):

    def __init__(
        self,
        uinput=None,
        *args,
        dimensions=None,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        deep=False,
        **kwargs,
    ):
        """Initialize the uniform grid."""
        super().__init__()

        # permit old behavior
        if isinstance(uinput, Sequence) and not isinstance(uinput, str):
            # Deprecated on v0.37.0, estimated removal on v0.40.0
            warnings.warn(
                "Behavior of pyvista.UniformGrid has changed. First argument must be "
                "either a ``vtk.vtkImageData`` or path.",
                PyVistaDeprecationWarning,
            )
            dimensions = uinput
            uinput = None

        if dimensions is None and 'dims' in kwargs:
            dimensions = kwargs.pop('dims')
            # Deprecated on v0.37.0, estimated removal on v0.40.0
            warnings.warn(
                '`dims` argument is deprecated. Please use `dimensions`.', PyVistaDeprecationWarning
            )
        assert_empty_kwargs(**kwargs)

        if args:
            # Deprecated on v0.37.0, estimated removal on v0.40.0
            warnings.warn(
                "Behavior of pyvista.UniformGrid has changed. Use keyword arguments "
                "to specify dimensions, spacing, and origin. For example:\n\n"
                "    >>> grid = pyvista.UniformGrid(\n"
                "    ...     dimensions=(10, 10, 10),\n"
                "    ...     spacing=(2, 1, 5),\n"
                "    ...     origin=(10, 35, 50),\n"
                "    ... )\n",
                PyVistaDeprecationWarning,
            )
            origin = args[0]
            if len(args) > 1:
                spacing = args[1]
            if len(args) > 2:
                raise ValueError(
                    "Too many additional arguments specified for UniformGrid. "
                    f"Accepts at most 2, and {len(args)} have been input."
                )

        # first argument must be either vtkImageData or a path
        if uinput is not None:
            if isinstance(uinput, _vtk.vtkImageData):
                if deep:
                    self.deep_copy(uinput)
                else:
                    self.shallow_copy(uinput)
            elif isinstance(uinput, (str, pathlib.Path)):
                self._from_file(uinput)
            else:
                raise TypeError(
                    "First argument, ``uinput`` must be either ``vtk.vtkImageData`` "
                    f"or a path, not {type(uinput)}.  Use keyword arguments to "
                    "specify dimensions, spacing, and origin. For example:\n\n"
                    "    >>> grid = pyvista.UniformGrid(\n"
                    "    ...     dimensions=(10, 10, 10),\n"
                    "    ...     spacing=(2, 1, 5),\n"
                    "    ...     origin=(10, 35, 50),\n"
                    "    ... )\n"
                )
        elif dimensions is not None:
            self._from_specs(dimensions, spacing, origin)

    def __repr__(self):
        """Return the default representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the default str representation."""
        return DataSet.__str__(self)
```
### 24 - pyvista/core/grid.py:

Start line: 749, End line: 776

```python
class UniformGrid(_vtk.vtkImageData, Grid, UniformGridFilters):

    @spacing.setter
    def spacing(self, spacing: Sequence[Union[float, int]]):
        """Set spacing."""
        if min(spacing) < 0:
            raise ValueError(f"Spacing must be non-negative, got {spacing}")
        self.SetSpacing(*spacing)
        self.Modified()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = Grid._get_attrs(self)
        fmt = "{}, {}, {}".format(*[pyvista.FLOAT_FORMAT] * 3)
        attrs.append(("Spacing", self.spacing, fmt))
        return attrs

    def cast_to_structured_grid(self) -> 'pyvista.StructuredGrid':
        """Cast this uniform grid to a structured grid.

        Returns
        -------
        pyvista.StructuredGrid
            This grid as a structured grid.

        """
        alg = _vtk.vtkImageToStructuredGrid()
        alg.SetInputData(self)
        alg.Update()
        return _get_output(alg)
```
### 25 - pyvista/core/grid.py:

Start line: 169, End line: 226

```python
class RectilinearGrid(_vtk.vtkRectilinearGrid, Grid, RectilinearGridFilters):

    def _from_arrays(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, check_duplicates: bool = False
    ):
        """Create VTK rectilinear grid directly from numpy arrays.

        Each array gives the uniques coordinates of the mesh along each axial
        direction. To help ensure you are using this correctly, we take the unique
        values of each argument.

        Parameters
        ----------
        x : numpy.ndarray
            Coordinates of the points in x direction.

        y : numpy.ndarray
            Coordinates of the points in y direction.

        z : numpy.ndarray
            Coordinates of the points in z direction.

        check_duplicates : bool, optional
            Check for duplications in any arrays that are passed.

        """
        # Set the coordinates along each axial direction
        # Must at least be an x array
        if check_duplicates:
            raise_has_duplicates(x)

        # edges are shown as triangles if x is not floating point
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(float)
        self.SetXCoordinates(helpers.convert_array(x.ravel()))
        if y is not None:
            if check_duplicates:
                raise_has_duplicates(y)
            if not np.issubdtype(y.dtype, np.floating):
                y = y.astype(float)
            self.SetYCoordinates(helpers.convert_array(y.ravel()))
        if z is not None:
            if check_duplicates:
                raise_has_duplicates(z)
            if not np.issubdtype(z.dtype, np.floating):
                z = z.astype(float)
            self.SetZCoordinates(helpers.convert_array(z.ravel()))
        # Ensure dimensions are properly set
        self._update_dimensions()

    @property
    def meshgrid(self) -> list:
        """Return a meshgrid of numpy arrays for this mesh.

        This simply returns a :func:`numpy.meshgrid` of the
        coordinates for this mesh in ``ij`` indexing. These are a copy
        of the points of this mesh.

        """
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')
```
### 26 - pyvista/core/grid.py:

Start line: 63, End line: 123

```python
class RectilinearGrid(_vtk.vtkRectilinearGrid, Grid, RectilinearGridFilters):
    """Dataset with variable spacing in the three coordinate directions.

    Can be initialized in several ways:

    * Create empty grid
    * Initialize from a ``vtk.vtkRectilinearGrid`` object
    * Initialize directly from the point arrays

    Parameters
    ----------
    uinput : str, pathlib.Path, vtk.vtkRectilinearGrid, numpy.ndarray, optional
        Filename, dataset, or array to initialize the rectilinear grid from. If a
        filename is passed, pyvista will attempt to load it as a
        :class:`RectilinearGrid`. If passed a ``vtk.vtkRectilinearGrid``, it
        will be wrapped. If a :class:`numpy.ndarray` is passed, this will be
        loaded as the x range.

    y : numpy.ndarray, optional
        Coordinates of the points in y direction. If this is passed, ``uinput``
        must be a :class:`numpy.ndarray`.

    z : numpy.ndarray, optional
        Coordinates of the points in z direction. If this is passed, ``uinput``
        and ``y`` must be a :class:`numpy.ndarray`.

    check_duplicates : bool, optional
        Check for duplications in any arrays that are passed. Defaults to
        ``False``. If ``True``, an error is raised if there are any duplicate
        values in any of the array-valued input arguments.

    deep : bool, optional
        Whether to deep copy a ``vtk.vtkRectilinearGrid`` object.
        Default is ``False``.  Keyword only.

    Examples
    --------
    >>> import pyvista
    >>> import vtk
    >>> import numpy as np

    Create an empty grid.

    >>> grid = pyvista.RectilinearGrid()

    Initialize from a vtk.vtkRectilinearGrid object

    >>> vtkgrid = vtk.vtkRectilinearGrid()
    >>> grid = pyvista.RectilinearGrid(vtkgrid)

    Create from NumPy arrays.

    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 5)
    >>> zrng = np.arange(-10, 10, 1)
    >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
    >>> grid.plot(show_edges=True)

    """

    _WRITERS = {'.vtk': _vtk.vtkRectilinearGridWriter, '.vtr': _vtk.vtkXMLRectilinearGridWriter}
```
### 29 - pyvista/core/grid.py:

Start line: 588, End line: 624

```python
class UniformGrid(_vtk.vtkImageData, Grid, UniformGridFilters):

    @property  # type: ignore
    def points(self) -> np.ndarray:  # type: ignore
        """Build a copy of the implicitly defined points as a numpy array.

        Notes
        -----
        The ``points`` for a :class:`pyvista.UniformGrid` cannot be set.

        Examples
        --------
        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
        >>> grid.points
        array([[0., 0., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [1., 1., 0.],
               [0., 0., 1.],
               [1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 1.]])

        """
        # Get grid dimensions
        nx, ny, nz = self.dimensions
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.spacing
        # Now make the cell arrays
        ox, oy, oz = np.array(self.origin) + np.array(self.extent[::2])  # type: ignore
        x = np.insert(np.cumsum(np.full(nx, dx)), 0, 0.0) + ox
        y = np.insert(np.cumsum(np.full(ny, dy)), 0, 0.0) + oy
        z = np.insert(np.cumsum(np.full(nz, dz)), 0, 0.0) + oz
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]
```
