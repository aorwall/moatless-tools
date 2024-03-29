# matplotlib__matplotlib-24013

| **matplotlib/matplotlib** | `394748d584d1cd5c361a6a4c7b70d7b8a8cd3ef0` |
| ---- | ---- |
| **No of patches** | 18 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 5 |
| **Missing snippets** | 20 |
| **Missing patch files** | 16 |


## Expected patch

```diff
diff --git a/lib/matplotlib/tri/__init__.py b/lib/matplotlib/tri/__init__.py
--- a/lib/matplotlib/tri/__init__.py
+++ b/lib/matplotlib/tri/__init__.py
@@ -2,15 +2,15 @@
 Unstructured triangular grid functions.
 """
 
-from .triangulation import Triangulation
-from .tricontour import TriContourSet, tricontour, tricontourf
-from .trifinder import TriFinder, TrapezoidMapTriFinder
-from .triinterpolate import (TriInterpolator, LinearTriInterpolator,
-                             CubicTriInterpolator)
-from .tripcolor import tripcolor
-from .triplot import triplot
-from .trirefine import TriRefiner, UniformTriRefiner
-from .tritools import TriAnalyzer
+from ._triangulation import Triangulation
+from ._tricontour import TriContourSet, tricontour, tricontourf
+from ._trifinder import TriFinder, TrapezoidMapTriFinder
+from ._triinterpolate import (TriInterpolator, LinearTriInterpolator,
+                              CubicTriInterpolator)
+from ._tripcolor import tripcolor
+from ._triplot import triplot
+from ._trirefine import TriRefiner, UniformTriRefiner
+from ._tritools import TriAnalyzer
 
 
 __all__ = ["Triangulation",
diff --git a/lib/matplotlib/tri/_triangulation.py b/lib/matplotlib/tri/_triangulation.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_triangulation.py
@@ -0,0 +1,240 @@
+import numpy as np
+
+from matplotlib import _api
+
+
+class Triangulation:
+    """
+    An unstructured triangular grid consisting of npoints points and
+    ntri triangles.  The triangles can either be specified by the user
+    or automatically generated using a Delaunay triangulation.
+
+    Parameters
+    ----------
+    x, y : (npoints,) array-like
+        Coordinates of grid points.
+    triangles : (ntri, 3) array-like of int, optional
+        For each triangle, the indices of the three points that make
+        up the triangle, ordered in an anticlockwise manner.  If not
+        specified, the Delaunay triangulation is calculated.
+    mask : (ntri,) array-like of bool, optional
+        Which triangles are masked out.
+
+    Attributes
+    ----------
+    triangles : (ntri, 3) array of int
+        For each triangle, the indices of the three points that make
+        up the triangle, ordered in an anticlockwise manner. If you want to
+        take the *mask* into account, use `get_masked_triangles` instead.
+    mask : (ntri, 3) array of bool
+        Masked out triangles.
+    is_delaunay : bool
+        Whether the Triangulation is a calculated Delaunay
+        triangulation (where *triangles* was not specified) or not.
+
+    Notes
+    -----
+    For a Triangulation to be valid it must not have duplicate points,
+    triangles formed from colinear points, or overlapping triangles.
+    """
+    def __init__(self, x, y, triangles=None, mask=None):
+        from matplotlib import _qhull
+
+        self.x = np.asarray(x, dtype=np.float64)
+        self.y = np.asarray(y, dtype=np.float64)
+        if self.x.shape != self.y.shape or self.x.ndim != 1:
+            raise ValueError("x and y must be equal-length 1D arrays, but "
+                             f"found shapes {self.x.shape!r} and "
+                             f"{self.y.shape!r}")
+
+        self.mask = None
+        self._edges = None
+        self._neighbors = None
+        self.is_delaunay = False
+
+        if triangles is None:
+            # No triangulation specified, so use matplotlib._qhull to obtain
+            # Delaunay triangulation.
+            self.triangles, self._neighbors = _qhull.delaunay(x, y)
+            self.is_delaunay = True
+        else:
+            # Triangulation specified. Copy, since we may correct triangle
+            # orientation.
+            try:
+                self.triangles = np.array(triangles, dtype=np.int32, order='C')
+            except ValueError as e:
+                raise ValueError('triangles must be a (N, 3) int array, not '
+                                 f'{triangles!r}') from e
+            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
+                raise ValueError(
+                    'triangles must be a (N, 3) int array, but found shape '
+                    f'{self.triangles.shape!r}')
+            if self.triangles.max() >= len(self.x):
+                raise ValueError(
+                    'triangles are indices into the points and must be in the '
+                    f'range 0 <= i < {len(self.x)} but found value '
+                    f'{self.triangles.max()}')
+            if self.triangles.min() < 0:
+                raise ValueError(
+                    'triangles are indices into the points and must be in the '
+                    f'range 0 <= i < {len(self.x)} but found value '
+                    f'{self.triangles.min()}')
+
+        # Underlying C++ object is not created until first needed.
+        self._cpp_triangulation = None
+
+        # Default TriFinder not created until needed.
+        self._trifinder = None
+
+        self.set_mask(mask)
+
+    def calculate_plane_coefficients(self, z):
+        """
+        Calculate plane equation coefficients for all unmasked triangles from
+        the point (x, y) coordinates and specified z-array of shape (npoints).
+        The returned array has shape (npoints, 3) and allows z-value at (x, y)
+        position in triangle tri to be calculated using
+        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
+        """
+        return self.get_cpp_triangulation().calculate_plane_coefficients(z)
+
+    @property
+    def edges(self):
+        """
+        Return integer array of shape (nedges, 2) containing all edges of
+        non-masked triangles.
+
+        Each row defines an edge by its start point index and end point
+        index.  Each edge appears only once, i.e. for an edge between points
+        *i*  and *j*, there will only be either *(i, j)* or *(j, i)*.
+        """
+        if self._edges is None:
+            self._edges = self.get_cpp_triangulation().get_edges()
+        return self._edges
+
+    def get_cpp_triangulation(self):
+        """
+        Return the underlying C++ Triangulation object, creating it
+        if necessary.
+        """
+        from matplotlib import _tri
+        if self._cpp_triangulation is None:
+            self._cpp_triangulation = _tri.Triangulation(
+                self.x, self.y, self.triangles, self.mask, self._edges,
+                self._neighbors, not self.is_delaunay)
+        return self._cpp_triangulation
+
+    def get_masked_triangles(self):
+        """
+        Return an array of triangles taking the mask into account.
+        """
+        if self.mask is not None:
+            return self.triangles[~self.mask]
+        else:
+            return self.triangles
+
+    @staticmethod
+    def get_from_args_and_kwargs(*args, **kwargs):
+        """
+        Return a Triangulation object from the args and kwargs, and
+        the remaining args and kwargs with the consumed values removed.
+
+        There are two alternatives: either the first argument is a
+        Triangulation object, in which case it is returned, or the args
+        and kwargs are sufficient to create a new Triangulation to
+        return.  In the latter case, see Triangulation.__init__ for
+        the possible args and kwargs.
+        """
+        if isinstance(args[0], Triangulation):
+            triangulation, *args = args
+            if 'triangles' in kwargs:
+                _api.warn_external(
+                    "Passing the keyword 'triangles' has no effect when also "
+                    "passing a Triangulation")
+            if 'mask' in kwargs:
+                _api.warn_external(
+                    "Passing the keyword 'mask' has no effect when also "
+                    "passing a Triangulation")
+        else:
+            x, y, triangles, mask, args, kwargs = \
+                Triangulation._extract_triangulation_params(args, kwargs)
+            triangulation = Triangulation(x, y, triangles, mask)
+        return triangulation, args, kwargs
+
+    @staticmethod
+    def _extract_triangulation_params(args, kwargs):
+        x, y, *args = args
+        # Check triangles in kwargs then args.
+        triangles = kwargs.pop('triangles', None)
+        from_args = False
+        if triangles is None and args:
+            triangles = args[0]
+            from_args = True
+        if triangles is not None:
+            try:
+                triangles = np.asarray(triangles, dtype=np.int32)
+            except ValueError:
+                triangles = None
+        if triangles is not None and (triangles.ndim != 2 or
+                                      triangles.shape[1] != 3):
+            triangles = None
+        if triangles is not None and from_args:
+            args = args[1:]  # Consumed first item in args.
+        # Check for mask in kwargs.
+        mask = kwargs.pop('mask', None)
+        return x, y, triangles, mask, args, kwargs
+
+    def get_trifinder(self):
+        """
+        Return the default `matplotlib.tri.TriFinder` of this
+        triangulation, creating it if necessary.  This allows the same
+        TriFinder object to be easily shared.
+        """
+        if self._trifinder is None:
+            # Default TriFinder class.
+            from matplotlib.tri._trifinder import TrapezoidMapTriFinder
+            self._trifinder = TrapezoidMapTriFinder(self)
+        return self._trifinder
+
+    @property
+    def neighbors(self):
+        """
+        Return integer array of shape (ntri, 3) containing neighbor triangles.
+
+        For each triangle, the indices of the three triangles that
+        share the same edges, or -1 if there is no such neighboring
+        triangle.  ``neighbors[i, j]`` is the triangle that is the neighbor
+        to the edge from point index ``triangles[i, j]`` to point index
+        ``triangles[i, (j+1)%3]``.
+        """
+        if self._neighbors is None:
+            self._neighbors = self.get_cpp_triangulation().get_neighbors()
+        return self._neighbors
+
+    def set_mask(self, mask):
+        """
+        Set or clear the mask array.
+
+        Parameters
+        ----------
+        mask : None or bool array of length ntri
+        """
+        if mask is None:
+            self.mask = None
+        else:
+            self.mask = np.asarray(mask, dtype=bool)
+            if self.mask.shape != (self.triangles.shape[0],):
+                raise ValueError('mask array must have same length as '
+                                 'triangles array')
+
+        # Set mask in C++ Triangulation.
+        if self._cpp_triangulation is not None:
+            self._cpp_triangulation.set_mask(self.mask)
+
+        # Clear derived fields so they are recalculated when needed.
+        self._edges = None
+        self._neighbors = None
+
+        # Recalculate TriFinder if it exists.
+        if self._trifinder is not None:
+            self._trifinder._initialize()
diff --git a/lib/matplotlib/tri/_tricontour.py b/lib/matplotlib/tri/_tricontour.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_tricontour.py
@@ -0,0 +1,271 @@
+import numpy as np
+
+from matplotlib import _docstring
+from matplotlib.contour import ContourSet
+from matplotlib.tri._triangulation import Triangulation
+
+
+@_docstring.dedent_interpd
+class TriContourSet(ContourSet):
+    """
+    Create and store a set of contour lines or filled regions for
+    a triangular grid.
+
+    This class is typically not instantiated directly by the user but by
+    `~.Axes.tricontour` and `~.Axes.tricontourf`.
+
+    %(contour_set_attributes)s
+    """
+    def __init__(self, ax, *args, **kwargs):
+        """
+        Draw triangular grid contour lines or filled regions,
+        depending on whether keyword arg *filled* is False
+        (default) or True.
+
+        The first argument of the initializer must be an `~.axes.Axes`
+        object.  The remaining arguments and keyword arguments
+        are described in the docstring of `~.Axes.tricontour`.
+        """
+        super().__init__(ax, *args, **kwargs)
+
+    def _process_args(self, *args, **kwargs):
+        """
+        Process args and kwargs.
+        """
+        if isinstance(args[0], TriContourSet):
+            C = args[0]._contour_generator
+            if self.levels is None:
+                self.levels = args[0].levels
+            self.zmin = args[0].zmin
+            self.zmax = args[0].zmax
+            self._mins = args[0]._mins
+            self._maxs = args[0]._maxs
+        else:
+            from matplotlib import _tri
+            tri, z = self._contour_args(args, kwargs)
+            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
+            self._mins = [tri.x.min(), tri.y.min()]
+            self._maxs = [tri.x.max(), tri.y.max()]
+
+        self._contour_generator = C
+        return kwargs
+
+    def _contour_args(self, args, kwargs):
+        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
+                                                                   **kwargs)
+        z = np.ma.asarray(args[0])
+        if z.shape != tri.x.shape:
+            raise ValueError('z array must have same length as triangulation x'
+                             ' and y arrays')
+
+        # z values must be finite, only need to check points that are included
+        # in the triangulation.
+        z_check = z[np.unique(tri.get_masked_triangles())]
+        if np.ma.is_masked(z_check):
+            raise ValueError('z must not contain masked points within the '
+                             'triangulation')
+        if not np.isfinite(z_check).all():
+            raise ValueError('z array must not contain non-finite values '
+                             'within the triangulation')
+
+        z = np.ma.masked_invalid(z, copy=False)
+        self.zmax = float(z_check.max())
+        self.zmin = float(z_check.min())
+        if self.logscale and self.zmin <= 0:
+            func = 'contourf' if self.filled else 'contour'
+            raise ValueError(f'Cannot {func} log of negative values.')
+        self._process_contour_level_args(args[1:])
+        return (tri, z)
+
+
+_docstring.interpd.update(_tricontour_doc="""
+Draw contour %%(type)s on an unstructured triangular grid.
+
+Call signatures::
+
+    %%(func)s(triangulation, z, [levels], ...)
+    %%(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)
+
+The triangular grid can be specified either by passing a `.Triangulation`
+object as the first parameter, or by passing the points *x*, *y* and
+optionally the *triangles* and a *mask*. See `.Triangulation` for an
+explanation of these parameters. If neither of *triangulation* or
+*triangles* are given, the triangulation is calculated on the fly.
+
+It is possible to pass *triangles* positionally, i.e.
+``%%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more
+clarity, pass *triangles* via keyword argument.
+
+Parameters
+----------
+triangulation : `.Triangulation`, optional
+    An already created triangular grid.
+
+x, y, triangles, mask
+    Parameters defining the triangular grid. See `.Triangulation`.
+    This is mutually exclusive with specifying *triangulation*.
+
+z : array-like
+    The height values over which the contour is drawn.  Color-mapping is
+    controlled by *cmap*, *norm*, *vmin*, and *vmax*.
+
+    .. note::
+        All values in *z* must be finite. Hence, nan and inf values must
+        either be removed or `~.Triangulation.set_mask` be used.
+
+levels : int or array-like, optional
+    Determines the number and positions of the contour lines / regions.
+
+    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
+    automatically choose no more than *n+1* "nice" contour levels between
+    between minimum and maximum numeric values of *Z*.
+
+    If array-like, draw contour lines at the specified levels.  The values must
+    be in increasing order.
+
+Returns
+-------
+`~matplotlib.tri.TriContourSet`
+
+Other Parameters
+----------------
+colors : color string or sequence of colors, optional
+    The colors of the levels, i.e., the contour %%(type)s.
+
+    The sequence is cycled for the levels in ascending order. If the sequence
+    is shorter than the number of levels, it is repeated.
+
+    As a shortcut, single color strings may be used in place of one-element
+    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
+    same color. This shortcut does only work for color strings, not for other
+    ways of specifying colors.
+
+    By default (value *None*), the colormap specified by *cmap* will be used.
+
+alpha : float, default: 1
+    The alpha blending value, between 0 (transparent) and 1 (opaque).
+
+%(cmap_doc)s
+
+    This parameter is ignored if *colors* is set.
+
+%(norm_doc)s
+
+    This parameter is ignored if *colors* is set.
+
+%(vmin_vmax_doc)s
+
+    If *vmin* or *vmax* are not given, the default color scaling is based on
+    *levels*.
+
+    This parameter is ignored if *colors* is set.
+
+origin : {*None*, 'upper', 'lower', 'image'}, default: None
+    Determines the orientation and exact position of *z* by specifying the
+    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.
+
+    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.
+    - 'lower': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
+    - 'upper': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
+    - 'image': Use the value from :rc:`image.origin`.
+
+extent : (x0, x1, y0, y1), optional
+    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
+    gives the outer pixel boundaries. In this case, the position of z[0, 0] is
+    the center of the pixel, not a corner. If *origin* is *None*, then
+    (*x0*, *y0*) is the position of z[0, 0], and (*x1*, *y1*) is the position
+    of z[-1, -1].
+
+    This argument is ignored if *X* and *Y* are specified in the call to
+    contour.
+
+locator : ticker.Locator subclass, optional
+    The locator is used to determine the contour levels if they are not given
+    explicitly via *levels*.
+    Defaults to `~.ticker.MaxNLocator`.
+
+extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
+    Determines the ``%%(func)s``-coloring of values that are outside the
+    *levels* range.
+
+    If 'neither', values outside the *levels* range are not colored.  If 'min',
+    'max' or 'both', color the values below, above or below and above the
+    *levels* range.
+
+    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
+    under/over values of the `.Colormap`. Note that most colormaps do not have
+    dedicated colors for these by default, so that the over and under values
+    are the edge values of the colormap.  You may want to set these values
+    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.
+
+    .. note::
+
+        An existing `.TriContourSet` does not get notified if properties of its
+        colormap are changed. Therefore, an explicit call to
+        `.ContourSet.changed()` is needed after modifying the colormap. The
+        explicit call can be left out, if a colorbar is assigned to the
+        `.TriContourSet` because it internally calls `.ContourSet.changed()`.
+
+xunits, yunits : registered units, optional
+    Override axis units by specifying an instance of a
+    :class:`matplotlib.units.ConversionInterface`.
+
+antialiased : bool, optional
+    Enable antialiasing, overriding the defaults.  For
+    filled contours, the default is *True*.  For line contours,
+    it is taken from :rc:`lines.antialiased`.""" % _docstring.interpd.params)
+
+
+@_docstring.Substitution(func='tricontour', type='lines')
+@_docstring.dedent_interpd
+def tricontour(ax, *args, **kwargs):
+    """
+    %(_tricontour_doc)s
+
+    linewidths : float or array-like, default: :rc:`contour.linewidth`
+        The line width of the contour lines.
+
+        If a number, all levels will be plotted with this linewidth.
+
+        If a sequence, the levels in ascending order will be plotted with
+        the linewidths in the order specified.
+
+        If None, this falls back to :rc:`lines.linewidth`.
+
+    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
+        If *linestyles* is *None*, the default is 'solid' unless the lines are
+        monochrome.  In that case, negative contours will take their linestyle
+        from :rc:`contour.negative_linestyle` setting.
+
+        *linestyles* can also be an iterable of the above strings specifying a
+        set of linestyles to be used. If this iterable is shorter than the
+        number of contour levels it will be repeated as necessary.
+    """
+    kwargs['filled'] = False
+    return TriContourSet(ax, *args, **kwargs)
+
+
+@_docstring.Substitution(func='tricontourf', type='regions')
+@_docstring.dedent_interpd
+def tricontourf(ax, *args, **kwargs):
+    """
+    %(_tricontour_doc)s
+
+    hatches : list[str], optional
+        A list of cross hatch patterns to use on the filled areas.
+        If None, no hatching will be added to the contour.
+        Hatching is supported in the PostScript, PDF, SVG and Agg
+        backends only.
+
+    Notes
+    -----
+    `.tricontourf` fills intervals that are closed at the top; that is, for
+    boundaries *z1* and *z2*, the filled region is::
+
+        z1 < Z <= z2
+
+    except for the lowest interval, which is closed on both sides (i.e. it
+    includes the lowest value).
+    """
+    kwargs['filled'] = True
+    return TriContourSet(ax, *args, **kwargs)
diff --git a/lib/matplotlib/tri/_trifinder.py b/lib/matplotlib/tri/_trifinder.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_trifinder.py
@@ -0,0 +1,93 @@
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri import Triangulation
+
+
+class TriFinder:
+    """
+    Abstract base class for classes used to find the triangles of a
+    Triangulation in which (x, y) points lie.
+
+    Rather than instantiate an object of a class derived from TriFinder, it is
+    usually better to use the function `.Triangulation.get_trifinder`.
+
+    Derived classes implement __call__(x, y) where x and y are array-like point
+    coordinates of the same shape.
+    """
+
+    def __init__(self, triangulation):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+
+class TrapezoidMapTriFinder(TriFinder):
+    """
+    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
+    map algorithm from the book "Computational Geometry, Algorithms and
+    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
+    and O. Schwarzkopf.
+
+    The triangulation must be valid, i.e. it must not have duplicate points,
+    triangles formed from colinear points, or overlapping triangles.  The
+    algorithm has some tolerance to triangles formed from colinear points, but
+    this should not be relied upon.
+    """
+
+    def __init__(self, triangulation):
+        from matplotlib import _tri
+        super().__init__(triangulation)
+        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
+            triangulation.get_cpp_triangulation())
+        self._initialize()
+
+    def __call__(self, x, y):
+        """
+        Return an array containing the indices of the triangles in which the
+        specified *x*, *y* points lie, or -1 for points that do not lie within
+        a triangle.
+
+        *x*, *y* are array-like x and y coordinates of the same shape and any
+        number of dimensions.
+
+        Returns integer array with the same shape and *x* and *y*.
+        """
+        x = np.asarray(x, dtype=np.float64)
+        y = np.asarray(y, dtype=np.float64)
+        if x.shape != y.shape:
+            raise ValueError("x and y must be array-like with the same shape")
+
+        # C++ does the heavy lifting, and expects 1D arrays.
+        indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
+                   .reshape(x.shape))
+        return indices
+
+    def _get_tree_stats(self):
+        """
+        Return a python list containing the statistics about the node tree:
+            0: number of nodes (tree size)
+            1: number of unique nodes
+            2: number of trapezoids (tree leaf nodes)
+            3: number of unique trapezoids
+            4: maximum parent count (max number of times a node is repeated in
+                   tree)
+            5: maximum depth of tree (one more than the maximum number of
+                   comparisons needed to search through the tree)
+            6: mean of all trapezoid depths (one more than the average number
+                   of comparisons needed to search through the tree)
+        """
+        return self._cpp_trifinder.get_tree_stats()
+
+    def _initialize(self):
+        """
+        Initialize the underlying C++ object.  Can be called multiple times if,
+        for example, the triangulation is modified.
+        """
+        self._cpp_trifinder.initialize()
+
+    def _print_tree(self):
+        """
+        Print a text representation of the node tree, which is useful for
+        debugging purposes.
+        """
+        self._cpp_trifinder.print_tree()
diff --git a/lib/matplotlib/tri/_triinterpolate.py b/lib/matplotlib/tri/_triinterpolate.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_triinterpolate.py
@@ -0,0 +1,1574 @@
+"""
+Interpolation inside triangular grids.
+"""
+
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri import Triangulation
+from matplotlib.tri._trifinder import TriFinder
+from matplotlib.tri._tritools import TriAnalyzer
+
+__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')
+
+
+class TriInterpolator:
+    """
+    Abstract base class for classes used to interpolate on a triangular grid.
+
+    Derived classes implement the following methods:
+
+    - ``__call__(x, y)``,
+      where x, y are array-like point coordinates of the same shape, and
+      that returns a masked array of the same shape containing the
+      interpolated z-values.
+
+    - ``gradient(x, y)``,
+      where x, y are array-like point coordinates of the same
+      shape, and that returns a list of 2 masked arrays of the same shape
+      containing the 2 derivatives of the interpolator (derivatives of
+      interpolated z values with respect to x and y).
+    """
+
+    def __init__(self, triangulation, z, trifinder=None):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+        self._z = np.asarray(z)
+        if self._z.shape != self._triangulation.x.shape:
+            raise ValueError("z array must have same length as triangulation x"
+                             " and y arrays")
+
+        _api.check_isinstance((TriFinder, None), trifinder=trifinder)
+        self._trifinder = trifinder or self._triangulation.get_trifinder()
+
+        # Default scaling factors : 1.0 (= no scaling)
+        # Scaling may be used for interpolations for which the order of
+        # magnitude of x, y has an impact on the interpolant definition.
+        # Please refer to :meth:`_interpolate_multikeys` for details.
+        self._unit_x = 1.0
+        self._unit_y = 1.0
+
+        # Default triangle renumbering: None (= no renumbering)
+        # Renumbering may be used to avoid unnecessary computations
+        # if complex calculations are done inside the Interpolator.
+        # Please refer to :meth:`_interpolate_multikeys` for details.
+        self._tri_renum = None
+
+    # __call__ and gradient docstrings are shared by all subclasses
+    # (except, if needed, relevant additions).
+    # However these methods are only implemented in subclasses to avoid
+    # confusion in the documentation.
+    _docstring__call__ = """
+        Returns a masked array containing interpolated values at the specified
+        (x, y) points.
+
+        Parameters
+        ----------
+        x, y : array-like
+            x and y coordinates of the same shape and any number of
+            dimensions.
+
+        Returns
+        -------
+        np.ma.array
+            Masked array of the same shape as *x* and *y*; values corresponding
+            to (*x*, *y*) points outside of the triangulation are masked out.
+
+        """
+
+    _docstringgradient = r"""
+        Returns a list of 2 masked arrays containing interpolated derivatives
+        at the specified (x, y) points.
+
+        Parameters
+        ----------
+        x, y : array-like
+            x and y coordinates of the same shape and any number of
+            dimensions.
+
+        Returns
+        -------
+        dzdx, dzdy : np.ma.array
+            2 masked arrays of the same shape as *x* and *y*; values
+            corresponding to (x, y) points outside of the triangulation
+            are masked out.
+            The first returned array contains the values of
+            :math:`\frac{\partial z}{\partial x}` and the second those of
+            :math:`\frac{\partial z}{\partial y}`.
+
+        """
+
+    def _interpolate_multikeys(self, x, y, tri_index=None,
+                               return_keys=('z',)):
+        """
+        Versatile (private) method defined for all TriInterpolators.
+
+        :meth:`_interpolate_multikeys` is a wrapper around method
+        :meth:`_interpolate_single_key` (to be defined in the child
+        subclasses).
+        :meth:`_interpolate_single_key actually performs the interpolation,
+        but only for 1-dimensional inputs and at valid locations (inside
+        unmasked triangles of the triangulation).
+
+        The purpose of :meth:`_interpolate_multikeys` is to implement the
+        following common tasks needed in all subclasses implementations:
+
+        - calculation of containing triangles
+        - dealing with more than one interpolation request at the same
+          location (e.g., if the 2 derivatives are requested, it is
+          unnecessary to compute the containing triangles twice)
+        - scaling according to self._unit_x, self._unit_y
+        - dealing with points outside of the grid (with fill value np.nan)
+        - dealing with multi-dimensional *x*, *y* arrays: flattening for
+          :meth:`_interpolate_params` call and final reshaping.
+
+        (Note that np.vectorize could do most of those things very well for
+        you, but it does it by function evaluations over successive tuples of
+        the input arrays. Therefore, this tends to be more time consuming than
+        using optimized numpy functions - e.g., np.dot - which can be used
+        easily on the flattened inputs, in the child-subclass methods
+        :meth:`_interpolate_single_key`.)
+
+        It is guaranteed that the calls to :meth:`_interpolate_single_key`
+        will be done with flattened (1-d) array-like input parameters *x*, *y*
+        and with flattened, valid `tri_index` arrays (no -1 index allowed).
+
+        Parameters
+        ----------
+        x, y : array-like
+            x and y coordinates where interpolated values are requested.
+        tri_index : array-like of int, optional
+            Array of the containing triangle indices, same shape as
+            *x* and *y*. Defaults to None. If None, these indices
+            will be computed by a TriFinder instance.
+            (Note: For point outside the grid, tri_index[ipt] shall be -1).
+        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
+            Defines the interpolation arrays to return, and in which order.
+
+        Returns
+        -------
+        list of arrays
+            Each array-like contains the expected interpolated values in the
+            order defined by *return_keys* parameter.
+        """
+        # Flattening and rescaling inputs arrays x, y
+        # (initial shape is stored for output)
+        x = np.asarray(x, dtype=np.float64)
+        y = np.asarray(y, dtype=np.float64)
+        sh_ret = x.shape
+        if x.shape != y.shape:
+            raise ValueError("x and y shall have same shapes."
+                             " Given: {0} and {1}".format(x.shape, y.shape))
+        x = np.ravel(x)
+        y = np.ravel(y)
+        x_scaled = x/self._unit_x
+        y_scaled = y/self._unit_y
+        size_ret = np.size(x_scaled)
+
+        # Computes & ravels the element indexes, extract the valid ones.
+        if tri_index is None:
+            tri_index = self._trifinder(x, y)
+        else:
+            if tri_index.shape != sh_ret:
+                raise ValueError(
+                    "tri_index array is provided and shall"
+                    " have same shape as x and y. Given: "
+                    "{0} and {1}".format(tri_index.shape, sh_ret))
+            tri_index = np.ravel(tri_index)
+
+        mask_in = (tri_index != -1)
+        if self._tri_renum is None:
+            valid_tri_index = tri_index[mask_in]
+        else:
+            valid_tri_index = self._tri_renum[tri_index[mask_in]]
+        valid_x = x_scaled[mask_in]
+        valid_y = y_scaled[mask_in]
+
+        ret = []
+        for return_key in return_keys:
+            # Find the return index associated with the key.
+            try:
+                return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
+            except KeyError as err:
+                raise ValueError("return_keys items shall take values in"
+                                 " {'z', 'dzdx', 'dzdy'}") from err
+
+            # Sets the scale factor for f & df components
+            scale = [1., 1./self._unit_x, 1./self._unit_y][return_index]
+
+            # Computes the interpolation
+            ret_loc = np.empty(size_ret, dtype=np.float64)
+            ret_loc[~mask_in] = np.nan
+            ret_loc[mask_in] = self._interpolate_single_key(
+                return_key, valid_tri_index, valid_x, valid_y) * scale
+            ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]
+
+        return ret
+
+    def _interpolate_single_key(self, return_key, tri_index, x, y):
+        """
+        Interpolate at points belonging to the triangulation
+        (inside an unmasked triangles).
+
+        Parameters
+        ----------
+        return_key : {'z', 'dzdx', 'dzdy'}
+            The requested values (z or its derivatives).
+        tri_index : 1D int array
+            Valid triangle index (cannot be -1).
+        x, y : 1D arrays, same shape as `tri_index`
+            Valid locations where interpolation is requested.
+
+        Returns
+        -------
+        1-d array
+            Returned array of the same size as *tri_index*
+        """
+        raise NotImplementedError("TriInterpolator subclasses" +
+                                  "should implement _interpolate_single_key!")
+
+
+class LinearTriInterpolator(TriInterpolator):
+    """
+    Linear interpolator on a triangular grid.
+
+    Each triangle is represented by a plane so that an interpolated value at
+    point (x, y) lies on the plane of the triangle containing (x, y).
+    Interpolated values are therefore continuous across the triangulation, but
+    their first derivatives are discontinuous at edges between triangles.
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The triangulation to interpolate over.
+    z : (npoints,) array-like
+        Array of values, defined at grid points, to interpolate between.
+    trifinder : `~matplotlib.tri.TriFinder`, optional
+        If this is not specified, the Triangulation's default TriFinder will
+        be used by calling `.Triangulation.get_trifinder`.
+
+    Methods
+    -------
+    `__call__` (x, y) : Returns interpolated values at (x, y) points.
+    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
+
+    """
+    def __init__(self, triangulation, z, trifinder=None):
+        super().__init__(triangulation, z, trifinder)
+
+        # Store plane coefficients for fast interpolation calculations.
+        self._plane_coefficients = \
+            self._triangulation.calculate_plane_coefficients(self._z)
+
+    def __call__(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('z',))[0]
+    __call__.__doc__ = TriInterpolator._docstring__call__
+
+    def gradient(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('dzdx', 'dzdy'))
+    gradient.__doc__ = TriInterpolator._docstringgradient
+
+    def _interpolate_single_key(self, return_key, tri_index, x, y):
+        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
+        if return_key == 'z':
+            return (self._plane_coefficients[tri_index, 0]*x +
+                    self._plane_coefficients[tri_index, 1]*y +
+                    self._plane_coefficients[tri_index, 2])
+        elif return_key == 'dzdx':
+            return self._plane_coefficients[tri_index, 0]
+        else:  # 'dzdy'
+            return self._plane_coefficients[tri_index, 1]
+
+
+class CubicTriInterpolator(TriInterpolator):
+    r"""
+    Cubic interpolator on a triangular grid.
+
+    In one-dimension - on a segment - a cubic interpolating function is
+    defined by the values of the function and its derivative at both ends.
+    This is almost the same in 2D inside a triangle, except that the values
+    of the function and its 2 derivatives have to be defined at each triangle
+    node.
+
+    The CubicTriInterpolator takes the value of the function at each node -
+    provided by the user - and internally computes the value of the
+    derivatives, resulting in a smooth interpolation.
+    (As a special feature, the user can also impose the value of the
+    derivatives at each node, but this is not supposed to be the common
+    usage.)
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The triangulation to interpolate over.
+    z : (npoints,) array-like
+        Array of values, defined at grid points, to interpolate between.
+    kind : {'min_E', 'geom', 'user'}, optional
+        Choice of the smoothing algorithm, in order to compute
+        the interpolant derivatives (defaults to 'min_E'):
+
+        - if 'min_E': (default) The derivatives at each node is computed
+          to minimize a bending energy.
+        - if 'geom': The derivatives at each node is computed as a
+          weighted average of relevant triangle normals. To be used for
+          speed optimization (large grids).
+        - if 'user': The user provides the argument *dz*, no computation
+          is hence needed.
+
+    trifinder : `~matplotlib.tri.TriFinder`, optional
+        If not specified, the Triangulation's default TriFinder will
+        be used by calling `.Triangulation.get_trifinder`.
+    dz : tuple of array-likes (dzdx, dzdy), optional
+        Used only if  *kind* ='user'. In this case *dz* must be provided as
+        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
+        are the interpolant first derivatives at the *triangulation* points.
+
+    Methods
+    -------
+    `__call__` (x, y) : Returns interpolated values at (x, y) points.
+    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
+
+    Notes
+    -----
+    This note is a bit technical and details how the cubic interpolation is
+    computed.
+
+    The interpolation is based on a Clough-Tocher subdivision scheme of
+    the *triangulation* mesh (to make it clearer, each triangle of the
+    grid will be divided in 3 child-triangles, and on each child triangle
+    the interpolated function is a cubic polynomial of the 2 coordinates).
+    This technique originates from FEM (Finite Element Method) analysis;
+    the element used is a reduced Hsieh-Clough-Tocher (HCT)
+    element. Its shape functions are described in [1]_.
+    The assembled function is guaranteed to be C1-smooth, i.e. it is
+    continuous and its first derivatives are also continuous (this
+    is easy to show inside the triangles but is also true when crossing the
+    edges).
+
+    In the default case (*kind* ='min_E'), the interpolant minimizes a
+    curvature energy on the functional space generated by the HCT element
+    shape functions - with imposed values but arbitrary derivatives at each
+    node. The minimized functional is the integral of the so-called total
+    curvature (implementation based on an algorithm from [2]_ - PCG sparse
+    solver):
+
+        .. math::
+
+            E(z) = \frac{1}{2} \int_{\Omega} \left(
+                \left( \frac{\partial^2{z}}{\partial{x}^2} \right)^2 +
+                \left( \frac{\partial^2{z}}{\partial{y}^2} \right)^2 +
+                2\left( \frac{\partial^2{z}}{\partial{y}\partial{x}} \right)^2
+            \right) dx\,dy
+
+    If the case *kind* ='geom' is chosen by the user, a simple geometric
+    approximation is used (weighted average of the triangle normal
+    vectors), which could improve speed on very large grids.
+
+    References
+    ----------
+    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
+        Hsieh-Clough-Tocher triangles, complete or reduced.",
+        International Journal for Numerical Methods in Engineering,
+        17(5):784 - 789. 2.01.
+    .. [2] C.T. Kelley, "Iterative Methods for Optimization".
+
+    """
+    def __init__(self, triangulation, z, kind='min_E', trifinder=None,
+                 dz=None):
+        super().__init__(triangulation, z, trifinder)
+
+        # Loads the underlying c++ _triangulation.
+        # (During loading, reordering of triangulation._triangles may occur so
+        # that all final triangles are now anti-clockwise)
+        self._triangulation.get_cpp_triangulation()
+
+        # To build the stiffness matrix and avoid zero-energy spurious modes
+        # we will only store internally the valid (unmasked) triangles and
+        # the necessary (used) points coordinates.
+        # 2 renumbering tables need to be computed and stored:
+        #  - a triangle renum table in order to translate the result from a
+        #    TriFinder instance into the internal stored triangle number.
+        #  - a node renum table to overwrite the self._z values into the new
+        #    (used) node numbering.
+        tri_analyzer = TriAnalyzer(self._triangulation)
+        (compressed_triangles, compressed_x, compressed_y, tri_renum,
+         node_renum) = tri_analyzer._get_compressed_triangulation()
+        self._triangles = compressed_triangles
+        self._tri_renum = tri_renum
+        # Taking into account the node renumbering in self._z:
+        valid_node = (node_renum != -1)
+        self._z[node_renum[valid_node]] = self._z[valid_node]
+
+        # Computing scale factors
+        self._unit_x = np.ptp(compressed_x)
+        self._unit_y = np.ptp(compressed_y)
+        self._pts = np.column_stack([compressed_x / self._unit_x,
+                                     compressed_y / self._unit_y])
+        # Computing triangle points
+        self._tris_pts = self._pts[self._triangles]
+        # Computing eccentricities
+        self._eccs = self._compute_tri_eccentricities(self._tris_pts)
+        # Computing dof estimations for HCT triangle shape function
+        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
+        self._dof = self._compute_dof(kind, dz=dz)
+        # Loading HCT element
+        self._ReferenceElement = _ReducedHCT_Element()
+
+    def __call__(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('z',))[0]
+    __call__.__doc__ = TriInterpolator._docstring__call__
+
+    def gradient(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('dzdx', 'dzdy'))
+    gradient.__doc__ = TriInterpolator._docstringgradient
+
+    def _interpolate_single_key(self, return_key, tri_index, x, y):
+        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
+        tris_pts = self._tris_pts[tri_index]
+        alpha = self._get_alpha_vec(x, y, tris_pts)
+        ecc = self._eccs[tri_index]
+        dof = np.expand_dims(self._dof[tri_index], axis=1)
+        if return_key == 'z':
+            return self._ReferenceElement.get_function_values(
+                alpha, ecc, dof)
+        else:  # 'dzdx', 'dzdy'
+            J = self._get_jacobian(tris_pts)
+            dzdx = self._ReferenceElement.get_function_derivatives(
+                alpha, J, ecc, dof)
+            if return_key == 'dzdx':
+                return dzdx[:, 0, 0]
+            else:
+                return dzdx[:, 1, 0]
+
+    def _compute_dof(self, kind, dz=None):
+        """
+        Compute and return nodal dofs according to kind.
+
+        Parameters
+        ----------
+        kind : {'min_E', 'geom', 'user'}
+            Choice of the _DOF_estimator subclass to estimate the gradient.
+        dz : tuple of array-likes (dzdx, dzdy), optional
+            Used only if *kind*=user; in this case passed to the
+            :class:`_DOF_estimator_user`.
+
+        Returns
+        -------
+        array-like, shape (npts, 2)
+            Estimation of the gradient at triangulation nodes (stored as
+            degree of freedoms of reduced-HCT triangle elements).
+        """
+        if kind == 'user':
+            if dz is None:
+                raise ValueError("For a CubicTriInterpolator with "
+                                 "*kind*='user', a valid *dz* "
+                                 "argument is expected.")
+            TE = _DOF_estimator_user(self, dz=dz)
+        elif kind == 'geom':
+            TE = _DOF_estimator_geom(self)
+        else:  # 'min_E', checked in __init__
+            TE = _DOF_estimator_min_E(self)
+        return TE.compute_dof_from_df()
+
+    @staticmethod
+    def _get_alpha_vec(x, y, tris_pts):
+        """
+        Fast (vectorized) function to compute barycentric coordinates alpha.
+
+        Parameters
+        ----------
+        x, y : array-like of dim 1 (shape (nx,))
+            Coordinates of the points whose points barycentric coordinates are
+            requested.
+        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
+            Coordinates of the containing triangles apexes.
+
+        Returns
+        -------
+        array of dim 2 (shape (nx, 3))
+            Barycentric coordinates of the points inside the containing
+            triangles.
+        """
+        ndim = tris_pts.ndim-2
+
+        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
+        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
+        abT = np.stack([a, b], axis=-1)
+        ab = _transpose_vectorized(abT)
+        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]
+
+        metric = ab @ abT
+        # Here we try to deal with the colinear cases.
+        # metric_inv is in this case set to the Moore-Penrose pseudo-inverse
+        # meaning that we will still return a set of valid barycentric
+        # coordinates.
+        metric_inv = _pseudo_inv22sym_vectorized(metric)
+        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
+        ksi = metric_inv @ Covar
+        alpha = _to_matrix_vectorized([
+            [1-ksi[:, 0, 0]-ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
+        return alpha
+
+    @staticmethod
+    def _get_jacobian(tris_pts):
+        """
+        Fast (vectorized) function to compute triangle jacobian matrix.
+
+        Parameters
+        ----------
+        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
+            Coordinates of the containing triangles apexes.
+
+        Returns
+        -------
+        array of dim 3 (shape (nx, 2, 2))
+            Barycentric coordinates of the points inside the containing
+            triangles.
+            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
+            itri, so that the following (matrix) relationship holds:
+               [dz/dksi] = [J] x [dz/dx]
+            with x: global coordinates
+                 ksi: element parametric coordinates in triangle first apex
+                 local basis.
+        """
+        a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
+        b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
+        J = _to_matrix_vectorized([[a[:, 0], a[:, 1]],
+                                   [b[:, 0], b[:, 1]]])
+        return J
+
+    @staticmethod
+    def _compute_tri_eccentricities(tris_pts):
+        """
+        Compute triangle eccentricities.
+
+        Parameters
+        ----------
+        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
+            Coordinates of the triangles apexes.
+
+        Returns
+        -------
+        array like of dim 2 (shape: (nx, 3))
+            The so-called eccentricity parameters [1] needed for HCT triangular
+            element.
+        """
+        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
+        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
+        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
+        # Do not use np.squeeze, this is dangerous if only one triangle
+        # in the triangulation...
+        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
+        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
+        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
+        # Note that this line will raise a warning for dot_a, dot_b or dot_c
+        # zeros, but we choose not to support triangles with duplicate points.
+        return _to_matrix_vectorized([[(dot_c-dot_b) / dot_a],
+                                      [(dot_a-dot_c) / dot_b],
+                                      [(dot_b-dot_a) / dot_c]])
+
+
+# FEM element used for interpolation and for solving minimisation
+# problem (Reduced HCT element)
+class _ReducedHCT_Element:
+    """
+    Implementation of reduced HCT triangular element with explicit shape
+    functions.
+
+    Computes z, dz, d2z and the element stiffness matrix for bending energy:
+    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)
+
+    *** Reference for the shape functions: ***
+    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
+        reduced.
+        Michel Bernadou, Kamal Hassan
+        International Journal for Numerical Methods in Engineering.
+        17(5):784 - 789.  2.01
+
+    *** Element description: ***
+    9 dofs: z and dz given at 3 apex
+    C1 (conform)
+
+    """
+    # 1) Loads matrices to generate shape functions as a function of
+    #    triangle eccentricities - based on [1] p.11 '''
+    M = np.array([
+        [ 0.00, 0.00, 0.00,  4.50,  4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00,  0.50,  1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00,  1.25,  0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.50, 1.00, 0.00, -1.50,  0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
+        [ 0.00, 0.00, 0.00, -0.25,  0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
+        [ 0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
+        [ 0.50, 0.00, 1.00,  0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
+        [ 0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
+        [ 0.00, 0.00, 0.00,  0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
+    M0 = np.array([
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [-1.00, 0.00, 0.00,  1.50,  1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
+        [-0.50, 0.00, 0.00,  0.75,  0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00,  3.00],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00,  1.50]])
+    M1 = np.array([
+        [-0.50, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
+    M2 = np.array([
+        [ 0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.50, 0.00, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
+
+    # 2) Loads matrices to rotate components of gradient & Hessian
+    #    vectors in the reference basis of triangle first apex (a0)
+    rotate_dV = np.array([[ 1.,  0.], [ 0.,  1.],
+                          [ 0.,  1.], [-1., -1.],
+                          [-1., -1.], [ 1.,  0.]])
+
+    rotate_d2V = np.array([[1., 0., 0.], [0., 1., 0.], [ 0.,  0.,  1.],
+                           [0., 1., 0.], [1., 1., 1.], [ 0., -2., -1.],
+                           [1., 1., 1.], [1., 0., 0.], [-2.,  0., -1.]])
+
+    # 3) Loads Gauss points & weights on the 3 sub-_triangles for P2
+    #    exact integral - 3 points on each subtriangles.
+    # NOTE: as the 2nd derivative is discontinuous , we really need those 9
+    # points!
+    n_gauss = 9
+    gauss_pts = np.array([[13./18.,  4./18.,  1./18.],
+                          [ 4./18., 13./18.,  1./18.],
+                          [ 7./18.,  7./18.,  4./18.],
+                          [ 1./18., 13./18.,  4./18.],
+                          [ 1./18.,  4./18., 13./18.],
+                          [ 4./18.,  7./18.,  7./18.],
+                          [ 4./18.,  1./18., 13./18.],
+                          [13./18.,  1./18.,  4./18.],
+                          [ 7./18.,  4./18.,  7./18.]], dtype=np.float64)
+    gauss_w = np.ones([9], dtype=np.float64) / 9.
+
+    #  4) Stiffness matrix for curvature energy
+    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
+
+    #  5) Loads the matrix to compute DOF_rot from tri_J at apex 0
+    J0_to_J1 = np.array([[-1.,  1.], [-1.,  0.]])
+    J0_to_J2 = np.array([[ 0., -1.], [ 1., -1.]])
+
+    def get_function_values(self, alpha, ecc, dofs):
+        """
+        Parameters
+        ----------
+        alpha : is a (N x 3 x 1) array (array of column-matrices) of
+        barycentric coordinates,
+        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities,
+        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
+        degrees of freedom.
+
+        Returns
+        -------
+        Returns the N-array of interpolated function values.
+        """
+        subtri = np.argmin(alpha, axis=1)[:, 0]
+        ksi = _roll_vectorized(alpha, -subtri, axis=0)
+        E = _roll_vectorized(ecc, -subtri, axis=0)
+        x = ksi[:, 0, 0]
+        y = ksi[:, 1, 0]
+        z = ksi[:, 2, 0]
+        x_sq = x*x
+        y_sq = y*y
+        z_sq = z*z
+        V = _to_matrix_vectorized([
+            [x_sq*x], [y_sq*y], [z_sq*z], [x_sq*z], [x_sq*y], [y_sq*x],
+            [y_sq*z], [z_sq*y], [z_sq*x], [x*y*z]])
+        prod = self.M @ V
+        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
+        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
+        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
+        s = _roll_vectorized(prod, 3*subtri, axis=0)
+        return (dofs @ s)[:, 0, 0]
+
+    def get_function_derivatives(self, alpha, J, ecc, dofs):
+        """
+        Parameters
+        ----------
+        *alpha* is a (N x 3 x 1) array (array of column-matrices of
+        barycentric coordinates)
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
+        eccentricities)
+        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
+        degrees of freedom.
+
+        Returns
+        -------
+        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
+        in global coordinates at locations alpha, as a column-matrices of
+        shape (N x 2 x 1).
+        """
+        subtri = np.argmin(alpha, axis=1)[:, 0]
+        ksi = _roll_vectorized(alpha, -subtri, axis=0)
+        E = _roll_vectorized(ecc, -subtri, axis=0)
+        x = ksi[:, 0, 0]
+        y = ksi[:, 1, 0]
+        z = ksi[:, 2, 0]
+        x_sq = x*x
+        y_sq = y*y
+        z_sq = z*z
+        dV = _to_matrix_vectorized([
+            [    -3.*x_sq,     -3.*x_sq],
+            [     3.*y_sq,           0.],
+            [          0.,      3.*z_sq],
+            [     -2.*x*z, -2.*x*z+x_sq],
+            [-2.*x*y+x_sq,      -2.*x*y],
+            [ 2.*x*y-y_sq,        -y_sq],
+            [      2.*y*z,         y_sq],
+            [        z_sq,       2.*y*z],
+            [       -z_sq,  2.*x*z-z_sq],
+            [     x*z-y*z,      x*y-y*z]])
+        # Puts back dV in first apex basis
+        dV = dV @ _extract_submatrices(
+            self.rotate_dV, subtri, block_size=2, axis=0)
+
+        prod = self.M @ dV
+        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
+        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
+        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
+        dsdksi = _roll_vectorized(prod, 3*subtri, axis=0)
+        dfdksi = dofs @ dsdksi
+        # In global coordinates:
+        # Here we try to deal with the simplest colinear cases, returning a
+        # null matrix.
+        J_inv = _safe_inv22_vectorized(J)
+        dfdx = J_inv @ _transpose_vectorized(dfdksi)
+        return dfdx
+
+    def get_function_hessians(self, alpha, J, ecc, dofs):
+        """
+        Parameters
+        ----------
+        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
+        barycentric coordinates
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
+        degrees of freedom.
+
+        Returns
+        -------
+        Returns the values of interpolated function 2nd-derivatives
+        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
+        as a column-matrices of shape (N x 3 x 1).
+        """
+        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
+        d2fdksi2 = dofs @ d2sdksi2
+        H_rot = self.get_Hrot_from_J(J)
+        d2fdx2 = d2fdksi2 @ H_rot
+        return _transpose_vectorized(d2fdx2)
+
+    def get_d2Sidksij2(self, alpha, ecc):
+        """
+        Parameters
+        ----------
+        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
+        barycentric coordinates
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+
+        Returns
+        -------
+        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
+        expressed in covariant coordinates in first apex basis.
+        """
+        subtri = np.argmin(alpha, axis=1)[:, 0]
+        ksi = _roll_vectorized(alpha, -subtri, axis=0)
+        E = _roll_vectorized(ecc, -subtri, axis=0)
+        x = ksi[:, 0, 0]
+        y = ksi[:, 1, 0]
+        z = ksi[:, 2, 0]
+        d2V = _to_matrix_vectorized([
+            [     6.*x,      6.*x,      6.*x],
+            [     6.*y,        0.,        0.],
+            [       0.,      6.*z,        0.],
+            [     2.*z, 2.*z-4.*x, 2.*z-2.*x],
+            [2.*y-4.*x,      2.*y, 2.*y-2.*x],
+            [2.*x-4.*y,        0.,     -2.*y],
+            [     2.*z,        0.,      2.*y],
+            [       0.,      2.*y,      2.*z],
+            [       0., 2.*x-4.*z,     -2.*z],
+            [    -2.*z,     -2.*y,     x-y-z]])
+        # Puts back d2V in first apex basis
+        d2V = d2V @ _extract_submatrices(
+            self.rotate_d2V, subtri, block_size=3, axis=0)
+        prod = self.M @ d2V
+        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
+        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
+        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
+        d2sdksi2 = _roll_vectorized(prod, 3*subtri, axis=0)
+        return d2sdksi2
+
+    def get_bending_matrices(self, J, ecc):
+        """
+        Parameters
+        ----------
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+
+        Returns
+        -------
+        Returns the element K matrices for bending energy expressed in
+        GLOBAL nodal coordinates.
+        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
+        tri_J is needed to rotate dofs from local basis to global basis
+        """
+        n = np.size(ecc, 0)
+
+        # 1) matrix to rotate dofs in global coordinates
+        J1 = self.J0_to_J1 @ J
+        J2 = self.J0_to_J2 @ J
+        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
+        DOF_rot[:, 0, 0] = 1
+        DOF_rot[:, 3, 3] = 1
+        DOF_rot[:, 6, 6] = 1
+        DOF_rot[:, 1:3, 1:3] = J
+        DOF_rot[:, 4:6, 4:6] = J1
+        DOF_rot[:, 7:9, 7:9] = J2
+
+        # 2) matrix to rotate Hessian in global coordinates.
+        H_rot, area = self.get_Hrot_from_J(J, return_area=True)
+
+        # 3) Computes stiffness matrix
+        # Gauss quadrature.
+        K = np.zeros([n, 9, 9], dtype=np.float64)
+        weights = self.gauss_w
+        pts = self.gauss_pts
+        for igauss in range(self.n_gauss):
+            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
+            alpha = np.expand_dims(alpha, 2)
+            weight = weights[igauss]
+            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
+            d2Skdx2 = d2Skdksi2 @ H_rot
+            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))
+
+        # 4) With nodal (not elem) dofs
+        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
+
+        # 5) Need the area to compute total element energy
+        return _scalar_vectorized(area, K)
+
+    def get_Hrot_from_J(self, J, return_area=False):
+        """
+        Parameters
+        ----------
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+
+        Returns
+        -------
+        Returns H_rot used to rotate Hessian from local basis of first apex,
+        to global coordinates.
+        if *return_area* is True, returns also the triangle area (0.5*det(J))
+        """
+        # Here we try to deal with the simplest colinear cases; a null
+        # energy and area is imposed.
+        J_inv = _safe_inv22_vectorized(J)
+        Ji00 = J_inv[:, 0, 0]
+        Ji11 = J_inv[:, 1, 1]
+        Ji10 = J_inv[:, 1, 0]
+        Ji01 = J_inv[:, 0, 1]
+        H_rot = _to_matrix_vectorized([
+            [Ji00*Ji00, Ji10*Ji10, Ji00*Ji10],
+            [Ji01*Ji01, Ji11*Ji11, Ji01*Ji11],
+            [2*Ji00*Ji01, 2*Ji11*Ji10, Ji00*Ji11+Ji10*Ji01]])
+        if not return_area:
+            return H_rot
+        else:
+            area = 0.5 * (J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0])
+            return H_rot, area
+
+    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
+        """
+        Build K and F for the following elliptic formulation:
+        minimization of curvature energy with value of function at node
+        imposed and derivatives 'free'.
+
+        Build the global Kff matrix in cco format.
+        Build the full Ff vec Ff = - Kfc x Uc.
+
+        Parameters
+        ----------
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+        *triangles* is a (N x 3) array of nodes indexes.
+        *Uc* is (N x 3) array of imposed displacements at nodes
+
+        Returns
+        -------
+        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
+        (row, col) entries must be summed.
+        Ff: force vector - dim npts * 3
+        """
+        ntri = np.size(ecc, 0)
+        vec_range = np.arange(ntri, dtype=np.int32)
+        c_indices = np.full(ntri, -1, dtype=np.int32)  # for unused dofs, -1
+        f_dof = [1, 2, 4, 5, 7, 8]
+        c_dof = [0, 3, 6]
+
+        # vals, rows and cols indices in global dof numbering
+        f_dof_indices = _to_matrix_vectorized([[
+            c_indices, triangles[:, 0]*2, triangles[:, 0]*2+1,
+            c_indices, triangles[:, 1]*2, triangles[:, 1]*2+1,
+            c_indices, triangles[:, 2]*2, triangles[:, 2]*2+1]])
+
+        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
+        f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
+        f_col_indices = expand_indices @ f_dof_indices
+        K_elem = self.get_bending_matrices(J, ecc)
+
+        # Extracting sub-matrices
+        # Explanation & notations:
+        # * Subscript f denotes 'free' degrees of freedom (i.e. dz/dx, dz/dx)
+        # * Subscript c denotes 'condensated' (imposed) degrees of freedom
+        #    (i.e. z at all nodes)
+        # * F = [Ff, Fc] is the force vector
+        # * U = [Uf, Uc] is the imposed dof vector
+        #        [ Kff Kfc ]
+        # * K =  [         ]  is the laplacian stiffness matrix
+        #        [ Kcf Kff ]
+        # * As F = K x U one gets straightforwardly: Ff = - Kfc x Uc
+
+        # Computing Kff stiffness matrix in sparse coo format
+        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
+        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
+        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])
+
+        # Computing Ff force vector in sparse coo format
+        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
+        Uc_elem = np.expand_dims(Uc, axis=2)
+        Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
+        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]
+
+        # Extracting Ff force vector in dense format
+        # We have to sum duplicate indices -  using bincount
+        Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
+        return Kff_rows, Kff_cols, Kff_vals, Ff
+
+
+# :class:_DOF_estimator, _DOF_estimator_user, _DOF_estimator_geom,
+# _DOF_estimator_min_E
+# Private classes used to compute the degree of freedom of each triangular
+# element for the TriCubicInterpolator.
+class _DOF_estimator:
+    """
+    Abstract base class for classes used to estimate a function's first
+    derivatives, and deduce the dofs for a CubicTriInterpolator using a
+    reduced HCT element formulation.
+
+    Derived classes implement ``compute_df(self, **kwargs)``, returning
+    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
+    gradient coordinates.
+    """
+    def __init__(self, interpolator, **kwargs):
+        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
+        self._pts = interpolator._pts
+        self._tris_pts = interpolator._tris_pts
+        self.z = interpolator._z
+        self._triangles = interpolator._triangles
+        (self._unit_x, self._unit_y) = (interpolator._unit_x,
+                                        interpolator._unit_y)
+        self.dz = self.compute_dz(**kwargs)
+        self.compute_dof_from_df()
+
+    def compute_dz(self, **kwargs):
+        raise NotImplementedError
+
+    def compute_dof_from_df(self):
+        """
+        Compute reduced-HCT elements degrees of freedom, from the gradient.
+        """
+        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
+        tri_z = self.z[self._triangles]
+        tri_dz = self.dz[self._triangles]
+        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
+        return tri_dof
+
+    @staticmethod
+    def get_dof_vec(tri_z, tri_dz, J):
+        """
+        Compute the dof vector of a triangle, from the value of f, df and
+        of the local Jacobian at each node.
+
+        Parameters
+        ----------
+        tri_z : shape (3,) array
+            f nodal values.
+        tri_dz : shape (3, 2) array
+            df/dx, df/dy nodal values.
+        J
+            Jacobian matrix in local basis of apex 0.
+
+        Returns
+        -------
+        dof : shape (9,) array
+            For each apex ``iapex``::
+
+                dof[iapex*3+0] = f(Ai)
+                dof[iapex*3+1] = df(Ai).(AiAi+)
+                dof[iapex*3+2] = df(Ai).(AiAi-)
+        """
+        npt = tri_z.shape[0]
+        dof = np.zeros([npt, 9], dtype=np.float64)
+        J1 = _ReducedHCT_Element.J0_to_J1 @ J
+        J2 = _ReducedHCT_Element.J0_to_J2 @ J
+
+        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
+        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
+        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
+
+        dfdksi = _to_matrix_vectorized([
+            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
+            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
+        dof[:, 0:7:3] = tri_z
+        dof[:, 1:8:3] = dfdksi[:, 0]
+        dof[:, 2:9:3] = dfdksi[:, 1]
+        return dof
+
+
+class _DOF_estimator_user(_DOF_estimator):
+    """dz is imposed by user; accounts for scaling if any."""
+
+    def compute_dz(self, dz):
+        (dzdx, dzdy) = dz
+        dzdx = dzdx * self._unit_x
+        dzdy = dzdy * self._unit_y
+        return np.vstack([dzdx, dzdy]).T
+
+
+class _DOF_estimator_geom(_DOF_estimator):
+    """Fast 'geometric' approximation, recommended for large arrays."""
+
+    def compute_dz(self):
+        """
+        self.df is computed as weighted average of _triangles sharing a common
+        node. On each triangle itri f is first assumed linear (= ~f), which
+        allows to compute d~f[itri]
+        Then the following approximation of df nodal values is then proposed:
+            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
+        The weighted coeff. w[itri] are proportional to the angle of the
+        triangle itri at apex ipt
+        """
+        el_geom_w = self.compute_geom_weights()
+        el_geom_grad = self.compute_geom_grads()
+
+        # Sum of weights coeffs
+        w_node_sum = np.bincount(np.ravel(self._triangles),
+                                 weights=np.ravel(el_geom_w))
+
+        # Sum of weighted df = (dfx, dfy)
+        dfx_el_w = np.empty_like(el_geom_w)
+        dfy_el_w = np.empty_like(el_geom_w)
+        for iapex in range(3):
+            dfx_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 0]
+            dfy_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 1]
+        dfx_node_sum = np.bincount(np.ravel(self._triangles),
+                                   weights=np.ravel(dfx_el_w))
+        dfy_node_sum = np.bincount(np.ravel(self._triangles),
+                                   weights=np.ravel(dfy_el_w))
+
+        # Estimation of df
+        dfx_estim = dfx_node_sum/w_node_sum
+        dfy_estim = dfy_node_sum/w_node_sum
+        return np.vstack([dfx_estim, dfy_estim]).T
+
+    def compute_geom_weights(self):
+        """
+        Build the (nelems, 3) weights coeffs of _triangles angles,
+        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
+        """
+        weights = np.zeros([np.size(self._triangles, 0), 3])
+        tris_pts = self._tris_pts
+        for ipt in range(3):
+            p0 = tris_pts[:, ipt % 3, :]
+            p1 = tris_pts[:, (ipt+1) % 3, :]
+            p2 = tris_pts[:, (ipt-1) % 3, :]
+            alpha1 = np.arctan2(p1[:, 1]-p0[:, 1], p1[:, 0]-p0[:, 0])
+            alpha2 = np.arctan2(p2[:, 1]-p0[:, 1], p2[:, 0]-p0[:, 0])
+            # In the below formula we could take modulo 2. but
+            # modulo 1. is safer regarding round-off errors (flat triangles).
+            angle = np.abs(((alpha2-alpha1) / np.pi) % 1)
+            # Weight proportional to angle up np.pi/2; null weight for
+            # degenerated cases 0 and np.pi (note that *angle* is normalized
+            # by np.pi).
+            weights[:, ipt] = 0.5 - np.abs(angle-0.5)
+        return weights
+
+    def compute_geom_grads(self):
+        """
+        Compute the (global) gradient component of f assumed linear (~f).
+        returns array df of shape (nelems, 2)
+        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
+        """
+        tris_pts = self._tris_pts
+        tris_f = self.z[self._triangles]
+
+        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
+        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
+        dM = np.dstack([dM1, dM2])
+        # Here we try to deal with the simplest colinear cases: a null
+        # gradient is assumed in this case.
+        dM_inv = _safe_inv22_vectorized(dM)
+
+        dZ1 = tris_f[:, 1] - tris_f[:, 0]
+        dZ2 = tris_f[:, 2] - tris_f[:, 0]
+        dZ = np.vstack([dZ1, dZ2]).T
+        df = np.empty_like(dZ)
+
+        # With np.einsum: could be ej,eji -> ej
+        df[:, 0] = dZ[:, 0]*dM_inv[:, 0, 0] + dZ[:, 1]*dM_inv[:, 1, 0]
+        df[:, 1] = dZ[:, 0]*dM_inv[:, 0, 1] + dZ[:, 1]*dM_inv[:, 1, 1]
+        return df
+
+
+class _DOF_estimator_min_E(_DOF_estimator_geom):
+    """
+    The 'smoothest' approximation, df is computed through global minimization
+    of the bending energy:
+      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
+    """
+    def __init__(self, Interpolator):
+        self._eccs = Interpolator._eccs
+        super().__init__(Interpolator)
+
+    def compute_dz(self):
+        """
+        Elliptic solver for bending energy minimization.
+        Uses a dedicated 'toy' sparse Jacobi PCG solver.
+        """
+        # Initial guess for iterative PCG solver.
+        dz_init = super().compute_dz()
+        Uf0 = np.ravel(dz_init)
+
+        reference_element = _ReducedHCT_Element()
+        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
+        eccs = self._eccs
+        triangles = self._triangles
+        Uc = self.z[self._triangles]
+
+        # Building stiffness matrix and force vector in coo format
+        Kff_rows, Kff_cols, Kff_vals, Ff = reference_element.get_Kff_and_Ff(
+            J, eccs, triangles, Uc)
+
+        # Building sparse matrix and solving minimization problem
+        # We could use scipy.sparse direct solver; however to avoid this
+        # external dependency an implementation of a simple PCG solver with
+        # a simple diagonal Jacobi preconditioner is implemented.
+        tol = 1.e-10
+        n_dof = Ff.shape[0]
+        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
+                                     shape=(n_dof, n_dof))
+        Kff_coo.compress_csc()
+        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
+        # If the PCG did not converge, we return the best guess between Uf0
+        # and Uf.
+        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
+        if err0 < err:
+            # Maybe a good occasion to raise a warning here ?
+            _api.warn_external("In TriCubicInterpolator initialization, "
+                               "PCG sparse solver did not converge after "
+                               "1000 iterations. `geom` approximation is "
+                               "used instead of `min_E`")
+            Uf = Uf0
+
+        # Building dz from Uf
+        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
+        dz[:, 0] = Uf[::2]
+        dz[:, 1] = Uf[1::2]
+        return dz
+
+
+# The following private :class:_Sparse_Matrix_coo and :func:_cg provide
+# a PCG sparse solver for (symmetric) elliptic problems.
+class _Sparse_Matrix_coo:
+    def __init__(self, vals, rows, cols, shape):
+        """
+        Create a sparse matrix in coo format.
+        *vals*: arrays of values of non-null entries of the matrix
+        *rows*: int arrays of rows of non-null entries of the matrix
+        *cols*: int arrays of cols of non-null entries of the matrix
+        *shape*: 2-tuple (n, m) of matrix shape
+        """
+        self.n, self.m = shape
+        self.vals = np.asarray(vals, dtype=np.float64)
+        self.rows = np.asarray(rows, dtype=np.int32)
+        self.cols = np.asarray(cols, dtype=np.int32)
+
+    def dot(self, V):
+        """
+        Dot product of self by a vector *V* in sparse-dense to dense format
+        *V* dense vector of shape (self.m,).
+        """
+        assert V.shape == (self.m,)
+        return np.bincount(self.rows,
+                           weights=self.vals*V[self.cols],
+                           minlength=self.m)
+
+    def compress_csc(self):
+        """
+        Compress rows, cols, vals / summing duplicates. Sort for csc format.
+        """
+        _, unique, indices = np.unique(
+            self.rows + self.n*self.cols,
+            return_index=True, return_inverse=True)
+        self.rows = self.rows[unique]
+        self.cols = self.cols[unique]
+        self.vals = np.bincount(indices, weights=self.vals)
+
+    def compress_csr(self):
+        """
+        Compress rows, cols, vals / summing duplicates. Sort for csr format.
+        """
+        _, unique, indices = np.unique(
+            self.m*self.rows + self.cols,
+            return_index=True, return_inverse=True)
+        self.rows = self.rows[unique]
+        self.cols = self.cols[unique]
+        self.vals = np.bincount(indices, weights=self.vals)
+
+    def to_dense(self):
+        """
+        Return a dense matrix representing self, mainly for debugging purposes.
+        """
+        ret = np.zeros([self.n, self.m], dtype=np.float64)
+        nvals = self.vals.size
+        for i in range(nvals):
+            ret[self.rows[i], self.cols[i]] += self.vals[i]
+        return ret
+
+    def __str__(self):
+        return self.to_dense().__str__()
+
+    @property
+    def diag(self):
+        """Return the (dense) vector of the diagonal elements."""
+        in_diag = (self.rows == self.cols)
+        diag = np.zeros(min(self.n, self.n), dtype=np.float64)  # default 0.
+        diag[self.rows[in_diag]] = self.vals[in_diag]
+        return diag
+
+
+def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
+    """
+    Use Preconditioned Conjugate Gradient iteration to solve A x = b
+    A simple Jacobi (diagonal) preconditioner is used.
+
+    Parameters
+    ----------
+    A : _Sparse_Matrix_coo
+        *A* must have been compressed before by compress_csc or
+        compress_csr method.
+    b : array
+        Right hand side of the linear system.
+    x0 : array, optional
+        Starting guess for the solution. Defaults to the zero vector.
+    tol : float, optional
+        Tolerance to achieve. The algorithm terminates when the relative
+        residual is below tol. Default is 1e-10.
+    maxiter : int, optional
+        Maximum number of iterations.  Iteration will stop after *maxiter*
+        steps even if the specified tolerance has not been achieved. Defaults
+        to 1000.
+
+    Returns
+    -------
+    x : array
+        The converged solution.
+    err : float
+        The absolute error np.linalg.norm(A.dot(x) - b)
+    """
+    n = b.size
+    assert A.n == n
+    assert A.m == n
+    b_norm = np.linalg.norm(b)
+
+    # Jacobi pre-conditioner
+    kvec = A.diag
+    # For diag elem < 1e-6 we keep 1e-6.
+    kvec = np.maximum(kvec, 1e-6)
+
+    # Initial guess
+    if x0 is None:
+        x = np.zeros(n)
+    else:
+        x = x0
+
+    r = b - A.dot(x)
+    w = r/kvec
+
+    p = np.zeros(n)
+    beta = 0.0
+    rho = np.dot(r, w)
+    k = 0
+
+    # Following C. T. Kelley
+    while (np.sqrt(abs(rho)) > tol*b_norm) and (k < maxiter):
+        p = w + beta*p
+        z = A.dot(p)
+        alpha = rho/np.dot(p, z)
+        r = r - alpha*z
+        w = r/kvec
+        rhoold = rho
+        rho = np.dot(r, w)
+        x = x + alpha*p
+        beta = rho/rhoold
+        # err = np.linalg.norm(A.dot(x) - b)  # absolute accuracy - not used
+        k += 1
+    err = np.linalg.norm(A.dot(x) - b)
+    return x, err
+
+
+# The following private functions:
+#     :func:`_safe_inv22_vectorized`
+#     :func:`_pseudo_inv22sym_vectorized`
+#     :func:`_scalar_vectorized`
+#     :func:`_transpose_vectorized`
+#     :func:`_roll_vectorized`
+#     :func:`_to_matrix_vectorized`
+#     :func:`_extract_submatrices`
+# provide fast numpy implementation of some standard operations on arrays of
+# matrices - stored as (:, n_rows, n_cols)-shaped np.arrays.
+
+# Development note: Dealing with pathologic 'flat' triangles in the
+# CubicTriInterpolator code and impact on (2, 2)-matrix inversion functions
+# :func:`_safe_inv22_vectorized` and :func:`_pseudo_inv22sym_vectorized`.
+#
+# Goals:
+# 1) The CubicTriInterpolator should be able to handle flat or almost flat
+#    triangles without raising an error,
+# 2) These degenerated triangles should have no impact on the automatic dof
+#    calculation (associated with null weight for the _DOF_estimator_geom and
+#    with null energy for the _DOF_estimator_min_E),
+# 3) Linear patch test should be passed exactly on degenerated meshes,
+# 4) Interpolation (with :meth:`_interpolate_single_key` or
+#    :meth:`_interpolate_multi_key`) shall be correctly handled even *inside*
+#    the pathologic triangles, to interact correctly with a TriRefiner class.
+#
+# Difficulties:
+# Flat triangles have rank-deficient *J* (so-called jacobian matrix) and
+# *metric* (the metric tensor = J x J.T). Computation of the local
+# tangent plane is also problematic.
+#
+# Implementation:
+# Most of the time, when computing the inverse of a rank-deficient matrix it
+# is safe to simply return the null matrix (which is the implementation in
+# :func:`_safe_inv22_vectorized`). This is because of point 2), itself
+# enforced by:
+#    - null area hence null energy in :class:`_DOF_estimator_min_E`
+#    - angles close or equal to 0 or np.pi hence null weight in
+#      :class:`_DOF_estimator_geom`.
+#      Note that the function angle -> weight is continuous and maximum for an
+#      angle np.pi/2 (refer to :meth:`compute_geom_weights`)
+# The exception is the computation of barycentric coordinates, which is done
+# by inversion of the *metric* matrix. In this case, we need to compute a set
+# of valid coordinates (1 among numerous possibilities), to ensure point 4).
+# We benefit here from the symmetry of metric = J x J.T, which makes it easier
+# to compute a pseudo-inverse in :func:`_pseudo_inv22sym_vectorized`
+def _safe_inv22_vectorized(M):
+    """
+    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
+    matrices.
+
+    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
+    """
+    _api.check_shape((None, 2, 2), M=M)
+    M_inv = np.empty_like(M)
+    prod1 = M[:, 0, 0]*M[:, 1, 1]
+    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
+
+    # We set delta_inv to 0. in case of a rank deficient matrix; a
+    # rank-deficient input matrix *M* will lead to a null matrix in output
+    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
+    if np.all(rank2):
+        # Normal 'optimized' flow.
+        delta_inv = 1./delta
+    else:
+        # 'Pathologic' flow.
+        delta_inv = np.zeros(M.shape[0])
+        delta_inv[rank2] = 1./delta[rank2]
+
+    M_inv[:, 0, 0] = M[:, 1, 1]*delta_inv
+    M_inv[:, 0, 1] = -M[:, 0, 1]*delta_inv
+    M_inv[:, 1, 0] = -M[:, 1, 0]*delta_inv
+    M_inv[:, 1, 1] = M[:, 0, 0]*delta_inv
+    return M_inv
+
+
+def _pseudo_inv22sym_vectorized(M):
+    """
+    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
+    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.
+
+    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
+    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
+    In case M is of rank 0, we return the null matrix.
+
+    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
+    """
+    _api.check_shape((None, 2, 2), M=M)
+    M_inv = np.empty_like(M)
+    prod1 = M[:, 0, 0]*M[:, 1, 1]
+    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
+    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
+
+    if np.all(rank2):
+        # Normal 'optimized' flow.
+        M_inv[:, 0, 0] = M[:, 1, 1] / delta
+        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
+        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
+        M_inv[:, 1, 1] = M[:, 0, 0] / delta
+    else:
+        # 'Pathologic' flow.
+        # Here we have to deal with 2 sub-cases
+        # 1) First sub-case: matrices of rank 2:
+        delta = delta[rank2]
+        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
+        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
+        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
+        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
+        # 2) Second sub-case: rank-deficient matrices of rank 0 and 1:
+        rank01 = ~rank2
+        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
+        tr_zeros = (np.abs(tr) < 1.e-8)
+        sq_tr_inv = (1.-tr_zeros) / (tr**2+tr_zeros)
+        # sq_tr_inv = 1. / tr**2
+        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
+        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
+        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
+        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv
+
+    return M_inv
+
+
+def _scalar_vectorized(scalar, M):
+    """
+    Scalar product between scalars and matrices.
+    """
+    return scalar[:, np.newaxis, np.newaxis]*M
+
+
+def _transpose_vectorized(M):
+    """
+    Transposition of an array of matrices *M*.
+    """
+    return np.transpose(M, [0, 2, 1])
+
+
+def _roll_vectorized(M, roll_indices, axis):
+    """
+    Roll an array of matrices along *axis* (0: rows, 1: columns) according to
+    an array of indices *roll_indices*.
+    """
+    assert axis in [0, 1]
+    ndim = M.ndim
+    assert ndim == 3
+    ndim_roll = roll_indices.ndim
+    assert ndim_roll == 1
+    sh = M.shape
+    r, c = sh[-2:]
+    assert sh[0] == roll_indices.shape[0]
+    vec_indices = np.arange(sh[0], dtype=np.int32)
+
+    # Builds the rolled matrix
+    M_roll = np.empty_like(M)
+    if axis == 0:
+        for ir in range(r):
+            for ic in range(c):
+                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices+ir) % r, ic]
+    else:  # 1
+        for ir in range(r):
+            for ic in range(c):
+                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices+ic) % c]
+    return M_roll
+
+
+def _to_matrix_vectorized(M):
+    """
+    Build an array of matrices from individuals np.arrays of identical shapes.
+
+    Parameters
+    ----------
+    M
+        ncols-list of nrows-lists of shape sh.
+
+    Returns
+    -------
+    M_res : np.array of shape (sh, nrow, ncols)
+        *M_res* satisfies ``M_res[..., i, j] = M[i][j]``.
+    """
+    assert isinstance(M, (tuple, list))
+    assert all(isinstance(item, (tuple, list)) for item in M)
+    c_vec = np.asarray([len(item) for item in M])
+    assert np.all(c_vec-c_vec[0] == 0)
+    r = len(M)
+    c = c_vec[0]
+    M00 = np.asarray(M[0][0])
+    dt = M00.dtype
+    sh = [M00.shape[0], r, c]
+    M_ret = np.empty(sh, dtype=dt)
+    for irow in range(r):
+        for icol in range(c):
+            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
+    return M_ret
+
+
+def _extract_submatrices(M, block_indices, block_size, axis):
+    """
+    Extract selected blocks of a matrices *M* depending on parameters
+    *block_indices* and *block_size*.
+
+    Returns the array of extracted matrices *Mres* so that ::
+
+        M_res[..., ir, :] = M[(block_indices*block_size+ir), :]
+    """
+    assert block_indices.ndim == 1
+    assert axis in [0, 1]
+
+    r, c = M.shape
+    if axis == 0:
+        sh = [block_indices.shape[0], block_size, c]
+    else:  # 1
+        sh = [block_indices.shape[0], r, block_size]
+
+    dt = M.dtype
+    M_res = np.empty(sh, dtype=dt)
+    if axis == 0:
+        for ir in range(block_size):
+            M_res[:, ir, :] = M[(block_indices*block_size+ir), :]
+    else:  # 1
+        for ic in range(block_size):
+            M_res[:, :, ic] = M[:, (block_indices*block_size+ic)]
+
+    return M_res
diff --git a/lib/matplotlib/tri/_tripcolor.py b/lib/matplotlib/tri/_tripcolor.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_tripcolor.py
@@ -0,0 +1,154 @@
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.collections import PolyCollection, TriMesh
+from matplotlib.colors import Normalize
+from matplotlib.tri._triangulation import Triangulation
+
+
+def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
+              vmax=None, shading='flat', facecolors=None, **kwargs):
+    """
+    Create a pseudocolor plot of an unstructured triangular grid.
+
+    Call signatures::
+
+      tripcolor(triangulation, c, *, ...)
+      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)
+
+    The triangular grid can be specified either by passing a `.Triangulation`
+    object as the first parameter, or by passing the points *x*, *y* and
+    optionally the *triangles* and a *mask*. See `.Triangulation` for an
+    explanation of these parameters.
+
+    It is possible to pass the triangles positionally, i.e.
+    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
+    For more clarity, pass *triangles* via keyword argument.
+
+    If neither of *triangulation* or *triangles* are given, the triangulation
+    is calculated on the fly. In this case, it does not make sense to provide
+    colors at the triangle faces via *c* or *facecolors* because there are
+    multiple possible triangulations for a group of points and you don't know
+    which triangles will be constructed.
+
+    Parameters
+    ----------
+    triangulation : `.Triangulation`
+        An already created triangular grid.
+    x, y, triangles, mask
+        Parameters defining the triangular grid. See `.Triangulation`.
+        This is mutually exclusive with specifying *triangulation*.
+    c : array-like
+        The color values, either for the points or for the triangles. Which one
+        is automatically inferred from the length of *c*, i.e. does it match
+        the number of points or the number of triangles. If there are the same
+        number of points and triangles in the triangulation it is assumed that
+        color values are defined at points; to force the use of color values at
+        triangles use the keyword argument ``facecolors=c`` instead of just
+        ``c``.
+        This parameter is position-only.
+    facecolors : array-like, optional
+        Can be used alternatively to *c* to specify colors at the triangle
+        faces. This parameter takes precedence over *c*.
+    shading : {'flat', 'gouraud'}, default: 'flat'
+        If  'flat' and the color values *c* are defined at points, the color
+        values used for each triangle are from the mean c of the triangle's
+        three points. If *shading* is 'gouraud' then color values must be
+        defined at points.
+    other_parameters
+        All other parameters are the same as for `~.Axes.pcolor`.
+    """
+    _api.check_in_list(['flat', 'gouraud'], shading=shading)
+
+    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
+
+    # Parse the color to be in one of (the other variable will be None):
+    # - facecolors: if specified at the triangle faces
+    # - point_colors: if specified at the points
+    if facecolors is not None:
+        if args:
+            _api.warn_external(
+                "Positional parameter c has no effect when the keyword "
+                "facecolors is given")
+        point_colors = None
+        if len(facecolors) != len(tri.triangles):
+            raise ValueError("The length of facecolors must match the number "
+                             "of triangles")
+    else:
+        # Color from positional parameter c
+        if not args:
+            raise TypeError(
+                "tripcolor() missing 1 required positional argument: 'c'; or "
+                "1 required keyword-only argument: 'facecolors'")
+        elif len(args) > 1:
+            _api.warn_deprecated(
+                "3.6", message=f"Additional positional parameters "
+                f"{args[1:]!r} are ignored; support for them is deprecated "
+                f"since %(since)s and will be removed %(removal)s")
+        c = np.asarray(args[0])
+        if len(c) == len(tri.x):
+            # having this before the len(tri.triangles) comparison gives
+            # precedence to nodes if there are as many nodes as triangles
+            point_colors = c
+            facecolors = None
+        elif len(c) == len(tri.triangles):
+            point_colors = None
+            facecolors = c
+        else:
+            raise ValueError('The length of c must match either the number '
+                             'of points or the number of triangles')
+
+    # Handling of linewidths, shading, edgecolors and antialiased as
+    # in Axes.pcolor
+    linewidths = (0.25,)
+    if 'linewidth' in kwargs:
+        kwargs['linewidths'] = kwargs.pop('linewidth')
+    kwargs.setdefault('linewidths', linewidths)
+
+    edgecolors = 'none'
+    if 'edgecolor' in kwargs:
+        kwargs['edgecolors'] = kwargs.pop('edgecolor')
+    ec = kwargs.setdefault('edgecolors', edgecolors)
+
+    if 'antialiased' in kwargs:
+        kwargs['antialiaseds'] = kwargs.pop('antialiased')
+    if 'antialiaseds' not in kwargs and ec.lower() == "none":
+        kwargs['antialiaseds'] = False
+
+    _api.check_isinstance((Normalize, None), norm=norm)
+    if shading == 'gouraud':
+        if facecolors is not None:
+            raise ValueError(
+                "shading='gouraud' can only be used when the colors "
+                "are specified at the points, not at the faces.")
+        collection = TriMesh(tri, alpha=alpha, array=point_colors,
+                             cmap=cmap, norm=norm, **kwargs)
+    else:  # 'flat'
+        # Vertices of triangles.
+        maskedTris = tri.get_masked_triangles()
+        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)
+
+        # Color values.
+        if facecolors is None:
+            # One color per triangle, the mean of the 3 vertex color values.
+            colors = point_colors[maskedTris].mean(axis=1)
+        elif tri.mask is not None:
+            # Remove color values of masked triangles.
+            colors = facecolors[~tri.mask]
+        else:
+            colors = facecolors
+        collection = PolyCollection(verts, alpha=alpha, array=colors,
+                                    cmap=cmap, norm=norm, **kwargs)
+
+    collection._scale_norm(norm, vmin, vmax)
+    ax.grid(False)
+
+    minx = tri.x.min()
+    maxx = tri.x.max()
+    miny = tri.y.min()
+    maxy = tri.y.max()
+    corners = (minx, miny), (maxx, maxy)
+    ax.update_datalim(corners)
+    ax.autoscale_view()
+    ax.add_collection(collection)
+    return collection
diff --git a/lib/matplotlib/tri/_triplot.py b/lib/matplotlib/tri/_triplot.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_triplot.py
@@ -0,0 +1,86 @@
+import numpy as np
+from matplotlib.tri._triangulation import Triangulation
+import matplotlib.cbook as cbook
+import matplotlib.lines as mlines
+
+
+def triplot(ax, *args, **kwargs):
+    """
+    Draw an unstructured triangular grid as lines and/or markers.
+
+    Call signatures::
+
+      triplot(triangulation, ...)
+      triplot(x, y, [triangles], *, [mask=mask], ...)
+
+    The triangular grid can be specified either by passing a `.Triangulation`
+    object as the first parameter, or by passing the points *x*, *y* and
+    optionally the *triangles* and a *mask*. If neither of *triangulation* or
+    *triangles* are given, the triangulation is calculated on the fly.
+
+    Parameters
+    ----------
+    triangulation : `.Triangulation`
+        An already created triangular grid.
+    x, y, triangles, mask
+        Parameters defining the triangular grid. See `.Triangulation`.
+        This is mutually exclusive with specifying *triangulation*.
+    other_parameters
+        All other args and kwargs are forwarded to `~.Axes.plot`.
+
+    Returns
+    -------
+    lines : `~matplotlib.lines.Line2D`
+        The drawn triangles edges.
+    markers : `~matplotlib.lines.Line2D`
+        The drawn marker nodes.
+    """
+    import matplotlib.axes
+
+    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
+    x, y, edges = (tri.x, tri.y, tri.edges)
+
+    # Decode plot format string, e.g., 'ro-'
+    fmt = args[0] if args else ""
+    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)
+
+    # Insert plot format string into a copy of kwargs (kwargs values prevail).
+    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
+    for key, val in zip(('linestyle', 'marker', 'color'),
+                        (linestyle, marker, color)):
+        if val is not None:
+            kw.setdefault(key, val)
+
+    # Draw lines without markers.
+    # Note 1: If we drew markers here, most markers would be drawn more than
+    #         once as they belong to several edges.
+    # Note 2: We insert nan values in the flattened edges arrays rather than
+    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
+    #         as it considerably speeds-up code execution.
+    linestyle = kw['linestyle']
+    kw_lines = {
+        **kw,
+        'marker': 'None',  # No marker to draw.
+        'zorder': kw.get('zorder', 1),  # Path default zorder is used.
+    }
+    if linestyle not in [None, 'None', '', ' ']:
+        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
+        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
+        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
+                            **kw_lines)
+    else:
+        tri_lines = ax.plot([], [], **kw_lines)
+
+    # Draw markers separately.
+    marker = kw['marker']
+    kw_markers = {
+        **kw,
+        'linestyle': 'None',  # No line to draw.
+    }
+    kw_markers.pop('label', None)
+    if marker not in [None, 'None', '', ' ']:
+        tri_markers = ax.plot(x, y, **kw_markers)
+    else:
+        tri_markers = ax.plot([], [], **kw_markers)
+
+    return tri_lines + tri_markers
diff --git a/lib/matplotlib/tri/_trirefine.py b/lib/matplotlib/tri/_trirefine.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_trirefine.py
@@ -0,0 +1,307 @@
+"""
+Mesh refinement for triangular grids.
+"""
+
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri._triangulation import Triangulation
+import matplotlib.tri._triinterpolate
+
+
+class TriRefiner:
+    """
+    Abstract base class for classes implementing mesh refinement.
+
+    A TriRefiner encapsulates a Triangulation object and provides tools for
+    mesh refinement and interpolation.
+
+    Derived classes must implement:
+
+    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
+      the optional keyword arguments *kwargs* are defined in each
+      TriRefiner concrete implementation, and which returns:
+
+      - a refined triangulation,
+      - optionally (depending on *return_tri_index*), for each
+        point of the refined triangulation: the index of
+        the initial triangulation triangle to which it belongs.
+
+    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:
+
+      - *z* array of field values (to refine) defined at the base
+        triangulation nodes,
+      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
+      - the other optional keyword arguments *kwargs* are defined in
+        each TriRefiner concrete implementation;
+
+      and which returns (as a tuple) a refined triangular mesh and the
+      interpolated values of the field at the refined triangulation nodes.
+    """
+
+    def __init__(self, triangulation):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+
+class UniformTriRefiner(TriRefiner):
+    """
+    Uniform mesh refinement by recursive subdivisions.
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The encapsulated triangulation (to be refined)
+    """
+#    See Also
+#    --------
+#    :class:`~matplotlib.tri.CubicTriInterpolator` and
+#    :class:`~matplotlib.tri.TriAnalyzer`.
+#    """
+    def __init__(self, triangulation):
+        super().__init__(triangulation)
+
+    def refine_triangulation(self, return_tri_index=False, subdiv=3):
+        """
+        Compute an uniformly refined triangulation *refi_triangulation* of
+        the encapsulated :attr:`triangulation`.
+
+        This function refines the encapsulated triangulation by splitting each
+        father triangle into 4 child sub-triangles built on the edges midside
+        nodes, recursing *subdiv* times.  In the end, each triangle is hence
+        divided into ``4**subdiv`` child triangles.
+
+        Parameters
+        ----------
+        return_tri_index : bool, default: False
+            Whether an index table indicating the father triangle index of each
+            point is returned.
+        subdiv : int, default: 3
+            Recursion level for the subdivision.
+            Each triangle is divided into ``4**subdiv`` child triangles;
+            hence, the default results in 64 refined subtriangles for each
+            triangle of the initial triangulation.
+
+        Returns
+        -------
+        refi_triangulation : `~matplotlib.tri.Triangulation`
+            The refined triangulation.
+        found_index : int array
+            Index of the initial triangulation containing triangle, for each
+            point of *refi_triangulation*.
+            Returned only if *return_tri_index* is set to True.
+        """
+        refi_triangulation = self._triangulation
+        ntri = refi_triangulation.triangles.shape[0]
+
+        # Computes the triangulation ancestors numbers in the reference
+        # triangulation.
+        ancestors = np.arange(ntri, dtype=np.int32)
+        for _ in range(subdiv):
+            refi_triangulation, ancestors = self._refine_triangulation_once(
+                refi_triangulation, ancestors)
+        refi_npts = refi_triangulation.x.shape[0]
+        refi_triangles = refi_triangulation.triangles
+
+        # Now we compute found_index table if needed
+        if return_tri_index:
+            # We have to initialize found_index with -1 because some nodes
+            # may very well belong to no triangle at all, e.g., in case of
+            # Delaunay Triangulation with DuplicatePointWarning.
+            found_index = np.full(refi_npts, -1, dtype=np.int32)
+            tri_mask = self._triangulation.mask
+            if tri_mask is None:
+                found_index[refi_triangles] = np.repeat(ancestors,
+                                                        3).reshape(-1, 3)
+            else:
+                # There is a subtlety here: we want to avoid whenever possible
+                # that refined points container is a masked triangle (which
+                # would result in artifacts in plots).
+                # So we impose the numbering from masked ancestors first,
+                # then overwrite it with unmasked ancestor numbers.
+                ancestor_mask = tri_mask[ancestors]
+                found_index[refi_triangles[ancestor_mask, :]
+                            ] = np.repeat(ancestors[ancestor_mask],
+                                          3).reshape(-1, 3)
+                found_index[refi_triangles[~ancestor_mask, :]
+                            ] = np.repeat(ancestors[~ancestor_mask],
+                                          3).reshape(-1, 3)
+            return refi_triangulation, found_index
+        else:
+            return refi_triangulation
+
+    def refine_field(self, z, triinterpolator=None, subdiv=3):
+        """
+        Refine a field defined on the encapsulated triangulation.
+
+        Parameters
+        ----------
+        z : (npoints,) array-like
+            Values of the field to refine, defined at the nodes of the
+            encapsulated triangulation. (``n_points`` is the number of points
+            in the initial triangulation)
+        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
+            Interpolator used for field interpolation. If not specified,
+            a `~matplotlib.tri.CubicTriInterpolator` will be used.
+        subdiv : int, default: 3
+            Recursion level for the subdivision.
+            Each triangle is divided into ``4**subdiv`` child triangles.
+
+        Returns
+        -------
+        refi_tri : `~matplotlib.tri.Triangulation`
+             The returned refined triangulation.
+        refi_z : 1D array of length: *refi_tri* node count.
+             The returned interpolated field (at *refi_tri* nodes).
+        """
+        if triinterpolator is None:
+            interp = matplotlib.tri.CubicTriInterpolator(
+                self._triangulation, z)
+        else:
+            _api.check_isinstance(matplotlib.tri.TriInterpolator,
+                                  triinterpolator=triinterpolator)
+            interp = triinterpolator
+
+        refi_tri, found_index = self.refine_triangulation(
+            subdiv=subdiv, return_tri_index=True)
+        refi_z = interp._interpolate_multikeys(
+            refi_tri.x, refi_tri.y, tri_index=found_index)[0]
+        return refi_tri, refi_z
+
+    @staticmethod
+    def _refine_triangulation_once(triangulation, ancestors=None):
+        """
+        Refine a `.Triangulation` by splitting each triangle into 4
+        child-masked_triangles built on the edges midside nodes.
+
+        Masked triangles, if present, are also split, but their children
+        returned masked.
+
+        If *ancestors* is not provided, returns only a new triangulation:
+        child_triangulation.
+
+        If the array-like key table *ancestor* is given, it shall be of shape
+        (ntri,) where ntri is the number of *triangulation* masked_triangles.
+        In this case, the function returns
+        (child_triangulation, child_ancestors)
+        child_ancestors is defined so that the 4 child masked_triangles share
+        the same index as their father: child_ancestors.shape = (4 * ntri,).
+        """
+
+        x = triangulation.x
+        y = triangulation.y
+
+        #    According to tri.triangulation doc:
+        #         neighbors[i, j] is the triangle that is the neighbor
+        #         to the edge from point index masked_triangles[i, j] to point
+        #         index masked_triangles[i, (j+1)%3].
+        neighbors = triangulation.neighbors
+        triangles = triangulation.triangles
+        npts = np.shape(x)[0]
+        ntri = np.shape(triangles)[0]
+        if ancestors is not None:
+            ancestors = np.asarray(ancestors)
+            if np.shape(ancestors) != (ntri,):
+                raise ValueError(
+                    "Incompatible shapes provide for triangulation"
+                    ".masked_triangles and ancestors: {0} and {1}".format(
+                        np.shape(triangles), np.shape(ancestors)))
+
+        # Initiating tables refi_x and refi_y of the refined triangulation
+        # points
+        # hint: each apex is shared by 2 masked_triangles except the borders.
+        borders = np.sum(neighbors == -1)
+        added_pts = (3*ntri + borders) // 2
+        refi_npts = npts + added_pts
+        refi_x = np.zeros(refi_npts)
+        refi_y = np.zeros(refi_npts)
+
+        # First part of refi_x, refi_y is just the initial points
+        refi_x[:npts] = x
+        refi_y[:npts] = y
+
+        # Second part contains the edge midside nodes.
+        # Each edge belongs to 1 triangle (if border edge) or is shared by 2
+        # masked_triangles (interior edge).
+        # We first build 2 * ntri arrays of edge starting nodes (edge_elems,
+        # edge_apexes); we then extract only the masters to avoid overlaps.
+        # The so-called 'master' is the triangle with biggest index
+        # The 'slave' is the triangle with lower index
+        # (can be -1 if border edge)
+        # For slave and master we will identify the apex pointing to the edge
+        # start
+        edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
+        edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
+        edge_neighbors = neighbors[edge_elems, edge_apexes]
+        mask_masters = (edge_elems > edge_neighbors)
+
+        # Identifying the "masters" and adding to refi_x, refi_y vec
+        masters = edge_elems[mask_masters]
+        apex_masters = edge_apexes[mask_masters]
+        x_add = (x[triangles[masters, apex_masters]] +
+                 x[triangles[masters, (apex_masters+1) % 3]]) * 0.5
+        y_add = (y[triangles[masters, apex_masters]] +
+                 y[triangles[masters, (apex_masters+1) % 3]]) * 0.5
+        refi_x[npts:] = x_add
+        refi_y[npts:] = y_add
+
+        # Building the new masked_triangles; each old masked_triangles hosts
+        # 4 new masked_triangles
+        # there are 6 pts to identify per 'old' triangle, 3 new_pt_corner and
+        # 3 new_pt_midside
+        new_pt_corner = triangles
+
+        # What is the index in refi_x, refi_y of point at middle of apex iapex
+        #  of elem ielem ?
+        # If ielem is the apex master: simple count, given the way refi_x was
+        #  built.
+        # If ielem is the apex slave: yet we do not know; but we will soon
+        # using the neighbors table.
+        new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
+        cum_sum = npts
+        for imid in range(3):
+            mask_st_loc = (imid == apex_masters)
+            n_masters_loc = np.sum(mask_st_loc)
+            elem_masters_loc = masters[mask_st_loc]
+            new_pt_midside[:, imid][elem_masters_loc] = np.arange(
+                n_masters_loc, dtype=np.int32) + cum_sum
+            cum_sum += n_masters_loc
+
+        # Now dealing with slave elems.
+        # for each slave element we identify the master and then the inode
+        # once slave_masters is identified, slave_masters_apex is such that:
+        # neighbors[slaves_masters, slave_masters_apex] == slaves
+        mask_slaves = np.logical_not(mask_masters)
+        slaves = edge_elems[mask_slaves]
+        slaves_masters = edge_neighbors[mask_slaves]
+        diff_table = np.abs(neighbors[slaves_masters, :] -
+                            np.outer(slaves, np.ones(3, dtype=np.int32)))
+        slave_masters_apex = np.argmin(diff_table, axis=1)
+        slaves_apex = edge_apexes[mask_slaves]
+        new_pt_midside[slaves, slaves_apex] = new_pt_midside[
+            slaves_masters, slave_masters_apex]
+
+        # Builds the 4 child masked_triangles
+        child_triangles = np.empty([ntri*4, 3], dtype=np.int32)
+        child_triangles[0::4, :] = np.vstack([
+            new_pt_corner[:, 0], new_pt_midside[:, 0],
+            new_pt_midside[:, 2]]).T
+        child_triangles[1::4, :] = np.vstack([
+            new_pt_corner[:, 1], new_pt_midside[:, 1],
+            new_pt_midside[:, 0]]).T
+        child_triangles[2::4, :] = np.vstack([
+            new_pt_corner[:, 2], new_pt_midside[:, 2],
+            new_pt_midside[:, 1]]).T
+        child_triangles[3::4, :] = np.vstack([
+            new_pt_midside[:, 0], new_pt_midside[:, 1],
+            new_pt_midside[:, 2]]).T
+        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
+
+        # Builds the child mask
+        if triangulation.mask is not None:
+            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
+
+        if ancestors is None:
+            return child_triangulation
+        else:
+            return child_triangulation, np.repeat(ancestors, 4)
diff --git a/lib/matplotlib/tri/_tritools.py b/lib/matplotlib/tri/_tritools.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_tritools.py
@@ -0,0 +1,263 @@
+"""
+Tools for triangular grids.
+"""
+
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri import Triangulation
+
+
+class TriAnalyzer:
+    """
+    Define basic tools for triangular mesh analysis and improvement.
+
+    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
+    tools for mesh analysis and mesh improvement.
+
+    Attributes
+    ----------
+    scale_factors
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The encapsulated triangulation to analyze.
+    """
+
+    def __init__(self, triangulation):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+    @property
+    def scale_factors(self):
+        """
+        Factors to rescale the triangulation into a unit square.
+
+        Returns
+        -------
+        (float, float)
+            Scaling factors (kx, ky) so that the triangulation
+            ``[triangulation.x * kx, triangulation.y * ky]``
+            fits exactly inside a unit square.
+        """
+        compressed_triangles = self._triangulation.get_masked_triangles()
+        node_used = (np.bincount(np.ravel(compressed_triangles),
+                                 minlength=self._triangulation.x.size) != 0)
+        return (1 / np.ptp(self._triangulation.x[node_used]),
+                1 / np.ptp(self._triangulation.y[node_used]))
+
+    def circle_ratios(self, rescale=True):
+        """
+        Return a measure of the triangulation triangles flatness.
+
+        The ratio of the incircle radius over the circumcircle radius is a
+        widely used indicator of a triangle flatness.
+        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
+        triangles. Circle ratios below 0.01 denote very flat triangles.
+
+        To avoid unduly low values due to a difference of scale between the 2
+        axis, the triangular mesh can first be rescaled to fit inside a unit
+        square with `scale_factors` (Only if *rescale* is True, which is
+        its default value).
+
+        Parameters
+        ----------
+        rescale : bool, default: True
+            If True, internally rescale (based on `scale_factors`), so that the
+            (unmasked) triangles fit exactly inside a unit square mesh.
+
+        Returns
+        -------
+        masked array
+            Ratio of the incircle radius over the circumcircle radius, for
+            each 'rescaled' triangle of the encapsulated triangulation.
+            Values corresponding to masked triangles are masked out.
+
+        """
+        # Coords rescaling
+        if rescale:
+            (kx, ky) = self.scale_factors
+        else:
+            (kx, ky) = (1.0, 1.0)
+        pts = np.vstack([self._triangulation.x*kx,
+                         self._triangulation.y*ky]).T
+        tri_pts = pts[self._triangulation.triangles]
+        # Computes the 3 side lengths
+        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
+        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
+        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
+        a = np.hypot(a[:, 0], a[:, 1])
+        b = np.hypot(b[:, 0], b[:, 1])
+        c = np.hypot(c[:, 0], c[:, 1])
+        # circumcircle and incircle radii
+        s = (a+b+c)*0.5
+        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
+        # We have to deal with flat triangles with infinite circum_radius
+        bool_flat = (prod == 0.)
+        if np.any(bool_flat):
+            # Pathologic flow
+            ntri = tri_pts.shape[0]
+            circum_radius = np.empty(ntri, dtype=np.float64)
+            circum_radius[bool_flat] = np.inf
+            abc = a*b*c
+            circum_radius[~bool_flat] = abc[~bool_flat] / (
+                4.0*np.sqrt(prod[~bool_flat]))
+        else:
+            # Normal optimized flow
+            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
+        in_radius = (a*b*c) / (4.0*circum_radius*s)
+        circle_ratio = in_radius/circum_radius
+        mask = self._triangulation.mask
+        if mask is None:
+            return circle_ratio
+        else:
+            return np.ma.array(circle_ratio, mask=mask)
+
+    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
+        """
+        Eliminate excessively flat border triangles from the triangulation.
+
+        Returns a mask *new_mask* which allows to clean the encapsulated
+        triangulation from its border-located flat triangles
+        (according to their :meth:`circle_ratios`).
+        This mask is meant to be subsequently applied to the triangulation
+        using `.Triangulation.set_mask`.
+        *new_mask* is an extension of the initial triangulation mask
+        in the sense that an initially masked triangle will remain masked.
+
+        The *new_mask* array is computed recursively; at each step flat
+        triangles are removed only if they share a side with the current mesh
+        border. Thus no new holes in the triangulated domain will be created.
+
+        Parameters
+        ----------
+        min_circle_ratio : float, default: 0.01
+            Border triangles with incircle/circumcircle radii ratio r/R will
+            be removed if r/R < *min_circle_ratio*.
+        rescale : bool, default: True
+            If True, first, internally rescale (based on `scale_factors`) so
+            that the (unmasked) triangles fit exactly inside a unit square
+            mesh.  This rescaling accounts for the difference of scale which
+            might exist between the 2 axis.
+
+        Returns
+        -------
+        array of bool
+            Mask to apply to encapsulated triangulation.
+            All the initially masked triangles remain masked in the
+            *new_mask*.
+
+        Notes
+        -----
+        The rationale behind this function is that a Delaunay
+        triangulation - of an unstructured set of points - sometimes contains
+        almost flat triangles at its border, leading to artifacts in plots
+        (especially for high-resolution contouring).
+        Masked with computed *new_mask*, the encapsulated
+        triangulation would contain no more unmasked border triangles
+        with a circle ratio below *min_circle_ratio*, thus improving the
+        mesh quality for subsequent plots or interpolation.
+        """
+        # Recursively computes the mask_current_borders, true if a triangle is
+        # at the border of the mesh OR touching the border through a chain of
+        # invalid aspect ratio masked_triangles.
+        ntri = self._triangulation.triangles.shape[0]
+        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
+
+        current_mask = self._triangulation.mask
+        if current_mask is None:
+            current_mask = np.zeros(ntri, dtype=bool)
+        valid_neighbors = np.copy(self._triangulation.neighbors)
+        renum_neighbors = np.arange(ntri, dtype=np.int32)
+        nadd = -1
+        while nadd != 0:
+            # The active wavefront is the triangles from the border (unmasked
+            # but with a least 1 neighbor equal to -1
+            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
+            # The element from the active wavefront will be masked if their
+            # circle ratio is bad.
+            added_mask = wavefront & mask_bad_ratio
+            current_mask = added_mask | current_mask
+            nadd = np.sum(added_mask)
+
+            # now we have to update the tables valid_neighbors
+            valid_neighbors[added_mask, :] = -1
+            renum_neighbors[added_mask] = -1
+            valid_neighbors = np.where(valid_neighbors == -1, -1,
+                                       renum_neighbors[valid_neighbors])
+
+        return np.ma.filled(current_mask, True)
+
+    def _get_compressed_triangulation(self):
+        """
+        Compress (if masked) the encapsulated triangulation.
+
+        Returns minimal-length triangles array (*compressed_triangles*) and
+        coordinates arrays (*compressed_x*, *compressed_y*) that can still
+        describe the unmasked triangles of the encapsulated triangulation.
+
+        Returns
+        -------
+        compressed_triangles : array-like
+            the returned compressed triangulation triangles
+        compressed_x : array-like
+            the returned compressed triangulation 1st coordinate
+        compressed_y : array-like
+            the returned compressed triangulation 2nd coordinate
+        tri_renum : int array
+            renumbering table to translate the triangle numbers from the
+            encapsulated triangulation into the new (compressed) renumbering.
+            -1 for masked triangles (deleted from *compressed_triangles*).
+        node_renum : int array
+            renumbering table to translate the point numbers from the
+            encapsulated triangulation into the new (compressed) renumbering.
+            -1 for unused points (i.e. those deleted from *compressed_x* and
+            *compressed_y*).
+
+        """
+        # Valid triangles and renumbering
+        tri_mask = self._triangulation.mask
+        compressed_triangles = self._triangulation.get_masked_triangles()
+        ntri = self._triangulation.triangles.shape[0]
+        if tri_mask is not None:
+            tri_renum = self._total_to_compress_renum(~tri_mask)
+        else:
+            tri_renum = np.arange(ntri, dtype=np.int32)
+
+        # Valid nodes and renumbering
+        valid_node = (np.bincount(np.ravel(compressed_triangles),
+                                  minlength=self._triangulation.x.size) != 0)
+        compressed_x = self._triangulation.x[valid_node]
+        compressed_y = self._triangulation.y[valid_node]
+        node_renum = self._total_to_compress_renum(valid_node)
+
+        # Now renumbering the valid triangles nodes
+        compressed_triangles = node_renum[compressed_triangles]
+
+        return (compressed_triangles, compressed_x, compressed_y, tri_renum,
+                node_renum)
+
+    @staticmethod
+    def _total_to_compress_renum(valid):
+        """
+        Parameters
+        ----------
+        valid : 1D bool array
+            Validity mask.
+
+        Returns
+        -------
+        int array
+            Array so that (`valid_array` being a compressed array
+            based on a `masked_array` with mask ~*valid*):
+
+            - For all i with valid[i] = True:
+              valid_array[renum[i]] = masked_array[i]
+            - For all i with valid[i] = False:
+              renum[i] = -1 (invalid value)
+        """
+        renum = np.full(np.size(valid), -1, dtype=np.int32)
+        n_valid = np.sum(valid)
+        renum[valid] = np.arange(n_valid, dtype=np.int32)
+        return renum
diff --git a/lib/matplotlib/tri/triangulation.py b/lib/matplotlib/tri/triangulation.py
--- a/lib/matplotlib/tri/triangulation.py
+++ b/lib/matplotlib/tri/triangulation.py
@@ -1,240 +1,9 @@
-import numpy as np
-
+from ._triangulation import *  # noqa: F401, F403
 from matplotlib import _api
 
 
-class Triangulation:
-    """
-    An unstructured triangular grid consisting of npoints points and
-    ntri triangles.  The triangles can either be specified by the user
-    or automatically generated using a Delaunay triangulation.
-
-    Parameters
-    ----------
-    x, y : (npoints,) array-like
-        Coordinates of grid points.
-    triangles : (ntri, 3) array-like of int, optional
-        For each triangle, the indices of the three points that make
-        up the triangle, ordered in an anticlockwise manner.  If not
-        specified, the Delaunay triangulation is calculated.
-    mask : (ntri,) array-like of bool, optional
-        Which triangles are masked out.
-
-    Attributes
-    ----------
-    triangles : (ntri, 3) array of int
-        For each triangle, the indices of the three points that make
-        up the triangle, ordered in an anticlockwise manner. If you want to
-        take the *mask* into account, use `get_masked_triangles` instead.
-    mask : (ntri, 3) array of bool
-        Masked out triangles.
-    is_delaunay : bool
-        Whether the Triangulation is a calculated Delaunay
-        triangulation (where *triangles* was not specified) or not.
-
-    Notes
-    -----
-    For a Triangulation to be valid it must not have duplicate points,
-    triangles formed from colinear points, or overlapping triangles.
-    """
-    def __init__(self, x, y, triangles=None, mask=None):
-        from matplotlib import _qhull
-
-        self.x = np.asarray(x, dtype=np.float64)
-        self.y = np.asarray(y, dtype=np.float64)
-        if self.x.shape != self.y.shape or self.x.ndim != 1:
-            raise ValueError("x and y must be equal-length 1D arrays, but "
-                             f"found shapes {self.x.shape!r} and "
-                             f"{self.y.shape!r}")
-
-        self.mask = None
-        self._edges = None
-        self._neighbors = None
-        self.is_delaunay = False
-
-        if triangles is None:
-            # No triangulation specified, so use matplotlib._qhull to obtain
-            # Delaunay triangulation.
-            self.triangles, self._neighbors = _qhull.delaunay(x, y)
-            self.is_delaunay = True
-        else:
-            # Triangulation specified. Copy, since we may correct triangle
-            # orientation.
-            try:
-                self.triangles = np.array(triangles, dtype=np.int32, order='C')
-            except ValueError as e:
-                raise ValueError('triangles must be a (N, 3) int array, not '
-                                 f'{triangles!r}') from e
-            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
-                raise ValueError(
-                    'triangles must be a (N, 3) int array, but found shape '
-                    f'{self.triangles.shape!r}')
-            if self.triangles.max() >= len(self.x):
-                raise ValueError(
-                    'triangles are indices into the points and must be in the '
-                    f'range 0 <= i < {len(self.x)} but found value '
-                    f'{self.triangles.max()}')
-            if self.triangles.min() < 0:
-                raise ValueError(
-                    'triangles are indices into the points and must be in the '
-                    f'range 0 <= i < {len(self.x)} but found value '
-                    f'{self.triangles.min()}')
-
-        # Underlying C++ object is not created until first needed.
-        self._cpp_triangulation = None
-
-        # Default TriFinder not created until needed.
-        self._trifinder = None
-
-        self.set_mask(mask)
-
-    def calculate_plane_coefficients(self, z):
-        """
-        Calculate plane equation coefficients for all unmasked triangles from
-        the point (x, y) coordinates and specified z-array of shape (npoints).
-        The returned array has shape (npoints, 3) and allows z-value at (x, y)
-        position in triangle tri to be calculated using
-        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
-        """
-        return self.get_cpp_triangulation().calculate_plane_coefficients(z)
-
-    @property
-    def edges(self):
-        """
-        Return integer array of shape (nedges, 2) containing all edges of
-        non-masked triangles.
-
-        Each row defines an edge by its start point index and end point
-        index.  Each edge appears only once, i.e. for an edge between points
-        *i*  and *j*, there will only be either *(i, j)* or *(j, i)*.
-        """
-        if self._edges is None:
-            self._edges = self.get_cpp_triangulation().get_edges()
-        return self._edges
-
-    def get_cpp_triangulation(self):
-        """
-        Return the underlying C++ Triangulation object, creating it
-        if necessary.
-        """
-        from matplotlib import _tri
-        if self._cpp_triangulation is None:
-            self._cpp_triangulation = _tri.Triangulation(
-                self.x, self.y, self.triangles, self.mask, self._edges,
-                self._neighbors, not self.is_delaunay)
-        return self._cpp_triangulation
-
-    def get_masked_triangles(self):
-        """
-        Return an array of triangles taking the mask into account.
-        """
-        if self.mask is not None:
-            return self.triangles[~self.mask]
-        else:
-            return self.triangles
-
-    @staticmethod
-    def get_from_args_and_kwargs(*args, **kwargs):
-        """
-        Return a Triangulation object from the args and kwargs, and
-        the remaining args and kwargs with the consumed values removed.
-
-        There are two alternatives: either the first argument is a
-        Triangulation object, in which case it is returned, or the args
-        and kwargs are sufficient to create a new Triangulation to
-        return.  In the latter case, see Triangulation.__init__ for
-        the possible args and kwargs.
-        """
-        if isinstance(args[0], Triangulation):
-            triangulation, *args = args
-            if 'triangles' in kwargs:
-                _api.warn_external(
-                    "Passing the keyword 'triangles' has no effect when also "
-                    "passing a Triangulation")
-            if 'mask' in kwargs:
-                _api.warn_external(
-                    "Passing the keyword 'mask' has no effect when also "
-                    "passing a Triangulation")
-        else:
-            x, y, triangles, mask, args, kwargs = \
-                Triangulation._extract_triangulation_params(args, kwargs)
-            triangulation = Triangulation(x, y, triangles, mask)
-        return triangulation, args, kwargs
-
-    @staticmethod
-    def _extract_triangulation_params(args, kwargs):
-        x, y, *args = args
-        # Check triangles in kwargs then args.
-        triangles = kwargs.pop('triangles', None)
-        from_args = False
-        if triangles is None and args:
-            triangles = args[0]
-            from_args = True
-        if triangles is not None:
-            try:
-                triangles = np.asarray(triangles, dtype=np.int32)
-            except ValueError:
-                triangles = None
-        if triangles is not None and (triangles.ndim != 2 or
-                                      triangles.shape[1] != 3):
-            triangles = None
-        if triangles is not None and from_args:
-            args = args[1:]  # Consumed first item in args.
-        # Check for mask in kwargs.
-        mask = kwargs.pop('mask', None)
-        return x, y, triangles, mask, args, kwargs
-
-    def get_trifinder(self):
-        """
-        Return the default `matplotlib.tri.TriFinder` of this
-        triangulation, creating it if necessary.  This allows the same
-        TriFinder object to be easily shared.
-        """
-        if self._trifinder is None:
-            # Default TriFinder class.
-            from matplotlib.tri.trifinder import TrapezoidMapTriFinder
-            self._trifinder = TrapezoidMapTriFinder(self)
-        return self._trifinder
-
-    @property
-    def neighbors(self):
-        """
-        Return integer array of shape (ntri, 3) containing neighbor triangles.
-
-        For each triangle, the indices of the three triangles that
-        share the same edges, or -1 if there is no such neighboring
-        triangle.  ``neighbors[i, j]`` is the triangle that is the neighbor
-        to the edge from point index ``triangles[i, j]`` to point index
-        ``triangles[i, (j+1)%3]``.
-        """
-        if self._neighbors is None:
-            self._neighbors = self.get_cpp_triangulation().get_neighbors()
-        return self._neighbors
-
-    def set_mask(self, mask):
-        """
-        Set or clear the mask array.
-
-        Parameters
-        ----------
-        mask : None or bool array of length ntri
-        """
-        if mask is None:
-            self.mask = None
-        else:
-            self.mask = np.asarray(mask, dtype=bool)
-            if self.mask.shape != (self.triangles.shape[0],):
-                raise ValueError('mask array must have same length as '
-                                 'triangles array')
-
-        # Set mask in C++ Triangulation.
-        if self._cpp_triangulation is not None:
-            self._cpp_triangulation.set_mask(self.mask)
-
-        # Clear derived fields so they are recalculated when needed.
-        self._edges = None
-        self._neighbors = None
-
-        # Recalculate TriFinder if it exists.
-        if self._trifinder is not None:
-            self._trifinder._initialize()
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/tricontour.py b/lib/matplotlib/tri/tricontour.py
--- a/lib/matplotlib/tri/tricontour.py
+++ b/lib/matplotlib/tri/tricontour.py
@@ -1,271 +1,9 @@
-import numpy as np
+from ._tricontour import *  # noqa: F401, F403
+from matplotlib import _api
 
-from matplotlib import _docstring
-from matplotlib.contour import ContourSet
-from matplotlib.tri.triangulation import Triangulation
 
-
-@_docstring.dedent_interpd
-class TriContourSet(ContourSet):
-    """
-    Create and store a set of contour lines or filled regions for
-    a triangular grid.
-
-    This class is typically not instantiated directly by the user but by
-    `~.Axes.tricontour` and `~.Axes.tricontourf`.
-
-    %(contour_set_attributes)s
-    """
-    def __init__(self, ax, *args, **kwargs):
-        """
-        Draw triangular grid contour lines or filled regions,
-        depending on whether keyword arg *filled* is False
-        (default) or True.
-
-        The first argument of the initializer must be an `~.axes.Axes`
-        object.  The remaining arguments and keyword arguments
-        are described in the docstring of `~.Axes.tricontour`.
-        """
-        super().__init__(ax, *args, **kwargs)
-
-    def _process_args(self, *args, **kwargs):
-        """
-        Process args and kwargs.
-        """
-        if isinstance(args[0], TriContourSet):
-            C = args[0]._contour_generator
-            if self.levels is None:
-                self.levels = args[0].levels
-            self.zmin = args[0].zmin
-            self.zmax = args[0].zmax
-            self._mins = args[0]._mins
-            self._maxs = args[0]._maxs
-        else:
-            from matplotlib import _tri
-            tri, z = self._contour_args(args, kwargs)
-            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
-            self._mins = [tri.x.min(), tri.y.min()]
-            self._maxs = [tri.x.max(), tri.y.max()]
-
-        self._contour_generator = C
-        return kwargs
-
-    def _contour_args(self, args, kwargs):
-        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
-                                                                   **kwargs)
-        z = np.ma.asarray(args[0])
-        if z.shape != tri.x.shape:
-            raise ValueError('z array must have same length as triangulation x'
-                             ' and y arrays')
-
-        # z values must be finite, only need to check points that are included
-        # in the triangulation.
-        z_check = z[np.unique(tri.get_masked_triangles())]
-        if np.ma.is_masked(z_check):
-            raise ValueError('z must not contain masked points within the '
-                             'triangulation')
-        if not np.isfinite(z_check).all():
-            raise ValueError('z array must not contain non-finite values '
-                             'within the triangulation')
-
-        z = np.ma.masked_invalid(z, copy=False)
-        self.zmax = float(z_check.max())
-        self.zmin = float(z_check.min())
-        if self.logscale and self.zmin <= 0:
-            func = 'contourf' if self.filled else 'contour'
-            raise ValueError(f'Cannot {func} log of negative values.')
-        self._process_contour_level_args(args[1:])
-        return (tri, z)
-
-
-_docstring.interpd.update(_tricontour_doc="""
-Draw contour %%(type)s on an unstructured triangular grid.
-
-Call signatures::
-
-    %%(func)s(triangulation, z, [levels], ...)
-    %%(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)
-
-The triangular grid can be specified either by passing a `.Triangulation`
-object as the first parameter, or by passing the points *x*, *y* and
-optionally the *triangles* and a *mask*. See `.Triangulation` for an
-explanation of these parameters. If neither of *triangulation* or
-*triangles* are given, the triangulation is calculated on the fly.
-
-It is possible to pass *triangles* positionally, i.e.
-``%%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more
-clarity, pass *triangles* via keyword argument.
-
-Parameters
-----------
-triangulation : `.Triangulation`, optional
-    An already created triangular grid.
-
-x, y, triangles, mask
-    Parameters defining the triangular grid. See `.Triangulation`.
-    This is mutually exclusive with specifying *triangulation*.
-
-z : array-like
-    The height values over which the contour is drawn.  Color-mapping is
-    controlled by *cmap*, *norm*, *vmin*, and *vmax*.
-
-    .. note::
-        All values in *z* must be finite. Hence, nan and inf values must
-        either be removed or `~.Triangulation.set_mask` be used.
-
-levels : int or array-like, optional
-    Determines the number and positions of the contour lines / regions.
-
-    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
-    automatically choose no more than *n+1* "nice" contour levels between
-    between minimum and maximum numeric values of *Z*.
-
-    If array-like, draw contour lines at the specified levels.  The values must
-    be in increasing order.
-
-Returns
--------
-`~matplotlib.tri.TriContourSet`
-
-Other Parameters
-----------------
-colors : color string or sequence of colors, optional
-    The colors of the levels, i.e., the contour %%(type)s.
-
-    The sequence is cycled for the levels in ascending order. If the sequence
-    is shorter than the number of levels, it is repeated.
-
-    As a shortcut, single color strings may be used in place of one-element
-    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
-    same color. This shortcut does only work for color strings, not for other
-    ways of specifying colors.
-
-    By default (value *None*), the colormap specified by *cmap* will be used.
-
-alpha : float, default: 1
-    The alpha blending value, between 0 (transparent) and 1 (opaque).
-
-%(cmap_doc)s
-
-    This parameter is ignored if *colors* is set.
-
-%(norm_doc)s
-
-    This parameter is ignored if *colors* is set.
-
-%(vmin_vmax_doc)s
-
-    If *vmin* or *vmax* are not given, the default color scaling is based on
-    *levels*.
-
-    This parameter is ignored if *colors* is set.
-
-origin : {*None*, 'upper', 'lower', 'image'}, default: None
-    Determines the orientation and exact position of *z* by specifying the
-    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.
-
-    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.
-    - 'lower': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
-    - 'upper': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
-    - 'image': Use the value from :rc:`image.origin`.
-
-extent : (x0, x1, y0, y1), optional
-    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
-    gives the outer pixel boundaries. In this case, the position of z[0, 0] is
-    the center of the pixel, not a corner. If *origin* is *None*, then
-    (*x0*, *y0*) is the position of z[0, 0], and (*x1*, *y1*) is the position
-    of z[-1, -1].
-
-    This argument is ignored if *X* and *Y* are specified in the call to
-    contour.
-
-locator : ticker.Locator subclass, optional
-    The locator is used to determine the contour levels if they are not given
-    explicitly via *levels*.
-    Defaults to `~.ticker.MaxNLocator`.
-
-extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
-    Determines the ``%%(func)s``-coloring of values that are outside the
-    *levels* range.
-
-    If 'neither', values outside the *levels* range are not colored.  If 'min',
-    'max' or 'both', color the values below, above or below and above the
-    *levels* range.
-
-    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
-    under/over values of the `.Colormap`. Note that most colormaps do not have
-    dedicated colors for these by default, so that the over and under values
-    are the edge values of the colormap.  You may want to set these values
-    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.
-
-    .. note::
-
-        An existing `.TriContourSet` does not get notified if properties of its
-        colormap are changed. Therefore, an explicit call to
-        `.ContourSet.changed()` is needed after modifying the colormap. The
-        explicit call can be left out, if a colorbar is assigned to the
-        `.TriContourSet` because it internally calls `.ContourSet.changed()`.
-
-xunits, yunits : registered units, optional
-    Override axis units by specifying an instance of a
-    :class:`matplotlib.units.ConversionInterface`.
-
-antialiased : bool, optional
-    Enable antialiasing, overriding the defaults.  For
-    filled contours, the default is *True*.  For line contours,
-    it is taken from :rc:`lines.antialiased`.""" % _docstring.interpd.params)
-
-
-@_docstring.Substitution(func='tricontour', type='lines')
-@_docstring.dedent_interpd
-def tricontour(ax, *args, **kwargs):
-    """
-    %(_tricontour_doc)s
-
-    linewidths : float or array-like, default: :rc:`contour.linewidth`
-        The line width of the contour lines.
-
-        If a number, all levels will be plotted with this linewidth.
-
-        If a sequence, the levels in ascending order will be plotted with
-        the linewidths in the order specified.
-
-        If None, this falls back to :rc:`lines.linewidth`.
-
-    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
-        If *linestyles* is *None*, the default is 'solid' unless the lines are
-        monochrome.  In that case, negative contours will take their linestyle
-        from :rc:`contour.negative_linestyle` setting.
-
-        *linestyles* can also be an iterable of the above strings specifying a
-        set of linestyles to be used. If this iterable is shorter than the
-        number of contour levels it will be repeated as necessary.
-    """
-    kwargs['filled'] = False
-    return TriContourSet(ax, *args, **kwargs)
-
-
-@_docstring.Substitution(func='tricontourf', type='regions')
-@_docstring.dedent_interpd
-def tricontourf(ax, *args, **kwargs):
-    """
-    %(_tricontour_doc)s
-
-    hatches : list[str], optional
-        A list of cross hatch patterns to use on the filled areas.
-        If None, no hatching will be added to the contour.
-        Hatching is supported in the PostScript, PDF, SVG and Agg
-        backends only.
-
-    Notes
-    -----
-    `.tricontourf` fills intervals that are closed at the top; that is, for
-    boundaries *z1* and *z2*, the filled region is::
-
-        z1 < Z <= z2
-
-    except for the lowest interval, which is closed on both sides (i.e. it
-    includes the lowest value).
-    """
-    kwargs['filled'] = True
-    return TriContourSet(ax, *args, **kwargs)
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/trifinder.py b/lib/matplotlib/tri/trifinder.py
--- a/lib/matplotlib/tri/trifinder.py
+++ b/lib/matplotlib/tri/trifinder.py
@@ -1,93 +1,9 @@
-import numpy as np
-
+from ._trifinder import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri import Triangulation
-
-
-class TriFinder:
-    """
-    Abstract base class for classes used to find the triangles of a
-    Triangulation in which (x, y) points lie.
-
-    Rather than instantiate an object of a class derived from TriFinder, it is
-    usually better to use the function `.Triangulation.get_trifinder`.
-
-    Derived classes implement __call__(x, y) where x and y are array-like point
-    coordinates of the same shape.
-    """
-
-    def __init__(self, triangulation):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-
-class TrapezoidMapTriFinder(TriFinder):
-    """
-    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
-    map algorithm from the book "Computational Geometry, Algorithms and
-    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
-    and O. Schwarzkopf.
-
-    The triangulation must be valid, i.e. it must not have duplicate points,
-    triangles formed from colinear points, or overlapping triangles.  The
-    algorithm has some tolerance to triangles formed from colinear points, but
-    this should not be relied upon.
-    """
-
-    def __init__(self, triangulation):
-        from matplotlib import _tri
-        super().__init__(triangulation)
-        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
-            triangulation.get_cpp_triangulation())
-        self._initialize()
-
-    def __call__(self, x, y):
-        """
-        Return an array containing the indices of the triangles in which the
-        specified *x*, *y* points lie, or -1 for points that do not lie within
-        a triangle.
-
-        *x*, *y* are array-like x and y coordinates of the same shape and any
-        number of dimensions.
-
-        Returns integer array with the same shape and *x* and *y*.
-        """
-        x = np.asarray(x, dtype=np.float64)
-        y = np.asarray(y, dtype=np.float64)
-        if x.shape != y.shape:
-            raise ValueError("x and y must be array-like with the same shape")
-
-        # C++ does the heavy lifting, and expects 1D arrays.
-        indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
-                   .reshape(x.shape))
-        return indices
-
-    def _get_tree_stats(self):
-        """
-        Return a python list containing the statistics about the node tree:
-            0: number of nodes (tree size)
-            1: number of unique nodes
-            2: number of trapezoids (tree leaf nodes)
-            3: number of unique trapezoids
-            4: maximum parent count (max number of times a node is repeated in
-                   tree)
-            5: maximum depth of tree (one more than the maximum number of
-                   comparisons needed to search through the tree)
-            6: mean of all trapezoid depths (one more than the average number
-                   of comparisons needed to search through the tree)
-        """
-        return self._cpp_trifinder.get_tree_stats()
 
-    def _initialize(self):
-        """
-        Initialize the underlying C++ object.  Can be called multiple times if,
-        for example, the triangulation is modified.
-        """
-        self._cpp_trifinder.initialize()
 
-    def _print_tree(self):
-        """
-        Print a text representation of the node tree, which is useful for
-        debugging purposes.
-        """
-        self._cpp_trifinder.print_tree()
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/triinterpolate.py b/lib/matplotlib/tri/triinterpolate.py
--- a/lib/matplotlib/tri/triinterpolate.py
+++ b/lib/matplotlib/tri/triinterpolate.py
@@ -1,1574 +1,9 @@
-"""
-Interpolation inside triangular grids.
-"""
-
-import numpy as np
-
+from ._triinterpolate import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri import Triangulation
-from matplotlib.tri.trifinder import TriFinder
-from matplotlib.tri.tritools import TriAnalyzer
-
-__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')
-
-
-class TriInterpolator:
-    """
-    Abstract base class for classes used to interpolate on a triangular grid.
-
-    Derived classes implement the following methods:
-
-    - ``__call__(x, y)``,
-      where x, y are array-like point coordinates of the same shape, and
-      that returns a masked array of the same shape containing the
-      interpolated z-values.
-
-    - ``gradient(x, y)``,
-      where x, y are array-like point coordinates of the same
-      shape, and that returns a list of 2 masked arrays of the same shape
-      containing the 2 derivatives of the interpolator (derivatives of
-      interpolated z values with respect to x and y).
-    """
-
-    def __init__(self, triangulation, z, trifinder=None):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-        self._z = np.asarray(z)
-        if self._z.shape != self._triangulation.x.shape:
-            raise ValueError("z array must have same length as triangulation x"
-                             " and y arrays")
-
-        _api.check_isinstance((TriFinder, None), trifinder=trifinder)
-        self._trifinder = trifinder or self._triangulation.get_trifinder()
-
-        # Default scaling factors : 1.0 (= no scaling)
-        # Scaling may be used for interpolations for which the order of
-        # magnitude of x, y has an impact on the interpolant definition.
-        # Please refer to :meth:`_interpolate_multikeys` for details.
-        self._unit_x = 1.0
-        self._unit_y = 1.0
-
-        # Default triangle renumbering: None (= no renumbering)
-        # Renumbering may be used to avoid unnecessary computations
-        # if complex calculations are done inside the Interpolator.
-        # Please refer to :meth:`_interpolate_multikeys` for details.
-        self._tri_renum = None
-
-    # __call__ and gradient docstrings are shared by all subclasses
-    # (except, if needed, relevant additions).
-    # However these methods are only implemented in subclasses to avoid
-    # confusion in the documentation.
-    _docstring__call__ = """
-        Returns a masked array containing interpolated values at the specified
-        (x, y) points.
-
-        Parameters
-        ----------
-        x, y : array-like
-            x and y coordinates of the same shape and any number of
-            dimensions.
-
-        Returns
-        -------
-        np.ma.array
-            Masked array of the same shape as *x* and *y*; values corresponding
-            to (*x*, *y*) points outside of the triangulation are masked out.
-
-        """
-
-    _docstringgradient = r"""
-        Returns a list of 2 masked arrays containing interpolated derivatives
-        at the specified (x, y) points.
-
-        Parameters
-        ----------
-        x, y : array-like
-            x and y coordinates of the same shape and any number of
-            dimensions.
-
-        Returns
-        -------
-        dzdx, dzdy : np.ma.array
-            2 masked arrays of the same shape as *x* and *y*; values
-            corresponding to (x, y) points outside of the triangulation
-            are masked out.
-            The first returned array contains the values of
-            :math:`\frac{\partial z}{\partial x}` and the second those of
-            :math:`\frac{\partial z}{\partial y}`.
-
-        """
-
-    def _interpolate_multikeys(self, x, y, tri_index=None,
-                               return_keys=('z',)):
-        """
-        Versatile (private) method defined for all TriInterpolators.
-
-        :meth:`_interpolate_multikeys` is a wrapper around method
-        :meth:`_interpolate_single_key` (to be defined in the child
-        subclasses).
-        :meth:`_interpolate_single_key actually performs the interpolation,
-        but only for 1-dimensional inputs and at valid locations (inside
-        unmasked triangles of the triangulation).
-
-        The purpose of :meth:`_interpolate_multikeys` is to implement the
-        following common tasks needed in all subclasses implementations:
-
-        - calculation of containing triangles
-        - dealing with more than one interpolation request at the same
-          location (e.g., if the 2 derivatives are requested, it is
-          unnecessary to compute the containing triangles twice)
-        - scaling according to self._unit_x, self._unit_y
-        - dealing with points outside of the grid (with fill value np.nan)
-        - dealing with multi-dimensional *x*, *y* arrays: flattening for
-          :meth:`_interpolate_params` call and final reshaping.
-
-        (Note that np.vectorize could do most of those things very well for
-        you, but it does it by function evaluations over successive tuples of
-        the input arrays. Therefore, this tends to be more time consuming than
-        using optimized numpy functions - e.g., np.dot - which can be used
-        easily on the flattened inputs, in the child-subclass methods
-        :meth:`_interpolate_single_key`.)
-
-        It is guaranteed that the calls to :meth:`_interpolate_single_key`
-        will be done with flattened (1-d) array-like input parameters *x*, *y*
-        and with flattened, valid `tri_index` arrays (no -1 index allowed).
-
-        Parameters
-        ----------
-        x, y : array-like
-            x and y coordinates where interpolated values are requested.
-        tri_index : array-like of int, optional
-            Array of the containing triangle indices, same shape as
-            *x* and *y*. Defaults to None. If None, these indices
-            will be computed by a TriFinder instance.
-            (Note: For point outside the grid, tri_index[ipt] shall be -1).
-        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
-            Defines the interpolation arrays to return, and in which order.
-
-        Returns
-        -------
-        list of arrays
-            Each array-like contains the expected interpolated values in the
-            order defined by *return_keys* parameter.
-        """
-        # Flattening and rescaling inputs arrays x, y
-        # (initial shape is stored for output)
-        x = np.asarray(x, dtype=np.float64)
-        y = np.asarray(y, dtype=np.float64)
-        sh_ret = x.shape
-        if x.shape != y.shape:
-            raise ValueError("x and y shall have same shapes."
-                             " Given: {0} and {1}".format(x.shape, y.shape))
-        x = np.ravel(x)
-        y = np.ravel(y)
-        x_scaled = x/self._unit_x
-        y_scaled = y/self._unit_y
-        size_ret = np.size(x_scaled)
-
-        # Computes & ravels the element indexes, extract the valid ones.
-        if tri_index is None:
-            tri_index = self._trifinder(x, y)
-        else:
-            if tri_index.shape != sh_ret:
-                raise ValueError(
-                    "tri_index array is provided and shall"
-                    " have same shape as x and y. Given: "
-                    "{0} and {1}".format(tri_index.shape, sh_ret))
-            tri_index = np.ravel(tri_index)
-
-        mask_in = (tri_index != -1)
-        if self._tri_renum is None:
-            valid_tri_index = tri_index[mask_in]
-        else:
-            valid_tri_index = self._tri_renum[tri_index[mask_in]]
-        valid_x = x_scaled[mask_in]
-        valid_y = y_scaled[mask_in]
-
-        ret = []
-        for return_key in return_keys:
-            # Find the return index associated with the key.
-            try:
-                return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
-            except KeyError as err:
-                raise ValueError("return_keys items shall take values in"
-                                 " {'z', 'dzdx', 'dzdy'}") from err
-
-            # Sets the scale factor for f & df components
-            scale = [1., 1./self._unit_x, 1./self._unit_y][return_index]
-
-            # Computes the interpolation
-            ret_loc = np.empty(size_ret, dtype=np.float64)
-            ret_loc[~mask_in] = np.nan
-            ret_loc[mask_in] = self._interpolate_single_key(
-                return_key, valid_tri_index, valid_x, valid_y) * scale
-            ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]
-
-        return ret
-
-    def _interpolate_single_key(self, return_key, tri_index, x, y):
-        """
-        Interpolate at points belonging to the triangulation
-        (inside an unmasked triangles).
-
-        Parameters
-        ----------
-        return_key : {'z', 'dzdx', 'dzdy'}
-            The requested values (z or its derivatives).
-        tri_index : 1D int array
-            Valid triangle index (cannot be -1).
-        x, y : 1D arrays, same shape as `tri_index`
-            Valid locations where interpolation is requested.
-
-        Returns
-        -------
-        1-d array
-            Returned array of the same size as *tri_index*
-        """
-        raise NotImplementedError("TriInterpolator subclasses" +
-                                  "should implement _interpolate_single_key!")
-
-
-class LinearTriInterpolator(TriInterpolator):
-    """
-    Linear interpolator on a triangular grid.
-
-    Each triangle is represented by a plane so that an interpolated value at
-    point (x, y) lies on the plane of the triangle containing (x, y).
-    Interpolated values are therefore continuous across the triangulation, but
-    their first derivatives are discontinuous at edges between triangles.
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The triangulation to interpolate over.
-    z : (npoints,) array-like
-        Array of values, defined at grid points, to interpolate between.
-    trifinder : `~matplotlib.tri.TriFinder`, optional
-        If this is not specified, the Triangulation's default TriFinder will
-        be used by calling `.Triangulation.get_trifinder`.
-
-    Methods
-    -------
-    `__call__` (x, y) : Returns interpolated values at (x, y) points.
-    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
-
-    """
-    def __init__(self, triangulation, z, trifinder=None):
-        super().__init__(triangulation, z, trifinder)
-
-        # Store plane coefficients for fast interpolation calculations.
-        self._plane_coefficients = \
-            self._triangulation.calculate_plane_coefficients(self._z)
-
-    def __call__(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('z',))[0]
-    __call__.__doc__ = TriInterpolator._docstring__call__
-
-    def gradient(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('dzdx', 'dzdy'))
-    gradient.__doc__ = TriInterpolator._docstringgradient
-
-    def _interpolate_single_key(self, return_key, tri_index, x, y):
-        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
-        if return_key == 'z':
-            return (self._plane_coefficients[tri_index, 0]*x +
-                    self._plane_coefficients[tri_index, 1]*y +
-                    self._plane_coefficients[tri_index, 2])
-        elif return_key == 'dzdx':
-            return self._plane_coefficients[tri_index, 0]
-        else:  # 'dzdy'
-            return self._plane_coefficients[tri_index, 1]
-
-
-class CubicTriInterpolator(TriInterpolator):
-    r"""
-    Cubic interpolator on a triangular grid.
-
-    In one-dimension - on a segment - a cubic interpolating function is
-    defined by the values of the function and its derivative at both ends.
-    This is almost the same in 2D inside a triangle, except that the values
-    of the function and its 2 derivatives have to be defined at each triangle
-    node.
-
-    The CubicTriInterpolator takes the value of the function at each node -
-    provided by the user - and internally computes the value of the
-    derivatives, resulting in a smooth interpolation.
-    (As a special feature, the user can also impose the value of the
-    derivatives at each node, but this is not supposed to be the common
-    usage.)
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The triangulation to interpolate over.
-    z : (npoints,) array-like
-        Array of values, defined at grid points, to interpolate between.
-    kind : {'min_E', 'geom', 'user'}, optional
-        Choice of the smoothing algorithm, in order to compute
-        the interpolant derivatives (defaults to 'min_E'):
-
-        - if 'min_E': (default) The derivatives at each node is computed
-          to minimize a bending energy.
-        - if 'geom': The derivatives at each node is computed as a
-          weighted average of relevant triangle normals. To be used for
-          speed optimization (large grids).
-        - if 'user': The user provides the argument *dz*, no computation
-          is hence needed.
-
-    trifinder : `~matplotlib.tri.TriFinder`, optional
-        If not specified, the Triangulation's default TriFinder will
-        be used by calling `.Triangulation.get_trifinder`.
-    dz : tuple of array-likes (dzdx, dzdy), optional
-        Used only if  *kind* ='user'. In this case *dz* must be provided as
-        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
-        are the interpolant first derivatives at the *triangulation* points.
-
-    Methods
-    -------
-    `__call__` (x, y) : Returns interpolated values at (x, y) points.
-    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
-
-    Notes
-    -----
-    This note is a bit technical and details how the cubic interpolation is
-    computed.
-
-    The interpolation is based on a Clough-Tocher subdivision scheme of
-    the *triangulation* mesh (to make it clearer, each triangle of the
-    grid will be divided in 3 child-triangles, and on each child triangle
-    the interpolated function is a cubic polynomial of the 2 coordinates).
-    This technique originates from FEM (Finite Element Method) analysis;
-    the element used is a reduced Hsieh-Clough-Tocher (HCT)
-    element. Its shape functions are described in [1]_.
-    The assembled function is guaranteed to be C1-smooth, i.e. it is
-    continuous and its first derivatives are also continuous (this
-    is easy to show inside the triangles but is also true when crossing the
-    edges).
-
-    In the default case (*kind* ='min_E'), the interpolant minimizes a
-    curvature energy on the functional space generated by the HCT element
-    shape functions - with imposed values but arbitrary derivatives at each
-    node. The minimized functional is the integral of the so-called total
-    curvature (implementation based on an algorithm from [2]_ - PCG sparse
-    solver):
-
-        .. math::
-
-            E(z) = \frac{1}{2} \int_{\Omega} \left(
-                \left( \frac{\partial^2{z}}{\partial{x}^2} \right)^2 +
-                \left( \frac{\partial^2{z}}{\partial{y}^2} \right)^2 +
-                2\left( \frac{\partial^2{z}}{\partial{y}\partial{x}} \right)^2
-            \right) dx\,dy
-
-    If the case *kind* ='geom' is chosen by the user, a simple geometric
-    approximation is used (weighted average of the triangle normal
-    vectors), which could improve speed on very large grids.
-
-    References
-    ----------
-    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
-        Hsieh-Clough-Tocher triangles, complete or reduced.",
-        International Journal for Numerical Methods in Engineering,
-        17(5):784 - 789. 2.01.
-    .. [2] C.T. Kelley, "Iterative Methods for Optimization".
-
-    """
-    def __init__(self, triangulation, z, kind='min_E', trifinder=None,
-                 dz=None):
-        super().__init__(triangulation, z, trifinder)
-
-        # Loads the underlying c++ _triangulation.
-        # (During loading, reordering of triangulation._triangles may occur so
-        # that all final triangles are now anti-clockwise)
-        self._triangulation.get_cpp_triangulation()
-
-        # To build the stiffness matrix and avoid zero-energy spurious modes
-        # we will only store internally the valid (unmasked) triangles and
-        # the necessary (used) points coordinates.
-        # 2 renumbering tables need to be computed and stored:
-        #  - a triangle renum table in order to translate the result from a
-        #    TriFinder instance into the internal stored triangle number.
-        #  - a node renum table to overwrite the self._z values into the new
-        #    (used) node numbering.
-        tri_analyzer = TriAnalyzer(self._triangulation)
-        (compressed_triangles, compressed_x, compressed_y, tri_renum,
-         node_renum) = tri_analyzer._get_compressed_triangulation()
-        self._triangles = compressed_triangles
-        self._tri_renum = tri_renum
-        # Taking into account the node renumbering in self._z:
-        valid_node = (node_renum != -1)
-        self._z[node_renum[valid_node]] = self._z[valid_node]
-
-        # Computing scale factors
-        self._unit_x = np.ptp(compressed_x)
-        self._unit_y = np.ptp(compressed_y)
-        self._pts = np.column_stack([compressed_x / self._unit_x,
-                                     compressed_y / self._unit_y])
-        # Computing triangle points
-        self._tris_pts = self._pts[self._triangles]
-        # Computing eccentricities
-        self._eccs = self._compute_tri_eccentricities(self._tris_pts)
-        # Computing dof estimations for HCT triangle shape function
-        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
-        self._dof = self._compute_dof(kind, dz=dz)
-        # Loading HCT element
-        self._ReferenceElement = _ReducedHCT_Element()
-
-    def __call__(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('z',))[0]
-    __call__.__doc__ = TriInterpolator._docstring__call__
-
-    def gradient(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('dzdx', 'dzdy'))
-    gradient.__doc__ = TriInterpolator._docstringgradient
-
-    def _interpolate_single_key(self, return_key, tri_index, x, y):
-        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
-        tris_pts = self._tris_pts[tri_index]
-        alpha = self._get_alpha_vec(x, y, tris_pts)
-        ecc = self._eccs[tri_index]
-        dof = np.expand_dims(self._dof[tri_index], axis=1)
-        if return_key == 'z':
-            return self._ReferenceElement.get_function_values(
-                alpha, ecc, dof)
-        else:  # 'dzdx', 'dzdy'
-            J = self._get_jacobian(tris_pts)
-            dzdx = self._ReferenceElement.get_function_derivatives(
-                alpha, J, ecc, dof)
-            if return_key == 'dzdx':
-                return dzdx[:, 0, 0]
-            else:
-                return dzdx[:, 1, 0]
-
-    def _compute_dof(self, kind, dz=None):
-        """
-        Compute and return nodal dofs according to kind.
-
-        Parameters
-        ----------
-        kind : {'min_E', 'geom', 'user'}
-            Choice of the _DOF_estimator subclass to estimate the gradient.
-        dz : tuple of array-likes (dzdx, dzdy), optional
-            Used only if *kind*=user; in this case passed to the
-            :class:`_DOF_estimator_user`.
-
-        Returns
-        -------
-        array-like, shape (npts, 2)
-            Estimation of the gradient at triangulation nodes (stored as
-            degree of freedoms of reduced-HCT triangle elements).
-        """
-        if kind == 'user':
-            if dz is None:
-                raise ValueError("For a CubicTriInterpolator with "
-                                 "*kind*='user', a valid *dz* "
-                                 "argument is expected.")
-            TE = _DOF_estimator_user(self, dz=dz)
-        elif kind == 'geom':
-            TE = _DOF_estimator_geom(self)
-        else:  # 'min_E', checked in __init__
-            TE = _DOF_estimator_min_E(self)
-        return TE.compute_dof_from_df()
-
-    @staticmethod
-    def _get_alpha_vec(x, y, tris_pts):
-        """
-        Fast (vectorized) function to compute barycentric coordinates alpha.
-
-        Parameters
-        ----------
-        x, y : array-like of dim 1 (shape (nx,))
-            Coordinates of the points whose points barycentric coordinates are
-            requested.
-        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
-            Coordinates of the containing triangles apexes.
-
-        Returns
-        -------
-        array of dim 2 (shape (nx, 3))
-            Barycentric coordinates of the points inside the containing
-            triangles.
-        """
-        ndim = tris_pts.ndim-2
-
-        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
-        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
-        abT = np.stack([a, b], axis=-1)
-        ab = _transpose_vectorized(abT)
-        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]
-
-        metric = ab @ abT
-        # Here we try to deal with the colinear cases.
-        # metric_inv is in this case set to the Moore-Penrose pseudo-inverse
-        # meaning that we will still return a set of valid barycentric
-        # coordinates.
-        metric_inv = _pseudo_inv22sym_vectorized(metric)
-        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
-        ksi = metric_inv @ Covar
-        alpha = _to_matrix_vectorized([
-            [1-ksi[:, 0, 0]-ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
-        return alpha
-
-    @staticmethod
-    def _get_jacobian(tris_pts):
-        """
-        Fast (vectorized) function to compute triangle jacobian matrix.
-
-        Parameters
-        ----------
-        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
-            Coordinates of the containing triangles apexes.
-
-        Returns
-        -------
-        array of dim 3 (shape (nx, 2, 2))
-            Barycentric coordinates of the points inside the containing
-            triangles.
-            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
-            itri, so that the following (matrix) relationship holds:
-               [dz/dksi] = [J] x [dz/dx]
-            with x: global coordinates
-                 ksi: element parametric coordinates in triangle first apex
-                 local basis.
-        """
-        a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
-        b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
-        J = _to_matrix_vectorized([[a[:, 0], a[:, 1]],
-                                   [b[:, 0], b[:, 1]]])
-        return J
-
-    @staticmethod
-    def _compute_tri_eccentricities(tris_pts):
-        """
-        Compute triangle eccentricities.
-
-        Parameters
-        ----------
-        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
-            Coordinates of the triangles apexes.
-
-        Returns
-        -------
-        array like of dim 2 (shape: (nx, 3))
-            The so-called eccentricity parameters [1] needed for HCT triangular
-            element.
-        """
-        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
-        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
-        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
-        # Do not use np.squeeze, this is dangerous if only one triangle
-        # in the triangulation...
-        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
-        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
-        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
-        # Note that this line will raise a warning for dot_a, dot_b or dot_c
-        # zeros, but we choose not to support triangles with duplicate points.
-        return _to_matrix_vectorized([[(dot_c-dot_b) / dot_a],
-                                      [(dot_a-dot_c) / dot_b],
-                                      [(dot_b-dot_a) / dot_c]])
-
-
-# FEM element used for interpolation and for solving minimisation
-# problem (Reduced HCT element)
-class _ReducedHCT_Element:
-    """
-    Implementation of reduced HCT triangular element with explicit shape
-    functions.
-
-    Computes z, dz, d2z and the element stiffness matrix for bending energy:
-    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)
-
-    *** Reference for the shape functions: ***
-    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
-        reduced.
-        Michel Bernadou, Kamal Hassan
-        International Journal for Numerical Methods in Engineering.
-        17(5):784 - 789.  2.01
-
-    *** Element description: ***
-    9 dofs: z and dz given at 3 apex
-    C1 (conform)
-
-    """
-    # 1) Loads matrices to generate shape functions as a function of
-    #    triangle eccentricities - based on [1] p.11 '''
-    M = np.array([
-        [ 0.00, 0.00, 0.00,  4.50,  4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00,  0.50,  1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00,  1.25,  0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.50, 1.00, 0.00, -1.50,  0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
-        [ 0.00, 0.00, 0.00, -0.25,  0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
-        [ 0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
-        [ 0.50, 0.00, 1.00,  0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
-        [ 0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
-        [ 0.00, 0.00, 0.00,  0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
-    M0 = np.array([
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [-1.00, 0.00, 0.00,  1.50,  1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
-        [-0.50, 0.00, 0.00,  0.75,  0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00,  3.00],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00,  1.50]])
-    M1 = np.array([
-        [-0.50, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
-    M2 = np.array([
-        [ 0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.50, 0.00, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
-
-    # 2) Loads matrices to rotate components of gradient & Hessian
-    #    vectors in the reference basis of triangle first apex (a0)
-    rotate_dV = np.array([[ 1.,  0.], [ 0.,  1.],
-                          [ 0.,  1.], [-1., -1.],
-                          [-1., -1.], [ 1.,  0.]])
-
-    rotate_d2V = np.array([[1., 0., 0.], [0., 1., 0.], [ 0.,  0.,  1.],
-                           [0., 1., 0.], [1., 1., 1.], [ 0., -2., -1.],
-                           [1., 1., 1.], [1., 0., 0.], [-2.,  0., -1.]])
-
-    # 3) Loads Gauss points & weights on the 3 sub-_triangles for P2
-    #    exact integral - 3 points on each subtriangles.
-    # NOTE: as the 2nd derivative is discontinuous , we really need those 9
-    # points!
-    n_gauss = 9
-    gauss_pts = np.array([[13./18.,  4./18.,  1./18.],
-                          [ 4./18., 13./18.,  1./18.],
-                          [ 7./18.,  7./18.,  4./18.],
-                          [ 1./18., 13./18.,  4./18.],
-                          [ 1./18.,  4./18., 13./18.],
-                          [ 4./18.,  7./18.,  7./18.],
-                          [ 4./18.,  1./18., 13./18.],
-                          [13./18.,  1./18.,  4./18.],
-                          [ 7./18.,  4./18.,  7./18.]], dtype=np.float64)
-    gauss_w = np.ones([9], dtype=np.float64) / 9.
-
-    #  4) Stiffness matrix for curvature energy
-    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
-
-    #  5) Loads the matrix to compute DOF_rot from tri_J at apex 0
-    J0_to_J1 = np.array([[-1.,  1.], [-1.,  0.]])
-    J0_to_J2 = np.array([[ 0., -1.], [ 1., -1.]])
-
-    def get_function_values(self, alpha, ecc, dofs):
-        """
-        Parameters
-        ----------
-        alpha : is a (N x 3 x 1) array (array of column-matrices) of
-        barycentric coordinates,
-        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities,
-        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
-        degrees of freedom.
-
-        Returns
-        -------
-        Returns the N-array of interpolated function values.
-        """
-        subtri = np.argmin(alpha, axis=1)[:, 0]
-        ksi = _roll_vectorized(alpha, -subtri, axis=0)
-        E = _roll_vectorized(ecc, -subtri, axis=0)
-        x = ksi[:, 0, 0]
-        y = ksi[:, 1, 0]
-        z = ksi[:, 2, 0]
-        x_sq = x*x
-        y_sq = y*y
-        z_sq = z*z
-        V = _to_matrix_vectorized([
-            [x_sq*x], [y_sq*y], [z_sq*z], [x_sq*z], [x_sq*y], [y_sq*x],
-            [y_sq*z], [z_sq*y], [z_sq*x], [x*y*z]])
-        prod = self.M @ V
-        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
-        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
-        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
-        s = _roll_vectorized(prod, 3*subtri, axis=0)
-        return (dofs @ s)[:, 0, 0]
-
-    def get_function_derivatives(self, alpha, J, ecc, dofs):
-        """
-        Parameters
-        ----------
-        *alpha* is a (N x 3 x 1) array (array of column-matrices of
-        barycentric coordinates)
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
-        eccentricities)
-        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
-        degrees of freedom.
-
-        Returns
-        -------
-        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
-        in global coordinates at locations alpha, as a column-matrices of
-        shape (N x 2 x 1).
-        """
-        subtri = np.argmin(alpha, axis=1)[:, 0]
-        ksi = _roll_vectorized(alpha, -subtri, axis=0)
-        E = _roll_vectorized(ecc, -subtri, axis=0)
-        x = ksi[:, 0, 0]
-        y = ksi[:, 1, 0]
-        z = ksi[:, 2, 0]
-        x_sq = x*x
-        y_sq = y*y
-        z_sq = z*z
-        dV = _to_matrix_vectorized([
-            [    -3.*x_sq,     -3.*x_sq],
-            [     3.*y_sq,           0.],
-            [          0.,      3.*z_sq],
-            [     -2.*x*z, -2.*x*z+x_sq],
-            [-2.*x*y+x_sq,      -2.*x*y],
-            [ 2.*x*y-y_sq,        -y_sq],
-            [      2.*y*z,         y_sq],
-            [        z_sq,       2.*y*z],
-            [       -z_sq,  2.*x*z-z_sq],
-            [     x*z-y*z,      x*y-y*z]])
-        # Puts back dV in first apex basis
-        dV = dV @ _extract_submatrices(
-            self.rotate_dV, subtri, block_size=2, axis=0)
-
-        prod = self.M @ dV
-        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
-        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
-        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
-        dsdksi = _roll_vectorized(prod, 3*subtri, axis=0)
-        dfdksi = dofs @ dsdksi
-        # In global coordinates:
-        # Here we try to deal with the simplest colinear cases, returning a
-        # null matrix.
-        J_inv = _safe_inv22_vectorized(J)
-        dfdx = J_inv @ _transpose_vectorized(dfdksi)
-        return dfdx
-
-    def get_function_hessians(self, alpha, J, ecc, dofs):
-        """
-        Parameters
-        ----------
-        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
-        barycentric coordinates
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
-        degrees of freedom.
-
-        Returns
-        -------
-        Returns the values of interpolated function 2nd-derivatives
-        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
-        as a column-matrices of shape (N x 3 x 1).
-        """
-        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
-        d2fdksi2 = dofs @ d2sdksi2
-        H_rot = self.get_Hrot_from_J(J)
-        d2fdx2 = d2fdksi2 @ H_rot
-        return _transpose_vectorized(d2fdx2)
-
-    def get_d2Sidksij2(self, alpha, ecc):
-        """
-        Parameters
-        ----------
-        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
-        barycentric coordinates
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-
-        Returns
-        -------
-        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
-        expressed in covariant coordinates in first apex basis.
-        """
-        subtri = np.argmin(alpha, axis=1)[:, 0]
-        ksi = _roll_vectorized(alpha, -subtri, axis=0)
-        E = _roll_vectorized(ecc, -subtri, axis=0)
-        x = ksi[:, 0, 0]
-        y = ksi[:, 1, 0]
-        z = ksi[:, 2, 0]
-        d2V = _to_matrix_vectorized([
-            [     6.*x,      6.*x,      6.*x],
-            [     6.*y,        0.,        0.],
-            [       0.,      6.*z,        0.],
-            [     2.*z, 2.*z-4.*x, 2.*z-2.*x],
-            [2.*y-4.*x,      2.*y, 2.*y-2.*x],
-            [2.*x-4.*y,        0.,     -2.*y],
-            [     2.*z,        0.,      2.*y],
-            [       0.,      2.*y,      2.*z],
-            [       0., 2.*x-4.*z,     -2.*z],
-            [    -2.*z,     -2.*y,     x-y-z]])
-        # Puts back d2V in first apex basis
-        d2V = d2V @ _extract_submatrices(
-            self.rotate_d2V, subtri, block_size=3, axis=0)
-        prod = self.M @ d2V
-        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
-        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
-        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
-        d2sdksi2 = _roll_vectorized(prod, 3*subtri, axis=0)
-        return d2sdksi2
-
-    def get_bending_matrices(self, J, ecc):
-        """
-        Parameters
-        ----------
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-
-        Returns
-        -------
-        Returns the element K matrices for bending energy expressed in
-        GLOBAL nodal coordinates.
-        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
-        tri_J is needed to rotate dofs from local basis to global basis
-        """
-        n = np.size(ecc, 0)
-
-        # 1) matrix to rotate dofs in global coordinates
-        J1 = self.J0_to_J1 @ J
-        J2 = self.J0_to_J2 @ J
-        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
-        DOF_rot[:, 0, 0] = 1
-        DOF_rot[:, 3, 3] = 1
-        DOF_rot[:, 6, 6] = 1
-        DOF_rot[:, 1:3, 1:3] = J
-        DOF_rot[:, 4:6, 4:6] = J1
-        DOF_rot[:, 7:9, 7:9] = J2
-
-        # 2) matrix to rotate Hessian in global coordinates.
-        H_rot, area = self.get_Hrot_from_J(J, return_area=True)
-
-        # 3) Computes stiffness matrix
-        # Gauss quadrature.
-        K = np.zeros([n, 9, 9], dtype=np.float64)
-        weights = self.gauss_w
-        pts = self.gauss_pts
-        for igauss in range(self.n_gauss):
-            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
-            alpha = np.expand_dims(alpha, 2)
-            weight = weights[igauss]
-            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
-            d2Skdx2 = d2Skdksi2 @ H_rot
-            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))
-
-        # 4) With nodal (not elem) dofs
-        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
-
-        # 5) Need the area to compute total element energy
-        return _scalar_vectorized(area, K)
-
-    def get_Hrot_from_J(self, J, return_area=False):
-        """
-        Parameters
-        ----------
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-
-        Returns
-        -------
-        Returns H_rot used to rotate Hessian from local basis of first apex,
-        to global coordinates.
-        if *return_area* is True, returns also the triangle area (0.5*det(J))
-        """
-        # Here we try to deal with the simplest colinear cases; a null
-        # energy and area is imposed.
-        J_inv = _safe_inv22_vectorized(J)
-        Ji00 = J_inv[:, 0, 0]
-        Ji11 = J_inv[:, 1, 1]
-        Ji10 = J_inv[:, 1, 0]
-        Ji01 = J_inv[:, 0, 1]
-        H_rot = _to_matrix_vectorized([
-            [Ji00*Ji00, Ji10*Ji10, Ji00*Ji10],
-            [Ji01*Ji01, Ji11*Ji11, Ji01*Ji11],
-            [2*Ji00*Ji01, 2*Ji11*Ji10, Ji00*Ji11+Ji10*Ji01]])
-        if not return_area:
-            return H_rot
-        else:
-            area = 0.5 * (J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0])
-            return H_rot, area
-
-    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
-        """
-        Build K and F for the following elliptic formulation:
-        minimization of curvature energy with value of function at node
-        imposed and derivatives 'free'.
-
-        Build the global Kff matrix in cco format.
-        Build the full Ff vec Ff = - Kfc x Uc.
-
-        Parameters
-        ----------
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-        *triangles* is a (N x 3) array of nodes indexes.
-        *Uc* is (N x 3) array of imposed displacements at nodes
-
-        Returns
-        -------
-        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
-        (row, col) entries must be summed.
-        Ff: force vector - dim npts * 3
-        """
-        ntri = np.size(ecc, 0)
-        vec_range = np.arange(ntri, dtype=np.int32)
-        c_indices = np.full(ntri, -1, dtype=np.int32)  # for unused dofs, -1
-        f_dof = [1, 2, 4, 5, 7, 8]
-        c_dof = [0, 3, 6]
-
-        # vals, rows and cols indices in global dof numbering
-        f_dof_indices = _to_matrix_vectorized([[
-            c_indices, triangles[:, 0]*2, triangles[:, 0]*2+1,
-            c_indices, triangles[:, 1]*2, triangles[:, 1]*2+1,
-            c_indices, triangles[:, 2]*2, triangles[:, 2]*2+1]])
-
-        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
-        f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
-        f_col_indices = expand_indices @ f_dof_indices
-        K_elem = self.get_bending_matrices(J, ecc)
-
-        # Extracting sub-matrices
-        # Explanation & notations:
-        # * Subscript f denotes 'free' degrees of freedom (i.e. dz/dx, dz/dx)
-        # * Subscript c denotes 'condensated' (imposed) degrees of freedom
-        #    (i.e. z at all nodes)
-        # * F = [Ff, Fc] is the force vector
-        # * U = [Uf, Uc] is the imposed dof vector
-        #        [ Kff Kfc ]
-        # * K =  [         ]  is the laplacian stiffness matrix
-        #        [ Kcf Kff ]
-        # * As F = K x U one gets straightforwardly: Ff = - Kfc x Uc
-
-        # Computing Kff stiffness matrix in sparse coo format
-        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
-        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
-        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])
-
-        # Computing Ff force vector in sparse coo format
-        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
-        Uc_elem = np.expand_dims(Uc, axis=2)
-        Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
-        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]
-
-        # Extracting Ff force vector in dense format
-        # We have to sum duplicate indices -  using bincount
-        Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
-        return Kff_rows, Kff_cols, Kff_vals, Ff
-
-
-# :class:_DOF_estimator, _DOF_estimator_user, _DOF_estimator_geom,
-# _DOF_estimator_min_E
-# Private classes used to compute the degree of freedom of each triangular
-# element for the TriCubicInterpolator.
-class _DOF_estimator:
-    """
-    Abstract base class for classes used to estimate a function's first
-    derivatives, and deduce the dofs for a CubicTriInterpolator using a
-    reduced HCT element formulation.
-
-    Derived classes implement ``compute_df(self, **kwargs)``, returning
-    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
-    gradient coordinates.
-    """
-    def __init__(self, interpolator, **kwargs):
-        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
-        self._pts = interpolator._pts
-        self._tris_pts = interpolator._tris_pts
-        self.z = interpolator._z
-        self._triangles = interpolator._triangles
-        (self._unit_x, self._unit_y) = (interpolator._unit_x,
-                                        interpolator._unit_y)
-        self.dz = self.compute_dz(**kwargs)
-        self.compute_dof_from_df()
-
-    def compute_dz(self, **kwargs):
-        raise NotImplementedError
-
-    def compute_dof_from_df(self):
-        """
-        Compute reduced-HCT elements degrees of freedom, from the gradient.
-        """
-        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
-        tri_z = self.z[self._triangles]
-        tri_dz = self.dz[self._triangles]
-        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
-        return tri_dof
-
-    @staticmethod
-    def get_dof_vec(tri_z, tri_dz, J):
-        """
-        Compute the dof vector of a triangle, from the value of f, df and
-        of the local Jacobian at each node.
-
-        Parameters
-        ----------
-        tri_z : shape (3,) array
-            f nodal values.
-        tri_dz : shape (3, 2) array
-            df/dx, df/dy nodal values.
-        J
-            Jacobian matrix in local basis of apex 0.
-
-        Returns
-        -------
-        dof : shape (9,) array
-            For each apex ``iapex``::
-
-                dof[iapex*3+0] = f(Ai)
-                dof[iapex*3+1] = df(Ai).(AiAi+)
-                dof[iapex*3+2] = df(Ai).(AiAi-)
-        """
-        npt = tri_z.shape[0]
-        dof = np.zeros([npt, 9], dtype=np.float64)
-        J1 = _ReducedHCT_Element.J0_to_J1 @ J
-        J2 = _ReducedHCT_Element.J0_to_J2 @ J
-
-        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
-        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
-        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
-
-        dfdksi = _to_matrix_vectorized([
-            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
-            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
-        dof[:, 0:7:3] = tri_z
-        dof[:, 1:8:3] = dfdksi[:, 0]
-        dof[:, 2:9:3] = dfdksi[:, 1]
-        return dof
-
-
-class _DOF_estimator_user(_DOF_estimator):
-    """dz is imposed by user; accounts for scaling if any."""
-
-    def compute_dz(self, dz):
-        (dzdx, dzdy) = dz
-        dzdx = dzdx * self._unit_x
-        dzdy = dzdy * self._unit_y
-        return np.vstack([dzdx, dzdy]).T
-
-
-class _DOF_estimator_geom(_DOF_estimator):
-    """Fast 'geometric' approximation, recommended for large arrays."""
-
-    def compute_dz(self):
-        """
-        self.df is computed as weighted average of _triangles sharing a common
-        node. On each triangle itri f is first assumed linear (= ~f), which
-        allows to compute d~f[itri]
-        Then the following approximation of df nodal values is then proposed:
-            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
-        The weighted coeff. w[itri] are proportional to the angle of the
-        triangle itri at apex ipt
-        """
-        el_geom_w = self.compute_geom_weights()
-        el_geom_grad = self.compute_geom_grads()
-
-        # Sum of weights coeffs
-        w_node_sum = np.bincount(np.ravel(self._triangles),
-                                 weights=np.ravel(el_geom_w))
-
-        # Sum of weighted df = (dfx, dfy)
-        dfx_el_w = np.empty_like(el_geom_w)
-        dfy_el_w = np.empty_like(el_geom_w)
-        for iapex in range(3):
-            dfx_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 0]
-            dfy_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 1]
-        dfx_node_sum = np.bincount(np.ravel(self._triangles),
-                                   weights=np.ravel(dfx_el_w))
-        dfy_node_sum = np.bincount(np.ravel(self._triangles),
-                                   weights=np.ravel(dfy_el_w))
-
-        # Estimation of df
-        dfx_estim = dfx_node_sum/w_node_sum
-        dfy_estim = dfy_node_sum/w_node_sum
-        return np.vstack([dfx_estim, dfy_estim]).T
-
-    def compute_geom_weights(self):
-        """
-        Build the (nelems, 3) weights coeffs of _triangles angles,
-        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
-        """
-        weights = np.zeros([np.size(self._triangles, 0), 3])
-        tris_pts = self._tris_pts
-        for ipt in range(3):
-            p0 = tris_pts[:, ipt % 3, :]
-            p1 = tris_pts[:, (ipt+1) % 3, :]
-            p2 = tris_pts[:, (ipt-1) % 3, :]
-            alpha1 = np.arctan2(p1[:, 1]-p0[:, 1], p1[:, 0]-p0[:, 0])
-            alpha2 = np.arctan2(p2[:, 1]-p0[:, 1], p2[:, 0]-p0[:, 0])
-            # In the below formula we could take modulo 2. but
-            # modulo 1. is safer regarding round-off errors (flat triangles).
-            angle = np.abs(((alpha2-alpha1) / np.pi) % 1)
-            # Weight proportional to angle up np.pi/2; null weight for
-            # degenerated cases 0 and np.pi (note that *angle* is normalized
-            # by np.pi).
-            weights[:, ipt] = 0.5 - np.abs(angle-0.5)
-        return weights
-
-    def compute_geom_grads(self):
-        """
-        Compute the (global) gradient component of f assumed linear (~f).
-        returns array df of shape (nelems, 2)
-        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
-        """
-        tris_pts = self._tris_pts
-        tris_f = self.z[self._triangles]
-
-        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
-        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
-        dM = np.dstack([dM1, dM2])
-        # Here we try to deal with the simplest colinear cases: a null
-        # gradient is assumed in this case.
-        dM_inv = _safe_inv22_vectorized(dM)
-
-        dZ1 = tris_f[:, 1] - tris_f[:, 0]
-        dZ2 = tris_f[:, 2] - tris_f[:, 0]
-        dZ = np.vstack([dZ1, dZ2]).T
-        df = np.empty_like(dZ)
-
-        # With np.einsum: could be ej,eji -> ej
-        df[:, 0] = dZ[:, 0]*dM_inv[:, 0, 0] + dZ[:, 1]*dM_inv[:, 1, 0]
-        df[:, 1] = dZ[:, 0]*dM_inv[:, 0, 1] + dZ[:, 1]*dM_inv[:, 1, 1]
-        return df
-
-
-class _DOF_estimator_min_E(_DOF_estimator_geom):
-    """
-    The 'smoothest' approximation, df is computed through global minimization
-    of the bending energy:
-      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
-    """
-    def __init__(self, Interpolator):
-        self._eccs = Interpolator._eccs
-        super().__init__(Interpolator)
-
-    def compute_dz(self):
-        """
-        Elliptic solver for bending energy minimization.
-        Uses a dedicated 'toy' sparse Jacobi PCG solver.
-        """
-        # Initial guess for iterative PCG solver.
-        dz_init = super().compute_dz()
-        Uf0 = np.ravel(dz_init)
-
-        reference_element = _ReducedHCT_Element()
-        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
-        eccs = self._eccs
-        triangles = self._triangles
-        Uc = self.z[self._triangles]
-
-        # Building stiffness matrix and force vector in coo format
-        Kff_rows, Kff_cols, Kff_vals, Ff = reference_element.get_Kff_and_Ff(
-            J, eccs, triangles, Uc)
-
-        # Building sparse matrix and solving minimization problem
-        # We could use scipy.sparse direct solver; however to avoid this
-        # external dependency an implementation of a simple PCG solver with
-        # a simple diagonal Jacobi preconditioner is implemented.
-        tol = 1.e-10
-        n_dof = Ff.shape[0]
-        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
-                                     shape=(n_dof, n_dof))
-        Kff_coo.compress_csc()
-        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
-        # If the PCG did not converge, we return the best guess between Uf0
-        # and Uf.
-        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
-        if err0 < err:
-            # Maybe a good occasion to raise a warning here ?
-            _api.warn_external("In TriCubicInterpolator initialization, "
-                               "PCG sparse solver did not converge after "
-                               "1000 iterations. `geom` approximation is "
-                               "used instead of `min_E`")
-            Uf = Uf0
-
-        # Building dz from Uf
-        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
-        dz[:, 0] = Uf[::2]
-        dz[:, 1] = Uf[1::2]
-        return dz
-
-
-# The following private :class:_Sparse_Matrix_coo and :func:_cg provide
-# a PCG sparse solver for (symmetric) elliptic problems.
-class _Sparse_Matrix_coo:
-    def __init__(self, vals, rows, cols, shape):
-        """
-        Create a sparse matrix in coo format.
-        *vals*: arrays of values of non-null entries of the matrix
-        *rows*: int arrays of rows of non-null entries of the matrix
-        *cols*: int arrays of cols of non-null entries of the matrix
-        *shape*: 2-tuple (n, m) of matrix shape
-        """
-        self.n, self.m = shape
-        self.vals = np.asarray(vals, dtype=np.float64)
-        self.rows = np.asarray(rows, dtype=np.int32)
-        self.cols = np.asarray(cols, dtype=np.int32)
-
-    def dot(self, V):
-        """
-        Dot product of self by a vector *V* in sparse-dense to dense format
-        *V* dense vector of shape (self.m,).
-        """
-        assert V.shape == (self.m,)
-        return np.bincount(self.rows,
-                           weights=self.vals*V[self.cols],
-                           minlength=self.m)
-
-    def compress_csc(self):
-        """
-        Compress rows, cols, vals / summing duplicates. Sort for csc format.
-        """
-        _, unique, indices = np.unique(
-            self.rows + self.n*self.cols,
-            return_index=True, return_inverse=True)
-        self.rows = self.rows[unique]
-        self.cols = self.cols[unique]
-        self.vals = np.bincount(indices, weights=self.vals)
-
-    def compress_csr(self):
-        """
-        Compress rows, cols, vals / summing duplicates. Sort for csr format.
-        """
-        _, unique, indices = np.unique(
-            self.m*self.rows + self.cols,
-            return_index=True, return_inverse=True)
-        self.rows = self.rows[unique]
-        self.cols = self.cols[unique]
-        self.vals = np.bincount(indices, weights=self.vals)
-
-    def to_dense(self):
-        """
-        Return a dense matrix representing self, mainly for debugging purposes.
-        """
-        ret = np.zeros([self.n, self.m], dtype=np.float64)
-        nvals = self.vals.size
-        for i in range(nvals):
-            ret[self.rows[i], self.cols[i]] += self.vals[i]
-        return ret
-
-    def __str__(self):
-        return self.to_dense().__str__()
-
-    @property
-    def diag(self):
-        """Return the (dense) vector of the diagonal elements."""
-        in_diag = (self.rows == self.cols)
-        diag = np.zeros(min(self.n, self.n), dtype=np.float64)  # default 0.
-        diag[self.rows[in_diag]] = self.vals[in_diag]
-        return diag
-
-
-def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
-    """
-    Use Preconditioned Conjugate Gradient iteration to solve A x = b
-    A simple Jacobi (diagonal) preconditioner is used.
-
-    Parameters
-    ----------
-    A : _Sparse_Matrix_coo
-        *A* must have been compressed before by compress_csc or
-        compress_csr method.
-    b : array
-        Right hand side of the linear system.
-    x0 : array, optional
-        Starting guess for the solution. Defaults to the zero vector.
-    tol : float, optional
-        Tolerance to achieve. The algorithm terminates when the relative
-        residual is below tol. Default is 1e-10.
-    maxiter : int, optional
-        Maximum number of iterations.  Iteration will stop after *maxiter*
-        steps even if the specified tolerance has not been achieved. Defaults
-        to 1000.
-
-    Returns
-    -------
-    x : array
-        The converged solution.
-    err : float
-        The absolute error np.linalg.norm(A.dot(x) - b)
-    """
-    n = b.size
-    assert A.n == n
-    assert A.m == n
-    b_norm = np.linalg.norm(b)
-
-    # Jacobi pre-conditioner
-    kvec = A.diag
-    # For diag elem < 1e-6 we keep 1e-6.
-    kvec = np.maximum(kvec, 1e-6)
-
-    # Initial guess
-    if x0 is None:
-        x = np.zeros(n)
-    else:
-        x = x0
-
-    r = b - A.dot(x)
-    w = r/kvec
-
-    p = np.zeros(n)
-    beta = 0.0
-    rho = np.dot(r, w)
-    k = 0
-
-    # Following C. T. Kelley
-    while (np.sqrt(abs(rho)) > tol*b_norm) and (k < maxiter):
-        p = w + beta*p
-        z = A.dot(p)
-        alpha = rho/np.dot(p, z)
-        r = r - alpha*z
-        w = r/kvec
-        rhoold = rho
-        rho = np.dot(r, w)
-        x = x + alpha*p
-        beta = rho/rhoold
-        # err = np.linalg.norm(A.dot(x) - b)  # absolute accuracy - not used
-        k += 1
-    err = np.linalg.norm(A.dot(x) - b)
-    return x, err
-
-
-# The following private functions:
-#     :func:`_safe_inv22_vectorized`
-#     :func:`_pseudo_inv22sym_vectorized`
-#     :func:`_scalar_vectorized`
-#     :func:`_transpose_vectorized`
-#     :func:`_roll_vectorized`
-#     :func:`_to_matrix_vectorized`
-#     :func:`_extract_submatrices`
-# provide fast numpy implementation of some standard operations on arrays of
-# matrices - stored as (:, n_rows, n_cols)-shaped np.arrays.
-
-# Development note: Dealing with pathologic 'flat' triangles in the
-# CubicTriInterpolator code and impact on (2, 2)-matrix inversion functions
-# :func:`_safe_inv22_vectorized` and :func:`_pseudo_inv22sym_vectorized`.
-#
-# Goals:
-# 1) The CubicTriInterpolator should be able to handle flat or almost flat
-#    triangles without raising an error,
-# 2) These degenerated triangles should have no impact on the automatic dof
-#    calculation (associated with null weight for the _DOF_estimator_geom and
-#    with null energy for the _DOF_estimator_min_E),
-# 3) Linear patch test should be passed exactly on degenerated meshes,
-# 4) Interpolation (with :meth:`_interpolate_single_key` or
-#    :meth:`_interpolate_multi_key`) shall be correctly handled even *inside*
-#    the pathologic triangles, to interact correctly with a TriRefiner class.
-#
-# Difficulties:
-# Flat triangles have rank-deficient *J* (so-called jacobian matrix) and
-# *metric* (the metric tensor = J x J.T). Computation of the local
-# tangent plane is also problematic.
-#
-# Implementation:
-# Most of the time, when computing the inverse of a rank-deficient matrix it
-# is safe to simply return the null matrix (which is the implementation in
-# :func:`_safe_inv22_vectorized`). This is because of point 2), itself
-# enforced by:
-#    - null area hence null energy in :class:`_DOF_estimator_min_E`
-#    - angles close or equal to 0 or np.pi hence null weight in
-#      :class:`_DOF_estimator_geom`.
-#      Note that the function angle -> weight is continuous and maximum for an
-#      angle np.pi/2 (refer to :meth:`compute_geom_weights`)
-# The exception is the computation of barycentric coordinates, which is done
-# by inversion of the *metric* matrix. In this case, we need to compute a set
-# of valid coordinates (1 among numerous possibilities), to ensure point 4).
-# We benefit here from the symmetry of metric = J x J.T, which makes it easier
-# to compute a pseudo-inverse in :func:`_pseudo_inv22sym_vectorized`
-def _safe_inv22_vectorized(M):
-    """
-    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
-    matrices.
-
-    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
-    """
-    _api.check_shape((None, 2, 2), M=M)
-    M_inv = np.empty_like(M)
-    prod1 = M[:, 0, 0]*M[:, 1, 1]
-    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
-
-    # We set delta_inv to 0. in case of a rank deficient matrix; a
-    # rank-deficient input matrix *M* will lead to a null matrix in output
-    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
-    if np.all(rank2):
-        # Normal 'optimized' flow.
-        delta_inv = 1./delta
-    else:
-        # 'Pathologic' flow.
-        delta_inv = np.zeros(M.shape[0])
-        delta_inv[rank2] = 1./delta[rank2]
-
-    M_inv[:, 0, 0] = M[:, 1, 1]*delta_inv
-    M_inv[:, 0, 1] = -M[:, 0, 1]*delta_inv
-    M_inv[:, 1, 0] = -M[:, 1, 0]*delta_inv
-    M_inv[:, 1, 1] = M[:, 0, 0]*delta_inv
-    return M_inv
-
-
-def _pseudo_inv22sym_vectorized(M):
-    """
-    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
-    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.
-
-    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
-    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
-    In case M is of rank 0, we return the null matrix.
-
-    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
-    """
-    _api.check_shape((None, 2, 2), M=M)
-    M_inv = np.empty_like(M)
-    prod1 = M[:, 0, 0]*M[:, 1, 1]
-    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
-    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
-
-    if np.all(rank2):
-        # Normal 'optimized' flow.
-        M_inv[:, 0, 0] = M[:, 1, 1] / delta
-        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
-        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
-        M_inv[:, 1, 1] = M[:, 0, 0] / delta
-    else:
-        # 'Pathologic' flow.
-        # Here we have to deal with 2 sub-cases
-        # 1) First sub-case: matrices of rank 2:
-        delta = delta[rank2]
-        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
-        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
-        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
-        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
-        # 2) Second sub-case: rank-deficient matrices of rank 0 and 1:
-        rank01 = ~rank2
-        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
-        tr_zeros = (np.abs(tr) < 1.e-8)
-        sq_tr_inv = (1.-tr_zeros) / (tr**2+tr_zeros)
-        # sq_tr_inv = 1. / tr**2
-        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
-        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
-        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
-        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv
-
-    return M_inv
-
-
-def _scalar_vectorized(scalar, M):
-    """
-    Scalar product between scalars and matrices.
-    """
-    return scalar[:, np.newaxis, np.newaxis]*M
-
-
-def _transpose_vectorized(M):
-    """
-    Transposition of an array of matrices *M*.
-    """
-    return np.transpose(M, [0, 2, 1])
-
-
-def _roll_vectorized(M, roll_indices, axis):
-    """
-    Roll an array of matrices along *axis* (0: rows, 1: columns) according to
-    an array of indices *roll_indices*.
-    """
-    assert axis in [0, 1]
-    ndim = M.ndim
-    assert ndim == 3
-    ndim_roll = roll_indices.ndim
-    assert ndim_roll == 1
-    sh = M.shape
-    r, c = sh[-2:]
-    assert sh[0] == roll_indices.shape[0]
-    vec_indices = np.arange(sh[0], dtype=np.int32)
-
-    # Builds the rolled matrix
-    M_roll = np.empty_like(M)
-    if axis == 0:
-        for ir in range(r):
-            for ic in range(c):
-                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices+ir) % r, ic]
-    else:  # 1
-        for ir in range(r):
-            for ic in range(c):
-                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices+ic) % c]
-    return M_roll
-
-
-def _to_matrix_vectorized(M):
-    """
-    Build an array of matrices from individuals np.arrays of identical shapes.
-
-    Parameters
-    ----------
-    M
-        ncols-list of nrows-lists of shape sh.
-
-    Returns
-    -------
-    M_res : np.array of shape (sh, nrow, ncols)
-        *M_res* satisfies ``M_res[..., i, j] = M[i][j]``.
-    """
-    assert isinstance(M, (tuple, list))
-    assert all(isinstance(item, (tuple, list)) for item in M)
-    c_vec = np.asarray([len(item) for item in M])
-    assert np.all(c_vec-c_vec[0] == 0)
-    r = len(M)
-    c = c_vec[0]
-    M00 = np.asarray(M[0][0])
-    dt = M00.dtype
-    sh = [M00.shape[0], r, c]
-    M_ret = np.empty(sh, dtype=dt)
-    for irow in range(r):
-        for icol in range(c):
-            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
-    return M_ret
-
-
-def _extract_submatrices(M, block_indices, block_size, axis):
-    """
-    Extract selected blocks of a matrices *M* depending on parameters
-    *block_indices* and *block_size*.
-
-    Returns the array of extracted matrices *Mres* so that ::
-
-        M_res[..., ir, :] = M[(block_indices*block_size+ir), :]
-    """
-    assert block_indices.ndim == 1
-    assert axis in [0, 1]
-
-    r, c = M.shape
-    if axis == 0:
-        sh = [block_indices.shape[0], block_size, c]
-    else:  # 1
-        sh = [block_indices.shape[0], r, block_size]
 
-    dt = M.dtype
-    M_res = np.empty(sh, dtype=dt)
-    if axis == 0:
-        for ir in range(block_size):
-            M_res[:, ir, :] = M[(block_indices*block_size+ir), :]
-    else:  # 1
-        for ic in range(block_size):
-            M_res[:, :, ic] = M[:, (block_indices*block_size+ic)]
 
-    return M_res
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/tripcolor.py b/lib/matplotlib/tri/tripcolor.py
--- a/lib/matplotlib/tri/tripcolor.py
+++ b/lib/matplotlib/tri/tripcolor.py
@@ -1,154 +1,9 @@
-import numpy as np
-
+from ._tripcolor import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.collections import PolyCollection, TriMesh
-from matplotlib.colors import Normalize
-from matplotlib.tri.triangulation import Triangulation
-
-
-def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
-              vmax=None, shading='flat', facecolors=None, **kwargs):
-    """
-    Create a pseudocolor plot of an unstructured triangular grid.
-
-    Call signatures::
-
-      tripcolor(triangulation, c, *, ...)
-      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)
-
-    The triangular grid can be specified either by passing a `.Triangulation`
-    object as the first parameter, or by passing the points *x*, *y* and
-    optionally the *triangles* and a *mask*. See `.Triangulation` for an
-    explanation of these parameters.
-
-    It is possible to pass the triangles positionally, i.e.
-    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
-    For more clarity, pass *triangles* via keyword argument.
-
-    If neither of *triangulation* or *triangles* are given, the triangulation
-    is calculated on the fly. In this case, it does not make sense to provide
-    colors at the triangle faces via *c* or *facecolors* because there are
-    multiple possible triangulations for a group of points and you don't know
-    which triangles will be constructed.
-
-    Parameters
-    ----------
-    triangulation : `.Triangulation`
-        An already created triangular grid.
-    x, y, triangles, mask
-        Parameters defining the triangular grid. See `.Triangulation`.
-        This is mutually exclusive with specifying *triangulation*.
-    c : array-like
-        The color values, either for the points or for the triangles. Which one
-        is automatically inferred from the length of *c*, i.e. does it match
-        the number of points or the number of triangles. If there are the same
-        number of points and triangles in the triangulation it is assumed that
-        color values are defined at points; to force the use of color values at
-        triangles use the keyword argument ``facecolors=c`` instead of just
-        ``c``.
-        This parameter is position-only.
-    facecolors : array-like, optional
-        Can be used alternatively to *c* to specify colors at the triangle
-        faces. This parameter takes precedence over *c*.
-    shading : {'flat', 'gouraud'}, default: 'flat'
-        If  'flat' and the color values *c* are defined at points, the color
-        values used for each triangle are from the mean c of the triangle's
-        three points. If *shading* is 'gouraud' then color values must be
-        defined at points.
-    other_parameters
-        All other parameters are the same as for `~.Axes.pcolor`.
-    """
-    _api.check_in_list(['flat', 'gouraud'], shading=shading)
-
-    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
-
-    # Parse the color to be in one of (the other variable will be None):
-    # - facecolors: if specified at the triangle faces
-    # - point_colors: if specified at the points
-    if facecolors is not None:
-        if args:
-            _api.warn_external(
-                "Positional parameter c has no effect when the keyword "
-                "facecolors is given")
-        point_colors = None
-        if len(facecolors) != len(tri.triangles):
-            raise ValueError("The length of facecolors must match the number "
-                             "of triangles")
-    else:
-        # Color from positional parameter c
-        if not args:
-            raise TypeError(
-                "tripcolor() missing 1 required positional argument: 'c'; or "
-                "1 required keyword-only argument: 'facecolors'")
-        elif len(args) > 1:
-            _api.warn_deprecated(
-                "3.6", message=f"Additional positional parameters "
-                f"{args[1:]!r} are ignored; support for them is deprecated "
-                f"since %(since)s and will be removed %(removal)s")
-        c = np.asarray(args[0])
-        if len(c) == len(tri.x):
-            # having this before the len(tri.triangles) comparison gives
-            # precedence to nodes if there are as many nodes as triangles
-            point_colors = c
-            facecolors = None
-        elif len(c) == len(tri.triangles):
-            point_colors = None
-            facecolors = c
-        else:
-            raise ValueError('The length of c must match either the number '
-                             'of points or the number of triangles')
-
-    # Handling of linewidths, shading, edgecolors and antialiased as
-    # in Axes.pcolor
-    linewidths = (0.25,)
-    if 'linewidth' in kwargs:
-        kwargs['linewidths'] = kwargs.pop('linewidth')
-    kwargs.setdefault('linewidths', linewidths)
-
-    edgecolors = 'none'
-    if 'edgecolor' in kwargs:
-        kwargs['edgecolors'] = kwargs.pop('edgecolor')
-    ec = kwargs.setdefault('edgecolors', edgecolors)
-
-    if 'antialiased' in kwargs:
-        kwargs['antialiaseds'] = kwargs.pop('antialiased')
-    if 'antialiaseds' not in kwargs and ec.lower() == "none":
-        kwargs['antialiaseds'] = False
-
-    _api.check_isinstance((Normalize, None), norm=norm)
-    if shading == 'gouraud':
-        if facecolors is not None:
-            raise ValueError(
-                "shading='gouraud' can only be used when the colors "
-                "are specified at the points, not at the faces.")
-        collection = TriMesh(tri, alpha=alpha, array=point_colors,
-                             cmap=cmap, norm=norm, **kwargs)
-    else:  # 'flat'
-        # Vertices of triangles.
-        maskedTris = tri.get_masked_triangles()
-        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)
-
-        # Color values.
-        if facecolors is None:
-            # One color per triangle, the mean of the 3 vertex color values.
-            colors = point_colors[maskedTris].mean(axis=1)
-        elif tri.mask is not None:
-            # Remove color values of masked triangles.
-            colors = facecolors[~tri.mask]
-        else:
-            colors = facecolors
-        collection = PolyCollection(verts, alpha=alpha, array=colors,
-                                    cmap=cmap, norm=norm, **kwargs)
 
-    collection._scale_norm(norm, vmin, vmax)
-    ax.grid(False)
 
-    minx = tri.x.min()
-    maxx = tri.x.max()
-    miny = tri.y.min()
-    maxy = tri.y.max()
-    corners = (minx, miny), (maxx, maxy)
-    ax.update_datalim(corners)
-    ax.autoscale_view()
-    ax.add_collection(collection)
-    return collection
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/triplot.py b/lib/matplotlib/tri/triplot.py
--- a/lib/matplotlib/tri/triplot.py
+++ b/lib/matplotlib/tri/triplot.py
@@ -1,86 +1,9 @@
-import numpy as np
-from matplotlib.tri.triangulation import Triangulation
-import matplotlib.cbook as cbook
-import matplotlib.lines as mlines
+from ._triplot import *  # noqa: F401, F403
+from matplotlib import _api
 
 
-def triplot(ax, *args, **kwargs):
-    """
-    Draw an unstructured triangular grid as lines and/or markers.
-
-    Call signatures::
-
-      triplot(triangulation, ...)
-      triplot(x, y, [triangles], *, [mask=mask], ...)
-
-    The triangular grid can be specified either by passing a `.Triangulation`
-    object as the first parameter, or by passing the points *x*, *y* and
-    optionally the *triangles* and a *mask*. If neither of *triangulation* or
-    *triangles* are given, the triangulation is calculated on the fly.
-
-    Parameters
-    ----------
-    triangulation : `.Triangulation`
-        An already created triangular grid.
-    x, y, triangles, mask
-        Parameters defining the triangular grid. See `.Triangulation`.
-        This is mutually exclusive with specifying *triangulation*.
-    other_parameters
-        All other args and kwargs are forwarded to `~.Axes.plot`.
-
-    Returns
-    -------
-    lines : `~matplotlib.lines.Line2D`
-        The drawn triangles edges.
-    markers : `~matplotlib.lines.Line2D`
-        The drawn marker nodes.
-    """
-    import matplotlib.axes
-
-    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
-    x, y, edges = (tri.x, tri.y, tri.edges)
-
-    # Decode plot format string, e.g., 'ro-'
-    fmt = args[0] if args else ""
-    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)
-
-    # Insert plot format string into a copy of kwargs (kwargs values prevail).
-    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
-    for key, val in zip(('linestyle', 'marker', 'color'),
-                        (linestyle, marker, color)):
-        if val is not None:
-            kw.setdefault(key, val)
-
-    # Draw lines without markers.
-    # Note 1: If we drew markers here, most markers would be drawn more than
-    #         once as they belong to several edges.
-    # Note 2: We insert nan values in the flattened edges arrays rather than
-    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
-    #         as it considerably speeds-up code execution.
-    linestyle = kw['linestyle']
-    kw_lines = {
-        **kw,
-        'marker': 'None',  # No marker to draw.
-        'zorder': kw.get('zorder', 1),  # Path default zorder is used.
-    }
-    if linestyle not in [None, 'None', '', ' ']:
-        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
-        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
-        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
-                            **kw_lines)
-    else:
-        tri_lines = ax.plot([], [], **kw_lines)
-
-    # Draw markers separately.
-    marker = kw['marker']
-    kw_markers = {
-        **kw,
-        'linestyle': 'None',  # No line to draw.
-    }
-    kw_markers.pop('label', None)
-    if marker not in [None, 'None', '', ' ']:
-        tri_markers = ax.plot(x, y, **kw_markers)
-    else:
-        tri_markers = ax.plot([], [], **kw_markers)
-
-    return tri_lines + tri_markers
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/trirefine.py b/lib/matplotlib/tri/trirefine.py
--- a/lib/matplotlib/tri/trirefine.py
+++ b/lib/matplotlib/tri/trirefine.py
@@ -1,307 +1,9 @@
-"""
-Mesh refinement for triangular grids.
-"""
-
-import numpy as np
-
+from ._trirefine import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri.triangulation import Triangulation
-import matplotlib.tri.triinterpolate
-
-
-class TriRefiner:
-    """
-    Abstract base class for classes implementing mesh refinement.
-
-    A TriRefiner encapsulates a Triangulation object and provides tools for
-    mesh refinement and interpolation.
-
-    Derived classes must implement:
-
-    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
-      the optional keyword arguments *kwargs* are defined in each
-      TriRefiner concrete implementation, and which returns:
-
-      - a refined triangulation,
-      - optionally (depending on *return_tri_index*), for each
-        point of the refined triangulation: the index of
-        the initial triangulation triangle to which it belongs.
-
-    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:
-
-      - *z* array of field values (to refine) defined at the base
-        triangulation nodes,
-      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
-      - the other optional keyword arguments *kwargs* are defined in
-        each TriRefiner concrete implementation;
-
-      and which returns (as a tuple) a refined triangular mesh and the
-      interpolated values of the field at the refined triangulation nodes.
-    """
-
-    def __init__(self, triangulation):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-
-class UniformTriRefiner(TriRefiner):
-    """
-    Uniform mesh refinement by recursive subdivisions.
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The encapsulated triangulation (to be refined)
-    """
-#    See Also
-#    --------
-#    :class:`~matplotlib.tri.CubicTriInterpolator` and
-#    :class:`~matplotlib.tri.TriAnalyzer`.
-#    """
-    def __init__(self, triangulation):
-        super().__init__(triangulation)
-
-    def refine_triangulation(self, return_tri_index=False, subdiv=3):
-        """
-        Compute an uniformly refined triangulation *refi_triangulation* of
-        the encapsulated :attr:`triangulation`.
-
-        This function refines the encapsulated triangulation by splitting each
-        father triangle into 4 child sub-triangles built on the edges midside
-        nodes, recursing *subdiv* times.  In the end, each triangle is hence
-        divided into ``4**subdiv`` child triangles.
-
-        Parameters
-        ----------
-        return_tri_index : bool, default: False
-            Whether an index table indicating the father triangle index of each
-            point is returned.
-        subdiv : int, default: 3
-            Recursion level for the subdivision.
-            Each triangle is divided into ``4**subdiv`` child triangles;
-            hence, the default results in 64 refined subtriangles for each
-            triangle of the initial triangulation.
-
-        Returns
-        -------
-        refi_triangulation : `~matplotlib.tri.Triangulation`
-            The refined triangulation.
-        found_index : int array
-            Index of the initial triangulation containing triangle, for each
-            point of *refi_triangulation*.
-            Returned only if *return_tri_index* is set to True.
-        """
-        refi_triangulation = self._triangulation
-        ntri = refi_triangulation.triangles.shape[0]
-
-        # Computes the triangulation ancestors numbers in the reference
-        # triangulation.
-        ancestors = np.arange(ntri, dtype=np.int32)
-        for _ in range(subdiv):
-            refi_triangulation, ancestors = self._refine_triangulation_once(
-                refi_triangulation, ancestors)
-        refi_npts = refi_triangulation.x.shape[0]
-        refi_triangles = refi_triangulation.triangles
-
-        # Now we compute found_index table if needed
-        if return_tri_index:
-            # We have to initialize found_index with -1 because some nodes
-            # may very well belong to no triangle at all, e.g., in case of
-            # Delaunay Triangulation with DuplicatePointWarning.
-            found_index = np.full(refi_npts, -1, dtype=np.int32)
-            tri_mask = self._triangulation.mask
-            if tri_mask is None:
-                found_index[refi_triangles] = np.repeat(ancestors,
-                                                        3).reshape(-1, 3)
-            else:
-                # There is a subtlety here: we want to avoid whenever possible
-                # that refined points container is a masked triangle (which
-                # would result in artifacts in plots).
-                # So we impose the numbering from masked ancestors first,
-                # then overwrite it with unmasked ancestor numbers.
-                ancestor_mask = tri_mask[ancestors]
-                found_index[refi_triangles[ancestor_mask, :]
-                            ] = np.repeat(ancestors[ancestor_mask],
-                                          3).reshape(-1, 3)
-                found_index[refi_triangles[~ancestor_mask, :]
-                            ] = np.repeat(ancestors[~ancestor_mask],
-                                          3).reshape(-1, 3)
-            return refi_triangulation, found_index
-        else:
-            return refi_triangulation
-
-    def refine_field(self, z, triinterpolator=None, subdiv=3):
-        """
-        Refine a field defined on the encapsulated triangulation.
-
-        Parameters
-        ----------
-        z : (npoints,) array-like
-            Values of the field to refine, defined at the nodes of the
-            encapsulated triangulation. (``n_points`` is the number of points
-            in the initial triangulation)
-        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
-            Interpolator used for field interpolation. If not specified,
-            a `~matplotlib.tri.CubicTriInterpolator` will be used.
-        subdiv : int, default: 3
-            Recursion level for the subdivision.
-            Each triangle is divided into ``4**subdiv`` child triangles.
-
-        Returns
-        -------
-        refi_tri : `~matplotlib.tri.Triangulation`
-             The returned refined triangulation.
-        refi_z : 1D array of length: *refi_tri* node count.
-             The returned interpolated field (at *refi_tri* nodes).
-        """
-        if triinterpolator is None:
-            interp = matplotlib.tri.CubicTriInterpolator(
-                self._triangulation, z)
-        else:
-            _api.check_isinstance(matplotlib.tri.TriInterpolator,
-                                  triinterpolator=triinterpolator)
-            interp = triinterpolator
-
-        refi_tri, found_index = self.refine_triangulation(
-            subdiv=subdiv, return_tri_index=True)
-        refi_z = interp._interpolate_multikeys(
-            refi_tri.x, refi_tri.y, tri_index=found_index)[0]
-        return refi_tri, refi_z
-
-    @staticmethod
-    def _refine_triangulation_once(triangulation, ancestors=None):
-        """
-        Refine a `.Triangulation` by splitting each triangle into 4
-        child-masked_triangles built on the edges midside nodes.
-
-        Masked triangles, if present, are also split, but their children
-        returned masked.
-
-        If *ancestors* is not provided, returns only a new triangulation:
-        child_triangulation.
-
-        If the array-like key table *ancestor* is given, it shall be of shape
-        (ntri,) where ntri is the number of *triangulation* masked_triangles.
-        In this case, the function returns
-        (child_triangulation, child_ancestors)
-        child_ancestors is defined so that the 4 child masked_triangles share
-        the same index as their father: child_ancestors.shape = (4 * ntri,).
-        """
-
-        x = triangulation.x
-        y = triangulation.y
-
-        #    According to tri.triangulation doc:
-        #         neighbors[i, j] is the triangle that is the neighbor
-        #         to the edge from point index masked_triangles[i, j] to point
-        #         index masked_triangles[i, (j+1)%3].
-        neighbors = triangulation.neighbors
-        triangles = triangulation.triangles
-        npts = np.shape(x)[0]
-        ntri = np.shape(triangles)[0]
-        if ancestors is not None:
-            ancestors = np.asarray(ancestors)
-            if np.shape(ancestors) != (ntri,):
-                raise ValueError(
-                    "Incompatible shapes provide for triangulation"
-                    ".masked_triangles and ancestors: {0} and {1}".format(
-                        np.shape(triangles), np.shape(ancestors)))
-
-        # Initiating tables refi_x and refi_y of the refined triangulation
-        # points
-        # hint: each apex is shared by 2 masked_triangles except the borders.
-        borders = np.sum(neighbors == -1)
-        added_pts = (3*ntri + borders) // 2
-        refi_npts = npts + added_pts
-        refi_x = np.zeros(refi_npts)
-        refi_y = np.zeros(refi_npts)
-
-        # First part of refi_x, refi_y is just the initial points
-        refi_x[:npts] = x
-        refi_y[:npts] = y
-
-        # Second part contains the edge midside nodes.
-        # Each edge belongs to 1 triangle (if border edge) or is shared by 2
-        # masked_triangles (interior edge).
-        # We first build 2 * ntri arrays of edge starting nodes (edge_elems,
-        # edge_apexes); we then extract only the masters to avoid overlaps.
-        # The so-called 'master' is the triangle with biggest index
-        # The 'slave' is the triangle with lower index
-        # (can be -1 if border edge)
-        # For slave and master we will identify the apex pointing to the edge
-        # start
-        edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
-        edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
-        edge_neighbors = neighbors[edge_elems, edge_apexes]
-        mask_masters = (edge_elems > edge_neighbors)
-
-        # Identifying the "masters" and adding to refi_x, refi_y vec
-        masters = edge_elems[mask_masters]
-        apex_masters = edge_apexes[mask_masters]
-        x_add = (x[triangles[masters, apex_masters]] +
-                 x[triangles[masters, (apex_masters+1) % 3]]) * 0.5
-        y_add = (y[triangles[masters, apex_masters]] +
-                 y[triangles[masters, (apex_masters+1) % 3]]) * 0.5
-        refi_x[npts:] = x_add
-        refi_y[npts:] = y_add
-
-        # Building the new masked_triangles; each old masked_triangles hosts
-        # 4 new masked_triangles
-        # there are 6 pts to identify per 'old' triangle, 3 new_pt_corner and
-        # 3 new_pt_midside
-        new_pt_corner = triangles
-
-        # What is the index in refi_x, refi_y of point at middle of apex iapex
-        #  of elem ielem ?
-        # If ielem is the apex master: simple count, given the way refi_x was
-        #  built.
-        # If ielem is the apex slave: yet we do not know; but we will soon
-        # using the neighbors table.
-        new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
-        cum_sum = npts
-        for imid in range(3):
-            mask_st_loc = (imid == apex_masters)
-            n_masters_loc = np.sum(mask_st_loc)
-            elem_masters_loc = masters[mask_st_loc]
-            new_pt_midside[:, imid][elem_masters_loc] = np.arange(
-                n_masters_loc, dtype=np.int32) + cum_sum
-            cum_sum += n_masters_loc
-
-        # Now dealing with slave elems.
-        # for each slave element we identify the master and then the inode
-        # once slave_masters is identified, slave_masters_apex is such that:
-        # neighbors[slaves_masters, slave_masters_apex] == slaves
-        mask_slaves = np.logical_not(mask_masters)
-        slaves = edge_elems[mask_slaves]
-        slaves_masters = edge_neighbors[mask_slaves]
-        diff_table = np.abs(neighbors[slaves_masters, :] -
-                            np.outer(slaves, np.ones(3, dtype=np.int32)))
-        slave_masters_apex = np.argmin(diff_table, axis=1)
-        slaves_apex = edge_apexes[mask_slaves]
-        new_pt_midside[slaves, slaves_apex] = new_pt_midside[
-            slaves_masters, slave_masters_apex]
-
-        # Builds the 4 child masked_triangles
-        child_triangles = np.empty([ntri*4, 3], dtype=np.int32)
-        child_triangles[0::4, :] = np.vstack([
-            new_pt_corner[:, 0], new_pt_midside[:, 0],
-            new_pt_midside[:, 2]]).T
-        child_triangles[1::4, :] = np.vstack([
-            new_pt_corner[:, 1], new_pt_midside[:, 1],
-            new_pt_midside[:, 0]]).T
-        child_triangles[2::4, :] = np.vstack([
-            new_pt_corner[:, 2], new_pt_midside[:, 2],
-            new_pt_midside[:, 1]]).T
-        child_triangles[3::4, :] = np.vstack([
-            new_pt_midside[:, 0], new_pt_midside[:, 1],
-            new_pt_midside[:, 2]]).T
-        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
 
-        # Builds the child mask
-        if triangulation.mask is not None:
-            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
 
-        if ancestors is None:
-            return child_triangulation
-        else:
-            return child_triangulation, np.repeat(ancestors, 4)
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/tritools.py b/lib/matplotlib/tri/tritools.py
--- a/lib/matplotlib/tri/tritools.py
+++ b/lib/matplotlib/tri/tritools.py
@@ -1,263 +1,9 @@
-"""
-Tools for triangular grids.
-"""
-
-import numpy as np
-
+from ._tritools import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri import Triangulation
-
-
-class TriAnalyzer:
-    """
-    Define basic tools for triangular mesh analysis and improvement.
-
-    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
-    tools for mesh analysis and mesh improvement.
-
-    Attributes
-    ----------
-    scale_factors
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The encapsulated triangulation to analyze.
-    """
-
-    def __init__(self, triangulation):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-    @property
-    def scale_factors(self):
-        """
-        Factors to rescale the triangulation into a unit square.
-
-        Returns
-        -------
-        (float, float)
-            Scaling factors (kx, ky) so that the triangulation
-            ``[triangulation.x * kx, triangulation.y * ky]``
-            fits exactly inside a unit square.
-        """
-        compressed_triangles = self._triangulation.get_masked_triangles()
-        node_used = (np.bincount(np.ravel(compressed_triangles),
-                                 minlength=self._triangulation.x.size) != 0)
-        return (1 / np.ptp(self._triangulation.x[node_used]),
-                1 / np.ptp(self._triangulation.y[node_used]))
-
-    def circle_ratios(self, rescale=True):
-        """
-        Return a measure of the triangulation triangles flatness.
-
-        The ratio of the incircle radius over the circumcircle radius is a
-        widely used indicator of a triangle flatness.
-        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
-        triangles. Circle ratios below 0.01 denote very flat triangles.
-
-        To avoid unduly low values due to a difference of scale between the 2
-        axis, the triangular mesh can first be rescaled to fit inside a unit
-        square with `scale_factors` (Only if *rescale* is True, which is
-        its default value).
-
-        Parameters
-        ----------
-        rescale : bool, default: True
-            If True, internally rescale (based on `scale_factors`), so that the
-            (unmasked) triangles fit exactly inside a unit square mesh.
-
-        Returns
-        -------
-        masked array
-            Ratio of the incircle radius over the circumcircle radius, for
-            each 'rescaled' triangle of the encapsulated triangulation.
-            Values corresponding to masked triangles are masked out.
-
-        """
-        # Coords rescaling
-        if rescale:
-            (kx, ky) = self.scale_factors
-        else:
-            (kx, ky) = (1.0, 1.0)
-        pts = np.vstack([self._triangulation.x*kx,
-                         self._triangulation.y*ky]).T
-        tri_pts = pts[self._triangulation.triangles]
-        # Computes the 3 side lengths
-        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
-        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
-        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
-        a = np.hypot(a[:, 0], a[:, 1])
-        b = np.hypot(b[:, 0], b[:, 1])
-        c = np.hypot(c[:, 0], c[:, 1])
-        # circumcircle and incircle radii
-        s = (a+b+c)*0.5
-        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
-        # We have to deal with flat triangles with infinite circum_radius
-        bool_flat = (prod == 0.)
-        if np.any(bool_flat):
-            # Pathologic flow
-            ntri = tri_pts.shape[0]
-            circum_radius = np.empty(ntri, dtype=np.float64)
-            circum_radius[bool_flat] = np.inf
-            abc = a*b*c
-            circum_radius[~bool_flat] = abc[~bool_flat] / (
-                4.0*np.sqrt(prod[~bool_flat]))
-        else:
-            # Normal optimized flow
-            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
-        in_radius = (a*b*c) / (4.0*circum_radius*s)
-        circle_ratio = in_radius/circum_radius
-        mask = self._triangulation.mask
-        if mask is None:
-            return circle_ratio
-        else:
-            return np.ma.array(circle_ratio, mask=mask)
-
-    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
-        """
-        Eliminate excessively flat border triangles from the triangulation.
-
-        Returns a mask *new_mask* which allows to clean the encapsulated
-        triangulation from its border-located flat triangles
-        (according to their :meth:`circle_ratios`).
-        This mask is meant to be subsequently applied to the triangulation
-        using `.Triangulation.set_mask`.
-        *new_mask* is an extension of the initial triangulation mask
-        in the sense that an initially masked triangle will remain masked.
-
-        The *new_mask* array is computed recursively; at each step flat
-        triangles are removed only if they share a side with the current mesh
-        border. Thus no new holes in the triangulated domain will be created.
-
-        Parameters
-        ----------
-        min_circle_ratio : float, default: 0.01
-            Border triangles with incircle/circumcircle radii ratio r/R will
-            be removed if r/R < *min_circle_ratio*.
-        rescale : bool, default: True
-            If True, first, internally rescale (based on `scale_factors`) so
-            that the (unmasked) triangles fit exactly inside a unit square
-            mesh.  This rescaling accounts for the difference of scale which
-            might exist between the 2 axis.
-
-        Returns
-        -------
-        array of bool
-            Mask to apply to encapsulated triangulation.
-            All the initially masked triangles remain masked in the
-            *new_mask*.
-
-        Notes
-        -----
-        The rationale behind this function is that a Delaunay
-        triangulation - of an unstructured set of points - sometimes contains
-        almost flat triangles at its border, leading to artifacts in plots
-        (especially for high-resolution contouring).
-        Masked with computed *new_mask*, the encapsulated
-        triangulation would contain no more unmasked border triangles
-        with a circle ratio below *min_circle_ratio*, thus improving the
-        mesh quality for subsequent plots or interpolation.
-        """
-        # Recursively computes the mask_current_borders, true if a triangle is
-        # at the border of the mesh OR touching the border through a chain of
-        # invalid aspect ratio masked_triangles.
-        ntri = self._triangulation.triangles.shape[0]
-        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
-
-        current_mask = self._triangulation.mask
-        if current_mask is None:
-            current_mask = np.zeros(ntri, dtype=bool)
-        valid_neighbors = np.copy(self._triangulation.neighbors)
-        renum_neighbors = np.arange(ntri, dtype=np.int32)
-        nadd = -1
-        while nadd != 0:
-            # The active wavefront is the triangles from the border (unmasked
-            # but with a least 1 neighbor equal to -1
-            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
-            # The element from the active wavefront will be masked if their
-            # circle ratio is bad.
-            added_mask = wavefront & mask_bad_ratio
-            current_mask = added_mask | current_mask
-            nadd = np.sum(added_mask)
-
-            # now we have to update the tables valid_neighbors
-            valid_neighbors[added_mask, :] = -1
-            renum_neighbors[added_mask] = -1
-            valid_neighbors = np.where(valid_neighbors == -1, -1,
-                                       renum_neighbors[valid_neighbors])
-
-        return np.ma.filled(current_mask, True)
-
-    def _get_compressed_triangulation(self):
-        """
-        Compress (if masked) the encapsulated triangulation.
-
-        Returns minimal-length triangles array (*compressed_triangles*) and
-        coordinates arrays (*compressed_x*, *compressed_y*) that can still
-        describe the unmasked triangles of the encapsulated triangulation.
-
-        Returns
-        -------
-        compressed_triangles : array-like
-            the returned compressed triangulation triangles
-        compressed_x : array-like
-            the returned compressed triangulation 1st coordinate
-        compressed_y : array-like
-            the returned compressed triangulation 2nd coordinate
-        tri_renum : int array
-            renumbering table to translate the triangle numbers from the
-            encapsulated triangulation into the new (compressed) renumbering.
-            -1 for masked triangles (deleted from *compressed_triangles*).
-        node_renum : int array
-            renumbering table to translate the point numbers from the
-            encapsulated triangulation into the new (compressed) renumbering.
-            -1 for unused points (i.e. those deleted from *compressed_x* and
-            *compressed_y*).
-
-        """
-        # Valid triangles and renumbering
-        tri_mask = self._triangulation.mask
-        compressed_triangles = self._triangulation.get_masked_triangles()
-        ntri = self._triangulation.triangles.shape[0]
-        if tri_mask is not None:
-            tri_renum = self._total_to_compress_renum(~tri_mask)
-        else:
-            tri_renum = np.arange(ntri, dtype=np.int32)
-
-        # Valid nodes and renumbering
-        valid_node = (np.bincount(np.ravel(compressed_triangles),
-                                  minlength=self._triangulation.x.size) != 0)
-        compressed_x = self._triangulation.x[valid_node]
-        compressed_y = self._triangulation.y[valid_node]
-        node_renum = self._total_to_compress_renum(valid_node)
-
-        # Now renumbering the valid triangles nodes
-        compressed_triangles = node_renum[compressed_triangles]
-
-        return (compressed_triangles, compressed_x, compressed_y, tri_renum,
-                node_renum)
-
-    @staticmethod
-    def _total_to_compress_renum(valid):
-        """
-        Parameters
-        ----------
-        valid : 1D bool array
-            Validity mask.
 
-        Returns
-        -------
-        int array
-            Array so that (`valid_array` being a compressed array
-            based on a `masked_array` with mask ~*valid*):
 
-            - For all i with valid[i] = True:
-              valid_array[renum[i]] = masked_array[i]
-            - For all i with valid[i] = False:
-              renum[i] = -1 (invalid value)
-        """
-        renum = np.full(np.size(valid), -1, dtype=np.int32)
-        n_valid = np.sum(valid)
-        renum[valid] = np.arange(n_valid, dtype=np.int32)
-        return renum
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/mpl_toolkits/mplot3d/axes3d.py b/lib/mpl_toolkits/mplot3d/axes3d.py
--- a/lib/mpl_toolkits/mplot3d/axes3d.py
+++ b/lib/mpl_toolkits/mplot3d/axes3d.py
@@ -32,7 +32,7 @@
 from matplotlib.axes import Axes
 from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
 from matplotlib.transforms import Bbox
-from matplotlib.tri.triangulation import Triangulation
+from matplotlib.tri._triangulation import Triangulation
 
 from . import art3d
 from . import proj3d
@@ -2153,7 +2153,7 @@ def tricontour(self, *args,
 
         Returns
         -------
-        matplotlib.tri.tricontour.TriContourSet
+        matplotlib.tri._tricontour.TriContourSet
         """
         had_data = self.has_data()
 
@@ -2246,7 +2246,7 @@ def tricontourf(self, *args, zdir='z', offset=None, **kwargs):
 
         Returns
         -------
-        matplotlib.tri.tricontour.TriContourSet
+        matplotlib.tri._tricontour.TriContourSet
         """
         had_data = self.has_data()
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/tri/__init__.py | 5 | 13 | - | - | -
| lib/matplotlib/tri/_triangulation.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_tricontour.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_trifinder.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_triinterpolate.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_tripcolor.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_triplot.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_trirefine.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/_tritools.py | 0 | 0 | - | - | -
| lib/matplotlib/tri/triangulation.py | 1 | 240 | - | 26 | -
| lib/matplotlib/tri/tricontour.py | 1 | 271 | - | - | -
| lib/matplotlib/tri/trifinder.py | 1 | 93 | - | - | -
| lib/matplotlib/tri/triinterpolate.py | 1 | 1574 | - | - | -
| lib/matplotlib/tri/tripcolor.py | 1 | 154 | - | 5 | -
| lib/matplotlib/tri/triplot.py | 1 | 86 | - | - | -
| lib/matplotlib/tri/trirefine.py | 1 | 307 | - | - | -
| lib/matplotlib/tri/tritools.py | 1 | 263 | - | - | -
| lib/mpl_toolkits/mplot3d/axes3d.py | 35 | 35 | - | - | -
| lib/mpl_toolkits/mplot3d/axes3d.py | 2156 | 2156 | - | - | -
| lib/mpl_toolkits/mplot3d/axes3d.py | 2249 | 2249 | - | - | -


## Problem Statement

```
function shadowing their own definition modules
I'm not sure if this is really a "bug" report but more of an unexpected interaction. The short reason for this is that I'm working on improving the documentation in IPython and need a bijection object <-> fully qualified name which is made difficult by the following. I take the example of tripcolor, but this is not the only object that shadow it's module definition.

### Bug report

`matplotlib.tri.tripcolor` refer either as a module or function depending on context:

\`\`\`
>>> from matplotlib.tri.tripcolor import tripcolor
>>> tripcolor.__module__
'matplotlib.tri.tripcolor'
\`\`\`
Therefore those two lines confort us that `matplotlib.tri.tripcolor` is a module.

Though

\`\`\`
>>> matplotlib.tri.tripcolor is tripcolor
True
\`\`\`

This is not too shocking for the advanced pythonista, as `tri/__init__.py:` contains
\`\`\`
...
from .tripcolor import * 
\`\`\`

Though it makes it hard to get access to the tripcolor module, though still possible via `importlib.import_module`, but make getting the object from it's fully qualified name difficult:

\`\`\`
In [6]: qualname = tripcolor.__module__+ '.' + tripcolor.__name__
   ...: obj = matplotlib
   ...: for k in qualname.split('.')[1:]:
   ...:     obj = getattr(obj, k)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-25-8f431e6ed783> in <module>
      2 obj = matplotlib
      3 for k in qualname.split('.')[1:]:
----> 4     obj = getattr(obj, k)

AttributeError: 'function' object has no attribute 'tripcolor'
\`\`\`

I'd like to suggest to rename the tripcolor submodule to _tripcolor, or anything else which is different than the name of the function so that the function and module where it is defined have non-clashing fully-qualified names. 

Note that this is not completely API-compatible, as the shown above `from matplotlib.tri.tripcolor import tripcolor` would not work – though the correct import form is `from matplotlib.tri import tripcolor` that should still work.

Is that a general concern in the matplotlib codebase and is there a desire that `obj.__module__+'.'+obj.__name__` should allow to get the fully qualified name of the object and should allow recursive call to getattr/import in order to access the object?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 doc/conf.py | 111 | 158| 481 | 481 | 5611 | 
| 2 | 2 lib/matplotlib/pyplot.py | 2956 | 2971| 184 | 665 | 33691 | 
| 3 | 3 lib/matplotlib/__init__.py | 107 | 156| 328 | 993 | 45742 | 
| 4 | 3 lib/matplotlib/__init__.py | 1 | 106| 767 | 1760 | 45742 | 
| 5 | 3 lib/matplotlib/pyplot.py | 92 | 119| 223 | 1983 | 45742 | 
| 6 | 3 doc/conf.py | 635 | 708| 477 | 2460 | 45742 | 
| 7 | 4 lib/matplotlib/cbook/__init__.py | 33 | 43| 126 | 2586 | 65060 | 
| 8 | 4 lib/matplotlib/__init__.py | 203 | 229| 253 | 2839 | 65060 | 
| 9 | **5 lib/matplotlib/tri/tripcolor.py** | 68 | 155| 851 | 3690 | 66588 | 
| 10 | 6 plot_types/unstructured/tripcolor.py | 1 | 28| 170 | 3860 | 66758 | 
| 11 | 6 lib/matplotlib/pyplot.py | 569 | 601| 231 | 4091 | 66758 | 
| 12 | 7 doc/sphinxext/skip_deprecated.py | 1 | 18| 128 | 4219 | 66887 | 
| 13 | 7 doc/conf.py | 178 | 273| 760 | 4979 | 66887 | 
| 14 | 7 lib/matplotlib/pyplot.py | 1 | 89| 669 | 5648 | 66887 | 
| 15 | 8 lib/matplotlib/_docstring.py | 64 | 98| 265 | 5913 | 67575 | 
| 16 | 9 examples/images_contours_and_fields/tripcolor_demo.py | 1 | 61| 475 | 6388 | 69918 | 
| 17 | 10 tutorials/advanced/patheffects_guide.py | 1 | 101| 918 | 7306 | 70964 | 
| 18 | 10 examples/images_contours_and_fields/tripcolor_demo.py | 103 | 138| 319 | 7625 | 70964 | 
| 19 | 10 lib/matplotlib/__init__.py | 1444 | 1488| 404 | 8029 | 70964 | 
| 20 | 11 lib/matplotlib/pylab.py | 1 | 51| 387 | 8416 | 71352 | 
| 21 | 12 lib/matplotlib/axes/_base.py | 1 | 32| 212 | 8628 | 110714 | 
| 22 | 13 lib/matplotlib/testing/decorators.py | 1 | 45| 256 | 8884 | 114869 | 
| 23 | 13 lib/matplotlib/axes/_base.py | 469 | 4640| 813 | 9697 | 114869 | 
| 24 | 14 tutorials/colors/colormaps.py | 202 | 259| 653 | 10350 | 119642 | 
| 25 | 15 lib/matplotlib/_api/__init__.py | 1 | 24| 136 | 10486 | 122588 | 
| 26 | 16 lib/matplotlib/artist.py | 1533 | 1547| 162 | 10648 | 135963 | 
| 27 | 17 lib/matplotlib/backends/backend_gtk3.py | 1 | 52| 377 | 11025 | 140805 | 
| 28 | 17 doc/conf.py | 561 | 633| 639 | 11664 | 140805 | 
| 29 | 17 doc/conf.py | 1 | 84| 646 | 12310 | 140805 | 
| 30 | 17 lib/matplotlib/testing/decorators.py | 175 | 210| 309 | 12619 | 140805 | 
| 31 | 18 lib/matplotlib/animation.py | 1 | 49| 356 | 12975 | 155618 | 
| 32 | 18 lib/matplotlib/_api/__init__.py | 195 | 227| 292 | 13267 | 155618 | 
| 33 | 18 doc/conf.py | 87 | 109| 195 | 13462 | 155618 | 
| 34 | 19 lib/matplotlib/colorbar.py | 30 | 115| 961 | 14423 | 169667 | 
| 35 | 20 lib/matplotlib/patheffects.py | 259 | 285| 251 | 14674 | 173961 | 
| 36 | 21 tutorials/toolkits/axisartist.py | 1 | 563| 4711 | 19385 | 178672 | 
| 37 | 21 doc/conf.py | 274 | 331| 412 | 19797 | 178672 | 
| 38 | 22 examples/shapes_and_collections/artist_reference.py | 90 | 130| 319 | 20116 | 179832 | 
| 39 | 22 lib/matplotlib/testing/decorators.py | 212 | 235| 202 | 20318 | 179832 | 
| 40 | 22 lib/matplotlib/pyplot.py | 2060 | 2073| 138 | 20456 | 179832 | 
| 41 | 23 tests.py | 1 | 35| 238 | 20694 | 180070 | 
| 42 | 23 lib/matplotlib/patheffects.py | 321 | 342| 225 | 20919 | 180070 | 
| 43 | 23 lib/matplotlib/artist.py | 1 | 21| 117 | 21036 | 180070 | 
| 44 | 23 lib/matplotlib/cbook/__init__.py | 414 | 449| 274 | 21310 | 180070 | 
| 45 | 23 lib/matplotlib/pyplot.py | 341 | 359| 175 | 21485 | 180070 | 
| 46 | 23 tutorials/advanced/patheffects_guide.py | 102 | 119| 127 | 21612 | 180070 | 
| 47 | 24 examples/misc/pythonic_matplotlib.py | 1 | 81| 610 | 22222 | 180680 | 
| 48 | 24 lib/matplotlib/testing/decorators.py | 91 | 116| 214 | 22436 | 180680 | 
| 49 | 25 tutorials/intermediate/artists.py | 118 | 333| 2328 | 24764 | 188291 | 
| 50 | 25 lib/matplotlib/pyplot.py | 3010 | 3106| 792 | 25556 | 188291 | 
| 51 | **26 lib/matplotlib/tri/triangulation.py** | 164 | 185| 199 | 25755 | 190444 | 
| 52 | 27 examples/shapes_and_collections/path_patch.py | 1 | 53| 399 | 26154 | 190843 | 
| 53 | 28 lib/matplotlib/_api/deprecation.py | 223 | 254| 311 | 26465 | 195160 | 
| 54 | 28 lib/matplotlib/_api/deprecation.py | 24 | 47| 238 | 26703 | 195160 | 
| 55 | 28 lib/matplotlib/axes/_base.py | 355 | 405| 512 | 27215 | 195160 | 
| 56 | 28 lib/matplotlib/__init__.py | 1194 | 1231| 170 | 27385 | 195160 | 


## Missing Patch Files

 * 1: lib/matplotlib/tri/__init__.py
 * 2: lib/matplotlib/tri/_triangulation.py
 * 3: lib/matplotlib/tri/_tricontour.py
 * 4: lib/matplotlib/tri/_trifinder.py
 * 5: lib/matplotlib/tri/_triinterpolate.py
 * 6: lib/matplotlib/tri/_tripcolor.py
 * 7: lib/matplotlib/tri/_triplot.py
 * 8: lib/matplotlib/tri/_trirefine.py
 * 9: lib/matplotlib/tri/_tritools.py
 * 10: lib/matplotlib/tri/triangulation.py
 * 11: lib/matplotlib/tri/tricontour.py
 * 12: lib/matplotlib/tri/trifinder.py
 * 13: lib/matplotlib/tri/triinterpolate.py
 * 14: lib/matplotlib/tri/tripcolor.py
 * 15: lib/matplotlib/tri/triplot.py
 * 16: lib/matplotlib/tri/trirefine.py
 * 17: lib/matplotlib/tri/tritools.py
 * 18: lib/mpl_toolkits/mplot3d/axes3d.py

### Hint

```
I agree with renaming all the `tri/foo.py` modules to `tri/_foo.py` (`tri/__init__.py` already reexports everything anyways).  I'm not sure it's possible to have a proper deprecation for `from matplotlib.tri.tripcolor import tripcolor` (without ridiculous hacks)?
```

## Patch

```diff
diff --git a/lib/matplotlib/tri/__init__.py b/lib/matplotlib/tri/__init__.py
--- a/lib/matplotlib/tri/__init__.py
+++ b/lib/matplotlib/tri/__init__.py
@@ -2,15 +2,15 @@
 Unstructured triangular grid functions.
 """
 
-from .triangulation import Triangulation
-from .tricontour import TriContourSet, tricontour, tricontourf
-from .trifinder import TriFinder, TrapezoidMapTriFinder
-from .triinterpolate import (TriInterpolator, LinearTriInterpolator,
-                             CubicTriInterpolator)
-from .tripcolor import tripcolor
-from .triplot import triplot
-from .trirefine import TriRefiner, UniformTriRefiner
-from .tritools import TriAnalyzer
+from ._triangulation import Triangulation
+from ._tricontour import TriContourSet, tricontour, tricontourf
+from ._trifinder import TriFinder, TrapezoidMapTriFinder
+from ._triinterpolate import (TriInterpolator, LinearTriInterpolator,
+                              CubicTriInterpolator)
+from ._tripcolor import tripcolor
+from ._triplot import triplot
+from ._trirefine import TriRefiner, UniformTriRefiner
+from ._tritools import TriAnalyzer
 
 
 __all__ = ["Triangulation",
diff --git a/lib/matplotlib/tri/_triangulation.py b/lib/matplotlib/tri/_triangulation.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_triangulation.py
@@ -0,0 +1,240 @@
+import numpy as np
+
+from matplotlib import _api
+
+
+class Triangulation:
+    """
+    An unstructured triangular grid consisting of npoints points and
+    ntri triangles.  The triangles can either be specified by the user
+    or automatically generated using a Delaunay triangulation.
+
+    Parameters
+    ----------
+    x, y : (npoints,) array-like
+        Coordinates of grid points.
+    triangles : (ntri, 3) array-like of int, optional
+        For each triangle, the indices of the three points that make
+        up the triangle, ordered in an anticlockwise manner.  If not
+        specified, the Delaunay triangulation is calculated.
+    mask : (ntri,) array-like of bool, optional
+        Which triangles are masked out.
+
+    Attributes
+    ----------
+    triangles : (ntri, 3) array of int
+        For each triangle, the indices of the three points that make
+        up the triangle, ordered in an anticlockwise manner. If you want to
+        take the *mask* into account, use `get_masked_triangles` instead.
+    mask : (ntri, 3) array of bool
+        Masked out triangles.
+    is_delaunay : bool
+        Whether the Triangulation is a calculated Delaunay
+        triangulation (where *triangles* was not specified) or not.
+
+    Notes
+    -----
+    For a Triangulation to be valid it must not have duplicate points,
+    triangles formed from colinear points, or overlapping triangles.
+    """
+    def __init__(self, x, y, triangles=None, mask=None):
+        from matplotlib import _qhull
+
+        self.x = np.asarray(x, dtype=np.float64)
+        self.y = np.asarray(y, dtype=np.float64)
+        if self.x.shape != self.y.shape or self.x.ndim != 1:
+            raise ValueError("x and y must be equal-length 1D arrays, but "
+                             f"found shapes {self.x.shape!r} and "
+                             f"{self.y.shape!r}")
+
+        self.mask = None
+        self._edges = None
+        self._neighbors = None
+        self.is_delaunay = False
+
+        if triangles is None:
+            # No triangulation specified, so use matplotlib._qhull to obtain
+            # Delaunay triangulation.
+            self.triangles, self._neighbors = _qhull.delaunay(x, y)
+            self.is_delaunay = True
+        else:
+            # Triangulation specified. Copy, since we may correct triangle
+            # orientation.
+            try:
+                self.triangles = np.array(triangles, dtype=np.int32, order='C')
+            except ValueError as e:
+                raise ValueError('triangles must be a (N, 3) int array, not '
+                                 f'{triangles!r}') from e
+            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
+                raise ValueError(
+                    'triangles must be a (N, 3) int array, but found shape '
+                    f'{self.triangles.shape!r}')
+            if self.triangles.max() >= len(self.x):
+                raise ValueError(
+                    'triangles are indices into the points and must be in the '
+                    f'range 0 <= i < {len(self.x)} but found value '
+                    f'{self.triangles.max()}')
+            if self.triangles.min() < 0:
+                raise ValueError(
+                    'triangles are indices into the points and must be in the '
+                    f'range 0 <= i < {len(self.x)} but found value '
+                    f'{self.triangles.min()}')
+
+        # Underlying C++ object is not created until first needed.
+        self._cpp_triangulation = None
+
+        # Default TriFinder not created until needed.
+        self._trifinder = None
+
+        self.set_mask(mask)
+
+    def calculate_plane_coefficients(self, z):
+        """
+        Calculate plane equation coefficients for all unmasked triangles from
+        the point (x, y) coordinates and specified z-array of shape (npoints).
+        The returned array has shape (npoints, 3) and allows z-value at (x, y)
+        position in triangle tri to be calculated using
+        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
+        """
+        return self.get_cpp_triangulation().calculate_plane_coefficients(z)
+
+    @property
+    def edges(self):
+        """
+        Return integer array of shape (nedges, 2) containing all edges of
+        non-masked triangles.
+
+        Each row defines an edge by its start point index and end point
+        index.  Each edge appears only once, i.e. for an edge between points
+        *i*  and *j*, there will only be either *(i, j)* or *(j, i)*.
+        """
+        if self._edges is None:
+            self._edges = self.get_cpp_triangulation().get_edges()
+        return self._edges
+
+    def get_cpp_triangulation(self):
+        """
+        Return the underlying C++ Triangulation object, creating it
+        if necessary.
+        """
+        from matplotlib import _tri
+        if self._cpp_triangulation is None:
+            self._cpp_triangulation = _tri.Triangulation(
+                self.x, self.y, self.triangles, self.mask, self._edges,
+                self._neighbors, not self.is_delaunay)
+        return self._cpp_triangulation
+
+    def get_masked_triangles(self):
+        """
+        Return an array of triangles taking the mask into account.
+        """
+        if self.mask is not None:
+            return self.triangles[~self.mask]
+        else:
+            return self.triangles
+
+    @staticmethod
+    def get_from_args_and_kwargs(*args, **kwargs):
+        """
+        Return a Triangulation object from the args and kwargs, and
+        the remaining args and kwargs with the consumed values removed.
+
+        There are two alternatives: either the first argument is a
+        Triangulation object, in which case it is returned, or the args
+        and kwargs are sufficient to create a new Triangulation to
+        return.  In the latter case, see Triangulation.__init__ for
+        the possible args and kwargs.
+        """
+        if isinstance(args[0], Triangulation):
+            triangulation, *args = args
+            if 'triangles' in kwargs:
+                _api.warn_external(
+                    "Passing the keyword 'triangles' has no effect when also "
+                    "passing a Triangulation")
+            if 'mask' in kwargs:
+                _api.warn_external(
+                    "Passing the keyword 'mask' has no effect when also "
+                    "passing a Triangulation")
+        else:
+            x, y, triangles, mask, args, kwargs = \
+                Triangulation._extract_triangulation_params(args, kwargs)
+            triangulation = Triangulation(x, y, triangles, mask)
+        return triangulation, args, kwargs
+
+    @staticmethod
+    def _extract_triangulation_params(args, kwargs):
+        x, y, *args = args
+        # Check triangles in kwargs then args.
+        triangles = kwargs.pop('triangles', None)
+        from_args = False
+        if triangles is None and args:
+            triangles = args[0]
+            from_args = True
+        if triangles is not None:
+            try:
+                triangles = np.asarray(triangles, dtype=np.int32)
+            except ValueError:
+                triangles = None
+        if triangles is not None and (triangles.ndim != 2 or
+                                      triangles.shape[1] != 3):
+            triangles = None
+        if triangles is not None and from_args:
+            args = args[1:]  # Consumed first item in args.
+        # Check for mask in kwargs.
+        mask = kwargs.pop('mask', None)
+        return x, y, triangles, mask, args, kwargs
+
+    def get_trifinder(self):
+        """
+        Return the default `matplotlib.tri.TriFinder` of this
+        triangulation, creating it if necessary.  This allows the same
+        TriFinder object to be easily shared.
+        """
+        if self._trifinder is None:
+            # Default TriFinder class.
+            from matplotlib.tri._trifinder import TrapezoidMapTriFinder
+            self._trifinder = TrapezoidMapTriFinder(self)
+        return self._trifinder
+
+    @property
+    def neighbors(self):
+        """
+        Return integer array of shape (ntri, 3) containing neighbor triangles.
+
+        For each triangle, the indices of the three triangles that
+        share the same edges, or -1 if there is no such neighboring
+        triangle.  ``neighbors[i, j]`` is the triangle that is the neighbor
+        to the edge from point index ``triangles[i, j]`` to point index
+        ``triangles[i, (j+1)%3]``.
+        """
+        if self._neighbors is None:
+            self._neighbors = self.get_cpp_triangulation().get_neighbors()
+        return self._neighbors
+
+    def set_mask(self, mask):
+        """
+        Set or clear the mask array.
+
+        Parameters
+        ----------
+        mask : None or bool array of length ntri
+        """
+        if mask is None:
+            self.mask = None
+        else:
+            self.mask = np.asarray(mask, dtype=bool)
+            if self.mask.shape != (self.triangles.shape[0],):
+                raise ValueError('mask array must have same length as '
+                                 'triangles array')
+
+        # Set mask in C++ Triangulation.
+        if self._cpp_triangulation is not None:
+            self._cpp_triangulation.set_mask(self.mask)
+
+        # Clear derived fields so they are recalculated when needed.
+        self._edges = None
+        self._neighbors = None
+
+        # Recalculate TriFinder if it exists.
+        if self._trifinder is not None:
+            self._trifinder._initialize()
diff --git a/lib/matplotlib/tri/_tricontour.py b/lib/matplotlib/tri/_tricontour.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_tricontour.py
@@ -0,0 +1,271 @@
+import numpy as np
+
+from matplotlib import _docstring
+from matplotlib.contour import ContourSet
+from matplotlib.tri._triangulation import Triangulation
+
+
+@_docstring.dedent_interpd
+class TriContourSet(ContourSet):
+    """
+    Create and store a set of contour lines or filled regions for
+    a triangular grid.
+
+    This class is typically not instantiated directly by the user but by
+    `~.Axes.tricontour` and `~.Axes.tricontourf`.
+
+    %(contour_set_attributes)s
+    """
+    def __init__(self, ax, *args, **kwargs):
+        """
+        Draw triangular grid contour lines or filled regions,
+        depending on whether keyword arg *filled* is False
+        (default) or True.
+
+        The first argument of the initializer must be an `~.axes.Axes`
+        object.  The remaining arguments and keyword arguments
+        are described in the docstring of `~.Axes.tricontour`.
+        """
+        super().__init__(ax, *args, **kwargs)
+
+    def _process_args(self, *args, **kwargs):
+        """
+        Process args and kwargs.
+        """
+        if isinstance(args[0], TriContourSet):
+            C = args[0]._contour_generator
+            if self.levels is None:
+                self.levels = args[0].levels
+            self.zmin = args[0].zmin
+            self.zmax = args[0].zmax
+            self._mins = args[0]._mins
+            self._maxs = args[0]._maxs
+        else:
+            from matplotlib import _tri
+            tri, z = self._contour_args(args, kwargs)
+            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
+            self._mins = [tri.x.min(), tri.y.min()]
+            self._maxs = [tri.x.max(), tri.y.max()]
+
+        self._contour_generator = C
+        return kwargs
+
+    def _contour_args(self, args, kwargs):
+        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
+                                                                   **kwargs)
+        z = np.ma.asarray(args[0])
+        if z.shape != tri.x.shape:
+            raise ValueError('z array must have same length as triangulation x'
+                             ' and y arrays')
+
+        # z values must be finite, only need to check points that are included
+        # in the triangulation.
+        z_check = z[np.unique(tri.get_masked_triangles())]
+        if np.ma.is_masked(z_check):
+            raise ValueError('z must not contain masked points within the '
+                             'triangulation')
+        if not np.isfinite(z_check).all():
+            raise ValueError('z array must not contain non-finite values '
+                             'within the triangulation')
+
+        z = np.ma.masked_invalid(z, copy=False)
+        self.zmax = float(z_check.max())
+        self.zmin = float(z_check.min())
+        if self.logscale and self.zmin <= 0:
+            func = 'contourf' if self.filled else 'contour'
+            raise ValueError(f'Cannot {func} log of negative values.')
+        self._process_contour_level_args(args[1:])
+        return (tri, z)
+
+
+_docstring.interpd.update(_tricontour_doc="""
+Draw contour %%(type)s on an unstructured triangular grid.
+
+Call signatures::
+
+    %%(func)s(triangulation, z, [levels], ...)
+    %%(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)
+
+The triangular grid can be specified either by passing a `.Triangulation`
+object as the first parameter, or by passing the points *x*, *y* and
+optionally the *triangles* and a *mask*. See `.Triangulation` for an
+explanation of these parameters. If neither of *triangulation* or
+*triangles* are given, the triangulation is calculated on the fly.
+
+It is possible to pass *triangles* positionally, i.e.
+``%%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more
+clarity, pass *triangles* via keyword argument.
+
+Parameters
+----------
+triangulation : `.Triangulation`, optional
+    An already created triangular grid.
+
+x, y, triangles, mask
+    Parameters defining the triangular grid. See `.Triangulation`.
+    This is mutually exclusive with specifying *triangulation*.
+
+z : array-like
+    The height values over which the contour is drawn.  Color-mapping is
+    controlled by *cmap*, *norm*, *vmin*, and *vmax*.
+
+    .. note::
+        All values in *z* must be finite. Hence, nan and inf values must
+        either be removed or `~.Triangulation.set_mask` be used.
+
+levels : int or array-like, optional
+    Determines the number and positions of the contour lines / regions.
+
+    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
+    automatically choose no more than *n+1* "nice" contour levels between
+    between minimum and maximum numeric values of *Z*.
+
+    If array-like, draw contour lines at the specified levels.  The values must
+    be in increasing order.
+
+Returns
+-------
+`~matplotlib.tri.TriContourSet`
+
+Other Parameters
+----------------
+colors : color string or sequence of colors, optional
+    The colors of the levels, i.e., the contour %%(type)s.
+
+    The sequence is cycled for the levels in ascending order. If the sequence
+    is shorter than the number of levels, it is repeated.
+
+    As a shortcut, single color strings may be used in place of one-element
+    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
+    same color. This shortcut does only work for color strings, not for other
+    ways of specifying colors.
+
+    By default (value *None*), the colormap specified by *cmap* will be used.
+
+alpha : float, default: 1
+    The alpha blending value, between 0 (transparent) and 1 (opaque).
+
+%(cmap_doc)s
+
+    This parameter is ignored if *colors* is set.
+
+%(norm_doc)s
+
+    This parameter is ignored if *colors* is set.
+
+%(vmin_vmax_doc)s
+
+    If *vmin* or *vmax* are not given, the default color scaling is based on
+    *levels*.
+
+    This parameter is ignored if *colors* is set.
+
+origin : {*None*, 'upper', 'lower', 'image'}, default: None
+    Determines the orientation and exact position of *z* by specifying the
+    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.
+
+    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.
+    - 'lower': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
+    - 'upper': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
+    - 'image': Use the value from :rc:`image.origin`.
+
+extent : (x0, x1, y0, y1), optional
+    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
+    gives the outer pixel boundaries. In this case, the position of z[0, 0] is
+    the center of the pixel, not a corner. If *origin* is *None*, then
+    (*x0*, *y0*) is the position of z[0, 0], and (*x1*, *y1*) is the position
+    of z[-1, -1].
+
+    This argument is ignored if *X* and *Y* are specified in the call to
+    contour.
+
+locator : ticker.Locator subclass, optional
+    The locator is used to determine the contour levels if they are not given
+    explicitly via *levels*.
+    Defaults to `~.ticker.MaxNLocator`.
+
+extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
+    Determines the ``%%(func)s``-coloring of values that are outside the
+    *levels* range.
+
+    If 'neither', values outside the *levels* range are not colored.  If 'min',
+    'max' or 'both', color the values below, above or below and above the
+    *levels* range.
+
+    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
+    under/over values of the `.Colormap`. Note that most colormaps do not have
+    dedicated colors for these by default, so that the over and under values
+    are the edge values of the colormap.  You may want to set these values
+    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.
+
+    .. note::
+
+        An existing `.TriContourSet` does not get notified if properties of its
+        colormap are changed. Therefore, an explicit call to
+        `.ContourSet.changed()` is needed after modifying the colormap. The
+        explicit call can be left out, if a colorbar is assigned to the
+        `.TriContourSet` because it internally calls `.ContourSet.changed()`.
+
+xunits, yunits : registered units, optional
+    Override axis units by specifying an instance of a
+    :class:`matplotlib.units.ConversionInterface`.
+
+antialiased : bool, optional
+    Enable antialiasing, overriding the defaults.  For
+    filled contours, the default is *True*.  For line contours,
+    it is taken from :rc:`lines.antialiased`.""" % _docstring.interpd.params)
+
+
+@_docstring.Substitution(func='tricontour', type='lines')
+@_docstring.dedent_interpd
+def tricontour(ax, *args, **kwargs):
+    """
+    %(_tricontour_doc)s
+
+    linewidths : float or array-like, default: :rc:`contour.linewidth`
+        The line width of the contour lines.
+
+        If a number, all levels will be plotted with this linewidth.
+
+        If a sequence, the levels in ascending order will be plotted with
+        the linewidths in the order specified.
+
+        If None, this falls back to :rc:`lines.linewidth`.
+
+    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
+        If *linestyles* is *None*, the default is 'solid' unless the lines are
+        monochrome.  In that case, negative contours will take their linestyle
+        from :rc:`contour.negative_linestyle` setting.
+
+        *linestyles* can also be an iterable of the above strings specifying a
+        set of linestyles to be used. If this iterable is shorter than the
+        number of contour levels it will be repeated as necessary.
+    """
+    kwargs['filled'] = False
+    return TriContourSet(ax, *args, **kwargs)
+
+
+@_docstring.Substitution(func='tricontourf', type='regions')
+@_docstring.dedent_interpd
+def tricontourf(ax, *args, **kwargs):
+    """
+    %(_tricontour_doc)s
+
+    hatches : list[str], optional
+        A list of cross hatch patterns to use on the filled areas.
+        If None, no hatching will be added to the contour.
+        Hatching is supported in the PostScript, PDF, SVG and Agg
+        backends only.
+
+    Notes
+    -----
+    `.tricontourf` fills intervals that are closed at the top; that is, for
+    boundaries *z1* and *z2*, the filled region is::
+
+        z1 < Z <= z2
+
+    except for the lowest interval, which is closed on both sides (i.e. it
+    includes the lowest value).
+    """
+    kwargs['filled'] = True
+    return TriContourSet(ax, *args, **kwargs)
diff --git a/lib/matplotlib/tri/_trifinder.py b/lib/matplotlib/tri/_trifinder.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_trifinder.py
@@ -0,0 +1,93 @@
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri import Triangulation
+
+
+class TriFinder:
+    """
+    Abstract base class for classes used to find the triangles of a
+    Triangulation in which (x, y) points lie.
+
+    Rather than instantiate an object of a class derived from TriFinder, it is
+    usually better to use the function `.Triangulation.get_trifinder`.
+
+    Derived classes implement __call__(x, y) where x and y are array-like point
+    coordinates of the same shape.
+    """
+
+    def __init__(self, triangulation):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+
+class TrapezoidMapTriFinder(TriFinder):
+    """
+    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
+    map algorithm from the book "Computational Geometry, Algorithms and
+    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
+    and O. Schwarzkopf.
+
+    The triangulation must be valid, i.e. it must not have duplicate points,
+    triangles formed from colinear points, or overlapping triangles.  The
+    algorithm has some tolerance to triangles formed from colinear points, but
+    this should not be relied upon.
+    """
+
+    def __init__(self, triangulation):
+        from matplotlib import _tri
+        super().__init__(triangulation)
+        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
+            triangulation.get_cpp_triangulation())
+        self._initialize()
+
+    def __call__(self, x, y):
+        """
+        Return an array containing the indices of the triangles in which the
+        specified *x*, *y* points lie, or -1 for points that do not lie within
+        a triangle.
+
+        *x*, *y* are array-like x and y coordinates of the same shape and any
+        number of dimensions.
+
+        Returns integer array with the same shape and *x* and *y*.
+        """
+        x = np.asarray(x, dtype=np.float64)
+        y = np.asarray(y, dtype=np.float64)
+        if x.shape != y.shape:
+            raise ValueError("x and y must be array-like with the same shape")
+
+        # C++ does the heavy lifting, and expects 1D arrays.
+        indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
+                   .reshape(x.shape))
+        return indices
+
+    def _get_tree_stats(self):
+        """
+        Return a python list containing the statistics about the node tree:
+            0: number of nodes (tree size)
+            1: number of unique nodes
+            2: number of trapezoids (tree leaf nodes)
+            3: number of unique trapezoids
+            4: maximum parent count (max number of times a node is repeated in
+                   tree)
+            5: maximum depth of tree (one more than the maximum number of
+                   comparisons needed to search through the tree)
+            6: mean of all trapezoid depths (one more than the average number
+                   of comparisons needed to search through the tree)
+        """
+        return self._cpp_trifinder.get_tree_stats()
+
+    def _initialize(self):
+        """
+        Initialize the underlying C++ object.  Can be called multiple times if,
+        for example, the triangulation is modified.
+        """
+        self._cpp_trifinder.initialize()
+
+    def _print_tree(self):
+        """
+        Print a text representation of the node tree, which is useful for
+        debugging purposes.
+        """
+        self._cpp_trifinder.print_tree()
diff --git a/lib/matplotlib/tri/_triinterpolate.py b/lib/matplotlib/tri/_triinterpolate.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_triinterpolate.py
@@ -0,0 +1,1574 @@
+"""
+Interpolation inside triangular grids.
+"""
+
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri import Triangulation
+from matplotlib.tri._trifinder import TriFinder
+from matplotlib.tri._tritools import TriAnalyzer
+
+__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')
+
+
+class TriInterpolator:
+    """
+    Abstract base class for classes used to interpolate on a triangular grid.
+
+    Derived classes implement the following methods:
+
+    - ``__call__(x, y)``,
+      where x, y are array-like point coordinates of the same shape, and
+      that returns a masked array of the same shape containing the
+      interpolated z-values.
+
+    - ``gradient(x, y)``,
+      where x, y are array-like point coordinates of the same
+      shape, and that returns a list of 2 masked arrays of the same shape
+      containing the 2 derivatives of the interpolator (derivatives of
+      interpolated z values with respect to x and y).
+    """
+
+    def __init__(self, triangulation, z, trifinder=None):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+        self._z = np.asarray(z)
+        if self._z.shape != self._triangulation.x.shape:
+            raise ValueError("z array must have same length as triangulation x"
+                             " and y arrays")
+
+        _api.check_isinstance((TriFinder, None), trifinder=trifinder)
+        self._trifinder = trifinder or self._triangulation.get_trifinder()
+
+        # Default scaling factors : 1.0 (= no scaling)
+        # Scaling may be used for interpolations for which the order of
+        # magnitude of x, y has an impact on the interpolant definition.
+        # Please refer to :meth:`_interpolate_multikeys` for details.
+        self._unit_x = 1.0
+        self._unit_y = 1.0
+
+        # Default triangle renumbering: None (= no renumbering)
+        # Renumbering may be used to avoid unnecessary computations
+        # if complex calculations are done inside the Interpolator.
+        # Please refer to :meth:`_interpolate_multikeys` for details.
+        self._tri_renum = None
+
+    # __call__ and gradient docstrings are shared by all subclasses
+    # (except, if needed, relevant additions).
+    # However these methods are only implemented in subclasses to avoid
+    # confusion in the documentation.
+    _docstring__call__ = """
+        Returns a masked array containing interpolated values at the specified
+        (x, y) points.
+
+        Parameters
+        ----------
+        x, y : array-like
+            x and y coordinates of the same shape and any number of
+            dimensions.
+
+        Returns
+        -------
+        np.ma.array
+            Masked array of the same shape as *x* and *y*; values corresponding
+            to (*x*, *y*) points outside of the triangulation are masked out.
+
+        """
+
+    _docstringgradient = r"""
+        Returns a list of 2 masked arrays containing interpolated derivatives
+        at the specified (x, y) points.
+
+        Parameters
+        ----------
+        x, y : array-like
+            x and y coordinates of the same shape and any number of
+            dimensions.
+
+        Returns
+        -------
+        dzdx, dzdy : np.ma.array
+            2 masked arrays of the same shape as *x* and *y*; values
+            corresponding to (x, y) points outside of the triangulation
+            are masked out.
+            The first returned array contains the values of
+            :math:`\frac{\partial z}{\partial x}` and the second those of
+            :math:`\frac{\partial z}{\partial y}`.
+
+        """
+
+    def _interpolate_multikeys(self, x, y, tri_index=None,
+                               return_keys=('z',)):
+        """
+        Versatile (private) method defined for all TriInterpolators.
+
+        :meth:`_interpolate_multikeys` is a wrapper around method
+        :meth:`_interpolate_single_key` (to be defined in the child
+        subclasses).
+        :meth:`_interpolate_single_key actually performs the interpolation,
+        but only for 1-dimensional inputs and at valid locations (inside
+        unmasked triangles of the triangulation).
+
+        The purpose of :meth:`_interpolate_multikeys` is to implement the
+        following common tasks needed in all subclasses implementations:
+
+        - calculation of containing triangles
+        - dealing with more than one interpolation request at the same
+          location (e.g., if the 2 derivatives are requested, it is
+          unnecessary to compute the containing triangles twice)
+        - scaling according to self._unit_x, self._unit_y
+        - dealing with points outside of the grid (with fill value np.nan)
+        - dealing with multi-dimensional *x*, *y* arrays: flattening for
+          :meth:`_interpolate_params` call and final reshaping.
+
+        (Note that np.vectorize could do most of those things very well for
+        you, but it does it by function evaluations over successive tuples of
+        the input arrays. Therefore, this tends to be more time consuming than
+        using optimized numpy functions - e.g., np.dot - which can be used
+        easily on the flattened inputs, in the child-subclass methods
+        :meth:`_interpolate_single_key`.)
+
+        It is guaranteed that the calls to :meth:`_interpolate_single_key`
+        will be done with flattened (1-d) array-like input parameters *x*, *y*
+        and with flattened, valid `tri_index` arrays (no -1 index allowed).
+
+        Parameters
+        ----------
+        x, y : array-like
+            x and y coordinates where interpolated values are requested.
+        tri_index : array-like of int, optional
+            Array of the containing triangle indices, same shape as
+            *x* and *y*. Defaults to None. If None, these indices
+            will be computed by a TriFinder instance.
+            (Note: For point outside the grid, tri_index[ipt] shall be -1).
+        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
+            Defines the interpolation arrays to return, and in which order.
+
+        Returns
+        -------
+        list of arrays
+            Each array-like contains the expected interpolated values in the
+            order defined by *return_keys* parameter.
+        """
+        # Flattening and rescaling inputs arrays x, y
+        # (initial shape is stored for output)
+        x = np.asarray(x, dtype=np.float64)
+        y = np.asarray(y, dtype=np.float64)
+        sh_ret = x.shape
+        if x.shape != y.shape:
+            raise ValueError("x and y shall have same shapes."
+                             " Given: {0} and {1}".format(x.shape, y.shape))
+        x = np.ravel(x)
+        y = np.ravel(y)
+        x_scaled = x/self._unit_x
+        y_scaled = y/self._unit_y
+        size_ret = np.size(x_scaled)
+
+        # Computes & ravels the element indexes, extract the valid ones.
+        if tri_index is None:
+            tri_index = self._trifinder(x, y)
+        else:
+            if tri_index.shape != sh_ret:
+                raise ValueError(
+                    "tri_index array is provided and shall"
+                    " have same shape as x and y. Given: "
+                    "{0} and {1}".format(tri_index.shape, sh_ret))
+            tri_index = np.ravel(tri_index)
+
+        mask_in = (tri_index != -1)
+        if self._tri_renum is None:
+            valid_tri_index = tri_index[mask_in]
+        else:
+            valid_tri_index = self._tri_renum[tri_index[mask_in]]
+        valid_x = x_scaled[mask_in]
+        valid_y = y_scaled[mask_in]
+
+        ret = []
+        for return_key in return_keys:
+            # Find the return index associated with the key.
+            try:
+                return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
+            except KeyError as err:
+                raise ValueError("return_keys items shall take values in"
+                                 " {'z', 'dzdx', 'dzdy'}") from err
+
+            # Sets the scale factor for f & df components
+            scale = [1., 1./self._unit_x, 1./self._unit_y][return_index]
+
+            # Computes the interpolation
+            ret_loc = np.empty(size_ret, dtype=np.float64)
+            ret_loc[~mask_in] = np.nan
+            ret_loc[mask_in] = self._interpolate_single_key(
+                return_key, valid_tri_index, valid_x, valid_y) * scale
+            ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]
+
+        return ret
+
+    def _interpolate_single_key(self, return_key, tri_index, x, y):
+        """
+        Interpolate at points belonging to the triangulation
+        (inside an unmasked triangles).
+
+        Parameters
+        ----------
+        return_key : {'z', 'dzdx', 'dzdy'}
+            The requested values (z or its derivatives).
+        tri_index : 1D int array
+            Valid triangle index (cannot be -1).
+        x, y : 1D arrays, same shape as `tri_index`
+            Valid locations where interpolation is requested.
+
+        Returns
+        -------
+        1-d array
+            Returned array of the same size as *tri_index*
+        """
+        raise NotImplementedError("TriInterpolator subclasses" +
+                                  "should implement _interpolate_single_key!")
+
+
+class LinearTriInterpolator(TriInterpolator):
+    """
+    Linear interpolator on a triangular grid.
+
+    Each triangle is represented by a plane so that an interpolated value at
+    point (x, y) lies on the plane of the triangle containing (x, y).
+    Interpolated values are therefore continuous across the triangulation, but
+    their first derivatives are discontinuous at edges between triangles.
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The triangulation to interpolate over.
+    z : (npoints,) array-like
+        Array of values, defined at grid points, to interpolate between.
+    trifinder : `~matplotlib.tri.TriFinder`, optional
+        If this is not specified, the Triangulation's default TriFinder will
+        be used by calling `.Triangulation.get_trifinder`.
+
+    Methods
+    -------
+    `__call__` (x, y) : Returns interpolated values at (x, y) points.
+    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
+
+    """
+    def __init__(self, triangulation, z, trifinder=None):
+        super().__init__(triangulation, z, trifinder)
+
+        # Store plane coefficients for fast interpolation calculations.
+        self._plane_coefficients = \
+            self._triangulation.calculate_plane_coefficients(self._z)
+
+    def __call__(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('z',))[0]
+    __call__.__doc__ = TriInterpolator._docstring__call__
+
+    def gradient(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('dzdx', 'dzdy'))
+    gradient.__doc__ = TriInterpolator._docstringgradient
+
+    def _interpolate_single_key(self, return_key, tri_index, x, y):
+        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
+        if return_key == 'z':
+            return (self._plane_coefficients[tri_index, 0]*x +
+                    self._plane_coefficients[tri_index, 1]*y +
+                    self._plane_coefficients[tri_index, 2])
+        elif return_key == 'dzdx':
+            return self._plane_coefficients[tri_index, 0]
+        else:  # 'dzdy'
+            return self._plane_coefficients[tri_index, 1]
+
+
+class CubicTriInterpolator(TriInterpolator):
+    r"""
+    Cubic interpolator on a triangular grid.
+
+    In one-dimension - on a segment - a cubic interpolating function is
+    defined by the values of the function and its derivative at both ends.
+    This is almost the same in 2D inside a triangle, except that the values
+    of the function and its 2 derivatives have to be defined at each triangle
+    node.
+
+    The CubicTriInterpolator takes the value of the function at each node -
+    provided by the user - and internally computes the value of the
+    derivatives, resulting in a smooth interpolation.
+    (As a special feature, the user can also impose the value of the
+    derivatives at each node, but this is not supposed to be the common
+    usage.)
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The triangulation to interpolate over.
+    z : (npoints,) array-like
+        Array of values, defined at grid points, to interpolate between.
+    kind : {'min_E', 'geom', 'user'}, optional
+        Choice of the smoothing algorithm, in order to compute
+        the interpolant derivatives (defaults to 'min_E'):
+
+        - if 'min_E': (default) The derivatives at each node is computed
+          to minimize a bending energy.
+        - if 'geom': The derivatives at each node is computed as a
+          weighted average of relevant triangle normals. To be used for
+          speed optimization (large grids).
+        - if 'user': The user provides the argument *dz*, no computation
+          is hence needed.
+
+    trifinder : `~matplotlib.tri.TriFinder`, optional
+        If not specified, the Triangulation's default TriFinder will
+        be used by calling `.Triangulation.get_trifinder`.
+    dz : tuple of array-likes (dzdx, dzdy), optional
+        Used only if  *kind* ='user'. In this case *dz* must be provided as
+        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
+        are the interpolant first derivatives at the *triangulation* points.
+
+    Methods
+    -------
+    `__call__` (x, y) : Returns interpolated values at (x, y) points.
+    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
+
+    Notes
+    -----
+    This note is a bit technical and details how the cubic interpolation is
+    computed.
+
+    The interpolation is based on a Clough-Tocher subdivision scheme of
+    the *triangulation* mesh (to make it clearer, each triangle of the
+    grid will be divided in 3 child-triangles, and on each child triangle
+    the interpolated function is a cubic polynomial of the 2 coordinates).
+    This technique originates from FEM (Finite Element Method) analysis;
+    the element used is a reduced Hsieh-Clough-Tocher (HCT)
+    element. Its shape functions are described in [1]_.
+    The assembled function is guaranteed to be C1-smooth, i.e. it is
+    continuous and its first derivatives are also continuous (this
+    is easy to show inside the triangles but is also true when crossing the
+    edges).
+
+    In the default case (*kind* ='min_E'), the interpolant minimizes a
+    curvature energy on the functional space generated by the HCT element
+    shape functions - with imposed values but arbitrary derivatives at each
+    node. The minimized functional is the integral of the so-called total
+    curvature (implementation based on an algorithm from [2]_ - PCG sparse
+    solver):
+
+        .. math::
+
+            E(z) = \frac{1}{2} \int_{\Omega} \left(
+                \left( \frac{\partial^2{z}}{\partial{x}^2} \right)^2 +
+                \left( \frac{\partial^2{z}}{\partial{y}^2} \right)^2 +
+                2\left( \frac{\partial^2{z}}{\partial{y}\partial{x}} \right)^2
+            \right) dx\,dy
+
+    If the case *kind* ='geom' is chosen by the user, a simple geometric
+    approximation is used (weighted average of the triangle normal
+    vectors), which could improve speed on very large grids.
+
+    References
+    ----------
+    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
+        Hsieh-Clough-Tocher triangles, complete or reduced.",
+        International Journal for Numerical Methods in Engineering,
+        17(5):784 - 789. 2.01.
+    .. [2] C.T. Kelley, "Iterative Methods for Optimization".
+
+    """
+    def __init__(self, triangulation, z, kind='min_E', trifinder=None,
+                 dz=None):
+        super().__init__(triangulation, z, trifinder)
+
+        # Loads the underlying c++ _triangulation.
+        # (During loading, reordering of triangulation._triangles may occur so
+        # that all final triangles are now anti-clockwise)
+        self._triangulation.get_cpp_triangulation()
+
+        # To build the stiffness matrix and avoid zero-energy spurious modes
+        # we will only store internally the valid (unmasked) triangles and
+        # the necessary (used) points coordinates.
+        # 2 renumbering tables need to be computed and stored:
+        #  - a triangle renum table in order to translate the result from a
+        #    TriFinder instance into the internal stored triangle number.
+        #  - a node renum table to overwrite the self._z values into the new
+        #    (used) node numbering.
+        tri_analyzer = TriAnalyzer(self._triangulation)
+        (compressed_triangles, compressed_x, compressed_y, tri_renum,
+         node_renum) = tri_analyzer._get_compressed_triangulation()
+        self._triangles = compressed_triangles
+        self._tri_renum = tri_renum
+        # Taking into account the node renumbering in self._z:
+        valid_node = (node_renum != -1)
+        self._z[node_renum[valid_node]] = self._z[valid_node]
+
+        # Computing scale factors
+        self._unit_x = np.ptp(compressed_x)
+        self._unit_y = np.ptp(compressed_y)
+        self._pts = np.column_stack([compressed_x / self._unit_x,
+                                     compressed_y / self._unit_y])
+        # Computing triangle points
+        self._tris_pts = self._pts[self._triangles]
+        # Computing eccentricities
+        self._eccs = self._compute_tri_eccentricities(self._tris_pts)
+        # Computing dof estimations for HCT triangle shape function
+        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
+        self._dof = self._compute_dof(kind, dz=dz)
+        # Loading HCT element
+        self._ReferenceElement = _ReducedHCT_Element()
+
+    def __call__(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('z',))[0]
+    __call__.__doc__ = TriInterpolator._docstring__call__
+
+    def gradient(self, x, y):
+        return self._interpolate_multikeys(x, y, tri_index=None,
+                                           return_keys=('dzdx', 'dzdy'))
+    gradient.__doc__ = TriInterpolator._docstringgradient
+
+    def _interpolate_single_key(self, return_key, tri_index, x, y):
+        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
+        tris_pts = self._tris_pts[tri_index]
+        alpha = self._get_alpha_vec(x, y, tris_pts)
+        ecc = self._eccs[tri_index]
+        dof = np.expand_dims(self._dof[tri_index], axis=1)
+        if return_key == 'z':
+            return self._ReferenceElement.get_function_values(
+                alpha, ecc, dof)
+        else:  # 'dzdx', 'dzdy'
+            J = self._get_jacobian(tris_pts)
+            dzdx = self._ReferenceElement.get_function_derivatives(
+                alpha, J, ecc, dof)
+            if return_key == 'dzdx':
+                return dzdx[:, 0, 0]
+            else:
+                return dzdx[:, 1, 0]
+
+    def _compute_dof(self, kind, dz=None):
+        """
+        Compute and return nodal dofs according to kind.
+
+        Parameters
+        ----------
+        kind : {'min_E', 'geom', 'user'}
+            Choice of the _DOF_estimator subclass to estimate the gradient.
+        dz : tuple of array-likes (dzdx, dzdy), optional
+            Used only if *kind*=user; in this case passed to the
+            :class:`_DOF_estimator_user`.
+
+        Returns
+        -------
+        array-like, shape (npts, 2)
+            Estimation of the gradient at triangulation nodes (stored as
+            degree of freedoms of reduced-HCT triangle elements).
+        """
+        if kind == 'user':
+            if dz is None:
+                raise ValueError("For a CubicTriInterpolator with "
+                                 "*kind*='user', a valid *dz* "
+                                 "argument is expected.")
+            TE = _DOF_estimator_user(self, dz=dz)
+        elif kind == 'geom':
+            TE = _DOF_estimator_geom(self)
+        else:  # 'min_E', checked in __init__
+            TE = _DOF_estimator_min_E(self)
+        return TE.compute_dof_from_df()
+
+    @staticmethod
+    def _get_alpha_vec(x, y, tris_pts):
+        """
+        Fast (vectorized) function to compute barycentric coordinates alpha.
+
+        Parameters
+        ----------
+        x, y : array-like of dim 1 (shape (nx,))
+            Coordinates of the points whose points barycentric coordinates are
+            requested.
+        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
+            Coordinates of the containing triangles apexes.
+
+        Returns
+        -------
+        array of dim 2 (shape (nx, 3))
+            Barycentric coordinates of the points inside the containing
+            triangles.
+        """
+        ndim = tris_pts.ndim-2
+
+        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
+        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
+        abT = np.stack([a, b], axis=-1)
+        ab = _transpose_vectorized(abT)
+        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]
+
+        metric = ab @ abT
+        # Here we try to deal with the colinear cases.
+        # metric_inv is in this case set to the Moore-Penrose pseudo-inverse
+        # meaning that we will still return a set of valid barycentric
+        # coordinates.
+        metric_inv = _pseudo_inv22sym_vectorized(metric)
+        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
+        ksi = metric_inv @ Covar
+        alpha = _to_matrix_vectorized([
+            [1-ksi[:, 0, 0]-ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
+        return alpha
+
+    @staticmethod
+    def _get_jacobian(tris_pts):
+        """
+        Fast (vectorized) function to compute triangle jacobian matrix.
+
+        Parameters
+        ----------
+        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
+            Coordinates of the containing triangles apexes.
+
+        Returns
+        -------
+        array of dim 3 (shape (nx, 2, 2))
+            Barycentric coordinates of the points inside the containing
+            triangles.
+            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
+            itri, so that the following (matrix) relationship holds:
+               [dz/dksi] = [J] x [dz/dx]
+            with x: global coordinates
+                 ksi: element parametric coordinates in triangle first apex
+                 local basis.
+        """
+        a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
+        b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
+        J = _to_matrix_vectorized([[a[:, 0], a[:, 1]],
+                                   [b[:, 0], b[:, 1]]])
+        return J
+
+    @staticmethod
+    def _compute_tri_eccentricities(tris_pts):
+        """
+        Compute triangle eccentricities.
+
+        Parameters
+        ----------
+        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
+            Coordinates of the triangles apexes.
+
+        Returns
+        -------
+        array like of dim 2 (shape: (nx, 3))
+            The so-called eccentricity parameters [1] needed for HCT triangular
+            element.
+        """
+        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
+        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
+        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
+        # Do not use np.squeeze, this is dangerous if only one triangle
+        # in the triangulation...
+        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
+        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
+        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
+        # Note that this line will raise a warning for dot_a, dot_b or dot_c
+        # zeros, but we choose not to support triangles with duplicate points.
+        return _to_matrix_vectorized([[(dot_c-dot_b) / dot_a],
+                                      [(dot_a-dot_c) / dot_b],
+                                      [(dot_b-dot_a) / dot_c]])
+
+
+# FEM element used for interpolation and for solving minimisation
+# problem (Reduced HCT element)
+class _ReducedHCT_Element:
+    """
+    Implementation of reduced HCT triangular element with explicit shape
+    functions.
+
+    Computes z, dz, d2z and the element stiffness matrix for bending energy:
+    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)
+
+    *** Reference for the shape functions: ***
+    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
+        reduced.
+        Michel Bernadou, Kamal Hassan
+        International Journal for Numerical Methods in Engineering.
+        17(5):784 - 789.  2.01
+
+    *** Element description: ***
+    9 dofs: z and dz given at 3 apex
+    C1 (conform)
+
+    """
+    # 1) Loads matrices to generate shape functions as a function of
+    #    triangle eccentricities - based on [1] p.11 '''
+    M = np.array([
+        [ 0.00, 0.00, 0.00,  4.50,  4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00,  0.50,  1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00,  1.25,  0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.50, 1.00, 0.00, -1.50,  0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
+        [ 0.00, 0.00, 0.00, -0.25,  0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
+        [ 0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
+        [ 0.50, 0.00, 1.00,  0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
+        [ 0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
+        [ 0.00, 0.00, 0.00,  0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
+    M0 = np.array([
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [-1.00, 0.00, 0.00,  1.50,  1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
+        [-0.50, 0.00, 0.00,  0.75,  0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00,  3.00],
+        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
+        [ 0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00,  1.50]])
+    M1 = np.array([
+        [-0.50, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
+    M2 = np.array([
+        [ 0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.50, 0.00, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [-0.25, 0.00, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
+        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
+
+    # 2) Loads matrices to rotate components of gradient & Hessian
+    #    vectors in the reference basis of triangle first apex (a0)
+    rotate_dV = np.array([[ 1.,  0.], [ 0.,  1.],
+                          [ 0.,  1.], [-1., -1.],
+                          [-1., -1.], [ 1.,  0.]])
+
+    rotate_d2V = np.array([[1., 0., 0.], [0., 1., 0.], [ 0.,  0.,  1.],
+                           [0., 1., 0.], [1., 1., 1.], [ 0., -2., -1.],
+                           [1., 1., 1.], [1., 0., 0.], [-2.,  0., -1.]])
+
+    # 3) Loads Gauss points & weights on the 3 sub-_triangles for P2
+    #    exact integral - 3 points on each subtriangles.
+    # NOTE: as the 2nd derivative is discontinuous , we really need those 9
+    # points!
+    n_gauss = 9
+    gauss_pts = np.array([[13./18.,  4./18.,  1./18.],
+                          [ 4./18., 13./18.,  1./18.],
+                          [ 7./18.,  7./18.,  4./18.],
+                          [ 1./18., 13./18.,  4./18.],
+                          [ 1./18.,  4./18., 13./18.],
+                          [ 4./18.,  7./18.,  7./18.],
+                          [ 4./18.,  1./18., 13./18.],
+                          [13./18.,  1./18.,  4./18.],
+                          [ 7./18.,  4./18.,  7./18.]], dtype=np.float64)
+    gauss_w = np.ones([9], dtype=np.float64) / 9.
+
+    #  4) Stiffness matrix for curvature energy
+    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
+
+    #  5) Loads the matrix to compute DOF_rot from tri_J at apex 0
+    J0_to_J1 = np.array([[-1.,  1.], [-1.,  0.]])
+    J0_to_J2 = np.array([[ 0., -1.], [ 1., -1.]])
+
+    def get_function_values(self, alpha, ecc, dofs):
+        """
+        Parameters
+        ----------
+        alpha : is a (N x 3 x 1) array (array of column-matrices) of
+        barycentric coordinates,
+        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities,
+        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
+        degrees of freedom.
+
+        Returns
+        -------
+        Returns the N-array of interpolated function values.
+        """
+        subtri = np.argmin(alpha, axis=1)[:, 0]
+        ksi = _roll_vectorized(alpha, -subtri, axis=0)
+        E = _roll_vectorized(ecc, -subtri, axis=0)
+        x = ksi[:, 0, 0]
+        y = ksi[:, 1, 0]
+        z = ksi[:, 2, 0]
+        x_sq = x*x
+        y_sq = y*y
+        z_sq = z*z
+        V = _to_matrix_vectorized([
+            [x_sq*x], [y_sq*y], [z_sq*z], [x_sq*z], [x_sq*y], [y_sq*x],
+            [y_sq*z], [z_sq*y], [z_sq*x], [x*y*z]])
+        prod = self.M @ V
+        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
+        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
+        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
+        s = _roll_vectorized(prod, 3*subtri, axis=0)
+        return (dofs @ s)[:, 0, 0]
+
+    def get_function_derivatives(self, alpha, J, ecc, dofs):
+        """
+        Parameters
+        ----------
+        *alpha* is a (N x 3 x 1) array (array of column-matrices of
+        barycentric coordinates)
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
+        eccentricities)
+        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
+        degrees of freedom.
+
+        Returns
+        -------
+        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
+        in global coordinates at locations alpha, as a column-matrices of
+        shape (N x 2 x 1).
+        """
+        subtri = np.argmin(alpha, axis=1)[:, 0]
+        ksi = _roll_vectorized(alpha, -subtri, axis=0)
+        E = _roll_vectorized(ecc, -subtri, axis=0)
+        x = ksi[:, 0, 0]
+        y = ksi[:, 1, 0]
+        z = ksi[:, 2, 0]
+        x_sq = x*x
+        y_sq = y*y
+        z_sq = z*z
+        dV = _to_matrix_vectorized([
+            [    -3.*x_sq,     -3.*x_sq],
+            [     3.*y_sq,           0.],
+            [          0.,      3.*z_sq],
+            [     -2.*x*z, -2.*x*z+x_sq],
+            [-2.*x*y+x_sq,      -2.*x*y],
+            [ 2.*x*y-y_sq,        -y_sq],
+            [      2.*y*z,         y_sq],
+            [        z_sq,       2.*y*z],
+            [       -z_sq,  2.*x*z-z_sq],
+            [     x*z-y*z,      x*y-y*z]])
+        # Puts back dV in first apex basis
+        dV = dV @ _extract_submatrices(
+            self.rotate_dV, subtri, block_size=2, axis=0)
+
+        prod = self.M @ dV
+        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
+        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
+        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
+        dsdksi = _roll_vectorized(prod, 3*subtri, axis=0)
+        dfdksi = dofs @ dsdksi
+        # In global coordinates:
+        # Here we try to deal with the simplest colinear cases, returning a
+        # null matrix.
+        J_inv = _safe_inv22_vectorized(J)
+        dfdx = J_inv @ _transpose_vectorized(dfdksi)
+        return dfdx
+
+    def get_function_hessians(self, alpha, J, ecc, dofs):
+        """
+        Parameters
+        ----------
+        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
+        barycentric coordinates
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
+        degrees of freedom.
+
+        Returns
+        -------
+        Returns the values of interpolated function 2nd-derivatives
+        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
+        as a column-matrices of shape (N x 3 x 1).
+        """
+        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
+        d2fdksi2 = dofs @ d2sdksi2
+        H_rot = self.get_Hrot_from_J(J)
+        d2fdx2 = d2fdksi2 @ H_rot
+        return _transpose_vectorized(d2fdx2)
+
+    def get_d2Sidksij2(self, alpha, ecc):
+        """
+        Parameters
+        ----------
+        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
+        barycentric coordinates
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+
+        Returns
+        -------
+        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
+        expressed in covariant coordinates in first apex basis.
+        """
+        subtri = np.argmin(alpha, axis=1)[:, 0]
+        ksi = _roll_vectorized(alpha, -subtri, axis=0)
+        E = _roll_vectorized(ecc, -subtri, axis=0)
+        x = ksi[:, 0, 0]
+        y = ksi[:, 1, 0]
+        z = ksi[:, 2, 0]
+        d2V = _to_matrix_vectorized([
+            [     6.*x,      6.*x,      6.*x],
+            [     6.*y,        0.,        0.],
+            [       0.,      6.*z,        0.],
+            [     2.*z, 2.*z-4.*x, 2.*z-2.*x],
+            [2.*y-4.*x,      2.*y, 2.*y-2.*x],
+            [2.*x-4.*y,        0.,     -2.*y],
+            [     2.*z,        0.,      2.*y],
+            [       0.,      2.*y,      2.*z],
+            [       0., 2.*x-4.*z,     -2.*z],
+            [    -2.*z,     -2.*y,     x-y-z]])
+        # Puts back d2V in first apex basis
+        d2V = d2V @ _extract_submatrices(
+            self.rotate_d2V, subtri, block_size=3, axis=0)
+        prod = self.M @ d2V
+        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
+        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
+        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
+        d2sdksi2 = _roll_vectorized(prod, 3*subtri, axis=0)
+        return d2sdksi2
+
+    def get_bending_matrices(self, J, ecc):
+        """
+        Parameters
+        ----------
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+
+        Returns
+        -------
+        Returns the element K matrices for bending energy expressed in
+        GLOBAL nodal coordinates.
+        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
+        tri_J is needed to rotate dofs from local basis to global basis
+        """
+        n = np.size(ecc, 0)
+
+        # 1) matrix to rotate dofs in global coordinates
+        J1 = self.J0_to_J1 @ J
+        J2 = self.J0_to_J2 @ J
+        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
+        DOF_rot[:, 0, 0] = 1
+        DOF_rot[:, 3, 3] = 1
+        DOF_rot[:, 6, 6] = 1
+        DOF_rot[:, 1:3, 1:3] = J
+        DOF_rot[:, 4:6, 4:6] = J1
+        DOF_rot[:, 7:9, 7:9] = J2
+
+        # 2) matrix to rotate Hessian in global coordinates.
+        H_rot, area = self.get_Hrot_from_J(J, return_area=True)
+
+        # 3) Computes stiffness matrix
+        # Gauss quadrature.
+        K = np.zeros([n, 9, 9], dtype=np.float64)
+        weights = self.gauss_w
+        pts = self.gauss_pts
+        for igauss in range(self.n_gauss):
+            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
+            alpha = np.expand_dims(alpha, 2)
+            weight = weights[igauss]
+            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
+            d2Skdx2 = d2Skdksi2 @ H_rot
+            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))
+
+        # 4) With nodal (not elem) dofs
+        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
+
+        # 5) Need the area to compute total element energy
+        return _scalar_vectorized(area, K)
+
+    def get_Hrot_from_J(self, J, return_area=False):
+        """
+        Parameters
+        ----------
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+
+        Returns
+        -------
+        Returns H_rot used to rotate Hessian from local basis of first apex,
+        to global coordinates.
+        if *return_area* is True, returns also the triangle area (0.5*det(J))
+        """
+        # Here we try to deal with the simplest colinear cases; a null
+        # energy and area is imposed.
+        J_inv = _safe_inv22_vectorized(J)
+        Ji00 = J_inv[:, 0, 0]
+        Ji11 = J_inv[:, 1, 1]
+        Ji10 = J_inv[:, 1, 0]
+        Ji01 = J_inv[:, 0, 1]
+        H_rot = _to_matrix_vectorized([
+            [Ji00*Ji00, Ji10*Ji10, Ji00*Ji10],
+            [Ji01*Ji01, Ji11*Ji11, Ji01*Ji11],
+            [2*Ji00*Ji01, 2*Ji11*Ji10, Ji00*Ji11+Ji10*Ji01]])
+        if not return_area:
+            return H_rot
+        else:
+            area = 0.5 * (J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0])
+            return H_rot, area
+
+    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
+        """
+        Build K and F for the following elliptic formulation:
+        minimization of curvature energy with value of function at node
+        imposed and derivatives 'free'.
+
+        Build the global Kff matrix in cco format.
+        Build the full Ff vec Ff = - Kfc x Uc.
+
+        Parameters
+        ----------
+        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
+        triangle first apex)
+        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
+        eccentricities
+        *triangles* is a (N x 3) array of nodes indexes.
+        *Uc* is (N x 3) array of imposed displacements at nodes
+
+        Returns
+        -------
+        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
+        (row, col) entries must be summed.
+        Ff: force vector - dim npts * 3
+        """
+        ntri = np.size(ecc, 0)
+        vec_range = np.arange(ntri, dtype=np.int32)
+        c_indices = np.full(ntri, -1, dtype=np.int32)  # for unused dofs, -1
+        f_dof = [1, 2, 4, 5, 7, 8]
+        c_dof = [0, 3, 6]
+
+        # vals, rows and cols indices in global dof numbering
+        f_dof_indices = _to_matrix_vectorized([[
+            c_indices, triangles[:, 0]*2, triangles[:, 0]*2+1,
+            c_indices, triangles[:, 1]*2, triangles[:, 1]*2+1,
+            c_indices, triangles[:, 2]*2, triangles[:, 2]*2+1]])
+
+        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
+        f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
+        f_col_indices = expand_indices @ f_dof_indices
+        K_elem = self.get_bending_matrices(J, ecc)
+
+        # Extracting sub-matrices
+        # Explanation & notations:
+        # * Subscript f denotes 'free' degrees of freedom (i.e. dz/dx, dz/dx)
+        # * Subscript c denotes 'condensated' (imposed) degrees of freedom
+        #    (i.e. z at all nodes)
+        # * F = [Ff, Fc] is the force vector
+        # * U = [Uf, Uc] is the imposed dof vector
+        #        [ Kff Kfc ]
+        # * K =  [         ]  is the laplacian stiffness matrix
+        #        [ Kcf Kff ]
+        # * As F = K x U one gets straightforwardly: Ff = - Kfc x Uc
+
+        # Computing Kff stiffness matrix in sparse coo format
+        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
+        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
+        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])
+
+        # Computing Ff force vector in sparse coo format
+        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
+        Uc_elem = np.expand_dims(Uc, axis=2)
+        Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
+        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]
+
+        # Extracting Ff force vector in dense format
+        # We have to sum duplicate indices -  using bincount
+        Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
+        return Kff_rows, Kff_cols, Kff_vals, Ff
+
+
+# :class:_DOF_estimator, _DOF_estimator_user, _DOF_estimator_geom,
+# _DOF_estimator_min_E
+# Private classes used to compute the degree of freedom of each triangular
+# element for the TriCubicInterpolator.
+class _DOF_estimator:
+    """
+    Abstract base class for classes used to estimate a function's first
+    derivatives, and deduce the dofs for a CubicTriInterpolator using a
+    reduced HCT element formulation.
+
+    Derived classes implement ``compute_df(self, **kwargs)``, returning
+    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
+    gradient coordinates.
+    """
+    def __init__(self, interpolator, **kwargs):
+        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
+        self._pts = interpolator._pts
+        self._tris_pts = interpolator._tris_pts
+        self.z = interpolator._z
+        self._triangles = interpolator._triangles
+        (self._unit_x, self._unit_y) = (interpolator._unit_x,
+                                        interpolator._unit_y)
+        self.dz = self.compute_dz(**kwargs)
+        self.compute_dof_from_df()
+
+    def compute_dz(self, **kwargs):
+        raise NotImplementedError
+
+    def compute_dof_from_df(self):
+        """
+        Compute reduced-HCT elements degrees of freedom, from the gradient.
+        """
+        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
+        tri_z = self.z[self._triangles]
+        tri_dz = self.dz[self._triangles]
+        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
+        return tri_dof
+
+    @staticmethod
+    def get_dof_vec(tri_z, tri_dz, J):
+        """
+        Compute the dof vector of a triangle, from the value of f, df and
+        of the local Jacobian at each node.
+
+        Parameters
+        ----------
+        tri_z : shape (3,) array
+            f nodal values.
+        tri_dz : shape (3, 2) array
+            df/dx, df/dy nodal values.
+        J
+            Jacobian matrix in local basis of apex 0.
+
+        Returns
+        -------
+        dof : shape (9,) array
+            For each apex ``iapex``::
+
+                dof[iapex*3+0] = f(Ai)
+                dof[iapex*3+1] = df(Ai).(AiAi+)
+                dof[iapex*3+2] = df(Ai).(AiAi-)
+        """
+        npt = tri_z.shape[0]
+        dof = np.zeros([npt, 9], dtype=np.float64)
+        J1 = _ReducedHCT_Element.J0_to_J1 @ J
+        J2 = _ReducedHCT_Element.J0_to_J2 @ J
+
+        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
+        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
+        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
+
+        dfdksi = _to_matrix_vectorized([
+            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
+            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
+        dof[:, 0:7:3] = tri_z
+        dof[:, 1:8:3] = dfdksi[:, 0]
+        dof[:, 2:9:3] = dfdksi[:, 1]
+        return dof
+
+
+class _DOF_estimator_user(_DOF_estimator):
+    """dz is imposed by user; accounts for scaling if any."""
+
+    def compute_dz(self, dz):
+        (dzdx, dzdy) = dz
+        dzdx = dzdx * self._unit_x
+        dzdy = dzdy * self._unit_y
+        return np.vstack([dzdx, dzdy]).T
+
+
+class _DOF_estimator_geom(_DOF_estimator):
+    """Fast 'geometric' approximation, recommended for large arrays."""
+
+    def compute_dz(self):
+        """
+        self.df is computed as weighted average of _triangles sharing a common
+        node. On each triangle itri f is first assumed linear (= ~f), which
+        allows to compute d~f[itri]
+        Then the following approximation of df nodal values is then proposed:
+            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
+        The weighted coeff. w[itri] are proportional to the angle of the
+        triangle itri at apex ipt
+        """
+        el_geom_w = self.compute_geom_weights()
+        el_geom_grad = self.compute_geom_grads()
+
+        # Sum of weights coeffs
+        w_node_sum = np.bincount(np.ravel(self._triangles),
+                                 weights=np.ravel(el_geom_w))
+
+        # Sum of weighted df = (dfx, dfy)
+        dfx_el_w = np.empty_like(el_geom_w)
+        dfy_el_w = np.empty_like(el_geom_w)
+        for iapex in range(3):
+            dfx_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 0]
+            dfy_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 1]
+        dfx_node_sum = np.bincount(np.ravel(self._triangles),
+                                   weights=np.ravel(dfx_el_w))
+        dfy_node_sum = np.bincount(np.ravel(self._triangles),
+                                   weights=np.ravel(dfy_el_w))
+
+        # Estimation of df
+        dfx_estim = dfx_node_sum/w_node_sum
+        dfy_estim = dfy_node_sum/w_node_sum
+        return np.vstack([dfx_estim, dfy_estim]).T
+
+    def compute_geom_weights(self):
+        """
+        Build the (nelems, 3) weights coeffs of _triangles angles,
+        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
+        """
+        weights = np.zeros([np.size(self._triangles, 0), 3])
+        tris_pts = self._tris_pts
+        for ipt in range(3):
+            p0 = tris_pts[:, ipt % 3, :]
+            p1 = tris_pts[:, (ipt+1) % 3, :]
+            p2 = tris_pts[:, (ipt-1) % 3, :]
+            alpha1 = np.arctan2(p1[:, 1]-p0[:, 1], p1[:, 0]-p0[:, 0])
+            alpha2 = np.arctan2(p2[:, 1]-p0[:, 1], p2[:, 0]-p0[:, 0])
+            # In the below formula we could take modulo 2. but
+            # modulo 1. is safer regarding round-off errors (flat triangles).
+            angle = np.abs(((alpha2-alpha1) / np.pi) % 1)
+            # Weight proportional to angle up np.pi/2; null weight for
+            # degenerated cases 0 and np.pi (note that *angle* is normalized
+            # by np.pi).
+            weights[:, ipt] = 0.5 - np.abs(angle-0.5)
+        return weights
+
+    def compute_geom_grads(self):
+        """
+        Compute the (global) gradient component of f assumed linear (~f).
+        returns array df of shape (nelems, 2)
+        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
+        """
+        tris_pts = self._tris_pts
+        tris_f = self.z[self._triangles]
+
+        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
+        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
+        dM = np.dstack([dM1, dM2])
+        # Here we try to deal with the simplest colinear cases: a null
+        # gradient is assumed in this case.
+        dM_inv = _safe_inv22_vectorized(dM)
+
+        dZ1 = tris_f[:, 1] - tris_f[:, 0]
+        dZ2 = tris_f[:, 2] - tris_f[:, 0]
+        dZ = np.vstack([dZ1, dZ2]).T
+        df = np.empty_like(dZ)
+
+        # With np.einsum: could be ej,eji -> ej
+        df[:, 0] = dZ[:, 0]*dM_inv[:, 0, 0] + dZ[:, 1]*dM_inv[:, 1, 0]
+        df[:, 1] = dZ[:, 0]*dM_inv[:, 0, 1] + dZ[:, 1]*dM_inv[:, 1, 1]
+        return df
+
+
+class _DOF_estimator_min_E(_DOF_estimator_geom):
+    """
+    The 'smoothest' approximation, df is computed through global minimization
+    of the bending energy:
+      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
+    """
+    def __init__(self, Interpolator):
+        self._eccs = Interpolator._eccs
+        super().__init__(Interpolator)
+
+    def compute_dz(self):
+        """
+        Elliptic solver for bending energy minimization.
+        Uses a dedicated 'toy' sparse Jacobi PCG solver.
+        """
+        # Initial guess for iterative PCG solver.
+        dz_init = super().compute_dz()
+        Uf0 = np.ravel(dz_init)
+
+        reference_element = _ReducedHCT_Element()
+        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
+        eccs = self._eccs
+        triangles = self._triangles
+        Uc = self.z[self._triangles]
+
+        # Building stiffness matrix and force vector in coo format
+        Kff_rows, Kff_cols, Kff_vals, Ff = reference_element.get_Kff_and_Ff(
+            J, eccs, triangles, Uc)
+
+        # Building sparse matrix and solving minimization problem
+        # We could use scipy.sparse direct solver; however to avoid this
+        # external dependency an implementation of a simple PCG solver with
+        # a simple diagonal Jacobi preconditioner is implemented.
+        tol = 1.e-10
+        n_dof = Ff.shape[0]
+        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
+                                     shape=(n_dof, n_dof))
+        Kff_coo.compress_csc()
+        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
+        # If the PCG did not converge, we return the best guess between Uf0
+        # and Uf.
+        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
+        if err0 < err:
+            # Maybe a good occasion to raise a warning here ?
+            _api.warn_external("In TriCubicInterpolator initialization, "
+                               "PCG sparse solver did not converge after "
+                               "1000 iterations. `geom` approximation is "
+                               "used instead of `min_E`")
+            Uf = Uf0
+
+        # Building dz from Uf
+        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
+        dz[:, 0] = Uf[::2]
+        dz[:, 1] = Uf[1::2]
+        return dz
+
+
+# The following private :class:_Sparse_Matrix_coo and :func:_cg provide
+# a PCG sparse solver for (symmetric) elliptic problems.
+class _Sparse_Matrix_coo:
+    def __init__(self, vals, rows, cols, shape):
+        """
+        Create a sparse matrix in coo format.
+        *vals*: arrays of values of non-null entries of the matrix
+        *rows*: int arrays of rows of non-null entries of the matrix
+        *cols*: int arrays of cols of non-null entries of the matrix
+        *shape*: 2-tuple (n, m) of matrix shape
+        """
+        self.n, self.m = shape
+        self.vals = np.asarray(vals, dtype=np.float64)
+        self.rows = np.asarray(rows, dtype=np.int32)
+        self.cols = np.asarray(cols, dtype=np.int32)
+
+    def dot(self, V):
+        """
+        Dot product of self by a vector *V* in sparse-dense to dense format
+        *V* dense vector of shape (self.m,).
+        """
+        assert V.shape == (self.m,)
+        return np.bincount(self.rows,
+                           weights=self.vals*V[self.cols],
+                           minlength=self.m)
+
+    def compress_csc(self):
+        """
+        Compress rows, cols, vals / summing duplicates. Sort for csc format.
+        """
+        _, unique, indices = np.unique(
+            self.rows + self.n*self.cols,
+            return_index=True, return_inverse=True)
+        self.rows = self.rows[unique]
+        self.cols = self.cols[unique]
+        self.vals = np.bincount(indices, weights=self.vals)
+
+    def compress_csr(self):
+        """
+        Compress rows, cols, vals / summing duplicates. Sort for csr format.
+        """
+        _, unique, indices = np.unique(
+            self.m*self.rows + self.cols,
+            return_index=True, return_inverse=True)
+        self.rows = self.rows[unique]
+        self.cols = self.cols[unique]
+        self.vals = np.bincount(indices, weights=self.vals)
+
+    def to_dense(self):
+        """
+        Return a dense matrix representing self, mainly for debugging purposes.
+        """
+        ret = np.zeros([self.n, self.m], dtype=np.float64)
+        nvals = self.vals.size
+        for i in range(nvals):
+            ret[self.rows[i], self.cols[i]] += self.vals[i]
+        return ret
+
+    def __str__(self):
+        return self.to_dense().__str__()
+
+    @property
+    def diag(self):
+        """Return the (dense) vector of the diagonal elements."""
+        in_diag = (self.rows == self.cols)
+        diag = np.zeros(min(self.n, self.n), dtype=np.float64)  # default 0.
+        diag[self.rows[in_diag]] = self.vals[in_diag]
+        return diag
+
+
+def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
+    """
+    Use Preconditioned Conjugate Gradient iteration to solve A x = b
+    A simple Jacobi (diagonal) preconditioner is used.
+
+    Parameters
+    ----------
+    A : _Sparse_Matrix_coo
+        *A* must have been compressed before by compress_csc or
+        compress_csr method.
+    b : array
+        Right hand side of the linear system.
+    x0 : array, optional
+        Starting guess for the solution. Defaults to the zero vector.
+    tol : float, optional
+        Tolerance to achieve. The algorithm terminates when the relative
+        residual is below tol. Default is 1e-10.
+    maxiter : int, optional
+        Maximum number of iterations.  Iteration will stop after *maxiter*
+        steps even if the specified tolerance has not been achieved. Defaults
+        to 1000.
+
+    Returns
+    -------
+    x : array
+        The converged solution.
+    err : float
+        The absolute error np.linalg.norm(A.dot(x) - b)
+    """
+    n = b.size
+    assert A.n == n
+    assert A.m == n
+    b_norm = np.linalg.norm(b)
+
+    # Jacobi pre-conditioner
+    kvec = A.diag
+    # For diag elem < 1e-6 we keep 1e-6.
+    kvec = np.maximum(kvec, 1e-6)
+
+    # Initial guess
+    if x0 is None:
+        x = np.zeros(n)
+    else:
+        x = x0
+
+    r = b - A.dot(x)
+    w = r/kvec
+
+    p = np.zeros(n)
+    beta = 0.0
+    rho = np.dot(r, w)
+    k = 0
+
+    # Following C. T. Kelley
+    while (np.sqrt(abs(rho)) > tol*b_norm) and (k < maxiter):
+        p = w + beta*p
+        z = A.dot(p)
+        alpha = rho/np.dot(p, z)
+        r = r - alpha*z
+        w = r/kvec
+        rhoold = rho
+        rho = np.dot(r, w)
+        x = x + alpha*p
+        beta = rho/rhoold
+        # err = np.linalg.norm(A.dot(x) - b)  # absolute accuracy - not used
+        k += 1
+    err = np.linalg.norm(A.dot(x) - b)
+    return x, err
+
+
+# The following private functions:
+#     :func:`_safe_inv22_vectorized`
+#     :func:`_pseudo_inv22sym_vectorized`
+#     :func:`_scalar_vectorized`
+#     :func:`_transpose_vectorized`
+#     :func:`_roll_vectorized`
+#     :func:`_to_matrix_vectorized`
+#     :func:`_extract_submatrices`
+# provide fast numpy implementation of some standard operations on arrays of
+# matrices - stored as (:, n_rows, n_cols)-shaped np.arrays.
+
+# Development note: Dealing with pathologic 'flat' triangles in the
+# CubicTriInterpolator code and impact on (2, 2)-matrix inversion functions
+# :func:`_safe_inv22_vectorized` and :func:`_pseudo_inv22sym_vectorized`.
+#
+# Goals:
+# 1) The CubicTriInterpolator should be able to handle flat or almost flat
+#    triangles without raising an error,
+# 2) These degenerated triangles should have no impact on the automatic dof
+#    calculation (associated with null weight for the _DOF_estimator_geom and
+#    with null energy for the _DOF_estimator_min_E),
+# 3) Linear patch test should be passed exactly on degenerated meshes,
+# 4) Interpolation (with :meth:`_interpolate_single_key` or
+#    :meth:`_interpolate_multi_key`) shall be correctly handled even *inside*
+#    the pathologic triangles, to interact correctly with a TriRefiner class.
+#
+# Difficulties:
+# Flat triangles have rank-deficient *J* (so-called jacobian matrix) and
+# *metric* (the metric tensor = J x J.T). Computation of the local
+# tangent plane is also problematic.
+#
+# Implementation:
+# Most of the time, when computing the inverse of a rank-deficient matrix it
+# is safe to simply return the null matrix (which is the implementation in
+# :func:`_safe_inv22_vectorized`). This is because of point 2), itself
+# enforced by:
+#    - null area hence null energy in :class:`_DOF_estimator_min_E`
+#    - angles close or equal to 0 or np.pi hence null weight in
+#      :class:`_DOF_estimator_geom`.
+#      Note that the function angle -> weight is continuous and maximum for an
+#      angle np.pi/2 (refer to :meth:`compute_geom_weights`)
+# The exception is the computation of barycentric coordinates, which is done
+# by inversion of the *metric* matrix. In this case, we need to compute a set
+# of valid coordinates (1 among numerous possibilities), to ensure point 4).
+# We benefit here from the symmetry of metric = J x J.T, which makes it easier
+# to compute a pseudo-inverse in :func:`_pseudo_inv22sym_vectorized`
+def _safe_inv22_vectorized(M):
+    """
+    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
+    matrices.
+
+    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
+    """
+    _api.check_shape((None, 2, 2), M=M)
+    M_inv = np.empty_like(M)
+    prod1 = M[:, 0, 0]*M[:, 1, 1]
+    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
+
+    # We set delta_inv to 0. in case of a rank deficient matrix; a
+    # rank-deficient input matrix *M* will lead to a null matrix in output
+    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
+    if np.all(rank2):
+        # Normal 'optimized' flow.
+        delta_inv = 1./delta
+    else:
+        # 'Pathologic' flow.
+        delta_inv = np.zeros(M.shape[0])
+        delta_inv[rank2] = 1./delta[rank2]
+
+    M_inv[:, 0, 0] = M[:, 1, 1]*delta_inv
+    M_inv[:, 0, 1] = -M[:, 0, 1]*delta_inv
+    M_inv[:, 1, 0] = -M[:, 1, 0]*delta_inv
+    M_inv[:, 1, 1] = M[:, 0, 0]*delta_inv
+    return M_inv
+
+
+def _pseudo_inv22sym_vectorized(M):
+    """
+    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
+    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.
+
+    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
+    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
+    In case M is of rank 0, we return the null matrix.
+
+    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
+    """
+    _api.check_shape((None, 2, 2), M=M)
+    M_inv = np.empty_like(M)
+    prod1 = M[:, 0, 0]*M[:, 1, 1]
+    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
+    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
+
+    if np.all(rank2):
+        # Normal 'optimized' flow.
+        M_inv[:, 0, 0] = M[:, 1, 1] / delta
+        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
+        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
+        M_inv[:, 1, 1] = M[:, 0, 0] / delta
+    else:
+        # 'Pathologic' flow.
+        # Here we have to deal with 2 sub-cases
+        # 1) First sub-case: matrices of rank 2:
+        delta = delta[rank2]
+        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
+        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
+        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
+        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
+        # 2) Second sub-case: rank-deficient matrices of rank 0 and 1:
+        rank01 = ~rank2
+        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
+        tr_zeros = (np.abs(tr) < 1.e-8)
+        sq_tr_inv = (1.-tr_zeros) / (tr**2+tr_zeros)
+        # sq_tr_inv = 1. / tr**2
+        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
+        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
+        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
+        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv
+
+    return M_inv
+
+
+def _scalar_vectorized(scalar, M):
+    """
+    Scalar product between scalars and matrices.
+    """
+    return scalar[:, np.newaxis, np.newaxis]*M
+
+
+def _transpose_vectorized(M):
+    """
+    Transposition of an array of matrices *M*.
+    """
+    return np.transpose(M, [0, 2, 1])
+
+
+def _roll_vectorized(M, roll_indices, axis):
+    """
+    Roll an array of matrices along *axis* (0: rows, 1: columns) according to
+    an array of indices *roll_indices*.
+    """
+    assert axis in [0, 1]
+    ndim = M.ndim
+    assert ndim == 3
+    ndim_roll = roll_indices.ndim
+    assert ndim_roll == 1
+    sh = M.shape
+    r, c = sh[-2:]
+    assert sh[0] == roll_indices.shape[0]
+    vec_indices = np.arange(sh[0], dtype=np.int32)
+
+    # Builds the rolled matrix
+    M_roll = np.empty_like(M)
+    if axis == 0:
+        for ir in range(r):
+            for ic in range(c):
+                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices+ir) % r, ic]
+    else:  # 1
+        for ir in range(r):
+            for ic in range(c):
+                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices+ic) % c]
+    return M_roll
+
+
+def _to_matrix_vectorized(M):
+    """
+    Build an array of matrices from individuals np.arrays of identical shapes.
+
+    Parameters
+    ----------
+    M
+        ncols-list of nrows-lists of shape sh.
+
+    Returns
+    -------
+    M_res : np.array of shape (sh, nrow, ncols)
+        *M_res* satisfies ``M_res[..., i, j] = M[i][j]``.
+    """
+    assert isinstance(M, (tuple, list))
+    assert all(isinstance(item, (tuple, list)) for item in M)
+    c_vec = np.asarray([len(item) for item in M])
+    assert np.all(c_vec-c_vec[0] == 0)
+    r = len(M)
+    c = c_vec[0]
+    M00 = np.asarray(M[0][0])
+    dt = M00.dtype
+    sh = [M00.shape[0], r, c]
+    M_ret = np.empty(sh, dtype=dt)
+    for irow in range(r):
+        for icol in range(c):
+            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
+    return M_ret
+
+
+def _extract_submatrices(M, block_indices, block_size, axis):
+    """
+    Extract selected blocks of a matrices *M* depending on parameters
+    *block_indices* and *block_size*.
+
+    Returns the array of extracted matrices *Mres* so that ::
+
+        M_res[..., ir, :] = M[(block_indices*block_size+ir), :]
+    """
+    assert block_indices.ndim == 1
+    assert axis in [0, 1]
+
+    r, c = M.shape
+    if axis == 0:
+        sh = [block_indices.shape[0], block_size, c]
+    else:  # 1
+        sh = [block_indices.shape[0], r, block_size]
+
+    dt = M.dtype
+    M_res = np.empty(sh, dtype=dt)
+    if axis == 0:
+        for ir in range(block_size):
+            M_res[:, ir, :] = M[(block_indices*block_size+ir), :]
+    else:  # 1
+        for ic in range(block_size):
+            M_res[:, :, ic] = M[:, (block_indices*block_size+ic)]
+
+    return M_res
diff --git a/lib/matplotlib/tri/_tripcolor.py b/lib/matplotlib/tri/_tripcolor.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_tripcolor.py
@@ -0,0 +1,154 @@
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.collections import PolyCollection, TriMesh
+from matplotlib.colors import Normalize
+from matplotlib.tri._triangulation import Triangulation
+
+
+def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
+              vmax=None, shading='flat', facecolors=None, **kwargs):
+    """
+    Create a pseudocolor plot of an unstructured triangular grid.
+
+    Call signatures::
+
+      tripcolor(triangulation, c, *, ...)
+      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)
+
+    The triangular grid can be specified either by passing a `.Triangulation`
+    object as the first parameter, or by passing the points *x*, *y* and
+    optionally the *triangles* and a *mask*. See `.Triangulation` for an
+    explanation of these parameters.
+
+    It is possible to pass the triangles positionally, i.e.
+    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
+    For more clarity, pass *triangles* via keyword argument.
+
+    If neither of *triangulation* or *triangles* are given, the triangulation
+    is calculated on the fly. In this case, it does not make sense to provide
+    colors at the triangle faces via *c* or *facecolors* because there are
+    multiple possible triangulations for a group of points and you don't know
+    which triangles will be constructed.
+
+    Parameters
+    ----------
+    triangulation : `.Triangulation`
+        An already created triangular grid.
+    x, y, triangles, mask
+        Parameters defining the triangular grid. See `.Triangulation`.
+        This is mutually exclusive with specifying *triangulation*.
+    c : array-like
+        The color values, either for the points or for the triangles. Which one
+        is automatically inferred from the length of *c*, i.e. does it match
+        the number of points or the number of triangles. If there are the same
+        number of points and triangles in the triangulation it is assumed that
+        color values are defined at points; to force the use of color values at
+        triangles use the keyword argument ``facecolors=c`` instead of just
+        ``c``.
+        This parameter is position-only.
+    facecolors : array-like, optional
+        Can be used alternatively to *c* to specify colors at the triangle
+        faces. This parameter takes precedence over *c*.
+    shading : {'flat', 'gouraud'}, default: 'flat'
+        If  'flat' and the color values *c* are defined at points, the color
+        values used for each triangle are from the mean c of the triangle's
+        three points. If *shading* is 'gouraud' then color values must be
+        defined at points.
+    other_parameters
+        All other parameters are the same as for `~.Axes.pcolor`.
+    """
+    _api.check_in_list(['flat', 'gouraud'], shading=shading)
+
+    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
+
+    # Parse the color to be in one of (the other variable will be None):
+    # - facecolors: if specified at the triangle faces
+    # - point_colors: if specified at the points
+    if facecolors is not None:
+        if args:
+            _api.warn_external(
+                "Positional parameter c has no effect when the keyword "
+                "facecolors is given")
+        point_colors = None
+        if len(facecolors) != len(tri.triangles):
+            raise ValueError("The length of facecolors must match the number "
+                             "of triangles")
+    else:
+        # Color from positional parameter c
+        if not args:
+            raise TypeError(
+                "tripcolor() missing 1 required positional argument: 'c'; or "
+                "1 required keyword-only argument: 'facecolors'")
+        elif len(args) > 1:
+            _api.warn_deprecated(
+                "3.6", message=f"Additional positional parameters "
+                f"{args[1:]!r} are ignored; support for them is deprecated "
+                f"since %(since)s and will be removed %(removal)s")
+        c = np.asarray(args[0])
+        if len(c) == len(tri.x):
+            # having this before the len(tri.triangles) comparison gives
+            # precedence to nodes if there are as many nodes as triangles
+            point_colors = c
+            facecolors = None
+        elif len(c) == len(tri.triangles):
+            point_colors = None
+            facecolors = c
+        else:
+            raise ValueError('The length of c must match either the number '
+                             'of points or the number of triangles')
+
+    # Handling of linewidths, shading, edgecolors and antialiased as
+    # in Axes.pcolor
+    linewidths = (0.25,)
+    if 'linewidth' in kwargs:
+        kwargs['linewidths'] = kwargs.pop('linewidth')
+    kwargs.setdefault('linewidths', linewidths)
+
+    edgecolors = 'none'
+    if 'edgecolor' in kwargs:
+        kwargs['edgecolors'] = kwargs.pop('edgecolor')
+    ec = kwargs.setdefault('edgecolors', edgecolors)
+
+    if 'antialiased' in kwargs:
+        kwargs['antialiaseds'] = kwargs.pop('antialiased')
+    if 'antialiaseds' not in kwargs and ec.lower() == "none":
+        kwargs['antialiaseds'] = False
+
+    _api.check_isinstance((Normalize, None), norm=norm)
+    if shading == 'gouraud':
+        if facecolors is not None:
+            raise ValueError(
+                "shading='gouraud' can only be used when the colors "
+                "are specified at the points, not at the faces.")
+        collection = TriMesh(tri, alpha=alpha, array=point_colors,
+                             cmap=cmap, norm=norm, **kwargs)
+    else:  # 'flat'
+        # Vertices of triangles.
+        maskedTris = tri.get_masked_triangles()
+        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)
+
+        # Color values.
+        if facecolors is None:
+            # One color per triangle, the mean of the 3 vertex color values.
+            colors = point_colors[maskedTris].mean(axis=1)
+        elif tri.mask is not None:
+            # Remove color values of masked triangles.
+            colors = facecolors[~tri.mask]
+        else:
+            colors = facecolors
+        collection = PolyCollection(verts, alpha=alpha, array=colors,
+                                    cmap=cmap, norm=norm, **kwargs)
+
+    collection._scale_norm(norm, vmin, vmax)
+    ax.grid(False)
+
+    minx = tri.x.min()
+    maxx = tri.x.max()
+    miny = tri.y.min()
+    maxy = tri.y.max()
+    corners = (minx, miny), (maxx, maxy)
+    ax.update_datalim(corners)
+    ax.autoscale_view()
+    ax.add_collection(collection)
+    return collection
diff --git a/lib/matplotlib/tri/_triplot.py b/lib/matplotlib/tri/_triplot.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_triplot.py
@@ -0,0 +1,86 @@
+import numpy as np
+from matplotlib.tri._triangulation import Triangulation
+import matplotlib.cbook as cbook
+import matplotlib.lines as mlines
+
+
+def triplot(ax, *args, **kwargs):
+    """
+    Draw an unstructured triangular grid as lines and/or markers.
+
+    Call signatures::
+
+      triplot(triangulation, ...)
+      triplot(x, y, [triangles], *, [mask=mask], ...)
+
+    The triangular grid can be specified either by passing a `.Triangulation`
+    object as the first parameter, or by passing the points *x*, *y* and
+    optionally the *triangles* and a *mask*. If neither of *triangulation* or
+    *triangles* are given, the triangulation is calculated on the fly.
+
+    Parameters
+    ----------
+    triangulation : `.Triangulation`
+        An already created triangular grid.
+    x, y, triangles, mask
+        Parameters defining the triangular grid. See `.Triangulation`.
+        This is mutually exclusive with specifying *triangulation*.
+    other_parameters
+        All other args and kwargs are forwarded to `~.Axes.plot`.
+
+    Returns
+    -------
+    lines : `~matplotlib.lines.Line2D`
+        The drawn triangles edges.
+    markers : `~matplotlib.lines.Line2D`
+        The drawn marker nodes.
+    """
+    import matplotlib.axes
+
+    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
+    x, y, edges = (tri.x, tri.y, tri.edges)
+
+    # Decode plot format string, e.g., 'ro-'
+    fmt = args[0] if args else ""
+    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)
+
+    # Insert plot format string into a copy of kwargs (kwargs values prevail).
+    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
+    for key, val in zip(('linestyle', 'marker', 'color'),
+                        (linestyle, marker, color)):
+        if val is not None:
+            kw.setdefault(key, val)
+
+    # Draw lines without markers.
+    # Note 1: If we drew markers here, most markers would be drawn more than
+    #         once as they belong to several edges.
+    # Note 2: We insert nan values in the flattened edges arrays rather than
+    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
+    #         as it considerably speeds-up code execution.
+    linestyle = kw['linestyle']
+    kw_lines = {
+        **kw,
+        'marker': 'None',  # No marker to draw.
+        'zorder': kw.get('zorder', 1),  # Path default zorder is used.
+    }
+    if linestyle not in [None, 'None', '', ' ']:
+        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
+        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
+        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
+                            **kw_lines)
+    else:
+        tri_lines = ax.plot([], [], **kw_lines)
+
+    # Draw markers separately.
+    marker = kw['marker']
+    kw_markers = {
+        **kw,
+        'linestyle': 'None',  # No line to draw.
+    }
+    kw_markers.pop('label', None)
+    if marker not in [None, 'None', '', ' ']:
+        tri_markers = ax.plot(x, y, **kw_markers)
+    else:
+        tri_markers = ax.plot([], [], **kw_markers)
+
+    return tri_lines + tri_markers
diff --git a/lib/matplotlib/tri/_trirefine.py b/lib/matplotlib/tri/_trirefine.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_trirefine.py
@@ -0,0 +1,307 @@
+"""
+Mesh refinement for triangular grids.
+"""
+
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri._triangulation import Triangulation
+import matplotlib.tri._triinterpolate
+
+
+class TriRefiner:
+    """
+    Abstract base class for classes implementing mesh refinement.
+
+    A TriRefiner encapsulates a Triangulation object and provides tools for
+    mesh refinement and interpolation.
+
+    Derived classes must implement:
+
+    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
+      the optional keyword arguments *kwargs* are defined in each
+      TriRefiner concrete implementation, and which returns:
+
+      - a refined triangulation,
+      - optionally (depending on *return_tri_index*), for each
+        point of the refined triangulation: the index of
+        the initial triangulation triangle to which it belongs.
+
+    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:
+
+      - *z* array of field values (to refine) defined at the base
+        triangulation nodes,
+      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
+      - the other optional keyword arguments *kwargs* are defined in
+        each TriRefiner concrete implementation;
+
+      and which returns (as a tuple) a refined triangular mesh and the
+      interpolated values of the field at the refined triangulation nodes.
+    """
+
+    def __init__(self, triangulation):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+
+class UniformTriRefiner(TriRefiner):
+    """
+    Uniform mesh refinement by recursive subdivisions.
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The encapsulated triangulation (to be refined)
+    """
+#    See Also
+#    --------
+#    :class:`~matplotlib.tri.CubicTriInterpolator` and
+#    :class:`~matplotlib.tri.TriAnalyzer`.
+#    """
+    def __init__(self, triangulation):
+        super().__init__(triangulation)
+
+    def refine_triangulation(self, return_tri_index=False, subdiv=3):
+        """
+        Compute an uniformly refined triangulation *refi_triangulation* of
+        the encapsulated :attr:`triangulation`.
+
+        This function refines the encapsulated triangulation by splitting each
+        father triangle into 4 child sub-triangles built on the edges midside
+        nodes, recursing *subdiv* times.  In the end, each triangle is hence
+        divided into ``4**subdiv`` child triangles.
+
+        Parameters
+        ----------
+        return_tri_index : bool, default: False
+            Whether an index table indicating the father triangle index of each
+            point is returned.
+        subdiv : int, default: 3
+            Recursion level for the subdivision.
+            Each triangle is divided into ``4**subdiv`` child triangles;
+            hence, the default results in 64 refined subtriangles for each
+            triangle of the initial triangulation.
+
+        Returns
+        -------
+        refi_triangulation : `~matplotlib.tri.Triangulation`
+            The refined triangulation.
+        found_index : int array
+            Index of the initial triangulation containing triangle, for each
+            point of *refi_triangulation*.
+            Returned only if *return_tri_index* is set to True.
+        """
+        refi_triangulation = self._triangulation
+        ntri = refi_triangulation.triangles.shape[0]
+
+        # Computes the triangulation ancestors numbers in the reference
+        # triangulation.
+        ancestors = np.arange(ntri, dtype=np.int32)
+        for _ in range(subdiv):
+            refi_triangulation, ancestors = self._refine_triangulation_once(
+                refi_triangulation, ancestors)
+        refi_npts = refi_triangulation.x.shape[0]
+        refi_triangles = refi_triangulation.triangles
+
+        # Now we compute found_index table if needed
+        if return_tri_index:
+            # We have to initialize found_index with -1 because some nodes
+            # may very well belong to no triangle at all, e.g., in case of
+            # Delaunay Triangulation with DuplicatePointWarning.
+            found_index = np.full(refi_npts, -1, dtype=np.int32)
+            tri_mask = self._triangulation.mask
+            if tri_mask is None:
+                found_index[refi_triangles] = np.repeat(ancestors,
+                                                        3).reshape(-1, 3)
+            else:
+                # There is a subtlety here: we want to avoid whenever possible
+                # that refined points container is a masked triangle (which
+                # would result in artifacts in plots).
+                # So we impose the numbering from masked ancestors first,
+                # then overwrite it with unmasked ancestor numbers.
+                ancestor_mask = tri_mask[ancestors]
+                found_index[refi_triangles[ancestor_mask, :]
+                            ] = np.repeat(ancestors[ancestor_mask],
+                                          3).reshape(-1, 3)
+                found_index[refi_triangles[~ancestor_mask, :]
+                            ] = np.repeat(ancestors[~ancestor_mask],
+                                          3).reshape(-1, 3)
+            return refi_triangulation, found_index
+        else:
+            return refi_triangulation
+
+    def refine_field(self, z, triinterpolator=None, subdiv=3):
+        """
+        Refine a field defined on the encapsulated triangulation.
+
+        Parameters
+        ----------
+        z : (npoints,) array-like
+            Values of the field to refine, defined at the nodes of the
+            encapsulated triangulation. (``n_points`` is the number of points
+            in the initial triangulation)
+        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
+            Interpolator used for field interpolation. If not specified,
+            a `~matplotlib.tri.CubicTriInterpolator` will be used.
+        subdiv : int, default: 3
+            Recursion level for the subdivision.
+            Each triangle is divided into ``4**subdiv`` child triangles.
+
+        Returns
+        -------
+        refi_tri : `~matplotlib.tri.Triangulation`
+             The returned refined triangulation.
+        refi_z : 1D array of length: *refi_tri* node count.
+             The returned interpolated field (at *refi_tri* nodes).
+        """
+        if triinterpolator is None:
+            interp = matplotlib.tri.CubicTriInterpolator(
+                self._triangulation, z)
+        else:
+            _api.check_isinstance(matplotlib.tri.TriInterpolator,
+                                  triinterpolator=triinterpolator)
+            interp = triinterpolator
+
+        refi_tri, found_index = self.refine_triangulation(
+            subdiv=subdiv, return_tri_index=True)
+        refi_z = interp._interpolate_multikeys(
+            refi_tri.x, refi_tri.y, tri_index=found_index)[0]
+        return refi_tri, refi_z
+
+    @staticmethod
+    def _refine_triangulation_once(triangulation, ancestors=None):
+        """
+        Refine a `.Triangulation` by splitting each triangle into 4
+        child-masked_triangles built on the edges midside nodes.
+
+        Masked triangles, if present, are also split, but their children
+        returned masked.
+
+        If *ancestors* is not provided, returns only a new triangulation:
+        child_triangulation.
+
+        If the array-like key table *ancestor* is given, it shall be of shape
+        (ntri,) where ntri is the number of *triangulation* masked_triangles.
+        In this case, the function returns
+        (child_triangulation, child_ancestors)
+        child_ancestors is defined so that the 4 child masked_triangles share
+        the same index as their father: child_ancestors.shape = (4 * ntri,).
+        """
+
+        x = triangulation.x
+        y = triangulation.y
+
+        #    According to tri.triangulation doc:
+        #         neighbors[i, j] is the triangle that is the neighbor
+        #         to the edge from point index masked_triangles[i, j] to point
+        #         index masked_triangles[i, (j+1)%3].
+        neighbors = triangulation.neighbors
+        triangles = triangulation.triangles
+        npts = np.shape(x)[0]
+        ntri = np.shape(triangles)[0]
+        if ancestors is not None:
+            ancestors = np.asarray(ancestors)
+            if np.shape(ancestors) != (ntri,):
+                raise ValueError(
+                    "Incompatible shapes provide for triangulation"
+                    ".masked_triangles and ancestors: {0} and {1}".format(
+                        np.shape(triangles), np.shape(ancestors)))
+
+        # Initiating tables refi_x and refi_y of the refined triangulation
+        # points
+        # hint: each apex is shared by 2 masked_triangles except the borders.
+        borders = np.sum(neighbors == -1)
+        added_pts = (3*ntri + borders) // 2
+        refi_npts = npts + added_pts
+        refi_x = np.zeros(refi_npts)
+        refi_y = np.zeros(refi_npts)
+
+        # First part of refi_x, refi_y is just the initial points
+        refi_x[:npts] = x
+        refi_y[:npts] = y
+
+        # Second part contains the edge midside nodes.
+        # Each edge belongs to 1 triangle (if border edge) or is shared by 2
+        # masked_triangles (interior edge).
+        # We first build 2 * ntri arrays of edge starting nodes (edge_elems,
+        # edge_apexes); we then extract only the masters to avoid overlaps.
+        # The so-called 'master' is the triangle with biggest index
+        # The 'slave' is the triangle with lower index
+        # (can be -1 if border edge)
+        # For slave and master we will identify the apex pointing to the edge
+        # start
+        edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
+        edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
+        edge_neighbors = neighbors[edge_elems, edge_apexes]
+        mask_masters = (edge_elems > edge_neighbors)
+
+        # Identifying the "masters" and adding to refi_x, refi_y vec
+        masters = edge_elems[mask_masters]
+        apex_masters = edge_apexes[mask_masters]
+        x_add = (x[triangles[masters, apex_masters]] +
+                 x[triangles[masters, (apex_masters+1) % 3]]) * 0.5
+        y_add = (y[triangles[masters, apex_masters]] +
+                 y[triangles[masters, (apex_masters+1) % 3]]) * 0.5
+        refi_x[npts:] = x_add
+        refi_y[npts:] = y_add
+
+        # Building the new masked_triangles; each old masked_triangles hosts
+        # 4 new masked_triangles
+        # there are 6 pts to identify per 'old' triangle, 3 new_pt_corner and
+        # 3 new_pt_midside
+        new_pt_corner = triangles
+
+        # What is the index in refi_x, refi_y of point at middle of apex iapex
+        #  of elem ielem ?
+        # If ielem is the apex master: simple count, given the way refi_x was
+        #  built.
+        # If ielem is the apex slave: yet we do not know; but we will soon
+        # using the neighbors table.
+        new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
+        cum_sum = npts
+        for imid in range(3):
+            mask_st_loc = (imid == apex_masters)
+            n_masters_loc = np.sum(mask_st_loc)
+            elem_masters_loc = masters[mask_st_loc]
+            new_pt_midside[:, imid][elem_masters_loc] = np.arange(
+                n_masters_loc, dtype=np.int32) + cum_sum
+            cum_sum += n_masters_loc
+
+        # Now dealing with slave elems.
+        # for each slave element we identify the master and then the inode
+        # once slave_masters is identified, slave_masters_apex is such that:
+        # neighbors[slaves_masters, slave_masters_apex] == slaves
+        mask_slaves = np.logical_not(mask_masters)
+        slaves = edge_elems[mask_slaves]
+        slaves_masters = edge_neighbors[mask_slaves]
+        diff_table = np.abs(neighbors[slaves_masters, :] -
+                            np.outer(slaves, np.ones(3, dtype=np.int32)))
+        slave_masters_apex = np.argmin(diff_table, axis=1)
+        slaves_apex = edge_apexes[mask_slaves]
+        new_pt_midside[slaves, slaves_apex] = new_pt_midside[
+            slaves_masters, slave_masters_apex]
+
+        # Builds the 4 child masked_triangles
+        child_triangles = np.empty([ntri*4, 3], dtype=np.int32)
+        child_triangles[0::4, :] = np.vstack([
+            new_pt_corner[:, 0], new_pt_midside[:, 0],
+            new_pt_midside[:, 2]]).T
+        child_triangles[1::4, :] = np.vstack([
+            new_pt_corner[:, 1], new_pt_midside[:, 1],
+            new_pt_midside[:, 0]]).T
+        child_triangles[2::4, :] = np.vstack([
+            new_pt_corner[:, 2], new_pt_midside[:, 2],
+            new_pt_midside[:, 1]]).T
+        child_triangles[3::4, :] = np.vstack([
+            new_pt_midside[:, 0], new_pt_midside[:, 1],
+            new_pt_midside[:, 2]]).T
+        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
+
+        # Builds the child mask
+        if triangulation.mask is not None:
+            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
+
+        if ancestors is None:
+            return child_triangulation
+        else:
+            return child_triangulation, np.repeat(ancestors, 4)
diff --git a/lib/matplotlib/tri/_tritools.py b/lib/matplotlib/tri/_tritools.py
new file mode 100644
--- /dev/null
+++ b/lib/matplotlib/tri/_tritools.py
@@ -0,0 +1,263 @@
+"""
+Tools for triangular grids.
+"""
+
+import numpy as np
+
+from matplotlib import _api
+from matplotlib.tri import Triangulation
+
+
+class TriAnalyzer:
+    """
+    Define basic tools for triangular mesh analysis and improvement.
+
+    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
+    tools for mesh analysis and mesh improvement.
+
+    Attributes
+    ----------
+    scale_factors
+
+    Parameters
+    ----------
+    triangulation : `~matplotlib.tri.Triangulation`
+        The encapsulated triangulation to analyze.
+    """
+
+    def __init__(self, triangulation):
+        _api.check_isinstance(Triangulation, triangulation=triangulation)
+        self._triangulation = triangulation
+
+    @property
+    def scale_factors(self):
+        """
+        Factors to rescale the triangulation into a unit square.
+
+        Returns
+        -------
+        (float, float)
+            Scaling factors (kx, ky) so that the triangulation
+            ``[triangulation.x * kx, triangulation.y * ky]``
+            fits exactly inside a unit square.
+        """
+        compressed_triangles = self._triangulation.get_masked_triangles()
+        node_used = (np.bincount(np.ravel(compressed_triangles),
+                                 minlength=self._triangulation.x.size) != 0)
+        return (1 / np.ptp(self._triangulation.x[node_used]),
+                1 / np.ptp(self._triangulation.y[node_used]))
+
+    def circle_ratios(self, rescale=True):
+        """
+        Return a measure of the triangulation triangles flatness.
+
+        The ratio of the incircle radius over the circumcircle radius is a
+        widely used indicator of a triangle flatness.
+        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
+        triangles. Circle ratios below 0.01 denote very flat triangles.
+
+        To avoid unduly low values due to a difference of scale between the 2
+        axis, the triangular mesh can first be rescaled to fit inside a unit
+        square with `scale_factors` (Only if *rescale* is True, which is
+        its default value).
+
+        Parameters
+        ----------
+        rescale : bool, default: True
+            If True, internally rescale (based on `scale_factors`), so that the
+            (unmasked) triangles fit exactly inside a unit square mesh.
+
+        Returns
+        -------
+        masked array
+            Ratio of the incircle radius over the circumcircle radius, for
+            each 'rescaled' triangle of the encapsulated triangulation.
+            Values corresponding to masked triangles are masked out.
+
+        """
+        # Coords rescaling
+        if rescale:
+            (kx, ky) = self.scale_factors
+        else:
+            (kx, ky) = (1.0, 1.0)
+        pts = np.vstack([self._triangulation.x*kx,
+                         self._triangulation.y*ky]).T
+        tri_pts = pts[self._triangulation.triangles]
+        # Computes the 3 side lengths
+        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
+        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
+        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
+        a = np.hypot(a[:, 0], a[:, 1])
+        b = np.hypot(b[:, 0], b[:, 1])
+        c = np.hypot(c[:, 0], c[:, 1])
+        # circumcircle and incircle radii
+        s = (a+b+c)*0.5
+        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
+        # We have to deal with flat triangles with infinite circum_radius
+        bool_flat = (prod == 0.)
+        if np.any(bool_flat):
+            # Pathologic flow
+            ntri = tri_pts.shape[0]
+            circum_radius = np.empty(ntri, dtype=np.float64)
+            circum_radius[bool_flat] = np.inf
+            abc = a*b*c
+            circum_radius[~bool_flat] = abc[~bool_flat] / (
+                4.0*np.sqrt(prod[~bool_flat]))
+        else:
+            # Normal optimized flow
+            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
+        in_radius = (a*b*c) / (4.0*circum_radius*s)
+        circle_ratio = in_radius/circum_radius
+        mask = self._triangulation.mask
+        if mask is None:
+            return circle_ratio
+        else:
+            return np.ma.array(circle_ratio, mask=mask)
+
+    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
+        """
+        Eliminate excessively flat border triangles from the triangulation.
+
+        Returns a mask *new_mask* which allows to clean the encapsulated
+        triangulation from its border-located flat triangles
+        (according to their :meth:`circle_ratios`).
+        This mask is meant to be subsequently applied to the triangulation
+        using `.Triangulation.set_mask`.
+        *new_mask* is an extension of the initial triangulation mask
+        in the sense that an initially masked triangle will remain masked.
+
+        The *new_mask* array is computed recursively; at each step flat
+        triangles are removed only if they share a side with the current mesh
+        border. Thus no new holes in the triangulated domain will be created.
+
+        Parameters
+        ----------
+        min_circle_ratio : float, default: 0.01
+            Border triangles with incircle/circumcircle radii ratio r/R will
+            be removed if r/R < *min_circle_ratio*.
+        rescale : bool, default: True
+            If True, first, internally rescale (based on `scale_factors`) so
+            that the (unmasked) triangles fit exactly inside a unit square
+            mesh.  This rescaling accounts for the difference of scale which
+            might exist between the 2 axis.
+
+        Returns
+        -------
+        array of bool
+            Mask to apply to encapsulated triangulation.
+            All the initially masked triangles remain masked in the
+            *new_mask*.
+
+        Notes
+        -----
+        The rationale behind this function is that a Delaunay
+        triangulation - of an unstructured set of points - sometimes contains
+        almost flat triangles at its border, leading to artifacts in plots
+        (especially for high-resolution contouring).
+        Masked with computed *new_mask*, the encapsulated
+        triangulation would contain no more unmasked border triangles
+        with a circle ratio below *min_circle_ratio*, thus improving the
+        mesh quality for subsequent plots or interpolation.
+        """
+        # Recursively computes the mask_current_borders, true if a triangle is
+        # at the border of the mesh OR touching the border through a chain of
+        # invalid aspect ratio masked_triangles.
+        ntri = self._triangulation.triangles.shape[0]
+        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
+
+        current_mask = self._triangulation.mask
+        if current_mask is None:
+            current_mask = np.zeros(ntri, dtype=bool)
+        valid_neighbors = np.copy(self._triangulation.neighbors)
+        renum_neighbors = np.arange(ntri, dtype=np.int32)
+        nadd = -1
+        while nadd != 0:
+            # The active wavefront is the triangles from the border (unmasked
+            # but with a least 1 neighbor equal to -1
+            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
+            # The element from the active wavefront will be masked if their
+            # circle ratio is bad.
+            added_mask = wavefront & mask_bad_ratio
+            current_mask = added_mask | current_mask
+            nadd = np.sum(added_mask)
+
+            # now we have to update the tables valid_neighbors
+            valid_neighbors[added_mask, :] = -1
+            renum_neighbors[added_mask] = -1
+            valid_neighbors = np.where(valid_neighbors == -1, -1,
+                                       renum_neighbors[valid_neighbors])
+
+        return np.ma.filled(current_mask, True)
+
+    def _get_compressed_triangulation(self):
+        """
+        Compress (if masked) the encapsulated triangulation.
+
+        Returns minimal-length triangles array (*compressed_triangles*) and
+        coordinates arrays (*compressed_x*, *compressed_y*) that can still
+        describe the unmasked triangles of the encapsulated triangulation.
+
+        Returns
+        -------
+        compressed_triangles : array-like
+            the returned compressed triangulation triangles
+        compressed_x : array-like
+            the returned compressed triangulation 1st coordinate
+        compressed_y : array-like
+            the returned compressed triangulation 2nd coordinate
+        tri_renum : int array
+            renumbering table to translate the triangle numbers from the
+            encapsulated triangulation into the new (compressed) renumbering.
+            -1 for masked triangles (deleted from *compressed_triangles*).
+        node_renum : int array
+            renumbering table to translate the point numbers from the
+            encapsulated triangulation into the new (compressed) renumbering.
+            -1 for unused points (i.e. those deleted from *compressed_x* and
+            *compressed_y*).
+
+        """
+        # Valid triangles and renumbering
+        tri_mask = self._triangulation.mask
+        compressed_triangles = self._triangulation.get_masked_triangles()
+        ntri = self._triangulation.triangles.shape[0]
+        if tri_mask is not None:
+            tri_renum = self._total_to_compress_renum(~tri_mask)
+        else:
+            tri_renum = np.arange(ntri, dtype=np.int32)
+
+        # Valid nodes and renumbering
+        valid_node = (np.bincount(np.ravel(compressed_triangles),
+                                  minlength=self._triangulation.x.size) != 0)
+        compressed_x = self._triangulation.x[valid_node]
+        compressed_y = self._triangulation.y[valid_node]
+        node_renum = self._total_to_compress_renum(valid_node)
+
+        # Now renumbering the valid triangles nodes
+        compressed_triangles = node_renum[compressed_triangles]
+
+        return (compressed_triangles, compressed_x, compressed_y, tri_renum,
+                node_renum)
+
+    @staticmethod
+    def _total_to_compress_renum(valid):
+        """
+        Parameters
+        ----------
+        valid : 1D bool array
+            Validity mask.
+
+        Returns
+        -------
+        int array
+            Array so that (`valid_array` being a compressed array
+            based on a `masked_array` with mask ~*valid*):
+
+            - For all i with valid[i] = True:
+              valid_array[renum[i]] = masked_array[i]
+            - For all i with valid[i] = False:
+              renum[i] = -1 (invalid value)
+        """
+        renum = np.full(np.size(valid), -1, dtype=np.int32)
+        n_valid = np.sum(valid)
+        renum[valid] = np.arange(n_valid, dtype=np.int32)
+        return renum
diff --git a/lib/matplotlib/tri/triangulation.py b/lib/matplotlib/tri/triangulation.py
--- a/lib/matplotlib/tri/triangulation.py
+++ b/lib/matplotlib/tri/triangulation.py
@@ -1,240 +1,9 @@
-import numpy as np
-
+from ._triangulation import *  # noqa: F401, F403
 from matplotlib import _api
 
 
-class Triangulation:
-    """
-    An unstructured triangular grid consisting of npoints points and
-    ntri triangles.  The triangles can either be specified by the user
-    or automatically generated using a Delaunay triangulation.
-
-    Parameters
-    ----------
-    x, y : (npoints,) array-like
-        Coordinates of grid points.
-    triangles : (ntri, 3) array-like of int, optional
-        For each triangle, the indices of the three points that make
-        up the triangle, ordered in an anticlockwise manner.  If not
-        specified, the Delaunay triangulation is calculated.
-    mask : (ntri,) array-like of bool, optional
-        Which triangles are masked out.
-
-    Attributes
-    ----------
-    triangles : (ntri, 3) array of int
-        For each triangle, the indices of the three points that make
-        up the triangle, ordered in an anticlockwise manner. If you want to
-        take the *mask* into account, use `get_masked_triangles` instead.
-    mask : (ntri, 3) array of bool
-        Masked out triangles.
-    is_delaunay : bool
-        Whether the Triangulation is a calculated Delaunay
-        triangulation (where *triangles* was not specified) or not.
-
-    Notes
-    -----
-    For a Triangulation to be valid it must not have duplicate points,
-    triangles formed from colinear points, or overlapping triangles.
-    """
-    def __init__(self, x, y, triangles=None, mask=None):
-        from matplotlib import _qhull
-
-        self.x = np.asarray(x, dtype=np.float64)
-        self.y = np.asarray(y, dtype=np.float64)
-        if self.x.shape != self.y.shape or self.x.ndim != 1:
-            raise ValueError("x and y must be equal-length 1D arrays, but "
-                             f"found shapes {self.x.shape!r} and "
-                             f"{self.y.shape!r}")
-
-        self.mask = None
-        self._edges = None
-        self._neighbors = None
-        self.is_delaunay = False
-
-        if triangles is None:
-            # No triangulation specified, so use matplotlib._qhull to obtain
-            # Delaunay triangulation.
-            self.triangles, self._neighbors = _qhull.delaunay(x, y)
-            self.is_delaunay = True
-        else:
-            # Triangulation specified. Copy, since we may correct triangle
-            # orientation.
-            try:
-                self.triangles = np.array(triangles, dtype=np.int32, order='C')
-            except ValueError as e:
-                raise ValueError('triangles must be a (N, 3) int array, not '
-                                 f'{triangles!r}') from e
-            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
-                raise ValueError(
-                    'triangles must be a (N, 3) int array, but found shape '
-                    f'{self.triangles.shape!r}')
-            if self.triangles.max() >= len(self.x):
-                raise ValueError(
-                    'triangles are indices into the points and must be in the '
-                    f'range 0 <= i < {len(self.x)} but found value '
-                    f'{self.triangles.max()}')
-            if self.triangles.min() < 0:
-                raise ValueError(
-                    'triangles are indices into the points and must be in the '
-                    f'range 0 <= i < {len(self.x)} but found value '
-                    f'{self.triangles.min()}')
-
-        # Underlying C++ object is not created until first needed.
-        self._cpp_triangulation = None
-
-        # Default TriFinder not created until needed.
-        self._trifinder = None
-
-        self.set_mask(mask)
-
-    def calculate_plane_coefficients(self, z):
-        """
-        Calculate plane equation coefficients for all unmasked triangles from
-        the point (x, y) coordinates and specified z-array of shape (npoints).
-        The returned array has shape (npoints, 3) and allows z-value at (x, y)
-        position in triangle tri to be calculated using
-        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
-        """
-        return self.get_cpp_triangulation().calculate_plane_coefficients(z)
-
-    @property
-    def edges(self):
-        """
-        Return integer array of shape (nedges, 2) containing all edges of
-        non-masked triangles.
-
-        Each row defines an edge by its start point index and end point
-        index.  Each edge appears only once, i.e. for an edge between points
-        *i*  and *j*, there will only be either *(i, j)* or *(j, i)*.
-        """
-        if self._edges is None:
-            self._edges = self.get_cpp_triangulation().get_edges()
-        return self._edges
-
-    def get_cpp_triangulation(self):
-        """
-        Return the underlying C++ Triangulation object, creating it
-        if necessary.
-        """
-        from matplotlib import _tri
-        if self._cpp_triangulation is None:
-            self._cpp_triangulation = _tri.Triangulation(
-                self.x, self.y, self.triangles, self.mask, self._edges,
-                self._neighbors, not self.is_delaunay)
-        return self._cpp_triangulation
-
-    def get_masked_triangles(self):
-        """
-        Return an array of triangles taking the mask into account.
-        """
-        if self.mask is not None:
-            return self.triangles[~self.mask]
-        else:
-            return self.triangles
-
-    @staticmethod
-    def get_from_args_and_kwargs(*args, **kwargs):
-        """
-        Return a Triangulation object from the args and kwargs, and
-        the remaining args and kwargs with the consumed values removed.
-
-        There are two alternatives: either the first argument is a
-        Triangulation object, in which case it is returned, or the args
-        and kwargs are sufficient to create a new Triangulation to
-        return.  In the latter case, see Triangulation.__init__ for
-        the possible args and kwargs.
-        """
-        if isinstance(args[0], Triangulation):
-            triangulation, *args = args
-            if 'triangles' in kwargs:
-                _api.warn_external(
-                    "Passing the keyword 'triangles' has no effect when also "
-                    "passing a Triangulation")
-            if 'mask' in kwargs:
-                _api.warn_external(
-                    "Passing the keyword 'mask' has no effect when also "
-                    "passing a Triangulation")
-        else:
-            x, y, triangles, mask, args, kwargs = \
-                Triangulation._extract_triangulation_params(args, kwargs)
-            triangulation = Triangulation(x, y, triangles, mask)
-        return triangulation, args, kwargs
-
-    @staticmethod
-    def _extract_triangulation_params(args, kwargs):
-        x, y, *args = args
-        # Check triangles in kwargs then args.
-        triangles = kwargs.pop('triangles', None)
-        from_args = False
-        if triangles is None and args:
-            triangles = args[0]
-            from_args = True
-        if triangles is not None:
-            try:
-                triangles = np.asarray(triangles, dtype=np.int32)
-            except ValueError:
-                triangles = None
-        if triangles is not None and (triangles.ndim != 2 or
-                                      triangles.shape[1] != 3):
-            triangles = None
-        if triangles is not None and from_args:
-            args = args[1:]  # Consumed first item in args.
-        # Check for mask in kwargs.
-        mask = kwargs.pop('mask', None)
-        return x, y, triangles, mask, args, kwargs
-
-    def get_trifinder(self):
-        """
-        Return the default `matplotlib.tri.TriFinder` of this
-        triangulation, creating it if necessary.  This allows the same
-        TriFinder object to be easily shared.
-        """
-        if self._trifinder is None:
-            # Default TriFinder class.
-            from matplotlib.tri.trifinder import TrapezoidMapTriFinder
-            self._trifinder = TrapezoidMapTriFinder(self)
-        return self._trifinder
-
-    @property
-    def neighbors(self):
-        """
-        Return integer array of shape (ntri, 3) containing neighbor triangles.
-
-        For each triangle, the indices of the three triangles that
-        share the same edges, or -1 if there is no such neighboring
-        triangle.  ``neighbors[i, j]`` is the triangle that is the neighbor
-        to the edge from point index ``triangles[i, j]`` to point index
-        ``triangles[i, (j+1)%3]``.
-        """
-        if self._neighbors is None:
-            self._neighbors = self.get_cpp_triangulation().get_neighbors()
-        return self._neighbors
-
-    def set_mask(self, mask):
-        """
-        Set or clear the mask array.
-
-        Parameters
-        ----------
-        mask : None or bool array of length ntri
-        """
-        if mask is None:
-            self.mask = None
-        else:
-            self.mask = np.asarray(mask, dtype=bool)
-            if self.mask.shape != (self.triangles.shape[0],):
-                raise ValueError('mask array must have same length as '
-                                 'triangles array')
-
-        # Set mask in C++ Triangulation.
-        if self._cpp_triangulation is not None:
-            self._cpp_triangulation.set_mask(self.mask)
-
-        # Clear derived fields so they are recalculated when needed.
-        self._edges = None
-        self._neighbors = None
-
-        # Recalculate TriFinder if it exists.
-        if self._trifinder is not None:
-            self._trifinder._initialize()
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/tricontour.py b/lib/matplotlib/tri/tricontour.py
--- a/lib/matplotlib/tri/tricontour.py
+++ b/lib/matplotlib/tri/tricontour.py
@@ -1,271 +1,9 @@
-import numpy as np
+from ._tricontour import *  # noqa: F401, F403
+from matplotlib import _api
 
-from matplotlib import _docstring
-from matplotlib.contour import ContourSet
-from matplotlib.tri.triangulation import Triangulation
 
-
-@_docstring.dedent_interpd
-class TriContourSet(ContourSet):
-    """
-    Create and store a set of contour lines or filled regions for
-    a triangular grid.
-
-    This class is typically not instantiated directly by the user but by
-    `~.Axes.tricontour` and `~.Axes.tricontourf`.
-
-    %(contour_set_attributes)s
-    """
-    def __init__(self, ax, *args, **kwargs):
-        """
-        Draw triangular grid contour lines or filled regions,
-        depending on whether keyword arg *filled* is False
-        (default) or True.
-
-        The first argument of the initializer must be an `~.axes.Axes`
-        object.  The remaining arguments and keyword arguments
-        are described in the docstring of `~.Axes.tricontour`.
-        """
-        super().__init__(ax, *args, **kwargs)
-
-    def _process_args(self, *args, **kwargs):
-        """
-        Process args and kwargs.
-        """
-        if isinstance(args[0], TriContourSet):
-            C = args[0]._contour_generator
-            if self.levels is None:
-                self.levels = args[0].levels
-            self.zmin = args[0].zmin
-            self.zmax = args[0].zmax
-            self._mins = args[0]._mins
-            self._maxs = args[0]._maxs
-        else:
-            from matplotlib import _tri
-            tri, z = self._contour_args(args, kwargs)
-            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
-            self._mins = [tri.x.min(), tri.y.min()]
-            self._maxs = [tri.x.max(), tri.y.max()]
-
-        self._contour_generator = C
-        return kwargs
-
-    def _contour_args(self, args, kwargs):
-        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args,
-                                                                   **kwargs)
-        z = np.ma.asarray(args[0])
-        if z.shape != tri.x.shape:
-            raise ValueError('z array must have same length as triangulation x'
-                             ' and y arrays')
-
-        # z values must be finite, only need to check points that are included
-        # in the triangulation.
-        z_check = z[np.unique(tri.get_masked_triangles())]
-        if np.ma.is_masked(z_check):
-            raise ValueError('z must not contain masked points within the '
-                             'triangulation')
-        if not np.isfinite(z_check).all():
-            raise ValueError('z array must not contain non-finite values '
-                             'within the triangulation')
-
-        z = np.ma.masked_invalid(z, copy=False)
-        self.zmax = float(z_check.max())
-        self.zmin = float(z_check.min())
-        if self.logscale and self.zmin <= 0:
-            func = 'contourf' if self.filled else 'contour'
-            raise ValueError(f'Cannot {func} log of negative values.')
-        self._process_contour_level_args(args[1:])
-        return (tri, z)
-
-
-_docstring.interpd.update(_tricontour_doc="""
-Draw contour %%(type)s on an unstructured triangular grid.
-
-Call signatures::
-
-    %%(func)s(triangulation, z, [levels], ...)
-    %%(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)
-
-The triangular grid can be specified either by passing a `.Triangulation`
-object as the first parameter, or by passing the points *x*, *y* and
-optionally the *triangles* and a *mask*. See `.Triangulation` for an
-explanation of these parameters. If neither of *triangulation* or
-*triangles* are given, the triangulation is calculated on the fly.
-
-It is possible to pass *triangles* positionally, i.e.
-``%%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more
-clarity, pass *triangles* via keyword argument.
-
-Parameters
-----------
-triangulation : `.Triangulation`, optional
-    An already created triangular grid.
-
-x, y, triangles, mask
-    Parameters defining the triangular grid. See `.Triangulation`.
-    This is mutually exclusive with specifying *triangulation*.
-
-z : array-like
-    The height values over which the contour is drawn.  Color-mapping is
-    controlled by *cmap*, *norm*, *vmin*, and *vmax*.
-
-    .. note::
-        All values in *z* must be finite. Hence, nan and inf values must
-        either be removed or `~.Triangulation.set_mask` be used.
-
-levels : int or array-like, optional
-    Determines the number and positions of the contour lines / regions.
-
-    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
-    automatically choose no more than *n+1* "nice" contour levels between
-    between minimum and maximum numeric values of *Z*.
-
-    If array-like, draw contour lines at the specified levels.  The values must
-    be in increasing order.
-
-Returns
--------
-`~matplotlib.tri.TriContourSet`
-
-Other Parameters
-----------------
-colors : color string or sequence of colors, optional
-    The colors of the levels, i.e., the contour %%(type)s.
-
-    The sequence is cycled for the levels in ascending order. If the sequence
-    is shorter than the number of levels, it is repeated.
-
-    As a shortcut, single color strings may be used in place of one-element
-    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
-    same color. This shortcut does only work for color strings, not for other
-    ways of specifying colors.
-
-    By default (value *None*), the colormap specified by *cmap* will be used.
-
-alpha : float, default: 1
-    The alpha blending value, between 0 (transparent) and 1 (opaque).
-
-%(cmap_doc)s
-
-    This parameter is ignored if *colors* is set.
-
-%(norm_doc)s
-
-    This parameter is ignored if *colors* is set.
-
-%(vmin_vmax_doc)s
-
-    If *vmin* or *vmax* are not given, the default color scaling is based on
-    *levels*.
-
-    This parameter is ignored if *colors* is set.
-
-origin : {*None*, 'upper', 'lower', 'image'}, default: None
-    Determines the orientation and exact position of *z* by specifying the
-    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.
-
-    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.
-    - 'lower': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
-    - 'upper': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
-    - 'image': Use the value from :rc:`image.origin`.
-
-extent : (x0, x1, y0, y1), optional
-    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
-    gives the outer pixel boundaries. In this case, the position of z[0, 0] is
-    the center of the pixel, not a corner. If *origin* is *None*, then
-    (*x0*, *y0*) is the position of z[0, 0], and (*x1*, *y1*) is the position
-    of z[-1, -1].
-
-    This argument is ignored if *X* and *Y* are specified in the call to
-    contour.
-
-locator : ticker.Locator subclass, optional
-    The locator is used to determine the contour levels if they are not given
-    explicitly via *levels*.
-    Defaults to `~.ticker.MaxNLocator`.
-
-extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
-    Determines the ``%%(func)s``-coloring of values that are outside the
-    *levels* range.
-
-    If 'neither', values outside the *levels* range are not colored.  If 'min',
-    'max' or 'both', color the values below, above or below and above the
-    *levels* range.
-
-    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
-    under/over values of the `.Colormap`. Note that most colormaps do not have
-    dedicated colors for these by default, so that the over and under values
-    are the edge values of the colormap.  You may want to set these values
-    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.
-
-    .. note::
-
-        An existing `.TriContourSet` does not get notified if properties of its
-        colormap are changed. Therefore, an explicit call to
-        `.ContourSet.changed()` is needed after modifying the colormap. The
-        explicit call can be left out, if a colorbar is assigned to the
-        `.TriContourSet` because it internally calls `.ContourSet.changed()`.
-
-xunits, yunits : registered units, optional
-    Override axis units by specifying an instance of a
-    :class:`matplotlib.units.ConversionInterface`.
-
-antialiased : bool, optional
-    Enable antialiasing, overriding the defaults.  For
-    filled contours, the default is *True*.  For line contours,
-    it is taken from :rc:`lines.antialiased`.""" % _docstring.interpd.params)
-
-
-@_docstring.Substitution(func='tricontour', type='lines')
-@_docstring.dedent_interpd
-def tricontour(ax, *args, **kwargs):
-    """
-    %(_tricontour_doc)s
-
-    linewidths : float or array-like, default: :rc:`contour.linewidth`
-        The line width of the contour lines.
-
-        If a number, all levels will be plotted with this linewidth.
-
-        If a sequence, the levels in ascending order will be plotted with
-        the linewidths in the order specified.
-
-        If None, this falls back to :rc:`lines.linewidth`.
-
-    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
-        If *linestyles* is *None*, the default is 'solid' unless the lines are
-        monochrome.  In that case, negative contours will take their linestyle
-        from :rc:`contour.negative_linestyle` setting.
-
-        *linestyles* can also be an iterable of the above strings specifying a
-        set of linestyles to be used. If this iterable is shorter than the
-        number of contour levels it will be repeated as necessary.
-    """
-    kwargs['filled'] = False
-    return TriContourSet(ax, *args, **kwargs)
-
-
-@_docstring.Substitution(func='tricontourf', type='regions')
-@_docstring.dedent_interpd
-def tricontourf(ax, *args, **kwargs):
-    """
-    %(_tricontour_doc)s
-
-    hatches : list[str], optional
-        A list of cross hatch patterns to use on the filled areas.
-        If None, no hatching will be added to the contour.
-        Hatching is supported in the PostScript, PDF, SVG and Agg
-        backends only.
-
-    Notes
-    -----
-    `.tricontourf` fills intervals that are closed at the top; that is, for
-    boundaries *z1* and *z2*, the filled region is::
-
-        z1 < Z <= z2
-
-    except for the lowest interval, which is closed on both sides (i.e. it
-    includes the lowest value).
-    """
-    kwargs['filled'] = True
-    return TriContourSet(ax, *args, **kwargs)
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/trifinder.py b/lib/matplotlib/tri/trifinder.py
--- a/lib/matplotlib/tri/trifinder.py
+++ b/lib/matplotlib/tri/trifinder.py
@@ -1,93 +1,9 @@
-import numpy as np
-
+from ._trifinder import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri import Triangulation
-
-
-class TriFinder:
-    """
-    Abstract base class for classes used to find the triangles of a
-    Triangulation in which (x, y) points lie.
-
-    Rather than instantiate an object of a class derived from TriFinder, it is
-    usually better to use the function `.Triangulation.get_trifinder`.
-
-    Derived classes implement __call__(x, y) where x and y are array-like point
-    coordinates of the same shape.
-    """
-
-    def __init__(self, triangulation):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-
-class TrapezoidMapTriFinder(TriFinder):
-    """
-    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
-    map algorithm from the book "Computational Geometry, Algorithms and
-    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
-    and O. Schwarzkopf.
-
-    The triangulation must be valid, i.e. it must not have duplicate points,
-    triangles formed from colinear points, or overlapping triangles.  The
-    algorithm has some tolerance to triangles formed from colinear points, but
-    this should not be relied upon.
-    """
-
-    def __init__(self, triangulation):
-        from matplotlib import _tri
-        super().__init__(triangulation)
-        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
-            triangulation.get_cpp_triangulation())
-        self._initialize()
-
-    def __call__(self, x, y):
-        """
-        Return an array containing the indices of the triangles in which the
-        specified *x*, *y* points lie, or -1 for points that do not lie within
-        a triangle.
-
-        *x*, *y* are array-like x and y coordinates of the same shape and any
-        number of dimensions.
-
-        Returns integer array with the same shape and *x* and *y*.
-        """
-        x = np.asarray(x, dtype=np.float64)
-        y = np.asarray(y, dtype=np.float64)
-        if x.shape != y.shape:
-            raise ValueError("x and y must be array-like with the same shape")
-
-        # C++ does the heavy lifting, and expects 1D arrays.
-        indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
-                   .reshape(x.shape))
-        return indices
-
-    def _get_tree_stats(self):
-        """
-        Return a python list containing the statistics about the node tree:
-            0: number of nodes (tree size)
-            1: number of unique nodes
-            2: number of trapezoids (tree leaf nodes)
-            3: number of unique trapezoids
-            4: maximum parent count (max number of times a node is repeated in
-                   tree)
-            5: maximum depth of tree (one more than the maximum number of
-                   comparisons needed to search through the tree)
-            6: mean of all trapezoid depths (one more than the average number
-                   of comparisons needed to search through the tree)
-        """
-        return self._cpp_trifinder.get_tree_stats()
 
-    def _initialize(self):
-        """
-        Initialize the underlying C++ object.  Can be called multiple times if,
-        for example, the triangulation is modified.
-        """
-        self._cpp_trifinder.initialize()
 
-    def _print_tree(self):
-        """
-        Print a text representation of the node tree, which is useful for
-        debugging purposes.
-        """
-        self._cpp_trifinder.print_tree()
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/triinterpolate.py b/lib/matplotlib/tri/triinterpolate.py
--- a/lib/matplotlib/tri/triinterpolate.py
+++ b/lib/matplotlib/tri/triinterpolate.py
@@ -1,1574 +1,9 @@
-"""
-Interpolation inside triangular grids.
-"""
-
-import numpy as np
-
+from ._triinterpolate import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri import Triangulation
-from matplotlib.tri.trifinder import TriFinder
-from matplotlib.tri.tritools import TriAnalyzer
-
-__all__ = ('TriInterpolator', 'LinearTriInterpolator', 'CubicTriInterpolator')
-
-
-class TriInterpolator:
-    """
-    Abstract base class for classes used to interpolate on a triangular grid.
-
-    Derived classes implement the following methods:
-
-    - ``__call__(x, y)``,
-      where x, y are array-like point coordinates of the same shape, and
-      that returns a masked array of the same shape containing the
-      interpolated z-values.
-
-    - ``gradient(x, y)``,
-      where x, y are array-like point coordinates of the same
-      shape, and that returns a list of 2 masked arrays of the same shape
-      containing the 2 derivatives of the interpolator (derivatives of
-      interpolated z values with respect to x and y).
-    """
-
-    def __init__(self, triangulation, z, trifinder=None):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-        self._z = np.asarray(z)
-        if self._z.shape != self._triangulation.x.shape:
-            raise ValueError("z array must have same length as triangulation x"
-                             " and y arrays")
-
-        _api.check_isinstance((TriFinder, None), trifinder=trifinder)
-        self._trifinder = trifinder or self._triangulation.get_trifinder()
-
-        # Default scaling factors : 1.0 (= no scaling)
-        # Scaling may be used for interpolations for which the order of
-        # magnitude of x, y has an impact on the interpolant definition.
-        # Please refer to :meth:`_interpolate_multikeys` for details.
-        self._unit_x = 1.0
-        self._unit_y = 1.0
-
-        # Default triangle renumbering: None (= no renumbering)
-        # Renumbering may be used to avoid unnecessary computations
-        # if complex calculations are done inside the Interpolator.
-        # Please refer to :meth:`_interpolate_multikeys` for details.
-        self._tri_renum = None
-
-    # __call__ and gradient docstrings are shared by all subclasses
-    # (except, if needed, relevant additions).
-    # However these methods are only implemented in subclasses to avoid
-    # confusion in the documentation.
-    _docstring__call__ = """
-        Returns a masked array containing interpolated values at the specified
-        (x, y) points.
-
-        Parameters
-        ----------
-        x, y : array-like
-            x and y coordinates of the same shape and any number of
-            dimensions.
-
-        Returns
-        -------
-        np.ma.array
-            Masked array of the same shape as *x* and *y*; values corresponding
-            to (*x*, *y*) points outside of the triangulation are masked out.
-
-        """
-
-    _docstringgradient = r"""
-        Returns a list of 2 masked arrays containing interpolated derivatives
-        at the specified (x, y) points.
-
-        Parameters
-        ----------
-        x, y : array-like
-            x and y coordinates of the same shape and any number of
-            dimensions.
-
-        Returns
-        -------
-        dzdx, dzdy : np.ma.array
-            2 masked arrays of the same shape as *x* and *y*; values
-            corresponding to (x, y) points outside of the triangulation
-            are masked out.
-            The first returned array contains the values of
-            :math:`\frac{\partial z}{\partial x}` and the second those of
-            :math:`\frac{\partial z}{\partial y}`.
-
-        """
-
-    def _interpolate_multikeys(self, x, y, tri_index=None,
-                               return_keys=('z',)):
-        """
-        Versatile (private) method defined for all TriInterpolators.
-
-        :meth:`_interpolate_multikeys` is a wrapper around method
-        :meth:`_interpolate_single_key` (to be defined in the child
-        subclasses).
-        :meth:`_interpolate_single_key actually performs the interpolation,
-        but only for 1-dimensional inputs and at valid locations (inside
-        unmasked triangles of the triangulation).
-
-        The purpose of :meth:`_interpolate_multikeys` is to implement the
-        following common tasks needed in all subclasses implementations:
-
-        - calculation of containing triangles
-        - dealing with more than one interpolation request at the same
-          location (e.g., if the 2 derivatives are requested, it is
-          unnecessary to compute the containing triangles twice)
-        - scaling according to self._unit_x, self._unit_y
-        - dealing with points outside of the grid (with fill value np.nan)
-        - dealing with multi-dimensional *x*, *y* arrays: flattening for
-          :meth:`_interpolate_params` call and final reshaping.
-
-        (Note that np.vectorize could do most of those things very well for
-        you, but it does it by function evaluations over successive tuples of
-        the input arrays. Therefore, this tends to be more time consuming than
-        using optimized numpy functions - e.g., np.dot - which can be used
-        easily on the flattened inputs, in the child-subclass methods
-        :meth:`_interpolate_single_key`.)
-
-        It is guaranteed that the calls to :meth:`_interpolate_single_key`
-        will be done with flattened (1-d) array-like input parameters *x*, *y*
-        and with flattened, valid `tri_index` arrays (no -1 index allowed).
-
-        Parameters
-        ----------
-        x, y : array-like
-            x and y coordinates where interpolated values are requested.
-        tri_index : array-like of int, optional
-            Array of the containing triangle indices, same shape as
-            *x* and *y*. Defaults to None. If None, these indices
-            will be computed by a TriFinder instance.
-            (Note: For point outside the grid, tri_index[ipt] shall be -1).
-        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
-            Defines the interpolation arrays to return, and in which order.
-
-        Returns
-        -------
-        list of arrays
-            Each array-like contains the expected interpolated values in the
-            order defined by *return_keys* parameter.
-        """
-        # Flattening and rescaling inputs arrays x, y
-        # (initial shape is stored for output)
-        x = np.asarray(x, dtype=np.float64)
-        y = np.asarray(y, dtype=np.float64)
-        sh_ret = x.shape
-        if x.shape != y.shape:
-            raise ValueError("x and y shall have same shapes."
-                             " Given: {0} and {1}".format(x.shape, y.shape))
-        x = np.ravel(x)
-        y = np.ravel(y)
-        x_scaled = x/self._unit_x
-        y_scaled = y/self._unit_y
-        size_ret = np.size(x_scaled)
-
-        # Computes & ravels the element indexes, extract the valid ones.
-        if tri_index is None:
-            tri_index = self._trifinder(x, y)
-        else:
-            if tri_index.shape != sh_ret:
-                raise ValueError(
-                    "tri_index array is provided and shall"
-                    " have same shape as x and y. Given: "
-                    "{0} and {1}".format(tri_index.shape, sh_ret))
-            tri_index = np.ravel(tri_index)
-
-        mask_in = (tri_index != -1)
-        if self._tri_renum is None:
-            valid_tri_index = tri_index[mask_in]
-        else:
-            valid_tri_index = self._tri_renum[tri_index[mask_in]]
-        valid_x = x_scaled[mask_in]
-        valid_y = y_scaled[mask_in]
-
-        ret = []
-        for return_key in return_keys:
-            # Find the return index associated with the key.
-            try:
-                return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
-            except KeyError as err:
-                raise ValueError("return_keys items shall take values in"
-                                 " {'z', 'dzdx', 'dzdy'}") from err
-
-            # Sets the scale factor for f & df components
-            scale = [1., 1./self._unit_x, 1./self._unit_y][return_index]
-
-            # Computes the interpolation
-            ret_loc = np.empty(size_ret, dtype=np.float64)
-            ret_loc[~mask_in] = np.nan
-            ret_loc[mask_in] = self._interpolate_single_key(
-                return_key, valid_tri_index, valid_x, valid_y) * scale
-            ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]
-
-        return ret
-
-    def _interpolate_single_key(self, return_key, tri_index, x, y):
-        """
-        Interpolate at points belonging to the triangulation
-        (inside an unmasked triangles).
-
-        Parameters
-        ----------
-        return_key : {'z', 'dzdx', 'dzdy'}
-            The requested values (z or its derivatives).
-        tri_index : 1D int array
-            Valid triangle index (cannot be -1).
-        x, y : 1D arrays, same shape as `tri_index`
-            Valid locations where interpolation is requested.
-
-        Returns
-        -------
-        1-d array
-            Returned array of the same size as *tri_index*
-        """
-        raise NotImplementedError("TriInterpolator subclasses" +
-                                  "should implement _interpolate_single_key!")
-
-
-class LinearTriInterpolator(TriInterpolator):
-    """
-    Linear interpolator on a triangular grid.
-
-    Each triangle is represented by a plane so that an interpolated value at
-    point (x, y) lies on the plane of the triangle containing (x, y).
-    Interpolated values are therefore continuous across the triangulation, but
-    their first derivatives are discontinuous at edges between triangles.
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The triangulation to interpolate over.
-    z : (npoints,) array-like
-        Array of values, defined at grid points, to interpolate between.
-    trifinder : `~matplotlib.tri.TriFinder`, optional
-        If this is not specified, the Triangulation's default TriFinder will
-        be used by calling `.Triangulation.get_trifinder`.
-
-    Methods
-    -------
-    `__call__` (x, y) : Returns interpolated values at (x, y) points.
-    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
-
-    """
-    def __init__(self, triangulation, z, trifinder=None):
-        super().__init__(triangulation, z, trifinder)
-
-        # Store plane coefficients for fast interpolation calculations.
-        self._plane_coefficients = \
-            self._triangulation.calculate_plane_coefficients(self._z)
-
-    def __call__(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('z',))[0]
-    __call__.__doc__ = TriInterpolator._docstring__call__
-
-    def gradient(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('dzdx', 'dzdy'))
-    gradient.__doc__ = TriInterpolator._docstringgradient
-
-    def _interpolate_single_key(self, return_key, tri_index, x, y):
-        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
-        if return_key == 'z':
-            return (self._plane_coefficients[tri_index, 0]*x +
-                    self._plane_coefficients[tri_index, 1]*y +
-                    self._plane_coefficients[tri_index, 2])
-        elif return_key == 'dzdx':
-            return self._plane_coefficients[tri_index, 0]
-        else:  # 'dzdy'
-            return self._plane_coefficients[tri_index, 1]
-
-
-class CubicTriInterpolator(TriInterpolator):
-    r"""
-    Cubic interpolator on a triangular grid.
-
-    In one-dimension - on a segment - a cubic interpolating function is
-    defined by the values of the function and its derivative at both ends.
-    This is almost the same in 2D inside a triangle, except that the values
-    of the function and its 2 derivatives have to be defined at each triangle
-    node.
-
-    The CubicTriInterpolator takes the value of the function at each node -
-    provided by the user - and internally computes the value of the
-    derivatives, resulting in a smooth interpolation.
-    (As a special feature, the user can also impose the value of the
-    derivatives at each node, but this is not supposed to be the common
-    usage.)
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The triangulation to interpolate over.
-    z : (npoints,) array-like
-        Array of values, defined at grid points, to interpolate between.
-    kind : {'min_E', 'geom', 'user'}, optional
-        Choice of the smoothing algorithm, in order to compute
-        the interpolant derivatives (defaults to 'min_E'):
-
-        - if 'min_E': (default) The derivatives at each node is computed
-          to minimize a bending energy.
-        - if 'geom': The derivatives at each node is computed as a
-          weighted average of relevant triangle normals. To be used for
-          speed optimization (large grids).
-        - if 'user': The user provides the argument *dz*, no computation
-          is hence needed.
-
-    trifinder : `~matplotlib.tri.TriFinder`, optional
-        If not specified, the Triangulation's default TriFinder will
-        be used by calling `.Triangulation.get_trifinder`.
-    dz : tuple of array-likes (dzdx, dzdy), optional
-        Used only if  *kind* ='user'. In this case *dz* must be provided as
-        (dzdx, dzdy) where dzdx, dzdy are arrays of the same shape as *z* and
-        are the interpolant first derivatives at the *triangulation* points.
-
-    Methods
-    -------
-    `__call__` (x, y) : Returns interpolated values at (x, y) points.
-    `gradient` (x, y) : Returns interpolated derivatives at (x, y) points.
-
-    Notes
-    -----
-    This note is a bit technical and details how the cubic interpolation is
-    computed.
-
-    The interpolation is based on a Clough-Tocher subdivision scheme of
-    the *triangulation* mesh (to make it clearer, each triangle of the
-    grid will be divided in 3 child-triangles, and on each child triangle
-    the interpolated function is a cubic polynomial of the 2 coordinates).
-    This technique originates from FEM (Finite Element Method) analysis;
-    the element used is a reduced Hsieh-Clough-Tocher (HCT)
-    element. Its shape functions are described in [1]_.
-    The assembled function is guaranteed to be C1-smooth, i.e. it is
-    continuous and its first derivatives are also continuous (this
-    is easy to show inside the triangles but is also true when crossing the
-    edges).
-
-    In the default case (*kind* ='min_E'), the interpolant minimizes a
-    curvature energy on the functional space generated by the HCT element
-    shape functions - with imposed values but arbitrary derivatives at each
-    node. The minimized functional is the integral of the so-called total
-    curvature (implementation based on an algorithm from [2]_ - PCG sparse
-    solver):
-
-        .. math::
-
-            E(z) = \frac{1}{2} \int_{\Omega} \left(
-                \left( \frac{\partial^2{z}}{\partial{x}^2} \right)^2 +
-                \left( \frac{\partial^2{z}}{\partial{y}^2} \right)^2 +
-                2\left( \frac{\partial^2{z}}{\partial{y}\partial{x}} \right)^2
-            \right) dx\,dy
-
-    If the case *kind* ='geom' is chosen by the user, a simple geometric
-    approximation is used (weighted average of the triangle normal
-    vectors), which could improve speed on very large grids.
-
-    References
-    ----------
-    .. [1] Michel Bernadou, Kamal Hassan, "Basis functions for general
-        Hsieh-Clough-Tocher triangles, complete or reduced.",
-        International Journal for Numerical Methods in Engineering,
-        17(5):784 - 789. 2.01.
-    .. [2] C.T. Kelley, "Iterative Methods for Optimization".
-
-    """
-    def __init__(self, triangulation, z, kind='min_E', trifinder=None,
-                 dz=None):
-        super().__init__(triangulation, z, trifinder)
-
-        # Loads the underlying c++ _triangulation.
-        # (During loading, reordering of triangulation._triangles may occur so
-        # that all final triangles are now anti-clockwise)
-        self._triangulation.get_cpp_triangulation()
-
-        # To build the stiffness matrix and avoid zero-energy spurious modes
-        # we will only store internally the valid (unmasked) triangles and
-        # the necessary (used) points coordinates.
-        # 2 renumbering tables need to be computed and stored:
-        #  - a triangle renum table in order to translate the result from a
-        #    TriFinder instance into the internal stored triangle number.
-        #  - a node renum table to overwrite the self._z values into the new
-        #    (used) node numbering.
-        tri_analyzer = TriAnalyzer(self._triangulation)
-        (compressed_triangles, compressed_x, compressed_y, tri_renum,
-         node_renum) = tri_analyzer._get_compressed_triangulation()
-        self._triangles = compressed_triangles
-        self._tri_renum = tri_renum
-        # Taking into account the node renumbering in self._z:
-        valid_node = (node_renum != -1)
-        self._z[node_renum[valid_node]] = self._z[valid_node]
-
-        # Computing scale factors
-        self._unit_x = np.ptp(compressed_x)
-        self._unit_y = np.ptp(compressed_y)
-        self._pts = np.column_stack([compressed_x / self._unit_x,
-                                     compressed_y / self._unit_y])
-        # Computing triangle points
-        self._tris_pts = self._pts[self._triangles]
-        # Computing eccentricities
-        self._eccs = self._compute_tri_eccentricities(self._tris_pts)
-        # Computing dof estimations for HCT triangle shape function
-        _api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
-        self._dof = self._compute_dof(kind, dz=dz)
-        # Loading HCT element
-        self._ReferenceElement = _ReducedHCT_Element()
-
-    def __call__(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('z',))[0]
-    __call__.__doc__ = TriInterpolator._docstring__call__
-
-    def gradient(self, x, y):
-        return self._interpolate_multikeys(x, y, tri_index=None,
-                                           return_keys=('dzdx', 'dzdy'))
-    gradient.__doc__ = TriInterpolator._docstringgradient
-
-    def _interpolate_single_key(self, return_key, tri_index, x, y):
-        _api.check_in_list(['z', 'dzdx', 'dzdy'], return_key=return_key)
-        tris_pts = self._tris_pts[tri_index]
-        alpha = self._get_alpha_vec(x, y, tris_pts)
-        ecc = self._eccs[tri_index]
-        dof = np.expand_dims(self._dof[tri_index], axis=1)
-        if return_key == 'z':
-            return self._ReferenceElement.get_function_values(
-                alpha, ecc, dof)
-        else:  # 'dzdx', 'dzdy'
-            J = self._get_jacobian(tris_pts)
-            dzdx = self._ReferenceElement.get_function_derivatives(
-                alpha, J, ecc, dof)
-            if return_key == 'dzdx':
-                return dzdx[:, 0, 0]
-            else:
-                return dzdx[:, 1, 0]
-
-    def _compute_dof(self, kind, dz=None):
-        """
-        Compute and return nodal dofs according to kind.
-
-        Parameters
-        ----------
-        kind : {'min_E', 'geom', 'user'}
-            Choice of the _DOF_estimator subclass to estimate the gradient.
-        dz : tuple of array-likes (dzdx, dzdy), optional
-            Used only if *kind*=user; in this case passed to the
-            :class:`_DOF_estimator_user`.
-
-        Returns
-        -------
-        array-like, shape (npts, 2)
-            Estimation of the gradient at triangulation nodes (stored as
-            degree of freedoms of reduced-HCT triangle elements).
-        """
-        if kind == 'user':
-            if dz is None:
-                raise ValueError("For a CubicTriInterpolator with "
-                                 "*kind*='user', a valid *dz* "
-                                 "argument is expected.")
-            TE = _DOF_estimator_user(self, dz=dz)
-        elif kind == 'geom':
-            TE = _DOF_estimator_geom(self)
-        else:  # 'min_E', checked in __init__
-            TE = _DOF_estimator_min_E(self)
-        return TE.compute_dof_from_df()
-
-    @staticmethod
-    def _get_alpha_vec(x, y, tris_pts):
-        """
-        Fast (vectorized) function to compute barycentric coordinates alpha.
-
-        Parameters
-        ----------
-        x, y : array-like of dim 1 (shape (nx,))
-            Coordinates of the points whose points barycentric coordinates are
-            requested.
-        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
-            Coordinates of the containing triangles apexes.
-
-        Returns
-        -------
-        array of dim 2 (shape (nx, 3))
-            Barycentric coordinates of the points inside the containing
-            triangles.
-        """
-        ndim = tris_pts.ndim-2
-
-        a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
-        b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
-        abT = np.stack([a, b], axis=-1)
-        ab = _transpose_vectorized(abT)
-        OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]
-
-        metric = ab @ abT
-        # Here we try to deal with the colinear cases.
-        # metric_inv is in this case set to the Moore-Penrose pseudo-inverse
-        # meaning that we will still return a set of valid barycentric
-        # coordinates.
-        metric_inv = _pseudo_inv22sym_vectorized(metric)
-        Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
-        ksi = metric_inv @ Covar
-        alpha = _to_matrix_vectorized([
-            [1-ksi[:, 0, 0]-ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
-        return alpha
-
-    @staticmethod
-    def _get_jacobian(tris_pts):
-        """
-        Fast (vectorized) function to compute triangle jacobian matrix.
-
-        Parameters
-        ----------
-        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
-            Coordinates of the containing triangles apexes.
-
-        Returns
-        -------
-        array of dim 3 (shape (nx, 2, 2))
-            Barycentric coordinates of the points inside the containing
-            triangles.
-            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
-            itri, so that the following (matrix) relationship holds:
-               [dz/dksi] = [J] x [dz/dx]
-            with x: global coordinates
-                 ksi: element parametric coordinates in triangle first apex
-                 local basis.
-        """
-        a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
-        b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
-        J = _to_matrix_vectorized([[a[:, 0], a[:, 1]],
-                                   [b[:, 0], b[:, 1]]])
-        return J
-
-    @staticmethod
-    def _compute_tri_eccentricities(tris_pts):
-        """
-        Compute triangle eccentricities.
-
-        Parameters
-        ----------
-        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
-            Coordinates of the triangles apexes.
-
-        Returns
-        -------
-        array like of dim 2 (shape: (nx, 3))
-            The so-called eccentricity parameters [1] needed for HCT triangular
-            element.
-        """
-        a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
-        b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
-        c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
-        # Do not use np.squeeze, this is dangerous if only one triangle
-        # in the triangulation...
-        dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
-        dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
-        dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
-        # Note that this line will raise a warning for dot_a, dot_b or dot_c
-        # zeros, but we choose not to support triangles with duplicate points.
-        return _to_matrix_vectorized([[(dot_c-dot_b) / dot_a],
-                                      [(dot_a-dot_c) / dot_b],
-                                      [(dot_b-dot_a) / dot_c]])
-
-
-# FEM element used for interpolation and for solving minimisation
-# problem (Reduced HCT element)
-class _ReducedHCT_Element:
-    """
-    Implementation of reduced HCT triangular element with explicit shape
-    functions.
-
-    Computes z, dz, d2z and the element stiffness matrix for bending energy:
-    E(f) = integral( (d2z/dx2 + d2z/dy2)**2 dA)
-
-    *** Reference for the shape functions: ***
-    [1] Basis functions for general Hsieh-Clough-Tocher _triangles, complete or
-        reduced.
-        Michel Bernadou, Kamal Hassan
-        International Journal for Numerical Methods in Engineering.
-        17(5):784 - 789.  2.01
-
-    *** Element description: ***
-    9 dofs: z and dz given at 3 apex
-    C1 (conform)
-
-    """
-    # 1) Loads matrices to generate shape functions as a function of
-    #    triangle eccentricities - based on [1] p.11 '''
-    M = np.array([
-        [ 0.00, 0.00, 0.00,  4.50,  4.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00,  0.50,  1.25, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00,  1.25,  0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.50, 1.00, 0.00, -1.50,  0.00, 3.00, 3.00, 0.00, 0.00, 3.00],
-        [ 0.00, 0.00, 0.00, -0.25,  0.25, 0.00, 1.00, 0.00, 0.00, 0.50],
-        [ 0.25, 0.00, 0.00, -0.50, -0.25, 1.00, 0.00, 0.00, 0.00, 1.00],
-        [ 0.50, 0.00, 1.00,  0.00, -1.50, 0.00, 0.00, 3.00, 3.00, 3.00],
-        [ 0.25, 0.00, 0.00, -0.25, -0.50, 0.00, 0.00, 0.00, 1.00, 1.00],
-        [ 0.00, 0.00, 0.00,  0.25, -0.25, 0.00, 0.00, 1.00, 0.00, 0.50]])
-    M0 = np.array([
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [-1.00, 0.00, 0.00,  1.50,  1.50, 0.00, 0.00, 0.00, 0.00, -3.00],
-        [-0.50, 0.00, 0.00,  0.75,  0.75, 0.00, 0.00, 0.00, 0.00, -1.50],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 1.00, 0.00, 0.00, -1.50, -1.50, 0.00, 0.00, 0.00, 0.00,  3.00],
-        [ 0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00, 0.00, 0.00,  0.00],
-        [ 0.50, 0.00, 0.00, -0.75, -0.75, 0.00, 0.00, 0.00, 0.00,  1.50]])
-    M1 = np.array([
-        [-0.50, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.50, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.25, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
-    M2 = np.array([
-        [ 0.50, 0.00, 0.00, 0.00, -1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.25, 0.00, 0.00, 0.00, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.50, 0.00, 0.00, 0.00,  1.50, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [-0.25, 0.00, 0.00, 0.00,  0.75, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
-        [ 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
-
-    # 2) Loads matrices to rotate components of gradient & Hessian
-    #    vectors in the reference basis of triangle first apex (a0)
-    rotate_dV = np.array([[ 1.,  0.], [ 0.,  1.],
-                          [ 0.,  1.], [-1., -1.],
-                          [-1., -1.], [ 1.,  0.]])
-
-    rotate_d2V = np.array([[1., 0., 0.], [0., 1., 0.], [ 0.,  0.,  1.],
-                           [0., 1., 0.], [1., 1., 1.], [ 0., -2., -1.],
-                           [1., 1., 1.], [1., 0., 0.], [-2.,  0., -1.]])
-
-    # 3) Loads Gauss points & weights on the 3 sub-_triangles for P2
-    #    exact integral - 3 points on each subtriangles.
-    # NOTE: as the 2nd derivative is discontinuous , we really need those 9
-    # points!
-    n_gauss = 9
-    gauss_pts = np.array([[13./18.,  4./18.,  1./18.],
-                          [ 4./18., 13./18.,  1./18.],
-                          [ 7./18.,  7./18.,  4./18.],
-                          [ 1./18., 13./18.,  4./18.],
-                          [ 1./18.,  4./18., 13./18.],
-                          [ 4./18.,  7./18.,  7./18.],
-                          [ 4./18.,  1./18., 13./18.],
-                          [13./18.,  1./18.,  4./18.],
-                          [ 7./18.,  4./18.,  7./18.]], dtype=np.float64)
-    gauss_w = np.ones([9], dtype=np.float64) / 9.
-
-    #  4) Stiffness matrix for curvature energy
-    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
-
-    #  5) Loads the matrix to compute DOF_rot from tri_J at apex 0
-    J0_to_J1 = np.array([[-1.,  1.], [-1.,  0.]])
-    J0_to_J2 = np.array([[ 0., -1.], [ 1., -1.]])
-
-    def get_function_values(self, alpha, ecc, dofs):
-        """
-        Parameters
-        ----------
-        alpha : is a (N x 3 x 1) array (array of column-matrices) of
-        barycentric coordinates,
-        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities,
-        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
-        degrees of freedom.
-
-        Returns
-        -------
-        Returns the N-array of interpolated function values.
-        """
-        subtri = np.argmin(alpha, axis=1)[:, 0]
-        ksi = _roll_vectorized(alpha, -subtri, axis=0)
-        E = _roll_vectorized(ecc, -subtri, axis=0)
-        x = ksi[:, 0, 0]
-        y = ksi[:, 1, 0]
-        z = ksi[:, 2, 0]
-        x_sq = x*x
-        y_sq = y*y
-        z_sq = z*z
-        V = _to_matrix_vectorized([
-            [x_sq*x], [y_sq*y], [z_sq*z], [x_sq*z], [x_sq*y], [y_sq*x],
-            [y_sq*z], [z_sq*y], [z_sq*x], [x*y*z]])
-        prod = self.M @ V
-        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
-        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
-        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
-        s = _roll_vectorized(prod, 3*subtri, axis=0)
-        return (dofs @ s)[:, 0, 0]
-
-    def get_function_derivatives(self, alpha, J, ecc, dofs):
-        """
-        Parameters
-        ----------
-        *alpha* is a (N x 3 x 1) array (array of column-matrices of
-        barycentric coordinates)
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices of triangle
-        eccentricities)
-        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
-        degrees of freedom.
-
-        Returns
-        -------
-        Returns the values of interpolated function derivatives [dz/dx, dz/dy]
-        in global coordinates at locations alpha, as a column-matrices of
-        shape (N x 2 x 1).
-        """
-        subtri = np.argmin(alpha, axis=1)[:, 0]
-        ksi = _roll_vectorized(alpha, -subtri, axis=0)
-        E = _roll_vectorized(ecc, -subtri, axis=0)
-        x = ksi[:, 0, 0]
-        y = ksi[:, 1, 0]
-        z = ksi[:, 2, 0]
-        x_sq = x*x
-        y_sq = y*y
-        z_sq = z*z
-        dV = _to_matrix_vectorized([
-            [    -3.*x_sq,     -3.*x_sq],
-            [     3.*y_sq,           0.],
-            [          0.,      3.*z_sq],
-            [     -2.*x*z, -2.*x*z+x_sq],
-            [-2.*x*y+x_sq,      -2.*x*y],
-            [ 2.*x*y-y_sq,        -y_sq],
-            [      2.*y*z,         y_sq],
-            [        z_sq,       2.*y*z],
-            [       -z_sq,  2.*x*z-z_sq],
-            [     x*z-y*z,      x*y-y*z]])
-        # Puts back dV in first apex basis
-        dV = dV @ _extract_submatrices(
-            self.rotate_dV, subtri, block_size=2, axis=0)
-
-        prod = self.M @ dV
-        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ dV)
-        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ dV)
-        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ dV)
-        dsdksi = _roll_vectorized(prod, 3*subtri, axis=0)
-        dfdksi = dofs @ dsdksi
-        # In global coordinates:
-        # Here we try to deal with the simplest colinear cases, returning a
-        # null matrix.
-        J_inv = _safe_inv22_vectorized(J)
-        dfdx = J_inv @ _transpose_vectorized(dfdksi)
-        return dfdx
-
-    def get_function_hessians(self, alpha, J, ecc, dofs):
-        """
-        Parameters
-        ----------
-        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
-        barycentric coordinates
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-        *dofs* is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
-        degrees of freedom.
-
-        Returns
-        -------
-        Returns the values of interpolated function 2nd-derivatives
-        [d2z/dx2, d2z/dy2, d2z/dxdy] in global coordinates at locations alpha,
-        as a column-matrices of shape (N x 3 x 1).
-        """
-        d2sdksi2 = self.get_d2Sidksij2(alpha, ecc)
-        d2fdksi2 = dofs @ d2sdksi2
-        H_rot = self.get_Hrot_from_J(J)
-        d2fdx2 = d2fdksi2 @ H_rot
-        return _transpose_vectorized(d2fdx2)
-
-    def get_d2Sidksij2(self, alpha, ecc):
-        """
-        Parameters
-        ----------
-        *alpha* is a (N x 3 x 1) array (array of column-matrices) of
-        barycentric coordinates
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-
-        Returns
-        -------
-        Returns the arrays d2sdksi2 (N x 3 x 1) Hessian of shape functions
-        expressed in covariant coordinates in first apex basis.
-        """
-        subtri = np.argmin(alpha, axis=1)[:, 0]
-        ksi = _roll_vectorized(alpha, -subtri, axis=0)
-        E = _roll_vectorized(ecc, -subtri, axis=0)
-        x = ksi[:, 0, 0]
-        y = ksi[:, 1, 0]
-        z = ksi[:, 2, 0]
-        d2V = _to_matrix_vectorized([
-            [     6.*x,      6.*x,      6.*x],
-            [     6.*y,        0.,        0.],
-            [       0.,      6.*z,        0.],
-            [     2.*z, 2.*z-4.*x, 2.*z-2.*x],
-            [2.*y-4.*x,      2.*y, 2.*y-2.*x],
-            [2.*x-4.*y,        0.,     -2.*y],
-            [     2.*z,        0.,      2.*y],
-            [       0.,      2.*y,      2.*z],
-            [       0., 2.*x-4.*z,     -2.*z],
-            [    -2.*z,     -2.*y,     x-y-z]])
-        # Puts back d2V in first apex basis
-        d2V = d2V @ _extract_submatrices(
-            self.rotate_d2V, subtri, block_size=3, axis=0)
-        prod = self.M @ d2V
-        prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ d2V)
-        prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ d2V)
-        prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ d2V)
-        d2sdksi2 = _roll_vectorized(prod, 3*subtri, axis=0)
-        return d2sdksi2
-
-    def get_bending_matrices(self, J, ecc):
-        """
-        Parameters
-        ----------
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-
-        Returns
-        -------
-        Returns the element K matrices for bending energy expressed in
-        GLOBAL nodal coordinates.
-        K_ij = integral [ (d2zi/dx2 + d2zi/dy2) * (d2zj/dx2 + d2zj/dy2) dA]
-        tri_J is needed to rotate dofs from local basis to global basis
-        """
-        n = np.size(ecc, 0)
-
-        # 1) matrix to rotate dofs in global coordinates
-        J1 = self.J0_to_J1 @ J
-        J2 = self.J0_to_J2 @ J
-        DOF_rot = np.zeros([n, 9, 9], dtype=np.float64)
-        DOF_rot[:, 0, 0] = 1
-        DOF_rot[:, 3, 3] = 1
-        DOF_rot[:, 6, 6] = 1
-        DOF_rot[:, 1:3, 1:3] = J
-        DOF_rot[:, 4:6, 4:6] = J1
-        DOF_rot[:, 7:9, 7:9] = J2
-
-        # 2) matrix to rotate Hessian in global coordinates.
-        H_rot, area = self.get_Hrot_from_J(J, return_area=True)
-
-        # 3) Computes stiffness matrix
-        # Gauss quadrature.
-        K = np.zeros([n, 9, 9], dtype=np.float64)
-        weights = self.gauss_w
-        pts = self.gauss_pts
-        for igauss in range(self.n_gauss):
-            alpha = np.tile(pts[igauss, :], n).reshape(n, 3)
-            alpha = np.expand_dims(alpha, 2)
-            weight = weights[igauss]
-            d2Skdksi2 = self.get_d2Sidksij2(alpha, ecc)
-            d2Skdx2 = d2Skdksi2 @ H_rot
-            K += weight * (d2Skdx2 @ self.E @ _transpose_vectorized(d2Skdx2))
-
-        # 4) With nodal (not elem) dofs
-        K = _transpose_vectorized(DOF_rot) @ K @ DOF_rot
-
-        # 5) Need the area to compute total element energy
-        return _scalar_vectorized(area, K)
-
-    def get_Hrot_from_J(self, J, return_area=False):
-        """
-        Parameters
-        ----------
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-
-        Returns
-        -------
-        Returns H_rot used to rotate Hessian from local basis of first apex,
-        to global coordinates.
-        if *return_area* is True, returns also the triangle area (0.5*det(J))
-        """
-        # Here we try to deal with the simplest colinear cases; a null
-        # energy and area is imposed.
-        J_inv = _safe_inv22_vectorized(J)
-        Ji00 = J_inv[:, 0, 0]
-        Ji11 = J_inv[:, 1, 1]
-        Ji10 = J_inv[:, 1, 0]
-        Ji01 = J_inv[:, 0, 1]
-        H_rot = _to_matrix_vectorized([
-            [Ji00*Ji00, Ji10*Ji10, Ji00*Ji10],
-            [Ji01*Ji01, Ji11*Ji11, Ji01*Ji11],
-            [2*Ji00*Ji01, 2*Ji11*Ji10, Ji00*Ji11+Ji10*Ji01]])
-        if not return_area:
-            return H_rot
-        else:
-            area = 0.5 * (J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0])
-            return H_rot, area
-
-    def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
-        """
-        Build K and F for the following elliptic formulation:
-        minimization of curvature energy with value of function at node
-        imposed and derivatives 'free'.
-
-        Build the global Kff matrix in cco format.
-        Build the full Ff vec Ff = - Kfc x Uc.
-
-        Parameters
-        ----------
-        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
-        triangle first apex)
-        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
-        eccentricities
-        *triangles* is a (N x 3) array of nodes indexes.
-        *Uc* is (N x 3) array of imposed displacements at nodes
-
-        Returns
-        -------
-        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
-        (row, col) entries must be summed.
-        Ff: force vector - dim npts * 3
-        """
-        ntri = np.size(ecc, 0)
-        vec_range = np.arange(ntri, dtype=np.int32)
-        c_indices = np.full(ntri, -1, dtype=np.int32)  # for unused dofs, -1
-        f_dof = [1, 2, 4, 5, 7, 8]
-        c_dof = [0, 3, 6]
-
-        # vals, rows and cols indices in global dof numbering
-        f_dof_indices = _to_matrix_vectorized([[
-            c_indices, triangles[:, 0]*2, triangles[:, 0]*2+1,
-            c_indices, triangles[:, 1]*2, triangles[:, 1]*2+1,
-            c_indices, triangles[:, 2]*2, triangles[:, 2]*2+1]])
-
-        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
-        f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
-        f_col_indices = expand_indices @ f_dof_indices
-        K_elem = self.get_bending_matrices(J, ecc)
-
-        # Extracting sub-matrices
-        # Explanation & notations:
-        # * Subscript f denotes 'free' degrees of freedom (i.e. dz/dx, dz/dx)
-        # * Subscript c denotes 'condensated' (imposed) degrees of freedom
-        #    (i.e. z at all nodes)
-        # * F = [Ff, Fc] is the force vector
-        # * U = [Uf, Uc] is the imposed dof vector
-        #        [ Kff Kfc ]
-        # * K =  [         ]  is the laplacian stiffness matrix
-        #        [ Kcf Kff ]
-        # * As F = K x U one gets straightforwardly: Ff = - Kfc x Uc
-
-        # Computing Kff stiffness matrix in sparse coo format
-        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
-        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
-        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])
-
-        # Computing Ff force vector in sparse coo format
-        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
-        Uc_elem = np.expand_dims(Uc, axis=2)
-        Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
-        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]
-
-        # Extracting Ff force vector in dense format
-        # We have to sum duplicate indices -  using bincount
-        Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
-        return Kff_rows, Kff_cols, Kff_vals, Ff
-
-
-# :class:_DOF_estimator, _DOF_estimator_user, _DOF_estimator_geom,
-# _DOF_estimator_min_E
-# Private classes used to compute the degree of freedom of each triangular
-# element for the TriCubicInterpolator.
-class _DOF_estimator:
-    """
-    Abstract base class for classes used to estimate a function's first
-    derivatives, and deduce the dofs for a CubicTriInterpolator using a
-    reduced HCT element formulation.
-
-    Derived classes implement ``compute_df(self, **kwargs)``, returning
-    ``np.vstack([dfx, dfy]).T`` where ``dfx, dfy`` are the estimation of the 2
-    gradient coordinates.
-    """
-    def __init__(self, interpolator, **kwargs):
-        _api.check_isinstance(CubicTriInterpolator, interpolator=interpolator)
-        self._pts = interpolator._pts
-        self._tris_pts = interpolator._tris_pts
-        self.z = interpolator._z
-        self._triangles = interpolator._triangles
-        (self._unit_x, self._unit_y) = (interpolator._unit_x,
-                                        interpolator._unit_y)
-        self.dz = self.compute_dz(**kwargs)
-        self.compute_dof_from_df()
-
-    def compute_dz(self, **kwargs):
-        raise NotImplementedError
-
-    def compute_dof_from_df(self):
-        """
-        Compute reduced-HCT elements degrees of freedom, from the gradient.
-        """
-        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
-        tri_z = self.z[self._triangles]
-        tri_dz = self.dz[self._triangles]
-        tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
-        return tri_dof
-
-    @staticmethod
-    def get_dof_vec(tri_z, tri_dz, J):
-        """
-        Compute the dof vector of a triangle, from the value of f, df and
-        of the local Jacobian at each node.
-
-        Parameters
-        ----------
-        tri_z : shape (3,) array
-            f nodal values.
-        tri_dz : shape (3, 2) array
-            df/dx, df/dy nodal values.
-        J
-            Jacobian matrix in local basis of apex 0.
-
-        Returns
-        -------
-        dof : shape (9,) array
-            For each apex ``iapex``::
-
-                dof[iapex*3+0] = f(Ai)
-                dof[iapex*3+1] = df(Ai).(AiAi+)
-                dof[iapex*3+2] = df(Ai).(AiAi-)
-        """
-        npt = tri_z.shape[0]
-        dof = np.zeros([npt, 9], dtype=np.float64)
-        J1 = _ReducedHCT_Element.J0_to_J1 @ J
-        J2 = _ReducedHCT_Element.J0_to_J2 @ J
-
-        col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
-        col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
-        col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
-
-        dfdksi = _to_matrix_vectorized([
-            [col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]],
-            [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
-        dof[:, 0:7:3] = tri_z
-        dof[:, 1:8:3] = dfdksi[:, 0]
-        dof[:, 2:9:3] = dfdksi[:, 1]
-        return dof
-
-
-class _DOF_estimator_user(_DOF_estimator):
-    """dz is imposed by user; accounts for scaling if any."""
-
-    def compute_dz(self, dz):
-        (dzdx, dzdy) = dz
-        dzdx = dzdx * self._unit_x
-        dzdy = dzdy * self._unit_y
-        return np.vstack([dzdx, dzdy]).T
-
-
-class _DOF_estimator_geom(_DOF_estimator):
-    """Fast 'geometric' approximation, recommended for large arrays."""
-
-    def compute_dz(self):
-        """
-        self.df is computed as weighted average of _triangles sharing a common
-        node. On each triangle itri f is first assumed linear (= ~f), which
-        allows to compute d~f[itri]
-        Then the following approximation of df nodal values is then proposed:
-            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
-        The weighted coeff. w[itri] are proportional to the angle of the
-        triangle itri at apex ipt
-        """
-        el_geom_w = self.compute_geom_weights()
-        el_geom_grad = self.compute_geom_grads()
-
-        # Sum of weights coeffs
-        w_node_sum = np.bincount(np.ravel(self._triangles),
-                                 weights=np.ravel(el_geom_w))
-
-        # Sum of weighted df = (dfx, dfy)
-        dfx_el_w = np.empty_like(el_geom_w)
-        dfy_el_w = np.empty_like(el_geom_w)
-        for iapex in range(3):
-            dfx_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 0]
-            dfy_el_w[:, iapex] = el_geom_w[:, iapex]*el_geom_grad[:, 1]
-        dfx_node_sum = np.bincount(np.ravel(self._triangles),
-                                   weights=np.ravel(dfx_el_w))
-        dfy_node_sum = np.bincount(np.ravel(self._triangles),
-                                   weights=np.ravel(dfy_el_w))
-
-        # Estimation of df
-        dfx_estim = dfx_node_sum/w_node_sum
-        dfy_estim = dfy_node_sum/w_node_sum
-        return np.vstack([dfx_estim, dfy_estim]).T
-
-    def compute_geom_weights(self):
-        """
-        Build the (nelems, 3) weights coeffs of _triangles angles,
-        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
-        """
-        weights = np.zeros([np.size(self._triangles, 0), 3])
-        tris_pts = self._tris_pts
-        for ipt in range(3):
-            p0 = tris_pts[:, ipt % 3, :]
-            p1 = tris_pts[:, (ipt+1) % 3, :]
-            p2 = tris_pts[:, (ipt-1) % 3, :]
-            alpha1 = np.arctan2(p1[:, 1]-p0[:, 1], p1[:, 0]-p0[:, 0])
-            alpha2 = np.arctan2(p2[:, 1]-p0[:, 1], p2[:, 0]-p0[:, 0])
-            # In the below formula we could take modulo 2. but
-            # modulo 1. is safer regarding round-off errors (flat triangles).
-            angle = np.abs(((alpha2-alpha1) / np.pi) % 1)
-            # Weight proportional to angle up np.pi/2; null weight for
-            # degenerated cases 0 and np.pi (note that *angle* is normalized
-            # by np.pi).
-            weights[:, ipt] = 0.5 - np.abs(angle-0.5)
-        return weights
-
-    def compute_geom_grads(self):
-        """
-        Compute the (global) gradient component of f assumed linear (~f).
-        returns array df of shape (nelems, 2)
-        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
-        """
-        tris_pts = self._tris_pts
-        tris_f = self.z[self._triangles]
-
-        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
-        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
-        dM = np.dstack([dM1, dM2])
-        # Here we try to deal with the simplest colinear cases: a null
-        # gradient is assumed in this case.
-        dM_inv = _safe_inv22_vectorized(dM)
-
-        dZ1 = tris_f[:, 1] - tris_f[:, 0]
-        dZ2 = tris_f[:, 2] - tris_f[:, 0]
-        dZ = np.vstack([dZ1, dZ2]).T
-        df = np.empty_like(dZ)
-
-        # With np.einsum: could be ej,eji -> ej
-        df[:, 0] = dZ[:, 0]*dM_inv[:, 0, 0] + dZ[:, 1]*dM_inv[:, 1, 0]
-        df[:, 1] = dZ[:, 0]*dM_inv[:, 0, 1] + dZ[:, 1]*dM_inv[:, 1, 1]
-        return df
-
-
-class _DOF_estimator_min_E(_DOF_estimator_geom):
-    """
-    The 'smoothest' approximation, df is computed through global minimization
-    of the bending energy:
-      E(f) = integral[(d2z/dx2 + d2z/dy2 + 2 d2z/dxdy)**2 dA]
-    """
-    def __init__(self, Interpolator):
-        self._eccs = Interpolator._eccs
-        super().__init__(Interpolator)
-
-    def compute_dz(self):
-        """
-        Elliptic solver for bending energy minimization.
-        Uses a dedicated 'toy' sparse Jacobi PCG solver.
-        """
-        # Initial guess for iterative PCG solver.
-        dz_init = super().compute_dz()
-        Uf0 = np.ravel(dz_init)
-
-        reference_element = _ReducedHCT_Element()
-        J = CubicTriInterpolator._get_jacobian(self._tris_pts)
-        eccs = self._eccs
-        triangles = self._triangles
-        Uc = self.z[self._triangles]
-
-        # Building stiffness matrix and force vector in coo format
-        Kff_rows, Kff_cols, Kff_vals, Ff = reference_element.get_Kff_and_Ff(
-            J, eccs, triangles, Uc)
-
-        # Building sparse matrix and solving minimization problem
-        # We could use scipy.sparse direct solver; however to avoid this
-        # external dependency an implementation of a simple PCG solver with
-        # a simple diagonal Jacobi preconditioner is implemented.
-        tol = 1.e-10
-        n_dof = Ff.shape[0]
-        Kff_coo = _Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
-                                     shape=(n_dof, n_dof))
-        Kff_coo.compress_csc()
-        Uf, err = _cg(A=Kff_coo, b=Ff, x0=Uf0, tol=tol)
-        # If the PCG did not converge, we return the best guess between Uf0
-        # and Uf.
-        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
-        if err0 < err:
-            # Maybe a good occasion to raise a warning here ?
-            _api.warn_external("In TriCubicInterpolator initialization, "
-                               "PCG sparse solver did not converge after "
-                               "1000 iterations. `geom` approximation is "
-                               "used instead of `min_E`")
-            Uf = Uf0
-
-        # Building dz from Uf
-        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
-        dz[:, 0] = Uf[::2]
-        dz[:, 1] = Uf[1::2]
-        return dz
-
-
-# The following private :class:_Sparse_Matrix_coo and :func:_cg provide
-# a PCG sparse solver for (symmetric) elliptic problems.
-class _Sparse_Matrix_coo:
-    def __init__(self, vals, rows, cols, shape):
-        """
-        Create a sparse matrix in coo format.
-        *vals*: arrays of values of non-null entries of the matrix
-        *rows*: int arrays of rows of non-null entries of the matrix
-        *cols*: int arrays of cols of non-null entries of the matrix
-        *shape*: 2-tuple (n, m) of matrix shape
-        """
-        self.n, self.m = shape
-        self.vals = np.asarray(vals, dtype=np.float64)
-        self.rows = np.asarray(rows, dtype=np.int32)
-        self.cols = np.asarray(cols, dtype=np.int32)
-
-    def dot(self, V):
-        """
-        Dot product of self by a vector *V* in sparse-dense to dense format
-        *V* dense vector of shape (self.m,).
-        """
-        assert V.shape == (self.m,)
-        return np.bincount(self.rows,
-                           weights=self.vals*V[self.cols],
-                           minlength=self.m)
-
-    def compress_csc(self):
-        """
-        Compress rows, cols, vals / summing duplicates. Sort for csc format.
-        """
-        _, unique, indices = np.unique(
-            self.rows + self.n*self.cols,
-            return_index=True, return_inverse=True)
-        self.rows = self.rows[unique]
-        self.cols = self.cols[unique]
-        self.vals = np.bincount(indices, weights=self.vals)
-
-    def compress_csr(self):
-        """
-        Compress rows, cols, vals / summing duplicates. Sort for csr format.
-        """
-        _, unique, indices = np.unique(
-            self.m*self.rows + self.cols,
-            return_index=True, return_inverse=True)
-        self.rows = self.rows[unique]
-        self.cols = self.cols[unique]
-        self.vals = np.bincount(indices, weights=self.vals)
-
-    def to_dense(self):
-        """
-        Return a dense matrix representing self, mainly for debugging purposes.
-        """
-        ret = np.zeros([self.n, self.m], dtype=np.float64)
-        nvals = self.vals.size
-        for i in range(nvals):
-            ret[self.rows[i], self.cols[i]] += self.vals[i]
-        return ret
-
-    def __str__(self):
-        return self.to_dense().__str__()
-
-    @property
-    def diag(self):
-        """Return the (dense) vector of the diagonal elements."""
-        in_diag = (self.rows == self.cols)
-        diag = np.zeros(min(self.n, self.n), dtype=np.float64)  # default 0.
-        diag[self.rows[in_diag]] = self.vals[in_diag]
-        return diag
-
-
-def _cg(A, b, x0=None, tol=1.e-10, maxiter=1000):
-    """
-    Use Preconditioned Conjugate Gradient iteration to solve A x = b
-    A simple Jacobi (diagonal) preconditioner is used.
-
-    Parameters
-    ----------
-    A : _Sparse_Matrix_coo
-        *A* must have been compressed before by compress_csc or
-        compress_csr method.
-    b : array
-        Right hand side of the linear system.
-    x0 : array, optional
-        Starting guess for the solution. Defaults to the zero vector.
-    tol : float, optional
-        Tolerance to achieve. The algorithm terminates when the relative
-        residual is below tol. Default is 1e-10.
-    maxiter : int, optional
-        Maximum number of iterations.  Iteration will stop after *maxiter*
-        steps even if the specified tolerance has not been achieved. Defaults
-        to 1000.
-
-    Returns
-    -------
-    x : array
-        The converged solution.
-    err : float
-        The absolute error np.linalg.norm(A.dot(x) - b)
-    """
-    n = b.size
-    assert A.n == n
-    assert A.m == n
-    b_norm = np.linalg.norm(b)
-
-    # Jacobi pre-conditioner
-    kvec = A.diag
-    # For diag elem < 1e-6 we keep 1e-6.
-    kvec = np.maximum(kvec, 1e-6)
-
-    # Initial guess
-    if x0 is None:
-        x = np.zeros(n)
-    else:
-        x = x0
-
-    r = b - A.dot(x)
-    w = r/kvec
-
-    p = np.zeros(n)
-    beta = 0.0
-    rho = np.dot(r, w)
-    k = 0
-
-    # Following C. T. Kelley
-    while (np.sqrt(abs(rho)) > tol*b_norm) and (k < maxiter):
-        p = w + beta*p
-        z = A.dot(p)
-        alpha = rho/np.dot(p, z)
-        r = r - alpha*z
-        w = r/kvec
-        rhoold = rho
-        rho = np.dot(r, w)
-        x = x + alpha*p
-        beta = rho/rhoold
-        # err = np.linalg.norm(A.dot(x) - b)  # absolute accuracy - not used
-        k += 1
-    err = np.linalg.norm(A.dot(x) - b)
-    return x, err
-
-
-# The following private functions:
-#     :func:`_safe_inv22_vectorized`
-#     :func:`_pseudo_inv22sym_vectorized`
-#     :func:`_scalar_vectorized`
-#     :func:`_transpose_vectorized`
-#     :func:`_roll_vectorized`
-#     :func:`_to_matrix_vectorized`
-#     :func:`_extract_submatrices`
-# provide fast numpy implementation of some standard operations on arrays of
-# matrices - stored as (:, n_rows, n_cols)-shaped np.arrays.
-
-# Development note: Dealing with pathologic 'flat' triangles in the
-# CubicTriInterpolator code and impact on (2, 2)-matrix inversion functions
-# :func:`_safe_inv22_vectorized` and :func:`_pseudo_inv22sym_vectorized`.
-#
-# Goals:
-# 1) The CubicTriInterpolator should be able to handle flat or almost flat
-#    triangles without raising an error,
-# 2) These degenerated triangles should have no impact on the automatic dof
-#    calculation (associated with null weight for the _DOF_estimator_geom and
-#    with null energy for the _DOF_estimator_min_E),
-# 3) Linear patch test should be passed exactly on degenerated meshes,
-# 4) Interpolation (with :meth:`_interpolate_single_key` or
-#    :meth:`_interpolate_multi_key`) shall be correctly handled even *inside*
-#    the pathologic triangles, to interact correctly with a TriRefiner class.
-#
-# Difficulties:
-# Flat triangles have rank-deficient *J* (so-called jacobian matrix) and
-# *metric* (the metric tensor = J x J.T). Computation of the local
-# tangent plane is also problematic.
-#
-# Implementation:
-# Most of the time, when computing the inverse of a rank-deficient matrix it
-# is safe to simply return the null matrix (which is the implementation in
-# :func:`_safe_inv22_vectorized`). This is because of point 2), itself
-# enforced by:
-#    - null area hence null energy in :class:`_DOF_estimator_min_E`
-#    - angles close or equal to 0 or np.pi hence null weight in
-#      :class:`_DOF_estimator_geom`.
-#      Note that the function angle -> weight is continuous and maximum for an
-#      angle np.pi/2 (refer to :meth:`compute_geom_weights`)
-# The exception is the computation of barycentric coordinates, which is done
-# by inversion of the *metric* matrix. In this case, we need to compute a set
-# of valid coordinates (1 among numerous possibilities), to ensure point 4).
-# We benefit here from the symmetry of metric = J x J.T, which makes it easier
-# to compute a pseudo-inverse in :func:`_pseudo_inv22sym_vectorized`
-def _safe_inv22_vectorized(M):
-    """
-    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
-    matrices.
-
-    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
-    """
-    _api.check_shape((None, 2, 2), M=M)
-    M_inv = np.empty_like(M)
-    prod1 = M[:, 0, 0]*M[:, 1, 1]
-    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
-
-    # We set delta_inv to 0. in case of a rank deficient matrix; a
-    # rank-deficient input matrix *M* will lead to a null matrix in output
-    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
-    if np.all(rank2):
-        # Normal 'optimized' flow.
-        delta_inv = 1./delta
-    else:
-        # 'Pathologic' flow.
-        delta_inv = np.zeros(M.shape[0])
-        delta_inv[rank2] = 1./delta[rank2]
-
-    M_inv[:, 0, 0] = M[:, 1, 1]*delta_inv
-    M_inv[:, 0, 1] = -M[:, 0, 1]*delta_inv
-    M_inv[:, 1, 0] = -M[:, 1, 0]*delta_inv
-    M_inv[:, 1, 1] = M[:, 0, 0]*delta_inv
-    return M_inv
-
-
-def _pseudo_inv22sym_vectorized(M):
-    """
-    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
-    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.
-
-    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
-    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
-    In case M is of rank 0, we return the null matrix.
-
-    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
-    """
-    _api.check_shape((None, 2, 2), M=M)
-    M_inv = np.empty_like(M)
-    prod1 = M[:, 0, 0]*M[:, 1, 1]
-    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
-    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
-
-    if np.all(rank2):
-        # Normal 'optimized' flow.
-        M_inv[:, 0, 0] = M[:, 1, 1] / delta
-        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
-        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
-        M_inv[:, 1, 1] = M[:, 0, 0] / delta
-    else:
-        # 'Pathologic' flow.
-        # Here we have to deal with 2 sub-cases
-        # 1) First sub-case: matrices of rank 2:
-        delta = delta[rank2]
-        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
-        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
-        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
-        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
-        # 2) Second sub-case: rank-deficient matrices of rank 0 and 1:
-        rank01 = ~rank2
-        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
-        tr_zeros = (np.abs(tr) < 1.e-8)
-        sq_tr_inv = (1.-tr_zeros) / (tr**2+tr_zeros)
-        # sq_tr_inv = 1. / tr**2
-        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
-        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
-        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
-        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv
-
-    return M_inv
-
-
-def _scalar_vectorized(scalar, M):
-    """
-    Scalar product between scalars and matrices.
-    """
-    return scalar[:, np.newaxis, np.newaxis]*M
-
-
-def _transpose_vectorized(M):
-    """
-    Transposition of an array of matrices *M*.
-    """
-    return np.transpose(M, [0, 2, 1])
-
-
-def _roll_vectorized(M, roll_indices, axis):
-    """
-    Roll an array of matrices along *axis* (0: rows, 1: columns) according to
-    an array of indices *roll_indices*.
-    """
-    assert axis in [0, 1]
-    ndim = M.ndim
-    assert ndim == 3
-    ndim_roll = roll_indices.ndim
-    assert ndim_roll == 1
-    sh = M.shape
-    r, c = sh[-2:]
-    assert sh[0] == roll_indices.shape[0]
-    vec_indices = np.arange(sh[0], dtype=np.int32)
-
-    # Builds the rolled matrix
-    M_roll = np.empty_like(M)
-    if axis == 0:
-        for ir in range(r):
-            for ic in range(c):
-                M_roll[:, ir, ic] = M[vec_indices, (-roll_indices+ir) % r, ic]
-    else:  # 1
-        for ir in range(r):
-            for ic in range(c):
-                M_roll[:, ir, ic] = M[vec_indices, ir, (-roll_indices+ic) % c]
-    return M_roll
-
-
-def _to_matrix_vectorized(M):
-    """
-    Build an array of matrices from individuals np.arrays of identical shapes.
-
-    Parameters
-    ----------
-    M
-        ncols-list of nrows-lists of shape sh.
-
-    Returns
-    -------
-    M_res : np.array of shape (sh, nrow, ncols)
-        *M_res* satisfies ``M_res[..., i, j] = M[i][j]``.
-    """
-    assert isinstance(M, (tuple, list))
-    assert all(isinstance(item, (tuple, list)) for item in M)
-    c_vec = np.asarray([len(item) for item in M])
-    assert np.all(c_vec-c_vec[0] == 0)
-    r = len(M)
-    c = c_vec[0]
-    M00 = np.asarray(M[0][0])
-    dt = M00.dtype
-    sh = [M00.shape[0], r, c]
-    M_ret = np.empty(sh, dtype=dt)
-    for irow in range(r):
-        for icol in range(c):
-            M_ret[:, irow, icol] = np.asarray(M[irow][icol])
-    return M_ret
-
-
-def _extract_submatrices(M, block_indices, block_size, axis):
-    """
-    Extract selected blocks of a matrices *M* depending on parameters
-    *block_indices* and *block_size*.
-
-    Returns the array of extracted matrices *Mres* so that ::
-
-        M_res[..., ir, :] = M[(block_indices*block_size+ir), :]
-    """
-    assert block_indices.ndim == 1
-    assert axis in [0, 1]
-
-    r, c = M.shape
-    if axis == 0:
-        sh = [block_indices.shape[0], block_size, c]
-    else:  # 1
-        sh = [block_indices.shape[0], r, block_size]
 
-    dt = M.dtype
-    M_res = np.empty(sh, dtype=dt)
-    if axis == 0:
-        for ir in range(block_size):
-            M_res[:, ir, :] = M[(block_indices*block_size+ir), :]
-    else:  # 1
-        for ic in range(block_size):
-            M_res[:, :, ic] = M[:, (block_indices*block_size+ic)]
 
-    return M_res
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/tripcolor.py b/lib/matplotlib/tri/tripcolor.py
--- a/lib/matplotlib/tri/tripcolor.py
+++ b/lib/matplotlib/tri/tripcolor.py
@@ -1,154 +1,9 @@
-import numpy as np
-
+from ._tripcolor import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.collections import PolyCollection, TriMesh
-from matplotlib.colors import Normalize
-from matplotlib.tri.triangulation import Triangulation
-
-
-def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
-              vmax=None, shading='flat', facecolors=None, **kwargs):
-    """
-    Create a pseudocolor plot of an unstructured triangular grid.
-
-    Call signatures::
-
-      tripcolor(triangulation, c, *, ...)
-      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)
-
-    The triangular grid can be specified either by passing a `.Triangulation`
-    object as the first parameter, or by passing the points *x*, *y* and
-    optionally the *triangles* and a *mask*. See `.Triangulation` for an
-    explanation of these parameters.
-
-    It is possible to pass the triangles positionally, i.e.
-    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.
-    For more clarity, pass *triangles* via keyword argument.
-
-    If neither of *triangulation* or *triangles* are given, the triangulation
-    is calculated on the fly. In this case, it does not make sense to provide
-    colors at the triangle faces via *c* or *facecolors* because there are
-    multiple possible triangulations for a group of points and you don't know
-    which triangles will be constructed.
-
-    Parameters
-    ----------
-    triangulation : `.Triangulation`
-        An already created triangular grid.
-    x, y, triangles, mask
-        Parameters defining the triangular grid. See `.Triangulation`.
-        This is mutually exclusive with specifying *triangulation*.
-    c : array-like
-        The color values, either for the points or for the triangles. Which one
-        is automatically inferred from the length of *c*, i.e. does it match
-        the number of points or the number of triangles. If there are the same
-        number of points and triangles in the triangulation it is assumed that
-        color values are defined at points; to force the use of color values at
-        triangles use the keyword argument ``facecolors=c`` instead of just
-        ``c``.
-        This parameter is position-only.
-    facecolors : array-like, optional
-        Can be used alternatively to *c* to specify colors at the triangle
-        faces. This parameter takes precedence over *c*.
-    shading : {'flat', 'gouraud'}, default: 'flat'
-        If  'flat' and the color values *c* are defined at points, the color
-        values used for each triangle are from the mean c of the triangle's
-        three points. If *shading* is 'gouraud' then color values must be
-        defined at points.
-    other_parameters
-        All other parameters are the same as for `~.Axes.pcolor`.
-    """
-    _api.check_in_list(['flat', 'gouraud'], shading=shading)
-
-    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
-
-    # Parse the color to be in one of (the other variable will be None):
-    # - facecolors: if specified at the triangle faces
-    # - point_colors: if specified at the points
-    if facecolors is not None:
-        if args:
-            _api.warn_external(
-                "Positional parameter c has no effect when the keyword "
-                "facecolors is given")
-        point_colors = None
-        if len(facecolors) != len(tri.triangles):
-            raise ValueError("The length of facecolors must match the number "
-                             "of triangles")
-    else:
-        # Color from positional parameter c
-        if not args:
-            raise TypeError(
-                "tripcolor() missing 1 required positional argument: 'c'; or "
-                "1 required keyword-only argument: 'facecolors'")
-        elif len(args) > 1:
-            _api.warn_deprecated(
-                "3.6", message=f"Additional positional parameters "
-                f"{args[1:]!r} are ignored; support for them is deprecated "
-                f"since %(since)s and will be removed %(removal)s")
-        c = np.asarray(args[0])
-        if len(c) == len(tri.x):
-            # having this before the len(tri.triangles) comparison gives
-            # precedence to nodes if there are as many nodes as triangles
-            point_colors = c
-            facecolors = None
-        elif len(c) == len(tri.triangles):
-            point_colors = None
-            facecolors = c
-        else:
-            raise ValueError('The length of c must match either the number '
-                             'of points or the number of triangles')
-
-    # Handling of linewidths, shading, edgecolors and antialiased as
-    # in Axes.pcolor
-    linewidths = (0.25,)
-    if 'linewidth' in kwargs:
-        kwargs['linewidths'] = kwargs.pop('linewidth')
-    kwargs.setdefault('linewidths', linewidths)
-
-    edgecolors = 'none'
-    if 'edgecolor' in kwargs:
-        kwargs['edgecolors'] = kwargs.pop('edgecolor')
-    ec = kwargs.setdefault('edgecolors', edgecolors)
-
-    if 'antialiased' in kwargs:
-        kwargs['antialiaseds'] = kwargs.pop('antialiased')
-    if 'antialiaseds' not in kwargs and ec.lower() == "none":
-        kwargs['antialiaseds'] = False
-
-    _api.check_isinstance((Normalize, None), norm=norm)
-    if shading == 'gouraud':
-        if facecolors is not None:
-            raise ValueError(
-                "shading='gouraud' can only be used when the colors "
-                "are specified at the points, not at the faces.")
-        collection = TriMesh(tri, alpha=alpha, array=point_colors,
-                             cmap=cmap, norm=norm, **kwargs)
-    else:  # 'flat'
-        # Vertices of triangles.
-        maskedTris = tri.get_masked_triangles()
-        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)
-
-        # Color values.
-        if facecolors is None:
-            # One color per triangle, the mean of the 3 vertex color values.
-            colors = point_colors[maskedTris].mean(axis=1)
-        elif tri.mask is not None:
-            # Remove color values of masked triangles.
-            colors = facecolors[~tri.mask]
-        else:
-            colors = facecolors
-        collection = PolyCollection(verts, alpha=alpha, array=colors,
-                                    cmap=cmap, norm=norm, **kwargs)
 
-    collection._scale_norm(norm, vmin, vmax)
-    ax.grid(False)
 
-    minx = tri.x.min()
-    maxx = tri.x.max()
-    miny = tri.y.min()
-    maxy = tri.y.max()
-    corners = (minx, miny), (maxx, maxy)
-    ax.update_datalim(corners)
-    ax.autoscale_view()
-    ax.add_collection(collection)
-    return collection
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/triplot.py b/lib/matplotlib/tri/triplot.py
--- a/lib/matplotlib/tri/triplot.py
+++ b/lib/matplotlib/tri/triplot.py
@@ -1,86 +1,9 @@
-import numpy as np
-from matplotlib.tri.triangulation import Triangulation
-import matplotlib.cbook as cbook
-import matplotlib.lines as mlines
+from ._triplot import *  # noqa: F401, F403
+from matplotlib import _api
 
 
-def triplot(ax, *args, **kwargs):
-    """
-    Draw an unstructured triangular grid as lines and/or markers.
-
-    Call signatures::
-
-      triplot(triangulation, ...)
-      triplot(x, y, [triangles], *, [mask=mask], ...)
-
-    The triangular grid can be specified either by passing a `.Triangulation`
-    object as the first parameter, or by passing the points *x*, *y* and
-    optionally the *triangles* and a *mask*. If neither of *triangulation* or
-    *triangles* are given, the triangulation is calculated on the fly.
-
-    Parameters
-    ----------
-    triangulation : `.Triangulation`
-        An already created triangular grid.
-    x, y, triangles, mask
-        Parameters defining the triangular grid. See `.Triangulation`.
-        This is mutually exclusive with specifying *triangulation*.
-    other_parameters
-        All other args and kwargs are forwarded to `~.Axes.plot`.
-
-    Returns
-    -------
-    lines : `~matplotlib.lines.Line2D`
-        The drawn triangles edges.
-    markers : `~matplotlib.lines.Line2D`
-        The drawn marker nodes.
-    """
-    import matplotlib.axes
-
-    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
-    x, y, edges = (tri.x, tri.y, tri.edges)
-
-    # Decode plot format string, e.g., 'ro-'
-    fmt = args[0] if args else ""
-    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)
-
-    # Insert plot format string into a copy of kwargs (kwargs values prevail).
-    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
-    for key, val in zip(('linestyle', 'marker', 'color'),
-                        (linestyle, marker, color)):
-        if val is not None:
-            kw.setdefault(key, val)
-
-    # Draw lines without markers.
-    # Note 1: If we drew markers here, most markers would be drawn more than
-    #         once as they belong to several edges.
-    # Note 2: We insert nan values in the flattened edges arrays rather than
-    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
-    #         as it considerably speeds-up code execution.
-    linestyle = kw['linestyle']
-    kw_lines = {
-        **kw,
-        'marker': 'None',  # No marker to draw.
-        'zorder': kw.get('zorder', 1),  # Path default zorder is used.
-    }
-    if linestyle not in [None, 'None', '', ' ']:
-        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
-        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
-        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(),
-                            **kw_lines)
-    else:
-        tri_lines = ax.plot([], [], **kw_lines)
-
-    # Draw markers separately.
-    marker = kw['marker']
-    kw_markers = {
-        **kw,
-        'linestyle': 'None',  # No line to draw.
-    }
-    kw_markers.pop('label', None)
-    if marker not in [None, 'None', '', ' ']:
-        tri_markers = ax.plot(x, y, **kw_markers)
-    else:
-        tri_markers = ax.plot([], [], **kw_markers)
-
-    return tri_lines + tri_markers
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/trirefine.py b/lib/matplotlib/tri/trirefine.py
--- a/lib/matplotlib/tri/trirefine.py
+++ b/lib/matplotlib/tri/trirefine.py
@@ -1,307 +1,9 @@
-"""
-Mesh refinement for triangular grids.
-"""
-
-import numpy as np
-
+from ._trirefine import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri.triangulation import Triangulation
-import matplotlib.tri.triinterpolate
-
-
-class TriRefiner:
-    """
-    Abstract base class for classes implementing mesh refinement.
-
-    A TriRefiner encapsulates a Triangulation object and provides tools for
-    mesh refinement and interpolation.
-
-    Derived classes must implement:
-
-    - ``refine_triangulation(return_tri_index=False, **kwargs)`` , where
-      the optional keyword arguments *kwargs* are defined in each
-      TriRefiner concrete implementation, and which returns:
-
-      - a refined triangulation,
-      - optionally (depending on *return_tri_index*), for each
-        point of the refined triangulation: the index of
-        the initial triangulation triangle to which it belongs.
-
-    - ``refine_field(z, triinterpolator=None, **kwargs)``, where:
-
-      - *z* array of field values (to refine) defined at the base
-        triangulation nodes,
-      - *triinterpolator* is an optional `~matplotlib.tri.TriInterpolator`,
-      - the other optional keyword arguments *kwargs* are defined in
-        each TriRefiner concrete implementation;
-
-      and which returns (as a tuple) a refined triangular mesh and the
-      interpolated values of the field at the refined triangulation nodes.
-    """
-
-    def __init__(self, triangulation):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-
-class UniformTriRefiner(TriRefiner):
-    """
-    Uniform mesh refinement by recursive subdivisions.
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The encapsulated triangulation (to be refined)
-    """
-#    See Also
-#    --------
-#    :class:`~matplotlib.tri.CubicTriInterpolator` and
-#    :class:`~matplotlib.tri.TriAnalyzer`.
-#    """
-    def __init__(self, triangulation):
-        super().__init__(triangulation)
-
-    def refine_triangulation(self, return_tri_index=False, subdiv=3):
-        """
-        Compute an uniformly refined triangulation *refi_triangulation* of
-        the encapsulated :attr:`triangulation`.
-
-        This function refines the encapsulated triangulation by splitting each
-        father triangle into 4 child sub-triangles built on the edges midside
-        nodes, recursing *subdiv* times.  In the end, each triangle is hence
-        divided into ``4**subdiv`` child triangles.
-
-        Parameters
-        ----------
-        return_tri_index : bool, default: False
-            Whether an index table indicating the father triangle index of each
-            point is returned.
-        subdiv : int, default: 3
-            Recursion level for the subdivision.
-            Each triangle is divided into ``4**subdiv`` child triangles;
-            hence, the default results in 64 refined subtriangles for each
-            triangle of the initial triangulation.
-
-        Returns
-        -------
-        refi_triangulation : `~matplotlib.tri.Triangulation`
-            The refined triangulation.
-        found_index : int array
-            Index of the initial triangulation containing triangle, for each
-            point of *refi_triangulation*.
-            Returned only if *return_tri_index* is set to True.
-        """
-        refi_triangulation = self._triangulation
-        ntri = refi_triangulation.triangles.shape[0]
-
-        # Computes the triangulation ancestors numbers in the reference
-        # triangulation.
-        ancestors = np.arange(ntri, dtype=np.int32)
-        for _ in range(subdiv):
-            refi_triangulation, ancestors = self._refine_triangulation_once(
-                refi_triangulation, ancestors)
-        refi_npts = refi_triangulation.x.shape[0]
-        refi_triangles = refi_triangulation.triangles
-
-        # Now we compute found_index table if needed
-        if return_tri_index:
-            # We have to initialize found_index with -1 because some nodes
-            # may very well belong to no triangle at all, e.g., in case of
-            # Delaunay Triangulation with DuplicatePointWarning.
-            found_index = np.full(refi_npts, -1, dtype=np.int32)
-            tri_mask = self._triangulation.mask
-            if tri_mask is None:
-                found_index[refi_triangles] = np.repeat(ancestors,
-                                                        3).reshape(-1, 3)
-            else:
-                # There is a subtlety here: we want to avoid whenever possible
-                # that refined points container is a masked triangle (which
-                # would result in artifacts in plots).
-                # So we impose the numbering from masked ancestors first,
-                # then overwrite it with unmasked ancestor numbers.
-                ancestor_mask = tri_mask[ancestors]
-                found_index[refi_triangles[ancestor_mask, :]
-                            ] = np.repeat(ancestors[ancestor_mask],
-                                          3).reshape(-1, 3)
-                found_index[refi_triangles[~ancestor_mask, :]
-                            ] = np.repeat(ancestors[~ancestor_mask],
-                                          3).reshape(-1, 3)
-            return refi_triangulation, found_index
-        else:
-            return refi_triangulation
-
-    def refine_field(self, z, triinterpolator=None, subdiv=3):
-        """
-        Refine a field defined on the encapsulated triangulation.
-
-        Parameters
-        ----------
-        z : (npoints,) array-like
-            Values of the field to refine, defined at the nodes of the
-            encapsulated triangulation. (``n_points`` is the number of points
-            in the initial triangulation)
-        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
-            Interpolator used for field interpolation. If not specified,
-            a `~matplotlib.tri.CubicTriInterpolator` will be used.
-        subdiv : int, default: 3
-            Recursion level for the subdivision.
-            Each triangle is divided into ``4**subdiv`` child triangles.
-
-        Returns
-        -------
-        refi_tri : `~matplotlib.tri.Triangulation`
-             The returned refined triangulation.
-        refi_z : 1D array of length: *refi_tri* node count.
-             The returned interpolated field (at *refi_tri* nodes).
-        """
-        if triinterpolator is None:
-            interp = matplotlib.tri.CubicTriInterpolator(
-                self._triangulation, z)
-        else:
-            _api.check_isinstance(matplotlib.tri.TriInterpolator,
-                                  triinterpolator=triinterpolator)
-            interp = triinterpolator
-
-        refi_tri, found_index = self.refine_triangulation(
-            subdiv=subdiv, return_tri_index=True)
-        refi_z = interp._interpolate_multikeys(
-            refi_tri.x, refi_tri.y, tri_index=found_index)[0]
-        return refi_tri, refi_z
-
-    @staticmethod
-    def _refine_triangulation_once(triangulation, ancestors=None):
-        """
-        Refine a `.Triangulation` by splitting each triangle into 4
-        child-masked_triangles built on the edges midside nodes.
-
-        Masked triangles, if present, are also split, but their children
-        returned masked.
-
-        If *ancestors* is not provided, returns only a new triangulation:
-        child_triangulation.
-
-        If the array-like key table *ancestor* is given, it shall be of shape
-        (ntri,) where ntri is the number of *triangulation* masked_triangles.
-        In this case, the function returns
-        (child_triangulation, child_ancestors)
-        child_ancestors is defined so that the 4 child masked_triangles share
-        the same index as their father: child_ancestors.shape = (4 * ntri,).
-        """
-
-        x = triangulation.x
-        y = triangulation.y
-
-        #    According to tri.triangulation doc:
-        #         neighbors[i, j] is the triangle that is the neighbor
-        #         to the edge from point index masked_triangles[i, j] to point
-        #         index masked_triangles[i, (j+1)%3].
-        neighbors = triangulation.neighbors
-        triangles = triangulation.triangles
-        npts = np.shape(x)[0]
-        ntri = np.shape(triangles)[0]
-        if ancestors is not None:
-            ancestors = np.asarray(ancestors)
-            if np.shape(ancestors) != (ntri,):
-                raise ValueError(
-                    "Incompatible shapes provide for triangulation"
-                    ".masked_triangles and ancestors: {0} and {1}".format(
-                        np.shape(triangles), np.shape(ancestors)))
-
-        # Initiating tables refi_x and refi_y of the refined triangulation
-        # points
-        # hint: each apex is shared by 2 masked_triangles except the borders.
-        borders = np.sum(neighbors == -1)
-        added_pts = (3*ntri + borders) // 2
-        refi_npts = npts + added_pts
-        refi_x = np.zeros(refi_npts)
-        refi_y = np.zeros(refi_npts)
-
-        # First part of refi_x, refi_y is just the initial points
-        refi_x[:npts] = x
-        refi_y[:npts] = y
-
-        # Second part contains the edge midside nodes.
-        # Each edge belongs to 1 triangle (if border edge) or is shared by 2
-        # masked_triangles (interior edge).
-        # We first build 2 * ntri arrays of edge starting nodes (edge_elems,
-        # edge_apexes); we then extract only the masters to avoid overlaps.
-        # The so-called 'master' is the triangle with biggest index
-        # The 'slave' is the triangle with lower index
-        # (can be -1 if border edge)
-        # For slave and master we will identify the apex pointing to the edge
-        # start
-        edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
-        edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
-        edge_neighbors = neighbors[edge_elems, edge_apexes]
-        mask_masters = (edge_elems > edge_neighbors)
-
-        # Identifying the "masters" and adding to refi_x, refi_y vec
-        masters = edge_elems[mask_masters]
-        apex_masters = edge_apexes[mask_masters]
-        x_add = (x[triangles[masters, apex_masters]] +
-                 x[triangles[masters, (apex_masters+1) % 3]]) * 0.5
-        y_add = (y[triangles[masters, apex_masters]] +
-                 y[triangles[masters, (apex_masters+1) % 3]]) * 0.5
-        refi_x[npts:] = x_add
-        refi_y[npts:] = y_add
-
-        # Building the new masked_triangles; each old masked_triangles hosts
-        # 4 new masked_triangles
-        # there are 6 pts to identify per 'old' triangle, 3 new_pt_corner and
-        # 3 new_pt_midside
-        new_pt_corner = triangles
-
-        # What is the index in refi_x, refi_y of point at middle of apex iapex
-        #  of elem ielem ?
-        # If ielem is the apex master: simple count, given the way refi_x was
-        #  built.
-        # If ielem is the apex slave: yet we do not know; but we will soon
-        # using the neighbors table.
-        new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
-        cum_sum = npts
-        for imid in range(3):
-            mask_st_loc = (imid == apex_masters)
-            n_masters_loc = np.sum(mask_st_loc)
-            elem_masters_loc = masters[mask_st_loc]
-            new_pt_midside[:, imid][elem_masters_loc] = np.arange(
-                n_masters_loc, dtype=np.int32) + cum_sum
-            cum_sum += n_masters_loc
-
-        # Now dealing with slave elems.
-        # for each slave element we identify the master and then the inode
-        # once slave_masters is identified, slave_masters_apex is such that:
-        # neighbors[slaves_masters, slave_masters_apex] == slaves
-        mask_slaves = np.logical_not(mask_masters)
-        slaves = edge_elems[mask_slaves]
-        slaves_masters = edge_neighbors[mask_slaves]
-        diff_table = np.abs(neighbors[slaves_masters, :] -
-                            np.outer(slaves, np.ones(3, dtype=np.int32)))
-        slave_masters_apex = np.argmin(diff_table, axis=1)
-        slaves_apex = edge_apexes[mask_slaves]
-        new_pt_midside[slaves, slaves_apex] = new_pt_midside[
-            slaves_masters, slave_masters_apex]
-
-        # Builds the 4 child masked_triangles
-        child_triangles = np.empty([ntri*4, 3], dtype=np.int32)
-        child_triangles[0::4, :] = np.vstack([
-            new_pt_corner[:, 0], new_pt_midside[:, 0],
-            new_pt_midside[:, 2]]).T
-        child_triangles[1::4, :] = np.vstack([
-            new_pt_corner[:, 1], new_pt_midside[:, 1],
-            new_pt_midside[:, 0]]).T
-        child_triangles[2::4, :] = np.vstack([
-            new_pt_corner[:, 2], new_pt_midside[:, 2],
-            new_pt_midside[:, 1]]).T
-        child_triangles[3::4, :] = np.vstack([
-            new_pt_midside[:, 0], new_pt_midside[:, 1],
-            new_pt_midside[:, 2]]).T
-        child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
 
-        # Builds the child mask
-        if triangulation.mask is not None:
-            child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
 
-        if ancestors is None:
-            return child_triangulation
-        else:
-            return child_triangulation, np.repeat(ancestors, 4)
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/matplotlib/tri/tritools.py b/lib/matplotlib/tri/tritools.py
--- a/lib/matplotlib/tri/tritools.py
+++ b/lib/matplotlib/tri/tritools.py
@@ -1,263 +1,9 @@
-"""
-Tools for triangular grids.
-"""
-
-import numpy as np
-
+from ._tritools import *  # noqa: F401, F403
 from matplotlib import _api
-from matplotlib.tri import Triangulation
-
-
-class TriAnalyzer:
-    """
-    Define basic tools for triangular mesh analysis and improvement.
-
-    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
-    tools for mesh analysis and mesh improvement.
-
-    Attributes
-    ----------
-    scale_factors
-
-    Parameters
-    ----------
-    triangulation : `~matplotlib.tri.Triangulation`
-        The encapsulated triangulation to analyze.
-    """
-
-    def __init__(self, triangulation):
-        _api.check_isinstance(Triangulation, triangulation=triangulation)
-        self._triangulation = triangulation
-
-    @property
-    def scale_factors(self):
-        """
-        Factors to rescale the triangulation into a unit square.
-
-        Returns
-        -------
-        (float, float)
-            Scaling factors (kx, ky) so that the triangulation
-            ``[triangulation.x * kx, triangulation.y * ky]``
-            fits exactly inside a unit square.
-        """
-        compressed_triangles = self._triangulation.get_masked_triangles()
-        node_used = (np.bincount(np.ravel(compressed_triangles),
-                                 minlength=self._triangulation.x.size) != 0)
-        return (1 / np.ptp(self._triangulation.x[node_used]),
-                1 / np.ptp(self._triangulation.y[node_used]))
-
-    def circle_ratios(self, rescale=True):
-        """
-        Return a measure of the triangulation triangles flatness.
-
-        The ratio of the incircle radius over the circumcircle radius is a
-        widely used indicator of a triangle flatness.
-        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
-        triangles. Circle ratios below 0.01 denote very flat triangles.
-
-        To avoid unduly low values due to a difference of scale between the 2
-        axis, the triangular mesh can first be rescaled to fit inside a unit
-        square with `scale_factors` (Only if *rescale* is True, which is
-        its default value).
-
-        Parameters
-        ----------
-        rescale : bool, default: True
-            If True, internally rescale (based on `scale_factors`), so that the
-            (unmasked) triangles fit exactly inside a unit square mesh.
-
-        Returns
-        -------
-        masked array
-            Ratio of the incircle radius over the circumcircle radius, for
-            each 'rescaled' triangle of the encapsulated triangulation.
-            Values corresponding to masked triangles are masked out.
-
-        """
-        # Coords rescaling
-        if rescale:
-            (kx, ky) = self.scale_factors
-        else:
-            (kx, ky) = (1.0, 1.0)
-        pts = np.vstack([self._triangulation.x*kx,
-                         self._triangulation.y*ky]).T
-        tri_pts = pts[self._triangulation.triangles]
-        # Computes the 3 side lengths
-        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
-        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
-        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
-        a = np.hypot(a[:, 0], a[:, 1])
-        b = np.hypot(b[:, 0], b[:, 1])
-        c = np.hypot(c[:, 0], c[:, 1])
-        # circumcircle and incircle radii
-        s = (a+b+c)*0.5
-        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
-        # We have to deal with flat triangles with infinite circum_radius
-        bool_flat = (prod == 0.)
-        if np.any(bool_flat):
-            # Pathologic flow
-            ntri = tri_pts.shape[0]
-            circum_radius = np.empty(ntri, dtype=np.float64)
-            circum_radius[bool_flat] = np.inf
-            abc = a*b*c
-            circum_radius[~bool_flat] = abc[~bool_flat] / (
-                4.0*np.sqrt(prod[~bool_flat]))
-        else:
-            # Normal optimized flow
-            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
-        in_radius = (a*b*c) / (4.0*circum_radius*s)
-        circle_ratio = in_radius/circum_radius
-        mask = self._triangulation.mask
-        if mask is None:
-            return circle_ratio
-        else:
-            return np.ma.array(circle_ratio, mask=mask)
-
-    def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
-        """
-        Eliminate excessively flat border triangles from the triangulation.
-
-        Returns a mask *new_mask* which allows to clean the encapsulated
-        triangulation from its border-located flat triangles
-        (according to their :meth:`circle_ratios`).
-        This mask is meant to be subsequently applied to the triangulation
-        using `.Triangulation.set_mask`.
-        *new_mask* is an extension of the initial triangulation mask
-        in the sense that an initially masked triangle will remain masked.
-
-        The *new_mask* array is computed recursively; at each step flat
-        triangles are removed only if they share a side with the current mesh
-        border. Thus no new holes in the triangulated domain will be created.
-
-        Parameters
-        ----------
-        min_circle_ratio : float, default: 0.01
-            Border triangles with incircle/circumcircle radii ratio r/R will
-            be removed if r/R < *min_circle_ratio*.
-        rescale : bool, default: True
-            If True, first, internally rescale (based on `scale_factors`) so
-            that the (unmasked) triangles fit exactly inside a unit square
-            mesh.  This rescaling accounts for the difference of scale which
-            might exist between the 2 axis.
-
-        Returns
-        -------
-        array of bool
-            Mask to apply to encapsulated triangulation.
-            All the initially masked triangles remain masked in the
-            *new_mask*.
-
-        Notes
-        -----
-        The rationale behind this function is that a Delaunay
-        triangulation - of an unstructured set of points - sometimes contains
-        almost flat triangles at its border, leading to artifacts in plots
-        (especially for high-resolution contouring).
-        Masked with computed *new_mask*, the encapsulated
-        triangulation would contain no more unmasked border triangles
-        with a circle ratio below *min_circle_ratio*, thus improving the
-        mesh quality for subsequent plots or interpolation.
-        """
-        # Recursively computes the mask_current_borders, true if a triangle is
-        # at the border of the mesh OR touching the border through a chain of
-        # invalid aspect ratio masked_triangles.
-        ntri = self._triangulation.triangles.shape[0]
-        mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
-
-        current_mask = self._triangulation.mask
-        if current_mask is None:
-            current_mask = np.zeros(ntri, dtype=bool)
-        valid_neighbors = np.copy(self._triangulation.neighbors)
-        renum_neighbors = np.arange(ntri, dtype=np.int32)
-        nadd = -1
-        while nadd != 0:
-            # The active wavefront is the triangles from the border (unmasked
-            # but with a least 1 neighbor equal to -1
-            wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
-            # The element from the active wavefront will be masked if their
-            # circle ratio is bad.
-            added_mask = wavefront & mask_bad_ratio
-            current_mask = added_mask | current_mask
-            nadd = np.sum(added_mask)
-
-            # now we have to update the tables valid_neighbors
-            valid_neighbors[added_mask, :] = -1
-            renum_neighbors[added_mask] = -1
-            valid_neighbors = np.where(valid_neighbors == -1, -1,
-                                       renum_neighbors[valid_neighbors])
-
-        return np.ma.filled(current_mask, True)
-
-    def _get_compressed_triangulation(self):
-        """
-        Compress (if masked) the encapsulated triangulation.
-
-        Returns minimal-length triangles array (*compressed_triangles*) and
-        coordinates arrays (*compressed_x*, *compressed_y*) that can still
-        describe the unmasked triangles of the encapsulated triangulation.
-
-        Returns
-        -------
-        compressed_triangles : array-like
-            the returned compressed triangulation triangles
-        compressed_x : array-like
-            the returned compressed triangulation 1st coordinate
-        compressed_y : array-like
-            the returned compressed triangulation 2nd coordinate
-        tri_renum : int array
-            renumbering table to translate the triangle numbers from the
-            encapsulated triangulation into the new (compressed) renumbering.
-            -1 for masked triangles (deleted from *compressed_triangles*).
-        node_renum : int array
-            renumbering table to translate the point numbers from the
-            encapsulated triangulation into the new (compressed) renumbering.
-            -1 for unused points (i.e. those deleted from *compressed_x* and
-            *compressed_y*).
-
-        """
-        # Valid triangles and renumbering
-        tri_mask = self._triangulation.mask
-        compressed_triangles = self._triangulation.get_masked_triangles()
-        ntri = self._triangulation.triangles.shape[0]
-        if tri_mask is not None:
-            tri_renum = self._total_to_compress_renum(~tri_mask)
-        else:
-            tri_renum = np.arange(ntri, dtype=np.int32)
-
-        # Valid nodes and renumbering
-        valid_node = (np.bincount(np.ravel(compressed_triangles),
-                                  minlength=self._triangulation.x.size) != 0)
-        compressed_x = self._triangulation.x[valid_node]
-        compressed_y = self._triangulation.y[valid_node]
-        node_renum = self._total_to_compress_renum(valid_node)
-
-        # Now renumbering the valid triangles nodes
-        compressed_triangles = node_renum[compressed_triangles]
-
-        return (compressed_triangles, compressed_x, compressed_y, tri_renum,
-                node_renum)
-
-    @staticmethod
-    def _total_to_compress_renum(valid):
-        """
-        Parameters
-        ----------
-        valid : 1D bool array
-            Validity mask.
 
-        Returns
-        -------
-        int array
-            Array so that (`valid_array` being a compressed array
-            based on a `masked_array` with mask ~*valid*):
 
-            - For all i with valid[i] = True:
-              valid_array[renum[i]] = masked_array[i]
-            - For all i with valid[i] = False:
-              renum[i] = -1 (invalid value)
-        """
-        renum = np.full(np.size(valid), -1, dtype=np.int32)
-        n_valid = np.sum(valid)
-        renum[valid] = np.arange(n_valid, dtype=np.int32)
-        return renum
+_api.warn_deprecated(
+    "3.7",
+    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
+            f"be removed two minor releases later. All functionality is "
+            f"available via the top-level module matplotlib.tri")
diff --git a/lib/mpl_toolkits/mplot3d/axes3d.py b/lib/mpl_toolkits/mplot3d/axes3d.py
--- a/lib/mpl_toolkits/mplot3d/axes3d.py
+++ b/lib/mpl_toolkits/mplot3d/axes3d.py
@@ -32,7 +32,7 @@
 from matplotlib.axes import Axes
 from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
 from matplotlib.transforms import Bbox
-from matplotlib.tri.triangulation import Triangulation
+from matplotlib.tri._triangulation import Triangulation
 
 from . import art3d
 from . import proj3d
@@ -2153,7 +2153,7 @@ def tricontour(self, *args,
 
         Returns
         -------
-        matplotlib.tri.tricontour.TriContourSet
+        matplotlib.tri._tricontour.TriContourSet
         """
         had_data = self.has_data()
 
@@ -2246,7 +2246,7 @@ def tricontourf(self, *args, zdir='z', offset=None, **kwargs):
 
         Returns
         -------
-        matplotlib.tri.tricontour.TriContourSet
+        matplotlib.tri._tricontour.TriContourSet
         """
         had_data = self.has_data()
 

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_triangulation.py b/lib/matplotlib/tests/test_triangulation.py
--- a/lib/matplotlib/tests/test_triangulation.py
+++ b/lib/matplotlib/tests/test_triangulation.py
@@ -614,15 +614,15 @@ def poisson_sparse_matrix(n, m):
 
     # Instantiating a sparse Poisson matrix of size 48 x 48:
     (n, m) = (12, 4)
-    mat = mtri.triinterpolate._Sparse_Matrix_coo(*poisson_sparse_matrix(n, m))
+    mat = mtri._triinterpolate._Sparse_Matrix_coo(*poisson_sparse_matrix(n, m))
     mat.compress_csc()
     mat_dense = mat.to_dense()
     # Testing a sparse solve for all 48 basis vector
     for itest in range(n*m):
         b = np.zeros(n*m, dtype=np.float64)
         b[itest] = 1.
-        x, _ = mtri.triinterpolate._cg(A=mat, b=b, x0=np.zeros(n*m),
-                                       tol=1.e-10)
+        x, _ = mtri._triinterpolate._cg(A=mat, b=b, x0=np.zeros(n*m),
+                                        tol=1.e-10)
         assert_array_almost_equal(np.dot(mat_dense, x), b)
 
     # 2) Same matrix with inserting 2 rows - cols with null diag terms
@@ -635,16 +635,16 @@ def poisson_sparse_matrix(n, m):
     rows = np.concatenate([rows, [i_zero, i_zero-1, j_zero, j_zero-1]])
     cols = np.concatenate([cols, [i_zero-1, i_zero, j_zero-1, j_zero]])
     vals = np.concatenate([vals, [1., 1., 1., 1.]])
-    mat = mtri.triinterpolate._Sparse_Matrix_coo(vals, rows, cols,
-                                                 (n*m + 2, n*m + 2))
+    mat = mtri._triinterpolate._Sparse_Matrix_coo(vals, rows, cols,
+                                                  (n*m + 2, n*m + 2))
     mat.compress_csc()
     mat_dense = mat.to_dense()
     # Testing a sparse solve for all 50 basis vec
     for itest in range(n*m + 2):
         b = np.zeros(n*m + 2, dtype=np.float64)
         b[itest] = 1.
-        x, _ = mtri.triinterpolate._cg(A=mat, b=b, x0=np.ones(n*m + 2),
-                                       tol=1.e-10)
+        x, _ = mtri._triinterpolate._cg(A=mat, b=b, x0=np.ones(n * m + 2),
+                                        tol=1.e-10)
         assert_array_almost_equal(np.dot(mat_dense, x), b)
 
     # 3) Now a simple test that summation of duplicate (i.e. with same rows,
@@ -655,7 +655,7 @@ def poisson_sparse_matrix(n, m):
     cols = np.array([0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                     dtype=np.int32)
     dim = (3, 3)
-    mat = mtri.triinterpolate._Sparse_Matrix_coo(vals, rows, cols, dim)
+    mat = mtri._triinterpolate._Sparse_Matrix_coo(vals, rows, cols, dim)
     mat.compress_csc()
     mat_dense = mat.to_dense()
     assert_array_almost_equal(mat_dense, np.array([
@@ -678,7 +678,7 @@ def test_triinterpcubic_geom_weights():
         y_rot = -np.sin(theta)*x + np.cos(theta)*y
         triang = mtri.Triangulation(x_rot, y_rot, triangles)
         cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
-        dof_estimator = mtri.triinterpolate._DOF_estimator_geom(cubic_geom)
+        dof_estimator = mtri._triinterpolate._DOF_estimator_geom(cubic_geom)
         weights = dof_estimator.compute_geom_weights()
         # Testing for the 4 possibilities...
         sum_w[0, :] = np.sum(weights, 1) - 1

```


## Code snippets

### 1 - doc/conf.py:

Start line: 111, End line: 158

```python
_check_dependencies()


# Import only after checking for dependencies.
# gallery_order.py from the sphinxext folder provides the classes that
# allow custom ordering of sections and subsections of the gallery
import sphinxext.gallery_order as gallery_order

# The following import is only necessary to monkey patch the signature later on
from sphinx_gallery import gen_rst

# On Linux, prevent plt.show() from emitting a non-GUI backend warning.
os.environ.pop("DISPLAY", None)

autosummary_generate = True

# we should ignore warnings coming from importing deprecated modules for
# autodoc purposes, as this will disappear automatically when they are removed
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='importlib',  # used by sphinx.autodoc.importer
                        message=r'(\n|.)*module was deprecated.*')

autodoc_docstring_signature = True
autodoc_default_options = {'members': None, 'undoc-members': None}

# make sure to ignore warnings that stem from simply inspecting deprecated
# class-level attributes
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='sphinx.util.inspect')

nitpicky = True
# change this to True to update the allowed failures
missing_references_write_json = False
missing_references_warn_unused_ignores = False

intersphinx_mapping = {
    'Pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'cycler': ('https://matplotlib.org/cycler/', None),
    'dateutil': ('https://dateutil.readthedocs.io/en/stable/', None),
    'ipykernel': ('https://ipykernel.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pytest': ('https://pytest.org/en/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'tornado': ('https://www.tornadoweb.org/en/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}
```
### 2 - lib/matplotlib/pyplot.py:

Start line: 2956, End line: 2971

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.tripcolor)
def tripcolor(
        *args, alpha=1.0, norm=None, cmap=None, vmin=None, vmax=None,
        shading='flat', facecolors=None, **kwargs):
    __ret = gca().tripcolor(
        *args, alpha=alpha, norm=norm, cmap=cmap, vmin=vmin,
        vmax=vmax, shading=shading, facecolors=facecolors, **kwargs)
    sci(__ret)
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.triplot)
def triplot(*args, **kwargs):
    return gca().triplot(*args, **kwargs)
```
### 3 - lib/matplotlib/__init__.py:

Start line: 107, End line: 156

```python
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy
from packaging.version import parse as parse_version

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler


_log = logging.getLogger(__name__)

__bibtex__ = r"""@Article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing in Science \& Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {90--95},
  abstract  = {Matplotlib is a 2D graphics package used for Python
  for application development, interactive scripting, and
  publication-quality image generation across user
  interfaces and operating systems.},
  publisher = {IEEE COMPUTER SOC},
  year      = 2007
}"""

# modelled after sys.version_info
_VersionInfo = namedtuple('_VersionInfo',
                          'major, minor, micro, releaselevel, serial')
```
### 4 - lib/matplotlib/__init__.py:

Start line: 1, End line: 106

```python
"""
An object-oriented plotting library.

A procedural interface is provided by the companion pyplot module,
which may be imported directly, e.g.::

    import matplotlib.pyplot as plt

or using ipython::

    ipython

at your terminal, followed by::

    In [1]: %matplotlib
    In [2]: import matplotlib.pyplot as plt

at the ipython shell prompt.

For the most part, direct use of the explicit object-oriented library is
encouraged when programming; the implicit pyplot interface is primarily for
working interactively. The exceptions to this suggestion are the pyplot
functions `.pyplot.figure`, `.pyplot.subplot`, `.pyplot.subplots`, and
`.pyplot.savefig`, which can greatly simplify scripting.  See
:ref:`api_interfaces` for an explanation of the tradeoffs between the implicit
and explicit interfaces.

Modules include:

    :mod:`matplotlib.axes`
        The `~.axes.Axes` class.  Most pyplot functions are wrappers for
        `~.axes.Axes` methods.  The axes module is the highest level of OO
        access to the library.

    :mod:`matplotlib.figure`
        The `.Figure` class.

    :mod:`matplotlib.artist`
        The `.Artist` base class for all classes that draw things.

    :mod:`matplotlib.lines`
        The `.Line2D` class for drawing lines and markers.

    :mod:`matplotlib.patches`
        Classes for drawing polygons.

    :mod:`matplotlib.text`
        The `.Text` and `.Annotation` classes.

    :mod:`matplotlib.image`
        The `.AxesImage` and `.FigureImage` classes.

    :mod:`matplotlib.collections`
        Classes for efficient drawing of groups of lines or polygons.

    :mod:`matplotlib.colors`
        Color specifications and making colormaps.

    :mod:`matplotlib.cm`
        Colormaps, and the `.ScalarMappable` mixin class for providing color
        mapping functionality to other classes.

    :mod:`matplotlib.ticker`
        Calculation of tick mark locations and formatting of tick labels.

    :mod:`matplotlib.backends`
        A subpackage with modules for various GUI libraries and output formats.

The base matplotlib namespace includes:

    `~matplotlib.rcParams`
        Default configuration settings; their defaults may be overridden using
        a :file:`matplotlibrc` file.

    `~matplotlib.use`
        Setting the Matplotlib backend.  This should be called before any
        figure is created, because it is not possible to switch between
        different GUI backends after that.

The following environment variables can be used to customize the behavior::

    .. envvar:: MPLBACKEND

      This optional variable can be set to choose the Matplotlib backend. See
      :ref:`what-is-a-backend`.

    .. envvar:: MPLCONFIGDIR

      This is the directory used to store user customizations to
      Matplotlib, as well as some caches to improve performance. If
      :envvar:`MPLCONFIGDIR` is not defined, :file:`{HOME}/.config/matplotlib`
      and :file:`{HOME}/.cache/matplotlib` are used on Linux, and
      :file:`{HOME}/.matplotlib` on other platforms, if they are
      writable. Otherwise, the Python standard library's `tempfile.gettempdir`
      is used to find a base directory in which the :file:`matplotlib`
      subdirectory is created.

Matplotlib was initially written by John D. Hunter (1968-2012) and is now
developed and maintained by a host of others.

Occasionally the internal documentation (python docstrings) will refer
to MATLAB®, a registered trademark of The MathWorks, Inc.

"""

import atexit
```
### 5 - lib/matplotlib/pyplot.py:

Start line: 92, End line: 119

```python
def _copy_docstring_and_deprecators(method, func=None):
    if func is None:
        return functools.partial(_copy_docstring_and_deprecators, method)
    decorators = [_docstring.copy(method)]
    # Check whether the definition of *method* includes @_api.rename_parameter
    # or @_api.make_keyword_only decorators; if so, propagate them to the
    # pyplot wrapper as well.
    while getattr(method, "__wrapped__", None) is not None:
        decorator = _api.deprecation.DECORATORS.get(method)
        if decorator:
            decorators.append(decorator)
        method = method.__wrapped__
    for decorator in decorators[::-1]:
        func = decorator(func)
    return func


## Global ##


# The state controlled by {,un}install_repl_displayhook().
_ReplDisplayHook = Enum("_ReplDisplayHook", ["NONE", "PLAIN", "IPYTHON"])
_REPL_DISPLAYHOOK = _ReplDisplayHook.NONE


def _draw_all_if_interactive():
    if matplotlib.is_interactive():
        draw_all()
```
### 6 - doc/conf.py:

Start line: 635, End line: 708

```python
if link_github:
    import inspect
    from packaging.version import parse

    extensions.append('sphinx.ext.linkcode')

    def linkcode_resolve(domain, info):
        """
        Determine the URL corresponding to Python object
        """
        if domain != 'py':
            return None

        modname = info['module']
        fullname = info['fullname']

        submod = sys.modules.get(modname)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split('.'):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        if inspect.isfunction(obj):
            obj = inspect.unwrap(obj)
        try:
            fn = inspect.getsourcefile(obj)
        except TypeError:
            fn = None
        if not fn or fn.endswith('__init__.py'):
            try:
                fn = inspect.getsourcefile(sys.modules[obj.__module__])
            except (TypeError, AttributeError, KeyError):
                fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        linespec = (f"#L{lineno:d}-L{lineno + len(source) - 1:d}"
                    if lineno else "")

        startdir = Path(matplotlib.__file__).parent.parent
        fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, '/')

        if not fn.startswith(('matplotlib/', 'mpl_toolkits/')):
            return None

        version = parse(matplotlib.__version__)
        tag = 'main' if version.is_devrelease else f'v{version.public}'
        return ("https://github.com/matplotlib/matplotlib/blob"
                f"/{tag}/lib/{fn}{linespec}")
else:
    extensions.append('sphinx.ext.viewcode')


# -----------------------------------------------------------------------------
# Sphinx setup
# -----------------------------------------------------------------------------
def setup(app):
    if any(st in version for st in ('post', 'dev', 'alpha', 'beta')):
        bld_type = 'dev'
    else:
        bld_type = 'rel'
    app.add_config_value('releaselevel', bld_type, 'env')
    app.connect('html-page-context', add_html_cache_busting, priority=1000)
```
### 7 - lib/matplotlib/cbook/__init__.py:

Start line: 33, End line: 43

```python
@_api.caching_module_getattr
class __getattr__:
    # module-level deprecations
    MatplotlibDeprecationWarning = _api.deprecated(
        "3.6", obj_type="",
        alternative="matplotlib.MatplotlibDeprecationWarning")(
        property(lambda self: _api.deprecation.MatplotlibDeprecationWarning))
    mplDeprecation = _api.deprecated(
        "3.6", obj_type="",
        alternative="matplotlib.MatplotlibDeprecationWarning")(
        property(lambda self: _api.deprecation.MatplotlibDeprecationWarning))
```
### 8 - lib/matplotlib/__init__.py:

Start line: 203, End line: 229

```python
@_api.caching_module_getattr
class __getattr__:
    __version__ = property(lambda self: _get_version())
    __version_info__ = property(
        lambda self: _parse_to_version_info(self.__version__))
    # module-level deprecations
    URL_REGEX = _api.deprecated("3.5", obj_type="")(property(
        lambda self: re.compile(r'^http://|^https://|^ftp://|^file:')))


def _check_versions():

    # Quickfix to ensure Microsoft Visual C++ redistributable
    # DLLs are loaded before importing kiwisolver
    from . import ft2font

    for modname, minver in [
            ("cycler", "0.10"),
            ("dateutil", "2.7"),
            ("kiwisolver", "1.0.1"),
            ("numpy", "1.19"),
            ("pyparsing", "2.3.1"),
    ]:
        module = importlib.import_module(modname)
        if parse_version(module.__version__) < parse_version(minver):
            raise ImportError(f"Matplotlib requires {modname}>={minver}; "
                              f"you have {module.__version__}")
```
### 9 - lib/matplotlib/tri/tripcolor.py:

Start line: 68, End line: 155

```python
def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
              vmax=None, shading='flat', facecolors=None, **kwargs):
    # ... other code
    if facecolors is not None:
        if args:
            _api.warn_external(
                "Positional parameter c has no effect when the keyword "
                "facecolors is given")
        point_colors = None
        if len(facecolors) != len(tri.triangles):
            raise ValueError("The length of facecolors must match the number "
                             "of triangles")
    else:
        # Color from positional parameter c
        if not args:
            raise TypeError(
                "tripcolor() missing 1 required positional argument: 'c'; or "
                "1 required keyword-only argument: 'facecolors'")
        elif len(args) > 1:
            _api.warn_deprecated(
                "3.6", message=f"Additional positional parameters "
                f"{args[1:]!r} are ignored; support for them is deprecated "
                f"since %(since)s and will be removed %(removal)s")
        c = np.asarray(args[0])
        if len(c) == len(tri.x):
            # having this before the len(tri.triangles) comparison gives
            # precedence to nodes if there are as many nodes as triangles
            point_colors = c
            facecolors = None
        elif len(c) == len(tri.triangles):
            point_colors = None
            facecolors = c
        else:
            raise ValueError('The length of c must match either the number '
                             'of points or the number of triangles')

    # Handling of linewidths, shading, edgecolors and antialiased as
    # in Axes.pcolor
    linewidths = (0.25,)
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    kwargs.setdefault('linewidths', linewidths)

    edgecolors = 'none'
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    ec = kwargs.setdefault('edgecolors', edgecolors)

    if 'antialiased' in kwargs:
        kwargs['antialiaseds'] = kwargs.pop('antialiased')
    if 'antialiaseds' not in kwargs and ec.lower() == "none":
        kwargs['antialiaseds'] = False

    _api.check_isinstance((Normalize, None), norm=norm)
    if shading == 'gouraud':
        if facecolors is not None:
            raise ValueError(
                "shading='gouraud' can only be used when the colors "
                "are specified at the points, not at the faces.")
        collection = TriMesh(tri, alpha=alpha, array=point_colors,
                             cmap=cmap, norm=norm, **kwargs)
    else:  # 'flat'
        # Vertices of triangles.
        maskedTris = tri.get_masked_triangles()
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)

        # Color values.
        if facecolors is None:
            # One color per triangle, the mean of the 3 vertex color values.
            colors = point_colors[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            # Remove color values of masked triangles.
            colors = facecolors[~tri.mask]
        else:
            colors = facecolors
        collection = PolyCollection(verts, alpha=alpha, array=colors,
                                    cmap=cmap, norm=norm, **kwargs)

    collection._scale_norm(norm, vmin, vmax)
    ax.grid(False)

    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection
```
### 10 - plot_types/unstructured/tripcolor.py:

Start line: 1, End line: 28

```python
"""
==================
tripcolor(x, y, z)
==================

See `~matplotlib.axes.Axes.tripcolor`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')

# make data:
np.random.seed(1)
x = np.random.uniform(-3, 3, 256)
y = np.random.uniform(-3, 3, 256)
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

# plot:
fig, ax = plt.subplots()

ax.plot(x, y, 'o', markersize=2, color='grey')
ax.tripcolor(x, y, z)

ax.set(xlim=(-3, 3), ylim=(-3, 3))

plt.show()
```
### 51 - lib/matplotlib/tri/triangulation.py:

Start line: 164, End line: 185

```python
class Triangulation:

    @staticmethod
    def _extract_triangulation_params(args, kwargs):
        x, y, *args = args
        # Check triangles in kwargs then args.
        triangles = kwargs.pop('triangles', None)
        from_args = False
        if triangles is None and args:
            triangles = args[0]
            from_args = True
        if triangles is not None:
            try:
                triangles = np.asarray(triangles, dtype=np.int32)
            except ValueError:
                triangles = None
        if triangles is not None and (triangles.ndim != 2 or
                                      triangles.shape[1] != 3):
            triangles = None
        if triangles is not None and from_args:
            args = args[1:]  # Consumed first item in args.
        # Check for mask in kwargs.
        mask = kwargs.pop('mask', None)
        return x, y, triangles, mask, args, kwargs
```
