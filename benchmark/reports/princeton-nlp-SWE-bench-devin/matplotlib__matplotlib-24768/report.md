# matplotlib__matplotlib-24768

| **matplotlib/matplotlib** | `ecf6e26f0b0241bdc80466e13ee0c13a0c12f412` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/lib/matplotlib/axes/_base.py b/lib/matplotlib/axes/_base.py
--- a/lib/matplotlib/axes/_base.py
+++ b/lib/matplotlib/axes/_base.py
@@ -3127,23 +3127,23 @@ def draw(self, renderer):
 
         if (rasterization_zorder is not None and
                 artists and artists[0].zorder < rasterization_zorder):
-            renderer.start_rasterizing()
-            artists_rasterized = [a for a in artists
-                                  if a.zorder < rasterization_zorder]
-            artists = [a for a in artists
-                       if a.zorder >= rasterization_zorder]
+            split_index = np.searchsorted(
+                [art.zorder for art in artists],
+                rasterization_zorder, side='right'
+            )
+            artists_rasterized = artists[:split_index]
+            artists = artists[split_index:]
         else:
             artists_rasterized = []
 
-        # the patch draws the background rectangle -- the frame below
-        # will draw the edges
         if self.axison and self._frameon:
-            self.patch.draw(renderer)
+            if artists_rasterized:
+                artists_rasterized = [self.patch] + artists_rasterized
+            else:
+                artists = [self.patch] + artists
 
         if artists_rasterized:
-            for a in artists_rasterized:
-                a.draw(renderer)
-            renderer.stop_rasterizing()
+            _draw_rasterized(self.figure, artists_rasterized, renderer)
 
         mimage._draw_list_compositing_images(
             renderer, self, artists, self.figure.suppressComposite)
@@ -4636,3 +4636,60 @@ def _label_outer_yaxis(self, *, check_patch):
             self.yaxis.set_tick_params(which="both", labelright=False)
             if self.yaxis.offsetText.get_position()[0] == 1:
                 self.yaxis.offsetText.set_visible(False)
+
+
+def _draw_rasterized(figure, artists, renderer):
+    """
+    A helper function for rasterizing the list of artists.
+
+    The bookkeeping to track if we are or are not in rasterizing mode
+    with the mixed-mode backends is relatively complicated and is now
+    handled in the matplotlib.artist.allow_rasterization decorator.
+
+    This helper defines the absolute minimum methods and attributes on a
+    shim class to be compatible with that decorator and then uses it to
+    rasterize the list of artists.
+
+    This is maybe too-clever, but allows us to re-use the same code that is
+    used on normal artists to participate in the "are we rasterizing"
+    accounting.
+
+    Please do not use this outside of the "rasterize below a given zorder"
+    functionality of Axes.
+
+    Parameters
+    ----------
+    figure : matplotlib.figure.Figure
+        The figure all of the artists belong to (not checked).  We need this
+        because we can at the figure level suppress composition and insert each
+        rasterized artist as its own image.
+
+    artists : List[matplotlib.artist.Artist]
+        The list of Artists to be rasterized.  These are assumed to all
+        be in the same Figure.
+
+    renderer : matplotlib.backendbases.RendererBase
+        The currently active renderer
+
+    Returns
+    -------
+    None
+
+    """
+    class _MinimalArtist:
+        def get_rasterized(self):
+            return True
+
+        def get_agg_filter(self):
+            return None
+
+        def __init__(self, figure, artists):
+            self.figure = figure
+            self.artists = artists
+
+        @martist.allow_rasterization
+        def draw(self, renderer):
+            for a in self.artists:
+                a.draw(renderer)
+
+    return _MinimalArtist(figure, artists).draw(renderer)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/axes/_base.py | 3130 | 3146 | - | - | -
| lib/matplotlib/axes/_base.py | 4639 | 4639 | - | - | -


## Problem Statement

```
[Bug]: pcolormesh(rasterized=True) conflicts with set_rasterization_zorder()
### Bug summary

According to the [documentation](https://matplotlib.org/stable/gallery/misc/rasterization_demo.html), a color plot can be rasterized in two ways:

* `pyplot.pcolormesh(…, rasterized=True)`
* `pyplot.gca().set_rasterization_zorder(…)`

The two ways cannot be used together.

### Code for reproduction

\`\`\`python
import math
import numpy
import numpy.random
import matplotlib
from matplotlib import pyplot

matplotlib.use('agg')

r = numpy.linspace(1, 10, 10+1)
p = numpy.linspace(-math.pi, math.pi, 36+1)
r, p = numpy.meshgrid(r, p)
x, y = r*numpy.cos(p), r*numpy.sin(p)
s = tuple(s-1 for s in x.shape)
z = numpy.random.default_rng(0).uniform(size=s)

pyplot.pcolormesh(x, y, z, rasterized=True, zorder=-11)
pyplot.gca().set_rasterization_zorder(-10)
pyplot.annotate(
  matplotlib.__version__,
  (0.5, 0.5), (0.5, 0.5), 'axes fraction', 'axes fraction',
  ha='center', va='center')

pyplot.savefig('test.pdf')
\`\`\`


### Actual outcome

\`\`\`
Traceback (most recent call last):
  File "test.py", line 23, in <module>
    pyplot.savefig('test.pdf')
  File "/home/edwin/matplotlib/lib/matplotlib/pyplot.py", line 954, in savefig
    res = fig.savefig(*args, **kwargs)
  File "/home/edwin/matplotlib/lib/matplotlib/figure.py", line 3273, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "/home/edwin/matplotlib/lib/matplotlib/backend_bases.py", line 2357, in print_figure
    result = print_method(
  File "/home/edwin/matplotlib/lib/matplotlib/backend_bases.py", line 2223, in <lambda>
    print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
  File "/home/edwin/matplotlib/lib/matplotlib/backends/backend_pdf.py", line 2815, in print_pdf
    self.figure.draw(renderer)
  File "/home/edwin/matplotlib/lib/matplotlib/artist.py", line 74, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "/home/edwin/matplotlib/lib/matplotlib/artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "/home/edwin/matplotlib/lib/matplotlib/figure.py", line 3070, in draw
    mimage._draw_list_compositing_images(
  File "/home/edwin/matplotlib/lib/matplotlib/image.py", line 131, in _draw_list_compositing_images
    a.draw(renderer)
  File "/home/edwin/matplotlib/lib/matplotlib/artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "/home/edwin/matplotlib/lib/matplotlib/axes/_base.py", line 3151, in draw
    mimage._draw_list_compositing_images(
  File "/home/edwin/matplotlib/lib/matplotlib/image.py", line 131, in _draw_list_compositing_images
    a.draw(renderer)
  File "/home/edwin/matplotlib/lib/matplotlib/artist.py", line 45, in draw_wrapper
    renderer.stop_rasterizing()
  File "/home/edwin/matplotlib/lib/matplotlib/backends/backend_mixed.py", line 97, in stop_rasterizing
    img = np.asarray(self._raster_renderer.buffer_rgba())
AttributeError: 'NoneType' object has no attribute 'buffer_rgba'
\`\`\`

### Expected outcome

![](https://user-images.githubusercontent.com/906137/197075452-25ed77c6-d343-480d-9396-0f776e1d124e.png)

### Additional information

The bug appears in version 3.5.1 and commit 2d18bba0ea0e9fb9ccab508fa0a60ffc5946771b, but not version 3.1.2.

The most immediate cause seems to be reentrance tracking being dropped from `MixedModeRenderer.start_rasterizing()` and `MixedModeRenderer.stop_rasterizing()` in commit b6a273989ffc8ef3889fe16ee61d40b24f79c3e6:

https://github.com/matplotlib/matplotlib/blob/b6a273989ffc8ef3889fe16ee61d40b24f79c3e6/lib/matplotlib/backends/backend_mixed.py#L87-L88
https://github.com/matplotlib/matplotlib/blob/b6a273989ffc8ef3889fe16ee61d40b24f79c3e6/lib/matplotlib/backends/backend_mixed.py#L116

However, these are probably not the right places to fix this bug.

### Operating system

Ubuntu 20.04, 22.04

### Matplotlib Version

3.1.2, 3.5.1, 3.7.0.dev447+g2d18bba0ea

### Matplotlib Backend

agg

### Python version

3.8.10, 3.10.6

### Jupyter version

_No response_

### Installation

git checkout

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/misc/rasterization_demo.py | 1 | 74| 762 | 762 | 928 | 
| 2 | 1 examples/misc/rasterization_demo.py | 75 | 95| 166 | 928 | 928 | 
| 3 | 2 lib/matplotlib/artist.py | 24 | 42| 194 | 1122 | 14593 | 
| 4 | 3 lib/matplotlib/backends/backend_mixed.py | 87 | 120| 305 | 1427 | 15583 | 
| 5 | 4 lib/matplotlib/backends/backend_agg.py | 109 | 187| 755 | 2182 | 20376 | 
| 6 | 4 lib/matplotlib/backends/backend_mixed.py | 71 | 85| 158 | 2340 | 20376 | 
| 7 | 5 examples/images_contours_and_fields/pcolor_demo.py | 1 | 97| 790 | 3130 | 21423 | 
| 8 | 6 lib/matplotlib/backends/backend_ps.py | 442 | 481| 366 | 3496 | 33689 | 
| 9 | 7 examples/misc/zorder_demo.py | 1 | 76| 715 | 4211 | 34404 | 
| 10 | 7 lib/matplotlib/backends/backend_ps.py | 712 | 756| 436 | 4647 | 34404 | 
| 11 | 8 lib/matplotlib/backends/backend_pgf.py | 654 | 697| 532 | 5179 | 44438 | 
| 12 | 9 lib/mpl_toolkits/mplot3d/axis3d.py | 498 | 531| 395 | 5574 | 50370 | 
| 13 | 9 lib/matplotlib/artist.py | 938 | 959| 165 | 5739 | 50370 | 
| 14 | 10 examples/images_contours_and_fields/quadmesh_demo.py | 1 | 51| 450 | 6189 | 50820 | 
| 15 | 11 lib/matplotlib/backends/backend_pdf.py | 2006 | 2046| 368 | 6557 | 75596 | 
| 16 | 12 examples/images_contours_and_fields/pcolormesh_grids.py | 81 | 128| 484 | 7041 | 76879 | 
| 17 | 12 lib/matplotlib/backends/backend_pdf.py | 2118 | 2158| 391 | 7432 | 76879 | 
| 18 | 13 examples/images_contours_and_fields/image_antialiasing.py | 1 | 69| 760 | 8192 | 78177 | 
| 19 | 13 examples/images_contours_and_fields/pcolormesh_grids.py | 1 | 80| 799 | 8991 | 78177 | 
| 20 | 14 lib/matplotlib/image.py | 1251 | 1285| 441 | 9432 | 95333 | 
| 21 | 15 lib/matplotlib/axes/_axes.py | 5231 | 5795| 5723 | 15155 | 170003 | 
| 22 | 15 lib/matplotlib/backends/backend_pdf.py | 2160 | 2198| 357 | 15512 | 170003 | 
| 23 | 15 lib/matplotlib/backends/backend_agg.py | 242 | 257| 176 | 15688 | 170003 | 
| 24 | 15 lib/matplotlib/backends/backend_pgf.py | 410 | 446| 435 | 16123 | 170003 | 
| 25 | 15 lib/matplotlib/backends/backend_pgf.py | 448 | 499| 665 | 16788 | 170003 | 
| 26 | 15 lib/matplotlib/backends/backend_agg.py | 59 | 107| 480 | 17268 | 170003 | 
| 27 | 16 examples/showcase/mandelbrot.py | 30 | 74| 509 | 17777 | 170751 | 
| 28 | 16 lib/matplotlib/backends/backend_pgf.py | 616 | 652| 441 | 18218 | 170751 | 
| 29 | 17 lib/matplotlib/backends/backend_svg.py | 818 | 934| 1000 | 19218 | 182338 | 


## Missing Patch Files

 * 1: lib/matplotlib/axes/_base.py

## Patch

```diff
diff --git a/lib/matplotlib/axes/_base.py b/lib/matplotlib/axes/_base.py
--- a/lib/matplotlib/axes/_base.py
+++ b/lib/matplotlib/axes/_base.py
@@ -3127,23 +3127,23 @@ def draw(self, renderer):
 
         if (rasterization_zorder is not None and
                 artists and artists[0].zorder < rasterization_zorder):
-            renderer.start_rasterizing()
-            artists_rasterized = [a for a in artists
-                                  if a.zorder < rasterization_zorder]
-            artists = [a for a in artists
-                       if a.zorder >= rasterization_zorder]
+            split_index = np.searchsorted(
+                [art.zorder for art in artists],
+                rasterization_zorder, side='right'
+            )
+            artists_rasterized = artists[:split_index]
+            artists = artists[split_index:]
         else:
             artists_rasterized = []
 
-        # the patch draws the background rectangle -- the frame below
-        # will draw the edges
         if self.axison and self._frameon:
-            self.patch.draw(renderer)
+            if artists_rasterized:
+                artists_rasterized = [self.patch] + artists_rasterized
+            else:
+                artists = [self.patch] + artists
 
         if artists_rasterized:
-            for a in artists_rasterized:
-                a.draw(renderer)
-            renderer.stop_rasterizing()
+            _draw_rasterized(self.figure, artists_rasterized, renderer)
 
         mimage._draw_list_compositing_images(
             renderer, self, artists, self.figure.suppressComposite)
@@ -4636,3 +4636,60 @@ def _label_outer_yaxis(self, *, check_patch):
             self.yaxis.set_tick_params(which="both", labelright=False)
             if self.yaxis.offsetText.get_position()[0] == 1:
                 self.yaxis.offsetText.set_visible(False)
+
+
+def _draw_rasterized(figure, artists, renderer):
+    """
+    A helper function for rasterizing the list of artists.
+
+    The bookkeeping to track if we are or are not in rasterizing mode
+    with the mixed-mode backends is relatively complicated and is now
+    handled in the matplotlib.artist.allow_rasterization decorator.
+
+    This helper defines the absolute minimum methods and attributes on a
+    shim class to be compatible with that decorator and then uses it to
+    rasterize the list of artists.
+
+    This is maybe too-clever, but allows us to re-use the same code that is
+    used on normal artists to participate in the "are we rasterizing"
+    accounting.
+
+    Please do not use this outside of the "rasterize below a given zorder"
+    functionality of Axes.
+
+    Parameters
+    ----------
+    figure : matplotlib.figure.Figure
+        The figure all of the artists belong to (not checked).  We need this
+        because we can at the figure level suppress composition and insert each
+        rasterized artist as its own image.
+
+    artists : List[matplotlib.artist.Artist]
+        The list of Artists to be rasterized.  These are assumed to all
+        be in the same Figure.
+
+    renderer : matplotlib.backendbases.RendererBase
+        The currently active renderer
+
+    Returns
+    -------
+    None
+
+    """
+    class _MinimalArtist:
+        def get_rasterized(self):
+            return True
+
+        def get_agg_filter(self):
+            return None
+
+        def __init__(self, figure, artists):
+            self.figure = figure
+            self.artists = artists
+
+        @martist.allow_rasterization
+        def draw(self, renderer):
+            for a in self.artists:
+                a.draw(renderer)
+
+    return _MinimalArtist(figure, artists).draw(renderer)

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_axes.py b/lib/matplotlib/tests/test_axes.py
--- a/lib/matplotlib/tests/test_axes.py
+++ b/lib/matplotlib/tests/test_axes.py
@@ -8449,3 +8449,11 @@ def get_next_color():
         c = 'red\n'
         mpl.axes.Axes._parse_scatter_color_args(
             c, None, kwargs={}, xsize=2, get_next_color_func=get_next_color)
+
+
+def test_zorder_and_explicit_rasterization():
+    fig, ax = plt.subplots()
+    ax.set_rasterization_zorder(5)
+    ln, = ax.plot(range(5), rasterized=True, zorder=1)
+    with io.BytesIO() as b:
+        fig.savefig(b, format='pdf')

```


## Code snippets

### 1 - examples/misc/rasterization_demo.py:

Start line: 1, End line: 74

```python
"""
=================================
Rasterization for vector graphics
=================================

Rasterization converts vector graphics into a raster image (pixels). It can
speed up rendering and produce smaller files for large data sets, but comes
at the cost of a fixed resolution.

Whether rasterization should be used can be specified per artist.  This can be
useful to reduce the file size of large artists, while maintaining the
advantages of vector graphics for other artists such as the axes
and text.  For instance a complicated `~.Axes.pcolormesh` or
`~.Axes.contourf` can be made significantly simpler by rasterizing.
Setting rasterization only affects vector backends such as PDF, SVG, or PS.

Rasterization is disabled by default. There are two ways to enable it, which
can also be combined:

- Set `~.Artist.set_rasterized` on individual artists, or use the keyword
  argument *rasterized* when creating the artist.
- Set `.Axes.set_rasterization_zorder` to rasterize all artists with a zorder
  less than the given value.

The storage size and the resolution of the rasterized artist is determined by
its physical size and the value of the ``dpi`` parameter passed to
`~.Figure.savefig`.

.. note::

    The image of this example shown in the HTML documentation is not a vector
    graphic. Therefore, it cannot illustrate the rasterization effect. Please
    run this example locally and check the generated graphics files.

"""

import numpy as np
import matplotlib.pyplot as plt

d = np.arange(100).reshape(10, 10)  # the values to be color-mapped
x, y = np.meshgrid(np.arange(11), np.arange(11))

theta = 0.25*np.pi
xx = x*np.cos(theta) - y*np.sin(theta)  # rotate x by -theta
yy = x*np.sin(theta) + y*np.cos(theta)  # rotate y by -theta

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)

# pcolormesh without rasterization
ax1.set_aspect(1)
ax1.pcolormesh(xx, yy, d)
ax1.set_title("No Rasterization")

# pcolormesh with rasterization; enabled by keyword argument
ax2.set_aspect(1)
ax2.set_title("Rasterization")
ax2.pcolormesh(xx, yy, d, rasterized=True)

# pcolormesh with an overlaid text without rasterization
ax3.set_aspect(1)
ax3.pcolormesh(xx, yy, d)
ax3.text(0.5, 0.5, "Text", alpha=0.2,
         va="center", ha="center", size=50, transform=ax3.transAxes)
ax3.set_title("No Rasterization")

# pcolormesh with an overlaid text without rasterization; enabled by zorder.
# Setting the rasterization zorder threshold to 0 and a negative zorder on the
# pcolormesh rasterizes it. All artists have a non-negative zorder by default,
# so they (e.g. the text here) are not affected.
ax4.set_aspect(1)
m = ax4.pcolormesh(xx, yy, d, zorder=-10)
ax4.text(0.5, 0.5, "Text", alpha=0.2,
         va="center", ha="center", size=50, transform=ax4.transAxes)
ax4.set_rasterization_zorder(0)
```
### 2 - examples/misc/rasterization_demo.py:

Start line: 75, End line: 95

```python
ax4.set_title("Rasterization z$<-10$")

# Save files in pdf and eps format
plt.savefig("test_rasterization.pdf", dpi=150)
plt.savefig("test_rasterization.eps", dpi=150)

if not plt.rcParams["text.usetex"]:
    plt.savefig("test_rasterization.svg", dpi=150)
    # svg backend currently ignores the dpi

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.artist.Artist.set_rasterized`
#    - `matplotlib.axes.Axes.set_rasterization_zorder`
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
```
### 3 - lib/matplotlib/artist.py:

Start line: 24, End line: 42

```python
def _prevent_rasterization(draw):
    # We assume that by default artists are not allowed to rasterize (unless
    # its draw method is explicitly decorated). If it is being drawn after a
    # rasterized artist and it has reached a raster_depth of 0, we stop
    # rasterization so that it does not affect the behavior of normal artist
    # (e.g., change in dpi).

    @wraps(draw)
    def draw_wrapper(artist, renderer, *args, **kwargs):
        if renderer._raster_depth == 0 and renderer._rasterizing:
            # Only stop when we are not in a rasterized parent
            # and something has been rasterized since last stop.
            renderer.stop_rasterizing()
            renderer._rasterizing = False

        return draw(artist, renderer, *args, **kwargs)

    draw_wrapper._supports_rasterization = False
    return draw_wrapper
```
### 4 - lib/matplotlib/backends/backend_mixed.py:

Start line: 87, End line: 120

```python
class MixedModeRenderer:

    def stop_rasterizing(self):
        """
        Exit "raster" mode.  All of the drawing that was done since
        the last `start_rasterizing` call will be copied to the
        vector backend by calling draw_image.
        """

        self._renderer = self._vector_renderer

        height = self._height * self.dpi
        img = np.asarray(self._raster_renderer.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(img[..., 3])
        cropped_img = img[slice_y, slice_x]
        if cropped_img.size:
            gc = self._renderer.new_gc()
            # TODO: If the mixedmode resolution differs from the figure's
            #       dpi, the image must be scaled (dpi->_figdpi). Not all
            #       backends support this.
            self._renderer.draw_image(
                gc,
                slice_x.start * self._figdpi / self.dpi,
                (height - slice_y.stop) * self._figdpi / self.dpi,
                cropped_img[::-1])
        self._raster_renderer = None

        # restore the figure dpi.
        self.figure.dpi = self._figdpi

        if self._bbox_inches_restore:  # when tight bbox is used
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore,
                                               self._figdpi)
            self._bbox_inches_restore = r
```
### 5 - lib/matplotlib/backends/backend_agg.py:

Start line: 109, End line: 187

```python
class RendererAgg(RendererBase):

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        nmax = mpl.rcParams['agg.path.chunksize']  # here at least for testing
        npts = path.vertices.shape[0]

        if (npts > nmax > 100 and path.should_simplify and
                rgbFace is None and gc.get_hatch() is None):
            nch = np.ceil(npts / nmax)
            chsize = int(np.ceil(npts / nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for ii0, ii1 in zip(i0, i1):
                v = path.vertices[ii0:ii1, :]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO  # move to end of last chunk
                p = Path(v, c)
                p.simplify_threshold = path.simplify_threshold
                try:
                    self._renderer.draw_path(gc, p, transform, rgbFace)
                except OverflowError:
                    msg = (
                        "Exceeded cell block limit in Agg.\n\n"
                        "Please reduce the value of "
                        f"rcParams['agg.path.chunksize'] (currently {nmax}) "
                        "or increase the path simplification threshold"
                        "(rcParams['path.simplify_threshold'] = "
                        f"{mpl.rcParams['path.simplify_threshold']:.2f} by "
                        "default and path.simplify_threshold = "
                        f"{path.simplify_threshold:.2f} on the input)."
                    )
                    raise OverflowError(msg) from None
        else:
            try:
                self._renderer.draw_path(gc, path, transform, rgbFace)
            except OverflowError:
                cant_chunk = ''
                if rgbFace is not None:
                    cant_chunk += "- can not split filled path\n"
                if gc.get_hatch() is not None:
                    cant_chunk += "- can not split hatched path\n"
                if not path.should_simplify:
                    cant_chunk += "- path.should_simplify is False\n"
                if len(cant_chunk):
                    msg = (
                        "Exceeded cell block limit in Agg, however for the "
                        "following reasons:\n\n"
                        f"{cant_chunk}\n"
                        "we can not automatically split up this path to draw."
                        "\n\nPlease manually simplify your path."
                    )

                else:
                    inc_threshold = (
                        "or increase the path simplification threshold"
                        "(rcParams['path.simplify_threshold'] = "
                        f"{mpl.rcParams['path.simplify_threshold']} "
                        "by default and path.simplify_threshold "
                        f"= {path.simplify_threshold} "
                        "on the input)."
                        )
                    if nmax > 100:
                        msg = (
                            "Exceeded cell block limit in Agg.  Please reduce "
                            "the value of rcParams['agg.path.chunksize'] "
                            f"(currently {nmax}) {inc_threshold}"
                        )
                    else:
                        msg = (
                            "Exceeded cell block limit in Agg.  Please set "
                            "the value of rcParams['agg.path.chunksize'], "
                            f"(currently {nmax}) to be greater than 100 "
                            + inc_threshold
                        )

                raise OverflowError(msg) from None
```
### 6 - lib/matplotlib/backends/backend_mixed.py:

Start line: 71, End line: 85

```python
class MixedModeRenderer:

    def start_rasterizing(self):
        """
        Enter "raster" mode.  All subsequent drawing commands (until
        `stop_rasterizing` is called) will be drawn with the raster backend.
        """
        # change the dpi of the figure temporarily.
        self.figure.dpi = self.dpi
        if self._bbox_inches_restore:  # when tight bbox is used
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore)
            self._bbox_inches_restore = r

        self._raster_renderer = self._raster_renderer_class(
            self._width*self.dpi, self._height*self.dpi, self.dpi)
        self._renderer = self._raster_renderer
```
### 7 - examples/images_contours_and_fields/pcolor_demo.py:

Start line: 1, End line: 97

```python
"""
===========
Pcolor demo
===========

Generating images with `~.axes.Axes.pcolor`.

Pcolor allows you to generate 2D image-style plots. Below we will show how
to do so in Matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# Fixing random state for reproducibility
np.random.seed(19680801)

###############################################################################
# A simple pcolor demo
# --------------------

Z = np.random.rand(6, 10)

fig, (ax0, ax1) = plt.subplots(2, 1)

c = ax0.pcolor(Z)
ax0.set_title('default: no edges')

c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
ax1.set_title('thick edges')

fig.tight_layout()
plt.show()

###############################################################################
# Comparing pcolor with similar functions
# ---------------------------------------
#
# Demonstrates similarities between `~.axes.Axes.pcolor`,
# `~.axes.Axes.pcolormesh`, `~.axes.Axes.imshow` and
# `~.axes.Axes.pcolorfast` for drawing quadrilateral grids.
# Note that we call ``imshow`` with ``aspect="auto"`` so that it doesn't force
# the data pixels to be square (the default is ``aspect="equal"``).

# make these smaller to increase the resolution
dx, dy = 0.15, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-3:3+dy:dy, -3:3+dx:dx]
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -abs(z).max(), abs(z).max()

fig, axs = plt.subplots(2, 2)

ax = axs[0, 0]
c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolor')
fig.colorbar(c, ax=ax)

ax = axs[0, 1]
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
fig.colorbar(c, ax=ax)

ax = axs[1, 0]
c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower', aspect='auto')
ax.set_title('image (nearest, aspect="auto")')
fig.colorbar(c, ax=ax)

ax = axs[1, 1]
c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolorfast')
fig.colorbar(c, ax=ax)

fig.tight_layout()
plt.show()


###############################################################################
# Pcolor with a log scale
# -----------------------
#
# The following shows pcolor plots with a log scale.

N = 100
X, Y = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))

# A low hump with a spike coming out.
# Needs to have z/colour axis on a log scale, so we see both hump and spike.
# A linear scale only shows the spike.
Z1 = np.exp(-X**2 - Y**2)
```
### 8 - lib/matplotlib/backends/backend_ps.py:

Start line: 442, End line: 481

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]
        imagecmd = "false 3 colorimage"
        data = im[::-1, :, :3]  # Vertically flipped rgb values.
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        if transform is None:
            matrix = "1 0 0 1 0 0"
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = " ".join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0

        self._pswriter.write(f"""\

        et_clip_cmd(gc)}
        :g} translate
        }] concat
        g} {yscale:g} scale
        ing {w:d} string def
        :d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]

        ile DataString readhexstring pop
        {imagecmd}
        }



    @_log_if_debug_on
    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)
```
### 9 - examples/misc/zorder_demo.py:

Start line: 1, End line: 76

```python
"""
===========
Zorder Demo
===========

The drawing order of artists is determined by their ``zorder`` attribute, which
is a floating point number. Artists with higher ``zorder`` are drawn on top.
You can change the order for individual artists by setting their ``zorder``.
The default value depends on the type of the Artist:

================================================================    =======
Artist                                                              Z-order
================================================================    =======
Images (`.AxesImage`, `.FigureImage`, `.BboxImage`)                 0
`.Patch`, `.PatchCollection`                                        1
`.Line2D`, `.LineCollection` (including minor ticks, grid lines)    2
Major ticks                                                         2.01
`.Text` (including axes labels and titles)                          3
`.Legend`                                                           5
================================================================    =======

Any call to a plotting method can set a value for the zorder of that particular
item explicitly.

.. note::

   `~.axes.Axes.set_axisbelow` and :rc:`axes.axisbelow` are convenient helpers
   for setting the zorder of ticks and grid lines.

Drawing is done per `~.axes.Axes` at a time. If you have overlapping Axes, all
elements of the second Axes are drawn on top of the first Axes, irrespective of
their relative zorder.
"""

import matplotlib.pyplot as plt
import numpy as np

r = np.linspace(0.3, 1, 30)
theta = np.linspace(0, 4*np.pi, 30)
x = r * np.sin(theta)
y = r * np.cos(theta)

###############################################################################
# The following example contains a `.Line2D` created by `~.axes.Axes.plot()`
# and the dots (a `.PatchCollection`) created by `~.axes.Axes.scatter()`.
# Hence, by default the dots are below the line (first subplot).
# In the second subplot, the ``zorder`` is set explicitly to move the dots
# on top of the line.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))

ax1.plot(x, y, 'C3', lw=3)
ax1.scatter(x, y, s=120)
ax1.set_title('Lines on top of dots')

ax2.plot(x, y, 'C3', lw=3)
ax2.scatter(x, y, s=120, zorder=2.5)  # move dots on top of line
ax2.set_title('Dots on top of lines')

plt.tight_layout()

###############################################################################
# Many functions that create a visible object accepts a ``zorder`` parameter.
# Alternatively, you can call ``set_zorder()`` on the created object later.

x = np.linspace(0, 7.5, 100)
plt.rcParams['lines.linewidth'] = 5
plt.figure()
plt.plot(x, np.sin(x), label='zorder=2', zorder=2)  # bottom
plt.plot(x, np.sin(x+0.5), label='zorder=3',  zorder=3)
plt.axhline(0, label='zorder=2.5', color='lightgrey', zorder=2.5)
plt.title('Custom order of elements')
l = plt.legend(loc='upper right')
l.set_zorder(2.5)  # legend between blue and orange line
plt.show()
```
### 10 - lib/matplotlib/backends/backend_ps.py:

Start line: 712, End line: 756

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4

        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        flat_points = trans.transform(flat_points)
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))

        xmin, ymin = points_min
        xmax, ymax = points_max

        data = np.empty(
            shape[0] * shape[1],
            dtype=[('flags', 'u1'), ('points', '2>u4'), ('colors', '3u1')])
        data['flags'] = 0
        data['points'] = (flat_points - points_min) * factor
        data['colors'] = flat_colors[:, :3] * 255.0
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        self._pswriter.write(f"""\

        ingType 4
        rSpace [/DeviceRGB]
        PerCoordinate 32
        PerComponent 8
        PerFlag 8
        Alias true
        de [ {xmin:g} {xmax:g} {ymin:g} {ymax:g} 0 1 0 1 0 1 ]
        Source <
        }
```
