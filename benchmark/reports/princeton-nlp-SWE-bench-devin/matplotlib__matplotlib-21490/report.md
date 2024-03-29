# matplotlib__matplotlib-21490

| **matplotlib/matplotlib** | `b09aad279b5dcfc49dcf43e0b064eee664ddaf68` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/examples/units/basic_units.py b/examples/units/basic_units.py
--- a/examples/units/basic_units.py
+++ b/examples/units/basic_units.py
@@ -132,6 +132,9 @@ def __init__(self, value, unit):
         self.unit = unit
         self.proxy_target = self.value
 
+    def __copy__(self):
+        return TaggedValue(self.value, self.unit)
+
     def __getattribute__(self, name):
         if name.startswith('__'):
             return object.__getattribute__(self, name)
diff --git a/lib/matplotlib/lines.py b/lib/matplotlib/lines.py
--- a/lib/matplotlib/lines.py
+++ b/lib/matplotlib/lines.py
@@ -2,6 +2,8 @@
 2D lines with support for a variety of line styles, markers, colors, etc.
 """
 
+import copy
+
 from numbers import Integral, Number, Real
 import logging
 
@@ -1230,7 +1232,7 @@ def set_xdata(self, x):
         ----------
         x : 1D array
         """
-        self._xorig = x
+        self._xorig = copy.copy(x)
         self._invalidx = True
         self.stale = True
 
@@ -1242,7 +1244,7 @@ def set_ydata(self, y):
         ----------
         y : 1D array
         """
-        self._yorig = y
+        self._yorig = copy.copy(y)
         self._invalidy = True
         self.stale = True
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| examples/units/basic_units.py | 135 | 135 | - | - | -
| lib/matplotlib/lines.py | 5 | 5 | - | 1 | -
| lib/matplotlib/lines.py | 1233 | 1233 | - | 1 | -
| lib/matplotlib/lines.py | 1245 | 1245 | - | 1 | -


## Problem Statement

```
[Bug]: Line2D should copy its inputs
### Bug summary

Currently, Line2D doesn't copy its inputs if they are already arrays.  Most of the time, in-place modifications to the input arrays do *not* affect the draw line, because there is a cache that doesn't get invalidated, but in some circumstances, it *is* possible for these modifications to affect the drawn line.

Instead, Line2D should just copy its inputs.  This was rejected in #736 on a memory-saving argument, but note that AxesImage (which would typically have much bigger (2D) inputs than Line2D (which has 1D inputs)) does a copy, which if anything is much worse memory-wise.

### Code for reproduction

\`\`\`python
from pylab import *
t = arange(0, 6, 2)
l, = plot(t, t, ".-")
savefig("/tmp/1.png")
t[:] = range(3)  # in place change
savefig("/tmp/2.png")  # no effect
l.set_drawstyle("steps")  # ... unless we trigger a cache invalidation
savefig("/tmp/3.png")  # in fact, only the x array got updated, not the y
\`\`\`


### Actual outcome

(1)
![1](https://user-images.githubusercontent.com/1322974/134257080-5f1afea6-59b0-429b-9ab4-bb4187942139.png)
(2) (same as (1))
![2](https://user-images.githubusercontent.com/1322974/134257087-a2dc2907-819e-4e50-8028-946677fff811.png)
(3) (different, but only x got updated, not y)
![3](https://user-images.githubusercontent.com/1322974/134257088-854fcbd6-407b-434e-b9cb-5583a8be3d77.png)


### Expected outcome

Modifying `t` a posteriori should not affect the Line2D.  Compare e.g. with AxesImage:
\`\`\`python
im = arange(9).reshape(3, 3)
imshow(im)
savefig("/tmp/4.png")
im[:, :] = im[::-1, ::-1]
savefig("/tmp/5.png")
\`\`\`
Both images are identical.

### Operating system

linux

### Matplotlib Version

3.5b1

### Matplotlib Backend

mplcairo

### Python version

39

### Jupyter version

_No response_

### Other libraries

_No response_

### Installation

source

### Conda channel

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/lines.py** | 652 | 693| 513 | 513 | 11984 | 
| 2 | **1 lib/matplotlib/lines.py** | 727 | 860| 1291 | 1804 | 11984 | 
| 3 | **1 lib/matplotlib/lines.py** | 1412 | 1455| 454 | 2258 | 11984 | 
| 4 | **1 lib/matplotlib/lines.py** | 263 | 274| 265 | 2523 | 11984 | 
| 5 | **1 lib/matplotlib/lines.py** | 1270 | 1292| 366 | 2889 | 11984 | 
| 6 | **1 lib/matplotlib/lines.py** | 710 | 725| 236 | 3125 | 11984 | 
| 7 | 2 examples/shapes_and_collections/line_collection.py | 1 | 100| 856 | 3981 | 12840 | 
| 8 | **2 lib/matplotlib/lines.py** | 924 | 1035| 769 | 4750 | 12840 | 
| 9 | **2 lib/matplotlib/lines.py** | 276 | 417| 1198 | 5948 | 12840 | 
| 10 | 3 lib/mpl_toolkits/axisartist/axislines.py | 1 | 49| 443 | 6391 | 17280 | 
| 11 | 4 lib/mpl_toolkits/axisartist/axis_artist.py | 175 | 199| 198 | 6589 | 25338 | 
| 12 | **4 lib/matplotlib/lines.py** | 862 | 904| 317 | 6906 | 25338 | 
| 13 | 5 tutorials/toolkits/axisartist.py | 1 | 564| 4724 | 11630 | 30062 | 
| 14 | 6 lib/matplotlib/axes/_base.py | 486 | 4737| 535 | 12165 | 69773 | 
| 15 | 6 lib/mpl_toolkits/axisartist/axislines.py | 315 | 348| 278 | 12443 | 69773 | 
| 16 | 7 lib/matplotlib/axes/_axes.py | 940 | 1631| 3911 | 16354 | 142578 | 
| 17 | 8 tutorials/introductory/pyplot.py | 88 | 247| 1519 | 17873 | 146981 | 
| 18 | **8 lib/matplotlib/lines.py** | 1394 | 1410| 146 | 18019 | 146981 | 
| 19 | 9 examples/text_labels_and_annotations/line_with_text.py | 51 | 87| 260 | 18279 | 147573 | 
| 20 | **9 lib/matplotlib/lines.py** | 419 | 487| 678 | 18957 | 147573 | 
| 21 | 10 examples/pyplots/axline.py | 1 | 53| 490 | 19447 | 148063 | 
| 22 | 11 examples/images_contours_and_fields/image_antialiasing.py | 70 | 121| 537 | 19984 | 149358 | 
| 23 | **11 lib/matplotlib/lines.py** | 198 | 261| 497 | 20481 | 149358 | 
| 24 | **11 lib/matplotlib/lines.py** | 695 | 708| 254 | 20735 | 149358 | 
| 25 | 12 examples/shapes_and_collections/artist_reference.py | 90 | 130| 319 | 21054 | 150517 | 
| 26 | 12 lib/mpl_toolkits/axisartist/axis_artist.py | 89 | 173| 650 | 21704 | 150517 | 
| 27 | 13 examples/subplots_axes_and_figures/shared_axis_demo.py | 1 | 58| 490 | 22194 | 151007 | 
| 28 | 13 lib/mpl_toolkits/axisartist/axislines.py | 550 | 580| 198 | 22392 | 151007 | 
| 29 | 14 lib/mpl_toolkits/axisartist/clip_path.py | 1 | 80| 682 | 23074 | 152239 | 
| 30 | 15 lib/matplotlib/image.py | 395 | 527| 1709 | 24783 | 168944 | 
| 31 | 15 lib/matplotlib/axes/_axes.py | 5026 | 5834| 5873 | 30656 | 168944 | 
| 32 | 16 examples/misc/demo_agg_filter.py | 198 | 231| 308 | 30964 | 171394 | 
| 33 | 17 examples/misc/rasterization_demo.py | 1 | 73| 753 | 31717 | 172324 | 
| 34 | 18 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 32242 | 172849 | 
| 35 | 19 tutorials/advanced/blitting.py | 1 | 87| 695 | 32937 | 174611 | 
| 36 | 20 examples/misc/tickedstroke_demo.py | 1 | 90| 755 | 33692 | 175507 | 
| 37 | 21 lib/mpl_toolkits/mplot3d/axis3d.py | 419 | 452| 373 | 34065 | 180709 | 
| 38 | 21 lib/matplotlib/axes/_axes.py | 2842 | 3778| 6351 | 40416 | 180709 | 
| 39 | 22 examples/pyplots/fig_x.py | 1 | 28| 139 | 40555 | 180848 | 
| 40 | 23 examples/misc/zorder_demo.py | 1 | 76| 714 | 41269 | 181562 | 
| 41 | 24 lib/matplotlib/legend_handler.py | 211 | 245| 323 | 41592 | 188100 | 
| 42 | 25 tutorials/advanced/transforms_tutorial.py | 233 | 330| 1003 | 42595 | 194201 | 
| 43 | 26 examples/misc/pythonic_matplotlib.py | 1 | 80| 596 | 43191 | 194797 | 
| 44 | 27 examples/widgets/annotated_cursor.py | 286 | 343| 471 | 43662 | 197645 | 
| 45 | 27 tutorials/advanced/blitting.py | 89 | 101| 157 | 43819 | 197645 | 


## Missing Patch Files

 * 1: examples/units/basic_units.py
 * 2: lib/matplotlib/lines.py

### Hint

```
I agree, for most practical purposes, the memory consumption should be negligable.

If one wanted to be on the safe side, one could add a flag, but I tend to think that's not neccesary.
Seems like a well defined what-to-do (with a lot of examples at other places in the code) -- adding it as a good first issue/hacktoberfest-accepted
Hi 🙋‍♂️ I would like to see if I can solve the problem.
Just to make sure that I understood the expected outcome in the example. Should the **y** be ending in 2, right?
```

## Patch

```diff
diff --git a/examples/units/basic_units.py b/examples/units/basic_units.py
--- a/examples/units/basic_units.py
+++ b/examples/units/basic_units.py
@@ -132,6 +132,9 @@ def __init__(self, value, unit):
         self.unit = unit
         self.proxy_target = self.value
 
+    def __copy__(self):
+        return TaggedValue(self.value, self.unit)
+
     def __getattribute__(self, name):
         if name.startswith('__'):
             return object.__getattribute__(self, name)
diff --git a/lib/matplotlib/lines.py b/lib/matplotlib/lines.py
--- a/lib/matplotlib/lines.py
+++ b/lib/matplotlib/lines.py
@@ -2,6 +2,8 @@
 2D lines with support for a variety of line styles, markers, colors, etc.
 """
 
+import copy
+
 from numbers import Integral, Number, Real
 import logging
 
@@ -1230,7 +1232,7 @@ def set_xdata(self, x):
         ----------
         x : 1D array
         """
-        self._xorig = x
+        self._xorig = copy.copy(x)
         self._invalidx = True
         self.stale = True
 
@@ -1242,7 +1244,7 @@ def set_ydata(self, y):
         ----------
         y : 1D array
         """
-        self._yorig = y
+        self._yorig = copy.copy(y)
         self._invalidy = True
         self.stale = True
 

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_lines.py b/lib/matplotlib/tests/test_lines.py
--- a/lib/matplotlib/tests/test_lines.py
+++ b/lib/matplotlib/tests/test_lines.py
@@ -332,3 +332,14 @@ def test_picking():
     found, indices = l2.contains(mouse_event)
     assert found
     assert_array_equal(indices['ind'], [0])
+
+
+@check_figures_equal()
+def test_input_copy(fig_test, fig_ref):
+
+    t = np.arange(0, 6, 2)
+    l, = fig_test.add_subplot().plot(t, t, ".-")
+    t[:] = range(3)
+    # Trigger cache invalidation
+    l.set_drawstyle("steps")
+    fig_ref.add_subplot().plot([0, 2, 4], [0, 2, 4], ".-", drawstyle="steps")
diff --git a/lib/matplotlib/tests/test_units.py b/lib/matplotlib/tests/test_units.py
--- a/lib/matplotlib/tests/test_units.py
+++ b/lib/matplotlib/tests/test_units.py
@@ -26,6 +26,9 @@ def to(self, new_units):
         else:
             return Quantity(self.magnitude, self.units)
 
+    def __copy__(self):
+        return Quantity(self.magnitude, self.units)
+
     def __getattr__(self, attr):
         return getattr(self.magnitude, attr)
 

```


## Code snippets

### 1 - lib/matplotlib/lines.py:

Start line: 652, End line: 693

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def recache(self, always=False):
        if always or self._invalidx:
            xconv = self.convert_xunits(self._xorig)
            x = _to_unmasked_float_array(xconv).ravel()
        else:
            x = self._x
        if always or self._invalidy:
            yconv = self.convert_yunits(self._yorig)
            y = _to_unmasked_float_array(yconv).ravel()
        else:
            y = self._y

        self._xy = np.column_stack(np.broadcast_arrays(x, y)).astype(float)
        self._x, self._y = self._xy.T  # views

        self._subslice = False
        if (self.axes and len(x) > 1000 and self._is_sorted(x) and
                self.axes.name == 'rectilinear' and
                self.axes.get_xscale() == 'linear' and
                self._markevery is None and
                self.get_clip_on() and
                self.get_transform() == self.axes.transData):
            self._subslice = True
            nanmask = np.isnan(x)
            if nanmask.any():
                self._x_filled = self._x.copy()
                indices = np.arange(len(x))
                self._x_filled[nanmask] = np.interp(
                    indices[nanmask], indices[~nanmask], self._x[~nanmask])
            else:
                self._x_filled = self._x

        if self._path is not None:
            interpolation_steps = self._path._interpolation_steps
        else:
            interpolation_steps = 1
        xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy.T)
        self._path = Path(np.asarray(xy).T,
                          _interpolation_steps=interpolation_steps)
        self._transformed_path = None
        self._invalidx = False
        self._invalidy = False
```
### 2 - lib/matplotlib/lines.py:

Start line: 727, End line: 860

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        if not self.get_visible():
            return

        if self._invalidy or self._invalidx:
            self.recache()
        self.ind_offset = 0  # Needed for contains() method.
        if self._subslice and self.axes:
            x0, x1 = self.axes.get_xbound()
            i0 = self._x_filled.searchsorted(x0, 'left')
            i1 = self._x_filled.searchsorted(x1, 'right')
            subslice = slice(max(i0 - 1, 0), i1 + 1)
            self.ind_offset = subslice.start
            self._transform_path(subslice)
        else:
            subslice = None

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        renderer.open_group('line2d', self.get_gid())
        if self._lineStyles[self._linestyle] != '_draw_nothing':
            tpath, affine = (self._get_transformed_path()
                             .get_transformed_path_and_affine())
            if len(tpath.vertices):
                gc = renderer.new_gc()
                self._set_gc_clip(gc)
                gc.set_url(self.get_url())

                lc_rgba = mcolors.to_rgba(self._color, self._alpha)
                gc.set_foreground(lc_rgba, isRGBA=True)

                gc.set_antialiased(self._antialiased)
                gc.set_linewidth(self._linewidth)

                if self.is_dashed():
                    cap = self._dashcapstyle
                    join = self._dashjoinstyle
                else:
                    cap = self._solidcapstyle
                    join = self._solidjoinstyle
                gc.set_joinstyle(join)
                gc.set_capstyle(cap)
                gc.set_snap(self.get_snap())
                if self.get_sketch_params() is not None:
                    gc.set_sketch_params(*self.get_sketch_params())

                gc.set_dashes(self._dashOffset, self._dashSeq)
                renderer.draw_path(gc, tpath, affine.frozen())
                gc.restore()

        if self._marker and self._markersize > 0:
            gc = renderer.new_gc()
            self._set_gc_clip(gc)
            gc.set_url(self.get_url())
            gc.set_linewidth(self._markeredgewidth)
            gc.set_antialiased(self._antialiased)

            ec_rgba = mcolors.to_rgba(
                self.get_markeredgecolor(), self._alpha)
            fc_rgba = mcolors.to_rgba(
                self._get_markerfacecolor(), self._alpha)
            fcalt_rgba = mcolors.to_rgba(
                self._get_markerfacecolor(alt=True), self._alpha)
            # If the edgecolor is "auto", it is set according to the *line*
            # color but inherits the alpha value of the *face* color, if any.
            if (cbook._str_equal(self._markeredgecolor, "auto")
                    and not cbook._str_lower_equal(
                        self.get_markerfacecolor(), "none")):
                ec_rgba = ec_rgba[:3] + (fc_rgba[3],)
            gc.set_foreground(ec_rgba, isRGBA=True)
            if self.get_sketch_params() is not None:
                scale, length, randomness = self.get_sketch_params()
                gc.set_sketch_params(scale/2, length/2, 2*randomness)

            marker = self._marker

            # Markers *must* be drawn ignoring the drawstyle (but don't pay the
            # recaching if drawstyle is already "default").
            if self.get_drawstyle() != "default":
                with cbook._setattr_cm(
                        self, _drawstyle="default", _transformed_path=None):
                    self.recache()
                    self._transform_path(subslice)
                    tpath, affine = (self._get_transformed_path()
                                     .get_transformed_points_and_affine())
            else:
                tpath, affine = (self._get_transformed_path()
                                 .get_transformed_points_and_affine())

            if len(tpath.vertices):
                # subsample the markers if markevery is not None
                markevery = self.get_markevery()
                if markevery is not None:
                    subsampled = _mark_every_path(
                        markevery, tpath, affine, self.axes)
                else:
                    subsampled = tpath

                snap = marker.get_snap_threshold()
                if isinstance(snap, Real):
                    snap = renderer.points_to_pixels(self._markersize) >= snap
                gc.set_snap(snap)
                gc.set_joinstyle(marker.get_joinstyle())
                gc.set_capstyle(marker.get_capstyle())
                marker_path = marker.get_path()
                marker_trans = marker.get_transform()
                w = renderer.points_to_pixels(self._markersize)

                if cbook._str_equal(marker.get_marker(), ","):
                    gc.set_linewidth(0)
                else:
                    # Don't scale for pixels, and don't stroke them
                    marker_trans = marker_trans.scale(w)
                renderer.draw_markers(gc, marker_path, marker_trans,
                                      subsampled, affine.frozen(),
                                      fc_rgba)

                alt_marker_path = marker.get_alt_path()
                if alt_marker_path:
                    alt_marker_trans = marker.get_alt_transform()
                    alt_marker_trans = alt_marker_trans.scale(w)
                    renderer.draw_markers(
                            gc, alt_marker_path, alt_marker_trans, subsampled,
                            affine.frozen(), fcalt_rgba)

            gc.restore()

        renderer.close_group('line2d')
        self.stale = False
```
### 3 - lib/matplotlib/lines.py:

Start line: 1412, End line: 1455

```python
class _AxLine(Line2D):

    def get_transform(self):
        ax = self.axes
        points_transform = self._transform - ax.transData + ax.transScale

        if self._xy2 is not None:
            # two points were given
            (x1, y1), (x2, y2) = \
                points_transform.transform([self._xy1, self._xy2])
            dx = x2 - x1
            dy = y2 - y1
            if np.allclose(x1, x2):
                if np.allclose(y1, y2):
                    raise ValueError(
                        f"Cannot draw a line through two identical points "
                        f"(x={(x1, x2)}, y={(y1, y2)})")
                slope = np.inf
            else:
                slope = dy / dx
        else:
            # one point and a slope were given
            x1, y1 = points_transform.transform(self._xy1)
            slope = self._slope
        (vxlo, vylo), (vxhi, vyhi) = ax.transScale.transform(ax.viewLim)
        # General case: find intersections with view limits in either
        # direction, and draw between the middle two points.
        if np.isclose(slope, 0):
            start = vxlo, y1
            stop = vxhi, y1
        elif np.isinf(slope):
            start = x1, vylo
            stop = x1, vyhi
        else:
            _, start, stop, _ = sorted([
                (vxlo, y1 + (vxlo - x1) * slope),
                (vxhi, y1 + (vxhi - x1) * slope),
                (x1 + (vylo - y1) / slope, vylo),
                (x1 + (vyhi - y1) / slope, vyhi),
            ])
        return (BboxTransformTo(Bbox([start, stop]))
                + ax.transLimits + ax.transAxes)

    def draw(self, renderer):
        self._transformed_path = None  # Force regen.
        super().draw(renderer)
```
### 4 - lib/matplotlib/lines.py:

Start line: 263, End line: 274

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def __str__(self):
        if self._label != "":
            return f"Line2D({self._label})"
        elif self._x is None:
            return "Line2D()"
        elif len(self._x) > 3:
            return "Line2D((%g,%g),(%g,%g),...,(%g,%g))" % (
                self._x[0], self._y[0], self._x[0],
                self._y[0], self._x[-1], self._y[-1])
        else:
            return "Line2D(%s)" % ",".join(
                map("({:g},{:g})".format, self._x, self._y))
```
### 5 - lib/matplotlib/lines.py:

Start line: 1270, End line: 1292

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def update_from(self, other):
        """Copy properties from *other* to self."""
        super().update_from(other)
        self._linestyle = other._linestyle
        self._linewidth = other._linewidth
        self._color = other._color
        self._markersize = other._markersize
        self._markerfacecolor = other._markerfacecolor
        self._markerfacecoloralt = other._markerfacecoloralt
        self._markeredgecolor = other._markeredgecolor
        self._markeredgewidth = other._markeredgewidth
        self._dashSeq = other._dashSeq
        self._us_dashSeq = other._us_dashSeq
        self._dashOffset = other._dashOffset
        self._us_dashOffset = other._us_dashOffset
        self._dashcapstyle = other._dashcapstyle
        self._dashjoinstyle = other._dashjoinstyle
        self._solidcapstyle = other._solidcapstyle
        self._solidjoinstyle = other._solidjoinstyle

        self._linestyle = other._linestyle
        self._marker = MarkerStyle(marker=other._marker)
        self._drawstyle = other._drawstyle
```
### 6 - lib/matplotlib/lines.py:

Start line: 710, End line: 725

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def _get_transformed_path(self):
        """Return this line's `~matplotlib.transforms.TransformedPath`."""
        if self._transformed_path is None:
            self._transform_path()
        return self._transformed_path

    def set_transform(self, t):
        # docstring inherited
        self._invalidx = True
        self._invalidy = True
        super().set_transform(t)

    def _is_sorted(self, x):
        """Return whether x is sorted in ascending order."""
        # We don't handle the monotonically decreasing case.
        return _path.is_sorted(x)
```
### 7 - examples/shapes_and_collections/line_collection.py:

Start line: 1, End line: 100

```python
"""
===============
Line Collection
===============

Plotting lines with Matplotlib.

`~matplotlib.collections.LineCollection` allows one to plot multiple
lines on a figure. Below we show off some of its properties.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

import numpy as np

# In order to efficiently plot many lines in a single set of axes,
# Matplotlib has the ability to add the lines all at once. Here is a
# simple example showing how it is done.

x = np.arange(100)
# Here are many sets of y to plot vs. x
ys = x[:50, np.newaxis] + x[np.newaxis, :]

segs = np.zeros((50, 100, 2))
segs[:, :, 1] = ys
segs[:, :, 0] = x

# Mask some values to test masked array support:
segs = np.ma.masked_where((segs > 50) & (segs < 60), segs)

# We need to set the plot limits.
fig, ax = plt.subplots()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(ys.min(), ys.max())

# *colors* is sequence of rgba tuples.
# *linestyle* is a string or dash tuple. Legal string values are
# solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
# onoffseq is an even length tuple of on and off ink in points.  If linestyle
# is omitted, 'solid' is used.
# See `matplotlib.collections.LineCollection` for more information.
colors = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

line_segments = LineCollection(segs, linewidths=(0.5, 1, 1.5, 2),
                               colors=colors, linestyle='solid')
ax.add_collection(line_segments)
ax.set_title('Line collection with masked arrays')
plt.show()

###############################################################################
# In order to efficiently plot many lines in a single set of axes,
# Matplotlib has the ability to add the lines all at once. Here is a
# simple example showing how it is done.

N = 50
x = np.arange(N)
# Here are many sets of y to plot vs. x
ys = [x + i for i in x]

# We need to set the plot limits, they will not autoscale
fig, ax = plt.subplots()
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(ys), np.max(ys))

# colors is sequence of rgba tuples
# linestyle is a string or dash tuple. Legal string values are
#          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
#          where onoffseq is an even length tuple of on and off ink in points.
#          If linestyle is omitted, 'solid' is used
# See `matplotlib.collections.LineCollection` for more information

# Make a sequence of (x, y) pairs.
line_segments = LineCollection([np.column_stack([x, y]) for y in ys],
                               linewidths=(0.5, 1, 1.5, 2),
                               linestyles='solid')
line_segments.set_array(x)
ax.add_collection(line_segments)
axcb = fig.colorbar(line_segments)
axcb.set_label('Line Number')
ax.set_title('Line Collection with mapped colors')
plt.sci(line_segments)  # This allows interactive changing of the colormap.
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.collections`
#    - `matplotlib.collections.LineCollection`
#    - `matplotlib.cm.ScalarMappable.set_array`
#    - `matplotlib.axes.Axes.add_collection`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.pyplot.sci`
```
### 8 - lib/matplotlib/lines.py:

Start line: 924, End line: 1035

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def get_markeredgewidth(self):
        """
        Return the marker edge width in points.

        See also `~.Line2D.set_markeredgewidth`.
        """
        return self._markeredgewidth

    def _get_markerfacecolor(self, alt=False):
        if self._marker.get_fillstyle() == 'none':
            return 'none'
        fc = self._markerfacecoloralt if alt else self._markerfacecolor
        if cbook._str_lower_equal(fc, 'auto'):
            return self._color
        else:
            return fc

    def get_markerfacecolor(self):
        """
        Return the marker face color.

        See also `~.Line2D.set_markerfacecolor`.
        """
        return self._get_markerfacecolor(alt=False)

    def get_markerfacecoloralt(self):
        """
        Return the alternate marker face color.

        See also `~.Line2D.set_markerfacecoloralt`.
        """
        return self._get_markerfacecolor(alt=True)

    def get_markersize(self):
        """
        Return the marker size in points.

        See also `~.Line2D.set_markersize`.
        """
        return self._markersize

    def get_data(self, orig=True):
        """
        Return the line data as an ``(xdata, ydata)`` pair.

        If *orig* is *True*, return the original data.
        """
        return self.get_xdata(orig=orig), self.get_ydata(orig=orig)

    def get_xdata(self, orig=True):
        """
        Return the xdata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        if orig:
            return self._xorig
        if self._invalidx:
            self.recache()
        return self._x

    def get_ydata(self, orig=True):
        """
        Return the ydata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        if orig:
            return self._yorig
        if self._invalidy:
            self.recache()
        return self._y

    def get_path(self):
        """Return the `~matplotlib.path.Path` associated with this line."""
        if self._invalidy or self._invalidx:
            self.recache()
        return self._path

    def get_xydata(self):
        """
        Return the *xy* data as a Nx2 numpy array.
        """
        if self._invalidy or self._invalidx:
            self.recache()
        return self._xy

    def set_antialiased(self, b):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        b : bool
        """
        if self._antialiased != b:
            self.stale = True
        self._antialiased = b

    def set_color(self, color):
        """
        Set the color of the line.

        Parameters
        ----------
        color : color
        """
        mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True
```
### 9 - lib/matplotlib/lines.py:

Start line: 276, End line: 417

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def __init__(self, xdata, ydata,
                 linewidth=None,  # all Nones default to rc
                 linestyle=None,
                 color=None,
                 marker=None,
                 markersize=None,
                 markeredgewidth=None,
                 markeredgecolor=None,
                 markerfacecolor=None,
                 markerfacecoloralt='none',
                 fillstyle=None,
                 antialiased=None,
                 dash_capstyle=None,
                 solid_capstyle=None,
                 dash_joinstyle=None,
                 solid_joinstyle=None,
                 pickradius=5,
                 drawstyle=None,
                 markevery=None,
                 **kwargs
                 ):
        """
        Create a `.Line2D` instance with *x* and *y* data in sequences of
        *xdata*, *ydata*.

        Additional keyword arguments are `.Line2D` properties:

        %(Line2D:kwdoc)s

        See :meth:`set_linestyle` for a description of the line styles,
        :meth:`set_marker` for a description of the markers, and
        :meth:`set_drawstyle` for a description of the draw styles.

        """
        super().__init__()

        # Convert sequences to NumPy arrays.
        if not np.iterable(xdata):
            raise RuntimeError('xdata must be a sequence')
        if not np.iterable(ydata):
            raise RuntimeError('ydata must be a sequence')

        if linewidth is None:
            linewidth = rcParams['lines.linewidth']

        if linestyle is None:
            linestyle = rcParams['lines.linestyle']
        if marker is None:
            marker = rcParams['lines.marker']
        if color is None:
            color = rcParams['lines.color']

        if markersize is None:
            markersize = rcParams['lines.markersize']
        if antialiased is None:
            antialiased = rcParams['lines.antialiased']
        if dash_capstyle is None:
            dash_capstyle = rcParams['lines.dash_capstyle']
        if dash_joinstyle is None:
            dash_joinstyle = rcParams['lines.dash_joinstyle']
        if solid_capstyle is None:
            solid_capstyle = rcParams['lines.solid_capstyle']
        if solid_joinstyle is None:
            solid_joinstyle = rcParams['lines.solid_joinstyle']

        if drawstyle is None:
            drawstyle = 'default'

        self._dashcapstyle = None
        self._dashjoinstyle = None
        self._solidjoinstyle = None
        self._solidcapstyle = None
        self.set_dash_capstyle(dash_capstyle)
        self.set_dash_joinstyle(dash_joinstyle)
        self.set_solid_capstyle(solid_capstyle)
        self.set_solid_joinstyle(solid_joinstyle)

        self._linestyles = None
        self._drawstyle = None
        self._linewidth = linewidth

        # scaled dash + offset
        self._dashSeq = None
        self._dashOffset = 0
        # unscaled dash + offset
        # this is needed scaling the dash pattern by linewidth
        self._us_dashSeq = None
        self._us_dashOffset = 0

        self.set_linewidth(linewidth)
        self.set_linestyle(linestyle)
        self.set_drawstyle(drawstyle)

        self._color = None
        self.set_color(color)
        if marker is None:
            marker = 'none'  # Default.
        if not isinstance(marker, MarkerStyle):
            self._marker = MarkerStyle(marker, fillstyle)
        else:
            self._marker = marker

        self._markevery = None
        self._markersize = None
        self._antialiased = None

        self.set_markevery(markevery)
        self.set_antialiased(antialiased)
        self.set_markersize(markersize)

        self._markeredgecolor = None
        self._markeredgewidth = None
        self._markerfacecolor = None
        self._markerfacecoloralt = None

        self.set_markerfacecolor(markerfacecolor)  # Normalizes None to rc.
        self.set_markerfacecoloralt(markerfacecoloralt)
        self.set_markeredgecolor(markeredgecolor)  # Normalizes None to rc.
        self.set_markeredgewidth(markeredgewidth)

        # update kwargs before updating data to give the caller a
        # chance to init axes (and hence unit support)
        self.update(kwargs)
        self.pickradius = pickradius
        self.ind_offset = 0
        if (isinstance(self._picker, Number) and
                not isinstance(self._picker, bool)):
            self.pickradius = self._picker

        self._xorig = np.asarray([])
        self._yorig = np.asarray([])
        self._invalidx = True
        self._invalidy = True
        self._x = None
        self._y = None
        self._xy = None
        self._path = None
        self._transformed_path = None
        self._subslice = False
        self._x_filled = None  # used in subslicing; only x is needed

        self.set_data(xdata, ydata)
```
### 10 - lib/mpl_toolkits/axisartist/axislines.py:

Start line: 1, End line: 49

```python
"""
Axislines includes modified implementation of the Axes class. The
biggest difference is that the artists responsible for drawing the axis spine,
ticks, ticklabels and axis labels are separated out from Matplotlib's Axis
class. Originally, this change was motivated to support curvilinear
grid. Here are a few reasons that I came up with a new axes class:

* "top" and "bottom" x-axis (or "left" and "right" y-axis) can have
  different ticks (tick locations and labels). This is not possible
  with the current Matplotlib, although some twin axes trick can help.

* Curvilinear grid.

* angled ticks.

In the new axes class, xaxis and yaxis is set to not visible by
default, and new set of artist (AxisArtist) are defined to draw axis
line, ticks, ticklabels and axis label. Axes.axis attribute serves as
a dictionary of these artists, i.e., ax.axis["left"] is a AxisArtist
instance responsible to draw left y-axis. The default Axes.axis contains
"bottom", "left", "top" and "right".

AxisArtist can be considered as a container artist and
has following children artists which will draw ticks, labels, etc.

* line
* major_ticks, major_ticklabels
* minor_ticks, minor_ticklabels
* offsetText
* label

Note that these are separate artists from `matplotlib.axis.Axis`, thus most
tick-related functions in Matplotlib won't work. For example, color and
markerwidth of the ``ax.axis["bottom"].major_ticks`` will follow those of
Axes.xaxis unless explicitly specified.

In addition to AxisArtist, the Axes will have *gridlines* attribute,
which obviously draws grid lines. The gridlines needs to be separated
from the axis as some gridlines can never pass any axis.
"""

import numpy as np

from matplotlib import _api, rcParams
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle
from .axis_artist import AxisArtist, GridlinesCollection
```
### 12 - lib/matplotlib/lines.py:

Start line: 862, End line: 904

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def get_antialiased(self):
        """Return whether antialiased rendering is used."""
        return self._antialiased

    def get_color(self):
        """
        Return the line color.

        See also `~.Line2D.set_color`.
        """
        return self._color

    def get_drawstyle(self):
        """
        Return the drawstyle.

        See also `~.Line2D.set_drawstyle`.
        """
        return self._drawstyle

    def get_linestyle(self):
        """
        Return the linestyle.

        See also `~.Line2D.set_linestyle`.
        """
        return self._linestyle

    def get_linewidth(self):
        """
        Return the linewidth in points.

        See also `~.Line2D.set_linewidth`.
        """
        return self._linewidth

    def get_marker(self):
        """
        Return the line marker.

        See also `~.Line2D.set_marker`.
        """
        return self._marker.get_marker()
```
### 18 - lib/matplotlib/lines.py:

Start line: 1394, End line: 1410

```python
class _AxLine(Line2D):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """

    def __init__(self, xy1, xy2, slope, **kwargs):
        super().__init__([0, 1], [0, 1], **kwargs)

        if (xy2 is None and slope is None or
                xy2 is not None and slope is not None):
            raise TypeError(
                "Exactly one of 'xy2' and 'slope' must be given")

        self._slope = slope
        self._xy1 = xy1
        self._xy2 = xy2
```
### 20 - lib/matplotlib/lines.py:

Start line: 419, End line: 487

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def contains(self, mouseevent):
        """
        Test whether *mouseevent* occurred on the line.

        An event is deemed to have occurred "on" the line if it is less
        than ``self.pickradius`` (default: 5 points) away from it.  Use
        `~.Line2D.get_pickradius` or `~.Line2D.set_pickradius` to get or set
        the pick radius.

        Parameters
        ----------
        mouseevent : `matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            A dictionary ``{'ind': pointlist}``, where *pointlist* is a
            list of points of the line that are within the pickradius around
            the event position.

            TODO: sort returned indices by distance
        """
        inside, info = self._default_contains(mouseevent)
        if inside is not None:
            return inside, info

        # Make sure we have data to plot
        if self._invalidy or self._invalidx:
            self.recache()
        if len(self._xy) == 0:
            return False, {}

        # Convert points to pixels
        transformed_path = self._get_transformed_path()
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # Convert pick radius from points to pixels
        if self.figure is None:
            _log.warning('no figure set when check if mouse is on line')
            pixels = self.pickradius
        else:
            pixels = self.figure.dpi / 72. * self.pickradius

        # The math involved in checking for containment (here and inside of
        # segment_hits) assumes that it is OK to overflow, so temporarily set
        # the error flags accordingly.
        with np.errstate(all='ignore'):
            # Check for collision
            if self._linestyle in ['None', None]:
                # If no line, return the nearby point(s)
                ind, = np.nonzero(
                    (xt - mouseevent.x) ** 2 + (yt - mouseevent.y) ** 2
                    <= pixels ** 2)
            else:
                # If line, return the nearby segment(s)
                ind = segment_hits(mouseevent.x, mouseevent.y, xt, yt, pixels)
                if self._drawstyle.startswith("steps"):
                    ind //= 2

        ind += self.ind_offset

        # Return the point(s) within radius
        return len(ind) > 0, dict(ind=ind)
```
### 23 - lib/matplotlib/lines.py:

Start line: 198, End line: 261

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, e.g., one
    can create "stepped" lines in various styles.
    """

    lineStyles = _lineStyles = {  # hidden names deprecated
        '-':    '_draw_solid',
        '--':   '_draw_dashed',
        '-.':   '_draw_dash_dot',
        ':':    '_draw_dotted',
        'None': '_draw_nothing',
        ' ':    '_draw_nothing',
        '':     '_draw_nothing',
    }

    _drawStyles_l = {
        'default':    '_draw_lines',
        'steps-mid':  '_draw_steps_mid',
        'steps-pre':  '_draw_steps_pre',
        'steps-post': '_draw_steps_post',
    }

    _drawStyles_s = {
        'steps': '_draw_steps_pre',
    }

    # drawStyles should now be deprecated.
    drawStyles = {**_drawStyles_l, **_drawStyles_s}
    # Need a list ordered with long names first:
    drawStyleKeys = [*_drawStyles_l, *_drawStyles_s]

    # Referenced here to maintain API.  These are defined in
    # MarkerStyle
    markers = MarkerStyle.markers
    filled_markers = MarkerStyle.filled_markers
    fillStyles = MarkerStyle.fillstyles

    zorder = 2

    @_api.deprecated("3.4")
    @_api.classproperty
    def validCap(cls):
        return tuple(cs.value for cs in CapStyle)

    @_api.deprecated("3.4")
    @_api.classproperty
    def validJoin(cls):
        return tuple(js.value for js in JoinStyle)
```
### 24 - lib/matplotlib/lines.py:

Start line: 695, End line: 708

```python
@docstring.interpd
@cbook._define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):

    def _transform_path(self, subslice=None):
        """
        Put a TransformedPath instance at self._transformed_path;
        all invalidation of the transform is then handled by the
        TransformedPath instance.
        """
        # Masked arrays are now handled by the Path class itself
        if subslice is not None:
            xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy[subslice, :].T)
            _path = Path(np.asarray(xy).T,
                         _interpolation_steps=self._path._interpolation_steps)
        else:
            _path = self._path
        self._transformed_path = TransformedPath(_path, self.get_transform())
```
