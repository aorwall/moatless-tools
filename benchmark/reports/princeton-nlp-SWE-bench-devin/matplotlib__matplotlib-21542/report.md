# matplotlib__matplotlib-21542

| **matplotlib/matplotlib** | `f0632c0fc7339f68e992ed63ae4cfac76cd41aad` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 25716 |
| **Any found context length** | 5976 |
| **Avg pos** | 50.0 |
| **Min pos** | 9 |
| **Max pos** | 41 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/colorbar.py b/lib/matplotlib/colorbar.py
--- a/lib/matplotlib/colorbar.py
+++ b/lib/matplotlib/colorbar.py
@@ -352,6 +352,8 @@ class Colorbar:
     ticks : `~matplotlib.ticker.Locator` or array-like of float
 
     format : str or `~matplotlib.ticker.Formatter`
+        If string, it supports '%' operator and `str.format` formats:
+        e.g. ``"%4.2e"`` or ``"{x:.2e}"``.
 
     drawedges : bool
 
@@ -487,7 +489,12 @@ def __init__(self, ax, mappable=None, *, cmap=None,
             self.locator = ticks    # Handle default in _ticker()
 
         if isinstance(format, str):
-            self.formatter = ticker.FormatStrFormatter(format)
+            # Check format between FormatStrFormatter and StrMethodFormatter
+            try:
+                self.formatter = ticker.FormatStrFormatter(format)
+                _ = self.formatter(0)
+            except TypeError:
+                self.formatter = ticker.StrMethodFormatter(format)
         else:
             self.formatter = format  # Assume it is a Formatter or None
         self.draw_all()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/colorbar.py | 355 | 355 | 41 | 1 | 25716
| lib/matplotlib/colorbar.py | 490 | 490 | 9 | 1 | 5976


## Problem Statement

```
[ENH]: use new style format strings for colorbar ticks
### Problem

At the moment, the default format strings in colorbar are old style ones, as in their init there is:

https://github.com/matplotlib/matplotlib/blob/67e18148d87db04d2c8d4293ff12c56fbbb7fde8/lib/matplotlib/colorbar.py#L489-L492

which is a different convention from the one of a normal axis, which was introduced in #16715. 

### Proposed solution

As in `update_ticks` we pass the colorbar's formatter to the long axis,

https://github.com/matplotlib/matplotlib/blob/67e18148d87db04d2c8d4293ff12c56fbbb7fde8/lib/matplotlib/colorbar.py#L801

the `if` statement above may be removed to keep the default logic only in `Axis`. Right now one can pass a callable directly (although that's not documented), and the default behaviour from #16715 is triggered. However, I guess making this change for format strings would imply a deprecation cycle, as it breaks current behaviour. Another option would be to check in `Axis._set_formatter`  for what kind of string format we've been passed (unsure how, although there must be way).

### Additional context and prior art

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/colorbar.py** | 197 | 210| 183 | 183 | 14249 | 
| 2 | **1 lib/matplotlib/colorbar.py** | 793 | 846| 460 | 643 | 14249 | 
| 3 | 2 examples/ticks/date_concise_formatter.py | 1 | 109| 1031 | 1674 | 15891 | 
| 4 | 3 examples/ticks/tick-formatters.py | 37 | 135| 963 | 2637 | 17126 | 
| 5 | 4 lib/matplotlib/axis.py | 1583 | 1621| 300 | 2937 | 37342 | 
| 6 | 5 lib/matplotlib/axes/_base.py | 122 | 209| 648 | 3585 | 77012 | 
| 7 | 5 lib/matplotlib/axis.py | 1557 | 1581| 228 | 3813 | 77012 | 
| 8 | 6 tutorials/intermediate/artists.py | 616 | 727| 1004 | 4817 | 84716 | 
| **-> 9 <-** | **6 lib/matplotlib/colorbar.py** | 369 | 511| 1159 | 5976 | 84716 | 
| 10 | 6 examples/ticks/tick-formatters.py | 1 | 34| 272 | 6248 | 84716 | 
| 11 | 7 tutorials/colors/colorbar_only.py | 1 | 87| 748 | 6996 | 85851 | 
| 12 | **7 lib/matplotlib/colorbar.py** | 63 | 124| 616 | 7612 | 85851 | 
| 13 | **7 lib/matplotlib/colorbar.py** | 1510 | 1568| 627 | 8239 | 85851 | 
| 14 | **7 lib/matplotlib/colorbar.py** | 889 | 924| 301 | 8540 | 85851 | 
| 15 | **7 lib/matplotlib/colorbar.py** | 1439 | 1456| 180 | 8720 | 85851 | 
| 16 | 8 examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 303 | 9023 | 86154 | 
| 17 | 8 lib/matplotlib/axis.py | 889 | 932| 492 | 9515 | 86154 | 
| 18 | 8 tutorials/colors/colorbar_only.py | 89 | 134| 387 | 9902 | 86154 | 
| 19 | 9 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 86| 750 | 10652 | 87040 | 
| 20 | 10 examples/ticks/scalarformatter.py | 1 | 85| 728 | 11380 | 87768 | 
| 21 | 11 tutorials/toolkits/axisartist.py | 1 | 564| 4724 | 16104 | 92492 | 
| 22 | **11 lib/matplotlib/colorbar.py** | 640 | 709| 712 | 16816 | 92492 | 
| 23 | **11 lib/matplotlib/colorbar.py** | 1356 | 1438| 792 | 17608 | 92492 | 
| 24 | **11 lib/matplotlib/colorbar.py** | 126 | 194| 642 | 18250 | 92492 | 
| 25 | **11 lib/matplotlib/colorbar.py** | 1332 | 1353| 322 | 18572 | 92492 | 
| 26 | 11 lib/matplotlib/axis.py | 1670 | 1747| 708 | 19280 | 92492 | 
| 27 | **11 lib/matplotlib/colorbar.py** | 1 | 61| 626 | 19906 | 92492 | 
| 28 | 12 tutorials/text/text_intro.py | 330 | 423| 764 | 20670 | 96370 | 
| 29 | 12 examples/ticks/date_concise_formatter.py | 111 | 182| 611 | 21281 | 96370 | 
| 30 | 13 tutorials/introductory/lifecycle.py | 105 | 198| 827 | 22108 | 98671 | 
| 31 | 14 examples/pyplots/pyplot_formatstr.py | 1 | 22| 128 | 22236 | 98799 | 
| 32 | 14 lib/matplotlib/axis.py | 1448 | 1480| 295 | 22531 | 98799 | 
| 33 | **14 lib/matplotlib/colorbar.py** | 926 | 964| 309 | 22840 | 98799 | 
| 34 | 15 examples/text_labels_and_annotations/date_index_formatter.py | 1 | 57| 447 | 23287 | 99246 | 
| 35 | 16 lib/mpl_toolkits/axes_grid1/axes_grid.py | 20 | 51| 229 | 23516 | 104135 | 
| 36 | 17 lib/mpl_toolkits/axisartist/angle_helper.py | 290 | 307| 230 | 23746 | 107832 | 
| 37 | 18 lib/matplotlib/pyplot.py | 2079 | 2092| 138 | 23884 | 134277 | 
| 38 | 18 lib/matplotlib/axis.py | 349 | 404| 600 | 24484 | 134277 | 
| 39 | 18 lib/matplotlib/axis.py | 850 | 887| 366 | 24850 | 134277 | 
| 40 | 19 lib/mpl_toolkits/axisartist/axis_artist.py | 883 | 908| 278 | 25128 | 142335 | 
| **-> 41 <-** | **19 lib/matplotlib/colorbar.py** | 289 | 367| 588 | 25716 | 142335 | 
| 42 | 20 examples/pyplots/dollar_ticks.py | 1 | 38| 207 | 25923 | 142542 | 
| 43 | 20 lib/matplotlib/axis.py | 1873 | 1907| 292 | 26215 | 142542 | 
| 44 | 21 lib/matplotlib/figure.py | 1127 | 1164| 384 | 26599 | 168809 | 
| 45 | 21 lib/matplotlib/axis.py | 185 | 200| 134 | 26733 | 168809 | 
| 46 | 22 examples/text_labels_and_annotations/engineering_formatter.py | 1 | 45| 367 | 27100 | 169176 | 
| 47 | 23 examples/color/colorbar_basics.py | 1 | 59| 513 | 27613 | 169689 | 
| 48 | 23 lib/mpl_toolkits/axisartist/angle_helper.py | 220 | 287| 633 | 28246 | 169689 | 
| 49 | 23 lib/matplotlib/axis.py | 1134 | 1169| 279 | 28525 | 169689 | 
| 50 | 24 lib/mpl_toolkits/axisartist/floating_axes.py | 48 | 115| 730 | 29255 | 172922 | 
| 51 | 24 lib/matplotlib/axis.py | 1034 | 1082| 459 | 29714 | 172922 | 
| 52 | 25 lib/mpl_toolkits/mplot3d/axis3d.py | 136 | 178| 353 | 30067 | 178124 | 
| 53 | 26 examples/misc/custom_projection.py | 259 | 282| 200 | 30267 | 181963 | 
| 54 | 27 tools/boilerplate.py | 82 | 108| 195 | 30462 | 184826 | 
| 55 | 28 tutorials/intermediate/constrainedlayout_guide.py | 122 | 194| 777 | 31239 | 190929 | 


## Patch

```diff
diff --git a/lib/matplotlib/colorbar.py b/lib/matplotlib/colorbar.py
--- a/lib/matplotlib/colorbar.py
+++ b/lib/matplotlib/colorbar.py
@@ -352,6 +352,8 @@ class Colorbar:
     ticks : `~matplotlib.ticker.Locator` or array-like of float
 
     format : str or `~matplotlib.ticker.Formatter`
+        If string, it supports '%' operator and `str.format` formats:
+        e.g. ``"%4.2e"`` or ``"{x:.2e}"``.
 
     drawedges : bool
 
@@ -487,7 +489,12 @@ def __init__(self, ax, mappable=None, *, cmap=None,
             self.locator = ticks    # Handle default in _ticker()
 
         if isinstance(format, str):
-            self.formatter = ticker.FormatStrFormatter(format)
+            # Check format between FormatStrFormatter and StrMethodFormatter
+            try:
+                self.formatter = ticker.FormatStrFormatter(format)
+                _ = self.formatter(0)
+            except TypeError:
+                self.formatter = ticker.StrMethodFormatter(format)
         else:
             self.formatter = format  # Assume it is a Formatter or None
         self.draw_all()

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_colorbar.py b/lib/matplotlib/tests/test_colorbar.py
--- a/lib/matplotlib/tests/test_colorbar.py
+++ b/lib/matplotlib/tests/test_colorbar.py
@@ -543,14 +543,15 @@ def test_colorbar_renorm():
     assert np.isclose(cbar.vmax, z.max() * 1000)
 
 
-def test_colorbar_format():
+@pytest.mark.parametrize('fmt', ['%4.2e', '{x:.2e}'])
+def test_colorbar_format(fmt):
     # make sure that format is passed properly
     x, y = np.ogrid[-4:4:31j, -4:4:31j]
     z = 120000*np.exp(-x**2 - y**2)
 
     fig, ax = plt.subplots()
     im = ax.imshow(z)
-    cbar = fig.colorbar(im, format='%4.2e')
+    cbar = fig.colorbar(im, format=fmt)
     fig.canvas.draw()
     assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '8.00e+04'
 

```


## Code snippets

### 1 - lib/matplotlib/colorbar.py:

Start line: 197, End line: 210

```python
@_api.caching_module_getattr  # module-level deprecations
class __getattr__:
    colorbar_doc = _api.deprecated("3.4", obj_type="")(property(
        lambda self: docstring.interpd.params["colorbar_doc"]))
    colorbar_kw_doc = _api.deprecated("3.4", obj_type="")(property(
        lambda self: _colormap_kw_doc))
    make_axes_kw_doc = _api.deprecated("3.4", obj_type="")(property(
        lambda self: _make_axes_param_doc + _make_axes_other_param_doc))


def _set_ticks_on_axis_warn(*args, **kwargs):
    # a top level function which gets put in at the axes'
    # set_xticks and set_yticks by Colorbar.__init__.
    _api.warn_external("Use the colorbar set_ticks() method instead.")
```
### 2 - lib/matplotlib/colorbar.py:

Start line: 793, End line: 846

```python
class Colorbar:

    def update_ticks(self):
        """
        Setup the ticks and ticklabels. This should not be needed by users.
        """
        # Get the locator and formatter; defaults to self.locator if not None.
        self._get_ticker_locator_formatter()
        self._long_axis().set_major_locator(self.locator)
        self._long_axis().set_minor_locator(self.minorlocator)
        self._long_axis().set_major_formatter(self.formatter)

    def _get_ticker_locator_formatter(self):
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
        locator = self.locator
        formatter = self.formatter
        minorlocator = self.minorlocator
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        elif self.boundaries is not None:
            b = self._boundaries[self._inside]
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        else:  # most cases:
            if locator is None:
                # we haven't set the locator explicitly, so use the default
                # for this axis:
                locator = self._long_axis().get_major_locator()
            if minorlocator is None:
                minorlocator = self._long_axis().get_minor_locator()
            if isinstance(self.norm, colors.NoNorm):
                # default locator:
                nv = len(self._values)
                base = 1 + int(nv / 10)
                locator = ticker.IndexLocator(base=base, offset=0)

        if minorlocator is None:
            minorlocator = ticker.NullLocator()

        if formatter is None:
            formatter = self._long_axis().get_major_formatter()

        self.locator = locator
        self.formatter = formatter
        self.minorlocator = minorlocator
        _log.debug('locator: %r', locator)
```
### 3 - examples/ticks/date_concise_formatter.py:

Start line: 1, End line: 109

```python
"""
================================================
Formatting date ticks using ConciseDateFormatter
================================================

Finding good tick values and formatting the ticks for an axis that
has date data is often a challenge.  `~.dates.ConciseDateFormatter` is
meant to improve the strings chosen for the ticklabels, and to minimize
the strings used in those tick labels as much as possible.

.. note::

    This formatter is a candidate to become the default date tick formatter
    in future versions of Matplotlib.  Please report any issues or
    suggestions for improvement to the github repository or mailing list.

"""
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#############################################################################
# First, the default formatter.

base = datetime.datetime(2005, 2, 1)
dates = [base + datetime.timedelta(hours=(2 * i)) for i in range(732)]
N = len(dates)
np.random.seed(19680801)
y = np.cumsum(np.random.randn(N))

fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(6, 6))
lims = [(np.datetime64('2005-02'), np.datetime64('2005-04')),
        (np.datetime64('2005-02-03'), np.datetime64('2005-02-15')),
        (np.datetime64('2005-02-03 11:00'), np.datetime64('2005-02-04 13:20'))]
for nn, ax in enumerate(axs):
    ax.plot(dates, y)
    ax.set_xlim(lims[nn])
    # rotate_labels...
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
axs[0].set_title('Default Date Formatter')
plt.show()

#############################################################################
# The default date formatter is quite verbose, so we have the option of
# using `~.dates.ConciseDateFormatter`, as shown below.  Note that
# for this example the labels do not need to be rotated as they do for the
# default formatter because the labels are as small as possible.

fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(6, 6))
for nn, ax in enumerate(axs):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.plot(dates, y)
    ax.set_xlim(lims[nn])
axs[0].set_title('Concise Date Formatter')

plt.show()

#############################################################################
# If all calls to axes that have dates are to be made using this converter,
# it is probably most convenient to use the units registry where you do
# imports:

import matplotlib.units as munits
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

fig, axs = plt.subplots(3, 1, figsize=(6, 6), constrained_layout=True)
for nn, ax in enumerate(axs):
    ax.plot(dates, y)
    ax.set_xlim(lims[nn])
axs[0].set_title('Concise Date Formatter')

plt.show()

#############################################################################
# Localization of date formats
# ============================
#
# Dates formats can be localized if the default formats are not desirable by
# manipulating one of three lists of strings.
#
# The ``formatter.formats`` list of formats is for the normal tick labels,
# There are six levels: years, months, days, hours, minutes, seconds.
# The ``formatter.offset_formats`` is how the "offset" string on the right
# of the axis is formatted.  This is usually much more verbose than the tick
# labels. Finally, the ``formatter.zero_formats`` are the formats of the
# ticks that are "zeros".  These are tick values that are either the first of
# the year, month, or day of month, or the zeroth hour, minute, or second.
# These are usually the same as the format of
# the ticks a level above.  For example if the axis limits mean the ticks are
# mostly days, then we label 1 Mar 2005 simply with a "Mar".  If the axis
# limits are mostly hours, we label Feb 4 00:00 as simply "Feb-4".
#
# Note that these format lists can also be passed to `.ConciseDateFormatter`
# as optional keyword arguments.
#
# Here we modify the labels to be "day month year", instead of the ISO
# "year month day":

fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(6, 6))
```
### 4 - examples/ticks/tick-formatters.py:

Start line: 37, End line: 135

```python
# Tick formatters can be set in one of two ways, either by passing a ``str``
# or function to `~.Axis.set_major_formatter` or `~.Axis.set_minor_formatter`,
# or by creating an instance of one of the various `~.ticker.Formatter` classes
# and providing that to `~.Axis.set_major_formatter` or
# `~.Axis.set_minor_formatter`.

# The first two examples directly pass a ``str`` or function.

fig0, axs0 = plt.subplots(2, 1, figsize=(8, 2))
fig0.suptitle('Simple Formatting')

# A ``str``, using format string function syntax, can be used directly as a
# formatter.  The variable ``x`` is the tick value and the variable ``pos`` is
# tick position.  This creates a StrMethodFormatter automatically.
setup(axs0[0], title="'{x} km'")
axs0[0].xaxis.set_major_formatter('{x} km')

# A function can also be used directly as a formatter. The function must take
# two arguments: ``x`` for the tick value and ``pos`` for the tick position,
# and must return a ``str``  This creates a FuncFormatter automatically.
setup(axs0[1], title="lambda x, pos: str(x-5)")
axs0[1].xaxis.set_major_formatter(lambda x, pos: str(x-5))

fig0.tight_layout()


# The remaining examples use Formatter objects.

fig1, axs1 = plt.subplots(7, 1, figsize=(8, 6))
fig1.suptitle('Formatter Object Formatting')

# Null formatter
setup(axs1[0], title="NullFormatter()")
axs1[0].xaxis.set_major_formatter(ticker.NullFormatter())

# StrMethod formatter
setup(axs1[1], title="StrMethodFormatter('{x:.3f}')")
axs1[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))


# FuncFormatter can be used as a decorator
@ticker.FuncFormatter
def major_formatter(x, pos):
    return f'[{x:.2f}]'


setup(axs1[2], title='FuncFormatter("[{:.2f}]".format)')
axs1[2].xaxis.set_major_formatter(major_formatter)

# Fixed formatter
setup(axs1[3], title="FixedFormatter(['A', 'B', 'C', ...])")
# FixedFormatter should only be used together with FixedLocator.
# Otherwise, one cannot be sure where the labels will end up.
positions = [0, 1, 2, 3, 4, 5]
labels = ['A', 'B', 'C', 'D', 'E', 'F']
axs1[3].xaxis.set_major_locator(ticker.FixedLocator(positions))
axs1[3].xaxis.set_major_formatter(ticker.FixedFormatter(labels))

# Scalar formatter
setup(axs1[4], title="ScalarFormatter()")
axs1[4].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# FormatStr formatter
setup(axs1[5], title="FormatStrFormatter('#%d')")
axs1[5].xaxis.set_major_formatter(ticker.FormatStrFormatter("#%d"))

# Percent formatter
setup(axs1[6], title="PercentFormatter(xmax=5)")
axs1[6].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))

fig1.tight_layout()
plt.show()


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.axes.Axes.text`
#    - `matplotlib.axis.Axis.set_major_formatter`
#    - `matplotlib.axis.Axis.set_major_locator`
#    - `matplotlib.axis.Axis.set_minor_locator`
#    - `matplotlib.axis.XAxis.set_ticks_position`
#    - `matplotlib.axis.YAxis.set_ticks_position`
#    - `matplotlib.ticker.FixedFormatter`
#    - `matplotlib.ticker.FixedLocator`
#    - `matplotlib.ticker.FormatStrFormatter`
#    - `matplotlib.ticker.FuncFormatter`
#    - `matplotlib.ticker.MultipleLocator`
#    - `matplotlib.ticker.NullFormatter`
#    - `matplotlib.ticker.NullLocator`
#    - `matplotlib.ticker.PercentFormatter`
#    - `matplotlib.ticker.ScalarFormatter`
#    - `matplotlib.ticker.StrMethodFormatter`
```
### 5 - lib/matplotlib/axis.py:

Start line: 1583, End line: 1621

```python
class Axis(martist.Artist):

    def set_minor_formatter(self, formatter):
        """
        Set the formatter of the minor ticker.

        In addition to a `~matplotlib.ticker.Formatter` instance,
        this also accepts a ``str`` or function.
        See `.Axis.set_major_formatter` for more information.

        Parameters
        ----------
        formatter : `~matplotlib.ticker.Formatter`, ``str``, or function
        """
        self._set_formatter(formatter, self.minor)

    def _set_formatter(self, formatter, level):
        if isinstance(formatter, str):
            formatter = mticker.StrMethodFormatter(formatter)
        # Don't allow any other TickHelper to avoid easy-to-make errors,
        # like using a Locator instead of a Formatter.
        elif (callable(formatter) and
              not isinstance(formatter, mticker.TickHelper)):
            formatter = mticker.FuncFormatter(formatter)
        else:
            _api.check_isinstance(mticker.Formatter, formatter=formatter)

        if (isinstance(formatter, mticker.FixedFormatter)
                and len(formatter.seq) > 0
                and not isinstance(level.locator, mticker.FixedLocator)):
            _api.warn_external('FixedFormatter should only be used together '
                               'with FixedLocator')

        if level == self.major:
            self.isDefault_majfmt = False
        else:
            self.isDefault_minfmt = False

        level.formatter = formatter
        formatter.set_axis(self)
        self.stale = True
```
### 6 - lib/matplotlib/axes/_base.py:

Start line: 122, End line: 209

```python
def _process_plot_format(fmt):
    """
    Convert a MATLAB style color/line style format string to a (*linestyle*,
    *marker*, *color*) tuple.

    Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines
    * 'C2--': the third color in the color cycle, dashed lines

    The format is absolute in the sense that if a linestyle or marker is not
    defined in *fmt*, there is no line or marker. This is expressed by
    returning 'None' for the respective quantity.

    See Also
    --------
    matplotlib.Line2D.lineStyles, matplotlib.colors.cnames
        All possible styles and color format strings.
    """

    linestyle = None
    marker = None
    color = None

    # Is fmt just a colorspec?
    try:
        color = mcolors.to_rgba(fmt)

        # We need to differentiate grayscale '1.0' from tri_down marker '1'
        try:
            fmtint = str(int(fmt))
        except ValueError:
            return linestyle, marker, color  # Yes
        else:
            if fmt != fmtint:
                # user definitely doesn't want tri_down marker
                return linestyle, marker, color  # Yes
            else:
                # ignore converted color
                color = None
    except ValueError:
        pass  # No, not just a color.

    i = 0
    while i < len(fmt):
        c = fmt[i]
        if fmt[i:i+2] in mlines.lineStyles:  # First, the two-char styles.
            if linestyle is not None:
                raise ValueError(
                    'Illegal format string "%s"; two linestyle symbols' % fmt)
            linestyle = fmt[i:i+2]
            i += 2
        elif c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(
                    'Illegal format string "%s"; two linestyle symbols' % fmt)
            linestyle = c
            i += 1
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(
                    'Illegal format string "%s"; two marker symbols' % fmt)
            marker = c
            i += 1
        elif c in mcolors.get_named_colors_mapping():
            if color is not None:
                raise ValueError(
                    'Illegal format string "%s"; two color symbols' % fmt)
            color = c
            i += 1
        elif c == 'C' and i < len(fmt) - 1:
            color_cycle_number = int(fmt[i + 1])
            color = mcolors.to_rgba("C{}".format(color_cycle_number))
            i += 2
        else:
            raise ValueError(
                'Unrecognized character %c in format string' % c)

    if linestyle is None and marker is None:
        linestyle = mpl.rcParams['lines.linestyle']
    if linestyle is None:
        linestyle = 'None'
    if marker is None:
        marker = 'None'

    return linestyle, marker, color
```
### 7 - lib/matplotlib/axis.py:

Start line: 1557, End line: 1581

```python
class Axis(martist.Artist):

    def set_major_formatter(self, formatter):
        """
        Set the formatter of the major ticker.

        In addition to a `~matplotlib.ticker.Formatter` instance,
        this also accepts a ``str`` or function.

        For a ``str`` a `~matplotlib.ticker.StrMethodFormatter` is used.
        The field used for the value must be labeled ``'x'`` and the field used
        for the position must be labeled ``'pos'``.
        See the  `~matplotlib.ticker.StrMethodFormatter` documentation for
        more information.

        For a function, a `~matplotlib.ticker.FuncFormatter` is used.
        The function must take two inputs (a tick value ``x`` and a
        position ``pos``), and return a string containing the corresponding
        tick label.
        See the  `~matplotlib.ticker.FuncFormatter` documentation for
        more information.

        Parameters
        ----------
        formatter : `~matplotlib.ticker.Formatter`, ``str``, or function
        """
        self._set_formatter(formatter, self.major)
```
### 8 - tutorials/intermediate/artists.py:

Start line: 616, End line: 727

```python
axis = ax.xaxis
axis.get_ticklocs()

###############################################################################

axis.get_ticklabels()

###############################################################################
# note there are twice as many ticklines as labels because by default there are
# tick lines at the top and bottom but only tick labels below the xaxis;
# however, this can be customized.

axis.get_ticklines()

###############################################################################
# And with the above methods, you only get lists of major ticks back by
# default, but you can also ask for the minor ticks:

axis.get_ticklabels(minor=True)
axis.get_ticklines(minor=True)

###############################################################################
# Here is a summary of some of the useful accessor methods of the ``Axis``
# (these have corresponding setters where useful, such as
# :meth:`~matplotlib.axis.Axis.set_major_formatter`.)
#
# =============================  ==============================================
# Axis accessor method           Description
# =============================  ==============================================
# `~.Axis.get_scale`             The scale of the Axis, e.g., 'log' or 'linear'
# `~.Axis.get_view_interval`     The interval instance of the Axis view limits
# `~.Axis.get_data_interval`     The interval instance of the Axis data limits
# `~.Axis.get_gridlines`         A list of grid lines for the Axis
# `~.Axis.get_label`             The Axis label - a `.Text` instance
# `~.Axis.get_offset_text`       The Axis offset text - a `.Text` instance
# `~.Axis.get_ticklabels`        A list of `.Text` instances -
#                                keyword minor=True|False
# `~.Axis.get_ticklines`         A list of `.Line2D` instances -
#                                keyword minor=True|False
# `~.Axis.get_ticklocs`          A list of Tick locations -
#                                keyword minor=True|False
# `~.Axis.get_major_locator`     The `.ticker.Locator` instance for major ticks
# `~.Axis.get_major_formatter`   The `.ticker.Formatter` instance for major
#                                ticks
# `~.Axis.get_minor_locator`     The `.ticker.Locator` instance for minor ticks
# `~.Axis.get_minor_formatter`   The `.ticker.Formatter` instance for minor
#                                ticks
# `~.axis.Axis.get_major_ticks`  A list of `.Tick` instances for major ticks
# `~.axis.Axis.get_minor_ticks`  A list of `.Tick` instances for minor ticks
# `~.Axis.grid`                  Turn the grid on or off for the major or minor
#                                ticks
# =============================  ==============================================
#
# Here is an example, not recommended for its beauty, which customizes
# the Axes and Tick properties.

# plt.figure creates a matplotlib.figure.Figure instance
fig = plt.figure()
rect = fig.patch  # a rectangle instance
rect.set_facecolor('lightgoldenrodyellow')

ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
rect = ax1.patch
rect.set_facecolor('lightslategray')


for label in ax1.xaxis.get_ticklabels():
    # label is a Text instance
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(16)

for line in ax1.yaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('green')
    line.set_markersize(25)
    line.set_markeredgewidth(3)

plt.show()

###############################################################################
# .. _tick-container:
#
# Tick containers
# ---------------
#
# The :class:`matplotlib.axis.Tick` is the final container object in our
# descent from the :class:`~matplotlib.figure.Figure` to the
# :class:`~matplotlib.axes.Axes` to the :class:`~matplotlib.axis.Axis`
# to the :class:`~matplotlib.axis.Tick`.  The ``Tick`` contains the tick
# and grid line instances, as well as the label instances for the upper
# and lower ticks.  Each of these is accessible directly as an attribute
# of the ``Tick``.
#
# ==============  ==========================================================
# Tick attribute  Description
# ==============  ==========================================================
# tick1line       A `.Line2D` instance
# tick2line       A `.Line2D` instance
# gridline        A `.Line2D` instance
# label1          A `.Text` instance
# label2          A `.Text` instance
# ==============  ==========================================================
#
# Here is an example which sets the formatter for the right side ticks with
# dollar signs and colors them green on the right side of the yaxis.
#
#
# .. include:: ../../gallery/pyplots/dollar_ticks.rst
#    :start-after: y axis labels.
#    :end-before: .. admonition:: References
```
### 9 - lib/matplotlib/colorbar.py:

Start line: 369, End line: 511

```python
class Colorbar:

    def __init__(self, ax, mappable=None, *, cmap=None,
                 norm=None,
                 alpha=None,
                 values=None,
                 boundaries=None,
                 orientation='vertical',
                 ticklocation='auto',
                 extend=None,
                 spacing='uniform',  # uniform or proportional
                 ticks=None,
                 format=None,
                 drawedges=False,
                 filled=True,
                 extendfrac=None,
                 extendrect=False,
                 label='',
                 ):

        if mappable is None:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        # Ensure the given mappable's norm has appropriate vmin and vmax
        # set even if mappable.draw has not yet been called.
        if mappable.get_array() is not None:
            mappable.autoscale_None()

        self.mappable = mappable
        cmap = mappable.cmap
        norm = mappable.norm

        if isinstance(mappable, contour.ContourSet):
            cs = mappable
            alpha = cs.get_alpha()
            boundaries = cs._levels
            values = cs.cvalues
            extend = cs.extend
            filled = cs.filled
            if ticks is None:
                ticks = ticker.FixedLocator(cs.levels, nbins=10)
        elif isinstance(mappable, martist.Artist):
            alpha = mappable.get_alpha()

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacks.connect(
            'changed', self.update_normal)

        _api.check_in_list(
            ['vertical', 'horizontal'], orientation=orientation)
        _api.check_in_list(
            ['auto', 'left', 'right', 'top', 'bottom'],
            ticklocation=ticklocation)
        _api.check_in_list(
            ['uniform', 'proportional'], spacing=spacing)

        self.ax = ax
        self.ax._axes_locator = _ColorbarAxesLocator(self)

        if extend is None:
            if (not isinstance(mappable, contour.ContourSet)
                    and getattr(cmap, 'colorbar_extend', False) is not False):
                extend = cmap.colorbar_extend
            elif hasattr(norm, 'extend'):
                extend = norm.extend
            else:
                extend = 'neither'
        self.alpha = None
        # Call set_alpha to handle array-like alphas properly
        self.set_alpha(alpha)
        self.cmap = cmap
        self.norm = norm
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = _api.check_getitem(
            {'neither': slice(0, None), 'both': slice(1, -1),
             'min': slice(1, None), 'max': slice(0, -1)},
            extend=extend)
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self.filled = filled
        self.extendfrac = extendfrac
        self.extendrect = extendrect
        self.solids = None
        self.solids_patches = []
        self.lines = []

        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)
        self._short_axis().set_visible(False)
        # Only kept for backcompat; remove after deprecation of .patch elapses.
        self._patch = mpatches.Polygon(
            np.empty((0, 2)),
            color=mpl.rcParams['axes.facecolor'], linewidth=0.01, zorder=-1)
        ax.add_artist(self._patch)

        self.dividers = collections.LineCollection(
            [],
            colors=[mpl.rcParams['axes.edgecolor']],
            linewidths=[0.5 * mpl.rcParams['axes.linewidth']])
        self.ax.add_collection(self.dividers)

        self.locator = None
        self.minorlocator = None
        self.formatter = None
        self.__scale = None  # linear, log10 for now.  Hopefully more?

        if ticklocation == 'auto':
            ticklocation = 'bottom' if orientation == 'horizontal' else 'right'
        self.ticklocation = ticklocation

        self.set_label(label)
        self._reset_locator_formatter_scale()

        if np.iterable(ticks):
            self.locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self.locator = ticks    # Handle default in _ticker()

        if isinstance(format, str):
            self.formatter = ticker.FormatStrFormatter(format)
        else:
            self.formatter = format  # Assume it is a Formatter or None
        self.draw_all()

        if isinstance(mappable, contour.ContourSet) and not mappable.filled:
            self.add_lines(mappable)

        # Link the Axes and Colorbar for interactive use
        self.ax._colorbar = self
        # Don't navigate on any of these types of mappables
        if (isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm)) or
                isinstance(self.mappable, contour.ContourSet)):
            self.ax.set_navigate(False)

        # These are the functions that set up interactivity on this colorbar
        self._interactive_funcs = ["_get_view", "_set_view",
                                   "_set_view_from_bbox", "drag_pan"]
        for x in self._interactive_funcs:
            setattr(self.ax, x, getattr(self, x))
        # Set the cla function to the cbar's method to override it
        self.ax.cla = self._cbar_cla
```
### 10 - examples/ticks/tick-formatters.py:

Start line: 1, End line: 34

```python
"""
===============
Tick formatters
===============

Tick formatters define how the numeric value associated with a tick on an axis
is formatted as a string.

This example illustrates the usage and effect of the most common formatters.
"""

import matplotlib.pyplot as plt
from matplotlib import ticker


def setup(ax, title):
    """Set up common parameters for the Axes in the example."""
    # only show the bottom spine
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines.right.set_color('none')
    ax.spines.left.set_color('none')
    ax.spines.top.set_color('none')

    # define tick positions
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.2, title, transform=ax.transAxes,
            fontsize=14, fontname='Monospace', color='tab:blue')
```
### 12 - lib/matplotlib/colorbar.py:

Start line: 63, End line: 124

```python
_colormap_kw_doc = """

    ============  ====================================================
    Property      Description
    ============  ====================================================
    *extend*      {'neither', 'both', 'min', 'max'}
                  If not 'neither', make pointed end(s) for out-of-
                  range values.  These are set for a given colormap
                  using the colormap set_under and set_over methods.
    *extendfrac*  {*None*, 'auto', length, lengths}
                  If set to *None*, both the minimum and maximum
                  triangular colorbar extensions with have a length of
                  5% of the interior colorbar length (this is the
                  default setting). If set to 'auto', makes the
                  triangular colorbar extensions the same lengths as
                  the interior boxes (when *spacing* is set to
                  'uniform') or the same lengths as the respective
                  adjacent interior boxes (when *spacing* is set to
                  'proportional'). If a scalar, indicates the length
                  of both the minimum and maximum triangular colorbar
                  extensions as a fraction of the interior colorbar
                  length. A two-element sequence of fractions may also
                  be given, indicating the lengths of the minimum and
                  maximum colorbar extensions respectively as a
                  fraction of the interior colorbar length.
    *extendrect*  bool
                  If *False* the minimum and maximum colorbar extensions
                  will be triangular (the default). If *True* the
                  extensions will be rectangular.
    *spacing*     {'uniform', 'proportional'}
                  Uniform spacing gives each discrete color the same
                  space; proportional makes the space proportional to
                  the data interval.
    *ticks*       *None* or list of ticks or Locator
                  If None, ticks are determined automatically from the
                  input.
    *format*      None or str or Formatter
                  If None, `~.ticker.ScalarFormatter` is used.
                  If a format string is given, e.g., '%.3f', that is used.
                  An alternative `~.ticker.Formatter` may be given instead.
    *drawedges*   bool
                  Whether to draw lines at color boundaries.
    *label*       str
                  The label on the colorbar's long axis.
    ============  ====================================================

    The following will probably be useful only in the context of
    indexed colors (that is, when the mappable has norm=NoNorm()),
    or other unusual circumstances.

    ============   ===================================================
    Property       Description
    ============   ===================================================
    *boundaries*   None or a sequence
    *values*       None or a sequence which must be of length 1 less
                   than the sequence of *boundaries*. For each region
                   delimited by adjacent entries in *boundaries*, the
                   colormapped to the corresponding value in values
                   will be used.
    ============   ===================================================

"""
```
### 13 - lib/matplotlib/colorbar.py:

Start line: 1510, End line: 1568

```python
@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes_gridspec(parent, *, location=None, orientation=None,
                       fraction=0.15, shrink=1.0, aspect=20, **kwargs):
    # ... other code

    if location in ('left', 'right'):
        # for shrinking
        height_ratios = [
                (1-anchor[1])*(1-shrink), shrink, anchor[1]*(1-shrink)]

        if location == 'left':
            gs = parent.get_subplotspec().subgridspec(
                    1, 2, wspace=wh_space,
                    width_ratios=[fraction, 1-fraction-pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(
                    3, 1, hspace=0, height_ratios=height_ratios)[1]
        else:
            gs = parent.get_subplotspec().subgridspec(
                    1, 2, wspace=wh_space,
                    width_ratios=[1-fraction-pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(
                    3, 1, hspace=0, height_ratios=height_ratios)[1]
    else:
        # for shrinking
        width_ratios = [
                anchor[0]*(1-shrink), shrink, (1-anchor[0])*(1-shrink)]

        if location == 'bottom':
            gs = parent.get_subplotspec().subgridspec(
                    2, 1, hspace=wh_space,
                    height_ratios=[1-fraction-pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(
                    1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect
        else:
            gs = parent.get_subplotspec().subgridspec(
                    2, 1, hspace=wh_space,
                    height_ratios=[fraction, 1-fraction-pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(
                    1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect

    parent.set_subplotspec(ss_main)
    parent.set_anchor(panchor)

    fig = parent.get_figure()
    cax = fig.add_subplot(ss_cb, label="<colorbar>")
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')
    cax._colorbar_info = dict(
        location=location,
        parents=[parent],
        shrink=shrink,
        anchor=anchor,
        panchor=panchor,
        fraction=fraction,
        aspect=aspect0,
        pad=pad)
    return cax, kwargs
```
### 14 - lib/matplotlib/colorbar.py:

Start line: 889, End line: 924

```python
class Colorbar:

    @_api.delete_parameter("3.5", "update_ticks")
    def set_ticklabels(self, ticklabels, update_ticks=True, *, minor=False,
                       **kwargs):
        """
        Set tick labels.

        .. admonition:: Discouraged

            The use of this method is discouraged, because of the dependency
            on tick positions. In most cases, you'll want to use
            ``set_ticks(positions, labels=labels)`` instead.

            If you are using this method, you should always fix the tick
            positions before, e.g. by using `.Colorbar.set_ticks` or by
            explicitly setting a `~.ticker.FixedLocator` on the long axis
            of the colorbar. Otherwise, ticks are free to move and the
            labels may end up in unexpected positions.

        Parameters
        ----------
        ticklabels : sequence of str or of `.Text`
            Texts for labeling each tick location in the sequence set by
            `.Colorbar.set_ticks`; the number of labels must match the number
            of locations.

        update_ticks : bool, default: True
            This keyword argument is ignored and will be be removed.
            Deprecated

         minor : bool
            If True, set minor ticks instead of major ticks.

        **kwargs
            `.Text` properties for the labels.
        """
        self._long_axis().set_ticklabels(ticklabels, minor=minor, **kwargs)
```
### 15 - lib/matplotlib/colorbar.py:

Start line: 1439, End line: 1456

```python
@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes(parents, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kwargs):
    # ... other code
    for a in parents:
        # tell the parent it has a colorbar
        a._colorbars += [cax]
    cax._colorbar_info = dict(
        location=location,
        parents=parents,
        shrink=shrink,
        anchor=anchor,
        panchor=panchor,
        fraction=fraction,
        aspect=aspect0,
        pad=pad)
    # and we need to set the aspect ratio by hand...
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')

    return cax, kwargs
```
### 22 - lib/matplotlib/colorbar.py:

Start line: 640, End line: 709

```python
class Colorbar:

    def _do_extends(self, extendlen):
        """
        Add the extend tri/rectangles on the outside of the axes.
        """
        # extend lengths are fraction of the *inner* part of colorbar,
        # not the total colorbar:
        bot = 0 - (extendlen[0] if self._extend_lower() else 0)
        top = 1 + (extendlen[1] if self._extend_upper() else 0)

        # xyout is the outline of the colorbar including the extend patches:
        if not self.extendrect:
            # triangle:
            xyout = np.array([[0, 0], [0.5, bot], [1, 0],
                              [1, 1], [0.5, top], [0, 1], [0, 0]])
        else:
            # rectangle:
            xyout = np.array([[0, 0], [0, bot], [1, bot], [1, 0],
                              [1, 1], [1, top], [0, top], [0, 1],
                              [0, 0]])

        if self.orientation == 'horizontal':
            xyout = xyout[:, ::-1]

        # xyout is the path for the spine:
        self.outline.set_xy(xyout)
        if not self.filled:
            return

        # Make extend triangles or rectangles filled patches.  These are
        # defined in the outer parent axes' coordinates:
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            hatches = mappable.hatches
        else:
            hatches = [None]

        if self._extend_lower():
            if not self.extendrect:
                # triangle
                xy = np.array([[0, 0], [0.5, bot], [1, 0]])
            else:
                # rectangle
                xy = np.array([[0, 0], [0, bot], [1., bot], [1, 0]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            color = self.cmap(self.norm(self._values[0]))
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color, linewidth=0,
                antialiased=False, transform=self.ax.transAxes,
                hatch=hatches[0], clip_on=False)
            self.ax.add_patch(patch)
        if self._extend_upper():
            if not self.extendrect:
                # triangle
                xy = np.array([[0, 1], [0.5, top], [1, 1]])
            else:
                # rectangle
                xy = np.array([[0, 1], [0, top], [1, top], [1, 1]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            color = self.cmap(self.norm(self._values[-1]))
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color,
                linewidth=0, antialiased=False,
                transform=self.ax.transAxes, hatch=hatches[-1], clip_on=False)
            self.ax.add_patch(patch)
        return
```
### 23 - lib/matplotlib/colorbar.py:

Start line: 1356, End line: 1438

```python
@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes(parents, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The axes is placed in the figure of the *parents* axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~.axes.Axes` or list of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %s

    Returns
    -------
    cax : `~.axes.Axes`
        The child axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.

    Other Parameters
    ----------------
    %s
    """
    loc_settings = _normalize_location_orientation(location, orientation)
    # put appropriate values into the kwargs dict for passing back to
    # the Colorbar class
    kwargs['orientation'] = loc_settings['orientation']
    location = kwargs['ticklocation'] = loc_settings['location']

    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    aspect0 = aspect
    # turn parents into a list if it is not already. We do this w/ np
    # because `plt.subplots` can return an ndarray and is natural to
    # pass to `colorbar`.
    parents = np.atleast_1d(parents).ravel()
    fig = parents[0].get_figure()

    pad0 = 0.05 if fig.get_constrained_layout() else loc_settings['pad']
    pad = kwargs.pop('pad', pad0)

    if not all(fig is ax.get_figure() for ax in parents):
        raise ValueError('Unable to create a colorbar axes as not all '
                         'parents share the same figure.')

    # take a bounding box around all of the given axes
    parents_bbox = mtransforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    pb = parents_bbox
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)

        # define the aspect ratio in terms of y's per x rather than x's per y
        aspect = 1.0 / aspect

    # define a transform which takes us from old axes coordinates to
    # new axes coordinates
    shrinking_trans = mtransforms.BboxTransform(parents_bbox, pb1)

    # transform each of the axes in parents using the new transform
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = mtransforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if panchor is not False:
            ax.set_anchor(panchor)

    cax = fig.add_axes(pbcb, label="<colorbar>")
    # ... other code
```
### 24 - lib/matplotlib/colorbar.py:

Start line: 126, End line: 194

```python
docstring.interpd.update(colorbar_doc="""
Add a colorbar to a plot.

Parameters
----------
mappable
    The `matplotlib.cm.ScalarMappable` (i.e., `~matplotlib.image.AxesImage`,
    `~matplotlib.contour.ContourSet`, etc.) described by this colorbar.
    This argument is mandatory for the `.Figure.colorbar` method but optional
    for the `.pyplot.colorbar` function, which sets the default to the current
    image.

    Note that one can create a `.ScalarMappable` "on-the-fly" to generate
    colorbars not attached to a previously drawn artist, e.g. ::

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

cax : `~matplotlib.axes.Axes`, optional
    Axes into which the colorbar will be drawn.

ax : `~matplotlib.axes.Axes`, list of Axes, optional
    One or more parent axes from which space for a new colorbar axes will be
    stolen, if *cax* is None.  This has no effect if *cax* is set.

use_gridspec : bool, optional
    If *cax* is ``None``, a new *cax* is created as an instance of Axes.  If
    *ax* is an instance of Subplot and *use_gridspec* is ``True``, *cax* is
    created as an instance of Subplot using the :mod:`.gridspec` module.

Returns
-------
colorbar : `~matplotlib.colorbar.Colorbar`

Notes
-----
Additional keyword arguments are of two kinds:

  axes properties:
%s
%s
  colorbar properties:
%s

If *mappable* is a `~.contour.ContourSet`, its *extend* kwarg is included
automatically.

The *shrink* kwarg provides a simple way to scale the colorbar with respect
to the axes. Note that if *cax* is specified, it determines the size of the
colorbar and *shrink* and *aspect* kwargs are ignored.

For more precise control, you can manually specify the positions of
the axes objects in which the mappable and the colorbar are drawn.  In
this case, do not use any of the axes properties kwargs.

It is known that some vector graphics viewers (svg and pdf) renders white gaps
between segments of the colorbar.  This is due to bugs in the viewers, not
Matplotlib.  As a workaround, the colorbar can be rendered with overlapping
segments::

    cbar = colorbar()
    cbar.solids.set_edgecolor("face")
    draw()

However this has negative consequences in other circumstances, e.g. with
semi-transparent images (alpha < 1) and colorbar extensions; therefore, this
workaround is not used by default (see issue #1188).
""" % (textwrap.indent(_make_axes_param_doc, "    "),
       textwrap.indent(_make_axes_other_param_doc, "    "),
       _colormap_kw_doc))
```
### 25 - lib/matplotlib/colorbar.py:

Start line: 1332, End line: 1353

```python
# Backcompat API


def _normalize_location_orientation(location, orientation):
    if location is None:
        location = _api.check_getitem(
            {None: "right", "vertical": "right", "horizontal": "bottom"},
            orientation=orientation)
    loc_settings = _api.check_getitem({
        "left":   {"location": "left", "orientation": "vertical",
                   "anchor": (1.0, 0.5), "panchor": (0.0, 0.5), "pad": 0.10},
        "right":  {"location": "right", "orientation": "vertical",
                   "anchor": (0.0, 0.5), "panchor": (1.0, 0.5), "pad": 0.05},
        "top":    {"location": "top", "orientation": "horizontal",
                   "anchor": (0.5, 0.0), "panchor": (0.5, 1.0), "pad": 0.05},
        "bottom": {"location": "bottom", "orientation": "horizontal",
                   "anchor": (0.5, 1.0), "panchor": (0.5, 0.0), "pad": 0.15},
    }, location=location)
    if orientation is not None and orientation != loc_settings["orientation"]:
        # Allow the user to pass both if they are consistent.
        raise TypeError("location and orientation are mutually exclusive")
    return loc_settings
```
### 27 - lib/matplotlib/colorbar.py:

Start line: 1, End line: 61

```python
"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which internally use `.Colorbar` together with
   `.make_axes_gridspec` (for `.GridSpec`-positioned axes) or `.make_axes` (for
   non-`.GridSpec`-positioned axes).

   End-users most likely won't need to directly use this module's API.
"""

import copy
import logging
import textwrap

import numpy as np

import matplotlib as mpl
from matplotlib import _api, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.scale as mscale
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import docstring

_log = logging.getLogger(__name__)

_make_axes_param_doc = """
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent axes, where the colorbar axes
    is created.  It also determines the *orientation* of the colorbar
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.
orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbar.  It is preferable to set the *location*
    of the colorbar, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.
fraction : float, default: 0.15
    Fraction of original axes to use for colorbar.
shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.
aspect : float, default: 20
    Ratio of long to short dimensions.
"""
_make_axes_other_param_doc = """
pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original axes between colorbar and new image axes.
anchor : (float, float), optional
    The anchor point of the colorbar axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.
panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
"""
```
### 33 - lib/matplotlib/colorbar.py:

Start line: 926, End line: 964

```python
class Colorbar:

    def minorticks_on(self):
        """
        Turn on colorbar minor ticks.
        """
        self.ax.minorticks_on()
        self.minorlocator = self._long_axis().get_minor_locator()
        self._short_axis().set_minor_locator(ticker.NullLocator())

    def minorticks_off(self):
        """Turn the minor ticks of the colorbar off."""
        self.minorlocator = ticker.NullLocator()
        self._long_axis().set_minor_locator(self.minorlocator)

    def set_label(self, label, *, loc=None, **kwargs):
        """
        Add a label to the long axis of the colorbar.

        Parameters
        ----------
        label : str
            The label text.
        loc : str, optional
            The location of the label.

            - For horizontal orientation one of {'left', 'center', 'right'}
            - For vertical orientation one of {'bottom', 'center', 'top'}

            Defaults to :rc:`xaxis.labellocation` or :rc:`yaxis.labellocation`
            depending on the orientation.
        **kwargs
            Keyword arguments are passed to `~.Axes.set_xlabel` /
            `~.Axes.set_ylabel`.
            Supported keywords are *labelpad* and `.Text` properties.
        """
        if self.orientation == "vertical":
            self.ax.set_ylabel(label, loc=loc, **kwargs)
        else:
            self.ax.set_xlabel(label, loc=loc, **kwargs)
        self.stale = True
```
### 41 - lib/matplotlib/colorbar.py:

Start line: 289, End line: 367

```python
class Colorbar:
    r"""
    Draw a colorbar in an existing axes.

    Typically, colorbars are created using `.Figure.colorbar` or
    `.pyplot.colorbar` and associated with `.ScalarMappable`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    In order to draw a colorbar not associated with other elements in the
    figure, e.g. when showing a colormap by itself, one can create an empty
    `.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
    to `Colorbar`.

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.

    mappable : `.ScalarMappable`
        The mappable whose colormap and norm will be used.

        To show the under- and over- value colors, the mappable's norm should
        be specified as ::

            norm = colors.Normalize(clip=False)

        To show the colors versus index instead of on a 0-1 scale, use::

            norm=colors.NoNorm()

    cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The colormap to use.  This parameter is ignored, unless *mappable* is
        None.

    norm : `~matplotlib.colors.Normalize`
        The normalization to use.  This parameter is ignored, unless *mappable*
        is None.

    alpha : float
        The colorbar transparency between 0 (transparent) and 1 (opaque).

    values, boundaries
        If unset, the colormap will be displayed on a 0-1 scale.

    orientation : {'vertical', 'horizontal'}

    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}

    extend : {'neither', 'both', 'min', 'max'}

    spacing : {'uniform', 'proportional'}

    ticks : `~matplotlib.ticker.Locator` or array-like of float

    format : str or `~matplotlib.ticker.Formatter`

    drawedges : bool

    filled : bool

    extendfrac

    extendrec

    label : str
    """

    n_rasterize = 50  # rasterize solids if number of colors >= n_rasterize
```
