# matplotlib__matplotlib-25651

| **matplotlib/matplotlib** | `bff46815c9b6b2300add1ed25f18b3d788b816de` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 3 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/lib/matplotlib/ticker.py b/lib/matplotlib/ticker.py
--- a/lib/matplotlib/ticker.py
+++ b/lib/matplotlib/ticker.py
@@ -2244,6 +2244,7 @@ class LogLocator(Locator):
 
     """
 
+    @_api.delete_parameter("3.8", "numdecs")
     def __init__(self, base=10.0, subs=(1.0,), numdecs=4, numticks=None):
         """Place ticks on the locations : subs[j] * base**i."""
         if numticks is None:
@@ -2253,9 +2254,10 @@ def __init__(self, base=10.0, subs=(1.0,), numdecs=4, numticks=None):
                 numticks = 'auto'
         self._base = float(base)
         self._set_subs(subs)
-        self.numdecs = numdecs
+        self._numdecs = numdecs
         self.numticks = numticks
 
+    @_api.delete_parameter("3.8", "numdecs")
     def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
         """Set parameters within this locator."""
         if base is not None:
@@ -2263,10 +2265,13 @@ def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
         if subs is not None:
             self._set_subs(subs)
         if numdecs is not None:
-            self.numdecs = numdecs
+            self._numdecs = numdecs
         if numticks is not None:
             self.numticks = numticks
 
+    numdecs = _api.deprecate_privatize_attribute(
+        "3.8", addendum="This attribute has no effect.")
+
     def _set_subs(self, subs):
         """
         Set the minor ticks for the log scaling every ``base**i*subs[j]``.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/ticker.py | 2247 | 2247 | - | - | -
| lib/matplotlib/ticker.py | 2256 | 2256 | - | - | -
| lib/matplotlib/ticker.py | 2266 | 2266 | - | - | -


## Problem Statement

```
[MNT]: numdecs parameter in `LogLocator`
### Summary

`LogLocator` takes a parameter *numdecs*, which is not described in its docstring.  I also can't find anywhere the parameter is used in the code.
https://matplotlib.org/devdocs/api/ticker_api.html#matplotlib.ticker.LogLocator

*numdec* (no s) is used within `tick_values`, but is calculated when needed

https://github.com/matplotlib/matplotlib/blob/61ed3f40057a48821ccad758fd5f04f0df1b8aab/lib/matplotlib/ticker.py#L2322

### Proposed fix

If *numdecs* really isn't used, maybe remove it.  Otherwise describe it in the docstring.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 galleries/examples/ticks/tick-locators.py | 32 | 94| 642 | 642 | 853 | 
| 2 | 1 galleries/examples/ticks/tick-locators.py | 1 | 29| 211 | 853 | 853 | 
| 3 | 2 galleries/examples/ticks/date_formatters_locators.py | 1 | 39| 467 | 1320 | 1833 | 
| 4 | 3 lib/matplotlib/dates.py | 1117 | 1172| 422 | 1742 | 18784 | 
| 5 | 3 galleries/examples/ticks/date_formatters_locators.py | 60 | 94| 304 | 2046 | 18784 | 
| 6 | 4 galleries/examples/ticks/ticks_too_many.py | 1 | 77| 744 | 2790 | 19528 | 
| 7 | 5 galleries/examples/ticks/date_concise_formatter.py | 114 | 185| 613 | 3403 | 21176 | 
| 8 | 6 galleries/users_explain/text/text_intro.py | 269 | 336| 832 | 4235 | 25053 | 
| 9 | 7 lib/matplotlib/mlab.py | 403 | 458| 756 | 4991 | 32571 | 
| 10 | 8 galleries/examples/ticks/major_minor_demo.py | 1 | 97| 828 | 5819 | 33399 | 
| 11 | 8 lib/matplotlib/dates.py | 1 | 172| 1565 | 7384 | 33399 | 
| 12 | 9 galleries/examples/ticks/date_precision_and_epochs.py | 86 | 159| 651 | 8035 | 34789 | 
| 13 | 9 lib/matplotlib/dates.py | 1372 | 1399| 280 | 8315 | 34789 | 
| 14 | 9 galleries/examples/ticks/date_precision_and_epochs.py | 1 | 84| 739 | 9054 | 34789 | 
| 15 | 9 lib/matplotlib/dates.py | 1708 | 1776| 537 | 9591 | 34789 | 
| 16 | 10 lib/mpl_toolkits/axisartist/angle_helper.py | 186 | 218| 310 | 9901 | 38482 | 
| 17 | 10 galleries/examples/ticks/date_concise_formatter.py | 1 | 112| 1035 | 10936 | 38482 | 
| 18 | 11 galleries/examples/images_contours_and_fields/contourf_log.py | 1 | 63| 494 | 11430 | 38976 | 
| 19 | 12 lib/matplotlib/axis.py | 433 | 456| 220 | 11650 | 60478 | 
| 20 | 13 lib/matplotlib/colorbar.py | 438 | 484| 334 | 11984 | 74712 | 
| 21 | 14 galleries/examples/text_labels_and_annotations/engineering_formatter.py | 1 | 45| 367 | 12351 | 75079 | 
| 22 | 14 galleries/users_explain/text/text_intro.py | 337 | 430| 752 | 13103 | 75079 | 
| 23 | 14 lib/matplotlib/dates.py | 1174 | 1190| 193 | 13296 | 75079 | 
| 24 | 15 galleries/examples/ticks/tick-formatters.py | 37 | 137| 970 | 14266 | 76308 | 
| 25 | 16 lib/mpl_toolkits/axisartist/grid_finder.py | 280 | 293| 140 | 14406 | 79225 | 
| 26 | 16 lib/matplotlib/dates.py | 1309 | 1370| 896 | 15302 | 79225 | 
| 27 | 17 galleries/examples/ticks/scalarformatter.py | 1 | 85| 731 | 16033 | 79956 | 
| 28 | 17 galleries/examples/ticks/tick-formatters.py | 1 | 34| 259 | 16292 | 79956 | 
| 29 | 17 lib/matplotlib/dates.py | 173 | 201| 312 | 16604 | 79956 | 
| 30 | 17 lib/matplotlib/dates.py | 1240 | 1266| 199 | 16803 | 79956 | 
| 31 | 18 lib/matplotlib/axes/_axes.py | 1 | 39| 286 | 17089 | 155563 | 
| 32 | 19 galleries/examples/text_labels_and_annotations/unicode_minus.py | 1 | 29| 252 | 17341 | 155815 | 
| 33 | 19 lib/mpl_toolkits/axisartist/grid_finder.py | 296 | 315| 170 | 17511 | 155815 | 
| 34 | 19 lib/matplotlib/axis.py | 494 | 517| 220 | 17731 | 155815 | 
| 35 | 19 galleries/examples/ticks/date_formatters_locators.py | 42 | 57| 209 | 17940 | 155815 | 
| 36 | 19 lib/matplotlib/axis.py | 340 | 395| 600 | 18540 | 155815 | 
| 37 | 20 lib/matplotlib/cm.py | 668 | 695| 360 | 18900 | 161480 | 
| 38 | 20 lib/matplotlib/dates.py | 1659 | 1681| 221 | 19121 | 161480 | 
| 39 | 21 galleries/examples/subplots_axes_and_figures/secondary_axis.py | 104 | 195| 656 | 19777 | 162913 | 
| 40 | 22 galleries/examples/scales/log_demo.py | 1 | 48| 351 | 20128 | 163264 | 
| 41 | 22 lib/matplotlib/axis.py | 520 | 558| 230 | 20358 | 163264 | 
| 42 | 22 lib/matplotlib/axis.py | 1833 | 1886| 381 | 20739 | 163264 | 
| 43 | 23 lib/matplotlib/_api/deprecation.py | 316 | 377| 566 | 21305 | 167581 | 
| 44 | 24 galleries/examples/scales/semilogx_demo.py | 1 | 24| 104 | 21409 | 167685 | 
| 45 | 24 lib/matplotlib/axis.py | 1592 | 1603| 119 | 21528 | 167685 | 
| 46 | 25 galleries/examples/text_labels_and_annotations/date.py | 1 | 65| 659 | 22187 | 168344 | 
| 47 | 26 lib/matplotlib/testing/decorators.py | 55 | 73| 141 | 22328 | 172014 | 
| 48 | 26 lib/matplotlib/_api/deprecation.py | 379 | 413| 391 | 22719 | 172014 | 
| 49 | 26 lib/matplotlib/dates.py | 790 | 873| 859 | 23578 | 172014 | 
| 50 | 26 lib/matplotlib/mlab.py | 1 | 77| 390 | 23968 | 172014 | 
| 51 | 27 lib/mpl_toolkits/mplot3d/axis3d.py | 166 | 184| 189 | 24157 | 177717 | 
| 52 | 28 galleries/examples/images_contours_and_fields/colormap_normalizations_symlognorm.py | 1 | 86| 822 | 24979 | 178539 | 
| 53 | 29 galleries/examples/ticks/date_index_formatter.py | 1 | 85| 748 | 25727 | 179287 | 
| 54 | 29 lib/matplotlib/axis.py | 1985 | 2001| 162 | 25889 | 179287 | 
| 55 | 30 lib/matplotlib/_pylab_helpers.py | 44 | 67| 179 | 26068 | 180207 | 
| 56 | 31 galleries/examples/scales/symlog_demo.py | 1 | 49| 325 | 26393 | 180532 | 
| 57 | 31 lib/matplotlib/colorbar.py | 29 | 114| 961 | 27354 | 180532 | 
| 58 | 31 lib/matplotlib/dates.py | 1606 | 1631| 284 | 27638 | 180532 | 
| 59 | 31 lib/matplotlib/dates.py | 1193 | 1214| 171 | 27809 | 180532 | 
| 60 | 31 lib/matplotlib/dates.py | 1478 | 1499| 259 | 28068 | 180532 | 
| 61 | 32 galleries/examples/statistics/barchart_demo.py | 27 | 44| 179 | 28247 | 181604 | 
| 62 | 33 lib/mpl_toolkits/axes_grid1/axes_divider.py | 202 | 225| 251 | 28498 | 187586 | 
| 63 | 33 lib/matplotlib/dates.py | 1401 | 1476| 801 | 29299 | 187586 | 


## Missing Patch Files

 * 1: lib/matplotlib/ticker.py

## Patch

```diff
diff --git a/lib/matplotlib/ticker.py b/lib/matplotlib/ticker.py
--- a/lib/matplotlib/ticker.py
+++ b/lib/matplotlib/ticker.py
@@ -2244,6 +2244,7 @@ class LogLocator(Locator):
 
     """
 
+    @_api.delete_parameter("3.8", "numdecs")
     def __init__(self, base=10.0, subs=(1.0,), numdecs=4, numticks=None):
         """Place ticks on the locations : subs[j] * base**i."""
         if numticks is None:
@@ -2253,9 +2254,10 @@ def __init__(self, base=10.0, subs=(1.0,), numdecs=4, numticks=None):
                 numticks = 'auto'
         self._base = float(base)
         self._set_subs(subs)
-        self.numdecs = numdecs
+        self._numdecs = numdecs
         self.numticks = numticks
 
+    @_api.delete_parameter("3.8", "numdecs")
     def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
         """Set parameters within this locator."""
         if base is not None:
@@ -2263,10 +2265,13 @@ def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
         if subs is not None:
             self._set_subs(subs)
         if numdecs is not None:
-            self.numdecs = numdecs
+            self._numdecs = numdecs
         if numticks is not None:
             self.numticks = numticks
 
+    numdecs = _api.deprecate_privatize_attribute(
+        "3.8", addendum="This attribute has no effect.")
+
     def _set_subs(self, subs):
         """
         Set the minor ticks for the log scaling every ``base**i*subs[j]``.

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_ticker.py b/lib/matplotlib/tests/test_ticker.py
--- a/lib/matplotlib/tests/test_ticker.py
+++ b/lib/matplotlib/tests/test_ticker.py
@@ -233,9 +233,11 @@ def test_set_params(self):
         See if change was successful. Should not raise exception.
         """
         loc = mticker.LogLocator()
-        loc.set_params(numticks=7, numdecs=8, subs=[2.0], base=4)
+        with pytest.warns(mpl.MatplotlibDeprecationWarning, match="numdecs"):
+            loc.set_params(numticks=7, numdecs=8, subs=[2.0], base=4)
         assert loc.numticks == 7
-        assert loc.numdecs == 8
+        with pytest.warns(mpl.MatplotlibDeprecationWarning, match="numdecs"):
+            assert loc.numdecs == 8
         assert loc._base == 4
         assert list(loc._subs) == [2.0]
 

```


## Code snippets

### 1 - galleries/examples/ticks/tick-locators.py:

Start line: 32, End line: 94

```python
fig, axs = plt.subplots(8, 1, figsize=(8, 6))

# Null Locator
setup(axs[0], title="NullLocator()")
axs[0].xaxis.set_major_locator(ticker.NullLocator())
axs[0].xaxis.set_minor_locator(ticker.NullLocator())

# Multiple Locator
setup(axs[1], title="MultipleLocator(0.5)")
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Fixed Locator
setup(axs[2], title="FixedLocator([0, 1, 5])")
axs[2].xaxis.set_major_locator(ticker.FixedLocator([0, 1, 5]))
axs[2].xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(0.2, 0.8, 4)))

# Linear Locator
setup(axs[3], title="LinearLocator(numticks=3)")
axs[3].xaxis.set_major_locator(ticker.LinearLocator(3))
axs[3].xaxis.set_minor_locator(ticker.LinearLocator(31))

# Index Locator
setup(axs[4], title="IndexLocator(base=0.5, offset=0.25)")
axs[4].plot(range(0, 5), [0]*5, color='white')
axs[4].xaxis.set_major_locator(ticker.IndexLocator(base=0.5, offset=0.25))

# Auto Locator
setup(axs[5], title="AutoLocator()")
axs[5].xaxis.set_major_locator(ticker.AutoLocator())
axs[5].xaxis.set_minor_locator(ticker.AutoMinorLocator())

# MaxN Locator
setup(axs[6], title="MaxNLocator(n=4)")
axs[6].xaxis.set_major_locator(ticker.MaxNLocator(4))
axs[6].xaxis.set_minor_locator(ticker.MaxNLocator(40))

# Log Locator
setup(axs[7], title="LogLocator(base=10, numticks=15)")
axs[7].set_xlim(10**3, 10**10)
axs[7].set_xscale('log')
axs[7].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

plt.tight_layout()
plt.show()

# %%
#
# .. admonition:: References
#
#    The following functions, methods, classes and modules are used in this example:
#
#    - `matplotlib.axis.Axis.set_major_locator`
#    - `matplotlib.axis.Axis.set_minor_locator`
#    - `matplotlib.ticker.NullLocator`
#    - `matplotlib.ticker.MultipleLocator`
#    - `matplotlib.ticker.FixedLocator`
#    - `matplotlib.ticker.LinearLocator`
#    - `matplotlib.ticker.IndexLocator`
#    - `matplotlib.ticker.AutoLocator`
#    - `matplotlib.ticker.MaxNLocator`
#    - `matplotlib.ticker.LogLocator`
```
### 2 - galleries/examples/ticks/tick-locators.py:

Start line: 1, End line: 29

```python
"""
=============
Tick locators
=============

Tick locators define the position of the ticks.

This example illustrates the usage and effect of the most common locators.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker


def setup(ax, title):
    """Set up common parameters for the Axes in the example."""
    # only show the bottom spine
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines[['left', 'right', 'top']].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.2, title, transform=ax.transAxes,
            fontsize=14, fontname='Monospace', color='tab:blue')
```
### 3 - galleries/examples/ticks/date_formatters_locators.py:

Start line: 1, End line: 39

```python
"""
=================================
Date tick locators and formatters
=================================

This example illustrates the usage and effect of the various date locators and
formatters.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import (FR, MO, MONTHLY, SA, SU, TH, TU, WE,
                              AutoDateFormatter, AutoDateLocator,
                              ConciseDateFormatter, DateFormatter, DayLocator,
                              HourLocator, MicrosecondLocator, MinuteLocator,
                              MonthLocator, RRuleLocator, SecondLocator,
                              WeekdayLocator, YearLocator, rrulewrapper)
import matplotlib.ticker as ticker

locators = [
    ('AutoDateLocator(maxticks=8)', '2003-02-01', '%Y-%m'),
    ('YearLocator(month=4)', '2003-02-01', '%Y-%m'),
    ('MonthLocator(bymonth=[4,8,12])', '2003-02-01', '%Y-%m'),
    ('DayLocator(interval=180)', '2003-02-01', '%Y-%m-%d'),
    ('WeekdayLocator(byweekday=SU, interval=4)', '2000-07-01', '%a %Y-%m-%d'),
    ('HourLocator(byhour=range(0,24,6))', '2000-02-04', '%H h'),
    ('MinuteLocator(interval=15)', '2000-02-01 02:00', '%H:%M'),
    ('SecondLocator(bysecond=(0,30))', '2000-02-01 00:02', '%H:%M:%S'),
    ('MicrosecondLocator(interval=1000)', '2000-02-01 00:00:00.005', '%S.%f'),
    ('RRuleLocator(rrulewrapper(freq=MONTHLY, \nbyweekday=(MO, TU, WE, TH,' +
     ' FR), bysetpos=-1))', '2000-07-01', '%Y-%m-%d')
]

formatters = [
    ('AutoDateFormatter(ax.xaxis.get_major_locator())'),
    ('ConciseDateFormatter(ax.xaxis.get_major_locator())'),
    ('DateFormatter("%b %Y")')
]
```
### 4 - lib/matplotlib/dates.py:

Start line: 1117, End line: 1172

```python
class DateLocator(ticker.Locator):
    """
    Determines the tick locations when plotting dates.

    This class is subclassed by other Locators and
    is not meant to be used on its own.
    """
    hms0d = {'byhour': 0, 'byminute': 0, 'bysecond': 0}

    def __init__(self, tz=None):
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        self.tz = _get_tzinfo(tz)

    def set_tzinfo(self, tz):
        """
        Set timezone info.

        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        self.tz = _get_tzinfo(tz)

    def datalim_to_dt(self):
        """Convert axis data interval to datetime objects."""
        dmin, dmax = self.axis.get_data_interval()
        if dmin > dmax:
            dmin, dmax = dmax, dmin

        return num2date(dmin, self.tz), num2date(dmax, self.tz)

    def viewlim_to_dt(self):
        """Convert the view interval to datetime objects."""
        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return num2date(vmin, self.tz), num2date(vmax, self.tz)

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        return 1

    def _get_interval(self):
        """
        Return the number of units for each tick.
        """
        return 1
```
### 5 - galleries/examples/ticks/date_formatters_locators.py:

Start line: 60, End line: 94

```python
fig, ax = plt.subplots(len(locators), 1, figsize=(8, len(locators) * .8),
                       layout='constrained')
fig.suptitle('Date Locators')
for i, loc in enumerate(locators):
    plot_axis(ax[i], *loc)

fig, ax = plt.subplots(len(formatters), 1, figsize=(8, len(formatters) * .8),
                       layout='constrained')
fig.suptitle('Date Formatters')
for i, fmt in enumerate(formatters):
    plot_axis(ax[i], formatter=fmt)


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.dates.AutoDateLocator`
#    - `matplotlib.dates.YearLocator`
#    - `matplotlib.dates.MonthLocator`
#    - `matplotlib.dates.DayLocator`
#    - `matplotlib.dates.WeekdayLocator`
#    - `matplotlib.dates.HourLocator`
#    - `matplotlib.dates.MinuteLocator`
#    - `matplotlib.dates.SecondLocator`
#    - `matplotlib.dates.MicrosecondLocator`
#    - `matplotlib.dates.RRuleLocator`
#    - `matplotlib.dates.rrulewrapper`
#    - `matplotlib.dates.DateFormatter`
#    - `matplotlib.dates.AutoDateFormatter`
#    - `matplotlib.dates.ConciseDateFormatter`
```
### 6 - galleries/examples/ticks/ticks_too_many.py:

Start line: 1, End line: 77

```python
"""
=====================
Fixing too many ticks
=====================

One common cause for unexpected tick behavior is passing a list of strings
instead of numbers or datetime objects. This can easily happen without notice
when reading in a comma-delimited text file. Matplotlib treats lists of strings
as *categorical* variables
(:doc:`/gallery/lines_bars_and_markers/categorical_variables`), and by default
puts one tick per category, and plots them in the order in which they are
supplied.  If this is not desired, the solution is to convert the strings to
a numeric type as in the following examples.

"""

# %%
# Example 1: Strings can lead to an unexpected order of number ticks
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(6, 2.5))
x = ['1', '5', '2', '3']
y = [1, 4, 2, 3]
ax[0].plot(x, y, 'd')
ax[0].tick_params(axis='x', color='r', labelcolor='r')
ax[0].set_xlabel('Categories')
ax[0].set_title('Ticks seem out of order / misplaced')

# convert to numbers:
x = np.asarray(x, dtype='float')
ax[1].plot(x, y, 'd')
ax[1].set_xlabel('Floats')
ax[1].set_title('Ticks as expected')

# %%
# Example 2: Strings can lead to very many ticks
# ----------------------------------------------
# If *x* has 100 elements, all strings, then we would have 100 (unreadable)
# ticks, and again the solution is to convert the strings to floats:

fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
x = [f'{xx}' for xx in np.arange(100)]
y = np.arange(100)
ax[0].plot(x, y)
ax[0].tick_params(axis='x', color='r', labelcolor='r')
ax[0].set_title('Too many ticks')
ax[0].set_xlabel('Categories')

ax[1].plot(np.asarray(x, float), y)
ax[1].set_title('x converted to numbers')
ax[1].set_xlabel('Floats')

# %%
# Example 3: Strings can lead to an unexpected order of datetime ticks
# --------------------------------------------------------------------
# A common case is when dates are read from a CSV file, they need to be
# converted from strings to datetime objects to get the proper date locators
# and formatters.

fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(6, 2.75))
x = ['2021-10-01', '2021-11-02', '2021-12-03', '2021-09-01']
y = [0, 2, 3, 1]
ax[0].plot(x, y, 'd')
ax[0].tick_params(axis='x', labelrotation=90, color='r', labelcolor='r')
ax[0].set_title('Dates out of order')

# convert to datetime64
x = np.asarray(x, dtype='datetime64[s]')
ax[1].plot(x, y, 'd')
ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_title('x converted to datetimes')

plt.show()
```
### 7 - galleries/examples/ticks/date_concise_formatter.py:

Start line: 114, End line: 185

```python
for nn, ax in enumerate(axs):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%y',  # ticks are mostly years
                         '%b',       # ticks are mostly months
                         '%d',       # ticks are mostly days
                         '%H:%M',    # hrs
                         '%H:%M',    # min
                         '%S.%f', ]  # secs
    # these are mostly just the level above...
    formatter.zero_formats = [''] + formatter.formats[:-1]
    # ...except for ticks that are mostly hours, then it is nice to have
    # month-day:
    formatter.zero_formats[3] = '%d-%b'

    formatter.offset_formats = ['',
                                '%Y',
                                '%b %Y',
                                '%d %b %Y',
                                '%d %b %Y',
                                '%d %b %Y %H:%M', ]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.plot(dates, y)
    ax.set_xlim(lims[nn])
axs[0].set_title('Concise Date Formatter')

plt.show()

# %%
# Registering a converter with localization
# =========================================
#
# `.ConciseDateFormatter` doesn't have rcParams entries, but localization can
# be accomplished by passing keyword arguments to `.ConciseDateConverter` and
# registering the datatypes you will use with the units registry:

import datetime

formats = ['%y',          # ticks are mostly years
           '%b',     # ticks are mostly months
           '%d',     # ticks are mostly days
           '%H:%M',  # hrs
           '%H:%M',  # min
           '%S.%f', ]  # secs
# these can be the same, except offset by one level....
zero_formats = [''] + formats[:-1]
# ...except for ticks that are mostly hours, then it's nice to have month-day
zero_formats[3] = '%d-%b'
offset_formats = ['',
                  '%Y',
                  '%b %Y',
                  '%d %b %Y',
                  '%d %b %Y',
                  '%d %b %Y %H:%M', ]

converter = mdates.ConciseDateConverter(
    formats=formats, zero_formats=zero_formats, offset_formats=offset_formats)

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(6, 6))
for nn, ax in enumerate(axs):
    ax.plot(dates, y)
    ax.set_xlim(lims[nn])
axs[0].set_title('Concise Date Formatter registered non-default')

plt.show()
```
### 8 - galleries/users_explain/text/text_intro.py:

Start line: 269, End line: 336

```python
axs[1].xaxis.set_ticks(np.arange(0., 8.1, 2.))
plt.show()

# %%
# We can of course fix this after the fact, but it does highlight a
# weakness of hard-coding the ticks.  This example also changes the format
# of the ticks:

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x1, y1)
axs[1].plot(x1, y1)
ticks = np.arange(0., 8.1, 2.)
# list comprehension to get all tick labels...
tickla = [f'{tick:1.2f}' for tick in ticks]
axs[1].xaxis.set_ticks(ticks)
axs[1].xaxis.set_ticklabels(tickla)
axs[1].set_xlim(axs[0].get_xlim())
plt.show()

# %%
# Tick Locators and Formatters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Instead of making a list of all the ticklabels, we could have
# used `matplotlib.ticker.StrMethodFormatter` (new-style ``str.format()``
# format string) or `matplotlib.ticker.FormatStrFormatter` (old-style '%'
# format string) and passed it to the ``ax.xaxis``.  A
# `matplotlib.ticker.StrMethodFormatter` can also be created by passing a
# ``str`` without having to explicitly create the formatter.

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x1, y1)
axs[1].plot(x1, y1)
ticks = np.arange(0., 8.1, 2.)
axs[1].xaxis.set_ticks(ticks)
axs[1].xaxis.set_major_formatter('{x:1.1f}')
axs[1].set_xlim(axs[0].get_xlim())
plt.show()

# %%
# And of course we could have used a non-default locator to set the
# tick locations.  Note we still pass in the tick values, but the
# x-limit fix used above is *not* needed.

fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
axs[0].plot(x1, y1)
axs[1].plot(x1, y1)
locator = matplotlib.ticker.FixedLocator(ticks)
axs[1].xaxis.set_major_locator(locator)
axs[1].xaxis.set_major_formatter('±{x}°')
plt.show()

# %%
# The default formatter is the `matplotlib.ticker.MaxNLocator` called as
# ``ticker.MaxNLocator(self, nbins='auto', steps=[1, 2, 2.5, 5, 10])``
# The *steps* keyword contains a list of multiples that can be used for
# tick values.  i.e. in this case, 2, 4, 6 would be acceptable ticks,
# as would 20, 40, 60 or 0.2, 0.4, 0.6. However, 3, 6, 9 would not be
# acceptable because 3 doesn't appear in the list of steps.
#
# ``nbins=auto`` uses an algorithm to determine how many ticks will
# be acceptable based on how long the axis is.  The fontsize of the
# ticklabel is taken into account, but the length of the tick string
# is not (because it's not yet known.)  In the bottom row, the
# ticklabels are quite large, so we set ``nbins=4`` to make the
# labels fit in the right-hand plot.

fig, axs = plt.subplots(2, 2, figsize=(8, 5), tight_layout=True)
```
### 9 - lib/matplotlib/mlab.py:

Start line: 403, End line: 458

```python
_docstring.interpd.update(
    Spectral="""\
Fs : float, default: 2
    The sampling frequency (samples per time unit).  It is used to calculate
    the Fourier frequencies, *freqs*, in cycles per time unit.

window : callable or ndarray, default: `.window_hanning`
    A function or a vector of length *NFFT*.  To create window vectors see
    `.window_hanning`, `.window_none`, `numpy.blackman`, `numpy.hamming`,
    `numpy.bartlett`, `scipy.signal`, `scipy.signal.get_window`, etc.  If a
    function is passed as the argument, it must take a data segment as an
    argument and return the windowed version of the segment.

sides : {'default', 'onesided', 'twosided'}, optional
    Which sides of the spectrum to return. 'default' is one-sided for real
    data and two-sided for complex data. 'onesided' forces the return of a
    one-sided spectrum, while 'twosided' forces two-sided.""",

    Single_Spectrum="""\
pad_to : int, optional
    The number of points to which the data segment is padded when performing
    the FFT.  While not increasing the actual resolution of the spectrum (the
    minimum distance between resolvable peaks), this can give more points in
    the plot, allowing for more detail. This corresponds to the *n* parameter
    in the call to `~numpy.fft.fft`.  The default is None, which sets *pad_to*
    equal to the length of the input signal (i.e. no padding).""",

    PSD="""\
pad_to : int, optional
    The number of points to which the data segment is padded when performing
    the FFT.  This can be different from *NFFT*, which specifies the number
    of data points used.  While not increasing the actual resolution of the
    spectrum (the minimum distance between resolvable peaks), this can give
    more points in the plot, allowing for more detail. This corresponds to
    the *n* parameter in the call to `~numpy.fft.fft`. The default is None,
    which sets *pad_to* equal to *NFFT*

NFFT : int, default: 256
    The number of data points used in each block for the FFT.  A power 2 is
    most efficient.  This should *NOT* be used to get zero padding, or the
    scaling of the result will be incorrect; use *pad_to* for this instead.

detrend : {'none', 'mean', 'linear'} or callable, default: 'none'
    The function applied to each segment before fft-ing, designed to remove
    the mean or linear trend.  Unlike in MATLAB, where the *detrend* parameter
    is a vector, in Matplotlib it is a function.  The :mod:`~matplotlib.mlab`
    module defines `.detrend_none`, `.detrend_mean`, and `.detrend_linear`,
    but you can use a custom function as well.  You can also use a string to
    choose one of the functions: 'none' calls `.detrend_none`. 'mean' calls
    `.detrend_mean`. 'linear' calls `.detrend_linear`.

scale_by_freq : bool, default: True
    Whether the resulting density values should be scaled by the scaling
    frequency, which gives density in units of 1/Hz.  This allows for
    integration over the returned frequency values.  The default is True for
    MATLAB compatibility.""")
```
### 10 - galleries/examples/ticks/major_minor_demo.py:

Start line: 1, End line: 97

```python
r"""
=====================
Major and minor ticks
=====================

Demonstrate how to use major and minor tickers.

The two relevant classes are `.Locator`\s and `.Formatter`\s.  Locators
determine where the ticks are, and formatters control the formatting of tick
labels.

Minor ticks are off by default (using `.NullLocator` and `.NullFormatter`).
Minor ticks can be turned on without labels by setting the minor locator.
Minor tick labels can be turned on by setting the minor formatter.

`.MultipleLocator` places ticks on multiples of some base.
`.StrMethodFormatter` uses a format string (e.g., ``'{x:d}'`` or ``'{x:1.2f}'``
or ``'{x:1.1f} cm'``) to format the tick labels (the variable in the format
string must be ``'x'``).  For a `.StrMethodFormatter`, the string can be passed
directly to `.Axis.set_major_formatter` or
`.Axis.set_minor_formatter`.  An appropriate `.StrMethodFormatter` will
be created and used automatically.

`.pyplot.grid` changes the grid settings of the major ticks of the x- and
y-axis together.  If you want to control the grid of the minor ticks for a
given axis, use for example ::

  ax.xaxis.grid(True, which='minor')

Note that a given locator or formatter instance can only be used on a single
axis (because the locator stores references to the axis data and view limits).
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import AutoMinorLocator, MultipleLocator

t = np.arange(0.0, 100.0, 0.1)
s = np.sin(0.1 * np.pi * t) * np.exp(-t * 0.01)

fig, ax = plt.subplots()
ax.plot(t, s)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_major_formatter('{x:.0f}')

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(5))

plt.show()

# %%
# Automatic tick selection for major and minor ticks.
#
# Use interactive pan and zoom to see how the tick intervals change. There will
# be either 4 or 5 minor tick intervals per major interval, depending on the
# major interval.
#
# One can supply an argument to `.AutoMinorLocator` to specify a fixed number
# of minor intervals per major interval, e.g. ``AutoMinorLocator(2)`` would
# lead to a single minor tick between major ticks.

t = np.arange(0.0, 100.0, 0.01)
s = np.sin(2 * np.pi * t) * np.exp(-t * 0.01)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.xaxis.set_minor_locator(AutoMinorLocator())

ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='r')

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.axis.Axis.set_major_formatter`
#    - `matplotlib.axis.Axis.set_major_locator`
#    - `matplotlib.axis.Axis.set_minor_locator`
#    - `matplotlib.ticker.AutoMinorLocator`
#    - `matplotlib.ticker.MultipleLocator`
#    - `matplotlib.ticker.StrMethodFormatter`
```
