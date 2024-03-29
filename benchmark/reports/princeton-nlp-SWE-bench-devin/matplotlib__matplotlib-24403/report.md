# matplotlib__matplotlib-24403

| **matplotlib/matplotlib** | `8d8ae7fe5b129af0fef45aefa0b3e11394fcbe51` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1425 |
| **Any found context length** | 1425 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -4416,7 +4416,7 @@ def invalid_shape_exception(csize, xsize):
                     # severe failure => one may appreciate a verbose feedback.
                     raise ValueError(
                         f"'c' argument must be a color, a sequence of colors, "
-                        f"or a sequence of numbers, not {c}") from err
+                        f"or a sequence of numbers, not {c!r}") from err
             else:
                 if len(colors) not in (0, 1, xsize):
                     # NB: remember that a single color is also acceptable.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/axes/_axes.py | 4419 | 4419 | 1 | 1 | 1425


## Problem Statement

```
[ENH]: Use `repr` instead of `str` in the error message
### Problem

I mistakenly supplied `"blue\n"` as the argument `c` for [`matplotlib.axes.Axes.scatter
`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib-axes-axes-scatter), then `matplitlib` claimed for illegal color name like this:

\`\`\`
ValueError: 'c' argument must be a color, a sequence of colors, or a sequence of numbers, not blue
\`\`\`

I was not aware that the argument actually contained a trailing newline so I was very confused. 

### Proposed solution

The error message would be nicer if it outputs user's input via `repr`.
For example, in this case the error message [here](https://github.com/matplotlib/matplotlib/blob/v3.5.1/lib/matplotlib/axes/_axes.py#L4230-L4232) can be easily replced with:

\`\`\`python
                    raise ValueError(
                        f"'c' argument must be a color, a sequence of colors, "
                        f"or a sequence of numbers, not {c!r}") from 
\`\`\`

so that we may now get an easy-to-troubleshoot error like this:

\`\`\`
ValueError: 'c' argument must be a color, a sequence of colors, or a sequence of numbers, not "blue\n"
\`\`\`

This kind of improvement can be applied to many other places.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 lib/matplotlib/axes/_axes.py** | 4279 | 4427| 1425 | 1425 | 74386 | 
| 2 | 2 lib/matplotlib/pyplot.py | 2492 | 2504| 229 | 1654 | 102466 | 
| 3 | 3 lib/matplotlib/backends/backend_wx.py | 10 | 60| 343 | 1997 | 114508 | 
| 4 | 4 lib/matplotlib/rcsetup.py | 289 | 307| 186 | 2183 | 126449 | 
| 5 | 5 lib/matplotlib/_api/deprecation.py | 24 | 47| 238 | 2421 | 130766 | 
| 6 | 5 lib/matplotlib/rcsetup.py | 310 | 332| 156 | 2577 | 130766 | 
| 7 | 5 lib/matplotlib/pyplot.py | 1 | 89| 669 | 3246 | 130766 | 
| 8 | 6 lib/matplotlib/axes/_base.py | 121 | 208| 677 | 3923 | 170128 | 
| 9 | 7 examples/ticks/ticks_too_many.py | 1 | 77| 742 | 4665 | 170870 | 
| 10 | 7 lib/matplotlib/pyplot.py | 2060 | 2073| 138 | 4803 | 170870 | 
| 11 | 7 lib/matplotlib/_api/deprecation.py | 50 | 96| 425 | 5228 | 170870 | 
| 13 | 8 lib/matplotlib/pyplot.py | 2956 | 2971| 184 | 6275 | 199313 | 


### Hint

```
Labeling this as a good first issue as this is a straight forward change that should not bring up any API design issues (yes technically changing the wording of an error message may break someone, but that is so brittle I am not going to worry about that).  Will need a test (the new line example is a good one!).
@e5f6bd If you are interested in opening a PR with this change, please have at it!
@e5f6bd Thanks, but I'm a bit busy recently, so I'll leave this chance to someone else.
If this issue has not been resolved yet and nobody working on it  I would like to try it
Looks like #21968 is addressing this issue 
yeah but the issue still open 
Working on this now in axes_.py,
future plans to work on this in _all_ files with error raising.
Also converting f strings to template strings where I see them in raised errors
> Also converting f strings to template strings where I see them in raised errors

f-string are usually preferred due to speed issues. Which is not critical, but if one can choose, f-strings are not worse.
```

## Patch

```diff
diff --git a/lib/matplotlib/axes/_axes.py b/lib/matplotlib/axes/_axes.py
--- a/lib/matplotlib/axes/_axes.py
+++ b/lib/matplotlib/axes/_axes.py
@@ -4416,7 +4416,7 @@ def invalid_shape_exception(csize, xsize):
                     # severe failure => one may appreciate a verbose feedback.
                     raise ValueError(
                         f"'c' argument must be a color, a sequence of colors, "
-                        f"or a sequence of numbers, not {c}") from err
+                        f"or a sequence of numbers, not {c!r}") from err
             else:
                 if len(colors) not in (0, 1, xsize):
                     # NB: remember that a single color is also acceptable.

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_axes.py b/lib/matplotlib/tests/test_axes.py
--- a/lib/matplotlib/tests/test_axes.py
+++ b/lib/matplotlib/tests/test_axes.py
@@ -8290,3 +8290,17 @@ def test_extent_units():
     with pytest.raises(ValueError,
                        match="set_extent did not consume all of the kwargs"):
         im.set_extent([2, 12, date_first, date_last], clip=False)
+
+
+def test_scatter_color_repr_error():
+
+    def get_next_color():
+        return 'blue'  # pragma: no cover
+    msg = (
+            r"'c' argument must be a color, a sequence of colors"
+            r", or a sequence of numbers, not 'red\\n'"
+        )
+    with pytest.raises(ValueError, match=msg):
+        c = 'red\n'
+        mpl.axes.Axes._parse_scatter_color_args(
+            c, None, kwargs={}, xsize=2, get_next_color_func=get_next_color)

```


## Code snippets

### 1 - lib/matplotlib/axes/_axes.py:

Start line: 4279, End line: 4427

```python
@_docstring.interpd
class Axes(_AxesBase):

    @staticmethod
    def _parse_scatter_color_args(c, edgecolors, kwargs, xsize,
                                  get_next_color_func):
        """
        Helper function to process color related arguments of `.Axes.scatter`.

        Argument precedence for facecolors:

        - c (if not None)
        - kwargs['facecolor']
        - kwargs['facecolors']
        - kwargs['color'] (==kwcolor)
        - 'b' if in classic mode else the result of ``get_next_color_func()``

        Argument precedence for edgecolors:

        - kwargs['edgecolor']
        - edgecolors (is an explicit kw argument in scatter())
        - kwargs['color'] (==kwcolor)
        - 'face' if not in classic mode else None

        Parameters
        ----------
        c : color or sequence or sequence of color or None
            See argument description of `.Axes.scatter`.
        edgecolors : color or sequence of color or {'face', 'none'} or None
            See argument description of `.Axes.scatter`.
        kwargs : dict
            Additional kwargs. If these keys exist, we pop and process them:
            'facecolors', 'facecolor', 'edgecolor', 'color'
            Note: The dict is modified by this function.
        xsize : int
            The size of the x and y arrays passed to `.Axes.scatter`.
        get_next_color_func : callable
            A callable that returns a color. This color is used as facecolor
            if no other color is provided.

            Note, that this is a function rather than a fixed color value to
            support conditional evaluation of the next color.  As of the
            current implementation obtaining the next color from the
            property cycle advances the cycle. This must only happen if we
            actually use the color, which will only be decided within this
            method.

        Returns
        -------
        c
            The input *c* if it was not *None*, else a color derived from the
            other inputs or defaults.
        colors : array(N, 4) or None
            The facecolors as RGBA values, or *None* if a colormap is used.
        edgecolors
            The edgecolor.

        """
        facecolors = kwargs.pop('facecolors', None)
        facecolors = kwargs.pop('facecolor', facecolors)
        edgecolors = kwargs.pop('edgecolor', edgecolors)

        kwcolor = kwargs.pop('color', None)

        if kwcolor is not None and c is not None:
            raise ValueError("Supply a 'c' argument or a 'color'"
                             " kwarg but not both; they differ but"
                             " their functionalities overlap.")

        if kwcolor is not None:
            try:
                mcolors.to_rgba_array(kwcolor)
            except ValueError as err:
                raise ValueError(
                    "'color' kwarg must be a color or sequence of color "
                    "specs.  For a sequence of values to be color-mapped, use "
                    "the 'c' argument instead.") from err
            if edgecolors is None:
                edgecolors = kwcolor
            if facecolors is None:
                facecolors = kwcolor

        if edgecolors is None and not mpl.rcParams['_internal.classic_mode']:
            edgecolors = mpl.rcParams['scatter.edgecolors']

        c_was_none = c is None
        if c is None:
            c = (facecolors if facecolors is not None
                 else "b" if mpl.rcParams['_internal.classic_mode']
                 else get_next_color_func())
        c_is_string_or_strings = (
            isinstance(c, str)
            or (np.iterable(c) and len(c) > 0
                and isinstance(cbook._safe_first_finite(c), str)))

        def invalid_shape_exception(csize, xsize):
            return ValueError(
                f"'c' argument has {csize} elements, which is inconsistent "
                f"with 'x' and 'y' with size {xsize}.")

        c_is_mapped = False  # Unless proven otherwise below.
        valid_shape = True  # Unless proven otherwise below.
        if not c_was_none and kwcolor is None and not c_is_string_or_strings:
            try:  # First, does 'c' look suitable for value-mapping?
                c = np.asanyarray(c, dtype=float)
            except ValueError:
                pass  # Failed to convert to float array; must be color specs.
            else:
                # handle the documented special case of a 2D array with 1
                # row which as RGB(A) to broadcast.
                if c.shape == (1, 4) or c.shape == (1, 3):
                    c_is_mapped = False
                    if c.size != xsize:
                        valid_shape = False
                # If c can be either mapped values or a RGB(A) color, prefer
                # the former if shapes match, the latter otherwise.
                elif c.size == xsize:
                    c = c.ravel()
                    c_is_mapped = True
                else:  # Wrong size; it must not be intended for mapping.
                    if c.shape in ((3,), (4,)):
                        _log.warning(
                            "*c* argument looks like a single numeric RGB or "
                            "RGBA sequence, which should be avoided as value-"
                            "mapping will have precedence in case its length "
                            "matches with *x* & *y*.  Please use the *color* "
                            "keyword-argument or provide a 2D array "
                            "with a single row if you intend to specify "
                            "the same RGB or RGBA value for all points.")
                    valid_shape = False
        if not c_is_mapped:
            try:  # Is 'c' acceptable as PathCollection facecolors?
                colors = mcolors.to_rgba_array(c)
            except (TypeError, ValueError) as err:
                if "RGBA values should be within 0-1 range" in str(err):
                    raise
                else:
                    if not valid_shape:
                        raise invalid_shape_exception(c.size, xsize) from err
                    # Both the mapping *and* the RGBA conversion failed: pretty
                    # severe failure => one may appreciate a verbose feedback.
                    raise ValueError(
                        f"'c' argument must be a color, a sequence of colors, "
                        f"or a sequence of numbers, not {c}") from err
            else:
                if len(colors) not in (0, 1, xsize):
                    # NB: remember that a single color is also acceptable.
                    # Besides *colors* will be an empty array if c == 'none'.
                    raise invalid_shape_exception(len(colors), xsize)
        else:
            colors = None  # use cmap, norm after collection is created
        return c, colors, edgecolors
```
### 2 - lib/matplotlib/pyplot.py:

Start line: 2492, End line: 2504

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.errorbar)
def errorbar(
        x, y, yerr=None, xerr=None, fmt='', ecolor=None,
        elinewidth=None, capsize=None, barsabove=False, lolims=False,
        uplims=False, xlolims=False, xuplims=False, errorevery=1,
        capthick=None, *, data=None, **kwargs):
    return gca().errorbar(
        x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor,
        elinewidth=elinewidth, capsize=capsize, barsabove=barsabove,
        lolims=lolims, uplims=uplims, xlolims=xlolims,
        xuplims=xuplims, errorevery=errorevery, capthick=capthick,
        **({"data": data} if data is not None else {}), **kwargs)
```
### 3 - lib/matplotlib/backends/backend_wx.py:

Start line: 10, End line: 60

```python
import functools
import logging
import math
import pathlib
import sys
import weakref

import numpy as np
import PIL

import matplotlib as mpl
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase,
    GraphicsContextBase, MouseButton, NavigationToolbar2, RendererBase,
    TimerBase, ToolContainerBase, cursors,
    CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent)

from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

import wx

_log = logging.getLogger(__name__)

# the True dots per inch on the screen; should be display dependent; see
# http://groups.google.com/d/msg/comp.lang.postscript/-/omHAc9FEuAsJ?hl=en
# for some info about screen dpi
PIXELS_PER_INCH = 75


@_api.deprecated("3.6")
def error_msg_wx(msg, parent=None):
    """Signal an error condition with a popup error dialog."""
    dialog = wx.MessageDialog(parent=parent,
                              message=msg,
                              caption='Matplotlib backend_wx error',
                              style=wx.OK | wx.CENTRE)
    dialog.ShowModal()
    dialog.Destroy()
    return None


# lru_cache holds a reference to the App and prevents it from being gc'ed.
@functools.lru_cache(1)
def _create_wxapp():
    wxapp = wx.App(False)
    wxapp.SetExitOnFrameDelete(True)
    cbook._setup_new_guiapp()
    return wxapp
```
### 4 - lib/matplotlib/rcsetup.py:

Start line: 289, End line: 307

```python
def _validate_color_or_linecolor(s):
    if cbook._str_equal(s, 'linecolor'):
        return s
    elif cbook._str_equal(s, 'mfc') or cbook._str_equal(s, 'markerfacecolor'):
        return 'markerfacecolor'
    elif cbook._str_equal(s, 'mec') or cbook._str_equal(s, 'markeredgecolor'):
        return 'markeredgecolor'
    elif s is None:
        return None
    elif isinstance(s, str) and len(s) == 6 or len(s) == 8:
        stmp = '#' + s
        if is_color_like(stmp):
            return stmp
        if s.lower() == 'none':
            return None
    elif is_color_like(s):
        return s

    raise ValueError(f'{s!r} does not look like a color arg')
```
### 5 - lib/matplotlib/_api/deprecation.py:

Start line: 24, End line: 47

```python
def _generate_deprecation_warning(
        since, message='', name='', alternative='', pending=False, obj_type='',
        addendum='', *, removal=''):
    if pending:
        if removal:
            raise ValueError(
                "A pending deprecation cannot have a scheduled removal")
    else:
        removal = f"in {removal}" if removal else "two minor releases later"
    if not message:
        message = (
            ("The %(name)s %(obj_type)s" if obj_type else "%(name)s")
            + (" will be deprecated in a future version"
               if pending else
               (" was deprecated in Matplotlib %(since)s"
                + (" and will be removed %(removal)s" if removal else "")))
            + "."
            + (" Use %(alternative)s instead." if alternative else "")
            + (" %(addendum)s" if addendum else ""))
    warning_cls = (PendingDeprecationWarning if pending
                   else MatplotlibDeprecationWarning)
    return warning_cls(message % dict(
        func=name, name=name, obj_type=obj_type, since=since, removal=removal,
        alternative=alternative, addendum=addendum))
```
### 6 - lib/matplotlib/rcsetup.py:

Start line: 310, End line: 332

```python
def validate_color(s):
    """Return a valid color arg."""
    if isinstance(s, str):
        if s.lower() == 'none':
            return 'none'
        if len(s) == 6 or len(s) == 8:
            stmp = '#' + s
            if is_color_like(stmp):
                return stmp

    if is_color_like(s):
        return s

    # If it is still valid, it must be a tuple (as a string from matplotlibrc).
    try:
        color = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        pass
    else:
        if is_color_like(color):
            return color

    raise ValueError(f'{s!r} does not look like a color arg')
```
### 7 - lib/matplotlib/pyplot.py:

Start line: 1, End line: 89

```python
# Note: The first part of this file can be modified in place, but the latter
# part is autogenerated by the boilerplate.py script.

"""
`matplotlib.pyplot` is a state-based interface to matplotlib. It provides
an implicit,  MATLAB-like, way of plotting.  It also opens figures on your
screen, and acts as the figure GUI manager.

pyplot is mainly intended for interactive plots and simple cases of
programmatic plot generation::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

The explicit object-oriented API is recommended for complex plots, though
pyplot is still usually used to create the figure and often the axes in the
figure. See `.pyplot.figure`, `.pyplot.subplots`, and
`.pyplot.subplot_mosaic` to create figures, and
:doc:`Axes API </api/axes_api>` for the plotting methods on an Axes::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)


See :ref:`api_interfaces` for an explanation of the tradeoffs between the
implicit and explicit interfaces.
"""

from contextlib import ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
from numbers import Number
import re
import sys
import threading
import time

from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import rcsetup, style
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import FigureCanvasBase, MouseButton
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcParams, rcParamsDefault, get_backend, rcParamsOrig
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names

from matplotlib import cm
from matplotlib.cm import _colormaps as colormaps, register_cmap
from matplotlib.colors import _color_sequences as color_sequences

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget

from .ticker import (
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator)

_log = logging.getLogger(__name__)
```
### 8 - lib/matplotlib/axes/_base.py:

Start line: 121, End line: 208

```python
def _process_plot_format(fmt, *, ambiguous_fmt_datakey=False):
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

    errfmt = ("{!r} is neither a data key nor a valid format string ({})"
              if ambiguous_fmt_datakey else
              "{!r} is not a valid format string ({})")

    i = 0
    while i < len(fmt):
        c = fmt[i]
        if fmt[i:i+2] in mlines.lineStyles:  # First, the two-char styles.
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = fmt[i:i+2]
            i += 2
        elif c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = c
            i += 1
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(errfmt.format(fmt, "two marker symbols"))
            marker = c
            i += 1
        elif c in mcolors.get_named_colors_mapping():
            if color is not None:
                raise ValueError(errfmt.format(fmt, "two color symbols"))
            color = c
            i += 1
        elif c == 'C' and i < len(fmt) - 1:
            color_cycle_number = int(fmt[i + 1])
            color = mcolors.to_rgba("C{}".format(color_cycle_number))
            i += 2
        else:
            raise ValueError(
                errfmt.format(fmt, f"unrecognized character {c!r}"))

    if linestyle is None and marker is None:
        linestyle = mpl.rcParams['lines.linestyle']
    if linestyle is None:
        linestyle = 'None'
    if marker is None:
        marker = 'None'

    return linestyle, marker, color
```
### 9 - examples/ticks/ticks_too_many.py:

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

############################################################################
# Example 1: Strings can lead to an unexpected order of number ticks
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.5))
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

############################################################################
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

############################################################################
# Example 3: Strings can lead to an unexpected order of datetime ticks
# --------------------------------------------------------------------
# A common case is when dates are read from a CSV file, they need to be
# converted from strings to datetime objects to get the proper date locators
# and formatters.

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.75))
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
### 10 - lib/matplotlib/pyplot.py:

Start line: 2060, End line: 2073

```python
## Plotting part 1: manually generated functions and wrappers ##


@_copy_docstring_and_deprecators(Figure.colorbar)
def colorbar(mappable=None, cax=None, ax=None, **kwargs):
    if mappable is None:
        mappable = gci()
        if mappable is None:
            raise RuntimeError('No mappable was found to use for colorbar '
                               'creation. First define a mappable such as '
                               'an image (with imshow) or a contour set ('
                               'with contourf).')
    ret = gcf().colorbar(mappable, cax=cax, ax=ax, **kwargs)
    return ret
```
