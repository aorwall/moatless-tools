# matplotlib__matplotlib-25281

| **matplotlib/matplotlib** | `5aee26d0a52c237c5b4fafcb843e392907ab45b3` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 16848 |
| **Any found context length** | 3702 |
| **Avg pos** | 35.0 |
| **Min pos** | 4 |
| **Max pos** | 27 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -23,6 +23,7 @@
 
 import itertools
 import logging
+import numbers
 import time
 
 import numpy as np
@@ -517,6 +518,9 @@ def val_or_rc(val, rc_name):
             if not self.isaxes and loc in [0, 'best']:
                 loc = 'upper right'
 
+        type_err_message = ("loc must be string, coordinate tuple, or"
+                            f" an integer 0-10, not {loc!r}")
+
         # handle outside legends:
         self._outside_loc = None
         if isinstance(loc, str):
@@ -535,6 +539,19 @@ def val_or_rc(val, rc_name):
                     loc = locs[0] + ' ' + locs[1]
             # check that loc is in acceptable strings
             loc = _api.check_getitem(self.codes, loc=loc)
+        elif np.iterable(loc):
+            # coerce iterable into tuple
+            loc = tuple(loc)
+            # validate the tuple represents Real coordinates
+            if len(loc) != 2 or not all(isinstance(e, numbers.Real) for e in loc):
+                raise ValueError(type_err_message)
+        elif isinstance(loc, int):
+            # validate the integer represents a string numeric value
+            if loc < 0 or loc > 10:
+                raise ValueError(type_err_message)
+        else:
+            # all other cases are invalid values of loc
+            raise ValueError(type_err_message)
 
         if self.isaxes and self._outside_loc:
             raise ValueError(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/legend.py | 26 | 26 | 27 | 1 | 16848
| lib/matplotlib/legend.py | 520 | 520 | 4 | 1 | 3702
| lib/matplotlib/legend.py | 538 | 538 | 4 | 1 | 3702


## Problem Statement

```
[Bug]: Validation not performed for `loc` argument to `legend`
### Bug summary

When passing non-str `loc` values to `legend`, validation is not performed. So even for invalid inputs, errors are raised only when we call `show()`

### Code for reproduction

\`\`\`python
>>> import matplotlib.pyplot as plt
>>> import matplotlib as mpl
>>> xs, ys = [1,2,3], [2,3,1]
>>> fig, ax = plt.subplots(3)
>>> ax[0].scatter(xs, ys, label='loc-tuple-arg')
<matplotlib.collections.PathCollection object at 0x0000019D4099ED60>
>>> ax[0].legend(loc=(1.1, .5, 1.1, "abc"))
<matplotlib.legend.Legend object at 0x0000019D4099EF10>
>>> plt.show()
\`\`\`


### Actual outcome

\`\`\`
Exception in Tkinter callback
Traceback (most recent call last):
  File "C:\Users\Me\anaconda3\envs\MPL\lib\tkinter\__init__.py", line 1892, in __call__
    return self.func(*args)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\tkinter\__init__.py", line 814, in callit
    func(*args)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\backends\_backend_tk.py", line 251, in idle_draw
    self.draw()
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\backends\backend_tkagg.py", line 10, in draw
    super().draw()
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\backends\backend_agg.py", line 405, in draw
    self.figure.draw(self.renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 74, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\figure.py", line 3071, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\image.py", line 131, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\axes\_base.py", line 3107, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\image.py", line 131, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\legend.py", line 641, in draw
    bbox = self._legend_box.get_window_extent(renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\offsetbox.py", line 354, in get_window_extent
    px, py = self.get_offset(w, h, xd, yd, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\offsetbox.py", line 291, in get_offset
    return (self._offset(width, height, xdescent, ydescent, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\legend.py", line 617, in _findoffset
    fx, fy = self._loc
ValueError: too many values to unpack (expected 2)
Exception in Tkinter callback
Traceback (most recent call last):
  File "C:\Users\Me\anaconda3\envs\MPL\lib\tkinter\__init__.py", line 1892, in __call__
    return self.func(*args)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\tkinter\__init__.py", line 814, in callit
    func(*args)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\backends\_backend_tk.py", line 251, in idle_draw
    self.draw()
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\backends\backend_tkagg.py", line 10, in draw
    super().draw()
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\backends\backend_agg.py", line 405, in draw
    self.figure.draw(self.renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 74, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\figure.py", line 3071, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\image.py", line 131, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\axes\_base.py", line 3107, in draw
    mimage._draw_list_compositing_images(
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\image.py", line 131, in _draw_list_compositing_images
    a.draw(renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\artist.py", line 51, in draw_wrapper
    return draw(artist, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\legend.py", line 641, in draw
    bbox = self._legend_box.get_window_extent(renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\offsetbox.py", line 354, in get_window_extent
    px, py = self.get_offset(w, h, xd, yd, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\offsetbox.py", line 291, in get_offset
    return (self._offset(width, height, xdescent, ydescent, renderer)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\legend.py", line 617, in _findoffset
    fx, fy = self._loc
ValueError: too many values to unpack (expected 2)
\`\`\`

### Expected outcome

Errors should be raised when invalid arguments are passed to `loc`. Similar to what we get when we pass an invalid string value as shown:
\`\`\`
>>> ax[0].legend(loc="abcd")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\axes\_axes.py", line 307, in legend
    self.legend_ = mlegend.Legend(self, handles, labels, **kwargs)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\_api\deprecation.py", line 454, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\legend.py", line 470, in __init__
    loc = _api.check_getitem(self.codes, loc=loc)
  File "C:\Users\Me\anaconda3\envs\MPL\lib\site-packages\matplotlib\_api\__init__.py", line 190, in check_getitem
    raise ValueError(
ValueError: 'abcd' is not a valid value for loc; supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
\`\`\`

### Additional information

- Do you know why this bug is happening?
https://github.com/matplotlib/matplotlib/blob/ab7917a89dc56165d695fa4b90200f2cacafcd59/lib/matplotlib/legend.py#L608-L615

No validation is done when setting values for `_loc_real`. We do check strings on line 473, which is why we don't face this issue there.

### Operating system

Windows

### Matplotlib Version

3.6.2

### Matplotlib Backend

'TkAgg'

### Python version

3.9.7

### Jupyter version

_No response_

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/legend.py** | 255 | 330| 761 | 761 | 11534 | 
| 2 | 2 galleries/tutorials/intermediate/legend_guide.py | 124 | 200| 789 | 1550 | 14650 | 
| 3 | 3 galleries/examples/ticks/tick-locators.py | 32 | 94| 642 | 2192 | 15503 | 
| **-> 4 <-** | **3 lib/matplotlib/legend.py** | 514 | 622| 1510 | 3702 | 15503 | 
| 5 | 3 galleries/tutorials/intermediate/legend_guide.py | 1 | 122| 892 | 4594 | 15503 | 
| 6 | 4 lib/matplotlib/pyplot.py | 2706 | 2721| 169 | 4763 | 44213 | 
| 7 | **4 lib/matplotlib/legend.py** | 1094 | 1111| 179 | 4942 | 44213 | 
| 8 | 4 galleries/tutorials/intermediate/legend_guide.py | 201 | 277| 791 | 5733 | 44213 | 
| 9 | 5 galleries/examples/ticks/date_formatters_locators.py | 60 | 94| 304 | 6037 | 45193 | 
| 10 | 6 galleries/examples/text_labels_and_annotations/legend_demo.py | 1 | 76| 737 | 6774 | 47099 | 
| 11 | 7 galleries/examples/text_labels_and_annotations/figlegend_demo.py | 1 | 54| 511 | 7285 | 47610 | 
| 12 | 7 galleries/examples/text_labels_and_annotations/legend_demo.py | 77 | 117| 569 | 7854 | 47610 | 
| 13 | 8 lib/mpl_toolkits/axisartist/grid_finder.py | 296 | 315| 170 | 8024 | 50527 | 
| 14 | 9 lib/matplotlib/legend_handler.py | 543 | 625| 762 | 8786 | 57137 | 
| 15 | 10 lib/matplotlib/axis.py | 342 | 397| 600 | 9386 | 78685 | 
| 16 | 11 galleries/tutorials/text/text_intro.py | 264 | 331| 832 | 10218 | 82566 | 
| 17 | 12 galleries/tutorials/text/annotations.py | 480 | 567| 877 | 11095 | 89955 | 
| 18 | 13 galleries/examples/lines_bars_and_markers/scatter_with_legend.py | 97 | 112| 103 | 11198 | 90885 | 
| 19 | 13 galleries/tutorials/text/annotations.py | 393 | 479| 923 | 12121 | 90885 | 
| 20 | **13 lib/matplotlib/legend.py** | 333 | 513| 1569 | 13690 | 90885 | 
| 21 | 13 galleries/tutorials/intermediate/legend_guide.py | 279 | 331| 450 | 14140 | 90885 | 
| 22 | **13 lib/matplotlib/legend.py** | 707 | 736| 238 | 14378 | 90885 | 
| 23 | 14 galleries/tutorials/intermediate/constrainedlayout_guide.py | 179 | 263| 863 | 15241 | 97579 | 
| 24 | 14 galleries/examples/lines_bars_and_markers/scatter_with_legend.py | 1 | 96| 827 | 16068 | 97579 | 
| 25 | **14 lib/matplotlib/legend.py** | 690 | 705| 189 | 16257 | 97579 | 
| 26 | 15 galleries/examples/userdemo/simple_legend01.py | 1 | 26| 198 | 16455 | 97777 | 
| **-> 27 <-** | **15 lib/matplotlib/legend.py** | 1 | 50| 393 | 16848 | 97777 | 
| 28 | 16 galleries/examples/shapes_and_collections/artist_reference.py | 16 | 79| 748 | 17596 | 98617 | 
| 29 | 17 lib/mpl_toolkits/axes_grid1/inset_locator.py | 60 | 77| 184 | 17780 | 103764 | 
| 30 | 18 lib/matplotlib/figure.py | 1114 | 1136| 242 | 18022 | 133303 | 
| 31 | 18 galleries/examples/text_labels_and_annotations/legend_demo.py | 165 | 183| 195 | 18217 | 133303 | 
| 32 | **18 lib/matplotlib/legend.py** | 658 | 688| 229 | 18446 | 133303 | 
| 33 | **18 lib/matplotlib/legend.py** | 623 | 656| 726 | 19172 | 133303 | 
| 34 | 19 lib/mpl_toolkits/axisartist/axis_artist.py | 176 | 200| 198 | 19370 | 141550 | 
| 35 | 20 galleries/examples/ticks/tick-formatters.py | 37 | 137| 970 | 20340 | 142779 | 
| 36 | 21 galleries/examples/text_labels_and_annotations/line_with_text.py | 52 | 88| 260 | 20600 | 143371 | 
| 37 | 21 galleries/examples/ticks/tick-locators.py | 1 | 29| 211 | 20811 | 143371 | 
| 38 | 22 galleries/tutorials/intermediate/tight_layout_guide.py | 105 | 221| 886 | 21697 | 145539 | 
| 39 | 23 galleries/examples/event_handling/legend_picking.py | 1 | 51| 401 | 22098 | 145940 | 
| 40 | 24 galleries/tutorials/toolkits/axisartist.py | 1 | 561| 4700 | 26798 | 150640 | 
| 41 | **24 lib/matplotlib/legend.py** | 1113 | 1157| 411 | 27209 | 150640 | 
| 42 | **24 lib/matplotlib/legend.py** | 907 | 940| 242 | 27451 | 150640 | 
| 43 | 25 galleries/examples/specialty_plots/leftventricle_bullseye.py | 94 | 155| 697 | 28148 | 152245 | 
| 44 | 26 lib/mpl_toolkits/axisartist/floating_axes.py | 53 | 113| 725 | 28873 | 154928 | 
| 45 | 26 lib/matplotlib/legend_handler.py | 740 | 769| 239 | 29112 | 154928 | 
| 46 | 26 galleries/tutorials/intermediate/tight_layout_guide.py | 222 | 293| 499 | 29611 | 154928 | 
| 47 | 27 galleries/examples/ticks/ticks_too_many.py | 1 | 77| 744 | 30355 | 155672 | 
| 48 | 27 galleries/tutorials/text/text_intro.py | 332 | 425| 764 | 31119 | 155672 | 
| 49 | 28 galleries/examples/ticks/major_minor_demo.py | 1 | 97| 828 | 31947 | 156500 | 
| 50 | 28 lib/matplotlib/axis.py | 435 | 458| 220 | 32167 | 156500 | 
| 51 | 29 lib/matplotlib/colorbar.py | 443 | 491| 351 | 32518 | 170811 | 
| 52 | 30 lib/mpl_toolkits/axisartist/grid_helper_curvelinear.py | 172 | 226| 649 | 33167 | 173985 | 
| 53 | 30 lib/matplotlib/axis.py | 58 | 189| 1063 | 34230 | 173985 | 
| 54 | 31 lib/matplotlib/offsetbox.py | 907 | 995| 742 | 34972 | 186311 | 
| 55 | 31 lib/matplotlib/axis.py | 496 | 519| 220 | 35192 | 186311 | 
| 56 | 32 lib/matplotlib/backends/backend_wx.py | 10 | 60| 343 | 35535 | 198426 | 
| 57 | 32 galleries/tutorials/intermediate/constrainedlayout_guide.py | 524 | 643| 1131 | 36666 | 198426 | 
| 58 | 33 galleries/examples/userdemo/simple_legend02.py | 1 | 24| 136 | 36802 | 198562 | 


### Hint

```
The work here is to :

 - sort out what the validation _should be_ (read the code where the above traceback starts)
 - add logic to `Legend.__init__` to validate loc
 - add tests
 - update docstring to legend (in both `Legend` and `Axes.legend`)

This is a good first issue because it should only require understanding a narrow section of the code and no API design (it is already broken for these inputs, we just want it to break _better_).
Hi. can i try this?
@sod-lol Please do!  We do not really assign issues or require you to get permission before you start working on an issue.
@tacaswell hello sir can you give me some resources to work on this issue
@tacaswell i want to work on this,please assign this issue to me
@Gairick52 there is already a PR for this....
@iofall  Hello sir,i want to work on this issue,please assign this to me
> @iofall Hello sir,i want to work on this issue,please assign this to me

Only maintainers can assign people to issues. Also, there is already a PR linked to this issue. You can try finding other issues to work on or provide any inputs if you have to the already linked PR.
@iofall please share me beginner's  developer's guide
> @iofall please share me beginner's developer's guide

Here is the link to the contributing guide - https://matplotlib.org/devdocs/devel/contributing.html
```

## Patch

```diff
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -23,6 +23,7 @@
 
 import itertools
 import logging
+import numbers
 import time
 
 import numpy as np
@@ -517,6 +518,9 @@ def val_or_rc(val, rc_name):
             if not self.isaxes and loc in [0, 'best']:
                 loc = 'upper right'
 
+        type_err_message = ("loc must be string, coordinate tuple, or"
+                            f" an integer 0-10, not {loc!r}")
+
         # handle outside legends:
         self._outside_loc = None
         if isinstance(loc, str):
@@ -535,6 +539,19 @@ def val_or_rc(val, rc_name):
                     loc = locs[0] + ' ' + locs[1]
             # check that loc is in acceptable strings
             loc = _api.check_getitem(self.codes, loc=loc)
+        elif np.iterable(loc):
+            # coerce iterable into tuple
+            loc = tuple(loc)
+            # validate the tuple represents Real coordinates
+            if len(loc) != 2 or not all(isinstance(e, numbers.Real) for e in loc):
+                raise ValueError(type_err_message)
+        elif isinstance(loc, int):
+            # validate the integer represents a string numeric value
+            if loc < 0 or loc > 10:
+                raise ValueError(type_err_message)
+        else:
+            # all other cases are invalid values of loc
+            raise ValueError(type_err_message)
 
         if self.isaxes and self._outside_loc:
             raise ValueError(

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_legend.py b/lib/matplotlib/tests/test_legend.py
--- a/lib/matplotlib/tests/test_legend.py
+++ b/lib/matplotlib/tests/test_legend.py
@@ -1219,3 +1219,79 @@ def test_ncol_ncols(fig_test, fig_ref):
     ncols = 3
     fig_test.legend(strings, ncol=ncols)
     fig_ref.legend(strings, ncols=ncols)
+
+
+def test_loc_invalid_tuple_exception():
+    # check that exception is raised if the loc arg
+    # of legend is not a 2-tuple of numbers
+    fig, ax = plt.subplots()
+    with pytest.raises(ValueError, match=('loc must be string, coordinate '
+                       'tuple, or an integer 0-10, not \\(1.1,\\)')):
+        ax.legend(loc=(1.1, ))
+
+    with pytest.raises(ValueError, match=('loc must be string, coordinate '
+                       'tuple, or an integer 0-10, not \\(0.481, 0.4227, 0.4523\\)')):
+        ax.legend(loc=(0.481, 0.4227, 0.4523))
+
+    with pytest.raises(ValueError, match=('loc must be string, coordinate '
+                       'tuple, or an integer 0-10, not \\(0.481, \'go blue\'\\)')):
+        ax.legend(loc=(0.481, "go blue"))
+
+
+def test_loc_valid_tuple():
+    fig, ax = plt.subplots()
+    ax.legend(loc=(0.481, 0.442))
+    ax.legend(loc=(1, 2))
+
+
+def test_loc_valid_list():
+    fig, ax = plt.subplots()
+    ax.legend(loc=[0.481, 0.442])
+    ax.legend(loc=[1, 2])
+
+
+def test_loc_invalid_list_exception():
+    fig, ax = plt.subplots()
+    with pytest.raises(ValueError, match=('loc must be string, coordinate '
+                       'tuple, or an integer 0-10, not \\[1.1, 2.2, 3.3\\]')):
+        ax.legend(loc=[1.1, 2.2, 3.3])
+
+
+def test_loc_invalid_type():
+    fig, ax = plt.subplots()
+    with pytest.raises(ValueError, match=("loc must be string, coordinate "
+                       "tuple, or an integer 0-10, not {'not': True}")):
+        ax.legend(loc={'not': True})
+
+
+def test_loc_validation_numeric_value():
+    fig, ax = plt.subplots()
+    ax.legend(loc=0)
+    ax.legend(loc=1)
+    ax.legend(loc=5)
+    ax.legend(loc=10)
+    with pytest.raises(ValueError, match=('loc must be string, coordinate '
+                       'tuple, or an integer 0-10, not 11')):
+        ax.legend(loc=11)
+
+    with pytest.raises(ValueError, match=('loc must be string, coordinate '
+                       'tuple, or an integer 0-10, not -1')):
+        ax.legend(loc=-1)
+
+
+def test_loc_validation_string_value():
+    fig, ax = plt.subplots()
+    ax.legend(loc='best')
+    ax.legend(loc='upper right')
+    ax.legend(loc='best')
+    ax.legend(loc='upper right')
+    ax.legend(loc='upper left')
+    ax.legend(loc='lower left')
+    ax.legend(loc='lower right')
+    ax.legend(loc='right')
+    ax.legend(loc='center left')
+    ax.legend(loc='center right')
+    ax.legend(loc='lower center')
+    ax.legend(loc='upper center')
+    with pytest.raises(ValueError, match="'wrong' is not a valid value for"):
+        ax.legend(loc='wrong')

```


## Code snippets

### 1 - lib/matplotlib/legend.py:

Start line: 255, End line: 330

```python
_loc_doc_base = """
loc : str or pair of floats, default: {default}
    The location of the legend.

    The strings ``'upper left'``, ``'upper right'``, ``'lower left'``,
    ``'lower right'`` place the legend at the corresponding corner of the
    {parent}.

    The strings ``'upper center'``, ``'lower center'``, ``'center left'``,
    ``'center right'`` place the legend at the center of the corresponding edge
    of the {parent}.

    The string ``'center'`` places the legend at the center of the {parent}.
{best}
    The location can also be a 2-tuple giving the coordinates of the lower-left
    corner of the legend in {parent} coordinates (in which case *bbox_to_anchor*
    will be ignored).

    For back-compatibility, ``'center right'`` (but no other location) can also
    be spelled ``'right'``, and each "string" location can also be given as a
    numeric value:

        ==================   =============
        Location String      Location Code
        ==================   =============
        'best' (Axes only)   0
        'upper right'        1
        'upper left'         2
        'lower left'         3
        'lower right'        4
        'right'              5
        'center left'        6
        'center right'       7
        'lower center'       8
        'upper center'       9
        'center'             10
        ==================   =============
    {outside}"""

_loc_doc_best = """
    The string ``'best'`` places the legend at the location, among the nine
    locations defined so far, with the minimum overlap with other drawn
    artists.  This option can be quite slow for plots with large amounts of
    data; your plotting speed may benefit from providing a specific location.
"""

_legend_kw_axes_st = (
    _loc_doc_base.format(parent='axes', default=':rc:`legend.loc`',
                         best=_loc_doc_best, outside='') +
    _legend_kw_doc_base)
_docstring.interpd.update(_legend_kw_axes=_legend_kw_axes_st)

_outside_doc = """
    If a figure is using the constrained layout manager, the string codes
    of the *loc* keyword argument can get better layout behaviour using the
    prefix 'outside'. There is ambiguity at the corners, so 'outside
    upper right' will make space for the legend above the rest of the
    axes in the layout, and 'outside right upper' will make space on the
    right side of the layout.  In addition to the values of *loc*
    listed above, we have 'outside right upper', 'outside right lower',
    'outside left upper', and 'outside left lower'.  See
    :doc:`/tutorials/intermediate/legend_guide` for more details.
"""

_legend_kw_figure_st = (
    _loc_doc_base.format(parent='figure', default="'upper right'",
                         best='', outside=_outside_doc) +
    _legend_kw_doc_base)
_docstring.interpd.update(_legend_kw_figure=_legend_kw_figure_st)

_legend_kw_both_st = (
    _loc_doc_base.format(parent='axes/figure',
                         default=":rc:`legend.loc` for Axes, 'upper right' for Figure",
                         best=_loc_doc_best, outside=_outside_doc) +
    _legend_kw_doc_base)
_docstring.interpd.update(_legend_kw_doc=_legend_kw_both_st)
```
### 2 - galleries/tutorials/intermediate/legend_guide.py:

Start line: 124, End line: 200

```python
fig, ax_dict = plt.subplot_mosaic([['top', 'top'], ['bottom', 'BLANK']],
                                  empty_sentinel="BLANK")
ax_dict['top'].plot([1, 2, 3], label="test1")
ax_dict['top'].plot([3, 2, 1], label="test2")
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
ax_dict['top'].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=2, mode="expand", borderaxespad=0.)

ax_dict['bottom'].plot([1, 2, 3], label="test1")
ax_dict['bottom'].plot([3, 2, 1], label="test2")
# Place a legend to the right of this smaller subplot.
ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)

# %%
# Figure legends
# --------------
#
# Sometimes it makes more sense to place a legend relative to the (sub)figure
# rather than individual Axes.  By using *constrained layout* and
# specifying "outside" at the beginning of the *loc* keyword argument,
# the legend is drawn outside the Axes on the (sub)figure.

fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')

axs['left'].plot([1, 2, 3], label="test1")
axs['left'].plot([3, 2, 1], label="test2")

axs['right'].plot([1, 2, 3], 'C2', label="test3")
axs['right'].plot([3, 2, 1], 'C3', label="test4")
# Place a legend to the right of this smaller subplot.
fig.legend(loc='outside upper right')

# %%
# This accepts a slightly different grammar than the normal *loc* keyword,
# where "outside right upper" is different from "outside upper right".
#
ucl = ['upper', 'center', 'lower']
lcr = ['left', 'center', 'right']
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')

ax.plot([1, 2], [1, 2], label='TEST')
# Place a legend to the right of this smaller subplot.
for loc in [
        'outside upper left',
        'outside upper center',
        'outside upper right',
        'outside lower left',
        'outside lower center',
        'outside lower right']:
    fig.legend(loc=loc, title=loc)

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
ax.plot([1, 2], [1, 2], label='test')

for loc in [
        'outside left upper',
        'outside right upper',
        'outside left lower',
        'outside right lower']:
    fig.legend(loc=loc, title=loc)


# %%
# Multiple legends on the same Axes
# =================================
#
# Sometimes it is more clear to split legend entries across multiple
# legends. Whilst the instinctive approach to doing this might be to call
# the :func:`legend` function multiple times, you will find that only one
# legend ever exists on the Axes. This has been done so that it is possible
# to call :func:`legend` repeatedly to update the legend to the latest
# handles on the Axes. To keep old legend instances, we must add them
# manually to the Axes:

fig, ax = plt.subplots()
```
### 3 - galleries/examples/ticks/tick-locators.py:

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
### 4 - lib/matplotlib/legend.py:

Start line: 514, End line: 622

```python
class Legend(Artist):

    @_docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        *,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
        reverse=False,       # reverse ordering of legend marker and label
        scatterpoints=None,  # number of scatter points
        scatteryoffsets=None,
        prop=None,           # properties for the legend texts
        fontsize=None,       # keyword to set font size directly
        labelcolor=None,     # keyword to set the text color

        # spacing & pad defined as a fraction of the font-size
        borderpad=None,      # whitespace inside the legend border
        labelspacing=None,   # vertical space between the legend entries
        handlelength=None,   # length of the legend handles
        handleheight=None,   # height of the legend handles
        handletextpad=None,  # pad between the legend handle and text
        borderaxespad=None,  # pad between the axes and legend border
        columnspacing=None,  # spacing between columns

        ncols=1,     # number of columns
        mode=None,  # horizontal distribution of columns: None or "expand"

        fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
        shadow=None,
        title=None,           # legend title
        title_fontsize=None,  # legend title font size
        framealpha=None,      # set frame alpha
        edgecolor=None,       # frame patch edgecolor
        facecolor=None,       # frame patch facecolor

        bbox_to_anchor=None,  # bbox to which the legend will be anchored
        bbox_transform=None,  # transform for the bbox
        frameon=None,         # draw frame
        handler_map=None,
        title_fontproperties=None,  # properties for the legend title
        alignment="center",       # control the alignment within the legend box
        ncol=1,  # synonym for ncols (backward compatibility)
        draggable=False  # whether the legend can be dragged with the mouse
    ):
        # ... other code
        if isinstance(loc, str):
            if loc.split()[0] == 'outside':
                # strip outside:
                loc = loc.split('outside ')[1]
                # strip "center" at the beginning
                self._outside_loc = loc.replace('center ', '')
                # strip first
                self._outside_loc = self._outside_loc.split()[0]
                locs = loc.split()
                if len(locs) > 1 and locs[0] in ('right', 'left'):
                    # locs doesn't accept "left upper", etc, so swap
                    if locs[0] != 'center':
                        locs = locs[::-1]
                    loc = locs[0] + ' ' + locs[1]
            # check that loc is in acceptable strings
            loc = _api.check_getitem(self.codes, loc=loc)

        if self.isaxes and self._outside_loc:
            raise ValueError(
                f"'outside' option for loc='{loc0}' keyword argument only "
                "works for figure legends")

        if not self.isaxes and loc == 0:
            raise ValueError(
                "Automatic legend placement (loc='best') not implemented for "
                "figure legend")

        self._mode = mode
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)

        # We use FancyBboxPatch to draw a legend frame. The location
        # and size of the box will be updated during the drawing time.

        if facecolor is None:
            facecolor = mpl.rcParams["legend.facecolor"]
        if facecolor == 'inherit':
            facecolor = mpl.rcParams["axes.facecolor"]

        if edgecolor is None:
            edgecolor = mpl.rcParams["legend.edgecolor"]
        if edgecolor == 'inherit':
            edgecolor = mpl.rcParams["axes.edgecolor"]

        if fancybox is None:
            fancybox = mpl.rcParams["legend.fancybox"]

        self.legendPatch = FancyBboxPatch(
            xy=(0, 0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            # If shadow is used, default to alpha=1 (#8943).
            alpha=(framealpha if framealpha is not None
                   else 1 if shadow
                   else mpl.rcParams["legend.framealpha"]),
            # The width and height of the legendPatch will be set (in draw())
            # to the length that includes the padding. Thus we set pad=0 here.
            boxstyle=("round,pad=0,rounding_size=0.2" if fancybox
                      else "square,pad=0"),
            mutation_scale=self._fontsize,
            snap=True,
            visible=(frameon if frameon is not None
                     else mpl.rcParams["legend.frameon"])
        )
        self._set_artist_props(self.legendPatch)

        _api.check_in_list(["center", "left", "right"], alignment=alignment)
        self._alignment = alignment

        # init with null renderer
        self._init_legend_box(handles, labels, markerfirst)

        tmp = self._loc_used_default
        self._set_loc(loc)
        self._loc_used_default = tmp  # ignore changes done by _set_loc

        # figure out title font properties:
        if title_fontsize is not None and title_fontproperties is not None:
            raise ValueError(
                "title_fontsize and title_fontproperties can't be specified "
                "at the same time. Only use one of them. ")
        title_prop_fp = FontProperties._from_any(title_fontproperties)
        if isinstance(title_fontproperties, dict):
            if "size" not in title_fontproperties:
                title_fontsize = mpl.rcParams["legend.title_fontsize"]
                title_prop_fp.set_size(title_fontsize)
        elif title_fontsize is not None:
            title_prop_fp.set_size(title_fontsize)
        elif not isinstance(title_fontproperties, FontProperties):
            title_fontsize = mpl.rcParams["legend.title_fontsize"]
            title_prop_fp.set_size(title_fontsize)

        self.set_title(title, prop=title_prop_fp)

        self._draggable = None
        self.set_draggable(state=draggable)

        # set the text color

        color_getters = {  # getter function depends on line or patch
            'linecolor':       ['get_color',           'get_facecolor'],
            'markerfacecolor': ['get_markerfacecolor', 'get_facecolor'],
            'mfc':             ['get_markerfacecolor', 'get_facecolor'],
            'markeredgecolor': ['get_markeredgecolor', 'get_edgecolor'],
            'mec':             ['get_markeredgecolor', 'get_edgecolor'],
        }
        if labelcolor is None:
            if mpl.rcParams['legend.labelcolor'] is not None:
                labelcolor = mpl.rcParams['legend.labelcolor']
            else:
                labelcolor = mpl.rcParams['text.color']
        # ... other code
```
### 5 - galleries/tutorials/intermediate/legend_guide.py:

Start line: 1, End line: 122

```python
"""
============
Legend guide
============

Generating legends flexibly in Matplotlib.

.. currentmodule:: matplotlib.pyplot

This legend guide is an extension of the documentation available at
:func:`~matplotlib.pyplot.legend` - please ensure you are familiar with
contents of that documentation before proceeding with this guide.


This guide makes use of some common terms, which are documented here for
clarity:

.. glossary::

    legend entry
        A legend is made up of one or more legend entries. An entry is made up
        of exactly one key and one label.

    legend key
        The colored/patterned marker to the left of each legend label.

    legend label
        The text which describes the handle represented by the key.

    legend handle
        The original object which is used to generate an appropriate entry in
        the legend.


Controlling the legend entries
==============================

Calling :func:`legend` with no arguments automatically fetches the legend
handles and their associated labels. This functionality is equivalent to::

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

The :meth:`~matplotlib.axes.Axes.get_legend_handles_labels` function returns
a list of handles/artists which exist on the Axes which can be used to
generate entries for the resulting legend - it is worth noting however that
not all artists can be added to a legend, at which point a "proxy" will have
to be created (see :ref:`proxy_legend_handles` for further details).

.. note::
    Artists with an empty string as label or with a label starting with an
    underscore, "_", will be ignored.

For full control of what is being added to the legend, it is common to pass
the appropriate handles directly to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend(handles=[line_up, line_down])

In some cases, it is not possible to set the label of the handle, so it is
possible to pass through the list of labels to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend([line_up, line_down], ['Line Up', 'Line Down'])


.. _proxy_legend_handles:

Creating artists specifically for adding to the legend (aka. Proxy artists)
===========================================================================

Not all handles can be turned into legend entries automatically,
so it is often necessary to create an artist which *can*. Legend handles
don't have to exist on the Figure or Axes in order to be used.

Suppose we wanted to create a legend which has an entry for some data which
is represented by a red color:
"""

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

fig, ax = plt.subplots()
red_patch = mpatches.Patch(color='red', label='The red data')
ax.legend(handles=[red_patch])

plt.show()

# %%
# There are many supported legend handles. Instead of creating a patch of color
# we could have created a line with a marker:

import matplotlib.lines as mlines

fig, ax = plt.subplots()
blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Blue stars')
ax.legend(handles=[blue_line])

plt.show()

# %%
# Legend location
# ===============
#
# The location of the legend can be specified by the keyword argument
# *loc*. Please see the documentation at :func:`legend` for more details.
#
# The ``bbox_to_anchor`` keyword gives a great degree of control for manual
# legend placement. For example, if you want your axes legend located at the
# figure's top right-hand corner instead of the axes' corner, simply specify
# the corner's location and the coordinate system of that location::
#
#     ax.legend(bbox_to_anchor=(1, 1),
#               bbox_transform=fig.transFigure)
#
# More examples of custom legend placement:
```
### 6 - lib/matplotlib/pyplot.py:

Start line: 2706, End line: 2721

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.legend)
def legend(*args, **kwargs):
    return gca().legend(*args, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.locator_params)
def locator_params(axis='both', tight=None, **kwargs):
    return gca().locator_params(axis=axis, tight=tight, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.loglog)
def loglog(*args, **kwargs):
    return gca().loglog(*args, **kwargs)
```
### 7 - lib/matplotlib/legend.py:

Start line: 1094, End line: 1111

```python
class Legend(Artist):

    def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
        """
        Place the *bbox* inside the *parentbbox* according to a given
        location code. Return the (x, y) coordinate of the bbox.

        Parameters
        ----------
        loc : int
            A location code in range(1, 11). This corresponds to the possible
            values for ``self._loc``, excluding "best".
        bbox : `~matplotlib.transforms.Bbox`
            bbox to be placed, in display coordinates.
        parentbbox : `~matplotlib.transforms.Bbox`
            A parent box which will contain the bbox, in display coordinates.
        """
        return offsetbox._get_anchored_bbox(
            loc, bbox, parentbbox,
            self.borderaxespad * renderer.points_to_pixels(self._fontsize))
```
### 8 - galleries/tutorials/intermediate/legend_guide.py:

Start line: 201, End line: 277

```python
line1, = ax.plot([1, 2, 3], label="Line 1", linestyle='--')
line2, = ax.plot([3, 2, 1], label="Line 2", linewidth=4)

# Create a legend for the first line.
first_legend = ax.legend(handles=[line1], loc='upper right')

# Add the legend manually to the Axes.
ax.add_artist(first_legend)

# Create another legend for the second line.
ax.legend(handles=[line2], loc='lower right')

plt.show()

# %%
# Legend Handlers
# ===============
#
# In order to create legend entries, handles are given as an argument to an
# appropriate :class:`~matplotlib.legend_handler.HandlerBase` subclass.
# The choice of handler subclass is determined by the following rules:
#
# 1. Update :func:`~matplotlib.legend.Legend.get_legend_handler_map`
#    with the value in the ``handler_map`` keyword.
# 2. Check if the ``handle`` is in the newly created ``handler_map``.
# 3. Check if the type of ``handle`` is in the newly created ``handler_map``.
# 4. Check if any of the types in the ``handle``'s mro is in the newly
#    created ``handler_map``.
#
# For completeness, this logic is mostly implemented in
# :func:`~matplotlib.legend.Legend.get_legend_handler`.
#
# All of this flexibility means that we have the necessary hooks to implement
# custom handlers for our own type of legend key.
#
# The simplest example of using custom handlers is to instantiate one of the
# existing `.legend_handler.HandlerBase` subclasses. For the
# sake of simplicity, let's choose `.legend_handler.HandlerLine2D`
# which accepts a *numpoints* argument (numpoints is also a keyword
# on the :func:`legend` function for convenience). We can then pass the mapping
# of instance to Handler as a keyword to legend.

from matplotlib.legend_handler import HandlerLine2D

fig, ax = plt.subplots()
line1, = ax.plot([3, 2, 1], marker='o', label='Line 1')
line2, = ax.plot([1, 2, 3], marker='o', label='Line 2')

ax.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

# %%
# As you can see, "Line 1" now has 4 marker points, where "Line 2" has 2 (the
# default). Try the above code, only change the map's key from ``line1`` to
# ``type(line1)``. Notice how now both `.Line2D` instances get 4 markers.
#
# Along with handlers for complex plot types such as errorbars, stem plots
# and histograms, the default ``handler_map`` has a special ``tuple`` handler
# (`.legend_handler.HandlerTuple`) which simply plots the handles on top of one
# another for each item in the given tuple. The following example demonstrates
# combining two legend keys on top of one another:

from numpy.random import randn

z = randn(10)

fig, ax = plt.subplots()
red_dot, = ax.plot(z, "ro", markersize=15)
# Put a white cross over some of the data.
white_cross, = ax.plot(z[:5], "w+", markeredgewidth=3, markersize=15)

ax.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])

# %%
# The `.legend_handler.HandlerTuple` class can also be used to
# assign several legend keys to the same entry:

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
```
### 9 - galleries/examples/ticks/date_formatters_locators.py:

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
### 10 - galleries/examples/text_labels_and_annotations/legend_demo.py:

Start line: 1, End line: 76

```python
"""
===========
Legend Demo
===========

Plotting legends in Matplotlib.

There are many ways to create and customize legends in Matplotlib. Below
we'll show a few examples for how to do so.

First we'll show off how to make a legend for specific lines.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D

t1 = np.arange(0.0, 2.0, 0.1)
t2 = np.arange(0.0, 2.0, 0.01)

fig, ax = plt.subplots()

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1, = ax.plot(t2, np.exp(-t2))
l2, l3 = ax.plot(t2, np.sin(2 * np.pi * t2), '--o', t1, np.log(1 + t1), '.')
l4, = ax.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 's-.')

ax.legend((l2, l4), ('oscillatory', 'damped'), loc='upper right', shadow=True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
ax.set_title('Damped oscillation')
plt.show()


# %%
# Next we'll demonstrate plotting more complex labels.

x = np.linspace(0, 1)

fig, (ax0, ax1) = plt.subplots(2, 1)

# Plot the lines y=x**n for n=1..4.
for n in range(1, 5):
    ax0.plot(x, x**n, label=f"{n=}")
leg = ax0.legend(loc="upper left", bbox_to_anchor=[0, 1],
                 ncols=2, shadow=True, title="Legend", fancybox=True)
leg.get_title().set_color("red")

# Demonstrate some more complex labels.
ax1.plot(x, x**2, label="multi\nline")
half_pi = np.linspace(0, np.pi / 2)
ax1.plot(np.sin(half_pi), np.cos(half_pi), label=r"$\frac{1}{2}\pi$")
ax1.plot(x, 2**(x**2), label="$2^{x^2}$")
ax1.legend(shadow=True, fancybox=True)

plt.show()


# %%
# Here we attach legends to more complex plots.

fig, axs = plt.subplots(3, 1, layout="constrained")
top_ax, middle_ax, bottom_ax = axs

top_ax.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4, label="Bar 1",
           align="center")
top_ax.bar([0.5, 1.5, 2.5], [0.3, 0.2, 0.2], color="red", width=0.4,
           label="Bar 2", align="center")
top_ax.legend()

middle_ax.errorbar([0, 1, 2], [2, 3, 1], xerr=0.4, fmt="s", label="test 1")
```
### 20 - lib/matplotlib/legend.py:

Start line: 333, End line: 513

```python
class Legend(Artist):
    """
    Place a legend on the figure/axes.
    """

    # 'best' is only implemented for axes legends
    codes = {'best': 0, **AnchoredOffsetbox.codes}
    zorder = 5

    def __str__(self):
        return "Legend"

    @_docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        *,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
        reverse=False,       # reverse ordering of legend marker and label
        scatterpoints=None,  # number of scatter points
        scatteryoffsets=None,
        prop=None,           # properties for the legend texts
        fontsize=None,       # keyword to set font size directly
        labelcolor=None,     # keyword to set the text color

        # spacing & pad defined as a fraction of the font-size
        borderpad=None,      # whitespace inside the legend border
        labelspacing=None,   # vertical space between the legend entries
        handlelength=None,   # length of the legend handles
        handleheight=None,   # height of the legend handles
        handletextpad=None,  # pad between the legend handle and text
        borderaxespad=None,  # pad between the axes and legend border
        columnspacing=None,  # spacing between columns

        ncols=1,     # number of columns
        mode=None,  # horizontal distribution of columns: None or "expand"

        fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
        shadow=None,
        title=None,           # legend title
        title_fontsize=None,  # legend title font size
        framealpha=None,      # set frame alpha
        edgecolor=None,       # frame patch edgecolor
        facecolor=None,       # frame patch facecolor

        bbox_to_anchor=None,  # bbox to which the legend will be anchored
        bbox_transform=None,  # transform for the bbox
        frameon=None,         # draw frame
        handler_map=None,
        title_fontproperties=None,  # properties for the legend title
        alignment="center",       # control the alignment within the legend box
        ncol=1,  # synonym for ncols (backward compatibility)
        draggable=False  # whether the legend can be dragged with the mouse
    ):
        """
        Parameters
        ----------
        parent : `~matplotlib.axes.Axes` or `.Figure`
            The artist that contains the legend.

        handles : list of `.Artist`
            A list of Artists (lines, patches) to be added to the legend.

        labels : list of str
            A list of labels to show next to the artists. The length of handles
            and labels should be the same. If they are not, they are truncated
            to the length of the shorter list.

        Other Parameters
        ----------------
        %(_legend_kw_doc)s

        Attributes
        ----------
        legend_handles
            List of `.Artist` objects added as legend entries.

            .. versionadded:: 3.7
        """
        # local import only to avoid circularity
        from matplotlib.axes import Axes
        from matplotlib.figure import FigureBase

        super().__init__()

        if prop is None:
            if fontsize is not None:
                self.prop = FontProperties(size=fontsize)
            else:
                self.prop = FontProperties(
                    size=mpl.rcParams["legend.fontsize"])
        else:
            self.prop = FontProperties._from_any(prop)
            if isinstance(prop, dict) and "size" not in prop:
                self.prop.set_size(mpl.rcParams["legend.fontsize"])

        self._fontsize = self.prop.get_size_in_points()

        self.texts = []
        self.legend_handles = []
        self._legend_title_box = None

        #: A dictionary with the extra handler mappings for this Legend
        #: instance.
        self._custom_handler_map = handler_map

        def val_or_rc(val, rc_name):
            return val if val is not None else mpl.rcParams[rc_name]

        self.numpoints = val_or_rc(numpoints, 'legend.numpoints')
        self.markerscale = val_or_rc(markerscale, 'legend.markerscale')
        self.scatterpoints = val_or_rc(scatterpoints, 'legend.scatterpoints')
        self.borderpad = val_or_rc(borderpad, 'legend.borderpad')
        self.labelspacing = val_or_rc(labelspacing, 'legend.labelspacing')
        self.handlelength = val_or_rc(handlelength, 'legend.handlelength')
        self.handleheight = val_or_rc(handleheight, 'legend.handleheight')
        self.handletextpad = val_or_rc(handletextpad, 'legend.handletextpad')
        self.borderaxespad = val_or_rc(borderaxespad, 'legend.borderaxespad')
        self.columnspacing = val_or_rc(columnspacing, 'legend.columnspacing')
        self.shadow = val_or_rc(shadow, 'legend.shadow')
        # trim handles and labels if illegal label...
        _lab, _hand = [], []
        for label, handle in zip(labels, handles):
            if isinstance(label, str) and label.startswith('_'):
                _api.warn_external(f"The label {label!r} of {handle!r} starts "
                                   "with '_'. It is thus excluded from the "
                                   "legend.")
            else:
                _lab.append(label)
                _hand.append(handle)
        labels, handles = _lab, _hand

        if reverse:
            labels.reverse()
            handles.reverse()

        if len(handles) < 2:
            ncols = 1
        self._ncols = ncols if ncols != 1 else ncol

        if self.numpoints <= 0:
            raise ValueError("numpoints must be > 0; it was %d" % numpoints)

        # introduce y-offset for handles of the scatter plot
        if scatteryoffsets is None:
            self._scatteryoffsets = np.array([3. / 8., 4. / 8., 2.5 / 8.])
        else:
            self._scatteryoffsets = np.asarray(scatteryoffsets)
        reps = self.scatterpoints // len(self._scatteryoffsets) + 1
        self._scatteryoffsets = np.tile(self._scatteryoffsets,
                                        reps)[:self.scatterpoints]

        # _legend_box is a VPacker instance that contains all
        # legend items and will be initialized from _init_legend_box()
        # method.
        self._legend_box = None

        if isinstance(parent, Axes):
            self.isaxes = True
            self.axes = parent
            self.set_figure(parent.figure)
        elif isinstance(parent, FigureBase):
            self.isaxes = False
            self.set_figure(parent)
        else:
            raise TypeError(
                "Legend needs either Axes or FigureBase as parent"
            )
        self.parent = parent

        loc0 = loc
        self._loc_used_default = loc is None
        if loc is None:
            loc = mpl.rcParams["legend.loc"]
            if not self.isaxes and loc in [0, 'best']:
                loc = 'upper right'

        # handle outside legends:
        self._outside_loc = None
        # ... other code
```
### 22 - lib/matplotlib/legend.py:

Start line: 707, End line: 736

```python
class Legend(Artist):

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return

        renderer.open_group('legend', gid=self.get_gid())

        fontsize = renderer.points_to_pixels(self._fontsize)

        # if mode == fill, set the width of the legend_box to the
        # width of the parent (minus pads)
        if self._mode in ["expand"]:
            pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
            self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)

        # update the location and size of the legend. This needs to
        # be done in any case to clip the figure right.
        bbox = self._legend_box.get_window_extent(renderer)
        self.legendPatch.set_bounds(bbox.bounds)
        self.legendPatch.set_mutation_scale(fontsize)

        if self.shadow:
            Shadow(self.legendPatch, 2, -2).draw(renderer)

        self.legendPatch.draw(renderer)
        self._legend_box.draw(renderer)

        renderer.close_group('legend')
        self.stale = False
```
### 25 - lib/matplotlib/legend.py:

Start line: 690, End line: 705

```python
class Legend(Artist):

    def _findoffset(self, width, height, xdescent, ydescent, renderer):
        """Helper function to locate the legend."""

        if self._loc == 0:  # "best".
            x, y = self._find_best_position(width, height, renderer)
        elif self._loc in Legend.codes.values():  # Fixed location.
            bbox = Bbox.from_bounds(0, 0, width, height)
            x, y = self._get_anchored_bbox(self._loc, bbox,
                                           self.get_bbox_to_anchor(),
                                           renderer)
        else:  # Axes or figure coordinates.
            fx, fy = self._loc
            bbox = self.get_bbox_to_anchor()
            x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy

        return x + xdescent, y + ydescent
```
### 27 - lib/matplotlib/legend.py:

Start line: 1, End line: 50

```python
"""
The legend module defines the Legend class, which is responsible for
drawing legends associated with axes and/or figures.

.. important::

    It is unlikely that you would ever create a Legend instance manually.
    Most users would normally create a legend via the `~.Axes.legend`
    function. For more details on legends there is also a :doc:`legend guide
    </tutorials/intermediate/legend_guide>`.

The `Legend` class is a container of legend handles and legend texts.

The legend handler map specifies how to create legend handles from artists
(lines, patches, etc.) in the axes or figures. Default legend handlers are
defined in the :mod:`~matplotlib.legend_handler` module. While not all artist
types are covered by the default legend handlers, custom legend handlers can be
defined to support arbitrary objects.

See the :doc:`legend guide </tutorials/intermediate/legend_guide>` for more
information.
"""

import itertools
import logging
import time

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
                                StepPatch)
from matplotlib.collections import (
    Collection, CircleCollection, LineCollection, PathCollection,
    PolyCollection, RegularPolyCollection)
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
    AnchoredOffsetbox, DraggableOffsetBox,
    HPacker, VPacker,
    DrawingArea, TextArea,
)
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
```
### 32 - lib/matplotlib/legend.py:

Start line: 658, End line: 688

```python
class Legend(Artist):

    legendHandles = _api.deprecated('3.7', alternative="legend_handles")(
        property(lambda self: self.legend_handles))

    def _set_artist_props(self, a):
        """
        Set the boilerplate props for artists added to axes.
        """
        a.set_figure(self.figure)
        if self.isaxes:
            # a.set_axes(self.axes)
            a.axes = self.axes

        a.set_transform(self.get_transform())

    def _set_loc(self, loc):
        # find_offset function will be provided to _legend_box and
        # _legend_box will draw itself at the location of the return
        # value of the find_offset.
        self._loc_used_default = False
        self._loc_real = loc
        self.stale = True
        self._legend_box.set_offset(self._findoffset)

    def set_ncols(self, ncols):
        """Set the number of columns."""
        self._ncols = ncols

    def _get_loc(self):
        return self._loc_real

    _loc = property(_get_loc, _set_loc)
```
### 33 - lib/matplotlib/legend.py:

Start line: 623, End line: 656

```python
class Legend(Artist):

    @_docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        *,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
        reverse=False,       # reverse ordering of legend marker and label
        scatterpoints=None,  # number of scatter points
        scatteryoffsets=None,
        prop=None,           # properties for the legend texts
        fontsize=None,       # keyword to set font size directly
        labelcolor=None,     # keyword to set the text color

        # spacing & pad defined as a fraction of the font-size
        borderpad=None,      # whitespace inside the legend border
        labelspacing=None,   # vertical space between the legend entries
        handlelength=None,   # length of the legend handles
        handleheight=None,   # height of the legend handles
        handletextpad=None,  # pad between the legend handle and text
        borderaxespad=None,  # pad between the axes and legend border
        columnspacing=None,  # spacing between columns

        ncols=1,     # number of columns
        mode=None,  # horizontal distribution of columns: None or "expand"

        fancybox=None,  # True: fancy box, False: rounded box, None: rcParam
        shadow=None,
        title=None,           # legend title
        title_fontsize=None,  # legend title font size
        framealpha=None,      # set frame alpha
        edgecolor=None,       # frame patch edgecolor
        facecolor=None,       # frame patch facecolor

        bbox_to_anchor=None,  # bbox to which the legend will be anchored
        bbox_transform=None,  # transform for the bbox
        frameon=None,         # draw frame
        handler_map=None,
        title_fontproperties=None,  # properties for the legend title
        alignment="center",       # control the alignment within the legend box
        ncol=1,  # synonym for ncols (backward compatibility)
        draggable=False  # whether the legend can be dragged with the mouse
    ):
        # ... other code
        if isinstance(labelcolor, str) and labelcolor in color_getters:
            getter_names = color_getters[labelcolor]
            for handle, text in zip(self.legend_handles, self.texts):
                try:
                    if handle.get_array() is not None:
                        continue
                except AttributeError:
                    pass
                for getter_name in getter_names:
                    try:
                        color = getattr(handle, getter_name)()
                        if isinstance(color, np.ndarray):
                            if (
                                    color.shape[0] == 1
                                    or np.isclose(color, color[0]).all()
                            ):
                                text.set_color(color[0])
                            else:
                                pass
                        else:
                            text.set_color(color)
                        break
                    except AttributeError:
                        pass
        elif isinstance(labelcolor, str) and labelcolor == 'none':
            for text in self.texts:
                text.set_color(labelcolor)
        elif np.iterable(labelcolor):
            for text, color in zip(self.texts,
                                   itertools.cycle(
                                       colors.to_rgba_array(labelcolor))):
                text.set_color(color)
        else:
            raise ValueError(f"Invalid labelcolor: {labelcolor!r}")
```
### 41 - lib/matplotlib/legend.py:

Start line: 1113, End line: 1157

```python
class Legend(Artist):

    def _find_best_position(self, width, height, renderer, consider=None):
        """
        Determine the best location to place the legend.

        *consider* is a list of ``(x, y)`` pairs to consider as a potential
        lower-left corner of the legend. All are display coords.
        """
        assert self.isaxes  # always holds, as this is only called internally

        start_time = time.perf_counter()

        bboxes, lines, offsets = self._auto_legend_data()

        bbox = Bbox.from_bounds(0, 0, width, height)
        if consider is None:
            consider = [self._get_anchored_bbox(x, bbox,
                                                self.get_bbox_to_anchor(),
                                                renderer)
                        for x in range(1, len(self.codes))]

        candidates = []
        for idx, (l, b) in enumerate(consider):
            legendBox = Bbox.from_bounds(l, b, width, height)
            badness = 0
            # XXX TODO: If markers are present, it would be good to take them
            # into account when checking vertex overlaps in the next line.
            badness = (sum(legendBox.count_contains(line.vertices)
                           for line in lines)
                       + legendBox.count_contains(offsets)
                       + legendBox.count_overlaps(bboxes)
                       + sum(line.intersects_bbox(legendBox, filled=False)
                             for line in lines))
            if badness == 0:
                return l, b
            # Include the index to favor lower codes in case of a tie.
            candidates.append((badness, idx, (l, b)))

        _, _, (l, b) = min(candidates)

        if self._loc_used_default and time.perf_counter() - start_time > 1:
            _api.warn_external(
                'Creating legend with loc="best" can be slow with large '
                'amounts of data.')

        return l, b
```
### 42 - lib/matplotlib/legend.py:

Start line: 907, End line: 940

```python
class Legend(Artist):

    def _auto_legend_data(self):
        """
        Return display coordinates for hit testing for "best" positioning.

        Returns
        -------
        bboxes
            List of bounding boxes of all patches.
        lines
            List of `.Path` corresponding to each line.
        offsets
            List of (x, y) offsets of all collection.
        """
        assert self.isaxes  # always holds, as this is only called internally
        bboxes = []
        lines = []
        offsets = []
        for artist in self.parent._children:
            if isinstance(artist, Line2D):
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, Rectangle):
                bboxes.append(
                    artist.get_bbox().transformed(artist.get_data_transform()))
            elif isinstance(artist, Patch):
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, Collection):
                transform, transOffset, hoffsets, _ = artist._prepare_points()
                if len(hoffsets):
                    for offset in transOffset.transform(hoffsets):
                        offsets.append(offset)

        return bboxes, lines, offsets
```
