# matplotlib__matplotlib-23742

| **matplotlib/matplotlib** | `942aa77a4ba1bd5b50e22c0246240b27ba925305` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 20752 |
| **Any found context length** | 20752 |
| **Avg pos** | 25.5 |
| **Min pos** | 51 |
| **Max pos** | 51 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/examples/user_interfaces/embedding_webagg_sgskip.py b/examples/user_interfaces/embedding_webagg_sgskip.py
--- a/examples/user_interfaces/embedding_webagg_sgskip.py
+++ b/examples/user_interfaces/embedding_webagg_sgskip.py
@@ -30,7 +30,7 @@
 
 
 import matplotlib as mpl
-from matplotlib.backends.backend_webagg_core import (
+from matplotlib.backends.backend_webagg import (
     FigureManagerWebAgg, new_figure_manager_given_figure)
 from matplotlib.figure import Figure
 
diff --git a/lib/matplotlib/backends/backend_webagg_core.py b/lib/matplotlib/backends/backend_webagg_core.py
--- a/lib/matplotlib/backends/backend_webagg_core.py
+++ b/lib/matplotlib/backends/backend_webagg_core.py
@@ -427,7 +427,9 @@ def set_history_buttons(self):
 
 
 class FigureManagerWebAgg(backend_bases.FigureManagerBase):
-    _toolbar2_class = ToolbarCls = NavigationToolbar2WebAgg
+    # This must be None to not break ipympl
+    _toolbar2_class = None
+    ToolbarCls = NavigationToolbar2WebAgg
 
     def __init__(self, canvas, num):
         self.web_sockets = set()

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| examples/user_interfaces/embedding_webagg_sgskip.py | 33 | 33 | - | - | -
| lib/matplotlib/backends/backend_webagg_core.py | 430 | 430 | 51 | 2 | 20752


## Problem Statement

```
[Bug]: Bug with toolbar instantiation in notebook
### Bug summary

In MNE-Python we have an abstraction layer for widgets+toolbars. Until today's latest `pip --pre` install it was working fine. Now it fails with:

\`\`\`
E   TraitError: The 'toolbar' trait of a Canvas instance expected a Toolbar or None, not the NavigationToolbar2WebAgg at '0x7fce12bf6f80'.
\`\`\`

See https://dev.azure.com/mne-tools/mne-python/_build/results?buildId=21230&view=logs&jobId=2b5832ae-6860-5681-a4e1-fd132048f8b4&j=2b5832ae-6860-5681-a4e1-fd132048f8b4&t=5b9d2bdb-d99e-53c3-c7bb-7166fe849ae1

### Code for reproduction

I'll work on boiling it down to a MWE tomorrow hopefully. Could also be a bug with traitlets. But the code that runs on the CIs is essentially this in a notebook I think:

\`\`\`python
plt.ioff()
fig, ax = plt.subplots()
\`\`\`


### Actual outcome

<details>
<summary>Full traceback</summary>

\`\`\`
E   nbclient.exceptions.CellExecutionError: An error occurred while executing the following cell:
E   ------------------
E   """Test the GUI widgets abstraction in notebook."""
E   from mne.viz import set_3d_backend
E   from mne.viz.backends.renderer import _get_backend
E   from mne.viz.backends.tests.test_abstract import _do_widget_tests
E   from IPython import get_ipython
E   
E   set_3d_backend('notebook')
E   backend = _get_backend()
E   
E   ipython = get_ipython()
E   ipython.magic('%matplotlib widget')
E   
E   _do_widget_tests(backend)
E   ------------------
E   
E   ---------------------------------------------------------------------------
E   TraitError                                Traceback (most recent call last)
E   Input In [1], in <cell line: 13>()
E        10 ipython = get_ipython()
E        11 ipython.magic('%matplotlib widget')
E   ---> 13 _do_widget_tests(backend)
E   
E   File ~/work/1/s/mne/viz/backends/tests/test_abstract.py:23, in _do_widget_tests(backend)
E        21 renderer.sphere([0, 0, 0], 'red', 1)
E        22 central_layout._add_widget(renderer.plotter)
E   ---> 23 canvas = backend._Canvas(5, 5, 96)
E        24 canvas.ax.plot(range(10), range(10), label='plot')
E        25 central_layout._add_widget(canvas)
E   
\`\`\`

</details>

### Expected outcome

No error

### Additional information

*EDIT*: ipympl 0.9.2

### Operating system

Ubuntu (GH actions), or macOS M1 (locally)

### Matplotlib Version

3.6.0rc1

### Matplotlib Backend

Notebook

### Python version

3.10

### Jupyter version

*EDIT*: 6.4.11

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 lib/matplotlib/backends/backend_macosx.py | 105 | 142| 379 | 379 | 1518 | 
| 2 | **2 lib/matplotlib/backends/backend_webagg_core.py** | 381 | 426| 407 | 786 | 5548 | 
| 3 | 3 lib/matplotlib/backends/backend_gtk3.py | 614 | 630| 115 | 901 | 10589 | 
| 4 | 3 lib/matplotlib/backends/backend_gtk3.py | 306 | 357| 458 | 1359 | 10589 | 
| 5 | 4 lib/matplotlib/backends/_backend_tk.py | 562 | 619| 511 | 1870 | 19883 | 
| 6 | 5 examples/user_interfaces/embedding_in_gtk4_panzoom_sgskip.py | 1 | 52| 319 | 2189 | 20202 | 
| 7 | 6 examples/user_interfaces/embedding_in_gtk3_panzoom_sgskip.py | 1 | 44| 273 | 2462 | 20475 | 
| 8 | 7 lib/matplotlib/backends/backend_gtk4.py | 259 | 304| 405 | 2867 | 24978 | 
| 9 | 7 lib/matplotlib/backends/backend_gtk4.py | 564 | 580| 115 | 2982 | 24978 | 
| 10 | 8 lib/matplotlib/backends/backend_nbagg.py | 64 | 90| 197 | 3179 | 26748 | 
| 11 | 9 lib/matplotlib/pyplot.py | 1 | 89| 669 | 3848 | 54715 | 
| 12 | 10 examples/user_interfaces/embedding_in_wx2_sgskip.py | 1 | 65| 423 | 4271 | 55138 | 
| 13 | 10 lib/matplotlib/backends/_backend_tk.py | 668 | 684| 203 | 4474 | 55138 | 
| 14 | 11 lib/matplotlib/backends/_backend_gtk.py | 249 | 289| 376 | 4850 | 57615 | 
| 15 | 11 lib/matplotlib/backends/backend_gtk3.py | 1 | 37| 254 | 5104 | 57615 | 
| 16 | 11 lib/matplotlib/backends/backend_gtk3.py | 359 | 405| 430 | 5534 | 57615 | 
| 17 | 11 lib/matplotlib/backends/_backend_tk.py | 992 | 1035| 292 | 5826 | 57615 | 
| 18 | 11 lib/matplotlib/backends/_backend_gtk.py | 292 | 326| 250 | 6076 | 57615 | 
| 19 | 12 lib/matplotlib/backend_bases.py | 2530 | 3316| 6572 | 12648 | 85592 | 
| 20 | 12 lib/matplotlib/backends/_backend_tk.py | 1 | 56| 368 | 13016 | 85592 | 
| 21 | 12 lib/matplotlib/backends/backend_gtk4.py | 343 | 366| 223 | 13239 | 85592 | 
| 22 | 12 lib/matplotlib/backends/backend_macosx.py | 145 | 184| 258 | 13497 | 85592 | 
| 23 | 12 lib/matplotlib/backends/_backend_tk.py | 763 | 795| 319 | 13816 | 85592 | 
| 24 | 13 lib/matplotlib/backends/backend_qt.py | 715 | 753| 337 | 14153 | 94124 | 
| 25 | 14 lib/matplotlib/backends/backend_wx.py | 1138 | 1169| 347 | 14500 | 106306 | 
| 26 | 14 lib/matplotlib/backends/_backend_tk.py | 797 | 835| 377 | 14877 | 106306 | 
| 27 | 14 lib/matplotlib/backends/_backend_tk.py | 920 | 940| 209 | 15086 | 106306 | 
| 28 | 15 lib/matplotlib/backends/backend_gtk4agg.py | 1 | 42| 305 | 15391 | 106612 | 
| 29 | 15 lib/matplotlib/backends/backend_gtk4.py | 1 | 31| 217 | 15608 | 106612 | 
| 30 | 15 lib/matplotlib/backends/_backend_tk.py | 899 | 917| 192 | 15800 | 106612 | 
| 31 | 15 lib/matplotlib/backends/backend_nbagg.py | 1 | 21| 174 | 15974 | 106612 | 
| 32 | 15 lib/matplotlib/backends/backend_nbagg.py | 42 | 61| 184 | 16158 | 106612 | 
| 33 | 15 lib/matplotlib/backends/backend_gtk4.py | 306 | 341| 318 | 16476 | 106612 | 
| 34 | 16 lib/matplotlib/backends/backend_wxagg.py | 1 | 14| 144 | 16620 | 107130 | 
| 35 | 16 lib/matplotlib/backends/_backend_tk.py | 648 | 666| 154 | 16774 | 107130 | 
| 36 | **16 lib/matplotlib/backends/backend_webagg_core.py** | 157 | 200| 387 | 17161 | 107130 | 
| 37 | 16 lib/matplotlib/backends/backend_wx.py | 10 | 39| 196 | 17357 | 107130 | 
| 38 | **16 lib/matplotlib/backends/backend_webagg_core.py** | 309 | 333| 242 | 17599 | 107130 | 
| 39 | 17 lib/matplotlib/backends/backend_gtk3agg.py | 1 | 48| 375 | 17974 | 107726 | 
| 40 | 17 lib/matplotlib/backends/backend_gtk4.py | 369 | 395| 240 | 18214 | 107726 | 
| 41 | 18 examples/user_interfaces/embedding_in_wx4_sgskip.py | 1 | 39| 301 | 18515 | 108323 | 
| 42 | 19 examples/user_interfaces/mpl_with_glade3_sgskip.py | 1 | 52| 309 | 18824 | 108632 | 
| 43 | 19 lib/matplotlib/backends/backend_wx.py | 1302 | 1310| 113 | 18937 | 108632 | 
| 44 | 20 lib/matplotlib/backends/backend_tkagg.py | 1 | 21| 160 | 19097 | 108793 | 
| 45 | 20 lib/matplotlib/backends/backend_wx.py | 1282 | 1299| 159 | 19256 | 108793 | 
| 46 | 20 lib/matplotlib/backends/_backend_tk.py | 958 | 989| 230 | 19486 | 108793 | 
| 47 | 20 lib/matplotlib/backends/backend_wx.py | 1171 | 1190| 215 | 19701 | 108793 | 
| 48 | 20 lib/matplotlib/backends/backend_qt.py | 685 | 713| 272 | 19973 | 108793 | 
| 49 | 21 examples/user_interfaces/embedding_in_wx3_sgskip.py | 24 | 54| 249 | 20222 | 109927 | 
| 50 | 21 lib/matplotlib/backends/backend_gtk3agg.py | 50 | 75| 246 | 20468 | 109927 | 
| **-> 51 <-** | **21 lib/matplotlib/backends/backend_webagg_core.py** | 429 | 469| 284 | 20752 | 109927 | 
| 52 | 21 lib/matplotlib/backends/backend_gtk3.py | 489 | 500| 114 | 20866 | 109927 | 
| 53 | 21 lib/matplotlib/backends/backend_gtk4.py | 31 | 89| 443 | 21309 | 109927 | 
| 54 | 21 lib/matplotlib/backends/backend_gtk3.py | 153 | 202| 461 | 21770 | 109927 | 
| 55 | 22 lib/matplotlib/axes/_base.py | 1 | 31| 200 | 21970 | 147917 | 
| 56 | 23 examples/user_interfaces/mathtext_wx_sgskip.py | 97 | 132| 262 | 22232 | 148977 | 
| 57 | 23 lib/matplotlib/backends/_backend_gtk.py | 1 | 46| 295 | 22527 | 148977 | 
| 58 | 24 lib/matplotlib/backends/backend_gtk4cairo.py | 1 | 30| 234 | 22761 | 149212 | 
| 59 | 25 lib/matplotlib/backends/backend_gtk3cairo.py | 1 | 28| 226 | 22987 | 149439 | 
| 60 | 26 lib/matplotlib/backends/backend_tkcairo.py | 1 | 27| 224 | 23211 | 149664 | 
| 61 | 26 lib/matplotlib/backends/backend_qt.py | 986 | 994| 108 | 23319 | 149664 | 
| 62 | 27 ci/check_wheel_licenses.py | 1 | 37| 213 | 23532 | 149921 | 
| 63 | 27 lib/matplotlib/backend_bases.py | 3319 | 3582| 1729 | 25261 | 149921 | 
| 64 | 28 lib/matplotlib/backends/backend_wxcairo.py | 1 | 14| 150 | 25411 | 150297 | 
| 65 | 29 examples/user_interfaces/embedding_in_tk_sgskip.py | 1 | 68| 466 | 25877 | 150763 | 
| 66 | 29 lib/matplotlib/backend_bases.py | 810 | 1660| 6378 | 32255 | 150763 | 
| 67 | 29 lib/matplotlib/backends/backend_qt.py | 962 | 983| 180 | 32435 | 150763 | 


## Missing Patch Files

 * 1: examples/user_interfaces/embedding_webagg_sgskip.py
 * 2: lib/matplotlib/backends/backend_webagg_core.py

### Hint

```
Okay I can replicate on 3.6.0.rc0 in a notebook with just:
\`\`\`
%matplotlib widget
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
\`\`\`

<details>
<summary>Traceback</summary>

\`\`\`
---------------------------------------------------------------------------
TraitError                                Traceback (most recent call last)
Input In [1], in <cell line: 3>()
      1 get_ipython().run_line_magic('matplotlib', 'widget')
      2 import matplotlib.pyplot as plt
----> 3 fig, ax = plt.subplots()

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/pyplot.py:1430, in subplots(nrows, ncols, sharex, sharey, squeeze, width_ratios, height_ratios, subplot_kw, gridspec_kw, **fig_kw)
   1284 def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True,
   1285              width_ratios=None, height_ratios=None,
   1286              subplot_kw=None, gridspec_kw=None, **fig_kw):
   1287     """
   1288     Create a figure and a set of subplots.
   1289 
   (...)
   1428 
   1429     """
-> 1430     fig = figure(**fig_kw)
   1431     axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
   1432                        squeeze=squeeze, subplot_kw=subplot_kw,
   1433                        gridspec_kw=gridspec_kw, height_ratios=height_ratios,
   1434                        width_ratios=width_ratios)
   1435     return fig, axs

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/_api/deprecation.py:454, in make_keyword_only.<locals>.wrapper(*args, **kwargs)
    448 if len(args) > name_idx:
    449     warn_deprecated(
    450         since, message="Passing the %(name)s %(obj_type)s "
    451         "positionally is deprecated since Matplotlib %(since)s; the "
    452         "parameter will become keyword-only %(removal)s.",
    453         name=name, obj_type=f"parameter of {func.__name__}()")
--> 454 return func(*args, **kwargs)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/pyplot.py:771, in figure(num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear, **kwargs)
    761 if len(allnums) == max_open_warning >= 1:
    762     _api.warn_external(
    763         f"More than {max_open_warning} figures have been opened. "
    764         f"Figures created through the pyplot interface "
   (...)
    768         f"Consider using `matplotlib.pyplot.close()`.",
    769         RuntimeWarning)
--> 771 manager = new_figure_manager(
    772     num, figsize=figsize, dpi=dpi,
    773     facecolor=facecolor, edgecolor=edgecolor, frameon=frameon,
    774     FigureClass=FigureClass, **kwargs)
    775 fig = manager.canvas.figure
    776 if fig_label:

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/pyplot.py:347, in new_figure_manager(*args, **kwargs)
    345 """Create a new figure manager instance."""
    346 _warn_if_gui_out_of_main_thread()
--> 347 return _get_backend_mod().new_figure_manager(*args, **kwargs)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/backend_bases.py:3505, in _Backend.new_figure_manager(cls, num, *args, **kwargs)
   3503 fig_cls = kwargs.pop('FigureClass', Figure)
   3504 fig = fig_cls(*args, **kwargs)
-> 3505 return cls.new_figure_manager_given_figure(num, fig)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/ipympl/backend_nbagg.py:487, in _Backend_ipympl.new_figure_manager_given_figure(num, figure)
    485 if 'nbagg.transparent' in rcParams and rcParams['nbagg.transparent']:
    486     figure.patch.set_alpha(0)
--> 487 manager = FigureManager(canvas, num)
    489 if is_interactive():
    490     _Backend_ipympl._to_show.append(figure)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/ipympl/backend_nbagg.py:459, in FigureManager.__init__(self, canvas, num)
    458 def __init__(self, canvas, num):
--> 459     FigureManagerWebAgg.__init__(self, canvas, num)
    460     self.web_sockets = [self.canvas]
    461     self.toolbar = Toolbar(self.canvas)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/backends/backend_webagg_core.py:434, in FigureManagerWebAgg.__init__(self, canvas, num)
    432 def __init__(self, canvas, num):
    433     self.web_sockets = set()
--> 434     super().__init__(canvas, num)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/backend_bases.py:2796, in FigureManagerBase.__init__(self, canvas, num)
   2791 self.toolmanager = (ToolManager(canvas.figure)
   2792                     if mpl.rcParams['toolbar'] == 'toolmanager'
   2793                     else None)
   2794 if (mpl.rcParams["toolbar"] == "toolbar2"
   2795         and self._toolbar2_class):
-> 2796     self.toolbar = self._toolbar2_class(self.canvas)
   2797 elif (mpl.rcParams["toolbar"] == "toolmanager"
   2798         and self._toolmanager_toolbar_class):
   2799     self.toolbar = self._toolmanager_toolbar_class(self.toolmanager)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/backends/backend_webagg_core.py:397, in NavigationToolbar2WebAgg.__init__(self, canvas)
    395 self.message = ''
    396 self._cursor = None  # Remove with deprecation.
--> 397 super().__init__(canvas)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/matplotlib/backend_bases.py:2938, in NavigationToolbar2.__init__(self, canvas)
   2936 def __init__(self, canvas):
   2937     self.canvas = canvas
-> 2938     canvas.toolbar = self
   2939     self._nav_stack = cbook.Stack()
   2940     # This cursor will be set after the initial draw.

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/traitlets/traitlets.py:712, in TraitType.__set__(self, obj, value)
    710     raise TraitError('The "%s" trait is read-only.' % self.name)
    711 else:
--> 712     self.set(obj, value)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/traitlets/traitlets.py:686, in TraitType.set(self, obj, value)
    685 def set(self, obj, value):
--> 686     new_value = self._validate(obj, value)
    687     try:
    688         old_value = obj._trait_values[self.name]

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/traitlets/traitlets.py:718, in TraitType._validate(self, obj, value)
    716     return value
    717 if hasattr(self, "validate"):
--> 718     value = self.validate(obj, value)  # type:ignore[attr-defined]
    719 if obj._cross_validation_lock is False:
    720     value = self._cross_validate(obj, value)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/traitlets/traitlets.py:2029, in Instance.validate(self, obj, value)
   2027     return value
   2028 else:
-> 2029     self.error(obj, value)

File ~/opt/miniconda3/envs/mne/lib/python3.10/site-packages/traitlets/traitlets.py:824, in TraitType.error(self, obj, value, error, info)
    818 else:
    819     e = "The '{}' trait expected {}, not {}.".format(
    820         self.name,
    821         self.info(),
    822         describe("the", value),
    823     )
--> 824 raise TraitError(e)

TraitError: The 'toolbar' trait of a Canvas instance expected a Toolbar or None, not the NavigationToolbar2WebAgg at '0x10f474310'.
\`\`\`

</details>
I have seen this error and even remember debugging it, but I do not remember if I found the problem or if I switch to an older environment with the plan to come back (because I was under time pressure for something else).  It is quite frustrating....
https://github.com/matplotlib/ipympl/issues/426 and https://github.com/matplotlib/matplotlib/pull/22454 look related, but this has apparent come back?
9369769ea7b4692043233b6f1463326a93120315 via https://github.com/matplotlib/matplotlib/pull/23498 un-did the fix in #22454 .

Looking how to address both this issue and `examples/user_interfaces/embedding_webagg_sgskip.py`
The facts here are:

 - we re-organized the backends to pull as much of the toolbar initialization logic into one place as possible
 - the `Toolbar` class to used is controlled by the `_toolbar2_class` private attribute which if None the backend_bases does not try to make a toolbar
 - `ipympl` inherits from webagg_core and does not (yet) set this private attribute
 - the Canvas in ipympl uses traitlets and type-checks the toolbar to be its toolbar (which is in turn also a `DOMWidget` and has all of the sync mechanics with the js frontend)
 - therefor webagg_core's classes must not set `FigureManager._toolbar2_class` (this is what #22454 did)
 - that fix in turn broke an example which is what #23498 fixed 
```

## Patch

```diff
diff --git a/examples/user_interfaces/embedding_webagg_sgskip.py b/examples/user_interfaces/embedding_webagg_sgskip.py
--- a/examples/user_interfaces/embedding_webagg_sgskip.py
+++ b/examples/user_interfaces/embedding_webagg_sgskip.py
@@ -30,7 +30,7 @@
 
 
 import matplotlib as mpl
-from matplotlib.backends.backend_webagg_core import (
+from matplotlib.backends.backend_webagg import (
     FigureManagerWebAgg, new_figure_manager_given_figure)
 from matplotlib.figure import Figure
 
diff --git a/lib/matplotlib/backends/backend_webagg_core.py b/lib/matplotlib/backends/backend_webagg_core.py
--- a/lib/matplotlib/backends/backend_webagg_core.py
+++ b/lib/matplotlib/backends/backend_webagg_core.py
@@ -427,7 +427,9 @@ def set_history_buttons(self):
 
 
 class FigureManagerWebAgg(backend_bases.FigureManagerBase):
-    _toolbar2_class = ToolbarCls = NavigationToolbar2WebAgg
+    # This must be None to not break ipympl
+    _toolbar2_class = None
+    ToolbarCls = NavigationToolbar2WebAgg
 
     def __init__(self, canvas, num):
         self.web_sockets = set()

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_backend_webagg.py b/lib/matplotlib/tests/test_backend_webagg.py
--- a/lib/matplotlib/tests/test_backend_webagg.py
+++ b/lib/matplotlib/tests/test_backend_webagg.py
@@ -2,6 +2,7 @@
 import os
 import sys
 import pytest
+import matplotlib.backends.backend_webagg_core
 
 
 @pytest.mark.parametrize("backend", ["webagg", "nbagg"])
@@ -25,3 +26,8 @@ def test_webagg_fallback(backend):
     ret = subprocess.call([sys.executable, "-c", test_code], env=env)
 
     assert ret == 0
+
+
+def test_webagg_core_no_toolbar():
+    fm = matplotlib.backends.backend_webagg_core.FigureManagerWebAgg
+    assert fm._toolbar2_class is None

```


## Code snippets

### 1 - lib/matplotlib/backends/backend_macosx.py:

Start line: 105, End line: 142

```python
class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        data_path = cbook._get_data_path('images')
        _, tooltips, image_names, _ = zip(*NavigationToolbar2.toolitems)
        _macosx.NavigationToolbar2.__init__(
            self, canvas,
            tuple(str(data_path / image_name) + ".pdf"
                  for image_name in image_names if image_name is not None),
            tuple(tooltip for tooltip in tooltips if tooltip is not None))
        NavigationToolbar2.__init__(self, canvas)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def remove_rubberband(self):
        self.canvas.remove_rubberband()

    def save_figure(self, *args):
        directory = os.path.expanduser(mpl.rcParams['savefig.directory'])
        filename = _macosx.choose_save_file('Save the figure',
                                            directory,
                                            self.canvas.get_default_filename())
        if filename is None:  # Cancel
            return
        # Save dir for next time, unless empty str (which means use cwd).
        if mpl.rcParams['savefig.directory']:
            mpl.rcParams['savefig.directory'] = os.path.dirname(filename)
        self.canvas.figure.savefig(filename)

    @_api.deprecated("3.6", alternative='configure_subplots()')
    def prepare_configure_subplots(self):
        toolfig = Figure(figsize=(6, 3))
        canvas = FigureCanvasMac(toolfig)
        toolfig.subplots_adjust(top=0.9)
        # Need to keep a reference to the tool.
        _tool = SubplotTool(self.canvas.figure, toolfig)
        return canvas
```
### 2 - lib/matplotlib/backends/backend_webagg_core.py:

Start line: 381, End line: 426

```python
class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):

    # Use the standard toolbar items + download button
    toolitems = [
        (text, tooltip_text, image_file, name_of_method)
        for text, tooltip_text, image_file, name_of_method
        in (*backend_bases.NavigationToolbar2.toolitems,
            ('Download', 'Download plot', 'filesave', 'download'))
        if name_of_method in _ALLOWED_TOOL_ITEMS
    ]

    cursor = _api.deprecate_privatize_attribute("3.5")

    def __init__(self, canvas):
        self.message = ''
        self._cursor = None  # Remove with deprecation.
        super().__init__(canvas)

    def set_message(self, message):
        if message != self.message:
            self.canvas.send_event("message", message=message)
        self.message = message

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.send_event("rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    def remove_rubberband(self):
        self.canvas.send_event("rubberband", x0=-1, y0=-1, x1=-1, y1=-1)

    def save_figure(self, *args):
        """Save the current figure"""
        self.canvas.send_event('save')

    def pan(self):
        super().pan()
        self.canvas.send_event('navigate_mode', mode=self.mode.name)

    def zoom(self):
        super().zoom()
        self.canvas.send_event('navigate_mode', mode=self.mode.name)

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        self.canvas.send_event('history_buttons',
                               Back=can_backward, Forward=can_forward)
```
### 3 - lib/matplotlib/backends/backend_gtk3.py:

Start line: 614, End line: 630

```python
Toolbar = ToolbarGTK3
backend_tools._register_tool_class(
    FigureCanvasGTK3, _backend_gtk.ConfigureSubplotsGTK)
backend_tools._register_tool_class(
    FigureCanvasGTK3, _backend_gtk.RubberbandGTK)


class FigureManagerGTK3(_FigureManagerGTK):
    _toolbar2_class = NavigationToolbar2GTK3
    _toolmanager_toolbar_class = ToolbarGTK3


@_BackendGTK.export
class _BackendGTK3(_BackendGTK):
    FigureCanvas = FigureCanvasGTK3
    FigureManager = FigureManagerGTK3
```
### 4 - lib/matplotlib/backends/backend_gtk3.py:

Start line: 306, End line: 357

```python
class NavigationToolbar2GTK3(_NavigationToolbar2GTK, Gtk.Toolbar):
    @_api.delete_parameter("3.6", "window")
    def __init__(self, canvas, window=None):
        self._win = window
        GObject.GObject.__init__(self)

        self.set_style(Gtk.ToolbarStyle.ICONS)

        self._gtk_ids = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.insert(Gtk.SeparatorToolItem(), -1)
                continue
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(
                    str(cbook._get_data_path('images',
                                             f'{image_file}-symbolic.svg'))),
                Gtk.IconSize.LARGE_TOOLBAR)
            self._gtk_ids[text] = button = (
                Gtk.ToggleToolButton() if callback in ['zoom', 'pan'] else
                Gtk.ToolButton())
            button.set_label(text)
            button.set_icon_widget(image)
            # Save the handler id, so that we can block it as needed.
            button._signal_handler = button.connect(
                'clicked', getattr(self, callback))
            button.set_tooltip_text(tooltip_text)
            self.insert(button, -1)

        # This filler item ensures the toolbar is always at least two text
        # lines high. Otherwise the canvas gets redrawn as the mouse hovers
        # over images because those use two-line messages which resize the
        # toolbar.
        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        label = Gtk.Label()
        label.set_markup(
            '<small>\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}</small>')
        toolitem.set_expand(True)  # Push real message to the right.
        toolitem.add(label)

        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        toolitem.add(self.message)

        self.show_all()

        _NavigationToolbar2GTK.__init__(self, canvas)

    win = _api.deprecated("3.6")(property(lambda self: self._win))
```
### 5 - lib/matplotlib/backends/_backend_tk.py:

Start line: 562, End line: 619

```python
class NavigationToolbar2Tk(NavigationToolbar2, tk.Frame):
    window = _api.deprecated("3.6", alternative="self.master")(
        property(lambda self: self.master))

    def __init__(self, canvas, window=None, *, pack_toolbar=True):
        """
        Parameters
        ----------
        canvas : `FigureCanvas`
            The figure canvas on which to operate.
        window : tk.Window
            The tk.Window which owns this toolbar.
        pack_toolbar : bool, default: True
            If True, add the toolbar to the parent's pack manager's packing
            list during initialization with ``side="bottom"`` and ``fill="x"``.
            If you want to use the toolbar with a different layout manager, use
            ``pack_toolbar=False``.
        """

        if window is None:
            window = canvas.get_tk_widget().master
        tk.Frame.__init__(self, master=window, borderwidth=2,
                          width=int(canvas.figure.bbox.width), height=50)

        self._buttons = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                self._buttons[text] = button = self._Button(
                    text,
                    str(cbook._get_data_path(f"images/{image_file}.png")),
                    toggle=callback in ["zoom", "pan"],
                    command=getattr(self, callback),
                )
                if tooltip_text is not None:
                    ToolTip.createToolTip(button, tooltip_text)

        self._label_font = tkinter.font.Font(root=window, size=10)

        # This filler item ensures the toolbar is always at least two text
        # lines high. Otherwise the canvas gets redrawn as the mouse hovers
        # over images because those use two-line messages which resize the
        # toolbar.
        label = tk.Label(master=self, font=self._label_font,
                         text='\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}')
        label.pack(side=tk.RIGHT)

        self.message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, font=self._label_font,
                                       textvariable=self.message,
                                       justify=tk.RIGHT)
        self._message_label.pack(side=tk.RIGHT)

        NavigationToolbar2.__init__(self, canvas)
        if pack_toolbar:
            self.pack(side=tk.BOTTOM, fill=tk.X)
```
### 6 - examples/user_interfaces/embedding_in_gtk4_panzoom_sgskip.py:

Start line: 1, End line: 52

```python
"""
===========================================
Embedding in GTK4 with a navigation toolbar
===========================================

Demonstrate NavigationToolbar with GTK4 accessed via pygobject.
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from matplotlib.backends.backend_gtk4 import (
    NavigationToolbar2GTK4 as NavigationToolbar)
from matplotlib.backends.backend_gtk4agg import (
    FigureCanvasGTK4Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np


def on_activate(app):
    win = Gtk.ApplicationWindow(application=app)
    win.set_default_size(400, 300)
    win.set_title("Embedding in GTK4")

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    t = np.arange(0.0, 3.0, 0.01)
    s = np.sin(2*np.pi*t)
    ax.plot(t, s)

    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    win.set_child(vbox)

    # Add canvas to vbox
    canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
    canvas.set_hexpand(True)
    canvas.set_vexpand(True)
    vbox.append(canvas)

    # Create toolbar
    toolbar = NavigationToolbar(canvas)
    vbox.append(toolbar)

    win.show()


app = Gtk.Application(
    application_id='org.matplotlib.examples.EmbeddingInGTK4PanZoom')
app.connect('activate', on_activate)
app.run(None)
```
### 7 - examples/user_interfaces/embedding_in_gtk3_panzoom_sgskip.py:

Start line: 1, End line: 44

```python
"""
===========================================
Embedding in GTK3 with a navigation toolbar
===========================================

Demonstrate NavigationToolbar with GTK3 accessed via pygobject.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)
from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit)
win.set_default_size(400, 300)
win.set_title("Embedding in GTK3")

fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(1, 1, 1)
t = np.arange(0.0, 3.0, 0.01)
s = np.sin(2*np.pi*t)
ax.plot(t, s)

vbox = Gtk.VBox()
win.add(vbox)

# Add canvas to vbox
canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
vbox.pack_start(canvas, True, True, 0)

# Create toolbar
toolbar = NavigationToolbar(canvas)
vbox.pack_start(toolbar, False, False, 0)

win.show_all()
Gtk.main()
```
### 8 - lib/matplotlib/backends/backend_gtk4.py:

Start line: 259, End line: 304

```python
class NavigationToolbar2GTK4(_NavigationToolbar2GTK, Gtk.Box):
    @_api.delete_parameter("3.6", "window")
    def __init__(self, canvas, window=None):
        self._win = window
        Gtk.Box.__init__(self)

        self.add_css_class('toolbar')

        self._gtk_ids = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.append(Gtk.Separator())
                continue
            image = Gtk.Image.new_from_gicon(
                Gio.Icon.new_for_string(
                    str(cbook._get_data_path('images',
                                             f'{image_file}-symbolic.svg'))))
            self._gtk_ids[text] = button = (
                Gtk.ToggleButton() if callback in ['zoom', 'pan'] else
                Gtk.Button())
            button.set_child(image)
            button.add_css_class('flat')
            button.add_css_class('image-button')
            # Save the handler id, so that we can block it as needed.
            button._signal_handler = button.connect(
                'clicked', getattr(self, callback))
            button.set_tooltip_text(tooltip_text)
            self.append(button)

        # This filler item ensures the toolbar is always at least two text
        # lines high. Otherwise the canvas gets redrawn as the mouse hovers
        # over images because those use two-line messages which resize the
        # toolbar.
        label = Gtk.Label()
        label.set_markup(
            '<small>\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}</small>')
        label.set_hexpand(True)  # Push real message to the right.
        self.append(label)

        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        self.append(self.message)

        _NavigationToolbar2GTK.__init__(self, canvas)

    win = _api.deprecated("3.6")(property(lambda self: self._win))
```
### 9 - lib/matplotlib/backends/backend_gtk4.py:

Start line: 564, End line: 580

```python
backend_tools._register_tool_class(
    FigureCanvasGTK4, _backend_gtk.ConfigureSubplotsGTK)
backend_tools._register_tool_class(
    FigureCanvasGTK4, _backend_gtk.RubberbandGTK)
Toolbar = ToolbarGTK4


class FigureManagerGTK4(_FigureManagerGTK):
    _toolbar2_class = NavigationToolbar2GTK4
    _toolmanager_toolbar_class = ToolbarGTK4


@_BackendGTK.export
class _BackendGTK4(_BackendGTK):
    FigureCanvas = FigureCanvasGTK4
    FigureManager = FigureManagerGTK4
```
### 10 - lib/matplotlib/backends/backend_nbagg.py:

Start line: 64, End line: 90

```python
class FigureManagerNbAgg(FigureManagerWebAgg):
    _toolbar2_class = ToolbarCls = NavigationIPy

    def __init__(self, canvas, num):
        self._shown = False
        super().__init__(canvas, num)

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        canvas = canvas_class(figure)
        manager = cls(canvas, num)
        if is_interactive():
            manager.show()
            canvas.draw_idle()

        def destroy(event):
            canvas.mpl_disconnect(cid)
            Gcf.destroy(manager)

        cid = canvas.mpl_connect('close_event', destroy)
        return manager

    def display_js(self):
        # XXX How to do this just once? It has to deal with multiple
        # browser instances using the same kernel (require.js - but the
        # file isn't static?).
        display(Javascript(FigureManagerNbAgg.get_javascript()))
```
### 36 - lib/matplotlib/backends/backend_webagg_core.py:

Start line: 157, End line: 200

```python
class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    manager_class = _api.classproperty(lambda cls: FigureManagerWebAgg)
    _timer_cls = TimerAsyncio
    # Webagg and friends having the right methods, but still
    # having bugs in practice.  Do not advertise that it works until
    # we can debug this.
    supports_blit = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set to True when the renderer contains data that is newer
        # than the PNG buffer.
        self._png_is_old = True
        # Set to True by the `refresh` message so that the next frame
        # sent to the clients will be a full frame.
        self._force_full = True
        # The last buffer, for diff mode.
        self._last_buff = np.empty((0, 0))
        # Store the current image mode so that at any point, clients can
        # request the information. This should be changed by calling
        # self.set_image_mode(mode) so that the notification can be given
        # to the connected clients.
        self._current_image_mode = 'full'
        # Track mouse events to fill in the x, y position of key events.
        self._last_mouse_xy = (None, None)

    def show(self):
        # show the figure window
        from matplotlib.pyplot import show
        show()

    def draw(self):
        self._png_is_old = True
        try:
            super().draw()
        finally:
            self.manager.refresh_all()  # Swap the frames.

    def blit(self, bbox=None):
        self._png_is_old = True
        self.manager.refresh_all()

    def draw_idle(self):
        self.send_event("draw")
```
### 38 - lib/matplotlib/backends/backend_webagg_core.py:

Start line: 309, End line: 333

```python
class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    handle_button_press = handle_button_release = handle_dblclick = \
        handle_figure_enter = handle_figure_leave = handle_motion_notify = \
        handle_scroll = _handle_mouse

    def _handle_key(self, event):
        KeyEvent(event['type'] + '_event', self,
                 _handle_key(event['key']), *self._last_mouse_xy,
                 guiEvent=event.get('guiEvent'))._process()
    handle_key_press = handle_key_release = _handle_key

    def handle_toolbar_button(self, event):
        # TODO: Be more suspicious of the input
        getattr(self.toolbar, event['name'])()

    def handle_refresh(self, event):
        figure_label = self.figure.get_label()
        if not figure_label:
            figure_label = "Figure {0}".format(self.manager.num)
        self.send_event('figure_label', label=figure_label)
        self._force_full = True
        if self.toolbar:
            # Normal toolbar init would refresh this, but it happens before the
            # browser canvas is set up.
            self.toolbar.set_history_buttons()
        self.draw_idle()
```
### 51 - lib/matplotlib/backends/backend_webagg_core.py:

Start line: 429, End line: 469

```python
class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    _toolbar2_class = ToolbarCls = NavigationToolbar2WebAgg

    def __init__(self, canvas, num):
        self.web_sockets = set()
        super().__init__(canvas, num)

    def show(self):
        pass

    def resize(self, w, h, forward=True):
        self._send_event(
            'resize',
            size=(w / self.canvas.device_pixel_ratio,
                  h / self.canvas.device_pixel_ratio),
            forward=forward)

    def set_window_title(self, title):
        self._send_event('figure_label', label=title)

    # The following methods are specific to FigureManagerWebAgg

    def add_web_socket(self, web_socket):
        assert hasattr(web_socket, 'send_binary')
        assert hasattr(web_socket, 'send_json')
        self.web_sockets.add(web_socket)
        self.resize(*self.canvas.figure.bbox.size)
        self._send_event('refresh')

    def remove_web_socket(self, web_socket):
        self.web_sockets.remove(web_socket)

    def handle_json(self, content):
        self.canvas.handle_event(content)

    def refresh_all(self):
        if self.web_sockets:
            diff = self.canvas.get_diff_image()
            if diff is not None:
                for s in self.web_sockets:
                    s.send_binary(diff)
```
