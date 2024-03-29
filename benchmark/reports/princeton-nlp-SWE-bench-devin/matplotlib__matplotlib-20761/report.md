# matplotlib__matplotlib-20761

| **matplotlib/matplotlib** | `7413aa92b5be5760c73e31641ab0770f328ad546` |
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
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -370,7 +370,10 @@ def _suplabels(self, t, info, **kwargs):
 
         x = kwargs.pop('x', None)
         y = kwargs.pop('y', None)
-        autopos = x is None and y is None
+        if info['name'] in ['_supxlabel', '_suptitle']:
+            autopos = y is None
+        elif info['name'] == '_supylabel':
+            autopos = x is None
         if x is None:
             x = info['x0']
         if y is None:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/figure.py | 373 | 373 | - | 1 | -


## Problem Statement

```
[Bug]: subfigure position shifts on y-axis when x kwarg added to supxlabel
### Bug summary

Location of subfigure shifts lower on y-axis when 'x' kwarg is used for supxlabel for that subfigure.
I've also posted to StackOverflow: https://stackoverflow.com/q/68567315/9249533

### Code for reproduction

\`\`\`python
fig = plt.figure(constrained_layout=True, figsize=(10, 8))

# create top/bottom subfigs
# see https://stackoverflow.com/a/68553015/9249533
(subfig_t, subfig_b) = fig.subfigures(2, 1, hspace=0.05, height_ratios=[1, 3])

# put ax0 in top subfig
ax0 = subfig_t.subplots()

# create left/right subfigs nested in bottom subfig
(subfig_bl, subfig_br) = subfig_b.subfigures(1, 2, wspace=0.1, width_ratios=[3, 1])

# put ax1-ax3 in gridspec of bottom-left subfig
gs = subfig_bl.add_gridspec(nrows=1, ncols=9)

ax1 = subfig_bl.add_subplot(gs[0, :3])
ax2 = subfig_bl.add_subplot(gs[0, 3:6], sharey=ax1)
ax3 = subfig_bl.add_subplot(gs[0, 6:9], sharey=ax1)


ax1.set_title('Nov. 7 to Nov. 13')
ax2.set_title('Nov. 13 to Nov. 27')
ax3.set_title('Nov. 27 to Dec. 31')
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

subfig_bl.supxlabel("My Subfigure Label", x=0.54, size=12, fontweight='bold')

# put ax4 in bottom-right subfig
ax4 = subfig_br.subplots()
ax4.set_title('Some Other Title')
subfig_br.supxlabel('Other Subfigure SubLabel', size=12, fontweight='bold')
\`\`\`


### Actual outcome

Body of subfigure shifts downward (lower on y-axis) and covers supxlabel

![image](https://user-images.githubusercontent.com/41835370/127401472-20570876-b098-4cc8-bed4-d58d5cfe9669.png)



### Expected outcome

subfigure position doesn't change. supxlabel shifts to right.

![image](https://user-images.githubusercontent.com/41835370/127401167-48803a9c-9d2c-4b52-b109-eec49cdc89de.png)


### Operating system

Windows 10 Pro

### Matplotlib Version

3.4.2

### Matplotlib Backend

_No response_

### Python version

3.9.5

### Jupyter version

3.0.16

### Other libraries

_No response_

### Installation

conda

### Conda channel

conda-forge

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/figure.py** | 414 | 421| 129 | 129 | 26046 | 
| 2 | 2 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 796 | 27469 | 
| 3 | **2 lib/matplotlib/figure.py** | 423 | 431| 137 | 933 | 27469 | 
| 4 | 3 examples/subplots_axes_and_figures/figure_title.py | 1 | 65| 544 | 1477 | 28013 | 
| 5 | 3 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 2233 | 28013 | 
| 6 | 4 examples/subplots_axes_and_figures/subplots_demo.py | 87 | 170| 785 | 3018 | 29980 | 
| 7 | **4 lib/matplotlib/figure.py** | 405 | 412| 132 | 3150 | 29980 | 
| 8 | 5 lib/matplotlib/pyplot.py | 2481 | 2556| 767 | 3917 | 58446 | 
| 9 | 6 tutorials/intermediate/constrainedlayout_guide.py | 196 | 268| 763 | 4680 | 64564 | 
| 10 | 6 examples/subplots_axes_and_figures/subplots_demo.py | 171 | 212| 441 | 5121 | 64564 | 
| 11 | 7 examples/pyplots/align_ylabels.py | 40 | 86| 346 | 5467 | 65198 | 
| 12 | 8 examples/text_labels_and_annotations/label_subplots.py | 1 | 71| 603 | 6070 | 65801 | 
| 13 | **8 lib/matplotlib/figure.py** | 253 | 294| 419 | 6489 | 65801 | 
| 14 | 9 lib/matplotlib/_constrained_layout.py | 275 | 309| 457 | 6946 | 71777 | 
| 15 | 10 tutorials/intermediate/gridspec.py | 137 | 206| 756 | 7702 | 74553 | 
| 16 | **10 lib/matplotlib/figure.py** | 2083 | 2105| 150 | 7852 | 74553 | 
| 17 | 10 lib/matplotlib/pyplot.py | 1212 | 1276| 649 | 8501 | 74553 | 
| 18 | 11 lib/matplotlib/axes/_subplots.py | 115 | 129| 175 | 8676 | 76191 | 
| 19 | 12 examples/lines_bars_and_markers/scatter_hist.py | 53 | 126| 595 | 9271 | 77159 | 
| 20 | 13 examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 128| 766 | 10037 | 78048 | 
| 21 | 14 examples/pyplots/auto_subplots_adjust.py | 1 | 46| 411 | 10448 | 78750 | 
| 22 | 15 examples/showcase/anatomy.py | 90 | 162| 554 | 11002 | 80064 | 
| 23 | 16 examples/subplots_axes_and_figures/broken_axis.py | 1 | 55| 525 | 11527 | 80589 | 
| 24 | 16 tutorials/intermediate/constrainedlayout_guide.py | 122 | 194| 777 | 12304 | 80589 | 
| 25 | **16 lib/matplotlib/figure.py** | 2001 | 2027| 304 | 12608 | 80589 | 
| 26 | 16 lib/matplotlib/pyplot.py | 3149 | 3185| 401 | 13009 | 80589 | 
| 27 | 16 lib/matplotlib/pyplot.py | 2669 | 2680| 146 | 13155 | 80589 | 
| 28 | 16 tutorials/intermediate/constrainedlayout_guide.py | 590 | 674| 815 | 13970 | 80589 | 
| 29 | 16 examples/showcase/anatomy.py | 1 | 89| 760 | 14730 | 80589 | 
| 30 | **16 lib/matplotlib/figure.py** | 1204 | 1263| 538 | 15268 | 80589 | 
| 31 | 17 examples/subplots_axes_and_figures/secondary_axis.py | 102 | 193| 656 | 15924 | 82023 | 
| 32 | 18 tutorials/advanced/transforms_tutorial.py | 233 | 330| 1003 | 16927 | 88124 | 
| 33 | 18 examples/pyplots/auto_subplots_adjust.py | 66 | 85| 140 | 17067 | 88124 | 
| 34 | 19 examples/subplots_axes_and_figures/colorbar_placement.py | 1 | 86| 750 | 17817 | 89010 | 
| 35 | 19 examples/subplots_axes_and_figures/demo_tight_layout.py | 129 | 148| 123 | 17940 | 89010 | 
| 36 | 19 lib/matplotlib/axes/_subplots.py | 131 | 145| 182 | 18122 | 89010 | 
| 37 | 19 lib/matplotlib/_constrained_layout.py | 343 | 406| 705 | 18827 | 89010 | 
| 38 | 20 examples/subplots_axes_and_figures/demo_constrained_layout.py | 1 | 72| 430 | 19257 | 89440 | 
| 39 | 21 tutorials/intermediate/tight_layout_guide.py | 105 | 221| 877 | 20134 | 91590 | 
| 40 | 21 tutorials/intermediate/constrainedlayout_guide.py | 269 | 350| 765 | 20899 | 91590 | 
| 41 | 21 lib/matplotlib/axes/_subplots.py | 1 | 86| 807 | 21706 | 91590 | 
| 42 | 21 tutorials/intermediate/constrainedlayout_guide.py | 352 | 393| 338 | 22044 | 91590 | 
| 43 | 22 examples/subplots_axes_and_figures/align_labels_demo.py | 1 | 38| 294 | 22338 | 91884 | 
| 44 | **22 lib/matplotlib/figure.py** | 1325 | 1347| 169 | 22507 | 91884 | 
| 45 | 22 tutorials/advanced/transforms_tutorial.py | 95 | 231| 1411 | 23918 | 91884 | 
| 46 | 23 examples/axes_grid1/make_room_for_ylabel_using_axesgrid.py | 1 | 60| 401 | 24319 | 92285 | 
| 47 | 24 examples/text_labels_and_annotations/titles_demo.py | 1 | 60| 367 | 24686 | 92652 | 
| 48 | 24 examples/pyplots/auto_subplots_adjust.py | 49 | 63| 151 | 24837 | 92652 | 
| 49 | 25 examples/specialty_plots/leftventricle_bulleye.py | 130 | 195| 733 | 25570 | 94844 | 
| 50 | 26 tutorials/text/text_intro.py | 173 | 261| 771 | 26341 | 98722 | 
| 51 | 26 tutorials/intermediate/constrainedlayout_guide.py | 426 | 588| 1487 | 27828 | 98722 | 
| 52 | 27 examples/ticks_and_spines/tick-formatters.py | 37 | 135| 963 | 28791 | 99957 | 
| 53 | 28 lib/matplotlib/backends/backend_wx.py | 1273 | 1299| 240 | 29031 | 112044 | 
| 54 | 28 tutorials/intermediate/gridspec.py | 1 | 78| 757 | 29788 | 112044 | 
| 55 | 28 lib/matplotlib/axes/_subplots.py | 88 | 113| 285 | 30073 | 112044 | 
| 56 | 29 examples/text_labels_and_annotations/line_with_text.py | 51 | 87| 260 | 30333 | 112636 | 
| 57 | **29 lib/matplotlib/figure.py** | 1265 | 1323| 537 | 30870 | 112636 | 
| 58 | 30 lib/matplotlib/_cm.py | 106 | 156| 787 | 31657 | 141079 | 
| 59 | **30 lib/matplotlib/figure.py** | 1 | 49| 321 | 31978 | 141079 | 
| 60 | 31 examples/widgets/annotated_cursor.py | 284 | 341| 471 | 32449 | 143914 | 
| 61 | 32 examples/shapes_and_collections/fancybox_demo.py | 80 | 127| 501 | 32950 | 145187 | 
| 62 | 32 tutorials/intermediate/gridspec.py | 79 | 136| 719 | 33669 | 145187 | 
| 63 | 32 lib/matplotlib/pyplot.py | 3070 | 3089| 226 | 33895 | 145187 | 
| 64 | 33 examples/subplots_axes_and_figures/axes_box_aspect.py | 111 | 156| 344 | 34239 | 146312 | 
| 65 | **33 lib/matplotlib/figure.py** | 1917 | 1999| 706 | 34945 | 146312 | 
| 66 | 33 examples/subplots_axes_and_figures/subplots_demo.py | 1 | 86| 741 | 35686 | 146312 | 
| 67 | 34 examples/text_labels_and_annotations/angle_annotation.py | 271 | 327| 707 | 36393 | 149765 | 
| 68 | 35 examples/subplots_axes_and_figures/shared_axis_demo.py | 1 | 58| 490 | 36883 | 150255 | 
| 69 | 35 lib/matplotlib/pyplot.py | 2468 | 2478| 137 | 37020 | 150255 | 
| 70 | 36 lib/matplotlib/axis.py | 2195 | 2239| 422 | 37442 | 170463 | 


### Hint

```
This has nothing to do with subfigures, right?  This happens if you specify x or y in supx/ylabel even on a normal figure, I think.  
Not sure.  I've only used suptitles to date.  Will do some more digging.  Cheers

\`\`\`python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(constrained_layout=True)
fig.supxlabel('Boo', x=0.54)
plt.show()
\`\`\`
does the same thing.  I think this is an easy-ish fix, but you'll need a private workaround for now:

\`\`\`python
lab = fig.supxlabel('Boo', x=0.7)
lab._autopos = True
\`\`\`
```

## Patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -370,7 +370,10 @@ def _suplabels(self, t, info, **kwargs):
 
         x = kwargs.pop('x', None)
         y = kwargs.pop('y', None)
-        autopos = x is None and y is None
+        if info['name'] in ['_supxlabel', '_suptitle']:
+            autopos = y is None
+        elif info['name'] == '_supylabel':
+            autopos = x is None
         if x is None:
             x = info['x0']
         if y is None:

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_constrainedlayout.py b/lib/matplotlib/tests/test_constrainedlayout.py
--- a/lib/matplotlib/tests/test_constrainedlayout.py
+++ b/lib/matplotlib/tests/test_constrainedlayout.py
@@ -537,3 +537,26 @@ def test_align_labels():
                                after_align[1].x0, rtol=0, atol=1e-05)
     # ensure labels do not go off the edge
     assert after_align[0].x0 >= 1
+
+
+def test_suplabels():
+    fig, ax = plt.subplots(constrained_layout=True)
+    fig.draw_no_output()
+    pos0 = ax.get_tightbbox(fig.canvas.get_renderer())
+    fig.supxlabel('Boo')
+    fig.supylabel('Booy')
+    fig.draw_no_output()
+    pos = ax.get_tightbbox(fig.canvas.get_renderer())
+    assert pos.y0 > pos0.y0 + 10.0
+    assert pos.x0 > pos0.x0 + 10.0
+
+    fig, ax = plt.subplots(constrained_layout=True)
+    fig.draw_no_output()
+    pos0 = ax.get_tightbbox(fig.canvas.get_renderer())
+    # check that specifying x (y) doesn't ruin the layout
+    fig.supxlabel('Boo', x=0.5)
+    fig.supylabel('Boo', y=0.5)
+    fig.draw_no_output()
+    pos = ax.get_tightbbox(fig.canvas.get_renderer())
+    assert pos.y0 > pos0.y0 + 10.0
+    assert pos.x0 > pos0.x0 + 10.0

```


## Code snippets

### 1 - lib/matplotlib/figure.py:

Start line: 414, End line: 421

```python
class FigureBase(Artist):

    @docstring.Substitution(x0=0.5, y0=0.01, name='supxlabel', ha='center',
                            va='bottom')
    @docstring.copy(_suplabels)
    def supxlabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supxlabel', 'x0': 0.5, 'y0': 0.01,
                'ha': 'center', 'va': 'bottom', 'rotation': 0}
        return self._suplabels(t, info, **kwargs)
```
### 2 - examples/subplots_axes_and_figures/subfigures.py:

Start line: 84, End line: 149

```python
subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

fig.suptitle('Figure suptitle', fontsize='xx-large')
plt.show()

##############################################################################
# Subfigures can have different widths and heights.  This is exactly the
# same example as the first example, but *width_ratios* has been changed:

fig = plt.figure(constrained_layout=True, figsize=(10, 4))
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[2, 1])

axsLeft = subfigs[0].subplots(1, 2, sharey=True)
subfigs[0].set_facecolor('0.75')
for ax in axsLeft:
    pc = example_plot(ax)
subfigs[0].suptitle('Left plots', fontsize='x-large')
subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

axsRight = subfigs[1].subplots(3, 1, sharex=True)
for nn, ax in enumerate(axsRight):
    pc = example_plot(ax, hide_labels=True)
    if nn == 2:
        ax.set_xlabel('xlabel')
    if nn == 1:
        ax.set_ylabel('ylabel')

subfigs[1].set_facecolor('0.85')
subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
subfigs[1].suptitle('Right plots', fontsize='x-large')

fig.suptitle('Figure suptitle', fontsize='xx-large')

plt.show()

##############################################################################
# Subfigures can be also be nested:

fig = plt.figure(constrained_layout=True, figsize=(10, 8))

fig.suptitle('fig')

subfigs = fig.subfigures(1, 2, wspace=0.07)

subfigs[0].set_facecolor('coral')
subfigs[0].suptitle('subfigs[0]')

subfigs[1].set_facecolor('coral')
subfigs[1].suptitle('subfigs[1]')

subfigsnest = subfigs[0].subfigures(2, 1, height_ratios=[1, 1.4])
subfigsnest[0].suptitle('subfigsnest[0]')
subfigsnest[0].set_facecolor('r')
axsnest0 = subfigsnest[0].subplots(1, 2, sharey=True)
for nn, ax in enumerate(axsnest0):
    pc = example_plot(ax, hide_labels=True)
subfigsnest[0].colorbar(pc, ax=axsnest0)

subfigsnest[1].suptitle('subfigsnest[1]')
subfigsnest[1].set_facecolor('g')
axsnest1 = subfigsnest[1].subplots(3, 1, sharex=True)

axsRight = subfigs[1].subplots(2, 2)

plt.show()
```
### 3 - lib/matplotlib/figure.py:

Start line: 423, End line: 431

```python
class FigureBase(Artist):

    @docstring.Substitution(x0=0.02, y0=0.5, name='supylabel', ha='left',
                            va='center')
    @docstring.copy(_suplabels)
    def supylabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supylabel', 'x0': 0.02, 'y0': 0.5,
                'ha': 'left', 'va': 'center', 'rotation': 'vertical',
                'rotation_mode': 'anchor'}
        return self._suplabels(t, info, **kwargs)
```
### 4 - examples/subplots_axes_and_figures/figure_title.py:

Start line: 1, End line: 65

```python
"""
=============================================
Figure labels: suptitle, supxlabel, supylabel
=============================================

Each axes can have a title (or actually three - one each with *loc* "left",
"center", and "right"), but is sometimes desirable to give a whole figure
(or `.SubFigure`) an overall title, using `.FigureBase.suptitle`.

We can also add figure-level x- and y-labels using `.FigureBase.supxlabel` and
`.FigureBase.supylabel`.
"""
from matplotlib.cbook import get_sample_data
import matplotlib.pyplot as plt

import numpy as np


x = np.linspace(0.0, 5.0, 501)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
ax1.plot(x, np.cos(6*x) * np.exp(-x))
ax1.set_title('damped')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('amplitude')

ax2.plot(x, np.cos(6*x))
ax2.set_xlabel('time (s)')
ax2.set_title('undamped')

fig.suptitle('Different types of oscillations', fontsize=16)

##############################################################################
# A global x- or y-label can be set using the `.FigureBase.supxlabel` and
# `.FigureBase.supylabel` methods.

fig, axs = plt.subplots(3, 5, figsize=(8, 5), constrained_layout=True,
                        sharex=True, sharey=True)

fname = get_sample_data('percent_bachelors_degrees_women_usa.csv',
                        asfileobj=False)
gender_degree_data = np.genfromtxt(fname, delimiter=',', names=True)

majors = ['Health Professions', 'Public Administration', 'Education',
          'Psychology', 'Foreign Languages', 'English',
          'Art and Performance', 'Biology',
          'Agriculture', 'Business',
          'Math and Statistics', 'Architecture', 'Physical Sciences',
          'Computer Science', 'Engineering']

for nn, ax in enumerate(axs.flat):
    ax.set_xlim(1969.5, 2011.1)
    column = majors[nn]
    column_rec_name = column.replace('\n', '_').replace(' ', '_')

    line, = ax.plot('Year', column_rec_name, data=gender_degree_data,
                    lw=2.5)
    ax.set_title(column, fontsize='small', loc='left')
    ax.set_ylim([0, 100])
    ax.grid()
fig.supxlabel('Year')
fig.supylabel('Percent Degrees Awarded To Women')

plt.show()
```
### 5 - examples/subplots_axes_and_figures/subfigures.py:

Start line: 1, End line: 83

```python
"""
=================
Figure subfigures
=================

Sometimes it is desirable to have a figure with two different layouts in it.
This can be achieved with
:doc:`nested gridspecs</gallery/subplots_axes_and_figures/gridspec_nested>`,
but having a virtual figure with its own artists is helpful, so
Matplotlib also has "subfigures", accessed by calling
`matplotlib.figure.Figure.add_subfigure` in a way that is analogous to
`matplotlib.figure.Figure.add_subplot`, or
`matplotlib.figure.Figure.subfigures` to make an array of subfigures.  Note
that subfigures can also have their own child subfigures.

.. note::
    ``subfigure`` is new in v3.4, and the API is still provisional.

"""
import matplotlib.pyplot as plt
import numpy as np


def example_plot(ax, fontsize=12, hide_labels=False):
    pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    return pc

np.random.seed(19680808)
# gridspec inside gridspec
fig = plt.figure(constrained_layout=True, figsize=(10, 4))
subfigs = fig.subfigures(1, 2, wspace=0.07)

axsLeft = subfigs[0].subplots(1, 2, sharey=True)
subfigs[0].set_facecolor('0.75')
for ax in axsLeft:
    pc = example_plot(ax)
subfigs[0].suptitle('Left plots', fontsize='x-large')
subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

axsRight = subfigs[1].subplots(3, 1, sharex=True)
for nn, ax in enumerate(axsRight):
    pc = example_plot(ax, hide_labels=True)
    if nn == 2:
        ax.set_xlabel('xlabel')
    if nn == 1:
        ax.set_ylabel('ylabel')

subfigs[1].set_facecolor('0.85')
subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
subfigs[1].suptitle('Right plots', fontsize='x-large')

fig.suptitle('Figure suptitle', fontsize='xx-large')

plt.show()

##############################################################################
# It is possible to mix subplots and subfigures using
# `matplotlib.figure.Figure.add_subfigure`.  This requires getting
# the gridspec that the subplots are laid out on.

fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 4))
gridspec = axs[0, 0].get_subplotspec().get_gridspec()

# clear the left column for the subfigure:
for a in axs[:, 0]:
    a.remove()

# plot data in remaining axes:
for a in axs[:, 1:].flat:
    a.plot(np.arange(10))

# make the subfigure in the empty gridspec slots:
subfig = fig.add_subfigure(gridspec[:, 0])

axsLeft = subfig.subplots(1, 2, sharey=True)
subfig.set_facecolor('0.75')
for ax in axsLeft:
    pc = example_plot(ax)
subfig.suptitle('Left plots', fontsize='x-large')
```
### 6 - examples/subplots_axes_and_figures/subplots_demo.py:

Start line: 87, End line: 170

```python
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

###############################################################################
# You can use tuple-unpacking also in 2D to assign all subplots to dedicated
# variables:

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y)
ax2.plot(x, y**2, 'tab:orange')
ax3.plot(x, -y, 'tab:green')
ax4.plot(x, -y**2, 'tab:red')

for ax in fig.get_axes():
    ax.label_outer()

###############################################################################
# Sharing axes
# """"""""""""
#
# By default, each Axes is scaled individually. Thus, if the ranges are
# different the tick values of the subplots do not align.

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Axes values are scaled individually by default')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

###############################################################################
# You can use *sharex* or *sharey* to align the horizontal or vertical axis.

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

###############################################################################
# Setting *sharex* or *sharey* to ``True`` enables global sharing across the
# whole grid, i.e. also the y-axes of vertically stacked subplots have the
# same scale when using ``sharey=True``.

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

###############################################################################
# For subplots that are sharing axes one set of tick labels is enough. Tick
# labels of inner Axes are automatically removed by *sharex* and *sharey*.
# Still there remains an unused empty space between the subplots.
#
# To precisely control the positioning of the subplots, one can explicitly
# create a `.GridSpec` with `.Figure.add_gridspec`, and then call its
# `~.GridSpecBase.subplots` method.  For example, we can reduce the height
# between vertical subplots using ``add_gridspec(hspace=0)``.
#
# `.label_outer` is a handy method to remove labels and ticks from subplots
# that are not at the edge of the grid.

fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[1].plot(x, 0.3 * y, 'o')
axs[2].plot(x, y, '+')

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

###############################################################################
# Apart from ``True`` and ``False``, both *sharex* and *sharey* accept the
# values 'row' and 'col' to share the values only per row or column.

fig = plt.figure()
```
### 7 - lib/matplotlib/figure.py:

Start line: 405, End line: 412

```python
class FigureBase(Artist):

    @docstring.Substitution(x0=0.5, y0=0.98, name='suptitle', ha='center',
                            va='top')
    @docstring.copy(_suplabels)
    def suptitle(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.98,
                'ha': 'center', 'va': 'top', 'rotation': 0}
        return self._suplabels(t, info, **kwargs)
```
### 8 - lib/matplotlib/pyplot.py:

Start line: 2481, End line: 2556

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.text)
def figtext(x, y, s, fontdict=None, **kwargs):
    return gcf().text(x, y, s, fontdict=fontdict, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.gca)
def gca(**kwargs):
    return gcf().gca(**kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure._gci)
def gci():
    return gcf()._gci()


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.ginput)
def ginput(
        n=1, timeout=30, show_clicks=True,
        mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.RIGHT,
        mouse_stop=MouseButton.MIDDLE):
    return gcf().ginput(
        n=n, timeout=timeout, show_clicks=show_clicks,
        mouse_add=mouse_add, mouse_pop=mouse_pop,
        mouse_stop=mouse_stop)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.subplots_adjust)
def subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=None,
        hspace=None):
    return gcf().subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, wspace=wspace,
        hspace=hspace)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.suptitle)
def suptitle(t, **kwargs):
    return gcf().suptitle(t, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.tight_layout)
def tight_layout(*, pad=1.08, h_pad=None, w_pad=None, rect=None):
    return gcf().tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.waitforbuttonpress)
def waitforbuttonpress(timeout=-1):
    return gcf().waitforbuttonpress(timeout=timeout)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.acorr)
def acorr(x, *, data=None, **kwargs):
    return gca().acorr(
        x, **({"data": data} if data is not None else {}), **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.angle_spectrum)
def angle_spectrum(
        x, Fs=None, Fc=None, window=None, pad_to=None, sides=None, *,
        data=None, **kwargs):
    return gca().angle_spectrum(
        x, Fs=Fs, Fc=Fc, window=window, pad_to=pad_to, sides=sides,
        **({"data": data} if data is not None else {}), **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
```
### 9 - tutorials/intermediate/constrainedlayout_guide.py:

Start line: 196, End line: 268

```python
axs[0].plot(np.arange(10))
axs[1].plot(np.arange(10), label='This is a plot')
leg = axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
leg.set_in_layout(False)
# trigger a draw so that constrained_layout is executed once
# before we turn it off when printing....
fig.canvas.draw()
# we want the legend included in the bbox_inches='tight' calcs.
leg.set_in_layout(True)
# we don't want the layout to change at this point.
fig.set_constrained_layout(False)
fig.savefig('../../doc/_static/constrained_layout_1b.png',
            bbox_inches='tight', dpi=100)

#############################################
# The saved file looks like:
#
# .. image:: /_static/constrained_layout_1b.png
#    :align: center
#
# A better way to get around this awkwardness is to simply
# use the legend method provided by `.Figure.legend`:
fig, axs = plt.subplots(1, 2, figsize=(4, 2), constrained_layout=True)
axs[0].plot(np.arange(10))
lines = axs[1].plot(np.arange(10), label='This is a plot')
labels = [l.get_label() for l in lines]
leg = fig.legend(lines, labels, loc='center left',
                 bbox_to_anchor=(0.8, 0.5), bbox_transform=axs[1].transAxes)
fig.savefig('../../doc/_static/constrained_layout_2b.png',
            bbox_inches='tight', dpi=100)

#############################################
# The saved file looks like:
#
# .. image:: /_static/constrained_layout_2b.png
#    :align: center
#

###############################################################################
# Padding and Spacing
# ===================
#
# Padding between axes is controlled in the horizontal by *w_pad* and
# *wspace*, and vertical by *h_pad* and *hspace*.  These can be edited
# via `~.figure.Figure.set_constrained_layout_pads`.  *w/h_pad* are
# the minimum space around the axes in units of inches:

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)

##########################################
# Spacing between subplots is further set by *wspace* and *hspace*. These
# are specified as a fraction of the size of the subplot group as a whole.
# If these values are smaller than *w_pad* or *h_pad*, then the fixed pads are
# used instead. Note in the below how the space at the edges doesn't change
# from the above, but the space between subplots does.

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                                wspace=0.2)

##########################################
# If there are more than two columns, the *wspace* is shared between them,
# so here the wspace is divided in 2, with a *wspace* of 0.1 between each
# column:

fig, axs = plt.subplots(2, 3, constrained_layout=True)
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
```
### 10 - examples/subplots_axes_and_figures/subplots_demo.py:

Start line: 171, End line: 212

```python
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y)
ax2.plot(x, y**2, 'tab:orange')
ax3.plot(x + 1, -y, 'tab:green')
ax4.plot(x + 2, -y**2, 'tab:red')

for ax in axs.flat:
    ax.label_outer()

###############################################################################
# If you want a more complex sharing structure, you can first create the
# grid of axes with no sharing, and then call `.axes.Axes.sharex` or
# `.axes.Axes.sharey` to add sharing info a posteriori.

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title("main")
axs[1, 0].plot(x, y**2)
axs[1, 0].set_title("shares x with main")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(x + 1, y + 1)
axs[0, 1].set_title("unrelated")
axs[1, 1].plot(x + 2, y + 2)
axs[1, 1].set_title("also unrelated")
fig.tight_layout()

###############################################################################
# Polar axes
# """"""""""
#
# The parameter *subplot_kw* of `.pyplot.subplots` controls the subplot
# properties (see also `.Figure.add_subplot`). In particular, this can be used
# to create a grid of polar Axes.

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax1.plot(x, y)
ax2.plot(x, y ** 2)

plt.show()
```
### 13 - lib/matplotlib/figure.py:

Start line: 253, End line: 294

```python
class FigureBase(Artist):

    def autofmt_xdate(
            self, bottom=0.2, rotation=30, ha='right', which='major'):
        """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared x-axis where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.

        Parameters
        ----------
        bottom : float, default: 0.2
            The bottom of the subplots for `subplots_adjust`.
        rotation : float, default: 30 degrees
            The rotation angle of the xtick labels in degrees.
        ha : {'left', 'center', 'right'}, default: 'right'
            The horizontal alignment of the xticklabels.
        which : {'major', 'minor', 'both'}, default: 'major'
            Selects which ticklabels to rotate.
        """
        _api.check_in_list(['major', 'minor', 'both'], which=which)
        allsubplots = all(hasattr(ax, 'get_subplotspec') for ax in self.axes)
        if len(self.axes) == 1:
            for label in self.axes[0].get_xticklabels(which=which):
                label.set_ha(ha)
                label.set_rotation(rotation)
        else:
            if allsubplots:
                for ax in self.get_axes():
                    if ax.get_subplotspec().is_last_row():
                        for label in ax.get_xticklabels(which=which):
                            label.set_ha(ha)
                            label.set_rotation(rotation)
                    else:
                        for label in ax.get_xticklabels(which=which):
                            label.set_visible(False)
                        ax.set_xlabel('')

        if allsubplots:
            self.subplots_adjust(bottom=bottom)
        self.stale = True
```
### 16 - lib/matplotlib/figure.py:

Start line: 2083, End line: 2105

```python
class SubFigure(FigureBase):

    get_axes = axes.fget

    def draw(self, renderer):
        # docstring inherited
        self._cachedRenderer = renderer

        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible():
            return

        artists = self._get_draw_artists(renderer)

        try:
            renderer.open_group('subfigure', gid=self.get_gid())
            self.patch.draw(renderer)
            mimage._draw_list_compositing_images(
                renderer, self, artists, self.figure.suppressComposite)
            for sfig in self.subfigs:
                sfig.draw(renderer)
            renderer.close_group('subfigure')

        finally:
            self.stale = False
```
### 25 - lib/matplotlib/figure.py:

Start line: 2001, End line: 2027

```python
class SubFigure(FigureBase):

    def _redo_transform_rel_fig(self, bbox=None):
        """
        Make the transSubfigure bbox relative to Figure transform.

        Parameters
        ----------
        bbox : bbox or None
            If not None, then the bbox is used for relative bounding box.
            Otherwise it is calculated from the subplotspec.
        """
        if bbox is not None:
            self.bbox_relative.p0 = bbox.p0
            self.bbox_relative.p1 = bbox.p1
            return
        # need to figure out *where* this subplotspec is.
        gs = self._subplotspec.get_gridspec()
        wr = np.asarray(gs.get_width_ratios())
        hr = np.asarray(gs.get_height_ratios())
        dx = wr[self._subplotspec.colspan].sum() / wr.sum()
        dy = hr[self._subplotspec.rowspan].sum() / hr.sum()
        x0 = wr[:self._subplotspec.colspan.start].sum() / wr.sum()
        y0 = 1 - hr[:self._subplotspec.rowspan.stop].sum() / hr.sum()
        if self.bbox_relative is None:
            self.bbox_relative = Bbox.from_bounds(x0, y0, dx, dy)
        else:
            self.bbox_relative.p0 = (x0, y0)
            self.bbox_relative.p1 = (x0 + dx, y0 + dy)
```
### 30 - lib/matplotlib/figure.py:

Start line: 1204, End line: 1263

```python
class FigureBase(Artist):

    def align_xlabels(self, axs=None):
        """
        Align the xlabels of subplots in the same subplot column if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the bottom, it is aligned with labels on Axes that
        also have their label on the bottom and that have the same
        bottom-most subplot row.  If the label is on the top,
        it is aligned with labels on Axes with the same top-most row.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list of (or ndarray) `~matplotlib.axes.Axes`
            to align the xlabels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with rotated xtick labels::

            fig, axs = plt.subplots(1, 2)
            for tick in axs[0].get_xticklabels():
                tick.set_rotation(55)
            axs[0].set_xlabel('XLabel 0')
            axs[1].set_xlabel('XLabel 1')
            fig.align_xlabels()
        """
        if axs is None:
            axs = self.axes
        axs = np.ravel(axs)
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_xlabel())
            rowspan = ax.get_subplotspec().rowspan
            pos = ax.xaxis.get_label_position()  # top or bottom
            # Search through other axes for label positions that are same as
            # this one and that share the appropriate row number.
            # Add to a grouper associated with each axes of siblings.
            # This list is inspected in `axis.draw` by
            # `axis._update_label_position`.
            for axc in axs:
                if axc.xaxis.get_label_position() == pos:
                    rowspanc = axc.get_subplotspec().rowspan
                    if (pos == 'top' and rowspan.start == rowspanc.start or
                            pos == 'bottom' and rowspan.stop == rowspanc.stop):
                        # grouper for groups of xlabels to align
                        self._align_label_groups['x'].join(ax, axc)
```
### 44 - lib/matplotlib/figure.py:

Start line: 1325, End line: 1347

```python
class FigureBase(Artist):

    def align_labels(self, axs=None):
        """
        Align the xlabels and ylabels of subplots with the same subplots
        row or column (respectively) if label alignment is being
        done automatically (i.e. the label position is not manually set).

        Alignment persists for draw events after this is called.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or ndarray) of `~matplotlib.axes.Axes`
            to align the labels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels

        matplotlib.figure.Figure.align_ylabels
        """
        self.align_xlabels(axs=axs)
        self.align_ylabels(axs=axs)
```
### 57 - lib/matplotlib/figure.py:

Start line: 1265, End line: 1323

```python
class FigureBase(Artist):

    def align_ylabels(self, axs=None):
        """
        Align the ylabels of subplots in the same subplot column if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the left, it is aligned with labels on Axes that
        also have their label on the left and that have the same
        left-most subplot column.  If the label is on the right,
        it is aligned with labels on Axes with the same right-most column.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or ndarray) of `~matplotlib.axes.Axes`
            to align the ylabels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with large yticks labels::

            fig, axs = plt.subplots(2, 1)
            axs[0].plot(np.arange(0, 1000, 50))
            axs[0].set_ylabel('YLabel 0')
            axs[1].set_ylabel('YLabel 1')
            fig.align_ylabels()
        """
        if axs is None:
            axs = self.axes
        axs = np.ravel(axs)
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_ylabel())
            colspan = ax.get_subplotspec().colspan
            pos = ax.yaxis.get_label_position()  # left or right
            # Search through other axes for label positions that are same as
            # this one and that share the appropriate column number.
            # Add to a list associated with each axes of siblings.
            # This list is inspected in `axis.draw` by
            # `axis._update_label_position`.
            for axc in axs:
                if axc.yaxis.get_label_position() == pos:
                    colspanc = axc.get_subplotspec().colspan
                    if (pos == 'left' and colspan.start == colspanc.start or
                            pos == 'right' and colspan.stop == colspanc.stop):
                        # grouper for groups of ylabels to align
                        self._align_label_groups['y'].join(ax, axc)
```
### 59 - lib/matplotlib/figure.py:

Start line: 1, End line: 49

```python
"""
`matplotlib.figure` implements the following classes:

`Figure`
    Top level `~matplotlib.artist.Artist`, which holds all plot elements.
    Many methods are implemented in `FigureBase`.

`SubFigure`
    A logical figure inside a figure, usually added to a figure (or parent
    `SubFigure`) with `Figure.add_subfigure` or `Figure.subfigures` methods
    (provisional API v3.4).

`SubplotParams`
    Control the default spacing between subplots.
"""

from contextlib import ExitStack
import inspect
import logging
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _blocking_input, docstring, projections
from matplotlib.artist import (
    Artist, allow_rasterization, _finalize_rasterization)
from matplotlib.backend_bases import (
    FigureCanvasBase, NonGuiException, MouseButton, _get_renderer)
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage

from matplotlib.axes import Axes, SubplotBase, subplot_class_factory
from matplotlib.gridspec import GridSpec
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)
import matplotlib._layoutgrid as layoutgrid

_log = logging.getLogger(__name__)


def _stale_figure_callback(self, val):
    if self.figure:
        self.figure.stale = val
```
### 65 - lib/matplotlib/figure.py:

Start line: 1917, End line: 1999

```python
class SubFigure(FigureBase):
    """
    Logical figure that can be placed inside a figure.

    Typically instantiated using `.Figure.add_subfigure` or
    `.SubFigure.add_subfigure`, or `.SubFigure.subfigures`.  A subfigure has
    the same methods as a figure except for those particularly tied to the size
    or dpi of the figure, and is confined to a prescribed region of the figure.
    For example the following puts two subfigures side-by-side::

        fig = plt.figure()
        sfigs = fig.subfigures(1, 2)
        axsL = sfigs[0].subplots(1, 2)
        axsR = sfigs[1].subplots(2, 1)

    See :doc:`/gallery/subplots_axes_and_figures/subfigures`
    """

    def __init__(self, parent, subplotspec, *,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None):
        """
        Parameters
        ----------
        parent : `.figure.Figure` or `.figure.SubFigure`
            Figure or subfigure that contains the SubFigure.  SubFigures
            can be nested.

        subplotspec : `.gridspec.SubplotSpec`
            Defines the region in a parent gridspec where the subfigure will
            be placed.

        facecolor : default: :rc:`figure.facecolor`
            The figure patch face color.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure background patch.
        """
        super().__init__()
        if facecolor is None:
            facecolor = mpl.rcParams['figure.facecolor']
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        self._subplotspec = subplotspec
        self._parent = parent
        self.figure = parent.figure
        # subfigures use the parent axstack
        self._axstack = parent._axstack
        self.subplotpars = parent.subplotpars
        self.dpi_scale_trans = parent.dpi_scale_trans
        self._axobservers = parent._axobservers
        self.dpi = parent.dpi
        self.canvas = parent.canvas
        self.transFigure = parent.transFigure
        self.bbox_relative = None
        self._redo_transform_rel_fig()
        self.figbbox = self._parent.figbbox
        self.bbox = TransformedBbox(self.bbox_relative,
                                    self._parent.transSubfigure)
        self.transSubfigure = BboxTransformTo(self.bbox)

        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1, visible=frameon,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            # Don't let the figure patch influence bbox calculation.
            in_layout=False, transform=self.transSubfigure)
        self._set_artist_props(self.patch)
        self.patch.set_antialiased(False)

        if parent._layoutgrid is not None:
            self.init_layoutgrid()
```
