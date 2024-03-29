# matplotlib__matplotlib-23266

| **matplotlib/matplotlib** | `dab648ac5eff66a39742f718a356ebe250e01880` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 867 |
| **Avg pos** | 25.0 |
| **Min pos** | 3 |
| **Max pos** | 11 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/contour.py b/lib/matplotlib/contour.py
--- a/lib/matplotlib/contour.py
+++ b/lib/matplotlib/contour.py
@@ -701,7 +701,7 @@ def __init__(self, ax, *args,
                  hatches=(None,), alpha=None, origin=None, extent=None,
                  cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                  extend='neither', antialiased=None, nchunk=0, locator=None,
-                 transform=None,
+                 transform=None, negative_linestyles=None,
                  **kwargs):
         """
         Draw contour lines or filled regions, depending on
@@ -786,6 +786,13 @@ def __init__(self, ax, *args,
 
         self._transform = transform
 
+        self.negative_linestyles = negative_linestyles
+        # If negative_linestyles was not defined as a kwarg,
+        # define negative_linestyles with rcParams
+        if self.negative_linestyles is None:
+            self.negative_linestyles = \
+                mpl.rcParams['contour.negative_linestyle']
+
         kwargs = self._process_args(*args, **kwargs)
         self._process_levels()
 
@@ -1276,11 +1283,10 @@ def _process_linestyles(self):
         if linestyles is None:
             tlinestyles = ['solid'] * Nlev
             if self.monochrome:
-                neg_ls = mpl.rcParams['contour.negative_linestyle']
                 eps = - (self.zmax - self.zmin) * 1e-15
                 for i, lev in enumerate(self.levels):
                     if lev < eps:
-                        tlinestyles[i] = neg_ls
+                        tlinestyles[i] = self.negative_linestyles
         else:
             if isinstance(linestyles, str):
                 tlinestyles = [linestyles] * Nlev
@@ -1751,6 +1757,18 @@ def _initialize_x_y(self, z):
     iterable is shorter than the number of contour levels
     it will be repeated as necessary.
 
+negative_linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, \
+                       optional
+    *Only applies to* `.contour`.
+
+    If *negative_linestyles* is None, the default is 'dashed' for
+    negative contours.
+
+    *negative_linestyles* can also be an iterable of the above
+    strings specifying a set of linestyles to be used. If this
+    iterable is shorter than the number of contour levels
+    it will be repeated as necessary.
+
 hatches : list[str], optional
     *Only applies to* `.contourf`.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/contour.py | 704 | 704 | 11 | 1 | 6374
| lib/matplotlib/contour.py | 789 | 789 | 11 | 1 | 6374
| lib/matplotlib/contour.py | 1279 | 1283 | 3 | 1 | 867
| lib/matplotlib/contour.py | 1754 | 1754 | - | 1 | -


## Problem Statement

```
[ENH]: contour kwarg for negative_linestyle
### Problem

if you contour a negative quantity, it gets dashed lines.  Leaving aside whether this is a good default or not, the only way to toggle this is via `rcParams['contour.negative_linestyle']=False`.  


### Proposed solution


I think this should be togglable via kwarg, though I appreciate that overlaps with `linestyle` and only is activated with monochrome contours.  

(I actually think the default should be False, FWIW - this surprises me every time, and I make quite a few contour plots).  

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/contour.py** | 1252 | 1271| 226 | 226 | 16683 | 
| 2 | 2 lib/matplotlib/tri/tricontour.py | 218 | 244| 279 | 505 | 19313 | 
| **-> 3 <-** | **2 lib/matplotlib/contour.py** | 1273 | 1308| 362 | 867 | 19313 | 
| 4 | **2 lib/matplotlib/contour.py** | 849 | 905| 644 | 1511 | 19313 | 
| 5 | 2 lib/matplotlib/tri/tricontour.py | 84 | 215| 1443 | 2954 | 19313 | 
| 6 | 3 examples/images_contours_and_fields/contourf_demo.py | 96 | 129| 338 | 3292 | 20430 | 
| 7 | 4 lib/matplotlib/pyplot.py | 2438 | 2455| 196 | 3488 | 48046 | 
| 8 | **4 lib/matplotlib/contour.py** | 538 | 597| 475 | 3963 | 48046 | 
| 9 | **4 lib/matplotlib/contour.py** | 1458 | 1482| 266 | 4229 | 48046 | 
| 10 | 5 examples/images_contours_and_fields/contour_demo.py | 1 | 83| 749 | 4978 | 49136 | 
| **-> 11 <-** | **5 lib/matplotlib/contour.py** | 699 | 847| 1396 | 6374 | 49136 | 
| 12 | **5 lib/matplotlib/contour.py** | 1069 | 1095| 293 | 6667 | 49136 | 
| 13 | 5 examples/images_contours_and_fields/contourf_demo.py | 1 | 95| 779 | 7446 | 49136 | 
| 14 | **5 lib/matplotlib/contour.py** | 388 | 418| 385 | 7831 | 49136 | 
| 15 | 5 examples/images_contours_and_fields/contour_demo.py | 84 | 122| 341 | 8172 | 49136 | 
| 16 | 5 lib/matplotlib/tri/tricontour.py | 53 | 81| 299 | 8471 | 49136 | 
| 17 | 6 examples/images_contours_and_fields/contour_corner_mask.py | 1 | 49| 370 | 8841 | 49506 | 
| 18 | 7 examples/images_contours_and_fields/colormap_normalizations_symlognorm.py | 1 | 85| 820 | 9661 | 50326 | 
| 19 | **7 lib/matplotlib/contour.py** | 636 | 652| 123 | 9784 | 50326 | 
| 20 | 8 examples/images_contours_and_fields/contourf_log.py | 1 | 62| 492 | 10276 | 50818 | 
| 21 | 9 examples/misc/contour_manual.py | 1 | 58| 621 | 10897 | 51439 | 
| 22 | 10 lib/matplotlib/axes/_axes.py | 6231 | 6820| 1195 | 12092 | 125169 | 
| 23 | 11 examples/images_contours_and_fields/contour_image.py | 1 | 76| 763 | 12855 | 126246 | 
| 24 | 12 examples/specialty_plots/advanced_hillshading.py | 33 | 53| 231 | 13086 | 126856 | 
| 25 | 12 examples/images_contours_and_fields/contour_image.py | 77 | 109| 314 | 13400 | 126856 | 
| 26 | **12 lib/matplotlib/contour.py** | 240 | 261| 246 | 13646 | 126856 | 
| 27 | **12 lib/matplotlib/contour.py** | 1207 | 1250| 426 | 14072 | 126856 | 
| 28 | **12 lib/matplotlib/contour.py** | 47 | 73| 309 | 14381 | 126856 | 
| 29 | 13 lib/matplotlib/colorbar.py | 67 | 118| 549 | 14930 | 141312 | 
| 30 | 14 plot_types/arrays/contourf.py | 1 | 24| 149 | 15079 | 141461 | 
| 31 | 15 tutorials/introductory/customizing.py | 1 | 113| 892 | 15971 | 143681 | 
| 32 | 15 lib/matplotlib/tri/tricontour.py | 247 | 271| 210 | 16181 | 143681 | 
| 33 | 16 examples/images_contours_and_fields/colormap_normalizations.py | 108 | 146| 427 | 16608 | 145188 | 
| 34 | **16 lib/matplotlib/contour.py** | 1037 | 1050| 151 | 16759 | 145188 | 
| 35 | 17 tutorials/colors/colormapnorms.py | 92 | 168| 831 | 17590 | 148878 | 
| 36 | **17 lib/matplotlib/contour.py** | 167 | 238| 712 | 18302 | 148878 | 
| 37 | 18 examples/images_contours_and_fields/contour_label_demo.py | 1 | 87| 613 | 18915 | 149491 | 
| 38 | 19 examples/images_contours_and_fields/contourf_hatching.py | 1 | 56| 476 | 19391 | 149967 | 
| 39 | 20 lib/matplotlib/lines.py | 1035 | 1083| 513 | 19904 | 161983 | 
| 40 | 21 examples/misc/customize_rc.py | 1 | 60| 446 | 20350 | 162429 | 
| 41 | 21 examples/images_contours_and_fields/colormap_normalizations.py | 1 | 75| 752 | 21102 | 162429 | 
| 42 | 22 examples/images_contours_and_fields/image_transparency_blend.py | 97 | 125| 235 | 21337 | 163477 | 
| 43 | 23 examples/images_contours_and_fields/contours_in_optimization_demo.py | 1 | 65| 605 | 21942 | 164082 | 
| 44 | **23 lib/matplotlib/contour.py** | 1097 | 1138| 385 | 22327 | 164082 | 
| 45 | 24 examples/text_labels_and_annotations/unicode_minus.py | 1 | 29| 252 | 22579 | 164334 | 
| 46 | **24 lib/matplotlib/contour.py** | 1 | 44| 290 | 22869 | 164334 | 
| 47 | 25 examples/lines_bars_and_markers/linestyles.py | 1 | 41| 543 | 23412 | 165233 | 
| 48 | 25 examples/images_contours_and_fields/colormap_normalizations.py | 92 | 105| 152 | 23564 | 165233 | 
| 49 | 26 lib/matplotlib/rcsetup.py | 452 | 486| 331 | 23895 | 177080 | 
| 50 | **26 lib/matplotlib/contour.py** | 1389 | 1456| 570 | 24465 | 177080 | 
| 51 | **26 lib/matplotlib/contour.py** | 984 | 1016| 336 | 24801 | 177080 | 
| 52 | 27 examples/images_contours_and_fields/irregulardatagrid.py | 1 | 78| 764 | 25565 | 177988 | 


### Hint

```
Should the current `linestyles` kwarg be used to accomplish this or should a new kwarg be added? I have a simple solution adding a new kwarg (though it would need a little more work before it is ready).

The following code snippet and images will show a solution with an added kwarg, but I expect this is not exactly what you had in mind.


\`\`\`
import numpy as np
import matplotlib.pyplot as plt

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# Negative contour defaults to dashed
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours dashed (default)')

# Set negative contours to be solid instead of dashed (default)
# using negative_linestyle='solid'
fig2, ax2 = plt.subplots()
CS = ax2.contour(X, Y, Z, 6, colors='k', negative_linestyles='solid')
ax2.clabel(CS, fontsize=9, inline=True)
ax2.set_title('Single color - negative contours solid')
\`\`\`

### Default
![image](https://user-images.githubusercontent.com/28690153/168208633-7e0ca845-e218-445c-a19a-850f279a15ec.png)

### `negative_linestyles='solid'`
![image](https://user-images.githubusercontent.com/28690153/168208620-88c3a5eb-7bb4-4f6a-8890-b23c76a30928.png)
> Should the current `linestyles` kwarg be used to accomplish this or should a new kwarg be added?

How would we squeeze this into the current `linestyles` parameter? The only thing I could imagine is something like `linestyles={'positive': 'solid', 'negative': 'dotted'}`, which I find a bit cumbersome. I suppose an extra kwarg is simpler.
> How would we squeeze this into the current linestyles parameter?

I had no good solution for this either, but thought I would ask just in case.

The next question is: 
1. should this be a boolean (the rcParams method seems to be)
2. or should the kwarg accept a separate entry for negative linestyles?

I think the latter makes a lot more sense, but I don't work with contour plots regularly.
I think since the rcParam is separate, its fine to keep the (potential) kwarg separate.  BTW I'm not wedded to this idea, but it does seem strange to have a feature only accessible by rcParam.  

As for bool vs string, I'd say both?   right now the rcParam defaults to "dashed" - I think that's fine for a kwarg as well, but being able to say *False* to skip the behaviour would be nice as well.  
> BTW I'm not wedded to this idea

I can mock-up a basic solution. Maybe we can get opinions from other contributors and maintainers in the meantime.

I think this would add value because it would make customization of negative contours more straightforward. Since this doesn't require removing any current functionality, I don't see any cons to adding the kwarg.
```

## Patch

```diff
diff --git a/lib/matplotlib/contour.py b/lib/matplotlib/contour.py
--- a/lib/matplotlib/contour.py
+++ b/lib/matplotlib/contour.py
@@ -701,7 +701,7 @@ def __init__(self, ax, *args,
                  hatches=(None,), alpha=None, origin=None, extent=None,
                  cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                  extend='neither', antialiased=None, nchunk=0, locator=None,
-                 transform=None,
+                 transform=None, negative_linestyles=None,
                  **kwargs):
         """
         Draw contour lines or filled regions, depending on
@@ -786,6 +786,13 @@ def __init__(self, ax, *args,
 
         self._transform = transform
 
+        self.negative_linestyles = negative_linestyles
+        # If negative_linestyles was not defined as a kwarg,
+        # define negative_linestyles with rcParams
+        if self.negative_linestyles is None:
+            self.negative_linestyles = \
+                mpl.rcParams['contour.negative_linestyle']
+
         kwargs = self._process_args(*args, **kwargs)
         self._process_levels()
 
@@ -1276,11 +1283,10 @@ def _process_linestyles(self):
         if linestyles is None:
             tlinestyles = ['solid'] * Nlev
             if self.monochrome:
-                neg_ls = mpl.rcParams['contour.negative_linestyle']
                 eps = - (self.zmax - self.zmin) * 1e-15
                 for i, lev in enumerate(self.levels):
                     if lev < eps:
-                        tlinestyles[i] = neg_ls
+                        tlinestyles[i] = self.negative_linestyles
         else:
             if isinstance(linestyles, str):
                 tlinestyles = [linestyles] * Nlev
@@ -1751,6 +1757,18 @@ def _initialize_x_y(self, z):
     iterable is shorter than the number of contour levels
     it will be repeated as necessary.
 
+negative_linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, \
+                       optional
+    *Only applies to* `.contour`.
+
+    If *negative_linestyles* is None, the default is 'dashed' for
+    negative contours.
+
+    *negative_linestyles* can also be an iterable of the above
+    strings specifying a set of linestyles to be used. If this
+    iterable is shorter than the number of contour levels
+    it will be repeated as necessary.
+
 hatches : list[str], optional
     *Only applies to* `.contourf`.
 

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_contour.py b/lib/matplotlib/tests/test_contour.py
--- a/lib/matplotlib/tests/test_contour.py
+++ b/lib/matplotlib/tests/test_contour.py
@@ -605,3 +605,80 @@ def test_subfigure_clabel():
         CS = ax.contour(X, Y, Z)
         ax.clabel(CS, inline=True, fontsize=10)
         ax.set_title("Simplest default with labels")
+
+
+@pytest.mark.parametrize(
+    "style", ['solid', 'dashed', 'dashdot', 'dotted'])
+def test_linestyles(style):
+    delta = 0.025
+    x = np.arange(-3.0, 3.0, delta)
+    y = np.arange(-2.0, 2.0, delta)
+    X, Y = np.meshgrid(x, y)
+    Z1 = np.exp(-X**2 - Y**2)
+    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
+    Z = (Z1 - Z2) * 2
+
+    # Positive contour defaults to solid
+    fig1, ax1 = plt.subplots()
+    CS1 = ax1.contour(X, Y, Z, 6, colors='k')
+    ax1.clabel(CS1, fontsize=9, inline=True)
+    ax1.set_title('Single color - positive contours solid (default)')
+    assert CS1.linestyles is None  # default
+
+    # Change linestyles using linestyles kwarg
+    fig2, ax2 = plt.subplots()
+    CS2 = ax2.contour(X, Y, Z, 6, colors='k', linestyles=style)
+    ax2.clabel(CS2, fontsize=9, inline=True)
+    ax2.set_title(f'Single color - positive contours {style}')
+    assert CS2.linestyles == style
+
+    # Ensure linestyles do not change when negative_linestyles is defined
+    fig3, ax3 = plt.subplots()
+    CS3 = ax3.contour(X, Y, Z, 6, colors='k', linestyles=style,
+                      negative_linestyles='dashdot')
+    ax3.clabel(CS3, fontsize=9, inline=True)
+    ax3.set_title(f'Single color - positive contours {style}')
+    assert CS3.linestyles == style
+
+
+@pytest.mark.parametrize(
+    "style", ['solid', 'dashed', 'dashdot', 'dotted'])
+def test_negative_linestyles(style):
+    delta = 0.025
+    x = np.arange(-3.0, 3.0, delta)
+    y = np.arange(-2.0, 2.0, delta)
+    X, Y = np.meshgrid(x, y)
+    Z1 = np.exp(-X**2 - Y**2)
+    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
+    Z = (Z1 - Z2) * 2
+
+    # Negative contour defaults to dashed
+    fig1, ax1 = plt.subplots()
+    CS1 = ax1.contour(X, Y, Z, 6, colors='k')
+    ax1.clabel(CS1, fontsize=9, inline=True)
+    ax1.set_title('Single color - negative contours dashed (default)')
+    assert CS1.negative_linestyles == 'dashed'  # default
+
+    # Change negative_linestyles using rcParams
+    plt.rcParams['contour.negative_linestyle'] = style
+    fig2, ax2 = plt.subplots()
+    CS2 = ax2.contour(X, Y, Z, 6, colors='k')
+    ax2.clabel(CS2, fontsize=9, inline=True)
+    ax2.set_title(f'Single color - negative contours {style}'
+                   '(using rcParams)')
+    assert CS2.negative_linestyles == style
+
+    # Change negative_linestyles using negative_linestyles kwarg
+    fig3, ax3 = plt.subplots()
+    CS3 = ax3.contour(X, Y, Z, 6, colors='k', negative_linestyles=style)
+    ax3.clabel(CS3, fontsize=9, inline=True)
+    ax3.set_title(f'Single color - negative contours {style}')
+    assert CS3.negative_linestyles == style
+
+    # Ensure negative_linestyles do not change when linestyles is defined
+    fig4, ax4 = plt.subplots()
+    CS4 = ax4.contour(X, Y, Z, 6, colors='k', linestyles='dashdot',
+                      negative_linestyles=style)
+    ax4.clabel(CS4, fontsize=9, inline=True)
+    ax4.set_title(f'Single color - negative contours {style}')
+    assert CS4.negative_linestyles == style

```


## Code snippets

### 1 - lib/matplotlib/contour.py:

Start line: 1252, End line: 1271

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def _process_linewidths(self):
        linewidths = self.linewidths
        Nlev = len(self.levels)
        if linewidths is None:
            default_linewidth = mpl.rcParams['contour.linewidth']
            if default_linewidth is None:
                default_linewidth = mpl.rcParams['lines.linewidth']
            tlinewidths = [(default_linewidth,)] * Nlev
        else:
            if not np.iterable(linewidths):
                linewidths = [linewidths] * Nlev
            else:
                linewidths = list(linewidths)
                if len(linewidths) < Nlev:
                    nreps = int(np.ceil(Nlev / len(linewidths)))
                    linewidths = linewidths * nreps
                if len(linewidths) > Nlev:
                    linewidths = linewidths[:Nlev]
            tlinewidths = [(w,) for w in linewidths]
        return tlinewidths
```
### 2 - lib/matplotlib/tri/tricontour.py:

Start line: 218, End line: 244

```python
@_docstring.Substitution(func='tricontour', type='lines')
@_docstring.dedent_interpd
def tricontour(ax, *args, **kwargs):
    """
    %(_tricontour_doc)s

    linewidths : float or array-like, default: :rc:`contour.linewidth`
        The line width of the contour lines.

        If a number, all levels will be plotted with this linewidth.

        If a sequence, the levels in ascending order will be plotted with
        the linewidths in the order specified.

        If None, this falls back to :rc:`lines.linewidth`.

    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
        If *linestyles* is *None*, the default is 'solid' unless the lines are
        monochrome.  In that case, negative contours will take their linestyle
        from :rc:`contour.negative_linestyle` setting.

        *linestyles* can also be an iterable of the above strings specifying a
        set of linestyles to be used. If this iterable is shorter than the
        number of contour levels it will be repeated as necessary.
    """
    kwargs['filled'] = False
    return TriContourSet(ax, *args, **kwargs)
```
### 3 - lib/matplotlib/contour.py:

Start line: 1273, End line: 1308

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def _process_linestyles(self):
        linestyles = self.linestyles
        Nlev = len(self.levels)
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
            if self.monochrome:
                neg_ls = mpl.rcParams['contour.negative_linestyle']
                eps = - (self.zmax - self.zmin) * 1e-15
                for i, lev in enumerate(self.levels):
                    if lev < eps:
                        tlinestyles[i] = neg_ls
        else:
            if isinstance(linestyles, str):
                tlinestyles = [linestyles] * Nlev
            elif np.iterable(linestyles):
                tlinestyles = list(linestyles)
                if len(tlinestyles) < Nlev:
                    nreps = int(np.ceil(Nlev / len(linestyles)))
                    tlinestyles = tlinestyles * nreps
                if len(tlinestyles) > Nlev:
                    tlinestyles = tlinestyles[:Nlev]
            else:
                raise ValueError("Unrecognized type for linestyles kwarg")
        return tlinestyles

    def get_alpha(self):
        """Return alpha to be applied to all ContourSet artists."""
        return self.alpha

    def set_alpha(self, alpha):
        """
        Set the alpha blending value for all ContourSet artists.
        *alpha* must be between 0 (transparent) and 1 (opaque).
        """
        self.alpha = alpha
        self.changed()
```
### 4 - lib/matplotlib/contour.py:

Start line: 849, End line: 905

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def __init__(self, ax, *args,
                 levels=None, filled=False, linewidths=None, linestyles=None,
                 hatches=(None,), alpha=None, origin=None, extent=None,
                 cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                 extend='neither', antialiased=None, nchunk=0, locator=None,
                 transform=None,
                 **kwargs):
        # ... other code

        if self.filled:
            if self.linewidths is not None:
                _api.warn_external('linewidths is ignored by contourf')
            # Lower and upper contour levels.
            lowers, uppers = self._get_lowers_and_uppers()
            # Default zorder taken from Collection
            self._contour_zorder = kwargs.pop('zorder', 1)

            self.collections[:] = [
                mcoll.PathCollection(
                    self._make_paths(segs, kinds),
                    antialiaseds=(self.antialiased,),
                    edgecolors='none',
                    alpha=self.alpha,
                    transform=self.get_transform(),
                    zorder=self._contour_zorder)
                for level, level_upper, segs, kinds
                in zip(lowers, uppers, self.allsegs, self.allkinds)]
        else:
            self.tlinewidths = tlinewidths = self._process_linewidths()
            tlinestyles = self._process_linestyles()
            aa = self.antialiased
            if aa is not None:
                aa = (self.antialiased,)
            # Default zorder taken from LineCollection, which is higher than
            # for filled contours so that lines are displayed on top.
            self._contour_zorder = kwargs.pop('zorder', 2)

            self.collections[:] = [
                mcoll.PathCollection(
                    self._make_paths(segs, kinds),
                    facecolors="none",
                    antialiaseds=aa,
                    linewidths=width,
                    linestyles=[lstyle],
                    alpha=self.alpha,
                    transform=self.get_transform(),
                    zorder=self._contour_zorder,
                    label='_nolegend_')
                for level, width, lstyle, segs, kinds
                in zip(self.levels, tlinewidths, tlinestyles, self.allsegs,
                       self.allkinds)]

        for col in self.collections:
            self.axes.add_collection(col, autolim=False)
            col.sticky_edges.x[:] = [self._mins[0], self._maxs[0]]
            col.sticky_edges.y[:] = [self._mins[1], self._maxs[1]]
        self.axes.update_datalim([self._mins, self._maxs])
        self.axes.autoscale_view(tight=True)

        self.changed()  # set the colors

        if kwargs:
            _api.warn_external(
                'The following kwargs were not used by contour: ' +
                ", ".join(map(repr, kwargs))
            )
```
### 5 - lib/matplotlib/tri/tricontour.py:

Start line: 84, End line: 215

```python
_docstring.interpd.update(_tricontour_doc="""
Draw contour %(type)s on an unstructured triangular grid.

Call signatures::

    %(func)s(triangulation, Z, [levels], ...)
    %(func)s(x, y, Z, [levels], *, [triangles=triangles], [mask=mask], ...)

The triangular grid can be specified either by passing a `.Triangulation`
object as the first parameter, or by passing the points *x*, *y* and
optionally the *triangles* and a *mask*. See `.Triangulation` for an
explanation of these parameters. If neither of *triangulation* or
*triangles* are given, the triangulation is calculated on the fly.

It is possible to pass *triangles* positionally, i.e.
``%(func)s(x, y, triangles, Z, ...)``. However, this is discouraged. For more
clarity, pass *triangles* via keyword argument.

Parameters
----------
triangulation : `.Triangulation`, optional
    An already created triangular grid.

x, y, triangles, mask
    Parameters defining the triangular grid. See `.Triangulation`.
    This is mutually exclusive with specifying *triangulation*.

Z : array-like
    The height values over which the contour is drawn.

levels : int or array-like, optional
    Determines the number and positions of the contour lines / regions.

    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to
    automatically choose no more than *n+1* "nice" contour levels between
    *vmin* and *vmax*.

    If array-like, draw contour lines at the specified levels.  The values must
    be in increasing order.

Returns
-------
`~matplotlib.tri.TriContourSet`

Other Parameters
----------------
colors : color string or sequence of colors, optional
    The colors of the levels, i.e., the contour %(type)s.

    The sequence is cycled for the levels in ascending order. If the sequence
    is shorter than the number of levels, it's repeated.

    As a shortcut, single color strings may be used in place of one-element
    lists, i.e. ``'red'`` instead of ``['red']`` to color all levels with the
    same color. This shortcut does only work for color strings, not for other
    ways of specifying colors.

    By default (value *None*), the colormap specified by *cmap* will be used.

alpha : float, default: 1
    The alpha blending value, between 0 (transparent) and 1 (opaque).

cmap : str or `.Colormap`, default: :rc:`image.cmap`
    A `.Colormap` instance or registered colormap name. The colormap maps the
    level values to colors.

    If both *colors* and *cmap* are given, an error is raised.

norm : `~matplotlib.colors.Normalize`, optional
    If a colormap is used, the `.Normalize` instance scales the level values to
    the canonical colormap range [0, 1] for mapping to colors. If not given,
    the default linear scaling is used.

vmin, vmax : float, optional
    If not *None*, either or both of these values will be supplied to
    the `.Normalize` instance, overriding the default color scaling
    based on *levels*.

origin : {*None*, 'upper', 'lower', 'image'}, default: None
    Determines the orientation and exact position of *Z* by specifying the
    position of ``Z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.

    - *None*: ``Z[0, 0]`` is at X=0, Y=0 in the lower left corner.
    - 'lower': ``Z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.
    - 'upper': ``Z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.
    - 'image': Use the value from :rc:`image.origin`.

extent : (x0, x1, y0, y1), optional
    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it
    gives the outer pixel boundaries. In this case, the position of Z[0, 0] is
    the center of the pixel, not a corner. If *origin* is *None*, then
    (*x0*, *y0*) is the position of Z[0, 0], and (*x1*, *y1*) is the position
    of Z[-1, -1].

    This argument is ignored if *X* and *Y* are specified in the call to
    contour.

locator : ticker.Locator subclass, optional
    The locator is used to determine the contour levels if they are not given
    explicitly via *levels*.
    Defaults to `~.ticker.MaxNLocator`.

extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
    Determines the ``%(func)s``-coloring of values that are outside the
    *levels* range.

    If 'neither', values outside the *levels* range are not colored.  If 'min',
    'max' or 'both', color the values below, above or below and above the
    *levels* range.

    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the
    under/over values of the `.Colormap`. Note that most colormaps do not have
    dedicated colors for these by default, so that the over and under values
    are the edge values of the colormap.  You may want to set these values
    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.

    .. note::

        An existing `.TriContourSet` does not get notified if properties of its
        colormap are changed. Therefore, an explicit call to
        `.ContourSet.changed()` is needed after modifying the colormap. The
        explicit call can be left out, if a colorbar is assigned to the
        `.TriContourSet` because it internally calls `.ContourSet.changed()`.

xunits, yunits : registered units, optional
    Override axis units by specifying an instance of a
    :class:`matplotlib.units.ConversionInterface`.

antialiased : bool, optional
    Enable antialiasing, overriding the defaults.  For
    filled contours, the default is *True*.  For line contours,
    it is taken from :rc:`lines.antialiased`.""")
```
### 6 - examples/images_contours_and_fields/contourf_demo.py:

Start line: 96, End line: 129

```python
extends = ["neither", "both", "min", "max"]
cmap = plt.colormaps["winter"].with_extremes(under="magenta", over="yellow")
# Note: contouring simply excludes masked or nan regions, so
# instead of using the "bad" colormap value for them, it draws
# nothing at all in them.  Therefore the following would have
# no effect:
# cmap.set_bad("red")

fig, axs = plt.subplots(2, 2, constrained_layout=True)

for ax, extend in zip(axs.flat, extends):
    cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
    fig.colorbar(cs, ax=ax, shrink=0.9)
    ax.set_title("extend = %s" % extend)
    ax.locator_params(nbins=4)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.set_bad`
#    - `matplotlib.colors.Colormap.set_under`
#    - `matplotlib.colors.Colormap.set_over`
```
### 7 - lib/matplotlib/pyplot.py:

Start line: 2438, End line: 2455

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.contour)
def contour(*args, data=None, **kwargs):
    __ret = gca().contour(
        *args, **({"data": data} if data is not None else {}),
        **kwargs)
    if __ret._A is not None: sci(__ret)  # noqa
    return __ret


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.contourf)
def contourf(*args, data=None, **kwargs):
    __ret = gca().contourf(
        *args, **({"data": data} if data is not None else {}),
        **kwargs)
    if __ret._A is not None: sci(__ret)  # noqa
    return __ret
```
### 8 - lib/matplotlib/contour.py:

Start line: 538, End line: 597

```python
class ContourLabeler:

    def pop_label(self, index=-1):
        """Defaults to removing last label, but any index can be supplied"""
        self.labelCValues.pop(index)
        t = self.labelTexts.pop(index)
        t.remove()

    def labels(self, inline, inline_spacing):

        if self._use_clabeltext:
            add_label = self.add_label_clabeltext
        else:
            add_label = self.add_label

        for idx, (icon, lev, cvalue) in enumerate(zip(
                self.labelIndiceList,
                self.labelLevelList,
                self.labelCValueList,
        )):

            con = self.collections[icon]
            trans = con.get_transform()
            lw = self._get_nth_label_width(idx)
            additions = []
            paths = con.get_paths()
            for segNum, linepath in enumerate(paths):
                lc = linepath.vertices  # Line contour
                slc = trans.transform(lc)  # Line contour in screen coords

                # Check if long enough for a label
                if self.print_label(slc, lw):
                    x, y, ind = self.locate_label(slc, lw)

                    rotation, new = self.calc_label_rot_and_inline(
                        slc, ind, lw, lc if inline else None, inline_spacing)

                    # Actually add the label
                    add_label(x, y, rotation, lev, cvalue)

                    # If inline, add new contours
                    if inline:
                        for n in new:
                            # Add path if not empty or single point
                            if len(n) > 1:
                                additions.append(mpath.Path(n))
                else:  # If not adding label, keep old path
                    additions.append(linepath)

            # After looping over all segments on a contour, replace old paths
            # by new ones if inlining.
            if inline:
                paths[:] = additions


def _is_closed_polygon(X):
    """
    Return whether first and last object in a sequence are the same. These are
    presumably coordinates on a polygonal curve, in which case this function
    tests if that curve is closed.
    """
    return np.allclose(X[0], X[-1], rtol=1e-10, atol=1e-13)
```
### 9 - lib/matplotlib/contour.py:

Start line: 1458, End line: 1482

```python
@_docstring.dedent_interpd
class QuadContourSet(ContourSet):

    def _contour_args(self, args, kwargs):
        if self.filled:
            fn = 'contourf'
        else:
            fn = 'contour'
        Nargs = len(args)
        if Nargs <= 2:
            z = ma.asarray(args[0], dtype=np.float64)
            x, y = self._initialize_x_y(z)
            args = args[1:]
        elif Nargs <= 4:
            x, y, z = self._check_xyz(args[:3], kwargs)
            args = args[3:]
        else:
            raise TypeError("Too many arguments to %s; see help(%s)" %
                            (fn, fn))
        z = ma.masked_invalid(z, copy=False)
        self.zmax = float(z.max())
        self.zmin = float(z.min())
        if self.logscale and self.zmin <= 0:
            z = ma.masked_where(z <= 0, z)
            _api.warn_external('Log scale: values of z <= 0 have been masked')
            self.zmin = float(z.min())
        self._process_contour_level_args(args)
        return (x, y, z)
```
### 10 - examples/images_contours_and_fields/contour_demo.py:

Start line: 1, End line: 83

```python
"""
============
Contour Demo
============

Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also the :doc:`contour image example
</gallery/images_contours_and_fields/contour_image>`.
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

###############################################################################
# Create a simple contour plot with labels using default colors.  The inline
# argument to clabel will control whether the labels are draw over the line
# segments of the contour, removing the lines beneath the label.

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')

###############################################################################
# Contour labels can be placed manually by providing list of positions (in data
# coordinate).  See :doc:`/gallery/event_handling/ginput_manual_clabel_sgskip`
# for interactive placement.

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
manual_locations = [
    (-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
ax.clabel(CS, inline=True, fontsize=10, manual=manual_locations)
ax.set_title('labels at selected locations')

###############################################################################
# You can force all the contours to be the same color.

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')  # Negative contours default to dashed.
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours dashed')

###############################################################################
# You can set negative contours to be solid instead of dashed:

plt.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')  # Negative contours default to dashed.
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours solid')

###############################################################################
# And you can manually specify the colors of the contour

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6,
                linewidths=np.arange(.5, 4, .5),
                colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'),
                )
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Crazy lines')

###############################################################################
# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines

fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cm.gray, extent=(-3, 3, -2, 2))
levels = np.arange(-1.2, 1.6, 0.2)
```
### 11 - lib/matplotlib/contour.py:

Start line: 699, End line: 847

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def __init__(self, ax, *args,
                 levels=None, filled=False, linewidths=None, linestyles=None,
                 hatches=(None,), alpha=None, origin=None, extent=None,
                 cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                 extend='neither', antialiased=None, nchunk=0, locator=None,
                 transform=None,
                 **kwargs):
        """
        Draw contour lines or filled regions, depending on
        whether keyword arg *filled* is ``False`` (default) or ``True``.

        Call signature::

            ContourSet(ax, levels, allsegs, [allkinds], **kwargs)

        Parameters
        ----------
        ax : `~.axes.Axes`
            The `~.axes.Axes` object to draw on.

        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the contour
            levels.

        allsegs : [level0segs, level1segs, ...]
            List of all the polygon segments for all the *levels*.
            For contour lines ``len(allsegs) == len(levels)``, and for
            filled contour regions ``len(allsegs) = len(levels)-1``. The lists
            should look like ::

                level0segs = [polygon0, polygon1, ...]
                polygon0 = [[x0, y0], [x1, y1], ...]

        allkinds : [level0kinds, level1kinds, ...], optional
            Optional list of all the polygon vertex kinds (code types), as
            described and used in Path. This is used to allow multiply-
            connected paths such as holes within filled polygons.
            If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
            should look like ::

                level0kinds = [polygon0kinds, ...]
                polygon0kinds = [vertexcode0, vertexcode1, ...]

            If *allkinds* is not ``None``, usually all polygons for a
            particular contour level are grouped together so that
            ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

        **kwargs
            Keyword arguments are as described in the docstring of
            `~.Axes.contour`.
        """
        self.axes = ax
        self.levels = levels
        self.filled = filled
        self.linewidths = linewidths
        self.linestyles = linestyles
        self.hatches = hatches
        self.alpha = alpha
        self.origin = origin
        self.extent = extent
        self.colors = colors
        self.extend = extend
        self.antialiased = antialiased
        if self.antialiased is None and self.filled:
            # Eliminate artifacts; we are not stroking the boundaries.
            self.antialiased = False
            # The default for line contours will be taken from the
            # LineCollection default, which uses :rc:`lines.antialiased`.

        self.nchunk = nchunk
        self.locator = locator
        if (isinstance(norm, mcolors.LogNorm)
                or isinstance(self.locator, ticker.LogLocator)):
            self.logscale = True
            if norm is None:
                norm = mcolors.LogNorm()
        else:
            self.logscale = False

        _api.check_in_list([None, 'lower', 'upper', 'image'], origin=origin)
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError(
                "If given, 'extent' must be None or (x0, x1, y0, y1)")
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image':
            self.origin = mpl.rcParams['image.origin']

        self._transform = transform

        kwargs = self._process_args(*args, **kwargs)
        self._process_levels()

        self._extend_min = self.extend in ['min', 'both']
        self._extend_max = self.extend in ['max', 'both']
        if self.colors is not None:
            ncolors = len(self.levels)
            if self.filled:
                ncolors -= 1
            i0 = 0

            # Handle the case where colors are given for the extended
            # parts of the contour.

            use_set_under_over = False
            # if we are extending the lower end, and we've been given enough
            # colors then skip the first color in the resulting cmap. For the
            # extend_max case we don't need to worry about passing more colors
            # than ncolors as ListedColormap will clip.
            total_levels = (ncolors +
                            int(self._extend_min) +
                            int(self._extend_max))
            if (len(self.colors) == total_levels and
                    (self._extend_min or self._extend_max)):
                use_set_under_over = True
                if self._extend_min:
                    i0 = 1

            cmap = mcolors.ListedColormap(self.colors[i0:None], N=ncolors)

            if use_set_under_over:
                if self._extend_min:
                    cmap.set_under(self.colors[0])
                if self._extend_max:
                    cmap.set_over(self.colors[-1])

        self.collections = cbook.silent_list(None)

        # label lists must be initialized here
        self.labelTexts = []
        self.labelCValues = []

        kw = {'cmap': cmap}
        if norm is not None:
            kw['norm'] = norm
        # sets self.cmap, norm if needed;
        cm.ScalarMappable.__init__(self, **kw)
        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax
        self._process_colors()

        if getattr(self, 'allsegs', None) is None:
            self.allsegs, self.allkinds = self._get_allsegs_and_allkinds()
        elif self.allkinds is None:
            # allsegs specified in constructor may or may not have allkinds as
            # well.  Must ensure allkinds can be zipped below.
            self.allkinds = [None] * len(self.allsegs)
        # ... other code
```
### 12 - lib/matplotlib/contour.py:

Start line: 1069, End line: 1095

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def changed(self):
        if not hasattr(self, "cvalues"):
            # Just return after calling the super() changed function
            cm.ScalarMappable.changed(self)
            return
        # Force an autoscale immediately because self.to_rgba() calls
        # autoscale_None() internally with the data passed to it,
        # so if vmin/vmax are not set yet, this would override them with
        # content from *cvalues* rather than levels like we want
        self.norm.autoscale_None(self.levels)
        tcolors = [(tuple(rgba),)
                   for rgba in self.to_rgba(self.cvalues, alpha=self.alpha)]
        self.tcolors = tcolors
        hatches = self.hatches * len(tcolors)
        for color, hatch, collection in zip(tcolors, hatches,
                                            self.collections):
            if self.filled:
                collection.set_facecolor(color)
                # update the collection's hatch (may be None)
                collection.set_hatch(hatch)
            else:
                collection.set_edgecolor(color)
        for label, cv in zip(self.labelTexts, self.labelCValues):
            label.set_alpha(self.alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        # add label colors
        cm.ScalarMappable.changed(self)
```
### 14 - lib/matplotlib/contour.py:

Start line: 388, End line: 418

```python
class ContourLabeler:

    def calc_label_rot_and_inline(self, slc, ind, lw, lc=None, spacing=5):
        # ... other code
        if len(lc):
            # Expand range by spacing
            xi = dp + xi + np.array([-spacing, spacing])

            # Get (integer) indices near points of interest; use -1 as marker
            # for out of bounds.
            I = np.interp(xi, pl, np.arange(len(pl)), left=-1, right=-1)
            I = [np.floor(I[0]).astype(int), np.ceil(I[1]).astype(int)]
            if I[0] != -1:
                xy1 = [np.interp(xi[0], pl, lc_col) for lc_col in lc.T]
            if I[1] != -1:
                xy2 = [np.interp(xi[1], pl, lc_col) for lc_col in lc.T]

            # Actually break contours
            if closed:
                # This will remove contour if shorter than label
                if all(i != -1 for i in I):
                    nlc.append(np.row_stack([xy2, lc[I[1]:I[0]+1], xy1]))
            else:
                # These will remove pieces of contour if they have length zero
                if I[0] != -1:
                    nlc.append(np.row_stack([lc[:I[0]+1], xy1]))
                if I[1] != -1:
                    nlc.append(np.row_stack([xy2, lc[I[1]:]]))

            # The current implementation removes contours completely
            # covered by labels.  Uncomment line below to keep
            # original contour if this is the preferred behavior.
            # if not len(nlc): nlc = [ lc ]

        return rotation, nlc
```
### 19 - lib/matplotlib/contour.py:

Start line: 636, End line: 652

```python
_docstring.interpd.update(contour_set_attributes=r"""
Attributes
----------
ax : `~matplotlib.axes.Axes`
    The Axes object in which the contours are drawn.

collections : `.silent_list` of `.PathCollection`\s
    The `.Artist`\s representing the contour. This is a list of
    `.PathCollection`\s for both line and filled contours.

levels : array
    The values of the contour levels.

layers : array
    Same as levels for line contours; half-way between
    levels for filled contours.  See ``ContourSet._process_colors``.
""")
```
### 26 - lib/matplotlib/contour.py:

Start line: 240, End line: 261

```python
class ContourLabeler:

    def print_label(self, linecontour, labelwidth):
        """Return whether a contour is long enough to hold a label."""
        return (len(linecontour) > 10 * labelwidth
                or (np.ptp(linecontour, axis=0) > 1.2 * labelwidth).any())

    def too_close(self, x, y, lw):
        """Return whether a label is already near this location."""
        thresh = (1.2 * lw) ** 2
        return any((x - loc[0]) ** 2 + (y - loc[1]) ** 2 < thresh
                   for loc in self.labelXYs)

    def _get_nth_label_width(self, nth):
        """Return the width of the *nth* label, in pixels."""
        fig = self.axes.figure
        renderer = fig._get_renderer()
        return (
            text.Text(0, 0,
                      self.get_text(self.labelLevelList[nth], self.labelFmt),
                      figure=fig,
                      size=self.labelFontSizeList[nth],
                      fontproperties=self.labelFontProps)
            .get_window_extent(renderer).width)
```
### 27 - lib/matplotlib/contour.py:

Start line: 1207, End line: 1250

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the colormapping on the contour levels
        and layers, not on the actual range of the Z values.  This
        means we don't have to worry about bad values in Z, and we
        always have the full dynamic range available for the selected
        levels.

        The color is based on the midpoint of the layer, except for
        extended end layers.  By default, the norm vmin and vmax
        are the extreme values of the non-extended levels.  Hence,
        the layer color extremes are not the extreme values of
        the colormap itself, but approach those values as the number
        of levels increases.  An advantage of this scheme is that
        line contours, when added to filled contours, take on
        colors that are consistent with those of the filled regions;
        for example, a contour line on the boundary between two
        regions will have a color intermediate between those
        of the regions.

        """
        self.monochrome = self.cmap.monochrome
        if self.colors is not None:
            # Generate integers for direct indexing.
            i0, i1 = 0, len(self.levels)
            if self.filled:
                i1 -= 1
                # Out of range indices for over and under:
                if self.extend in ('both', 'min'):
                    i0 -= 1
                if self.extend in ('both', 'max'):
                    i1 += 1
            self.cvalues = list(range(i0, i1))
            self.set_norm(mcolors.NoNorm())
        else:
            self.cvalues = self.layers
        self.set_array(self.levels)
        self.autoscale_None()
        if self.extend in ('both', 'max', 'min'):
            self.norm.clip = False

        # self.tcolors are set by the "changed" method
```
### 28 - lib/matplotlib/contour.py:

Start line: 47, End line: 73

```python
def _contour_labeler_event_handler(cs, inline, inline_spacing, event):
    canvas = cs.axes.figure.canvas
    is_button = event.name == "button_press_event"
    is_key = event.name == "key_press_event"
    # Quit (even if not in infinite mode; this is consistent with
    # MATLAB and sometimes quite useful, but will require the user to
    # test how many points were actually returned before using data).
    if (is_button and event.button == MouseButton.MIDDLE
            or is_key and event.key in ["escape", "enter"]):
        canvas.stop_event_loop()
    # Pop last click.
    elif (is_button and event.button == MouseButton.RIGHT
          or is_key and event.key in ["backspace", "delete"]):
        # Unfortunately, if one is doing inline labels, then there is currently
        # no way to fix the broken contour - once humpty-dumpty is broken, he
        # can't be put back together.  In inline mode, this does nothing.
        if not inline:
            cs.pop_label()
            canvas.draw()
    # Add new click.
    elif (is_button and event.button == MouseButton.LEFT
          # On macOS/gtk, some keys return None.
          or is_key and event.key is not None):
        if event.inaxes == cs.axes:
            cs.add_label_near(event.x, event.y, transform=False,
                              inline=inline, inline_spacing=inline_spacing)
            canvas.draw()
```
### 34 - lib/matplotlib/contour.py:

Start line: 1037, End line: 1050

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def _get_lowers_and_uppers(self):
        """
        Return ``(lowers, uppers)`` for filled contours.
        """
        lowers = self._levels[:-1]
        if self.zmin == lowers[0]:
            # Include minimum values in lowest interval
            lowers = lowers.copy()  # so we don't change self._levels
            if self.logscale:
                lowers[0] = 0.99 * self.zmin
            else:
                lowers[0] -= 1
        uppers = self._levels[1:]
        return (lowers, uppers)
```
### 36 - lib/matplotlib/contour.py:

Start line: 167, End line: 238

```python
class ContourLabeler:

    def clabel(self, levels=None, *,
               fontsize=None, inline=True, inline_spacing=5, fmt=None,
               colors=None, use_clabeltext=False, manual=False,
               rightside_up=True, zorder=None):

        # clabel basically takes the input arguments and uses them to
        # add a list of "label specific" attributes to the ContourSet
        # object.  These attributes are all of the form label* and names
        # should be fairly self explanatory.
        #
        # Once these attributes are set, clabel passes control to the
        # labels method (case of automatic label placement) or
        # `BlockingContourLabeler` (case of manual label placement).

        if fmt is None:
            fmt = ticker.ScalarFormatter(useOffset=False)
            fmt.create_dummy_axis()
        self.labelFmt = fmt
        self._use_clabeltext = use_clabeltext
        # Detect if manual selection is desired and remove from argument list.
        self.labelManual = manual
        self.rightside_up = rightside_up
        if zorder is None:
            self._clabel_zorder = 2+self._contour_zorder
        else:
            self._clabel_zorder = zorder

        if levels is None:
            levels = self.levels
            indices = list(range(len(self.cvalues)))
        else:
            levlabs = list(levels)
            indices, levels = [], []
            for i, lev in enumerate(self.levels):
                if lev in levlabs:
                    indices.append(i)
                    levels.append(lev)
            if len(levels) < len(levlabs):
                raise ValueError(f"Specified levels {levlabs} don't match "
                                 f"available levels {self.levels}")
        self.labelLevelList = levels
        self.labelIndiceList = indices

        self.labelFontProps = font_manager.FontProperties()
        self.labelFontProps.set_size(fontsize)
        font_size_pts = self.labelFontProps.get_size_in_points()
        self.labelFontSizeList = [font_size_pts] * len(levels)

        if colors is None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = mcolors.ListedColormap(colors, N=len(self.labelLevelList))
            self.labelCValueList = list(range(len(self.labelLevelList)))
            self.labelMappable = cm.ScalarMappable(cmap=cmap,
                                                   norm=mcolors.NoNorm())

        self.labelXYs = []

        if np.iterable(self.labelManual):
            for x, y in self.labelManual:
                self.add_label_near(x, y, inline, inline_spacing)
        elif self.labelManual:
            print('Select label locations manually using first mouse button.')
            print('End manual selection with second mouse button.')
            if not inline:
                print('Remove last label by clicking third mouse button.')
            mpl._blocking_input.blocking_input_loop(
                self.axes.figure, ["button_press_event", "key_press_event"],
                timeout=-1, handler=functools.partial(
                    _contour_labeler_event_handler,
                    self, inline, inline_spacing))
        else:
            self.labels(inline, inline_spacing)

        self.labelTextsList = cbook.silent_list('text.Text', self.labelTexts)
        return self.labelTextsList
```
### 44 - lib/matplotlib/contour.py:

Start line: 1097, End line: 1138

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def _autolev(self, N):
        """
        Select contour levels to span the data.

        The target number of levels, *N*, is used only when the
        scale is not log and default locator is used.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        """
        if self.locator is None:
            if self.logscale:
                self.locator = ticker.LogLocator()
            else:
                self.locator = ticker.MaxNLocator(N + 1, min_n_ticks=1)

        lev = self.locator.tick_values(self.zmin, self.zmax)

        try:
            if self.locator._symmetric:
                return lev
        except AttributeError:
            pass

        # Trim excess levels the locator may have supplied.
        under = np.nonzero(lev < self.zmin)[0]
        i0 = under[-1] if len(under) else 0
        over = np.nonzero(lev > self.zmax)[0]
        i1 = over[0] + 1 if len(over) else len(lev)
        if self.extend in ('min', 'both'):
            i0 += 1
        if self.extend in ('max', 'both'):
            i1 -= 1

        if i1 - i0 < 3:
            i0, i1 = 0, len(lev)

        return lev[i0:i1]
```
### 46 - lib/matplotlib/contour.py:

Start line: 1, End line: 44

```python
"""
Classes to support contour plotting and labelling for the Axes class.
"""

import functools
from numbers import Integral

import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
import matplotlib.path as mpath
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.text as text
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


# We can't use a single line collection for contour because a line
# collection can have only a single line style, and we want to be able to have
# dashed negative contours, for example, and solid positive contours.
# We could use a single polygon collection for filled contours, but it
# seems better to keep line and filled contours similar, with one collection
# per level.


class ClabelText(text.Text):
    """
    Unlike the ordinary text, the get_rotation returns an updated
    angle in the pixel coordinate assuming that the input rotation is
    an angle in data coordinate (or whatever transform set).
    """

    def get_rotation(self):
        new_angle, = self.get_transform().transform_angles(
            [super().get_rotation()], [self.get_position()])
        return new_angle
```
### 50 - lib/matplotlib/contour.py:

Start line: 1389, End line: 1456

```python
@_docstring.dedent_interpd
class QuadContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions.

    This class is typically not instantiated directly by the user but by
    `~.Axes.contour` and `~.Axes.contourf`.

    %(contour_set_attributes)s
    """

    def _process_args(self, *args, corner_mask=None, algorithm=None, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], QuadContourSet):
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._corner_mask = args[0]._corner_mask
            contour_generator = args[0]._contour_generator
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
            self._algorithm = args[0]._algorithm
        else:
            import contourpy

            if algorithm is None:
                algorithm = mpl.rcParams['contour.algorithm']
            mpl.rcParams.validate["contour.algorithm"](algorithm)
            self._algorithm = algorithm

            if corner_mask is None:
                if self._algorithm == "mpl2005":
                    # mpl2005 does not support corner_mask=True so if not
                    # specifically requested then disable it.
                    corner_mask = False
                else:
                    corner_mask = mpl.rcParams['contour.corner_mask']
            self._corner_mask = corner_mask

            x, y, z = self._contour_args(args, kwargs)

            contour_generator = contourpy.contour_generator(
                x, y, z, name=self._algorithm, corner_mask=self._corner_mask,
                line_type=contourpy.LineType.SeparateCode,
                fill_type=contourpy.FillType.OuterCode,
                chunk_size=self.nchunk)

            t = self.get_transform()

            # if the transform is not trans data, and some part of it
            # contains transData, transform the xs and ys to data coordinates
            if (t != self.axes.transData and
                    any(t.contains_branch_seperately(self.axes.transData))):
                trans_to_data = t - self.axes.transData
                pts = np.vstack([x.flat, y.flat]).T
                transformed_pts = trans_to_data.transform(pts)
                x = transformed_pts[..., 0]
                y = transformed_pts[..., 1]

            self._mins = [ma.min(x), ma.min(y)]
            self._maxs = [ma.max(x), ma.max(y)]

        self._contour_generator = contour_generator

        return kwargs
```
### 51 - lib/matplotlib/contour.py:

Start line: 984, End line: 1016

```python
@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):

    def _process_args(self, *args, **kwargs):
        """
        Process *args* and *kwargs*; override in derived classes.

        Must set self.levels, self.zmin and self.zmax, and update axes limits.
        """
        self.levels = args[0]
        self.allsegs = args[1]
        self.allkinds = args[2] if len(args) > 2 else None
        self.zmax = np.max(self.levels)
        self.zmin = np.min(self.levels)

        # Check lengths of levels and allsegs.
        if self.filled:
            if len(self.allsegs) != len(self.levels) - 1:
                raise ValueError('must be one less number of segments as '
                                 'levels')
        else:
            if len(self.allsegs) != len(self.levels):
                raise ValueError('must be same number of segments as levels')

        # Check length of allkinds.
        if (self.allkinds is not None and
                len(self.allkinds) != len(self.allsegs)):
            raise ValueError('allkinds has different length to allsegs')

        # Determine x, y bounds and update axes data limits.
        flatseglist = [s for seg in self.allsegs for s in seg]
        points = np.concatenate(flatseglist, axis=0)
        self._mins = points.min(axis=0)
        self._maxs = points.max(axis=0)

        return kwargs
```
