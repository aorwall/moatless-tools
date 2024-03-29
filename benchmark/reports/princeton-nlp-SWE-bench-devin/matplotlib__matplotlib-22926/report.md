# matplotlib__matplotlib-22926

| **matplotlib/matplotlib** | `e779b97174ff3ab2737fbdffb432ef8689201602` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 1268 |
| **Avg pos** | 18.0 |
| **Min pos** | 1 |
| **Max pos** | 13 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/widgets.py b/lib/matplotlib/widgets.py
--- a/lib/matplotlib/widgets.py
+++ b/lib/matplotlib/widgets.py
@@ -19,7 +19,7 @@
 from . import (_api, _docstring, backend_tools, cbook, colors, ticker,
                transforms)
 from .lines import Line2D
-from .patches import Circle, Rectangle, Ellipse
+from .patches import Circle, Rectangle, Ellipse, Polygon
 from .transforms import TransformedPatchPath, Affine2D
 
 
@@ -709,7 +709,7 @@ def __init__(
                 facecolor=track_color
             )
             ax.add_patch(self.track)
-            self.poly = ax.axhspan(valinit[0], valinit[1], 0, 1, **kwargs)
+            poly_transform = self.ax.get_yaxis_transform(which="grid")
             handleXY_1 = [.5, valinit[0]]
             handleXY_2 = [.5, valinit[1]]
         else:
@@ -719,9 +719,15 @@ def __init__(
                 facecolor=track_color
             )
             ax.add_patch(self.track)
-            self.poly = ax.axvspan(valinit[0], valinit[1], 0, 1, **kwargs)
+            poly_transform = self.ax.get_xaxis_transform(which="grid")
             handleXY_1 = [valinit[0], .5]
             handleXY_2 = [valinit[1], .5]
+        self.poly = Polygon(np.zeros([5, 2]), **kwargs)
+        self._update_selection_poly(*valinit)
+        self.poly.set_transform(poly_transform)
+        self.poly.get_path()._interpolation_steps = 100
+        self.ax.add_patch(self.poly)
+        self.ax._request_autoscale_view()
         self._handles = [
             ax.plot(
                 *handleXY_1,
@@ -777,6 +783,27 @@ def __init__(
         self._active_handle = None
         self.set_val(valinit)
 
+    def _update_selection_poly(self, vmin, vmax):
+        """
+        Update the vertices of the *self.poly* slider in-place
+        to cover the data range *vmin*, *vmax*.
+        """
+        # The vertices are positioned
+        #  1 ------ 2
+        #  |        |
+        # 0, 4 ---- 3
+        verts = self.poly.xy
+        if self.orientation == "vertical":
+            verts[0] = verts[4] = .25, vmin
+            verts[1] = .25, vmax
+            verts[2] = .75, vmax
+            verts[3] = .75, vmin
+        else:
+            verts[0] = verts[4] = vmin, .25
+            verts[1] = vmin, .75
+            verts[2] = vmax, .75
+            verts[3] = vmax, .25
+
     def _min_in_bounds(self, min):
         """Ensure the new min value is between valmin and self.val[1]."""
         if min <= self.valmin:
@@ -903,36 +930,24 @@ def set_val(self, val):
         """
         val = np.sort(val)
         _api.check_shape((2,), val=val)
-        val[0] = self._min_in_bounds(val[0])
-        val[1] = self._max_in_bounds(val[1])
-        xy = self.poly.xy
+        vmin, vmax = val
+        vmin = self._min_in_bounds(vmin)
+        vmax = self._max_in_bounds(vmax)
+        self._update_selection_poly(vmin, vmax)
         if self.orientation == "vertical":
-            xy[0] = .25, val[0]
-            xy[1] = .25, val[1]
-            xy[2] = .75, val[1]
-            xy[3] = .75, val[0]
-            xy[4] = .25, val[0]
-
-            self._handles[0].set_ydata([val[0]])
-            self._handles[1].set_ydata([val[1]])
+            self._handles[0].set_ydata([vmin])
+            self._handles[1].set_ydata([vmax])
         else:
-            xy[0] = val[0], .25
-            xy[1] = val[0], .75
-            xy[2] = val[1], .75
-            xy[3] = val[1], .25
-            xy[4] = val[0], .25
+            self._handles[0].set_xdata([vmin])
+            self._handles[1].set_xdata([vmax])
 
-            self._handles[0].set_xdata([val[0]])
-            self._handles[1].set_xdata([val[1]])
-
-        self.poly.xy = xy
-        self.valtext.set_text(self._format(val))
+        self.valtext.set_text(self._format((vmin, vmax)))
 
         if self.drawon:
             self.ax.figure.canvas.draw_idle()
-        self.val = val
+        self.val = (vmin, vmax)
         if self.eventson:
-            self._observers.process("changed", val)
+            self._observers.process("changed", (vmin, vmax))
 
     def on_changed(self, func):
         """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/widgets.py | 22 | 22 | - | 1 | -
| lib/matplotlib/widgets.py | 712 | 712 | 1 | 1 | 1268
| lib/matplotlib/widgets.py | 722 | 722 | 1 | 1 | 1268
| lib/matplotlib/widgets.py | 780 | 780 | 13 | 1 | 5673
| lib/matplotlib/widgets.py | 906 | 935 | 3 | 1 | 1892


## Problem Statement

```
[Bug]: cannot give init value for RangeSlider widget
### Bug summary

I think `xy[4] = .25, val[0]` should be commented in /matplotlib/widgets. py", line 915, in set_val
as it prevents to initialized value for RangeSlider

### Code for reproduction

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

# generate a fake image
np.random.seed(19680801)
N = 128
img = np.random.randn(N, N)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(bottom=0.25)

im = axs[0].imshow(img)
axs[1].hist(img.flatten(), bins='auto')
axs[1].set_title('Histogram of pixel intensities')

# Create the RangeSlider
slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])

# Create the Vertical lines on the histogram
lower_limit_line = axs[1].axvline(slider.val[0], color='k')
upper_limit_line = axs[1].axvline(slider.val[1], color='k')


def update(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the image's colormap
    im.norm.vmin = val[0]
    im.norm.vmax = val[1]

    # Update the position of the vertical lines
    lower_limit_line.set_xdata([val[0], val[0]])
    upper_limit_line.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
\`\`\`


### Actual outcome

\`\`\`python
  File "<ipython-input-52-b704c53e18d4>", line 19, in <module>
    slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])

  File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 778, in __init__
    self.set_val(valinit)

  File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 915, in set_val
    xy[4] = val[0], .25

IndexError: index 4 is out of bounds for axis 0 with size 4
\`\`\`

### Expected outcome

range slider with user initial values

### Additional information

error can be removed by commenting this line
\`\`\`python

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
        val = np.sort(np.asanyarray(val))
        if val.shape != (2,):
            raise ValueError(
                f"val must have shape (2,) but has shape {val.shape}"
            )
        val[0] = self._min_in_bounds(val[0])
        val[1] = self._max_in_bounds(val[1])
        xy = self.poly.xy
        if self.orientation == "vertical":
            xy[0] = .25, val[0]
            xy[1] = .25, val[1]
            xy[2] = .75, val[1]
            xy[3] = .75, val[0]
            # xy[4] = .25, val[0]
        else:
            xy[0] = val[0], .25
            xy[1] = val[0], .75
            xy[2] = val[1], .75
            xy[3] = val[1], .25
            # xy[4] = val[0], .25
        self.poly.xy = xy
        self.valtext.set_text(self._format(val))
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process("changed", val)

\`\`\`

### Operating system

OSX

### Matplotlib Version

3.5.1

### Matplotlib Backend

_No response_

### Python version

3.8

### Jupyter version

_No response_

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 lib/matplotlib/widgets.py** | 603 | 778| 1268 | 1268 | 32204 | 
| 2 | 2 examples/widgets/range_slider.py | 44 | 71| 168 | 1436 | 32712 | 
| **-> 3 <-** | **2 lib/matplotlib/widgets.py** | 896 | 952| 456 | 1892 | 32712 | 
| 4 | **2 lib/matplotlib/widgets.py** | 331 | 485| 1378 | 3270 | 32712 | 
| 5 | **2 lib/matplotlib/widgets.py** | 545 | 585| 270 | 3540 | 32712 | 
| 6 | **2 lib/matplotlib/widgets.py** | 806 | 819| 125 | 3665 | 32712 | 
| 7 | **2 lib/matplotlib/widgets.py** | 281 | 309| 229 | 3894 | 32712 | 
| 8 | **2 lib/matplotlib/widgets.py** | 511 | 543| 267 | 4161 | 32712 | 
| 9 | 3 examples/widgets/slider_snap_demo.py | 1 | 83| 584 | 4745 | 33296 | 
| 10 | **3 lib/matplotlib/widgets.py** | 862 | 894| 248 | 4993 | 33296 | 
| 11 | **3 lib/matplotlib/widgets.py** | 312 | 329| 147 | 5140 | 33296 | 
| 12 | **3 lib/matplotlib/widgets.py** | 821 | 860| 318 | 5458 | 33296 | 
| **-> 13 <-** | **3 lib/matplotlib/widgets.py** | 780 | 804| 215 | 5673 | 33296 | 
| 14 | **3 lib/matplotlib/widgets.py** | 487 | 509| 159 | 5832 | 33296 | 
| 15 | **3 lib/matplotlib/widgets.py** | 588 | 601| 118 | 5950 | 33296 | 
| 16 | 3 examples/widgets/range_slider.py | 1 | 41| 340 | 6290 | 33296 | 
| 17 | 4 examples/widgets/slider_demo.py | 1 | 92| 591 | 6881 | 33887 | 
| 18 | **4 lib/matplotlib/widgets.py** | 234 | 279| 374 | 7255 | 33887 | 
| 19 | 5 examples/user_interfaces/fourier_demo_wx_sgskip.py | 67 | 102| 308 | 7563 | 36031 | 
| 20 | 6 examples/widgets/annotated_cursor.py | 286 | 343| 471 | 8034 | 38879 | 
| 21 | **6 lib/matplotlib/widgets.py** | 1280 | 1309| 313 | 8347 | 38879 | 
| 22 | 7 lib/matplotlib/pyplot.py | 1 | 90| 665 | 9012 | 65464 | 
| 23 | 8 lib/matplotlib/_cm.py | 106 | 156| 787 | 9799 | 93907 | 
| 24 | 9 lib/matplotlib/axis.py | 219 | 346| 802 | 10601 | 115352 | 
| 25 | 10 examples/user_interfaces/embedding_in_tk_sgskip.py | 1 | 68| 466 | 11067 | 115818 | 
| 26 | 11 examples/user_interfaces/svg_histogram_sgskip.py | 37 | 160| 907 | 11974 | 116986 | 
| 27 | 12 examples/images_contours_and_fields/colormap_interactive_adjustment.py | 1 | 34| 306 | 12280 | 117292 | 
| 28 | 13 lib/mpl_toolkits/axisartist/grid_helper_curvelinear.py | 89 | 130| 470 | 12750 | 120717 | 
| 29 | 14 lib/mpl_toolkits/mplot3d/axis3d.py | 165 | 211| 398 | 13148 | 126265 | 
| 30 | 15 examples/event_handling/viewlims.py | 55 | 67| 150 | 13298 | 127089 | 
| 31 | 16 lib/matplotlib/axes/_base.py | 542 | 1299| 6388 | 19686 | 165694 | 
| 32 | 17 lib/matplotlib/backends/backend_qt.py | 891 | 899| 140 | 19826 | 174513 | 
| 33 | 18 examples/ticks/tick-formatters.py | 37 | 137| 970 | 20796 | 175755 | 
| 34 | 19 examples/user_interfaces/pylab_with_gtk3_sgskip.py | 1 | 61| 388 | 21184 | 176143 | 
| 35 | 20 examples/event_handling/image_slices_viewer.py | 25 | 60| 237 | 21421 | 176502 | 
| 36 | 21 examples/event_handling/ginput_manual_clabel_sgskip.py | 1 | 101| 631 | 22052 | 177133 | 
| 37 | 22 lib/matplotlib/colorbar.py | 1123 | 1174| 589 | 22641 | 191812 | 
| 38 | 22 lib/matplotlib/axis.py | 1804 | 1884| 736 | 23377 | 191812 | 


### Hint

```
Huh, the polygon object must have changed inadvertently. Usually, you have
to "close" the polygon by repeating the first vertex, but we make it
possible for polygons to auto-close themselves. I wonder how (when?) this
broke?

On Tue, Mar 22, 2022 at 10:29 PM vpicouet ***@***.***> wrote:

> Bug summary
>
> I think xy[4] = .25, val[0] should be commented in /matplotlib/widgets.
> py", line 915, in set_val
> as it prevents to initialized value for RangeSlider
> Code for reproduction
>
> import numpy as npimport matplotlib.pyplot as pltfrom matplotlib.widgets import RangeSlider
> # generate a fake imagenp.random.seed(19680801)N = 128img = np.random.randn(N, N)
> fig, axs = plt.subplots(1, 2, figsize=(10, 5))fig.subplots_adjust(bottom=0.25)
> im = axs[0].imshow(img)axs[1].hist(img.flatten(), bins='auto')axs[1].set_title('Histogram of pixel intensities')
> # Create the RangeSliderslider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])
> # Create the Vertical lines on the histogramlower_limit_line = axs[1].axvline(slider.val[0], color='k')upper_limit_line = axs[1].axvline(slider.val[1], color='k')
>
> def update(val):
>     # The val passed to a callback by the RangeSlider will
>     # be a tuple of (min, max)
>
>     # Update the image's colormap
>     im.norm.vmin = val[0]
>     im.norm.vmax = val[1]
>
>     # Update the position of the vertical lines
>     lower_limit_line.set_xdata([val[0], val[0]])
>     upper_limit_line.set_xdata([val[1], val[1]])
>
>     # Redraw the figure to ensure it updates
>     fig.canvas.draw_idle()
>
> slider.on_changed(update)plt.show()
>
> Actual outcome
>
>   File "<ipython-input-52-b704c53e18d4>", line 19, in <module>
>     slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max(),valinit=[0.0,0.0])
>
>   File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 778, in __init__
>     self.set_val(valinit)
>
>   File "/Users/Vincent/opt/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/widgets.py", line 915, in set_val
>     xy[4] = val[0], .25
>
> IndexError: index 4 is out of bounds for axis 0 with size 4
>
> Expected outcome
>
> range slider with user initial values
> Additional information
>
> error can be
>
>
>     def set_val(self, val):
>         """
>         Set slider value to *val*.
>
>         Parameters
>         ----------
>         val : tuple or array-like of float
>         """
>         val = np.sort(np.asanyarray(val))
>         if val.shape != (2,):
>             raise ValueError(
>                 f"val must have shape (2,) but has shape {val.shape}"
>             )
>         val[0] = self._min_in_bounds(val[0])
>         val[1] = self._max_in_bounds(val[1])
>         xy = self.poly.xy
>         if self.orientation == "vertical":
>             xy[0] = .25, val[0]
>             xy[1] = .25, val[1]
>             xy[2] = .75, val[1]
>             xy[3] = .75, val[0]
>             # xy[4] = .25, val[0]
>         else:
>             xy[0] = val[0], .25
>             xy[1] = val[0], .75
>             xy[2] = val[1], .75
>             xy[3] = val[1], .25
>             # xy[4] = val[0], .25
>         self.poly.xy = xy
>         self.valtext.set_text(self._format(val))
>         if self.drawon:
>             self.ax.figure.canvas.draw_idle()
>         self.val = val
>         if self.eventson:
>             self._observers.process("changed", val)
>
>
> Operating system
>
> OSX
> Matplotlib Version
>
> 3.5.1
> Matplotlib Backend
>
> *No response*
> Python version
>
> 3.8
> Jupyter version
>
> *No response*
> Installation
>
> pip
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/matplotlib/matplotlib/issues/22686>, or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AACHF6CW2HVLKT5Q56BVZDLVBJ6X7ANCNFSM5RMUEIDQ>
> .
> You are receiving this because you are subscribed to this thread.Message
> ID: ***@***.***>
>

Yes, i might have been too fast, cause it allows to skip the error but then it seems that the polygon is not right...
Let me know if you know how this should be solved...
![Capture d’écran, le 2022-03-22 à 23 20 23](https://user-images.githubusercontent.com/37241971/159617326-44c69bfc-bf0a-4f79-ab23-925c7066f2c2.jpg)


So I think you found an edge case because your valinit has both values equal. This means that the poly object created by `axhspan` is not as large as the rest of the code expects. 

https://github.com/matplotlib/matplotlib/blob/11737d0694109f71d2603ba67d764aa2fb302761/lib/matplotlib/widgets.py#L722

A quick workaround is to have the valinit contain two different numbers (even if only minuscule difference)
Yes you are right!
Thanks a lot for digging into this!!
Compare:

\`\`\`python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
poly_same_valinit = ax.axvspan(0, 0, 0, 1)
poly_diff_valinit = ax.axvspan(0, .5, 0, 1)
print(poly_same_valinit.xy.shape)
print(poly_diff_valinit.xy.shape)
\`\`\`

which gives:

\`\`\`
(4, 2)
(5, 2)
\`\`\`

Two solutions spring to mind:

1. Easier option
Throw an error in init if `valinit[0] == valinit[1]`

2. Maybe better option?
Don't use axhspan and manually create the poly to ensure it always has the expected number of vertices
Option 2 might be better yes
@vpicouet any interest in opening a PR?
I don't think I am qualified to do so, never opened a PR yet. 
RangeSlider might also contain another issue. 
When I call `RangeSlider.set_val([0.0,0.1])`
It changes only the blue poly object and the range value on the right side of the slider not the dot:
![Capture d’écran, le 2022-03-25 à 15 53 44](https://user-images.githubusercontent.com/37241971/160191943-aef5fbe2-2f54-42ae-9719-23375767b212.jpg)
 
> I don't think I am qualified to do so, never opened a PR yet.

That's always true until you've opened your first one :). But I also understand that it can be intimidating.


>  RangeSlider might also contain another issue.
> When I call RangeSlider.set_val([0.0,0.1])
> It changes only the blue poly object and the range value on the right side of the slider not the dot:


oh hmm - good catch! may be worth opening a separate issue there as these are two distinct bugs and this one may be a bit more comlicated to fix.
Haha true! I might try when I have more time!
Throwing an error at least as I have never worked with axhspan and polys.
Ok, openning another issue.
Can I try working on this? @ianhi @vpicouet 
From the discussion, I could identify that a quick fix would be to use a try-except block to throw an error 
if valinit[0] == valinit[1]

Please let me know your thoughts.
Sure! 
@nik1097 anyone can work on any issue at any point - no need to ask :) The only thing you should do is check that theres not already a PR for it.

> From the discussion, I could identify that a quick fix would be to use a try-except block to throw an error
> if valinit[0] == valinit[1]

> Please let me know your thoughts.

while this would have made this an explicit error I think we should use this an opportunity to improve the functionality by using option 2 from https://github.com/matplotlib/matplotlib/issues/22686#issuecomment-1076496982

That creation code will look something like what axhspan does internally ( replacing `self` with `self.ax`)

https://github.com/matplotlib/matplotlib/blob/710fce3df95e22701bd68bf6af2c8adbc9d67a79/lib/matplotlib/axes/_axes.py#L988-L993
with the key difference that the `verts` variable should be defined as `verts = np.zeros([5,2])` and then the values should be filled the same way they are in the `set_val` method here: https://github.com/matplotlib/matplotlib/blob/2e921df22ba6b2a7782241798b042403c04cbdaf/lib/matplotlib/widgets.py#L901-L912

Hey, I guess the below PR should fix this issue too. If I'm not wrong, I will wait for the PR to be approved.
https://github.com/matplotlib/matplotlib/pull/22711
@nik1097 that PR doesn't address this issue so no need to wait for that one
Yes, it makes sense now. Will get back to you asap @ianhi 

Is someone still working on this ?
I still am but I'm currently caught up with another project. Feel free to give it a shot, thanks! @AnnaMastori 

should this be the expected outcome of the solution?
![Στιγμιότυπο οθόνης (104)](https://user-images.githubusercontent.com/72826029/165545175-aca35912-1b46-491d-8fb3-841e1ae4a1c4.png)

@NickolasGiannatos that looks like the correct starting position if you give an init value of `(0, 0)`. The next things to check are that it looks correct when you move the slider around, and that it looks correct with init values like `(.3, .5)`
@ianhi Me and @NickolasGiannatos work together. It works for (.3, .5). Is there anything else that we should try ?
![Στιγμιότυπο οθόνης (107)](https://user-images.githubusercontent.com/72812754/165580835-cc41e3f7-7761-42e2-b276-2e1f9c12fffa.png)
 
@AnnaMastori @NickolasGiannatos can you please open a PR with your changes? It makes it much easier to discuss if we can all look at the same code.
okay we will do it as soon as possible although it's our first time so there might be mistakes.
>  our first time so there might be mistakes.

That's fine! I had lots of mistakes my first time
```

## Patch

```diff
diff --git a/lib/matplotlib/widgets.py b/lib/matplotlib/widgets.py
--- a/lib/matplotlib/widgets.py
+++ b/lib/matplotlib/widgets.py
@@ -19,7 +19,7 @@
 from . import (_api, _docstring, backend_tools, cbook, colors, ticker,
                transforms)
 from .lines import Line2D
-from .patches import Circle, Rectangle, Ellipse
+from .patches import Circle, Rectangle, Ellipse, Polygon
 from .transforms import TransformedPatchPath, Affine2D
 
 
@@ -709,7 +709,7 @@ def __init__(
                 facecolor=track_color
             )
             ax.add_patch(self.track)
-            self.poly = ax.axhspan(valinit[0], valinit[1], 0, 1, **kwargs)
+            poly_transform = self.ax.get_yaxis_transform(which="grid")
             handleXY_1 = [.5, valinit[0]]
             handleXY_2 = [.5, valinit[1]]
         else:
@@ -719,9 +719,15 @@ def __init__(
                 facecolor=track_color
             )
             ax.add_patch(self.track)
-            self.poly = ax.axvspan(valinit[0], valinit[1], 0, 1, **kwargs)
+            poly_transform = self.ax.get_xaxis_transform(which="grid")
             handleXY_1 = [valinit[0], .5]
             handleXY_2 = [valinit[1], .5]
+        self.poly = Polygon(np.zeros([5, 2]), **kwargs)
+        self._update_selection_poly(*valinit)
+        self.poly.set_transform(poly_transform)
+        self.poly.get_path()._interpolation_steps = 100
+        self.ax.add_patch(self.poly)
+        self.ax._request_autoscale_view()
         self._handles = [
             ax.plot(
                 *handleXY_1,
@@ -777,6 +783,27 @@ def __init__(
         self._active_handle = None
         self.set_val(valinit)
 
+    def _update_selection_poly(self, vmin, vmax):
+        """
+        Update the vertices of the *self.poly* slider in-place
+        to cover the data range *vmin*, *vmax*.
+        """
+        # The vertices are positioned
+        #  1 ------ 2
+        #  |        |
+        # 0, 4 ---- 3
+        verts = self.poly.xy
+        if self.orientation == "vertical":
+            verts[0] = verts[4] = .25, vmin
+            verts[1] = .25, vmax
+            verts[2] = .75, vmax
+            verts[3] = .75, vmin
+        else:
+            verts[0] = verts[4] = vmin, .25
+            verts[1] = vmin, .75
+            verts[2] = vmax, .75
+            verts[3] = vmax, .25
+
     def _min_in_bounds(self, min):
         """Ensure the new min value is between valmin and self.val[1]."""
         if min <= self.valmin:
@@ -903,36 +930,24 @@ def set_val(self, val):
         """
         val = np.sort(val)
         _api.check_shape((2,), val=val)
-        val[0] = self._min_in_bounds(val[0])
-        val[1] = self._max_in_bounds(val[1])
-        xy = self.poly.xy
+        vmin, vmax = val
+        vmin = self._min_in_bounds(vmin)
+        vmax = self._max_in_bounds(vmax)
+        self._update_selection_poly(vmin, vmax)
         if self.orientation == "vertical":
-            xy[0] = .25, val[0]
-            xy[1] = .25, val[1]
-            xy[2] = .75, val[1]
-            xy[3] = .75, val[0]
-            xy[4] = .25, val[0]
-
-            self._handles[0].set_ydata([val[0]])
-            self._handles[1].set_ydata([val[1]])
+            self._handles[0].set_ydata([vmin])
+            self._handles[1].set_ydata([vmax])
         else:
-            xy[0] = val[0], .25
-            xy[1] = val[0], .75
-            xy[2] = val[1], .75
-            xy[3] = val[1], .25
-            xy[4] = val[0], .25
+            self._handles[0].set_xdata([vmin])
+            self._handles[1].set_xdata([vmax])
 
-            self._handles[0].set_xdata([val[0]])
-            self._handles[1].set_xdata([val[1]])
-
-        self.poly.xy = xy
-        self.valtext.set_text(self._format(val))
+        self.valtext.set_text(self._format((vmin, vmax)))
 
         if self.drawon:
             self.ax.figure.canvas.draw_idle()
-        self.val = val
+        self.val = (vmin, vmax)
         if self.eventson:
-            self._observers.process("changed", val)
+            self._observers.process("changed", (vmin, vmax))
 
     def on_changed(self, func):
         """

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_widgets.py b/lib/matplotlib/tests/test_widgets.py
--- a/lib/matplotlib/tests/test_widgets.py
+++ b/lib/matplotlib/tests/test_widgets.py
@@ -1161,6 +1161,23 @@ def handle_positions(slider):
     assert_allclose(handle_positions(slider), (0.1, 0.34))
 
 
+@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
+def test_range_slider_same_init_values(orientation):
+    if orientation == "vertical":
+        idx = [1, 0, 3, 2]
+    else:
+        idx = [0, 1, 2, 3]
+
+    fig, ax = plt.subplots()
+
+    slider = widgets.RangeSlider(
+         ax=ax, label="", valmin=0.0, valmax=1.0, orientation=orientation,
+         valinit=[0, 0]
+     )
+    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
+    assert_allclose(box.get_points().flatten()[idx], [0, 0.25, 0, 0.75])
+
+
 def check_polygon_selector(event_sequence, expected_result, selections_count,
                            **kwargs):
     """

```


## Code snippets

### 1 - lib/matplotlib/widgets.py:

Start line: 603, End line: 778

```python
class RangeSlider(SliderBase):

    def __init__(
        self,
        ax,
        label,
        valmin,
        valmax,
        valinit=None,
        valfmt=None,
        closedmin=True,
        closedmax=True,
        dragging=True,
        valstep=None,
        orientation="horizontal",
        track_color='lightgrey',
        handle_style=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : tuple of float or None, default: None
            The initial positions of the slider. If None the initial positions
            will be at the 25th and 75th percentiles of the range.

        valfmt : str, default: None
            %-format string used to format the slider values.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float, default: None
            If given, the slider will snap to multiples of *valstep*.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        track_color : color, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handles. Default values are

            ========= ===== ======= =========================================
            Key       Value Default Description
            ========= ===== ======= =========================================
            facecolor color 'white' The facecolor of the slider handles.
            edgecolor color '.75'   The edgecolor of the slider handles.
            size      int   10      The size of the slider handles in points.
            ========= ===== ======= =========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Polygon` that draws the slider knob.  See the
        `.Polygon` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
        super().__init__(ax, orientation, closedmin, closedmax,
                         valmin, valmax, valfmt, dragging, valstep)

        # Set a value to allow _value_in_bounds() to work.
        self.val = [valmin, valmax]
        if valinit is None:
            # Place at the 25th and 75th percentiles
            extent = valmax - valmin
            valinit = np.array([valmin + extent * 0.25,
                                valmin + extent * 0.75])
        else:
            valinit = self._value_in_bounds(valinit)
        self.val = valinit
        self.valinit = valinit

        defaults = {'facecolor': 'white', 'edgecolor': '.75', 'size': 10}
        handle_style = {} if handle_style is None else handle_style
        marker_props = {
            f'marker{k}': v for k, v in {**defaults, **handle_style}.items()
        }

        if orientation == "vertical":
            self.track = Rectangle(
                (.25, 0), .5, 2,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            self.poly = ax.axhspan(valinit[0], valinit[1], 0, 1, **kwargs)
            handleXY_1 = [.5, valinit[0]]
            handleXY_2 = [.5, valinit[1]]
        else:
            self.track = Rectangle(
                (0, .25), 1, .5,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            self.poly = ax.axvspan(valinit[0], valinit[1], 0, 1, **kwargs)
            handleXY_1 = [valinit[0], .5]
            handleXY_2 = [valinit[1], .5]
        self._handles = [
            ax.plot(
                *handleXY_1,
                "o",
                **marker_props,
                clip_on=False
            )[0],
            ax.plot(
                *handleXY_2,
                "o",
                **marker_props,
                clip_on=False
            )[0]
        ]

        if orientation == "vertical":
            self.label = ax.text(
                0.5,
                1.02,
                label,
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

            self.valtext = ax.text(
                0.5,
                -0.02,
                self._format(valinit),
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
            )
        else:
            self.label = ax.text(
                -0.02,
                0.5,
                label,
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="right",
            )

            self.valtext = ax.text(
                1.02,
                0.5,
                self._format(valinit),
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="left",
            )

        self._active_handle = None
        self.set_val(valinit)
```
### 2 - examples/widgets/range_slider.py:

Start line: 44, End line: 71

```python
def update(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the image's colormap
    im.norm.vmin = val[0]
    im.norm.vmax = val[1]

    # Update the position of the vertical lines
    lower_limit_line.set_xdata([val[0], val[0]])
    upper_limit_line.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.RangeSlider`
```
### 3 - lib/matplotlib/widgets.py:

Start line: 896, End line: 952

```python
class RangeSlider(SliderBase):

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
        val = np.sort(val)
        _api.check_shape((2,), val=val)
        val[0] = self._min_in_bounds(val[0])
        val[1] = self._max_in_bounds(val[1])
        xy = self.poly.xy
        if self.orientation == "vertical":
            xy[0] = .25, val[0]
            xy[1] = .25, val[1]
            xy[2] = .75, val[1]
            xy[3] = .75, val[0]
            xy[4] = .25, val[0]

            self._handles[0].set_ydata([val[0]])
            self._handles[1].set_ydata([val[1]])
        else:
            xy[0] = val[0], .25
            xy[1] = val[0], .75
            xy[2] = val[1], .75
            xy[3] = val[1], .25
            xy[4] = val[0], .25

            self._handles[0].set_xdata([val[0]])
            self._handles[1].set_xdata([val[1]])

        self.poly.xy = xy
        self.valtext.set_text(self._format(val))

        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process("changed", val)

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed. The function
            must accept a numpy array with shape (2,) as its argument.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))
```
### 4 - lib/matplotlib/widgets.py:

Start line: 331, End line: 485

```python
class Slider(SliderBase):

    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt=None,
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, valstep=None,
                 orientation='horizontal', *, initcolor='r',
                 track_color='lightgrey', handle_style=None, **kwargs):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : float, default: 0.5
            The slider initial position.

        valfmt : str, default: None
            %-format string used to format the slider value.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        slidermin : Slider, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider *slidermin*.

        slidermax : Slider, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider *slidermax*.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float or array-like, default: None
            If a float, the slider will snap to multiples of *valstep*.
            If an array the slider will snap to the values in the array.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        initcolor : color, default: 'r'
            The color of the line at the *valinit* position. Set to ``'none'``
            for no line.

        track_color : color, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handle. Default values are

            ========= ===== ======= ========================================
            Key       Value Default Description
            ========= ===== ======= ========================================
            facecolor color 'white' The facecolor of the slider handle.
            edgecolor color '.75'   The edgecolor of the slider handle.
            size      int   10      The size of the slider handle in points.
            ========= ===== ======= ========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Polygon` that draws the slider knob.  See the
        `.Polygon` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
        super().__init__(ax, orientation, closedmin, closedmax,
                         valmin, valmax, valfmt, dragging, valstep)

        if slidermin is not None and not hasattr(slidermin, 'val'):
            raise ValueError(
                f"Argument slidermin ({type(slidermin)}) has no 'val'")
        if slidermax is not None and not hasattr(slidermax, 'val'):
            raise ValueError(
                f"Argument slidermax ({type(slidermax)}) has no 'val'")
        self.slidermin = slidermin
        self.slidermax = slidermax
        valinit = self._value_in_bounds(valinit)
        if valinit is None:
            valinit = valmin
        self.val = valinit
        self.valinit = valinit

        defaults = {'facecolor': 'white', 'edgecolor': '.75', 'size': 10}
        handle_style = {} if handle_style is None else handle_style
        marker_props = {
            f'marker{k}': v for k, v in {**defaults, **handle_style}.items()
        }

        if orientation == 'vertical':
            self.track = Rectangle(
                (.25, 0), .5, 1,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            self.poly = ax.axhspan(valmin, valinit, .25, .75, **kwargs)
            # Drawing a longer line and clipping it to the track avoids
            # pixellization-related asymmetries.
            self.hline = ax.axhline(valinit, 0, 1, color=initcolor, lw=1,
                                    clip_path=TransformedPatchPath(self.track))
            handleXY = [[0.5], [valinit]]
        else:
            self.track = Rectangle(
                (0, .25), 1, .5,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            self.poly = ax.axvspan(valmin, valinit, .25, .75, **kwargs)
            self.vline = ax.axvline(valinit, 0, 1, color=initcolor, lw=1,
                                    clip_path=TransformedPatchPath(self.track))
            handleXY = [[valinit], [0.5]]
        self._handle, = ax.plot(
            *handleXY,
            "o",
            **marker_props,
            clip_on=False
        )

        if orientation == 'vertical':
            self.label = ax.text(0.5, 1.02, label, transform=ax.transAxes,
                                 verticalalignment='bottom',
                                 horizontalalignment='center')

            self.valtext = ax.text(0.5, -0.02, self._format(valinit),
                                   transform=ax.transAxes,
                                   verticalalignment='top',
                                   horizontalalignment='center')
        else:
            self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                                 verticalalignment='center',
                                 horizontalalignment='right')

            self.valtext = ax.text(1.02, 0.5, self._format(valinit),
                                   transform=ax.transAxes,
                                   verticalalignment='center',
                                   horizontalalignment='left')

        self.set_val(valinit)
```
### 5 - lib/matplotlib/widgets.py:

Start line: 545, End line: 585

```python
class Slider(SliderBase):

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : float
        """
        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[1] = .25, val
            xy[2] = .75, val
            self._handle.set_ydata([val])
        else:
            xy[2] = val, .75
            xy[3] = val, .25
            self._handle.set_xdata([val])
        self.poly.xy = xy
        self.valtext.set_text(self._format(val))
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process('changed', val)

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))
```
### 6 - lib/matplotlib/widgets.py:

Start line: 806, End line: 819

```python
class RangeSlider(SliderBase):

    def _update_val_from_pos(self, pos):
        """Update the slider value based on a given position."""
        idx = np.argmin(np.abs(self.val - pos))
        if idx == 0:
            val = self._min_in_bounds(pos)
            self.set_min(val)
        else:
            val = self._max_in_bounds(pos)
            self.set_max(val)
        if self._active_handle:
            if self.orientation == "vertical":
                self._active_handle.set_ydata([val])
            else:
                self._active_handle.set_xdata([val])
```
### 7 - lib/matplotlib/widgets.py:

Start line: 281, End line: 309

```python
class SliderBase(AxesWidget):

    def _stepped_value(self, val):
        """Return *val* coerced to closest number in the ``valstep`` grid."""
        if isinstance(self.valstep, Number):
            val = (self.valmin
                   + round((val - self.valmin) / self.valstep) * self.valstep)
        elif self.valstep is not None:
            valstep = np.asanyarray(self.valstep)
            if valstep.ndim != 1:
                raise ValueError(
                    f"valstep must have 1 dimension but has {valstep.ndim}"
                )
            val = valstep[np.argmin(np.abs(valstep - val))]
        return val

    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*.

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed.
        """
        self._observers.disconnect(cid)

    def reset(self):
        """Reset the slider to the initial value."""
        if np.any(self.val != self.valinit):
            self.set_val(self.valinit)
```
### 8 - lib/matplotlib/widgets.py:

Start line: 511, End line: 543

```python
class Slider(SliderBase):

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif ((event.name == 'button_release_event') or
              (event.name == 'button_press_event' and
               event.inaxes != self.ax)):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return
        if self.orientation == 'vertical':
            val = self._value_in_bounds(event.ydata)
        else:
            val = self._value_in_bounds(event.xdata)
        if val not in [None, self.val]:
            self.set_val(val)

    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return self.valfmt % val
        else:
            _, s, _ = self._fmt.format_ticks([self.valmin, val, self.valmax])
            # fmt.get_offset is actually the multiplicative factor, if any.
            return s + self._fmt.get_offset()
```
### 9 - examples/widgets/slider_snap_demo.py:

Start line: 1, End line: 83

```python
"""
===================================
Snapping Sliders to Discrete Values
===================================

You can snap slider values to discrete values using the ``valstep`` argument.

In this example the Freq slider is constrained to be multiples of pi, and the
Amp slider uses an array as the ``valstep`` argument to more densely sample
the first part of its range.

See :doc:`/gallery/widgets/slider_demo` for an example of using
a ``Slider`` to control a single float.

See :doc:`/gallery/widgets/range_slider` for an example of using
a ``RangeSlider`` to define a range of values.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0 * np.sin(2 * np.pi * f0 * t)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
l, = ax.plot(t, s, lw=2)

ax_freq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ax_amp = fig.add_axes([0.25, 0.15, 0.65, 0.03])

# define the values to use for snapping
allowed_amplitudes = np.concatenate([np.linspace(.1, 5, 100), [6, 7, 8, 9]])

# create the sliders
samp = Slider(
    ax_amp, "Amp", 0.1, 9.0,
    valinit=a0, valstep=allowed_amplitudes,
    color="green"
)

sfreq = Slider(
    ax_freq, "Freq", 0, 10*np.pi,
    valinit=2*np.pi, valstep=np.pi,
    initcolor='none'  # Remove the line marking the valinit position.
)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)

ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)


plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.Slider`
#    - `matplotlib.widgets.Button`
```
### 10 - lib/matplotlib/widgets.py:

Start line: 862, End line: 894

```python
class RangeSlider(SliderBase):

    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return f"({self.valfmt % val[0]}, {self.valfmt % val[1]})"
        else:
            _, s1, s2, _ = self._fmt.format_ticks(
                [self.valmin, *val, self.valmax]
            )
            # fmt.get_offset is actually the multiplicative factor, if any.
            s1 += self._fmt.get_offset()
            s2 += self._fmt.get_offset()
            # Use f string to avoid issues with backslashes when cast to a str
            return f"({s1}, {s2})"

    def set_min(self, min):
        """
        Set the lower value of the slider to *min*.

        Parameters
        ----------
        min : float
        """
        self.set_val((min, self.val[1]))

    def set_max(self, max):
        """
        Set the lower value of the slider to *max*.

        Parameters
        ----------
        max : float
        """
        self.set_val((self.val[0], max))
```
### 11 - lib/matplotlib/widgets.py:

Start line: 312, End line: 329

```python
class Slider(SliderBase):
    """
    A slider representing a floating point range.

    Create a slider from *valmin* to *valmax* in Axes *ax*. For the slider to
    remain responsive you must maintain a reference to it. Call
    :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : float
        Slider value.
    """

    cnt = _api.deprecated("3.4")(property(  # Not real, but close enough.
        lambda self: len(self._observers.callbacks['changed'])))
    observers = _api.deprecated("3.4")(property(
        lambda self: self._observers.callbacks['changed']))
```
### 12 - lib/matplotlib/widgets.py:

Start line: 821, End line: 860

```python
class RangeSlider(SliderBase):

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == "button_press_event" and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif (event.name == "button_release_event") or (
            event.name == "button_press_event" and event.inaxes != self.ax
        ):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            self._active_handle = None
            return

        # determine which handle was grabbed
        if self.orientation == "vertical":
            handle_index = np.argmin(
                np.abs([h.get_ydata()[0] - event.ydata for h in self._handles])
            )
        else:
            handle_index = np.argmin(
                np.abs([h.get_xdata()[0] - event.xdata for h in self._handles])
            )
        handle = self._handles[handle_index]

        # these checks ensure smooth behavior if the handles swap which one
        # has a higher value. i.e. if one is dragged over and past the other.
        if handle is not self._active_handle:
            self._active_handle = handle

        if self.orientation == "vertical":
            self._update_val_from_pos(event.ydata)
        else:
            self._update_val_from_pos(event.xdata)
```
### 13 - lib/matplotlib/widgets.py:

Start line: 780, End line: 804

```python
class RangeSlider(SliderBase):

    def _min_in_bounds(self, min):
        """Ensure the new min value is between valmin and self.val[1]."""
        if min <= self.valmin:
            if not self.closedmin:
                return self.val[0]
            min = self.valmin

        if min > self.val[1]:
            min = self.val[1]
        return self._stepped_value(min)

    def _max_in_bounds(self, max):
        """Ensure the new max value is between valmax and self.val[0]."""
        if max >= self.valmax:
            if not self.closedmax:
                return self.val[1]
            max = self.valmax

        if max <= self.val[0]:
            max = self.val[0]
        return self._stepped_value(max)

    def _value_in_bounds(self, vals):
        """Clip min, max values to the bounds."""
        return (self._min_in_bounds(vals[0]), self._max_in_bounds(vals[1]))
```
### 14 - lib/matplotlib/widgets.py:

Start line: 487, End line: 509

```python
class Slider(SliderBase):

    def _value_in_bounds(self, val):
        """Makes sure *val* is with given bounds."""
        val = self._stepped_value(val)

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val
```
### 15 - lib/matplotlib/widgets.py:

Start line: 588, End line: 601

```python
class RangeSlider(SliderBase):
    """
    A slider representing a range of floating point values. Defines the min and
    max of the range via the *val* attribute as a tuple of (min, max).

    Create a slider that defines a range contained within [*valmin*, *valmax*]
    in Axes *ax*. For the slider to remain responsive you must maintain a
    reference to it. Call :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : tuple of float
        Slider value.
    """
```
### 18 - lib/matplotlib/widgets.py:

Start line: 234, End line: 279

```python
class SliderBase(AxesWidget):
    """
    The base class for constructing Slider widgets. Not intended for direct
    usage.

    For the slider to remain responsive you must maintain a reference to it.
    """
    def __init__(self, ax, orientation, closedmin, closedmax,
                 valmin, valmax, valfmt, dragging, valstep):
        if ax.name == '3d':
            raise ValueError('Sliders cannot be added to 3D Axes')

        super().__init__(ax)
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)

        self.orientation = orientation
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.valmin = valmin
        self.valmax = valmax
        self.valstep = valstep
        self.drag_active = False
        self.valfmt = valfmt

        if orientation == "vertical":
            ax.set_ylim((valmin, valmax))
            axis = ax.yaxis
        else:
            ax.set_xlim((valmin, valmax))
            axis = ax.xaxis

        self._fmt = axis.get_major_formatter()
        if not isinstance(self._fmt, ticker.ScalarFormatter):
            self._fmt = ticker.ScalarFormatter()
            self._fmt.set_axis(axis)
        self._fmt.set_useOffset(False)  # No additive offset.
        self._fmt.set_useMathText(True)  # x sign before multiplicative offset.

        ax.set_axis_off()
        ax.set_navigate(False)

        self.connect_event("button_press_event", self._update)
        self.connect_event("button_release_event", self._update)
        if dragging:
            self.connect_event("motion_notify_event", self._update)
        self._observers = cbook.CallbackRegistry(signals=["changed"])
```
### 21 - lib/matplotlib/widgets.py:

Start line: 1280, End line: 1309

```python
class TextBox(AxesWidget):

    def set_val(self, val):
        newval = str(val)
        if self.text == newval:
            return
        self.text_disp.set_text(newval)
        self._rendercursor()
        if self.eventson:
            self._observers.process('change', self.text)
            self._observers.process('submit', self.text)

    def begin_typing(self, x):
        self.capturekeystrokes = True
        # Disable keypress shortcuts, which may otherwise cause the figure to
        # be saved, closed, etc., until the user stops typing.  The way to
        # achieve this depends on whether toolmanager is in use.
        stack = ExitStack()  # Register cleanup actions when user stops typing.
        self._on_stop_typing = stack.close
        toolmanager = getattr(
            self.ax.figure.canvas.manager, "toolmanager", None)
        if toolmanager is not None:
            # If using toolmanager, lock keypresses, and plan to release the
            # lock when typing stops.
            toolmanager.keypresslock(self)
            stack.callback(toolmanager.keypresslock.release, self)
        else:
            # If not using toolmanager, disable all keypress-related rcParams.
            # Avoid spurious warnings if keymaps are getting deprecated.
            with _api.suppress_matplotlib_deprecation_warning():
                stack.enter_context(mpl.rc_context(
                    {k: [] for k in mpl.rcParams if k.startswith("keymap.")}))
```
