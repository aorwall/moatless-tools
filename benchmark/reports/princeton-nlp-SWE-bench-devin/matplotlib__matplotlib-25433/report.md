# matplotlib__matplotlib-25433

| **matplotlib/matplotlib** | `7eafdd8af3c523c1c77b027d378fb337dd489f18` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 12 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -931,6 +931,7 @@ def _break_share_link(ax, grouper):
         self._axobservers.process("_axes_change_event", self)
         self.stale = True
         self._localaxes.remove(ax)
+        self.canvas.release_mouse(ax)
 
         # Break link between any shared axes
         for name in ax._axis_names:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/figure.py | 934 | 934 | - | 12 | -


## Problem Statement

```
[Bug]: using clf and pyplot.draw in range slider on_changed callback blocks input to widgets
### Bug summary

When using clear figure, adding new widgets and then redrawing the current figure in the on_changed callback of a range slider the inputs to all the widgets in the figure are blocked. When doing the same in the button callback on_clicked, everything works fine.

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets

def onchanged(values):
    print("on changed")
    print(values)
    pyplot.clf()
    addElements()
    pyplot.draw()

def onclick(e):
    print("on click")
    pyplot.clf()
    addElements()
    pyplot.draw()

def addElements():
    ax = pyplot.axes([0.1, 0.45, 0.8, 0.1])
    global slider
    slider = widgets.RangeSlider(ax, "Test", valmin=1, valmax=10, valinit=(1, 10))
    slider.on_changed(onchanged)
    ax = pyplot.axes([0.1, 0.30, 0.8, 0.1])
    global button
    button = widgets.Button(ax, "Test")
    button.on_clicked(onclick)

addElements()

pyplot.show()
\`\`\`


### Actual outcome

The widgets can't receive any input from a mouse click, when redrawing in the on_changed callback of a range Slider. 
When using a button, there is no problem.

### Expected outcome

The range slider callback on_changed behaves the same as the button callback on_clicked.

### Additional information

The problem also occurred on Manjaro with:
- Python version: 3.10.9
- Matplotlib version: 3.6.2
- Matplotlib backend: QtAgg
- Installation of matplotlib via Linux package manager


### Operating system

Windows 10

### Matplotlib Version

3.6.2

### Matplotlib Backend

TkAgg

### Python version

3.11.0

### Jupyter version

_No response_

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 lib/matplotlib/widgets.py | 1 | 331| 2375 | 2375 | 34490 | 
| 2 | 2 galleries/examples/widgets/range_slider.py | 45 | 72| 168 | 2543 | 34998 | 
| 3 | 2 lib/matplotlib/widgets.py | 334 | 1143| 6393 | 8936 | 34998 | 
| 4 | 3 lib/matplotlib/backends/backend_gtk3.py | 134 | 194| 568 | 9504 | 39992 | 
| 5 | 4 galleries/examples/widgets/check_buttons.py | 1 | 61| 440 | 9944 | 40432 | 
| 6 | 5 lib/matplotlib/backends/_backend_tk.py | 325 | 343| 205 | 10149 | 50110 | 
| 7 | 5 lib/matplotlib/widgets.py | 1198 | 1942| 6229 | 16378 | 50110 | 
| 8 | 6 lib/matplotlib/backend_bases.py | 2411 | 2780| 2862 | 19240 | 79149 | 
| 9 | 6 galleries/examples/widgets/range_slider.py | 1 | 42| 340 | 19580 | 79149 | 
| 10 | 7 galleries/examples/widgets/slider_snap_demo.py | 1 | 84| 584 | 20164 | 79733 | 
| 11 | 8 galleries/examples/user_interfaces/pylab_with_gtk3_sgskip.py | 1 | 62| 388 | 20552 | 80121 | 
| 12 | 9 lib/matplotlib/backends/backend_gtk4.py | 115 | 176| 513 | 21065 | 84882 | 
| 13 | 10 galleries/examples/widgets/slider_demo.py | 1 | 93| 591 | 21656 | 85473 | 
| 14 | 11 galleries/examples/widgets/rectangle_selector.py | 31 | 75| 335 | 21991 | 86038 | 
| 15 | **12 lib/matplotlib/figure.py** | 3413 | 3455| 432 | 22423 | 115514 | 
| 16 | 12 lib/matplotlib/backends/_backend_tk.py | 313 | 323| 137 | 22560 | 115514 | 
| 17 | 12 lib/matplotlib/backends/_backend_tk.py | 164 | 235| 738 | 23298 | 115514 | 
| 18 | 13 galleries/examples/user_interfaces/pylab_with_gtk4_sgskip.py | 1 | 53| 335 | 23633 | 115849 | 
| 19 | 14 lib/matplotlib/backends/backend_webagg_core.py | 309 | 333| 240 | 23873 | 119875 | 
| 20 | 14 lib/matplotlib/backends/backend_gtk3.py | 273 | 298| 172 | 24045 | 119875 | 
| 21 | 15 galleries/examples/event_handling/ginput_manual_clabel_sgskip.py | 1 | 101| 631 | 24676 | 120506 | 
| 22 | 15 lib/matplotlib/backends/backend_gtk3.py | 55 | 110| 524 | 25200 | 120506 | 
| 23 | 16 galleries/examples/event_handling/data_browser.py | 91 | 111| 160 | 25360 | 121269 | 
| 24 | 16 lib/matplotlib/backends/backend_gtk3.py | 243 | 271| 282 | 25642 | 121269 | 
| 25 | 17 galleries/examples/widgets/annotated_cursor.py | 288 | 357| 568 | 26210 | 124224 | 
| 26 | 18 lib/matplotlib/backends/backend_gtk3agg.py | 50 | 75| 246 | 26456 | 124820 | 
| 27 | 19 galleries/examples/event_handling/path_editor.py | 92 | 123| 230 | 26686 | 125931 | 
| 28 | 19 lib/matplotlib/backends/backend_webagg_core.py | 283 | 308| 311 | 26997 | 125931 | 
| 29 | 19 lib/matplotlib/backends/backend_gtk4.py | 262 | 287| 172 | 27169 | 125931 | 
| 30 | 19 lib/matplotlib/backends/backend_gtk4.py | 225 | 260| 337 | 27506 | 125931 | 
| 31 | 19 lib/matplotlib/backends/backend_gtk3.py | 207 | 241| 351 | 27857 | 125931 | 
| 32 | 20 galleries/examples/user_interfaces/embedding_in_tk_sgskip.py | 1 | 67| 466 | 28323 | 126397 | 
| 33 | 21 galleries/examples/subplots_axes_and_figures/auto_subplots_adjust.py | 69 | 88| 140 | 28463 | 127112 | 
| 34 | 22 lib/matplotlib/backends/backend_qt.py | 887 | 895| 140 | 28603 | 135750 | 
| 35 | 23 galleries/examples/widgets/radio_buttons.py | 1 | 75| 559 | 29162 | 136309 | 
| 36 | 24 galleries/examples/user_interfaces/embedding_in_gtk3_sgskip.py | 1 | 43| 246 | 29408 | 136555 | 
| 37 | 25 galleries/examples/user_interfaces/mpl_with_glade3_sgskip.py | 1 | 55| 309 | 29717 | 136864 | 
| 38 | 25 lib/matplotlib/backends/backend_gtk4.py | 33 | 91| 466 | 30183 | 136864 | 
| 39 | 26 lib/matplotlib/backends/_backend_gtk.py | 311 | 333| 180 | 30363 | 139385 | 
| 40 | 27 galleries/examples/widgets/buttons.py | 1 | 61| 384 | 30747 | 139769 | 
| 41 | 27 galleries/examples/subplots_axes_and_figures/auto_subplots_adjust.py | 52 | 66| 151 | 30898 | 139769 | 
| 42 | 28 galleries/examples/user_interfaces/fourier_demo_wx_sgskip.py | 68 | 103| 299 | 31197 | 141896 | 
| 43 | 28 lib/matplotlib/backends/backend_qt.py | 269 | 313| 368 | 31565 | 141896 | 
| 44 | 29 galleries/examples/user_interfaces/mplcvd.py | 181 | 211| 240 | 31805 | 144175 | 
| 45 | 30 galleries/examples/widgets/lasso_selector_demo_sgskip.py | 76 | 112| 239 | 32044 | 144951 | 
| 46 | **30 lib/matplotlib/figure.py** | 3457 | 3475| 153 | 32197 | 144951 | 
| 47 | 31 lib/matplotlib/backends/qt_editor/figureoptions.py | 177 | 264| 735 | 32932 | 147127 | 
| 48 | 32 galleries/examples/user_interfaces/embedding_in_gtk4_sgskip.py | 1 | 48| 300 | 33232 | 147427 | 
| 49 | 32 galleries/examples/user_interfaces/mplcvd.py | 214 | 230| 157 | 33389 | 147427 | 
| 50 | 33 galleries/examples/event_handling/zoom_window.py | 1 | 55| 442 | 33831 | 147869 | 
| 51 | 34 lib/matplotlib/testing/widgets.py | 1 | 64| 420 | 34251 | 148757 | 
| 52 | 34 galleries/examples/user_interfaces/mplcvd.py | 116 | 178| 452 | 34703 | 148757 | 
| 53 | 35 galleries/examples/event_handling/figure_axes_enter_leave.py | 1 | 53| 323 | 35026 | 149080 | 
| 54 | 35 lib/matplotlib/backends/_backend_tk.py | 387 | 438| 419 | 35445 | 149080 | 
| 55 | 35 lib/matplotlib/backends/backend_gtk4.py | 210 | 223| 139 | 35584 | 149080 | 
| 56 | 36 galleries/examples/user_interfaces/embedding_in_wx3_sgskip.py | 24 | 55| 256 | 35840 | 150221 | 
| 57 | 36 galleries/examples/widgets/annotated_cursor.py | 260 | 285| 187 | 36027 | 150221 | 
| 58 | 36 lib/matplotlib/backends/_backend_tk.py | 1 | 54| 364 | 36391 | 150221 | 
| 59 | 37 lib/matplotlib/pyplot.py | 994 | 1016| 156 | 36547 | 178931 | 
| 60 | 37 lib/matplotlib/backends/_backend_gtk.py | 268 | 308| 376 | 36923 | 178931 | 
| 61 | 37 lib/matplotlib/backends/_backend_tk.py | 345 | 357| 154 | 37077 | 178931 | 
| 62 | 38 lib/matplotlib/backends/backend_wx.py | 10 | 60| 343 | 37420 | 191046 | 
| 63 | 38 lib/matplotlib/backends/backend_gtk3.py | 196 | 205| 121 | 37541 | 191046 | 
| 64 | 38 lib/matplotlib/backends/backend_webagg_core.py | 157 | 200| 387 | 37928 | 191046 | 
| 65 | 39 lib/matplotlib/backends/backend_gtk3cairo.py | 1 | 28| 226 | 38154 | 191273 | 
| 66 | 40 lib/matplotlib/backend_tools.py | 799 | 837| 330 | 38484 | 198613 | 
| 67 | 40 lib/matplotlib/backends/_backend_tk.py | 688 | 706| 154 | 38638 | 198613 | 
| 68 | 40 lib/matplotlib/backends/_backend_tk.py | 950 | 961| 137 | 38775 | 198613 | 


### Hint

```
A can confirm this behavior, but removing and recreating the objects that host the callbacks in the callbacks is definitely on the edge of the intended usage.  

Why are you doing this?  In your application can you get away with not destroying your slider?
I think there could be a way to not destroy the slider. But I don't have the time to test that currently.
My workaround for the problem was using a button to redraw everything. With that everything is working fine.

That was the weird part for me as they are both callbacks, but in one everything works fine and in the other one all inputs are blocked. If what I'm trying doing is not the intended usage, that's fine for me. 
Thanks for the answer.
The idiomatic way to destructively work on widgets that triggered an event in a UI toolkit is to do the destructive work in an idle callback:
\`\`\`python
def redraw():
    pyplot.clf()
    addElements()
    pyplot.draw()
    return False

def onchanged(values):
    print("on changed")
    print(values)
    pyplot.gcf().canvas.new_timer(redraw)
\`\`\`
Thanks for the answer. I tried that with the code in the original post.
The line:

\`\`\`python
pyplot.gcf().canvas.new_timer(redraw)
\`\`\`
doesn't work for me. After having a look at the documentation, I think the line should be:

\`\`\`python
pyplot.gcf().canvas.new_timer(callbacks=[(redraw, tuple(), dict())])
\`\`\`

But that still didn't work. The redraw callback doesn't seem to trigger.
Sorry, I mean to update that after testing and before posting, but forgot to. That line should be:
\`\`\`
    pyplot.gcf().canvas.new_timer(callbacks=[redraw])
\`\`\`
Sorry for double posting; now I see that it didn't actually get called!

That's because I forgot to call `start` as well, and the timer was garbage collected at the end of the function. It should be called as you've said, but stored globally and started:
\`\`\`
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets


def redraw():
    print("redraw")
    pyplot.clf()
    addElements()
    pyplot.draw()
    return False


def onchanged(values):
    print("on changed")
    print(values)
    global timer
    timer = pyplot.gcf().canvas.new_timer(callbacks=[(redraw, (), {})])
    timer.start()


def onclick(e):
    print("on click")
    global timer
    timer = pyplot.gcf().canvas.new_timer(callbacks=[(redraw, (), {})])
    timer.start()


def addElements():
    ax = pyplot.axes([0.1, 0.45, 0.8, 0.1])
    global slider
    slider = widgets.RangeSlider(ax, "Test", valmin=1, valmax=10, valinit=(1, 10))
    slider.on_changed(onchanged)
    ax = pyplot.axes([0.1, 0.30, 0.8, 0.1])
    global button
    button = widgets.Button(ax, "Test")
    button.on_clicked(onclick)


addElements()

pyplot.show()
\`\`\`
Thanks for the answer, the code works without errors, but I'm still able to break it when using the range slider. Then it results in the same problem as my original post. It seems like that happens when triggering the onchanged callback at the same time the timer callback gets called.
Are you sure it's not working, or is it just that it got replaced by a new one that is using the same initial value again? For me, it moves, but once the callback runs, it's reset because the slider is created with `valinit=(1, 10)`.
The code redraws everything fine, but when the onchanged callback of the range slider gets called at the same time as the redraw callback of the timer, I'm not able to give any new inputs to the widgets. You can maybe reproduce that with changing the value of the slider the whole time, while waiting for the redraw.
I think what is happening is that because you are destroying the slider every time the you are also destroying the state of what value you changed it to.  When you create it you re-create it at the default value which produces the effect that you can not seem to change it.  Put another way, the state of what the current value of the slider is is in the `Slider` object.   Replacing the slider object with a new one (correctly) forgets the old value.

-----

I agree it fails in surprising ways, but I think that this is best case "undefined behavior" and worst case incorrect usage of the tools.  In either case there is not much we can do upstream to address this (as if the user _did_ want to get fresh sliders that is not a case we should prevent from happening or warn on).
The "forgetting the value" is not the problem. My problem is, that the input just blocks for every widget in the figure. When that happens, you can click on any widget, but no callback gets triggered. That problem seems to happen when a callback of an object gets called that is/has been destroyed, but I don't know for sure.

But if that is from the incorrect usage of the tools, that's fine for me. I got a decent workaround that works currently, so I just have to keep that in my code for now.
```

## Patch

```diff
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -931,6 +931,7 @@ def _break_share_link(ax, grouper):
         self._axobservers.process("_axes_change_event", self)
         self.stale = True
         self._localaxes.remove(ax)
+        self.canvas.release_mouse(ax)
 
         # Break link between any shared axes
         for name in ax._axis_names:

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_backend_bases.py b/lib/matplotlib/tests/test_backend_bases.py
--- a/lib/matplotlib/tests/test_backend_bases.py
+++ b/lib/matplotlib/tests/test_backend_bases.py
@@ -95,6 +95,16 @@ def test_non_gui_warning(monkeypatch):
                 in str(rec[0].message))
 
 
+def test_grab_clear():
+    fig, ax = plt.subplots()
+
+    fig.canvas.grab_mouse(ax)
+    assert fig.canvas.mouse_grabber == ax
+
+    fig.clear()
+    assert fig.canvas.mouse_grabber is None
+
+
 @pytest.mark.parametrize(
     "x, y", [(42, 24), (None, 42), (None, None), (200, 100.01), (205.75, 2.0)])
 def test_location_event_position(x, y):

```


## Code snippets

### 1 - lib/matplotlib/widgets.py:

Start line: 1, End line: 331

```python
"""
GUI neutral widgets
===================

Widgets that are designed to work for any of the GUI backends.
All of these widgets require you to predefine a `matplotlib.axes.Axes`
instance and pass that as the first parameter.  Matplotlib doesn't try to
be too smart with respect to layout -- you will have to figure out how
wide and tall you want your Axes to be to accommodate your widget.
"""

from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number

from cycler import cycler
import numpy as np

import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
               text as mtext, ticker, transforms)
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D


class LockDraw:
    """
    Some widgets, like the cursor, draw onto the canvas, and this is not
    desirable under all circumstances, like when the toolbar is in zoom-to-rect
    mode and drawing a rectangle.  To avoid this, a widget can acquire a
    canvas' lock with ``canvas.widgetlock(widget)`` before drawing on the
    canvas; this will prevent other widgets from doing so at the same time (if
    they also try to acquire the lock first).
    """

    def __init__(self):
        self._owner = None

    def __call__(self, o):
        """Reserve the lock for *o*."""
        if not self.available(o):
            raise ValueError('already locked')
        self._owner = o

    def release(self, o):
        """Release the lock from *o*."""
        if not self.available(o):
            raise ValueError('you do not own this lock')
        self._owner = None

    def available(self, o):
        """Return whether drawing is available to *o*."""
        return not self.locked() or self.isowner(o)

    def isowner(self, o):
        """Return whether *o* owns this lock."""
        return self._owner is o

    def locked(self):
        """Return whether the lock is currently held by an owner."""
        return self._owner is not None


class Widget:
    """
    Abstract base class for GUI neutral widgets.
    """
    drawon = True
    eventson = True
    _active = True

    def set_active(self, active):
        """Set whether the widget is active."""
        self._active = active

    def get_active(self):
        """Get whether the widget is active."""
        return self._active

    # set_active is overridden by SelectorWidgets.
    active = property(get_active, set_active, doc="Is the widget active?")

    def ignore(self, event):
        """
        Return whether *event* should be ignored.

        This method should be called at the beginning of any event callback.
        """
        return not self.active

    def _changed_canvas(self):
        """
        Someone has switched the canvas on us!

        This happens if `savefig` needs to save to a format the previous
        backend did not support (e.g. saving a figure using an Agg based
        backend saved to a vector format).

        Returns
        -------
        bool
           True if the canvas has been changed.

        """
        return self.canvas is not self.ax.figure.canvas


class AxesWidget(Widget):
    """
    Widget connected to a single `~matplotlib.axes.Axes`.

    To guarantee that the widget remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the widget.  If there are no references to the widget
    object it may be garbage collected which will disconnect the callbacks.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    canvas : `~matplotlib.backend_bases.FigureCanvasBase`
        The parent figure canvas for the widget.
    active : bool
        If False, the widget does not respond to events.
    """

    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self._cids = []

    def connect_event(self, event, callback):
        """
        Connect a callback function with an event.

        This should be used in lieu of ``figure.canvas.mpl_connect`` since this
        function stores callback ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self._cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        for c in self._cids:
            self.canvas.mpl_disconnect(c)


class Button(AxesWidget):
    """
    A GUI neutral button.

    For the button to remain responsive you must keep a reference to it.
    Call `.on_clicked` to connect to the button.

    Attributes
    ----------
    ax
        The `matplotlib.axes.Axes` the button renders into.
    label
        A `matplotlib.text.Text` instance.
    color
        The color of the button when not hovering.
    hovercolor
        The color of the button when hovering.
    """

    def __init__(self, ax, label, image=None,
                 color='0.85', hovercolor='0.95', *, useblit=True):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            The button text.
        image : array-like or PIL Image
            The image to place in the button, if not *None*.  The parameter is
            directly forwarded to `~matplotlib.axes.Axes.imshow`.
        color : color
            The color of the button when not activated.
        hovercolor : color
            The color of the button when the mouse is over it.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :doc:`/tutorials/advanced/blitting` for details.

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        if image is not None:
            ax.imshow(image)
        self.label = ax.text(0.5, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='center',
                             transform=ax.transAxes)

        self._useblit = useblit and self.canvas.supports_blit

        self._observers = cbook.CallbackRegistry(signals=["clicked"])

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        ax.set_navigate(False)
        ax.set_facecolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.color = color
        self.hovercolor = hovercolor

    def _click(self, event):
        if self.ignore(event) or event.inaxes != self.ax or not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)

    def _release(self, event):
        if self.ignore(event) or event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)
        if self.eventson and event.inaxes == self.ax:
            self._observers.process('clicked', event)

    def _motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if event.inaxes == self.ax else self.color
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                if self._useblit:
                    self.ax.draw_artist(self.ax)
                    self.canvas.blit(self.ax.bbox)
                else:
                    self.canvas.draw()

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Returns a connection id, which can be used to disconnect the callback.
        """
        return self._observers.connect('clicked', lambda event: func(event))

    def disconnect(self, cid):
        """Remove the callback function with connection id *cid*."""
        self._observers.disconnect(cid)


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
### 2 - galleries/examples/widgets/range_slider.py:

Start line: 45, End line: 72

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

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.RangeSlider`
```
### 3 - lib/matplotlib/widgets.py:

Start line: 334, End line: 1143

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

    @_api.make_keyword_only("3.7", name="valinit")
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
            # pixelation-related asymmetries.
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

    @_api.make_keyword_only("3.7", name="valinit")
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
        self.val = (valmin, valmax)
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
            poly_transform = self.ax.get_yaxis_transform(which="grid")
            handleXY_1 = [.5, valinit[0]]
            handleXY_2 = [.5, valinit[1]]
        else:
            self.track = Rectangle(
                (0, .25), 1, .5,
                transform=ax.transAxes,
                facecolor=track_color
            )
            ax.add_patch(self.track)
            poly_transform = self.ax.get_xaxis_transform(which="grid")
            handleXY_1 = [valinit[0], .5]
            handleXY_2 = [valinit[1], .5]
        self.poly = Polygon(np.zeros([5, 2]), **kwargs)
        self._update_selection_poly(*valinit)
        self.poly.set_transform(poly_transform)
        self.poly.get_path()._interpolation_steps = 100
        self.ax.add_patch(self.poly)
        self.ax._request_autoscale_view()
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

    def _update_selection_poly(self, vmin, vmax):
        """
        Update the vertices of the *self.poly* slider in-place
        to cover the data range *vmin*, *vmax*.
        """
        # The vertices are positioned
        #  1 ------ 2
        #  |        |
        # 0, 4 ---- 3
        verts = self.poly.xy
        if self.orientation == "vertical":
            verts[0] = verts[4] = .25, vmin
            verts[1] = .25, vmax
            verts[2] = .75, vmax
            verts[3] = .75, vmin
        else:
            verts[0] = verts[4] = vmin, .25
            verts[1] = vmin, .75
            verts[2] = vmax, .75
            verts[3] = vmax, .25

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

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
        val = np.sort(val)
        _api.check_shape((2,), val=val)
        # Reset value to allow _value_in_bounds() to work.
        self.val = (self.valmin, self.valmax)
        vmin, vmax = self._value_in_bounds(val)
        self._update_selection_poly(vmin, vmax)
        if self.orientation == "vertical":
            self._handles[0].set_ydata([vmin])
            self._handles[1].set_ydata([vmax])
        else:
            self._handles[0].set_xdata([vmin])
            self._handles[1].set_xdata([vmax])

        self.valtext.set_text(self._format((vmin, vmax)))

        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = (vmin, vmax)
        if self.eventson:
            self._observers.process("changed", (vmin, vmax))

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed. The function
            must accept a 2-tuple of floats as its argument.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))


def _expand_text_props(props):
    props = cbook.normalize_kwargs(props, mtext.Text)
    return cycler(**props)() if props else itertools.repeat({})


class CheckButtons(AxesWidget):
    r"""
    A GUI neutral set of check buttons.

    For the check buttons to remain responsive you must keep a
    reference to this object.

    Connect to the CheckButtons with the `.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    labels : list of `.Text`

    rectangles : list of `.Rectangle`

    lines : list of (`.Line2D`, `.Line2D`) pairs
        List of lines for the x's in the checkboxes.  These lines exist for
        each box, but have ``set_visible(False)`` when its box is not checked.
    """

    def __init__(self, ax, labels, actives=None, *, useblit=True,
                 label_props=None, frame_props=None, check_props=None):
        """
        Add check buttons to `matplotlib.axes.Axes` instance *ax*.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The parent Axes for the widget.
        labels : list of str
            The labels of the check buttons.
        actives : list of bool, optional
            The initial check states of the buttons. The list must have the
            same length as *labels*. If not given, all buttons are unchecked.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :doc:`/tutorials/advanced/blitting` for details.

            .. versionadded:: 3.7
        label_props : dict, optional
            Dictionary of `.Text` properties to be used for the labels.

            .. versionadded:: 3.7
        frame_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            check button frame. Defaults (label font size / 2)**2 size, black
            edgecolor, no facecolor, and 1.0 linewidth.

            .. versionadded:: 3.7
        check_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            check button check. Defaults to (label font size / 2)**2 size,
            black color, and 1.0 linewidth.

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        _api.check_isinstance((dict, None), label_props=label_props,
                              frame_props=frame_props, check_props=check_props)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        if actives is None:
            actives = [False] * len(labels)

        self._useblit = useblit and self.canvas.supports_blit
        self._background = None

        ys = np.linspace(1, 0, len(labels)+2)[1:-1]

        label_props = _expand_text_props(label_props)
        self.labels = [
            ax.text(0.25, y, label, transform=ax.transAxes,
                    horizontalalignment="left", verticalalignment="center",
                    **props)
            for y, label, props in zip(ys, labels, label_props)]
        text_size = np.array([text.get_fontsize() for text in self.labels]) / 2

        frame_props = {
            's': text_size**2,
            'linewidth': 1,
            **cbook.normalize_kwargs(frame_props, collections.PathCollection),
            'marker': 's',
            'transform': ax.transAxes,
        }
        frame_props.setdefault('facecolor', frame_props.get('color', 'none'))
        frame_props.setdefault('edgecolor', frame_props.pop('color', 'black'))
        self._frames = ax.scatter([0.15] * len(ys), ys, **frame_props)
        check_props = {
            'linewidth': 1,
            's': text_size**2,
            **cbook.normalize_kwargs(check_props, collections.PathCollection),
            'marker': 'x',
            'transform': ax.transAxes,
            'animated': self._useblit,
        }
        check_props.setdefault('facecolor', check_props.pop('color', 'black'))
        self._checks = ax.scatter([0.15] * len(ys), ys, **check_props)
        # The user may have passed custom colours in check_props, so we need to
        # create the checks (above), and modify the visibility after getting
        # whatever the user set.
        self._init_status(actives)

        self.connect_event('button_press_event', self._clicked)
        if self._useblit:
            self.connect_event('draw_event', self._clear)

        self._observers = cbook.CallbackRegistry(signals=["clicked"])

    def _clear(self, event):
        """Internal event handler to clear the buttons."""
        if self.ignore(event) or self._changed_canvas():
            return
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._checks)
        if hasattr(self, '_lines'):
            for l1, l2 in self._lines:
                self.ax.draw_artist(l1)
                self.ax.draw_artist(l2)

    def _clicked(self, event):
        if self.ignore(event) or event.button != 1 or event.inaxes != self.ax:
            return
        pclicked = self.ax.transAxes.inverted().transform((event.x, event.y))
        distances = {}
        if hasattr(self, "_rectangles"):
            for i, (p, t) in enumerate(zip(self._rectangles, self.labels)):
                x0, y0 = p.get_xy()
                if (t.get_window_extent().contains(event.x, event.y)
                        or (x0 <= pclicked[0] <= x0 + p.get_width()
                            and y0 <= pclicked[1] <= y0 + p.get_height())):
                    distances[i] = np.linalg.norm(pclicked - p.get_center())
        else:
            _, frame_inds = self._frames.contains(event)
            coords = self._frames.get_offset_transform().transform(
                self._frames.get_offsets()
            )
            for i, t in enumerate(self.labels):
                if (i in frame_inds["ind"]
                        or t.get_window_extent().contains(event.x, event.y)):
                    distances[i] = np.linalg.norm(pclicked - coords[i])
        if len(distances) > 0:
            closest = min(distances, key=distances.get)
            self.set_active(closest)
```
### 4 - lib/matplotlib/backends/backend_gtk3.py:

Start line: 134, End line: 194

```python
class FigureCanvasGTK3(_FigureCanvasGTK, Gtk.DrawingArea):

    def scroll_event(self, widget, event):
        step = 1 if event.direction == Gdk.ScrollDirection.UP else -1
        MouseEvent("scroll_event", self,
                   *self._mpl_coords(event), step=step,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def button_press_event(self, widget, event):
        MouseEvent("button_press_event", self,
                   *self._mpl_coords(event), event.button,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def button_release_event(self, widget, event):
        MouseEvent("button_release_event", self,
                   *self._mpl_coords(event), event.button,
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def key_press_event(self, widget, event):
        KeyEvent("key_press_event", self,
                 self._get_key(event), *self._mpl_coords(),
                 guiEvent=event)._process()
        return True  # stop event propagation

    def key_release_event(self, widget, event):
        KeyEvent("key_release_event", self,
                 self._get_key(event), *self._mpl_coords(),
                 guiEvent=event)._process()
        return True  # stop event propagation

    def motion_notify_event(self, widget, event):
        MouseEvent("motion_notify_event", self, *self._mpl_coords(event),
                   modifiers=self._mpl_modifiers(event.state),
                   guiEvent=event)._process()
        return False  # finish event propagation?

    def enter_notify_event(self, widget, event):
        gtk_mods = Gdk.Keymap.get_for_display(
            self.get_display()).get_modifier_state()
        LocationEvent("figure_enter_event", self, *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(gtk_mods),
                      guiEvent=event)._process()

    def leave_notify_event(self, widget, event):
        gtk_mods = Gdk.Keymap.get_for_display(
            self.get_display()).get_modifier_state()
        LocationEvent("figure_leave_event", self, *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(gtk_mods),
                      guiEvent=event)._process()

    def size_allocate(self, widget, allocation):
        dpival = self.figure.dpi
        winch = allocation.width * self.device_pixel_ratio / dpival
        hinch = allocation.height * self.device_pixel_ratio / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        ResizeEvent("resize_event", self)._process()
        self.draw_idle()
```
### 5 - galleries/examples/widgets/check_buttons.py:

Start line: 1, End line: 61

```python
"""
=============
Check buttons
=============

Turning visual elements on and off with check buttons.

This program shows the use of `.CheckButtons` which is similar to
check boxes. There are 3 different sine waves shown, and we can choose which
waves are displayed with the check buttons.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import CheckButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(6*np.pi*t)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False, lw=2, color='black', label='1 Hz')
l1, = ax.plot(t, s1, lw=2, color='red', label='2 Hz')
l2, = ax.plot(t, s2, lw=2, color='green', label='3 Hz')
fig.subplots_adjust(left=0.2)

lines_by_label = {l.get_label(): l for l in [l0, l1, l2]}
line_colors = [l.get_color() for l in lines_by_label.values()]

# Make checkbuttons with all plotted lines with correct visibility
rax = fig.add_axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(
    ax=rax,
    labels=lines_by_label.keys(),
    actives=[l.get_visible() for l in lines_by_label.values()],
    label_props={'color': line_colors},
    frame_props={'edgecolor': line_colors},
    check_props={'facecolor': line_colors},
)


def callback(label):
    ln = lines_by_label[label]
    ln.set_visible(not ln.get_visible())
    ln.figure.canvas.draw_idle()

check.on_clicked(callback)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.CheckButtons`
```
### 6 - lib/matplotlib/backends/_backend_tk.py:

Start line: 325, End line: 343

```python
class FigureCanvasTk(FigureCanvasBase):

    def button_dblclick_event(self, event):
        self.button_press_event(event, dblclick=True)

    def button_release_event(self, event):
        num = getattr(event, 'num', None)
        if sys.platform == 'darwin':  # 2 and 3 are reversed.
            num = {2: 3, 3: 2}.get(num, num)
        MouseEvent("button_release_event", self,
                   *self._event_mpl_coords(event), num,
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()

    def scroll_event(self, event):
        num = getattr(event, 'num', None)
        step = 1 if num == 4 else -1 if num == 5 else 0
        MouseEvent("scroll_event", self,
                   *self._event_mpl_coords(event), step=step,
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()
```
### 7 - lib/matplotlib/widgets.py:

Start line: 1198, End line: 1942

```python
class CheckButtons(AxesWidget):

    def set_active(self, index):
        """
        Toggle (activate or deactivate) a check button by index.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            Index of the check button to toggle.

        Raises
        ------
        ValueError
            If *index* is invalid.
        """
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid CheckButton index: {index}')

        invisible = colors.to_rgba('none')

        facecolors = self._checks.get_facecolor()
        facecolors[index] = (
            self._active_check_colors[index]
            if colors.same_color(facecolors[index], invisible)
            else invisible
        )
        self._checks.set_facecolor(facecolors)

        if hasattr(self, "_lines"):
            l1, l2 = self._lines[index]
            l1.set_visible(not l1.get_visible())
            l2.set_visible(not l2.get_visible())

        if self.drawon:
            if self._useblit:
                if self._background is not None:
                    self.canvas.restore_region(self._background)
                self.ax.draw_artist(self._checks)
                if hasattr(self, "_lines"):
                    for l1, l2 in self._lines:
                        self.ax.draw_artist(l1)
                        self.ax.draw_artist(l2)
                self.canvas.blit(self.ax.bbox)
            else:
                self.canvas.draw()

        if self.eventson:
            self._observers.process('clicked', self.labels[index].get_text())

    def _init_status(self, actives):
        """
        Initialize properties to match active status.

        The user may have passed custom colours in *check_props* to the
        constructor, or to `.set_check_props`, so we need to modify the
        visibility after getting whatever the user set.
        """
        self._active_check_colors = self._checks.get_facecolor()
        if len(self._active_check_colors) == 1:
            self._active_check_colors = np.repeat(self._active_check_colors,
                                                  len(actives), axis=0)
        self._checks.set_facecolor(
            [ec if active else "none"
             for ec, active in zip(self._active_check_colors, actives)])

    def get_status(self):
        """
        Return a list of the status (True/False) of all of the check buttons.
        """
        return [not colors.same_color(color, colors.to_rgba("none"))
                for color in self._checks.get_facecolors()]

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Returns a connection id, which can be used to disconnect the callback.
        """
        return self._observers.connect('clicked', lambda text: func(text))

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)

    @_api.deprecated("3.7",
                     addendum="Any custom property styling may be lost.")
    @property
    def rectangles(self):
        if not hasattr(self, "_rectangles"):
            ys = np.linspace(1, 0, len(self.labels)+2)[1:-1]
            dy = 1. / (len(self.labels) + 1)
            w, h = dy / 2, dy / 2
            rectangles = self._rectangles = [
                Rectangle(xy=(0.05, ys[i] - h / 2), width=w, height=h,
                          edgecolor="black",
                          facecolor="none",
                          transform=self.ax.transAxes
                          )
                for i, y in enumerate(ys)
            ]
            self._frames.set_visible(False)
            for rectangle in rectangles:
                self.ax.add_patch(rectangle)
        if not hasattr(self, "_lines"):
            with _api.suppress_matplotlib_deprecation_warning():
                _ = self.lines
        return self._rectangles

    @_api.deprecated("3.7",
                     addendum="Any custom property styling may be lost.")
    @property
    def lines(self):
        if not hasattr(self, "_lines"):
            ys = np.linspace(1, 0, len(self.labels)+2)[1:-1]
            self._checks.set_visible(False)
            dy = 1. / (len(self.labels) + 1)
            w, h = dy / 2, dy / 2
            self._lines = []
            current_status = self.get_status()
            lineparams = {'color': 'k', 'linewidth': 1.25,
                          'transform': self.ax.transAxes,
                          'solid_capstyle': 'butt',
                          'animated': self._useblit}
            for i, y in enumerate(ys):
                x, y = 0.05, y - h / 2
                l1 = Line2D([x, x + w], [y + h, y], **lineparams)
                l2 = Line2D([x, x + w], [y, y + h], **lineparams)

                l1.set_visible(current_status[i])
                l2.set_visible(current_status[i])
                self._lines.append((l1, l2))
                self.ax.add_line(l1)
                self.ax.add_line(l2)
        if not hasattr(self, "_rectangles"):
            with _api.suppress_matplotlib_deprecation_warning():
                _ = self.rectangles
        return self._lines


class TextBox(AxesWidget):
    """
    A GUI neutral text input box.

    For the text box to remain responsive you must keep a reference to it.

    Call `.on_text_change` to be updated whenever the text changes.

    Call `.on_submit` to be updated whenever the user hits enter or
    leaves the text entry field.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    label : `.Text`

    color : color
        The color of the text box when not hovering.
    hovercolor : color
        The color of the text box when hovering.
    """

    @_api.make_keyword_only("3.7", name="color")
    def __init__(self, ax, label, initial='',
                 color='.95', hovercolor='1', label_pad=.01,
                 textalignment="left"):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            Label for this text box.
        initial : str
            Initial value in the text box.
        color : color
            The color of the box.
        hovercolor : color
            The color of the box when the mouse is over it.
        label_pad : float
            The distance between the label and the right side of the textbox.
        textalignment : {'left', 'center', 'right'}
            The horizontal location of the text.
        """
        super().__init__(ax)

        self._text_position = _api.check_getitem(
            {"left": 0.05, "center": 0.5, "right": 0.95},
            textalignment=textalignment)

        self.label = ax.text(
            -label_pad, 0.5, label, transform=ax.transAxes,
            verticalalignment='center', horizontalalignment='right')

        # TextBox's text object should not parse mathtext at all.
        self.text_disp = self.ax.text(
            self._text_position, 0.5, initial, transform=self.ax.transAxes,
            verticalalignment='center', horizontalalignment=textalignment,
            parse_math=False)

        self._observers = cbook.CallbackRegistry(signals=["change", "submit"])

        ax.set(
            xlim=(0, 1), ylim=(0, 1),  # s.t. cursor appears from first click.
            navigate=False, facecolor=color,
            xticks=[], yticks=[])

        self.cursor_index = 0

        self.cursor = ax.vlines(0, 0, 0, visible=False, color="k", lw=1,
                                transform=mpl.transforms.IdentityTransform())

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        self.connect_event('key_press_event', self._keypress)
        self.connect_event('resize_event', self._resize)

        self.color = color
        self.hovercolor = hovercolor

        self.capturekeystrokes = False

    @property
    def text(self):
        return self.text_disp.get_text()

    def _rendercursor(self):
        # this is a hack to figure out where the cursor should go.
        # we draw the text up to where the cursor should go, measure
        # and save its dimensions, draw the real text, then put the cursor
        # at the saved dimensions

        # This causes a single extra draw if the figure has never been rendered
        # yet, which should be fine as we're going to repeatedly re-render the
        # figure later anyways.
        if self.ax.figure._get_renderer() is None:
            self.ax.figure.canvas.draw()

        text = self.text_disp.get_text()  # Save value before overwriting it.
        widthtext = text[:self.cursor_index]

        bb_text = self.text_disp.get_window_extent()
        self.text_disp.set_text(widthtext or ",")
        bb_widthtext = self.text_disp.get_window_extent()

        if bb_text.y0 == bb_text.y1:  # Restoring the height if no text.
            bb_text.y0 -= bb_widthtext.height / 2
            bb_text.y1 += bb_widthtext.height / 2
        elif not widthtext:  # Keep width to 0.
            bb_text.x1 = bb_text.x0
        else:  # Move the cursor using width of bb_widthtext.
            bb_text.x1 = bb_text.x0 + bb_widthtext.width

        self.cursor.set(
            segments=[[(bb_text.x1, bb_text.y0), (bb_text.x1, bb_text.y1)]],
            visible=True)
        self.text_disp.set_text(text)

        self.ax.figure.canvas.draw()

    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)

    def _keypress(self, event):
        if self.ignore(event):
            return
        if self.capturekeystrokes:
            key = event.key
            text = self.text
            if len(key) == 1:
                text = (text[:self.cursor_index] + key +
                        text[self.cursor_index:])
                self.cursor_index += 1
            elif key == "right":
                if self.cursor_index != len(text):
                    self.cursor_index += 1
            elif key == "left":
                if self.cursor_index != 0:
                    self.cursor_index -= 1
            elif key == "home":
                self.cursor_index = 0
            elif key == "end":
                self.cursor_index = len(text)
            elif key == "backspace":
                if self.cursor_index != 0:
                    text = (text[:self.cursor_index - 1] +
                            text[self.cursor_index:])
                    self.cursor_index -= 1
            elif key == "delete":
                if self.cursor_index != len(self.text):
                    text = (text[:self.cursor_index] +
                            text[self.cursor_index + 1:])
            self.text_disp.set_text(text)
            self._rendercursor()
            if self.eventson:
                self._observers.process('change', self.text)
                if key in ["enter", "return"]:
                    self._observers.process('submit', self.text)

    def set_val(self, val):
        newval = str(val)
        if self.text == newval:
            return
        self.text_disp.set_text(newval)
        self._rendercursor()
        if self.eventson:
            self._observers.process('change', self.text)
            self._observers.process('submit', self.text)

    @_api.delete_parameter("3.7", "x")
    def begin_typing(self, x=None):
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

    def stop_typing(self):
        if self.capturekeystrokes:
            self._on_stop_typing()
            self._on_stop_typing = None
            notifysubmit = True
        else:
            notifysubmit = False
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self.ax.figure.canvas.draw()
        if notifysubmit and self.eventson:
            # Because process() might throw an error in the user's code, only
            # call it once we've already done our cleanup.
            self._observers.process('submit', self.text)

    def _click(self, event):
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            self.stop_typing()
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)
        if not self.capturekeystrokes:
            self.begin_typing()
        self.cursor_index = self.text_disp._char_index_at(event.x)
        self._rendercursor()

    def _resize(self, event):
        self.stop_typing()

    def _motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if event.inaxes == self.ax else self.color
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                self.ax.figure.canvas.draw()

    def on_text_change(self, func):
        """
        When the text changes, call this *func* with event.

        A connection id is returned which can be used to disconnect.
        """
        return self._observers.connect('change', lambda text: func(text))

    def on_submit(self, func):
        """
        When the user hits enter or leaves the submission box, call this
        *func* with event.

        A connection id is returned which can be used to disconnect.
        """
        return self._observers.connect('submit', lambda text: func(text))

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)


class RadioButtons(AxesWidget):
    """
    A GUI neutral radio button.

    For the buttons to remain responsive you must keep a reference to this
    object.

    Connect to the RadioButtons with the `.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    activecolor : color
        The color of the selected button.
    labels : list of `.Text`
        The button labels.
    circles : list of `~.patches.Circle`
        The buttons.
    value_selected : str
        The label text of the currently selected button.
    """

    def __init__(self, ax, labels, active=0, activecolor=None, *,
                 useblit=True, label_props=None, radio_props=None):
        """
        Add radio buttons to an `~.axes.Axes`.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button. The default is ``'blue'`` if not
            specified here or in *radio_props*.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :doc:`/tutorials/advanced/blitting` for details.

            .. versionadded:: 3.7
        label_props : dict or list of dict, optional
            Dictionary of `.Text` properties to be used for the labels.

            .. versionadded:: 3.7
        radio_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            radio buttons. Defaults to (label font size / 2)**2 size, black
            edgecolor, and *activecolor* facecolor (when active).

            .. note::
                If a facecolor is supplied in *radio_props*, it will override
                *activecolor*. This may be used to provide an active color per
                button.

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        _api.check_isinstance((dict, None), label_props=label_props,
                              radio_props=radio_props)

        radio_props = cbook.normalize_kwargs(radio_props,
                                             collections.PathCollection)
        if activecolor is not None:
            if 'facecolor' in radio_props:
                _api.warn_external(
                    'Both the *activecolor* parameter and the *facecolor* '
                    'key in the *radio_props* parameter has been specified. '
                    '*activecolor* will be ignored.')
        else:
            activecolor = 'blue'  # Default.

        self._activecolor = activecolor
        self.value_selected = labels[active]

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        ys = np.linspace(1, 0, len(labels) + 2)[1:-1]

        self._useblit = useblit and self.canvas.supports_blit
        self._background = None

        label_props = _expand_text_props(label_props)
        self.labels = [
            ax.text(0.25, y, label, transform=ax.transAxes,
                    horizontalalignment="left", verticalalignment="center",
                    **props)
            for y, label, props in zip(ys, labels, label_props)]
        text_size = np.array([text.get_fontsize() for text in self.labels]) / 2

        radio_props = {
            's': text_size**2,
            **radio_props,
            'marker': 'o',
            'transform': ax.transAxes,
            'animated': self._useblit,
        }
        radio_props.setdefault('edgecolor', radio_props.get('color', 'black'))
        radio_props.setdefault('facecolor',
                               radio_props.pop('color', activecolor))
        self._buttons = ax.scatter([.15] * len(ys), ys, **radio_props)
        # The user may have passed custom colours in radio_props, so we need to
        # create the radios, and modify the visibility after getting whatever
        # the user set.
        self._active_colors = self._buttons.get_facecolor()
        if len(self._active_colors) == 1:
            self._active_colors = np.repeat(self._active_colors, len(labels),
                                            axis=0)
        self._buttons.set_facecolor(
            [activecolor if i == active else "none"
             for i, activecolor in enumerate(self._active_colors)])

        self.connect_event('button_press_event', self._clicked)
        if self._useblit:
            self.connect_event('draw_event', self._clear)

        self._observers = cbook.CallbackRegistry(signals=["clicked"])

    def _clear(self, event):
        """Internal event handler to clear the buttons."""
        if self.ignore(event) or self._changed_canvas():
            return
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._buttons)
        if hasattr(self, "_circles"):
            for circle in self._circles:
                self.ax.draw_artist(circle)

    def _clicked(self, event):
        if self.ignore(event) or event.button != 1 or event.inaxes != self.ax:
            return
        pclicked = self.ax.transAxes.inverted().transform((event.x, event.y))
        _, inds = self._buttons.contains(event)
        coords = self._buttons.get_offset_transform().transform(
            self._buttons.get_offsets())
        distances = {}
        if hasattr(self, "_circles"):  # Remove once circles is removed.
            for i, (p, t) in enumerate(zip(self._circles, self.labels)):
                if (t.get_window_extent().contains(event.x, event.y)
                        or np.linalg.norm(pclicked - p.center) < p.radius):
                    distances[i] = np.linalg.norm(pclicked - p.center)
        else:
            for i, t in enumerate(self.labels):
                if (i in inds["ind"]
                        or t.get_window_extent().contains(event.x, event.y)):
                    distances[i] = np.linalg.norm(pclicked - coords[i])
        if len(distances) > 0:
            closest = min(distances, key=distances.get)
            self.set_active(closest)

    def set_label_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels.
        """
        _api.check_isinstance(dict, props=props)
        props = _expand_text_props(props)
        for text, prop in zip(self.labels, props):
            text.update(prop)

    def set_radio_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the radio
            buttons.
        """
        _api.check_isinstance(dict, props=props)
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        self._buttons.update(props)
        self._active_colors = self._buttons.get_facecolor()
        if len(self._active_colors) == 1:
            self._active_colors = np.repeat(self._active_colors,
                                            len(self.labels), axis=0)
        self._buttons.set_facecolor(
            [activecolor if text.get_text() == self.value_selected else "none"
             for text, activecolor in zip(self.labels, self._active_colors)])

    @property
    def activecolor(self):
        return self._activecolor

    @activecolor.setter
    def activecolor(self, activecolor):
        colors._check_color_like(activecolor=activecolor)
        self._activecolor = activecolor
        self.set_radio_props({'facecolor': activecolor})
        # Make sure the deprecated version is updated.
        # Remove once circles is removed.
        labels = [label.get_text() for label in self.labels]
        with cbook._setattr_cm(self, eventson=False):
            self.set_active(labels.index(self.value_selected))

    def set_active(self, index):
        """
        Select button with number *index*.

        Callbacks will be triggered if :attr:`eventson` is True.
        """
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid RadioButton index: {index}')
        self.value_selected = self.labels[index].get_text()
        button_facecolors = self._buttons.get_facecolor()
        button_facecolors[:] = colors.to_rgba("none")
        button_facecolors[index] = colors.to_rgba(self._active_colors[index])
        self._buttons.set_facecolor(button_facecolors)
        if hasattr(self, "_circles"):  # Remove once circles is removed.
            for i, p in enumerate(self._circles):
                p.set_facecolor(self.activecolor if i == index else "none")
                if self.drawon and self._useblit:
                    self.ax.draw_artist(p)
        if self.drawon:
            if self._useblit:
                if self._background is not None:
                    self.canvas.restore_region(self._background)
                self.ax.draw_artist(self._buttons)
                if hasattr(self, "_circles"):
                    for p in self._circles:
                        self.ax.draw_artist(p)
                self.canvas.blit(self.ax.bbox)
            else:
                self.canvas.draw()

        if self.eventson:
            self._observers.process('clicked', self.labels[index].get_text())

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Returns a connection id, which can be used to disconnect the callback.
        """
        return self._observers.connect('clicked', func)

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)

    @_api.deprecated("3.7",
                     addendum="Any custom property styling may be lost.")
    @property
    def circles(self):
        if not hasattr(self, "_circles"):
            radius = min(.5 / (len(self.labels) + 1) - .01, .05)
            circles = self._circles = [
                Circle(xy=self._buttons.get_offsets()[i], edgecolor="black",
                       facecolor=self._buttons.get_facecolor()[i],
                       radius=radius, transform=self.ax.transAxes,
                       animated=self._useblit)
                for i in range(len(self.labels))]
            self._buttons.set_visible(False)
            for circle in circles:
                self.ax.add_patch(circle)
        return self._circles


class SubplotTool(Widget):
    """
    A tool to adjust the subplot params of a `matplotlib.figure.Figure`.
    """

    def __init__(self, targetfig, toolfig):
        """
        Parameters
        ----------
        targetfig : `.Figure`
            The figure instance to adjust.
        toolfig : `.Figure`
            The figure instance to embed the subplot tool into.
        """

        self.figure = toolfig
        self.targetfig = targetfig
        toolfig.subplots_adjust(left=0.2, right=0.9)
        toolfig.suptitle("Click on slider to adjust subplot param")

        self._sliders = []
        names = ["left", "bottom", "right", "top", "wspace", "hspace"]
        # The last subplot, removed below, keeps space for the "Reset" button.
        for name, ax in zip(names, toolfig.subplots(len(names) + 1)):
            ax.set_navigate(False)
            slider = Slider(ax, name, 0, 1,
                            valinit=getattr(targetfig.subplotpars, name))
            slider.on_changed(self._on_slider_changed)
            self._sliders.append(slider)
        toolfig.axes[-1].remove()
        (self.sliderleft, self.sliderbottom, self.sliderright, self.slidertop,
         self.sliderwspace, self.sliderhspace) = self._sliders
        for slider in [self.sliderleft, self.sliderbottom,
                       self.sliderwspace, self.sliderhspace]:
            slider.closedmax = False
        for slider in [self.sliderright, self.slidertop]:
            slider.closedmin = False

        # constraints
        self.sliderleft.slidermax = self.sliderright
        self.sliderright.slidermin = self.sliderleft
        self.sliderbottom.slidermax = self.slidertop
        self.slidertop.slidermin = self.sliderbottom

        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonreset = Button(bax, 'Reset')
        self.buttonreset.on_clicked(self._on_reset)

    def _on_slider_changed(self, _):
        self.targetfig.subplots_adjust(
            **{slider.label.get_text(): slider.val
               for slider in self._sliders})
        if self.drawon:
            self.targetfig.canvas.draw()

    def _on_reset(self, event):
        with ExitStack() as stack:
            # Temporarily disable drawing on self and self's sliders, and
            # disconnect slider events (as the subplotparams can be temporarily
            # invalid, depending on the order in which they are restored).
            stack.enter_context(cbook._setattr_cm(self, drawon=False))
            for slider in self._sliders:
                stack.enter_context(
                    cbook._setattr_cm(slider, drawon=False, eventson=False))
            # Reset the slider to the initial position.
            for slider in self._sliders:
                slider.reset()
        if self.drawon:
            event.canvas.draw()  # Redraw the subplottool canvas.
        self._on_slider_changed(None)  # Apply changes to the target window.
```
### 8 - lib/matplotlib/backend_bases.py:

Start line: 2411, End line: 2780

```python
class FigureCanvasBase:

    @classmethod
    def get_default_filetype(cls):
        """
        Return the default savefig file format as specified in
        :rc:`savefig.format`.

        The returned string does not include a period. This method is
        overridden in backends that only support a single file type.
        """
        return rcParams['savefig.format']

    def get_default_filename(self):
        """
        Return a string, which includes extension, suitable for use as
        a default filename.
        """
        basename = (self.manager.get_window_title() if self.manager is not None
                    else '')
        basename = (basename or 'image').replace(' ', '_')
        filetype = self.get_default_filetype()
        filename = basename + '.' + filetype
        return filename

    def switch_backends(self, FigureCanvasClass):
        """
        Instantiate an instance of FigureCanvasClass

        This is used for backend switching, e.g., to instantiate a
        FigureCanvasPS from a FigureCanvasGTK.  Note, deep copying is
        not done, so any changes to one of the instances (e.g., setting
        figure size or line props), will be reflected in the other
        """
        newCanvas = FigureCanvasClass(self.figure)
        newCanvas._is_saving = self._is_saving
        return newCanvas

    def mpl_connect(self, s, func):
        """
        Bind function *func* to event *s*.

        Parameters
        ----------
        s : str
            One of the following events ids:

            - 'button_press_event'
            - 'button_release_event'
            - 'draw_event'
            - 'key_press_event'
            - 'key_release_event'
            - 'motion_notify_event'
            - 'pick_event'
            - 'resize_event'
            - 'scroll_event'
            - 'figure_enter_event',
            - 'figure_leave_event',
            - 'axes_enter_event',
            - 'axes_leave_event'
            - 'close_event'.

        func : callable
            The callback function to be executed, which must have the
            signature::

                def func(event: Event) -> Any

            For the location events (button and key press/release), if the
            mouse is over the Axes, the ``inaxes`` attribute of the event will
            be set to the `~matplotlib.axes.Axes` the event occurs is over, and
            additionally, the variables ``xdata`` and ``ydata`` attributes will
            be set to the mouse location in data coordinates.  See `.KeyEvent`
            and `.MouseEvent` for more info.

            .. note::

                If func is a method, this only stores a weak reference to the
                method. Thus, the figure does not influence the lifetime of
                the associated object. Usually, you want to make sure that the
                object is kept alive throughout the lifetime of the figure by
                holding a reference to it.

        Returns
        -------
        cid
            A connection id that can be used with
            `.FigureCanvasBase.mpl_disconnect`.

        Examples
        --------
        ::

            def on_press(event):
                print('you pressed', event.button, event.xdata, event.ydata)

            cid = canvas.mpl_connect('button_press_event', on_press)
        """

        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        """
        Disconnect the callback with id *cid*.

        Examples
        --------
        ::

            cid = canvas.mpl_connect('button_press_event', on_press)
            # ... later
            canvas.mpl_disconnect(cid)
        """
        return self.callbacks.disconnect(cid)

    # Internal subclasses can override _timer_cls instead of new_timer, though
    # this is not a public API for third-party subclasses.
    _timer_cls = TimerBase

    def new_timer(self, interval=None, callbacks=None):
        """
        Create a new backend-specific subclass of `.Timer`.

        This is useful for getting periodic events through the backend's native
        event loop.  Implemented only for backends with GUIs.

        Parameters
        ----------
        interval : int
            Timer interval in milliseconds.

        callbacks : list[tuple[callable, tuple, dict]]
            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
            will be executed by the timer every *interval*.

            Callbacks which return ``False`` or ``0`` will be removed from the
            timer.

        Examples
        --------
        >>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])
        """
        return self._timer_cls(interval=interval, callbacks=callbacks)

    def flush_events(self):
        """
        Flush the GUI events for the figure.

        Interactive backends need to reimplement this method.
        """

    def start_event_loop(self, timeout=0):
        """
        Start a blocking event loop.

        Such an event loop is used by interactive functions, such as
        `~.Figure.ginput` and `~.Figure.waitforbuttonpress`, to wait for
        events.

        The event loop blocks until a callback function triggers
        `stop_event_loop`, or *timeout* is reached.

        If *timeout* is 0 or negative, never timeout.

        Only interactive backends need to reimplement this method and it relies
        on `flush_events` being properly implemented.

        Interactive backends should implement this in a more native way.
        """
        if timeout <= 0:
            timeout = np.inf
        timestep = 0.01
        counter = 0
        self._looping = True
        while self._looping and counter * timestep < timeout:
            self.flush_events()
            time.sleep(timestep)
            counter += 1

    def stop_event_loop(self):
        """
        Stop the current blocking event loop.

        Interactive backends need to reimplement this to match
        `start_event_loop`
        """
        self._looping = False


def key_press_handler(event, canvas=None, toolbar=None):
    """
    Implement the default Matplotlib key bindings for the canvas and toolbar
    described at :ref:`key-event-handling`.

    Parameters
    ----------
    event : `KeyEvent`
        A key press/release event.
    canvas : `FigureCanvasBase`, default: ``event.canvas``
        The backend-specific canvas instance.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas``.
    toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``
        The navigation cursor toolbar.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas.toolbar``.
    """
    # these bindings happen whether you are over an Axes or not

    if event.key is None:
        return
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar

    # Load key-mappings from rcParams.
    fullscreen_keys = rcParams['keymap.fullscreen']
    home_keys = rcParams['keymap.home']
    back_keys = rcParams['keymap.back']
    forward_keys = rcParams['keymap.forward']
    pan_keys = rcParams['keymap.pan']
    zoom_keys = rcParams['keymap.zoom']
    save_keys = rcParams['keymap.save']
    quit_keys = rcParams['keymap.quit']
    quit_all_keys = rcParams['keymap.quit_all']
    grid_keys = rcParams['keymap.grid']
    grid_minor_keys = rcParams['keymap.grid_minor']
    toggle_yscale_keys = rcParams['keymap.yscale']
    toggle_xscale_keys = rcParams['keymap.xscale']

    # toggle fullscreen mode ('f', 'ctrl + f')
    if event.key in fullscreen_keys:
        try:
            canvas.manager.full_screen_toggle()
        except AttributeError:
            pass

    # quit the figure (default key 'ctrl+w')
    if event.key in quit_keys:
        Gcf.destroy_fig(canvas.figure)
    if event.key in quit_all_keys:
        Gcf.destroy_all()

    if toolbar is not None:
        # home or reset mnemonic  (default key 'h', 'home' and 'r')
        if event.key in home_keys:
            toolbar.home()
        # forward / backward keys to enable left handed quick navigation
        # (default key for backward: 'left', 'backspace' and 'c')
        elif event.key in back_keys:
            toolbar.back()
        # (default key for forward: 'right' and 'v')
        elif event.key in forward_keys:
            toolbar.forward()
        # pan mnemonic (default key 'p')
        elif event.key in pan_keys:
            toolbar.pan()
            toolbar._update_cursor(event)
        # zoom mnemonic (default key 'o')
        elif event.key in zoom_keys:
            toolbar.zoom()
            toolbar._update_cursor(event)
        # saving current figure (default key 's')
        elif event.key in save_keys:
            toolbar.save_figure()

    if event.inaxes is None:
        return

    # these bindings require the mouse to be over an Axes to trigger
    def _get_uniform_gridstate(ticks):
        # Return True/False if all grid lines are on or off, None if they are
        # not all in the same state.
        if all(tick.gridline.get_visible() for tick in ticks):
            return True
        elif not any(tick.gridline.get_visible() for tick in ticks):
            return False
        else:
            return None

    ax = event.inaxes
    # toggle major grids in current Axes (default key 'g')
    # Both here and below (for 'G'), we do nothing if *any* grid (major or
    # minor, x or y) is not in a uniform state, to avoid messing up user
    # customization.
    if (event.key in grid_keys
            # Exclude minor grids not in a uniform state.
            and None not in [_get_uniform_gridstate(ax.xaxis.minorTicks),
                             _get_uniform_gridstate(ax.yaxis.minorTicks)]):
        x_state = _get_uniform_gridstate(ax.xaxis.majorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.majorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            x_state, y_state = (
                cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        except ValueError:
            # Exclude major grids not in a uniform state.
            pass
        else:
            # If turning major grids off, also turn minor grids off.
            ax.grid(x_state, which="major" if x_state else "both", axis="x")
            ax.grid(y_state, which="major" if y_state else "both", axis="y")
            canvas.draw_idle()
    # toggle major and minor grids in current Axes (default key 'G')
    if (event.key in grid_minor_keys
            # Exclude major grids not in a uniform state.
            and None not in [_get_uniform_gridstate(ax.xaxis.majorTicks),
                             _get_uniform_gridstate(ax.yaxis.majorTicks)]):
        x_state = _get_uniform_gridstate(ax.xaxis.minorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.minorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            x_state, y_state = (
                cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        except ValueError:
            # Exclude minor grids not in a uniform state.
            pass
        else:
            ax.grid(x_state, which="both", axis="x")
            ax.grid(y_state, which="both", axis="y")
            canvas.draw_idle()
    # toggle scaling of y-axes between 'log and 'linear' (default key 'l')
    elif event.key in toggle_yscale_keys:
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
        elif scale == 'linear':
            try:
                ax.set_yscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
    # toggle scaling of x-axes between 'log and 'linear' (default key 'k')
    elif event.key in toggle_xscale_keys:
        scalex = ax.get_xscale()
        if scalex == 'log':
            ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()
        elif scalex == 'linear':
            try:
                ax.set_xscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()


def button_press_handler(event, canvas=None, toolbar=None):
    """
    The default Matplotlib button actions for extra mouse buttons.

    Parameters are as for `key_press_handler`, except that *event* is a
    `MouseEvent`.
    """
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar
    if toolbar is not None:
        button_name = str(MouseButton(event.button))
        if button_name in rcParams['keymap.back']:
            toolbar.back()
        elif button_name in rcParams['keymap.forward']:
            toolbar.forward()


class NonGuiException(Exception):
    """Raised when trying show a figure in a non-GUI backend."""
    pass
```
### 9 - galleries/examples/widgets/range_slider.py:

Start line: 1, End line: 42

```python
"""
======================================
Thresholding an Image with RangeSlider
======================================

Using the RangeSlider widget to control the thresholding of an image.

The RangeSlider widget can be used similarly to the `.widgets.Slider`
widget. The major difference is that RangeSlider's ``val`` attribute
is a tuple of floats ``(lower val, upper val)`` rather than a single float.

See :doc:`/gallery/widgets/slider_demo` for an example of using
a ``Slider`` to control a single float.

See :doc:`/gallery/widgets/slider_snap_demo` for an example of having
the ``Slider`` snap to discrete values.
"""

import matplotlib.pyplot as plt
import numpy as np

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
slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max())

# Create the Vertical lines on the histogram
lower_limit_line = axs[1].axvline(slider.val[0], color='k')
upper_limit_line = axs[1].axvline(slider.val[1], color='k')
```
### 10 - galleries/examples/widgets/slider_snap_demo.py:

Start line: 1, End line: 84

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

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

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

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.Slider`
#    - `matplotlib.widgets.Button`
```
### 15 - lib/matplotlib/figure.py:

Start line: 3413, End line: 3455

```python
@_docstring.interpd
class Figure(FigureBase):

    def ginput(self, n=1, timeout=30, show_clicks=True,
               mouse_add=MouseButton.LEFT,
               mouse_pop=MouseButton.RIGHT,
               mouse_stop=MouseButton.MIDDLE):
        # ... other code

        def handler(event):
            is_button = event.name == "button_press_event"
            is_key = event.name == "key_press_event"
            # Quit (even if not in infinite mode; this is consistent with
            # MATLAB and sometimes quite useful, but will require the user to
            # test how many points were actually returned before using data).
            if (is_button and event.button == mouse_stop
                    or is_key and event.key in ["escape", "enter"]):
                self.canvas.stop_event_loop()
            # Pop last click.
            elif (is_button and event.button == mouse_pop
                  or is_key and event.key in ["backspace", "delete"]):
                if clicks:
                    clicks.pop()
                    if show_clicks:
                        marks.pop().remove()
                        self.canvas.draw()
            # Add new click.
            elif (is_button and event.button == mouse_add
                  # On macOS/gtk, some keys return None.
                  or is_key and event.key is not None):
                if event.inaxes:
                    clicks.append((event.xdata, event.ydata))
                    _log.info("input %i: %f, %f",
                              len(clicks), event.xdata, event.ydata)
                    if show_clicks:
                        line = mpl.lines.Line2D([event.xdata], [event.ydata],
                                                marker="+", color="r")
                        event.inaxes.add_line(line)
                        marks.append(line)
                        self.canvas.draw()
            if len(clicks) == n and n > 0:
                self.canvas.stop_event_loop()

        _blocking_input.blocking_input_loop(
            self, ["button_press_event", "key_press_event"], timeout, handler)

        # Cleanup.
        for mark in marks:
            mark.remove()
        self.canvas.draw()

        return clicks
```
### 46 - lib/matplotlib/figure.py:

Start line: 3457, End line: 3475

```python
@_docstring.interpd
class Figure(FigureBase):

    def waitforbuttonpress(self, timeout=-1):
        """
        Blocking call to interact with the figure.

        Wait for user input and return True if a key was pressed, False if a
        mouse button was pressed and None if no input was given within
        *timeout* seconds.  Negative values deactivate *timeout*.
        """
        event = None

        def handler(ev):
            nonlocal event
            event = ev
            self.canvas.stop_event_loop()

        _blocking_input.blocking_input_loop(
            self, ["button_press_event", "key_press_event"], timeout, handler)

        return None if event is None else event.name == "key_press_event"
```
