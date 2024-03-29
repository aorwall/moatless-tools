# matplotlib__matplotlib-25404

| **matplotlib/matplotlib** | `b51a227058e16cdbc56890f49e3a0888ea02b9d2` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 5654 |
| **Avg pos** | 22.0 |
| **Min pos** | 6 |
| **Max pos** | 8 |
| **Top file pos** | 2 |
| **Missing snippets** | 8 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/widgets.py b/lib/matplotlib/widgets.py
--- a/lib/matplotlib/widgets.py
+++ b/lib/matplotlib/widgets.py
@@ -2457,15 +2457,16 @@ def artists(self):
 
     def set_props(self, **props):
         """
-        Set the properties of the selector artist. See the `props` argument
-        in the selector docstring to know which properties are supported.
+        Set the properties of the selector artist.
+
+        See the *props* argument in the selector docstring to know which properties are
+        supported.
         """
         artist = self._selection_artist
         props = cbook.normalize_kwargs(props, artist)
         artist.set(**props)
         if self.useblit:
             self.update()
-        self._props.update(props)
 
     def set_handle_props(self, **handle_props):
         """
@@ -2658,7 +2659,6 @@ def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
         # but we maintain it until it is removed
         self._pressv = None
 
-        self._props = props
         self.onmove_callback = onmove_callback
         self.minspan = minspan
 
@@ -2670,7 +2670,7 @@ def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
 
         # Reset canvas so that `new_axes` connects events.
         self.canvas = None
-        self.new_axes(ax)
+        self.new_axes(ax, _props=props)
 
         # Setup handles
         self._handle_props = {
@@ -2686,7 +2686,7 @@ def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
         # prev attribute is deprecated but we still need to maintain it
         self._prev = (0, 0)
 
-    def new_axes(self, ax):
+    def new_axes(self, ax, *, _props=None):
         """Set SpanSelector to operate on a new Axes."""
         self.ax = ax
         if self.canvas is not ax.figure.canvas:
@@ -2705,10 +2705,11 @@ def new_axes(self, ax):
         else:
             trans = ax.get_yaxis_transform()
             w, h = 1, 0
-        rect_artist = Rectangle((0, 0), w, h,
-                                transform=trans,
-                                visible=False,
-                                **self._props)
+        rect_artist = Rectangle((0, 0), w, h, transform=trans, visible=False)
+        if _props is not None:
+            rect_artist.update(_props)
+        elif self._selection_artist is not None:
+            rect_artist.update_from(self._selection_artist)
 
         self.ax.add_patch(rect_artist)
         self._selection_artist = rect_artist
@@ -3287,9 +3288,9 @@ def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False,
         if props is None:
             props = dict(facecolor='red', edgecolor='black',
                          alpha=0.2, fill=True)
-        self._props = {**props, 'animated': self.useblit}
-        self._visible = self._props.pop('visible', self._visible)
-        to_draw = self._init_shape(**self._props)
+        props = {**props, 'animated': self.useblit}
+        self._visible = props.pop('visible', self._visible)
+        to_draw = self._init_shape(**props)
         self.ax.add_patch(to_draw)
 
         self._selection_artist = to_draw
@@ -3305,8 +3306,7 @@ def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False,
 
         if self._interactive:
             self._handle_props = {
-                'markeredgecolor': (self._props or {}).get(
-                    'edgecolor', 'black'),
+                'markeredgecolor': (props or {}).get('edgecolor', 'black'),
                 **cbook.normalize_kwargs(handle_props, Line2D)}
 
             self._corner_order = ['SW', 'SE', 'NE', 'NW']
@@ -3942,13 +3942,13 @@ def __init__(self, ax, onselect, useblit=False,
 
         if props is None:
             props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
-        self._props = {**props, 'animated': self.useblit}
-        self._selection_artist = line = Line2D([], [], **self._props)
+        props = {**props, 'animated': self.useblit}
+        self._selection_artist = line = Line2D([], [], **props)
         self.ax.add_line(line)
 
         if handle_props is None:
             handle_props = dict(markeredgecolor='k',
-                                markerfacecolor=self._props.get('color', 'k'))
+                                markerfacecolor=props.get('color', 'k'))
         self._handle_props = handle_props
         self._polygon_handles = ToolHandles(self.ax, [], [],
                                             useblit=self.useblit,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/widgets.py | 2460 | 2468 | - | 2 | -
| lib/matplotlib/widgets.py | 2661 | 2661 | - | 2 | -
| lib/matplotlib/widgets.py | 2673 | 2673 | - | 2 | -
| lib/matplotlib/widgets.py | 2689 | 2689 | - | 2 | -
| lib/matplotlib/widgets.py | 2708 | 2711 | - | 2 | -
| lib/matplotlib/widgets.py | 3290 | 3292 | 8 | 2 | 12129
| lib/matplotlib/widgets.py | 3308 | 3309 | 8 | 2 | 12129
| lib/matplotlib/widgets.py | 3945 | 3951 | 6 | 2 | 5654


## Problem Statement

```
[Bug]: AttributeError: 'LassoSelector' object has no attribute '_props'
### Summary

I used the LassoSelector object to select the single point in the scatterplot. But when I try to update the line color of LassoSelector with the set_props function, I get an error like this **AttributeError: 'LassoSelector' object has no attribute '_props'**.

### Proposed fix

This warning does not occur when the comment symbol is placed at the beginning of the line "**self._props.update(props)**" in the "**set_ props**" function of the matplotlib library's widget.py code.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 galleries/examples/widgets/lasso_selector_demo_sgskip.py | 76 | 112| 239 | 239 | 776 | 
| 2 | **2 lib/matplotlib/widgets.py** | 1145 | 1196| 361 | 600 | 35266 | 
| 3 | 3 galleries/examples/event_handling/lasso_demo.py | 44 | 107| 450 | 1050 | 35971 | 
| 4 | 4 lib/matplotlib/artist.py | 1205 | 1243| 277 | 1327 | 49655 | 
| 5 | 4 galleries/examples/widgets/lasso_selector_demo_sgskip.py | 1 | 73| 537 | 1864 | 49655 | 
| **-> 6 <-** | **4 lib/matplotlib/widgets.py** | 3846 | 4276| 3790 | 5654 | 49655 | 
| 7 | 4 galleries/examples/event_handling/lasso_demo.py | 1 | 41| 254 | 5908 | 49655 | 
| **-> 8 <-** | **4 lib/matplotlib/widgets.py** | 3147 | 3843| 6221 | 12129 | 49655 | 
| 9 | 5 lib/matplotlib/legend.py | 659 | 689| 229 | 12358 | 61202 | 
| 10 | 5 lib/matplotlib/artist.py | 1180 | 1203| 196 | 12554 | 61202 | 
| 11 | 6 galleries/examples/subplots_axes_and_figures/axes_props.py | 1 | 22| 0 | 12554 | 61303 | 
| 12 | 7 lib/matplotlib/rcsetup.py | 592 | 626| 298 | 12852 | 73217 | 
| 13 | 8 galleries/examples/widgets/span_selector.py | 53 | 75| 112 | 12964 | 73687 | 
| 14 | 9 lib/matplotlib/legend_handler.py | 665 | 719| 504 | 13468 | 80354 | 
| 15 | 10 galleries/examples/misc/set_and_get.py | 1 | 102| 737 | 14205 | 81091 | 
| 16 | 10 lib/matplotlib/artist.py | 152 | 178| 239 | 14444 | 81091 | 
| 17 | 11 galleries/examples/statistics/boxplot.py | 74 | 106| 314 | 14758 | 82154 | 
| 18 | 11 lib/matplotlib/artist.py | 1602 | 1624| 185 | 14943 | 82154 | 
| 19 | 11 lib/matplotlib/legend.py | 985 | 1012| 204 | 15147 | 82154 | 
| 20 | 12 galleries/examples/statistics/bxp.py | 86 | 116| 271 | 15418 | 83189 | 
| 21 | 12 lib/matplotlib/legend_handler.py | 546 | 628| 762 | 16180 | 83189 | 
| 22 | 13 lib/matplotlib/lines.py | 1340 | 1361| 356 | 16536 | 95847 | 
| 23 | 14 lib/matplotlib/collections.py | 1087 | 1159| 676 | 17212 | 114174 | 
| 24 | **14 lib/matplotlib/widgets.py** | 1198 | 1942| 6229 | 23441 | 114174 | 
| 25 | 15 galleries/examples/widgets/annotated_cursor.py | 288 | 357| 568 | 24009 | 117129 | 
| 26 | 16 galleries/examples/text_labels_and_annotations/line_with_text.py | 52 | 88| 260 | 24269 | 117721 | 
| 27 | 16 lib/matplotlib/legend_handler.py | 238 | 272| 329 | 24598 | 117721 | 
| 28 | 17 lib/mpl_toolkits/axisartist/axis_artist.py | 90 | 174| 654 | 25252 | 125968 | 
| 29 | 18 lib/matplotlib/axes/_base.py | 468 | 4553| 813 | 26065 | 164942 | 
| 30 | 18 lib/matplotlib/legend.py | 515 | 623| 1524 | 27589 | 164942 | 
| 31 | 19 lib/matplotlib/pyplot.py | 2444 | 2468| 252 | 27841 | 193688 | 
| 32 | 19 lib/mpl_toolkits/axisartist/axis_artist.py | 176 | 200| 198 | 28039 | 193688 | 
| 33 | 19 lib/matplotlib/artist.py | 1570 | 1584| 162 | 28201 | 193688 | 
| 34 | 19 lib/matplotlib/artist.py | 1626 | 1679| 497 | 28698 | 193688 | 
| 35 | 20 galleries/examples/widgets/menu.py | 122 | 139| 147 | 28845 | 194614 | 
| 36 | 21 lib/matplotlib/_api/deprecation.py | 223 | 254| 311 | 29156 | 198931 | 


### Hint

```
The properties for `LassoSelector` is applied to the line stored as `self._selection_artist`. As such `self._props` is not defined in the constructor.

I *think* the correct solution is to redefine `set_props` for `LassoSelector` (and in that method set the props of the line), but there may be someone knowing better.
From a quick look, I'd perhaps try to just get rid of the _props attribute and always store the properties directly in the instantiated artist (creating it as early as possible).
It appears that the artist _is_ generally used, and the only real need for `_SelectorWidget._props` is in `SpanSelector.new_axes`, which needs to know the properties when attaching a new `Axes`.
```

## Patch

```diff
diff --git a/lib/matplotlib/widgets.py b/lib/matplotlib/widgets.py
--- a/lib/matplotlib/widgets.py
+++ b/lib/matplotlib/widgets.py
@@ -2457,15 +2457,16 @@ def artists(self):
 
     def set_props(self, **props):
         """
-        Set the properties of the selector artist. See the `props` argument
-        in the selector docstring to know which properties are supported.
+        Set the properties of the selector artist.
+
+        See the *props* argument in the selector docstring to know which properties are
+        supported.
         """
         artist = self._selection_artist
         props = cbook.normalize_kwargs(props, artist)
         artist.set(**props)
         if self.useblit:
             self.update()
-        self._props.update(props)
 
     def set_handle_props(self, **handle_props):
         """
@@ -2658,7 +2659,6 @@ def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
         # but we maintain it until it is removed
         self._pressv = None
 
-        self._props = props
         self.onmove_callback = onmove_callback
         self.minspan = minspan
 
@@ -2670,7 +2670,7 @@ def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
 
         # Reset canvas so that `new_axes` connects events.
         self.canvas = None
-        self.new_axes(ax)
+        self.new_axes(ax, _props=props)
 
         # Setup handles
         self._handle_props = {
@@ -2686,7 +2686,7 @@ def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
         # prev attribute is deprecated but we still need to maintain it
         self._prev = (0, 0)
 
-    def new_axes(self, ax):
+    def new_axes(self, ax, *, _props=None):
         """Set SpanSelector to operate on a new Axes."""
         self.ax = ax
         if self.canvas is not ax.figure.canvas:
@@ -2705,10 +2705,11 @@ def new_axes(self, ax):
         else:
             trans = ax.get_yaxis_transform()
             w, h = 1, 0
-        rect_artist = Rectangle((0, 0), w, h,
-                                transform=trans,
-                                visible=False,
-                                **self._props)
+        rect_artist = Rectangle((0, 0), w, h, transform=trans, visible=False)
+        if _props is not None:
+            rect_artist.update(_props)
+        elif self._selection_artist is not None:
+            rect_artist.update_from(self._selection_artist)
 
         self.ax.add_patch(rect_artist)
         self._selection_artist = rect_artist
@@ -3287,9 +3288,9 @@ def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False,
         if props is None:
             props = dict(facecolor='red', edgecolor='black',
                          alpha=0.2, fill=True)
-        self._props = {**props, 'animated': self.useblit}
-        self._visible = self._props.pop('visible', self._visible)
-        to_draw = self._init_shape(**self._props)
+        props = {**props, 'animated': self.useblit}
+        self._visible = props.pop('visible', self._visible)
+        to_draw = self._init_shape(**props)
         self.ax.add_patch(to_draw)
 
         self._selection_artist = to_draw
@@ -3305,8 +3306,7 @@ def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False,
 
         if self._interactive:
             self._handle_props = {
-                'markeredgecolor': (self._props or {}).get(
-                    'edgecolor', 'black'),
+                'markeredgecolor': (props or {}).get('edgecolor', 'black'),
                 **cbook.normalize_kwargs(handle_props, Line2D)}
 
             self._corner_order = ['SW', 'SE', 'NE', 'NW']
@@ -3942,13 +3942,13 @@ def __init__(self, ax, onselect, useblit=False,
 
         if props is None:
             props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
-        self._props = {**props, 'animated': self.useblit}
-        self._selection_artist = line = Line2D([], [], **self._props)
+        props = {**props, 'animated': self.useblit}
+        self._selection_artist = line = Line2D([], [], **props)
         self.ax.add_line(line)
 
         if handle_props is None:
             handle_props = dict(markeredgecolor='k',
-                                markerfacecolor=self._props.get('color', 'k'))
+                                markerfacecolor=props.get('color', 'k'))
         self._handle_props = handle_props
         self._polygon_handles = ToolHandles(self.ax, [], [],
                                             useblit=self.useblit,

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_widgets.py b/lib/matplotlib/tests/test_widgets.py
--- a/lib/matplotlib/tests/test_widgets.py
+++ b/lib/matplotlib/tests/test_widgets.py
@@ -987,6 +987,19 @@ def test_lasso_selector(ax, kwargs):
     onselect.assert_called_once_with([(100, 100), (125, 125), (150, 150)])
 
 
+def test_lasso_selector_set_props(ax):
+    onselect = mock.Mock(spec=noop, return_value=None)
+
+    tool = widgets.LassoSelector(ax, onselect, props=dict(color='b', alpha=0.2))
+
+    artist = tool._selection_artist
+    assert mcolors.same_color(artist.get_color(), 'b')
+    assert artist.get_alpha() == 0.2
+    tool.set_props(color='r', alpha=0.3)
+    assert mcolors.same_color(artist.get_color(), 'r')
+    assert artist.get_alpha() == 0.3
+
+
 def test_CheckButtons(ax):
     check = widgets.CheckButtons(ax, ('a', 'b', 'c'), (True, False, True))
     assert check.get_status() == [True, False, True]

```


## Code snippets

### 1 - galleries/examples/widgets/lasso_selector_demo_sgskip.py:

Start line: 76, End line: 112

```python
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.LassoSelector`
#    - `matplotlib.path.Path`
```
### 2 - lib/matplotlib/widgets.py:

Start line: 1145, End line: 1196

```python
class CheckButtons(AxesWidget):

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

    def set_frame_props(self, props):
        """
        Set properties of the check button frames.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button frames.
        """
        _api.check_isinstance(dict, props=props)
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        self._frames.update(props)

    def set_check_props(self, props):
        """
        Set properties of the check button checks.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button check.
        """
        _api.check_isinstance(dict, props=props)
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        actives = self.get_status()
        self._checks.update(props)
        # If new colours are supplied, then we must re-apply the status.
        self._init_status(actives)
```
### 3 - galleries/examples/event_handling/lasso_demo.py:

Start line: 44, End line: 107

```python
class LassoManager:
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data

        self.Nxy = len(data)

        facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        self.collection = RegularPolyCollection(
            6, sizes=(100,),
            facecolors=facecolors,
            offsets=self.xys,
            offset_transform=ax.transData)

        ax.add_collection(self.collection)

        self.cid_press = self.canvas.mpl_connect('button_press_event',
                                                 self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event',
                                                   self.on_release)

    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = Datum.colorin
            else:
                facecolors[i] = Datum.colorout

        self.canvas.draw_idle()
        del self.lasso

    def on_press(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def on_release(self, event):
        if hasattr(self, 'lasso') and self.canvas.widgetlock.isowner(self.lasso):
            self.canvas.widgetlock.release(self.lasso)


if __name__ == '__main__':

    np.random.seed(19680801)

    data = [Datum(*xy) for xy in np.random.rand(100, 2)]
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    ax.set_title('Lasso points using left mouse button')

    lman = LassoManager(ax, data)

    plt.show()
```
### 4 - lib/matplotlib/artist.py:

Start line: 1205, End line: 1243

```python
class Artist:

    def update(self, props):
        """
        Update this artist's properties from the dict *props*.

        Parameters
        ----------
        props : dict
        """
        return self._update_props(
            props, "{cls.__name__!r} object has no property {prop_name!r}")

    def _internal_update(self, kwargs):
        """
        Update artist properties without prenormalizing them, but generating
        errors as if calling `set`.

        The lack of prenormalization is to maintain backcompatibility.
        """
        return self._update_props(
            kwargs, "{cls.__name__}.set() got an unexpected keyword argument "
            "{prop_name!r}")

    def set(self, **kwargs):
        # docstring and signature are auto-generated via
        # Artist._update_set_signature_and_docstring() at the end of the
        # module.
        return self._internal_update(cbook.normalize_kwargs(kwargs, self))

    @contextlib.contextmanager
    def _cm_set(self, **kwargs):
        """
        `.Artist.set` context-manager that restores original values at exit.
        """
        orig_vals = {k: getattr(self, f"get_{k}")() for k in kwargs}
        try:
            self.set(**kwargs)
            yield
        finally:
            self.set(**orig_vals)
```
### 5 - galleries/examples/widgets/lasso_selector_demo_sgskip.py:

Start line: 1, End line: 73

```python
"""
==============
Lasso Selector
==============

Interactively selecting data points with the lasso tool.

This examples plots a scatter plot. You can then select a few points by drawing
a lasso loop around the points on the graph. To draw, just click
on the graph, hold, and drag it around the points you need to select.
"""


import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
```
### 6 - lib/matplotlib/widgets.py:

Start line: 3846, End line: 4276

```python
class PolygonSelector(_SelectorWidget):
    """
    Select a polygon region of an Axes.

    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Once drawn individual vertices
    can be moved by clicking and dragging with the left mouse button, or
    removed by clicking the right mouse button.

    In addition, the following modifier keys can be used:

    - Hold *ctrl* and click and drag a vertex to reposition it before the
      polygon has been completed.
    - Hold the *shift* key and click and drag anywhere in the Axes to move
      all vertices.
    - Press the *esc* key to start a new polygon.

    For the selector to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    onselect : function
        When a polygon is completed or modified after completion,
        the *onselect* function is called and passed a list of the vertices as
        ``(xdata, ydata)`` tuples.

    useblit : bool, default: False
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :doc:`/tutorials/advanced/blitting`
        for details.

    props : dict, optional
        Properties with which the line is drawn, see `matplotlib.lines.Line2D`
        for valid properties.
        Default:

            ``dict(color='k', linestyle='-', linewidth=2, alpha=0.5)``

    handle_props : dict, optional
        Artist properties for the markers drawn at the vertices of the polygon.
        See the marker arguments in `matplotlib.lines.Line2D` for valid
        properties.  Default values are defined in ``mpl.rcParams`` except for
        the default value of ``markeredgecolor`` which will be the same as the
        ``color`` property in *props*.

    grab_range : float, default: 10
        A vertex is selected (to complete the polygon or to move a vertex) if
        the mouse click is within *grab_range* pixels of the vertex.

    draw_bounding_box : bool, optional
        If `True`, a bounding box will be drawn around the polygon selector
        once it is complete. This box can be used to move and resize the
        selector.

    box_handle_props : dict, optional
        Properties to set for the box handles. See the documentation for the
        *handle_props* argument to `RectangleSelector` for more info.

    box_props : dict, optional
        Properties to set for the box. See the documentation for the *props*
        argument to `RectangleSelector` for more info.

    Examples
    --------
    :doc:`/gallery/widgets/polygon_selector_simple`
    :doc:`/gallery/widgets/polygon_selector_demo`

    Notes
    -----
    If only one point remains after removing points, the selector reverts to an
    incomplete state and you can start drawing a new polygon from the existing
    point.
    """

    @_api.make_keyword_only("3.7", name="useblit")
    def __init__(self, ax, onselect, useblit=False,
                 props=None, handle_props=None, grab_range=10, *,
                 draw_bounding_box=False, box_handle_props=None,
                 box_props=None):
        # The state modifiers 'move', 'square', and 'center' are expected by
        # _SelectorWidget but are not supported by PolygonSelector
        # Note: could not use the existing 'move' state modifier in-place of
        # 'move_all' because _SelectorWidget automatically discards 'move'
        # from the state on button release.
        state_modifier_keys = dict(clear='escape', move_vertex='control',
                                   move_all='shift', move='not-applicable',
                                   square='not-applicable',
                                   center='not-applicable',
                                   rotate='not-applicable')
        super().__init__(ax, onselect, useblit=useblit,
                         state_modifier_keys=state_modifier_keys)

        self._xys = [(0, 0)]

        if props is None:
            props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        self._props = {**props, 'animated': self.useblit}
        self._selection_artist = line = Line2D([], [], **self._props)
        self.ax.add_line(line)

        if handle_props is None:
            handle_props = dict(markeredgecolor='k',
                                markerfacecolor=self._props.get('color', 'k'))
        self._handle_props = handle_props
        self._polygon_handles = ToolHandles(self.ax, [], [],
                                            useblit=self.useblit,
                                            marker_props=self._handle_props)

        self._active_handle_idx = -1
        self.grab_range = grab_range

        self.set_visible(True)
        self._draw_box = draw_bounding_box
        self._box = None

        if box_handle_props is None:
            box_handle_props = {}
        self._box_handle_props = self._handle_props.update(box_handle_props)
        self._box_props = box_props

    def _get_bbox(self):
        return self._selection_artist.get_bbox()

    def _add_box(self):
        self._box = RectangleSelector(self.ax,
                                      onselect=lambda *args, **kwargs: None,
                                      useblit=self.useblit,
                                      grab_range=self.grab_range,
                                      handle_props=self._box_handle_props,
                                      props=self._box_props,
                                      interactive=True)
        self._box._state_modifier_keys.pop('rotate')
        self._box.connect_event('motion_notify_event', self._scale_polygon)
        self._update_box()
        # Set state that prevents the RectangleSelector from being created
        # by the user
        self._box._allow_creation = False
        self._box._selection_completed = True
        self._draw_polygon()

    def _remove_box(self):
        if self._box is not None:
            self._box.set_visible(False)
            self._box = None

    def _update_box(self):
        # Update selection box extents to the extents of the polygon
        if self._box is not None:
            bbox = self._get_bbox()
            self._box.extents = [bbox.x0, bbox.x1, bbox.y0, bbox.y1]
            # Save a copy
            self._old_box_extents = self._box.extents

    def _scale_polygon(self, event):
        """
        Scale the polygon selector points when the bounding box is moved or
        scaled.

        This is set as a callback on the bounding box RectangleSelector.
        """
        if not self._selection_completed:
            return

        if self._old_box_extents == self._box.extents:
            return

        # Create transform from old box to new box
        x1, y1, w1, h1 = self._box._rect_bbox
        old_bbox = self._get_bbox()
        t = (transforms.Affine2D()
             .translate(-old_bbox.x0, -old_bbox.y0)
             .scale(1 / old_bbox.width, 1 / old_bbox.height)
             .scale(w1, h1)
             .translate(x1, y1))

        # Update polygon verts.  Must be a list of tuples for consistency.
        new_verts = [(x, y) for x, y in t.transform(np.array(self.verts))]
        self._xys = [*new_verts, new_verts[0]]
        self._draw_polygon()
        self._old_box_extents = self._box.extents

    @property
    def _handles_artists(self):
        return self._polygon_handles.artists

    def _remove_vertex(self, i):
        """Remove vertex with index i."""
        if (len(self._xys) > 2 and
                self._selection_completed and
                i in (0, len(self._xys) - 1)):
            # If selecting the first or final vertex, remove both first and
            # last vertex as they are the same for a closed polygon
            self._xys.pop(0)
            self._xys.pop(-1)
            # Close the polygon again by appending the new first vertex to the
            # end
            self._xys.append(self._xys[0])
        else:
            self._xys.pop(i)
        if len(self._xys) <= 2:
            # If only one point left, return to incomplete state to let user
            # start drawing again
            self._selection_completed = False
            self._remove_box()

    def _press(self, event):
        """Button press event handler."""
        # Check for selection of a tool handle.
        if ((self._selection_completed or 'move_vertex' in self._state)
                and len(self._xys) > 0):
            h_idx, h_dist = self._polygon_handles.closest(event.x, event.y)
            if h_dist < self.grab_range:
                self._active_handle_idx = h_idx
        # Save the vertex positions at the time of the press event (needed to
        # support the 'move_all' state modifier).
        self._xys_at_press = self._xys.copy()

    def _release(self, event):
        """Button release event handler."""
        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1

        # Complete the polygon.
        elif len(self._xys) > 3 and self._xys[-1] == self._xys[0]:
            self._selection_completed = True
            if self._draw_box and self._box is None:
                self._add_box()

        # Place new vertex.
        elif (not self._selection_completed
              and 'move_all' not in self._state
              and 'move_vertex' not in self._state):
            self._xys.insert(-1, (event.xdata, event.ydata))

        if self._selection_completed:
            self.onselect(self.verts)

    def onmove(self, event):
        """Cursor move event handler and validator."""
        # Method overrides _SelectorWidget.onmove because the polygon selector
        # needs to process the move callback even if there is no button press.
        # _SelectorWidget.onmove include logic to ignore move event if
        # _eventpress is None.
        if not self.ignore(event):
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler."""
        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xys[idx] = event.xdata, event.ydata
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._selection_completed:
                self._xys[-1] = event.xdata, event.ydata

        # Move all vertices.
        elif 'move_all' in self._state and self._eventpress:
            dx = event.xdata - self._eventpress.xdata
            dy = event.ydata - self._eventpress.ydata
            for k in range(len(self._xys)):
                x_at_press, y_at_press = self._xys_at_press[k]
                self._xys[k] = x_at_press + dx, y_at_press + dy

        # Do nothing if completed or waiting for a move.
        elif (self._selection_completed
              or 'move_vertex' in self._state or 'move_all' in self._state):
            return

        # Position pending vertex.
        else:
            # Calculate distance to the start vertex.
            x0, y0 = \
                self._selection_artist.get_transform().transform(self._xys[0])
            v0_dist = np.hypot(x0 - event.x, y0 - event.y)
            # Lock on to the start vertex if near it and ready to complete.
            if len(self._xys) > 3 and v0_dist < self.grab_range:
                self._xys[-1] = self._xys[0]
            else:
                self._xys[-1] = event.xdata, event.ydata

        self._draw_polygon()

    def _on_key_press(self, event):
        """Key press event handler."""
        # Remove the pending vertex if entering the 'move_vertex' or
        # 'move_all' mode
        if (not self._selection_completed
                and ('move_vertex' in self._state or
                     'move_all' in self._state)):
            self._xys.pop()
            self._draw_polygon()

    def _on_key_release(self, event):
        """Key release event handler."""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if (not self._selection_completed
                and
                (event.key == self._state_modifier_keys.get('move_vertex')
                 or event.key == self._state_modifier_keys.get('move_all'))):
            self._xys.append((event.xdata, event.ydata))
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self._state_modifier_keys.get('clear'):
            event = self._clean_event(event)
            self._xys = [(event.xdata, event.ydata)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)

    def _draw_polygon(self):
        """Redraw the polygon based on the new vertex positions."""
        xs, ys = zip(*self._xys) if self._xys else ([], [])
        self._selection_artist.set_data(xs, ys)
        self._update_box()
        # Only show one tool handle at the start and end vertex of the polygon
        # if the polygon is completed or the user is locked on to the start
        # vertex.
        if (self._selection_completed
                or (len(self._xys) > 3
                    and self._xys[-1] == self._xys[0])):
            self._polygon_handles.set_data(xs[:-1], ys[:-1])
        else:
            self._polygon_handles.set_data(xs, ys)
        self.update()

    @property
    def verts(self):
        """The polygon vertices, as a list of ``(x, y)`` pairs."""
        return self._xys[:-1]

    @verts.setter
    def verts(self, xys):
        """
        Set the polygon vertices.

        This will remove any preexisting vertices, creating a complete polygon
        with the new vertices.
        """
        self._xys = [*xys, xys[0]]
        self._selection_completed = True
        self.set_visible(True)
        if self._draw_box and self._box is None:
            self._add_box()
        self._draw_polygon()


class Lasso(AxesWidget):
    """
    Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    `~matplotlib.path.Path.contains_point` to select data points from an image.

    Unlike `LassoSelector`, this must be initialized with a starting
    point *xy*, and the `Lasso` events are destroyed upon release.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    xy : (float, float)
        Coordinates of the start of the lasso.
    callback : callable
        Whenever the lasso is released, the *callback* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :doc:`/tutorials/advanced/blitting`
        for details.
    """

    @_api.make_keyword_only("3.7", name="useblit")
    def __init__(self, ax, xy, callback, useblit=True):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        x, y = xy
        self.verts = [(x, y)]
        self.line = Line2D([x], [y], linestyle='-', color='black', lw=2)
        self.ax.add_line(self.line)
        self.callback = callback
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if len(self.verts) > 2:
                self.callback(self.verts)
            self.line.remove()
        self.verts = None
        self.disconnect_events()

    def onmove(self, event):
        if self.ignore(event):
            return
        if self.verts is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self.verts.append((event.xdata, event.ydata))

        self.line.set_data(list(zip(*self.verts)))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
```
### 7 - galleries/examples/event_handling/lasso_demo.py:

Start line: 1, End line: 41

```python
"""
==========
Lasso Demo
==========

Show how to use a lasso to select a set of points and get the indices
of the selected points.  A callback is used to change the color of the
selected points

This is currently a proof-of-concept implementation (though it is
usable as is).  There will be some refinement of the API.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors as mcolors
from matplotlib import path
from matplotlib.collections import RegularPolyCollection
from matplotlib.widgets import Lasso


class Datum:
    colorin = mcolors.to_rgba("red")
    colorout = mcolors.to_rgba("blue")

    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include:
            self.color = self.colorin
        else:
            self.color = self.colorout
```
### 8 - lib/matplotlib/widgets.py:

Start line: 3147, End line: 3843

```python
_RECTANGLESELECTOR_PARAMETERS_DOCSTRING =
    r"""
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent axes for the widget.

    onselect : function
        A callback function that is called after a release event and the
        selection is created, changed or removed.
        It must have the signature::

            def onselect(eclick: MouseEvent, erelease: MouseEvent)

        where *eclick* and *erelease* are the mouse click and release
        `.MouseEvent`\s that start and complete the selection.

    minspanx : float, default: 0
        Selections with an x-span less than or equal to *minspanx* are removed
        (when already existing) or cancelled.

    minspany : float, default: 0
        Selections with an y-span less than or equal to *minspanx* are removed
        (when already existing) or cancelled.

    useblit : bool, default: False
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :doc:`/tutorials/advanced/blitting`
        for details.

    props : dict, optional
        Properties with which the __ARTIST_NAME__ is drawn. See
        `matplotlib.patches.Patch` for valid properties.
        Default:

        ``dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)``

    spancoords : {"data", "pixels"}, default: "data"
        Whether to interpret *minspanx* and *minspany* in data or in pixel
        coordinates.

    button : `.MouseButton`, list of `.MouseButton`, default: all buttons
        Button(s) that trigger rectangle selection.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be
        activated.

    handle_props : dict, optional
        Properties with which the interactive handles (marker artists) are
        drawn. See the marker arguments in `matplotlib.lines.Line2D` for valid
        properties.  Default values are defined in ``mpl.rcParams`` except for
        the default value of ``markeredgecolor`` which will be the same as the
        ``edgecolor`` property in *props*.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    state_modifier_keys : dict, optional
        Keyboard modifiers which affect the widget's behavior.  Values
        amend the defaults, which are:

        - "move": Move the existing shape, default: no modifier.
        - "clear": Clear the current shape, default: "escape".
        - "square": Make the shape square, default: "shift".
        - "center": change the shape around its center, default: "ctrl".
        - "rotate": Rotate the shape around its center between -45° and 45°,
          default: "r".

        "square" and "center" can be combined. The square shape can be defined
        in data or display coordinates as determined by the
        ``use_data_coordinates`` argument specified when creating the selector.

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within
        its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be
        ignored.

    use_data_coordinates : bool, default: False
        If `True`, the "square" shape of the selector is defined in
        data coordinates instead of display coordinates.
    """


@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace(
    '__ARTIST_NAME__', 'rectangle'))
class RectangleSelector(_SelectorWidget):
    """
    Select a rectangular region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(eclick, erelease):
    ...     print(eclick.xdata, eclick.ydata)
    ...     print(erelease.xdata, erelease.ydata)
    >>> props = dict(facecolor='blue', alpha=0.5)
    >>> rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,
    ...                                   props=props)
    >>> fig.show()
    >>> rect.add_state('square')

    See also: :doc:`/gallery/widgets/rectangle_selector`
    """

    def __init__(self, ax, onselect, *, minspanx=0, minspany=0, useblit=False,
                 props=None, spancoords='data', button=None, grab_range=10,
                 handle_props=None, interactive=False,
                 state_modifier_keys=None, drag_from_anywhere=False,
                 ignore_event_outside=False, use_data_coordinates=False):
        super().__init__(ax, onselect, useblit=useblit, button=button,
                         state_modifier_keys=state_modifier_keys,
                         use_data_coordinates=use_data_coordinates)

        self._interactive = interactive
        self.drag_from_anywhere = drag_from_anywhere
        self.ignore_event_outside = ignore_event_outside
        self._rotation = 0.0
        self._aspect_ratio_correction = 1.0

        # State to allow the option of an interactive selector that can't be
        # interactively drawn. This is used in PolygonSelector as an
        # interactive bounding box to allow the polygon to be easily resized
        self._allow_creation = True

        if props is None:
            props = dict(facecolor='red', edgecolor='black',
                         alpha=0.2, fill=True)
        self._props = {**props, 'animated': self.useblit}
        self._visible = self._props.pop('visible', self._visible)
        to_draw = self._init_shape(**self._props)
        self.ax.add_patch(to_draw)

        self._selection_artist = to_draw
        self._set_aspect_ratio_correction()

        self.minspanx = minspanx
        self.minspany = minspany

        _api.check_in_list(['data', 'pixels'], spancoords=spancoords)
        self.spancoords = spancoords

        self.grab_range = grab_range

        if self._interactive:
            self._handle_props = {
                'markeredgecolor': (self._props or {}).get(
                    'edgecolor', 'black'),
                **cbook.normalize_kwargs(handle_props, Line2D)}

            self._corner_order = ['SW', 'SE', 'NE', 'NW']
            xc, yc = self.corners
            self._corner_handles = ToolHandles(self.ax, xc, yc,
                                               marker_props=self._handle_props,
                                               useblit=self.useblit)

            self._edge_order = ['W', 'S', 'E', 'N']
            xe, ye = self.edge_centers
            self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s',
                                             marker_props=self._handle_props,
                                             useblit=self.useblit)

            xc, yc = self.center
            self._center_handle = ToolHandles(self.ax, [xc], [yc], marker='s',
                                              marker_props=self._handle_props,
                                              useblit=self.useblit)

            self._active_handle = None

        self._extents_on_press = None

    @property
    def _handles_artists(self):
        return (*self._center_handle.artists, *self._corner_handles.artists,
                *self._edge_handles.artists)

    def _init_shape(self, **props):
        return Rectangle((0, 0), 0, 1, visible=False,
                         rotation_point='center', **props)

    def _press(self, event):
        """Button press event handler."""
        # make the drawn box/line visible get the click-coordinates,
        # button, ...
        if self._interactive and self._selection_artist.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if ((self._active_handle is None or not self._interactive) and
                self._allow_creation):
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        if (self._active_handle is None and not self.ignore_event_outside and
                self._allow_creation):
            x = event.xdata
            y = event.ydata
            self._visible = False
            self.extents = x, x, y, y
            self._visible = True
        else:
            self.set_visible(True)

        self._extents_on_press = self.extents
        self._rotation_on_press = self._rotation
        self._set_aspect_ratio_correction()

        return False

    def _release(self, event):
        """Button release event handler."""
        if not self._interactive:
            self._selection_artist.set_visible(False)

        if (self._active_handle is None and self._selection_completed and
                self.ignore_event_outside):
            return

        # update the eventpress and eventrelease with the resulting extents
        x0, x1, y0, y1 = self.extents
        self._eventpress.xdata = x0
        self._eventpress.ydata = y0
        xy0 = self.ax.transData.transform([x0, y0])
        self._eventpress.x, self._eventpress.y = xy0

        self._eventrelease.xdata = x1
        self._eventrelease.ydata = y1
        xy1 = self.ax.transData.transform([x1, y1])
        self._eventrelease.x, self._eventrelease.y = xy1

        # calculate dimensions of box or line
        if self.spancoords == 'data':
            spanx = abs(self._eventpress.xdata - self._eventrelease.xdata)
            spany = abs(self._eventpress.ydata - self._eventrelease.ydata)
        elif self.spancoords == 'pixels':
            spanx = abs(self._eventpress.x - self._eventrelease.x)
            spany = abs(self._eventpress.y - self._eventrelease.y)
        else:
            _api.check_in_list(['data', 'pixels'],
                               spancoords=self.spancoords)
        # check if drawn distance (if it exists) is not too small in
        # either x or y-direction
        if spanx <= self.minspanx or spany <= self.minspany:
            if self._selection_completed:
                # Call onselect, only when the selection is already existing
                self.onselect(self._eventpress, self._eventrelease)
            self._clear_without_update()
        else:
            self.onselect(self._eventpress, self._eventrelease)
            self._selection_completed = True

        self.update()
        self._active_handle = None
        self._extents_on_press = None

        return False

    def _onmove(self, event):
        """
        Motion notify event handler.

        This can do one of four things:
        - Translate
        - Rotate
        - Re-size
        - Continue the creation of a new shape
        """
        eventpress = self._eventpress
        # The calculations are done for rotation at zero: we apply inverse
        # transformation to events except when we rotate and move
        state = self._state
        rotate = ('rotate' in state and
                  self._active_handle in self._corner_order)
        move = self._active_handle == 'C'
        resize = self._active_handle and not move

        if resize:
            inv_tr = self._get_rotation_transform().inverted()
            event.xdata, event.ydata = inv_tr.transform(
                [event.xdata, event.ydata])
            eventpress.xdata, eventpress.ydata = inv_tr.transform(
                [eventpress.xdata, eventpress.ydata]
                )

        dx = event.xdata - eventpress.xdata
        dy = event.ydata - eventpress.ydata
        # refmax is used when moving the corner handle with the square state
        # and is the maximum between refx and refy
        refmax = None
        if self._use_data_coordinates:
            refx, refy = dx, dy
        else:
            # Get dx/dy in display coordinates
            refx = event.x - eventpress.x
            refy = event.y - eventpress.y

        x0, x1, y0, y1 = self._extents_on_press
        # rotate an existing shape
        if rotate:
            # calculate angle abc
            a = np.array([eventpress.xdata, eventpress.ydata])
            b = np.array(self.center)
            c = np.array([event.xdata, event.ydata])
            angle = (np.arctan2(c[1]-b[1], c[0]-b[0]) -
                     np.arctan2(a[1]-b[1], a[0]-b[0]))
            self.rotation = np.rad2deg(self._rotation_on_press + angle)

        elif resize:
            size_on_press = [x1 - x0, y1 - y0]
            center = [x0 + size_on_press[0] / 2, y0 + size_on_press[1] / 2]

            # Keeping the center fixed
            if 'center' in state:
                # hh, hw are half-height and half-width
                if 'square' in state:
                    # when using a corner, find which reference to use
                    if self._active_handle in self._corner_order:
                        refmax = max(refx, refy, key=abs)
                    if self._active_handle in ['E', 'W'] or refmax == refx:
                        hw = event.xdata - center[0]
                        hh = hw / self._aspect_ratio_correction
                    else:
                        hh = event.ydata - center[1]
                        hw = hh * self._aspect_ratio_correction
                else:
                    hw = size_on_press[0] / 2
                    hh = size_on_press[1] / 2
                    # cancel changes in perpendicular direction
                    if self._active_handle in ['E', 'W'] + self._corner_order:
                        hw = abs(event.xdata - center[0])
                    if self._active_handle in ['N', 'S'] + self._corner_order:
                        hh = abs(event.ydata - center[1])

                x0, x1, y0, y1 = (center[0] - hw, center[0] + hw,
                                  center[1] - hh, center[1] + hh)

            else:
                # change sign of relative changes to simplify calculation
                # Switch variables so that x1 and/or y1 are updated on move
                if 'W' in self._active_handle:
                    x0 = x1
                if 'S' in self._active_handle:
                    y0 = y1
                if self._active_handle in ['E', 'W'] + self._corner_order:
                    x1 = event.xdata
                if self._active_handle in ['N', 'S'] + self._corner_order:
                    y1 = event.ydata
                if 'square' in state:
                    # when using a corner, find which reference to use
                    if self._active_handle in self._corner_order:
                        refmax = max(refx, refy, key=abs)
                    if self._active_handle in ['E', 'W'] or refmax == refx:
                        sign = np.sign(event.ydata - y0)
                        y1 = y0 + sign * abs(x1 - x0) / \
                            self._aspect_ratio_correction
                    else:
                        sign = np.sign(event.xdata - x0)
                        x1 = x0 + sign * abs(y1 - y0) * \
                            self._aspect_ratio_correction

        elif move:
            x0, x1, y0, y1 = self._extents_on_press
            dx = event.xdata - eventpress.xdata
            dy = event.ydata - eventpress.ydata
            x0 += dx
            x1 += dx
            y0 += dy
            y1 += dy

        else:
            # Create a new shape
            self._rotation = 0
            # Don't create a new rectangle if there is already one when
            # ignore_event_outside=True
            if ((self.ignore_event_outside and self._selection_completed) or
                    not self._allow_creation):
                return
            center = [eventpress.xdata, eventpress.ydata]
            dx = (event.xdata - center[0]) / 2.
            dy = (event.ydata - center[1]) / 2.

            # square shape
            if 'square' in state:
                refmax = max(refx, refy, key=abs)
                if refmax == refx:
                    dy = np.sign(dy) * abs(dx) / self._aspect_ratio_correction
                else:
                    dx = np.sign(dx) * abs(dy) * self._aspect_ratio_correction

            # from center
            if 'center' in state:
                dx *= 2
                dy *= 2

            # from corner
            else:
                center[0] += dx
                center[1] += dy

            x0, x1, y0, y1 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        self.extents = x0, x1, y0, y1

    @property
    def _rect_bbox(self):
        return self._selection_artist.get_bbox().bounds

    def _set_aspect_ratio_correction(self):
        aspect_ratio = self.ax._get_aspect_ratio()
        self._selection_artist._aspect_ratio_correction = aspect_ratio
        if self._use_data_coordinates:
            self._aspect_ratio_correction = 1
        else:
            self._aspect_ratio_correction = aspect_ratio

    def _get_rotation_transform(self):
        aspect_ratio = self.ax._get_aspect_ratio()
        return Affine2D().translate(-self.center[0], -self.center[1]) \
                .scale(1, aspect_ratio) \
                .rotate(self._rotation) \
                .scale(1, 1 / aspect_ratio) \
                .translate(*self.center)

    @property
    def corners(self):
        """
        Corners of rectangle in data coordinates from lower left,
        moving clockwise.
        """
        x0, y0, width, height = self._rect_bbox
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        transform = self._get_rotation_transform()
        coords = transform.transform(np.array([xc, yc]).T).T
        return coords[0], coords[1]

    @property
    def edge_centers(self):
        """
        Midpoint of rectangle edges in data coordinates from left,
        moving anti-clockwise.
        """
        x0, y0, width, height = self._rect_bbox
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        transform = self._get_rotation_transform()
        coords = transform.transform(np.array([xe, ye]).T).T
        return coords[0], coords[1]

    @property
    def center(self):
        """Center of rectangle in data coordinates."""
        x0, y0, width, height = self._rect_bbox
        return x0 + width / 2., y0 + height / 2.

    @property
    def extents(self):
        """
        Return (xmin, xmax, ymin, ymax) in data coordinates as defined by the
        bounding box before rotation.
        """
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        # Update displayed shape
        self._draw_shape(extents)
        if self._interactive:
            # Update displayed handles
            self._corner_handles.set_data(*self.corners)
            self._edge_handles.set_data(*self.edge_centers)
            x, y = self.center
            self._center_handle.set_data([x], [y])
        self.set_visible(self._visible)
        self.update()

    @property
    def rotation(self):
        """
        Rotation in degree in interval [-45°, 45°]. The rotation is limited in
        range to keep the implementation simple.
        """
        return np.rad2deg(self._rotation)

    @rotation.setter
    def rotation(self, value):
        # Restrict to a limited range of rotation [-45°, 45°] to avoid changing
        # order of handles
        if -45 <= value and value <= 45:
            self._rotation = np.deg2rad(value)
            # call extents setter to draw shape and update handles positions
            self.extents = self.extents

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())

        xmin = max(xlim[0], xmin)
        ymin = max(ylim[0], ymin)
        xmax = min(xmax, xlim[1])
        ymax = min(ymax, ylim[1])

        self._selection_artist.set_x(xmin)
        self._selection_artist.set_y(ymin)
        self._selection_artist.set_width(xmax - xmin)
        self._selection_artist.set_height(ymax - ymin)
        self._selection_artist.set_angle(self.rotation)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        m_idx, m_dist = self._center_handle.closest(event.x, event.y)

        if 'move' in self._state:
            self._active_handle = 'C'
        # Set active handle as closest handle, if mouse click is close enough.
        elif m_dist < self.grab_range * 2:
            # Prioritise center handle over other handles
            self._active_handle = 'C'
        elif c_dist > self.grab_range and e_dist > self.grab_range:
            # Not close to any handles
            if self.drag_from_anywhere and self._contains(event):
                # Check if we've clicked inside the region
                self._active_handle = 'C'
            else:
                self._active_handle = None
                return
        elif c_dist < e_dist:
            # Closest to a corner handle
            self._active_handle = self._corner_order[c_idx]
        else:
            # Closest to an edge handle
            self._active_handle = self._edge_order[e_idx]

    def _contains(self, event):
        """Return True if event is within the patch."""
        return self._selection_artist.contains(event, radius=0)[0]

    @property
    def geometry(self):
        """
        Return an array of shape (2, 5) containing the
        x (``RectangleSelector.geometry[1, :]``) and
        y (``RectangleSelector.geometry[0, :]``) data coordinates of the four
        corners of the rectangle starting and ending in the top left corner.
        """
        if hasattr(self._selection_artist, 'get_verts'):
            xfm = self.ax.transData.inverted()
            y, x = xfm.transform(self._selection_artist.get_verts()).T
            return np.array([x, y])
        else:
            return np.array(self._selection_artist.get_data())


@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace(
    '__ARTIST_NAME__', 'ellipse'))
class EllipseSelector(RectangleSelector):
    """
    Select an elliptical region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    :doc:`/gallery/widgets/rectangle_selector`
    """
    def _init_shape(self, **props):
        return Ellipse((0, 0), 0, 1, visible=False, **props)

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        center = [x0 + (x1 - x0) / 2., y0 + (y1 - y0) / 2.]
        a = (xmax - xmin) / 2.
        b = (ymax - ymin) / 2.

        self._selection_artist.center = center
        self._selection_artist.width = 2 * a
        self._selection_artist.height = 2 * b
        self._selection_artist.angle = self.rotation

    @property
    def _rect_bbox(self):
        x, y = self._selection_artist.center
        width = self._selection_artist.width
        height = self._selection_artist.height
        return x - width / 2., y - height / 2., width, height


class LassoSelector(_SelectorWidget):
    """
    Selection curve of an arbitrary shape.

    For the selector to remain responsive you must keep a reference to it.

    The selected path can be used in conjunction with `~.Path.contains_point`
    to select data points from an image.

    In contrast to `Lasso`, `LassoSelector` is written with an interface
    similar to `RectangleSelector` and `SpanSelector`, and will continue to
    interact with the Axes until disconnected.

    Example usage::

        ax = plt.subplot()
        ax.plot(x, y)

        def onselect(verts):
            print(verts)
        lasso = LassoSelector(ax, onselect)

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    onselect : function
        Whenever the lasso is released, the *onselect* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :doc:`/tutorials/advanced/blitting`
        for details.
    props : dict, optional
        Properties with which the line is drawn, see `matplotlib.lines.Line2D`
        for valid properties. Default values are defined in ``mpl.rcParams``.
    button : `.MouseButton` or list of `.MouseButton`, optional
        The mouse buttons used for rectangle selection.  Default is ``None``,
        which corresponds to all buttons.
    """

    @_api.make_keyword_only("3.7", name="useblit")
    def __init__(self, ax, onselect, useblit=True, props=None, button=None):
        super().__init__(ax, onselect, useblit=useblit, button=button)
        self.verts = None
        props = {
            **(props if props is not None else {}),
            # Note that self.useblit may be != useblit, if the canvas doesn't
            # support blitting.
            'animated': self.useblit, 'visible': False,
        }
        line = Line2D([], [], **props)
        self.ax.add_line(line)
        self._selection_artist = line

    def _press(self, event):
        self.verts = [self._get_data(event)]
        self._selection_artist.set_visible(True)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts)
        self._selection_artist.set_data([[], []])
        self._selection_artist.set_visible(False)
        self.verts = None

    def _onmove(self, event):
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))
        self._selection_artist.set_data(list(zip(*self.verts)))

        self.update()
```
### 9 - lib/matplotlib/legend.py:

Start line: 659, End line: 689

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
### 10 - lib/matplotlib/artist.py:

Start line: 1180, End line: 1203

```python
class Artist:

    def _update_props(self, props, errfmt):
        """
        Helper for `.Artist.set` and `.Artist.update`.

        *errfmt* is used to generate error messages for invalid property
        names; it gets formatted with ``type(self)`` and the property name.
        """
        ret = []
        with cbook._setattr_cm(self, eventson=False):
            for k, v in props.items():
                # Allow attributes we want to be able to update through
                # art.update, art.set, setp.
                if k == "axes":
                    ret.append(setattr(self, k, v))
                else:
                    func = getattr(self, f"set_{k}", None)
                    if not callable(func):
                        raise AttributeError(
                            errfmt.format(cls=type(self), prop_name=k))
                    ret.append(func(v))
        if ret:
            self.pchanged()
            self.stale = True
        return ret
```
### 24 - lib/matplotlib/widgets.py:

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
