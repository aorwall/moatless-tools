# matplotlib__matplotlib-26311

| **matplotlib/matplotlib** | `3044bded1b23ae8dc73c1611b124e88db98308ac` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6036 |
| **Any found context length** | 6036 |
| **Avg pos** | 11.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/contour.py b/lib/matplotlib/contour.py
--- a/lib/matplotlib/contour.py
+++ b/lib/matplotlib/contour.py
@@ -370,7 +370,7 @@ def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=
         # path always starts with a MOVETO, and we consider there's an implicit
         # MOVETO (closing the last path) at the end.
         movetos = (codes == Path.MOVETO).nonzero()[0]
-        start = movetos[movetos < idx][-1]
+        start = movetos[movetos <= idx][-1]
         try:
             stop = movetos[movetos > idx][0]
         except IndexError:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/contour.py | 373 | 373 | 11 | 1 | 6036


## Problem Statement

```
[Bug]: labels can't be placed at start of contours
### Bug summary

For some combinations of contour shape and fontsize, the automatic label placement tries to put the label right at the start of the contour.  This is not currently possible on `main`.

### Code for reproduction

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()

_, ax = plt.subplots()
lats = lons = np.linspace(-np.pi / 2, np.pi / 2, 50, dtype=np.longdouble)
lons, lats = np.meshgrid(lons, lats)
wave = 0.75 * (np.sin(2 * lats) ** 8) * np.cos(4 * lons)
mean = 0.5 * np.cos(2 * lats) * ((np.sin(2 * lats)) ** 2 + 2)
data = wave + mean

cs = ax.contour(lons, lats, data)
cs.clabel(fontsize=9)
\`\`\`


### Actual outcome

\`\`\`
Traceback (most recent call last):
  File "[snip]/contour_clabel_start.py", line 14, in <module>
    cs.clabel(fontsize=9)
  File "[git-path]/matplotlib/lib/matplotlib/contour.py", line 222, in clabel
    self.labels(inline, inline_spacing)
  File "[git-path]/matplotlib/lib/matplotlib/contour.py", line 622, in labels
    rotation, path = self._split_path_and_get_label_rotation(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "[git-path]/matplotlib/lib/matplotlib/contour.py", line 373, in _split_path_and_get_label_rotation
    start = movetos[movetos < idx][-1]
            ~~~~~~~~~~~~~~~~~~~~~~^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
\`\`\`

### Expected outcome

With v3.7.1 I get

![image](https://github.com/matplotlib/matplotlib/assets/10599679/655bde83-dd20-428b-84e6-8318d7001911)


### Additional information

The fix is easy: https://github.com/matplotlib/matplotlib/commit/07f694dc3f0ef90e95e3dce44d4f4857b5dc6e55

Writing a test seems harder.  I tried pasting the above code into a test, and it passed against `main`.  I assume that is because the tests have different "screen space" than when I just run it as a script.

Marking as "release critical" because this is a regression.

### Operating system

RHEL7

### Matplotlib Version

main

### Matplotlib Backend

QtAgg

### Python version

3.11.3

### Jupyter version

N/A

### Installation

git checkout

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/contour.py** | 160 | 224| 647 | 647 | 18001 | 
| 2 | **1 lib/matplotlib/contour.py** | 226 | 282| 566 | 1213 | 18001 | 
| 3 | **1 lib/matplotlib/contour.py** | 398 | 436| 591 | 1804 | 18001 | 
| 4 | **1 lib/matplotlib/contour.py** | 497 | 527| 395 | 2199 | 18001 | 
| 5 | 2 galleries/examples/images_contours_and_fields/contour_label_demo.py | 1 | 88| 617 | 2816 | 18618 | 
| 6 | **2 lib/matplotlib/contour.py** | 284 | 316| 484 | 3300 | 18618 | 
| 7 | **2 lib/matplotlib/contour.py** | 42 | 68| 309 | 3609 | 18618 | 
| 8 | **2 lib/matplotlib/contour.py** | 596 | 647| 459 | 4068 | 18618 | 
| 9 | **2 lib/matplotlib/contour.py** | 438 | 496| 556 | 4624 | 18618 | 
| 10 | 3 galleries/examples/text_labels_and_annotations/label_subplots.py | 1 | 72| 606 | 5230 | 19224 | 
| **-> 11 <-** | **3 lib/matplotlib/contour.py** | 318 | 396| 806 | 6036 | 19224 | 
| 12 | **3 lib/matplotlib/contour.py** | 1130 | 1144| 198 | 6234 | 19224 | 
| 13 | **3 lib/matplotlib/contour.py** | 1 | 39| 237 | 6471 | 19224 | 
| 14 | 4 galleries/examples/text_labels_and_annotations/line_with_text.py | 52 | 88| 260 | 6731 | 19816 | 
| 15 | 5 galleries/examples/images_contours_and_fields/contour_demo.py | 84 | 124| 360 | 7091 | 20931 | 
| 16 | 5 galleries/examples/images_contours_and_fields/contour_demo.py | 1 | 83| 755 | 7846 | 20931 | 
| 17 | 6 galleries/examples/text_labels_and_annotations/align_ylabels.py | 41 | 87| 346 | 8192 | 21578 | 
| 18 | **6 lib/matplotlib/contour.py** | 71 | 158| 792 | 8984 | 21578 | 
| 19 | 7 galleries/examples/images_contours_and_fields/contourf_demo.py | 96 | 129| 340 | 9324 | 22697 | 
| 20 | 8 lib/matplotlib/pyplot.py | 2112 | 2847| 6074 | 15398 | 56778 | 
| 21 | 9 galleries/examples/specialty_plots/leftventricle_bullseye.py | 94 | 155| 697 | 16095 | 58383 | 
| 22 | **9 lib/matplotlib/contour.py** | 934 | 943| 199 | 16294 | 58383 | 
| 23 | 10 galleries/users_explain/text/text_intro.py | 269 | 336| 832 | 17126 | 62260 | 
| 24 | **10 lib/matplotlib/contour.py** | 529 | 545| 191 | 17317 | 62260 | 
| 25 | 11 lib/mpl_toolkits/mplot3d/axis3d.py | 325 | 402| 834 | 18151 | 68131 | 
| 26 | 12 galleries/examples/ticks/colorbar_tick_labelling_demo.py | 1 | 65| 470 | 18621 | 68601 | 
| 27 | **12 lib/matplotlib/contour.py** | 945 | 977| 369 | 18990 | 68601 | 
| 28 | 13 galleries/examples/text_labels_and_annotations/titles_demo.py | 1 | 60| 369 | 19359 | 68970 | 
| 29 | 13 galleries/examples/images_contours_and_fields/contourf_demo.py | 1 | 95| 779 | 20138 | 68970 | 
| 30 | 13 galleries/users_explain/text/text_intro.py | 180 | 268| 762 | 20900 | 68970 | 
| 31 | 13 galleries/users_explain/text/text_intro.py | 92 | 178| 780 | 21680 | 68970 | 
| 32 | 14 galleries/users_explain/axes/constrainedlayout_guide.py | 184 | 268| 863 | 22543 | 75680 | 
| 33 | **14 lib/matplotlib/contour.py** | 1308 | 1329| 266 | 22809 | 75680 | 
| 34 | 15 galleries/examples/text_labels_and_annotations/angle_annotation.py | 271 | 327| 707 | 23516 | 79115 | 
| 35 | 16 galleries/examples/ticks/tick-locators.py | 32 | 94| 647 | 24163 | 79973 | 
| 36 | 17 galleries/examples/lines_bars_and_markers/filled_step.py | 180 | 238| 493 | 24656 | 81599 | 
| 37 | 18 galleries/examples/lines_bars_and_markers/bar_label_demo.py | 1 | 107| 773 | 25429 | 82505 | 
| 38 | **18 lib/matplotlib/contour.py** | 1295 | 1306| 150 | 25579 | 82505 | 
| 39 | 19 galleries/examples/subplots_axes_and_figures/figure_title.py | 1 | 54| 456 | 26035 | 82961 | 
| 40 | 20 lib/matplotlib/axis.py | 1996 | 2012| 184 | 26219 | 104762 | 
| 41 | 21 galleries/examples/text_labels_and_annotations/multiline.py | 1 | 46| 386 | 26605 | 105148 | 
| 42 | 22 galleries/examples/text_labels_and_annotations/demo_text_path.py | 48 | 124| 731 | 27336 | 106202 | 
| 43 | 23 galleries/examples/text_labels_and_annotations/date.py | 1 | 65| 659 | 27995 | 106861 | 
| 44 | 23 galleries/users_explain/axes/constrainedlayout_guide.py | 529 | 648| 1131 | 29126 | 106861 | 
| 45 | 24 galleries/examples/images_contours_and_fields/contour_image.py | 1 | 75| 746 | 29872 | 107921 | 
| 46 | 25 galleries/examples/showcase/anatomy.py | 85 | 122| 453 | 30325 | 109161 | 
| 47 | 26 galleries/users_explain/axes/colorbar_placement.py | 1 | 85| 764 | 31089 | 110056 | 
| 48 | 27 galleries/examples/images_contours_and_fields/contourf_log.py | 1 | 63| 494 | 31583 | 110550 | 
| 49 | 28 galleries/examples/ticks/tick-formatters.py | 54 | 106| 563 | 32146 | 111548 | 
| 50 | 29 galleries/examples/ticks/centered_ticklabels.py | 1 | 50| 361 | 32507 | 111909 | 
| 51 | 30 galleries/examples/misc/contour_manual.py | 1 | 61| 617 | 33124 | 112526 | 
| 52 | 31 galleries/users_explain/artists/transforms_tutorial.py | 259 | 356| 1006 | 34130 | 118888 | 
| 53 | 32 galleries/examples/text_labels_and_annotations/text_alignment.py | 82 | 119| 239 | 34369 | 119871 | 
| 54 | 33 galleries/examples/subplots_axes_and_figures/align_labels_demo.py | 1 | 38| 289 | 34658 | 120160 | 
| 55 | 34 galleries/examples/text_labels_and_annotations/engineering_formatter.py | 1 | 45| 367 | 35025 | 120527 | 
| 56 | 34 galleries/users_explain/axes/constrainedlayout_guide.py | 1 | 105| 783 | 35808 | 120527 | 
| 57 | 35 galleries/users_explain/quick_start.py | 305 | 402| 968 | 36776 | 126534 | 
| 58 | 35 galleries/examples/text_labels_and_annotations/align_ylabels.py | 1 | 38| 301 | 37077 | 126534 | 
| 59 | 36 galleries/examples/specialty_plots/anscombe.py | 37 | 66| 346 | 37423 | 127446 | 
| 60 | 36 galleries/examples/images_contours_and_fields/contour_image.py | 76 | 108| 314 | 37737 | 127446 | 
| 61 | 36 galleries/users_explain/quick_start.py | 489 | 575| 1016 | 38753 | 127446 | 
| 62 | 36 galleries/users_explain/axes/constrainedlayout_guide.py | 106 | 183| 750 | 39503 | 127446 | 
| 63 | 37 galleries/examples/event_handling/ginput_manual_clabel_sgskip.py | 1 | 101| 631 | 40134 | 128077 | 
| 64 | 38 galleries/examples/text_labels_and_annotations/mathtext_examples.py | 58 | 99| 421 | 40555 | 129346 | 
| 65 | 39 galleries/examples/misc/tickedstroke_demo.py | 93 | 120| 237 | 40792 | 130345 | 
| 66 | 40 galleries/users_explain/axes/legend_guide.py | 104 | 178| 759 | 41551 | 133480 | 
| 67 | 41 galleries/examples/text_labels_and_annotations/demo_annotation_box.py | 101 | 120| 124 | 41675 | 134357 | 
| 68 | 42 galleries/examples/ticks/major_minor_demo.py | 1 | 97| 828 | 42503 | 135185 | 
| 69 | 42 lib/mpl_toolkits/mplot3d/axis3d.py | 465 | 486| 263 | 42766 | 135185 | 
| 70 | 43 lib/mpl_toolkits/mplot3d/axes3d.py | 3016 | 3076| 806 | 43572 | 166668 | 
| 71 | 44 galleries/examples/images_contours_and_fields/irregulardatagrid.py | 80 | 96| 144 | 43716 | 167576 | 
| 72 | **44 lib/matplotlib/contour.py** | 1461 | 1475| 187 | 43903 | 167576 | 
| 73 | 45 galleries/examples/axisartist/demo_ticklabel_alignment.py | 1 | 41| 269 | 44172 | 167845 | 
| 74 | 45 lib/matplotlib/axis.py | 349 | 404| 606 | 44778 | 167845 | 
| 75 | 46 galleries/examples/text_labels_and_annotations/annotation_demo.py | 226 | 285| 624 | 45402 | 171778 | 
| 76 | 46 galleries/users_explain/axes/constrainedlayout_guide.py | 442 | 527| 795 | 46197 | 171778 | 
| 77 | 47 galleries/examples/widgets/annotated_cursor.py | 288 | 357| 568 | 46765 | 174733 | 
| 78 | 47 galleries/users_explain/text/text_intro.py | 337 | 430| 752 | 47517 | 174733 | 
| 79 | 48 galleries/examples/text_labels_and_annotations/legend_demo.py | 165 | 183| 195 | 47712 | 176639 | 
| 80 | 48 galleries/users_explain/quick_start.py | 227 | 304| 835 | 48547 | 176639 | 
| 81 | 49 lib/mpl_toolkits/axisartist/floating_axes.py | 53 | 113| 725 | 49272 | 179322 | 
| 82 | 49 lib/mpl_toolkits/mplot3d/axis3d.py | 403 | 464| 708 | 49980 | 179322 | 
| 83 | 50 galleries/examples/subplots_axes_and_figures/secondary_axis.py | 104 | 195| 656 | 50636 | 180755 | 


### Hint

```
I left a comment on your commit.

Trying to target the end of a broken contour might be easier?
> Writing a test seems harder. I tried pasting the above code into a test, and it passed against main. I assume that is because the tests have different "screen space" than when I just run it as a script.

Can you set the DPI of the figure in your test to whatever you're using locally so that the rendered size is the same and therefore transforms to the same screen-size?
```

## Patch

```diff
diff --git a/lib/matplotlib/contour.py b/lib/matplotlib/contour.py
--- a/lib/matplotlib/contour.py
+++ b/lib/matplotlib/contour.py
@@ -370,7 +370,7 @@ def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=
         # path always starts with a MOVETO, and we consider there's an implicit
         # MOVETO (closing the last path) at the end.
         movetos = (codes == Path.MOVETO).nonzero()[0]
-        start = movetos[movetos < idx][-1]
+        start = movetos[movetos <= idx][-1]
         try:
             stop = movetos[movetos > idx][0]
         except IndexError:

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_contour.py b/lib/matplotlib/tests/test_contour.py
--- a/lib/matplotlib/tests/test_contour.py
+++ b/lib/matplotlib/tests/test_contour.py
@@ -1,6 +1,7 @@
 import datetime
 import platform
 import re
+from unittest import mock
 
 import contourpy  # type: ignore
 import numpy as np
@@ -233,6 +234,31 @@ def test_labels(split_collections):
     _maybe_split_collections(split_collections)
 
 
+def test_label_contour_start():
+    # Set up data and figure/axes that result in automatic labelling adding the
+    # label to the start of a contour
+
+    _, ax = plt.subplots(dpi=100)
+    lats = lons = np.linspace(-np.pi / 2, np.pi / 2, 50)
+    lons, lats = np.meshgrid(lons, lats)
+    wave = 0.75 * (np.sin(2 * lats) ** 8) * np.cos(4 * lons)
+    mean = 0.5 * np.cos(2 * lats) * ((np.sin(2 * lats)) ** 2 + 2)
+    data = wave + mean
+
+    cs = ax.contour(lons, lats, data)
+
+    with mock.patch.object(
+            cs, '_split_path_and_get_label_rotation',
+            wraps=cs._split_path_and_get_label_rotation) as mocked_splitter:
+        # Smoke test that we can add the labels
+        cs.clabel(fontsize=9)
+
+    # Verify at least one label was added to the start of a contour.  I.e. the
+    # splitting method was called with idx=0 at least once.
+    idxs = [cargs[0][1] for cargs in mocked_splitter.call_args_list]
+    assert 0 in idxs
+
+
 @pytest.mark.parametrize("split_collections", [False, True])
 @image_comparison(['contour_corner_mask_False.png', 'contour_corner_mask_True.png'],
                   remove_text=True, tol=1.88)

```


## Code snippets

### 1 - lib/matplotlib/contour.py:

Start line: 160, End line: 224

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
        self._clabel_zorder = 2 + self.get_zorder() if zorder is None else zorder

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

        self._label_font_props = font_manager.FontProperties(size=fontsize)

        if colors is None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = mcolors.ListedColormap(colors, N=len(self.labelLevelList))
            self.labelCValueList = list(range(len(self.labelLevelList)))
            self.labelMappable = cm.ScalarMappable(cmap=cmap,
                                                   norm=mcolors.NoNorm())

        self.labelXYs = []

        if np.iterable(manual):
            for x, y in manual:
                self.add_label_near(x, y, inline, inline_spacing)
        elif manual:
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

        return cbook.silent_list('text.Text', self.labelTexts)
```
### 2 - lib/matplotlib/contour.py:

Start line: 226, End line: 282

```python
class ContourLabeler:

    @_api.deprecated("3.7", alternative="cs.labelTexts[0].get_font()")
    @property
    def labelFontProps(self):
        return self._label_font_props

    @_api.deprecated("3.7", alternative=(
        "[cs.labelTexts[0].get_font().get_size()] * len(cs.labelLevelList)"))
    @property
    def labelFontSizeList(self):
        return [self._label_font_props.get_size()] * len(self.labelLevelList)

    @_api.deprecated("3.7", alternative="cs.labelTexts")
    @property
    def labelTextsList(self):
        return cbook.silent_list('text.Text', self.labelTexts)

    def print_label(self, linecontour, labelwidth):
        """Return whether a contour is long enough to hold a label."""
        return (len(linecontour) > 10 * labelwidth
                or (len(linecontour)
                    and (np.ptp(linecontour, axis=0) > 1.2 * labelwidth).any()))

    def too_close(self, x, y, lw):
        """Return whether a label is already near this location."""
        thresh = (1.2 * lw) ** 2
        return any((x - loc[0]) ** 2 + (y - loc[1]) ** 2 < thresh
                   for loc in self.labelXYs)

    def _get_nth_label_width(self, nth):
        """Return the width of the *nth* label, in pixels."""
        fig = self.axes.figure
        renderer = fig._get_renderer()
        return (Text(0, 0,
                     self.get_text(self.labelLevelList[nth], self.labelFmt),
                     figure=fig, fontproperties=self._label_font_props)
                .get_window_extent(renderer).width)

    @_api.deprecated("3.7", alternative="Artist.set")
    def set_label_props(self, label, text, color):
        """Set the label properties - color, fontsize, text."""
        label.set_text(text)
        label.set_color(color)
        label.set_fontproperties(self._label_font_props)
        label.set_clip_box(self.axes.bbox)

    def get_text(self, lev, fmt):
        """Get the text of the label."""
        if isinstance(lev, str):
            return lev
        elif isinstance(fmt, dict):
            return fmt.get(lev, '%1.3f')
        elif callable(getattr(fmt, "format_ticks", None)):
            return fmt.format_ticks([*self.labelLevelList, lev])[-1]
        elif callable(fmt):
            return fmt(lev)
        else:
            return fmt % lev
```
### 3 - lib/matplotlib/contour.py:

Start line: 398, End line: 436

```python
class ContourLabeler:

    def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=5):

        # Use linear interpolation to get end coordinates of label.
        target_cpls = np.array([-lw/2, lw/2])
        if is_closed_path:  # For closed paths, target from the other end.
            target_cpls[0] += (path_cpls[-1] - path_cpls[0])
        (sx0, sx1), (sy0, sy1) = interp_vec(target_cpls, path_cpls, screen_xys)
        angle = np.rad2deg(np.arctan2(sy1 - sy0, sx1 - sx0))  # Screen space.
        if self.rightside_up:  # Fix angle so text is never upside-down
            angle = (angle + 90) % 180 - 90

        target_cpls += [-spacing, +spacing]  # Expand range by spacing.

        # Get indices near points of interest; use -1 as out of bounds marker.
        i0, i1 = np.interp(target_cpls, path_cpls, range(len(path_cpls)),
                           left=-1, right=-1)
        i0 = math.floor(i0)
        i1 = math.ceil(i1)
        (x0, x1), (y0, y1) = interp_vec(target_cpls, path_cpls, cc_xys)

        # Actually break contours (dropping zero-len parts).
        new_xy_blocks = []
        new_code_blocks = []
        if is_closed_path:
            if i0 != -1 and i1 != -1:
                new_xy_blocks.extend([[(x1, y1)], cc_xys[i1:i0+1], [(x0, y0)]])
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (i0 + 2 - i1)])
        else:
            if i0 != -1:
                new_xy_blocks.extend([cc_xys[:i0 + 1], [(x0, y0)]])
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (i0 + 1)])
            if i1 != -1:
                new_xy_blocks.extend([[(x1, y1)], cc_xys[i1:]])
                new_code_blocks.extend([
                    [Path.MOVETO], [Path.LINETO] * (len(cc_xys) - i1)])

        # Back to the full path.
        xys = np.concatenate([xys[:start], *new_xy_blocks, xys[stop:]])
        codes = np.concatenate([codes[:start], *new_code_blocks, codes[stop:]])

        return angle, Path(xys, codes)
```
### 4 - lib/matplotlib/contour.py:

Start line: 497, End line: 527

```python
class ContourLabeler:

    @_api.deprecated("3.8")
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
            # if not len(nlc): nlc = [lc]

        return rotation, nlc
```
### 5 - galleries/examples/images_contours_and_fields/contour_label_demo.py:

Start line: 1, End line: 88

```python
"""
==================
Contour Label Demo
==================

Illustrate some of the more advanced things that one can do with
contour labels.

See also the :doc:`contour demo example
</gallery/images_contours_and_fields/contour_demo>`.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker

# %%
# Define our surface

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# %%
# Make contour labels with custom level formatters


# This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# then adds a percent sign.
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


# Basic contour plot
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)

ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

# %%
# Label contours with arbitrary strings using a dictionary

fig1, ax1 = plt.subplots()

# Basic contour plot
CS1 = ax1.contour(X, Y, Z)

fmt = {}
strs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s

# Label every other level using strings
ax1.clabel(CS1, CS1.levels[::2], inline=True, fmt=fmt, fontsize=10)

# %%
# Use a Formatter

fig2, ax2 = plt.subplots()

CS2 = ax2.contour(X, Y, 100**Z, locator=plt.LogLocator())
fmt = ticker.LogFormatterMathtext()
fmt.create_dummy_axis()
ax2.clabel(CS2, CS2.levels, fmt=fmt)
ax2.set_title("$100^Z$")

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.ticker.LogFormatterMathtext`
#    - `matplotlib.ticker.TickHelper.create_dummy_axis`
```
### 6 - lib/matplotlib/contour.py:

Start line: 284, End line: 316

```python
class ContourLabeler:

    def locate_label(self, linecontour, labelwidth):
        """
        Find good place to draw a label (relatively flat part of the contour).
        """
        ctr_size = len(linecontour)
        n_blocks = int(np.ceil(ctr_size / labelwidth)) if labelwidth > 1 else 1
        block_size = ctr_size if n_blocks == 1 else int(labelwidth)
        # Split contour into blocks of length ``block_size``, filling the last
        # block by cycling the contour start (per `np.resize` semantics).  (Due
        # to cycling, the index returned is taken modulo ctr_size.)
        xx = np.resize(linecontour[:, 0], (n_blocks, block_size))
        yy = np.resize(linecontour[:, 1], (n_blocks, block_size))
        yfirst = yy[:, :1]
        ylast = yy[:, -1:]
        xfirst = xx[:, :1]
        xlast = xx[:, -1:]
        s = (yfirst - yy) * (xlast - xfirst) - (xfirst - xx) * (ylast - yfirst)
        l = np.hypot(xlast - xfirst, ylast - yfirst)
        # Ignore warning that divide by zero throws, as this is a valid option
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = (abs(s) / l).sum(axis=-1)
        # Labels are drawn in the middle of the block (``hbsize``) where the
        # contour is the closest (per ``distances``) to a straight line, but
        # not `too_close()` to a preexisting label.
        hbsize = block_size // 2
        adist = np.argsort(distances)
        # If all candidates are `too_close()`, go back to the straightest part
        # (``adist[0]``).
        for idx in np.append(adist, adist[0]):
            x, y = xx[idx, hbsize], yy[idx, hbsize]
            if not self.too_close(x, y, labelwidth):
                break
        return x, y, (idx * block_size + hbsize) % ctr_size
```
### 7 - lib/matplotlib/contour.py:

Start line: 42, End line: 68

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
        if cs.axes.contains(event)[0]:
            cs.add_label_near(event.x, event.y, transform=False,
                              inline=inline, inline_spacing=inline_spacing)
            canvas.draw()
```
### 8 - lib/matplotlib/contour.py:

Start line: 596, End line: 647

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
            trans = self.get_transform()
            label_width = self._get_nth_label_width(idx)
            additions = []
            for subpath in self._paths[icon]._iter_connected_components():
                screen_xys = trans.transform(subpath.vertices)
                # Check if long enough for a label
                if self.print_label(screen_xys, label_width):
                    x, y, idx = self.locate_label(screen_xys, label_width)
                    rotation, path = self._split_path_and_get_label_rotation(
                        subpath, idx, (x, y),
                        label_width, inline_spacing)
                    add_label(x, y, rotation, lev, cvalue)  # Really add label.
                    if inline:  # If inline, add new contours
                        additions.append(path)
                else:  # If not adding label, keep old path
                    additions.append(subpath)
            # After looping over all segments on a contour, replace old path by new one
            # if inlining.
            if inline:
                self._paths[icon] = Path.make_compound_path(*additions)

    def remove(self):
        super().remove()
        for text in self.labelTexts:
            text.remove()


def _is_closed_polygon(X):
    """
    Return whether first and last object in a sequence are the same. These are
    presumably coordinates on a polygonal curve, in which case this function
    tests if that curve is closed.
    """
    return np.allclose(X[0], X[-1], rtol=1e-10, atol=1e-13)
```
### 9 - lib/matplotlib/contour.py:

Start line: 438, End line: 496

```python
class ContourLabeler:

    @_api.deprecated("3.8")
    def calc_label_rot_and_inline(self, slc, ind, lw, lc=None, spacing=5):
        """
        Calculate the appropriate label rotation given the linecontour
        coordinates in screen units, the index of the label location and the
        label width.

        If *lc* is not None or empty, also break contours and compute
        inlining.

        *spacing* is the empty space to leave around the label, in pixels.

        Both tasks are done together to avoid calculating path lengths
        multiple times, which is relatively costly.

        The method used here involves computing the path length along the
        contour in pixel coordinates and then looking approximately (label
        width / 2) away from central point to determine rotation and then to
        break contour if desired.
        """

        if lc is None:
            lc = []
        # Half the label width
        hlw = lw / 2.0

        # Check if closed and, if so, rotate contour so label is at edge
        closed = _is_closed_polygon(slc)
        if closed:
            slc = np.concatenate([slc[ind:-1], slc[:ind + 1]])
            if len(lc):  # Rotate lc also if not empty
                lc = np.concatenate([lc[ind:-1], lc[:ind + 1]])
            ind = 0

        # Calculate path lengths
        pl = np.zeros(slc.shape[0], dtype=float)
        dx = np.diff(slc, axis=0)
        pl[1:] = np.cumsum(np.hypot(dx[:, 0], dx[:, 1]))
        pl = pl - pl[ind]

        # Use linear interpolation to get points around label
        xi = np.array([-hlw, hlw])
        if closed:  # Look at end also for closed contours
            dp = np.array([pl[-1], 0])
        else:
            dp = np.zeros_like(xi)

        # Get angle of vector between the two ends of the label - must be
        # calculated in pixel space for text rotation to work correctly.
        (dx,), (dy,) = (np.diff(np.interp(dp + xi, pl, slc_col))
                        for slc_col in slc.T)
        rotation = np.rad2deg(np.arctan2(dy, dx))

        if self.rightside_up:
            # Fix angle so text is never upside-down
            rotation = (rotation + 90) % 180 - 90

        # Break contour if desired
        nlc = []
        # ... other code
```
### 10 - galleries/examples/text_labels_and_annotations/label_subplots.py:

Start line: 1, End line: 72

```python
"""
==================
Labelling subplots
==================

Labelling subplots is relatively straightforward, and varies,
so Matplotlib does not have a general method for doing this.

Simplest is putting the label inside the axes.  Note, here
we use `.pyplot.subplot_mosaic`, and use the subplot labels
as keys for the subplots, which is a nice convenience.  However,
the same method works with `.pyplot.subplots` or keys that are
different than what you want to label the subplot with.
"""

import matplotlib.pyplot as plt

import matplotlib.transforms as mtransforms

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')

for label, ax in axs.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

plt.show()

# %%
# We may prefer the labels outside the axes, but still aligned
# with each other, in which case we use a slightly different transform:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')

plt.show()

# %%
# If we want it aligned with the title, either incorporate in the title or
# use the *loc* keyword argument:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')

for label, ax in axs.items():
    ax.set_title('Normal Title', fontstyle='italic')
    ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.subplot_mosaic` /
#      `matplotlib.pyplot.subplot_mosaic`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.text`
#    - `matplotlib.transforms.ScaledTranslation`
```
### 11 - lib/matplotlib/contour.py:

Start line: 318, End line: 396

```python
class ContourLabeler:

    def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=5):
        """
        Prepare for insertion of a label at index *idx* of *path*.

        Parameters
        ----------
        path : Path
            The path where the label will be inserted, in data space.
        idx : int
            The vertex index after which the label will be inserted.
        screen_pos : (float, float)
            The position where the label will be inserted, in screen space.
        lw : float
            The label width, in screen space.
        spacing : float
            Extra spacing around the label, in screen space.

        Returns
        -------
        path : Path
            The path, broken so that the label can be drawn over it.
        angle : float
            The rotation of the label.

        Notes
        -----
        Both tasks are done together to avoid calculating path lengths multiple times,
        which is relatively costly.

        The method used here involves computing the path length along the contour in
        pixel coordinates and then looking (label width / 2) away from central point to
        determine rotation and then to break contour if desired.  The extra spacing is
        taken into account when breaking the path, but not when computing the angle.
        """
        if hasattr(self, "_old_style_split_collections"):
            del self._old_style_split_collections  # Invalidate them.

        xys = path.vertices
        codes = path.codes

        # Insert a vertex at idx/pos (converting back to data space), if there isn't yet
        # a vertex there.  With infinite precision one could also always insert the
        # extra vertex (it will get masked out by the label below anyways), but floating
        # point inaccuracies (the point can have undergone a data->screen->data
        # transform loop) can slightly shift the point and e.g. shift the angle computed
        # below from exactly zero to nonzero.
        pos = self.get_transform().inverted().transform(screen_pos)
        if not np.allclose(pos, xys[idx]):
            xys = np.insert(xys, idx, pos, axis=0)
            codes = np.insert(codes, idx, Path.LINETO)

        # Find the connected component where the label will be inserted.  Note that a
        # path always starts with a MOVETO, and we consider there's an implicit
        # MOVETO (closing the last path) at the end.
        movetos = (codes == Path.MOVETO).nonzero()[0]
        start = movetos[movetos < idx][-1]
        try:
            stop = movetos[movetos > idx][0]
        except IndexError:
            stop = len(codes)

        # Restrict ourselves to the connected component.
        cc_xys = xys[start:stop]
        idx -= start

        # If the path is closed, rotate it s.t. it starts at the label.
        is_closed_path = codes[stop - 1] == Path.CLOSEPOLY
        if is_closed_path:
            cc_xys = np.concatenate([xys[idx:-1], xys[:idx+1]])
            idx = 0

        # Like np.interp, but additionally vectorized over fp.
        def interp_vec(x, xp, fp): return [np.interp(x, xp, col) for col in fp.T]

        # Use cumulative path lengths ("cpl") as curvilinear coordinate along contour.
        screen_xys = self.get_transform().transform(cc_xys)
        path_cpls = np.insert(
            np.cumsum(np.hypot(*np.diff(screen_xys, axis=0).T)), 0, 0)
        path_cpls -= path_cpls[idx]
        # ... other code
```
### 12 - lib/matplotlib/contour.py:

Start line: 1130, End line: 1144

```python
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):

    def changed(self):
        if not hasattr(self, "cvalues"):
            self._process_colors()  # Sets cvalues.
        # Force an autoscale immediately because self.to_rgba() calls
        # autoscale_None() internally with the data passed to it,
        # so if vmin/vmax are not set yet, this would override them with
        # content from *cvalues* rather than levels like we want
        self.norm.autoscale_None(self.levels)
        self.set_array(self.cvalues)
        self.update_scalarmappable()
        alphas = np.broadcast_to(self.get_alpha(), len(self.cvalues))
        for label, cv, alpha in zip(self.labelTexts, self.labelCValues, alphas):
            label.set_alpha(alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        super().changed()
```
### 13 - lib/matplotlib/contour.py:

Start line: 1, End line: 39

```python
"""
Classes to support contour plotting and labelling for the Axes class.
"""

import functools
import math
from numbers import Integral

import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


@_api.deprecated("3.7", alternative="Text.set_transform_rotates_text")
class ClabelText(Text):
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
### 18 - lib/matplotlib/contour.py:

Start line: 71, End line: 158

```python
class ContourLabeler:
    """Mixin to provide labelling capability to `.ContourSet`."""

    def clabel(self, levels=None, *,
               fontsize=None, inline=True, inline_spacing=5, fmt=None,
               colors=None, use_clabeltext=False, manual=False,
               rightside_up=True, zorder=None):
        """
        Label a contour plot.

        Adds labels to line contours in this `.ContourSet` (which inherits from
        this mixin class).

        Parameters
        ----------
        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``cs.levels``. If not given, all levels are labeled.

        fontsize : str or float, default: :rc:`font.size`
            Size in points or relative size e.g., 'smaller', 'x-large'.
            See `.Text.set_size` for accepted string values.

        colors : color or colors or None, default: None
            The label colors:

            - If *None*, the color of each label matches the color of
              the corresponding contour.

            - If one string color, e.g., *colors* = 'r' or *colors* =
              'red', all labels will be plotted in this color.

            - If a tuple of colors (string, float, RGB, etc), different labels
              will be plotted in different colors in the order specified.

        inline : bool, default: True
            If ``True`` the underlying contour is removed where the label is
            placed.

        inline_spacing : float, default: 5
            Space in pixels to leave on each side of label when placing inline.

            This spacing will be exact for labels at locations where the
            contour is straight, less so for labels on curved contours.

        fmt : `.Formatter` or str or callable or dict, optional
            How the levels are formatted:

            - If a `.Formatter`, it is used to format all levels at once, using
              its `.Formatter.format_ticks` method.
            - If a str, it is interpreted as a %-style format string.
            - If a callable, it is called with one level at a time and should
              return the corresponding label.
            - If a dict, it should directly map levels to labels.

            The default is to use a standard `.ScalarFormatter`.

        manual : bool or iterable, default: False
            If ``True``, contour labels will be placed manually using
            mouse clicks. Click the first button near a contour to
            add a label, click the second button (or potentially both
            mouse buttons at once) to finish adding labels. The third
            button can be used to remove the last label added, but
            only if labels are not inline. Alternatively, the keyboard
            can be used to select label locations (enter to end label
            placement, delete or backspace act like the third mouse button,
            and any other key will select a label location).

            *manual* can also be an iterable object of (x, y) tuples.
            Contour labels will be created as if mouse is clicked at each
            (x, y) position.

        rightside_up : bool, default: True
            If ``True``, label rotations will always be plus
            or minus 90 degrees from level.

        use_clabeltext : bool, default: False
            If ``True``, use `.Text.set_transform_rotates_text` to ensure that
            label rotation is updated whenever the axes aspect changes.

        zorder : float or None, default: ``(2 + contour.get_zorder())``
            zorder of the contour labels.

        Returns
        -------
        labels
            A list of `.Text` instances for the labels.
        """
        # ... other code
```
### 22 - lib/matplotlib/contour.py:

Start line: 934, End line: 943

```python
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):

    allsegs = _api.deprecated("3.8", pending=True)(property(lambda self: [
        p.vertices for c in self.collections for p in c.get_paths()]))
    allkinds = _api.deprecated("3.8", pending=True)(property(lambda self: [
        p.codes for c in self.collections for p in c.get_paths()]))
    tcolors = _api.deprecated("3.8")(property(lambda self: [
        (tuple(rgba),) for rgba in self.to_rgba(self.cvalues, self.alpha)]))
    tlinewidths = _api.deprecated("3.8")(property(lambda self: [
        (w,) for w in self.get_linewidths()]))
    alpha = property(lambda self: self.get_alpha())
    linestyles = property(lambda self: self._orig_linestyles)
```
### 24 - lib/matplotlib/contour.py:

Start line: 529, End line: 545

```python
class ContourLabeler:

    def add_label(self, x, y, rotation, lev, cvalue):
        """Add contour label without `.Text.set_transform_rotates_text`."""
        data_x, data_y = self.axes.transData.inverted().transform((x, y))
        t = Text(
            data_x, data_y,
            text=self.get_text(lev, self.labelFmt),
            rotation=rotation,
            horizontalalignment='center', verticalalignment='center',
            zorder=self._clabel_zorder,
            color=self.labelMappable.to_rgba(cvalue, alpha=self.get_alpha()),
            fontproperties=self._label_font_props,
            clip_box=self.axes.bbox)
        self.labelTexts.append(t)
        self.labelCValues.append(cvalue)
        self.labelXYs.append((x, y))
        # Add label to plot here - useful for manual mode label selection
        self.axes.add_artist(t)
```
### 27 - lib/matplotlib/contour.py:

Start line: 945, End line: 977

```python
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):

    @_api.deprecated("3.8")
    @property
    def collections(self):
        # On access, make oneself invisible and instead add the old-style collections
        # (one PathCollection per level).  We do not try to further split contours into
        # connected components as we already lost track of what pairs of contours need
        # to be considered as single units to draw filled regions with holes.
        if not hasattr(self, "_old_style_split_collections"):
            self.set_visible(False)
            fcs = self.get_facecolor()
            ecs = self.get_edgecolor()
            lws = self.get_linewidth()
            lss = self.get_linestyle()
            self._old_style_split_collections = []
            for idx, path in enumerate(self._paths):
                pc = mcoll.PathCollection(
                    [path] if len(path.vertices) else [],
                    alpha=self.get_alpha(),
                    antialiaseds=self._antialiaseds[idx % len(self._antialiaseds)],
                    transform=self.get_transform(),
                    zorder=self.get_zorder(),
                    label="_nolegend_",
                    facecolor=fcs[idx] if len(fcs) else "none",
                    edgecolor=ecs[idx] if len(ecs) else "none",
                    linewidths=[lws[idx % len(lws)]],
                    linestyles=[lss[idx % len(lss)]],
                )
                if self.filled:
                    pc.set(hatch=self.hatches[idx % len(self.hatches)])
                self._old_style_split_collections.append(pc)
            for col in self._old_style_split_collections:
                self.axes.add_collection(col)
        return self._old_style_split_collections
```
### 33 - lib/matplotlib/contour.py:

Start line: 1308, End line: 1329

```python
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):

    def _process_linestyles(self, linestyles):
        Nlev = len(self.levels)
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
            if self.monochrome:
                eps = - (self.zmax - self.zmin) * 1e-15
                for i, lev in enumerate(self.levels):
                    if lev < eps:
                        tlinestyles[i] = self.negative_linestyles
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
```
### 38 - lib/matplotlib/contour.py:

Start line: 1295, End line: 1306

```python
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):

    def _process_linewidths(self, linewidths):
        Nlev = len(self.levels)
        if linewidths is None:
            default_linewidth = mpl.rcParams['contour.linewidth']
            if default_linewidth is None:
                default_linewidth = mpl.rcParams['lines.linewidth']
            return [default_linewidth] * Nlev
        elif not np.iterable(linewidths):
            return [linewidths] * Nlev
        else:
            linewidths = list(linewidths)
            return (linewidths * math.ceil(Nlev / len(linewidths)))[:Nlev]
```
### 72 - lib/matplotlib/contour.py:

Start line: 1461, End line: 1475

```python
@_docstring.dedent_interpd
class ContourSet(ContourLabeler, mcoll.Collection):

    def draw(self, renderer):
        paths = self._paths
        n_paths = len(paths)
        if not self.filled or all(hatch is None for hatch in self.hatches):
            super().draw(renderer)
            return
        # In presence of hatching, draw contours one at a time.
        for idx in range(n_paths):
            with cbook._setattr_cm(self, _paths=[paths[idx]]), self._cm_set(
                hatch=self.hatches[idx % len(self.hatches)],
                array=[self.get_array()[idx]],
                linewidths=[self.get_linewidths()[idx % len(self.get_linewidths())]],
                linestyles=[self.get_linestyles()[idx % len(self.get_linestyles())]],
            ):
                super().draw(renderer)
```
