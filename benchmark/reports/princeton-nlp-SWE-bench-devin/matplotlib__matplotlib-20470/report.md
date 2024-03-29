# matplotlib__matplotlib-20470

| **matplotlib/matplotlib** | `f0632c0fc7339f68e992ed63ae4cfac76cd41aad` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 14382 |
| **Any found context length** | 10623 |
| **Avg pos** | 113.5 |
| **Min pos** | 25 |
| **Max pos** | 93 |
| **Top file pos** | 3 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -38,6 +38,7 @@
 from matplotlib.collections import (
     Collection, CircleCollection, LineCollection, PathCollection,
     PolyCollection, RegularPolyCollection)
+from matplotlib.text import Text
 from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
 from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
 from matplotlib.offsetbox import (
@@ -740,11 +741,12 @@ def _init_legend_box(self, handles, labels, markerfirst=True):
             handler = self.get_legend_handler(legend_handler_map, orig_handle)
             if handler is None:
                 _api.warn_external(
-                    "Legend does not support {!r} instances.\nA proxy artist "
-                    "may be used instead.\nSee: "
-                    "https://matplotlib.org/users/legend_guide.html"
-                    "#creating-artists-specifically-for-adding-to-the-legend-"
-                    "aka-proxy-artists".format(orig_handle))
+                             "Legend does not support handles for {0} "
+                             "instances.\nA proxy artist may be used "
+                             "instead.\nSee: https://matplotlib.org/"
+                             "stable/tutorials/intermediate/legend_guide.html"
+                             "#controlling-the-legend-entries".format(
+                                 type(orig_handle).__name__))
                 # No handle for this artist, so we just defer to None.
                 handle_list.append(None)
             else:
@@ -1074,14 +1076,14 @@ def _get_legend_handles(axs, legend_handler_map=None):
     for ax in axs:
         handles_original += [
             *(a for a in ax._children
-              if isinstance(a, (Line2D, Patch, Collection))),
+              if isinstance(a, (Line2D, Patch, Collection, Text))),
             *ax.containers]
         # support parasite axes:
         if hasattr(ax, 'parasites'):
             for axx in ax.parasites:
                 handles_original += [
                     *(a for a in axx._children
-                      if isinstance(a, (Line2D, Patch, Collection))),
+                      if isinstance(a, (Line2D, Patch, Collection, Text))),
                     *axx.containers]
 
     handler_map = {**Legend.get_default_handler_map(),
@@ -1091,6 +1093,15 @@ def _get_legend_handles(axs, legend_handler_map=None):
         label = handle.get_label()
         if label != '_nolegend_' and has_handler(handler_map, handle):
             yield handle
+        elif (label not in ['_nolegend_', ''] and
+                not has_handler(handler_map, handle)):
+            _api.warn_external(
+                             "Legend does not support handles for {0} "
+                             "instances.\nSee: https://matplotlib.org/stable/"
+                             "tutorials/intermediate/legend_guide.html"
+                             "#implementing-a-custom-legend-handler".format(
+                                 type(handle).__name__))
+            continue
 
 
 def _get_legend_handles_labels(axs, legend_handler_map=None):
diff --git a/lib/matplotlib/text.py b/lib/matplotlib/text.py
--- a/lib/matplotlib/text.py
+++ b/lib/matplotlib/text.py
@@ -132,6 +132,9 @@ def __init__(self,
         """
         Create a `.Text` instance at *x*, *y* with string *text*.
 
+        While Text accepts the 'label' keyword argument, by default it is not
+        added to the handles of a legend.
+
         Valid keyword arguments are:
 
         %(Text:kwdoc)s

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/legend.py | 41 | 41 | 54 | 3 | 26343
| lib/matplotlib/legend.py | 743 | 747 | 93 | 3 | 41534
| lib/matplotlib/legend.py | 1077 | 1084 | 25 | 3 | 10623
| lib/matplotlib/legend.py | 1094 | 1094 | 25 | 3 | 10623
| lib/matplotlib/text.py | 135 | 135 | 30 | 12 | 14382


## Problem Statement

```
Handle and label not created for Text with label
### Bug report

**Bug summary**

Text accepts a `label` keyword argument but neither its handle nor its label is created and added to the legend.

**Code for reproduction**

\`\`\`python
import matplotlib.pyplot as plt

x = [0, 10]
y = [0, 10]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x, y, label="line")
ax.text(x=2, y=5, s="text", label="label")

ax.legend()

plt.show()
\`\`\`

**Actual outcome**

![t](https://user-images.githubusercontent.com/9297904/102268707-a4e97f00-3ee9-11eb-9bd9-cca098f69c29.png)

**Expected outcome**

I expect a legend entry for the text.

**Matplotlib version**
  * Matplotlib version: 3.3.3


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 lib/matplotlib/contour.py | 465 | 482| 181 | 181 | 16669 | 
| 2 | 2 examples/text_labels_and_annotations/legend_demo.py | 76 | 116| 567 | 748 | 18576 | 
| 3 | **3 lib/matplotlib/legend.py** | 787 | 802| 175 | 923 | 28951 | 
| 4 | 4 examples/text_labels_and_annotations/line_with_text.py | 51 | 87| 260 | 1183 | 29543 | 
| 5 | 5 examples/text_labels_and_annotations/label_subplots.py | 1 | 71| 603 | 1786 | 30146 | 
| 6 | 5 lib/matplotlib/contour.py | 443 | 463| 235 | 2021 | 30146 | 
| 7 | 5 lib/matplotlib/contour.py | 484 | 493| 130 | 2151 | 30146 | 
| 8 | 6 lib/matplotlib/legend_handler.py | 540 | 622| 756 | 2907 | 36684 | 
| 9 | 6 lib/matplotlib/legend_handler.py | 736 | 767| 258 | 3165 | 36684 | 
| 10 | 6 examples/text_labels_and_annotations/legend_demo.py | 1 | 75| 738 | 3903 | 36684 | 
| 11 | 6 examples/text_labels_and_annotations/legend_demo.py | 164 | 183| 198 | 4101 | 36684 | 
| 12 | 7 tutorials/intermediate/legend_guide.py | 196 | 282| 812 | 4913 | 39341 | 
| 13 | **7 lib/matplotlib/legend.py** | 1108 | 1204| 715 | 5628 | 39341 | 
| 14 | 7 tutorials/intermediate/legend_guide.py | 122 | 194| 768 | 6396 | 39341 | 
| 15 | 8 examples/text_labels_and_annotations/legend.py | 1 | 40| 233 | 6629 | 39574 | 
| 16 | 8 lib/matplotlib/legend_handler.py | 409 | 420| 129 | 6758 | 39574 | 
| 17 | 9 lib/matplotlib/pyplot.py | 2927 | 2963| 401 | 7159 | 66019 | 
| 18 | 9 lib/matplotlib/legend_handler.py | 248 | 263| 144 | 7303 | 66019 | 
| 19 | **9 lib/matplotlib/legend.py** | 470 | 561| 1270 | 8573 | 66019 | 
| 20 | 9 lib/matplotlib/contour.py | 167 | 238| 712 | 9285 | 66019 | 
| 21 | 9 lib/matplotlib/legend_handler.py | 659 | 712| 498 | 9783 | 66019 | 
| 22 | 10 lib/matplotlib/figure.py | 1053 | 1075| 242 | 10025 | 92286 | 
| 23 | 11 lib/mpl_toolkits/axisartist/axis_artist.py | 245 | 263| 161 | 10186 | 100344 | 
| 24 | 11 lib/matplotlib/legend_handler.py | 378 | 388| 137 | 10323 | 100344 | 
| **-> 25 <-** | **11 lib/matplotlib/legend.py** | 1069 | 1105| 300 | 10623 | 100344 | 
| 26 | 11 lib/mpl_toolkits/axisartist/axis_artist.py | 285 | 352| 470 | 11093 | 100344 | 
| 27 | **11 lib/matplotlib/legend.py** | 285 | 468| 1668 | 12761 | 100344 | 
| 28 | 11 tutorials/intermediate/legend_guide.py | 1 | 120| 883 | 13644 | 100344 | 
| 29 | 11 lib/matplotlib/legend_handler.py | 331 | 348| 199 | 13843 | 100344 | 
| **-> 30 <-** | **12 lib/matplotlib/text.py** | 91 | 160| 539 | 14382 | 115578 | 
| 31 | 13 examples/text_labels_and_annotations/figlegend_demo.py | 1 | 31| 254 | 14636 | 115832 | 
| 32 | **13 lib/matplotlib/legend.py** | 877 | 913| 240 | 14876 | 115832 | 
| 33 | 13 lib/matplotlib/figure.py | 942 | 1051| 815 | 15691 | 115832 | 
| 34 | 13 lib/matplotlib/legend_handler.py | 211 | 245| 323 | 16014 | 115832 | 
| 35 | 14 examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 303 | 16317 | 116135 | 
| 36 | 14 examples/text_labels_and_annotations/legend_demo.py | 118 | 162| 403 | 16720 | 116135 | 
| 37 | 15 lib/matplotlib/axis.py | 185 | 200| 134 | 16854 | 136351 | 
| 38 | 16 tutorials/text/text_intro.py | 85 | 171| 776 | 17630 | 140229 | 
| 39 | 16 lib/matplotlib/contour.py | 495 | 559| 667 | 18297 | 140229 | 
| 40 | 16 lib/matplotlib/axis.py | 1670 | 1747| 708 | 19005 | 140229 | 
| 41 | 17 examples/text_labels_and_annotations/demo_text_path.py | 47 | 134| 775 | 19780 | 141327 | 
| 42 | 18 examples/text_labels_and_annotations/mathtext_examples.py | 59 | 102| 434 | 20214 | 142609 | 
| 43 | 18 lib/matplotlib/legend_handler.py | 1 | 44| 317 | 20531 | 142609 | 
| 44 | 19 examples/text_labels_and_annotations/angle_annotation.py | 270 | 326| 707 | 21238 | 146049 | 
| 45 | 20 examples/lines_bars_and_markers/bar_label_demo.py | 1 | 107| 820 | 22058 | 146869 | 
| 46 | 20 lib/mpl_toolkits/axisartist/axis_artist.py | 400 | 583| 1482 | 23540 | 146869 | 
| 47 | 20 tutorials/text/text_intro.py | 173 | 261| 771 | 24311 | 146869 | 
| 48 | 21 examples/userdemo/simple_legend01.py | 1 | 27| 199 | 24510 | 147068 | 
| 49 | 21 lib/matplotlib/contour.py | 47 | 73| 309 | 24819 | 147068 | 
| 50 | 21 lib/matplotlib/pyplot.py | 2655 | 2670| 169 | 24988 | 147068 | 
| 51 | 21 examples/text_labels_and_annotations/line_with_text.py | 1 | 49| 332 | 25320 | 147068 | 
| 52 | 21 lib/matplotlib/legend_handler.py | 93 | 136| 357 | 25677 | 147068 | 
| 53 | 21 lib/mpl_toolkits/axisartist/axis_artist.py | 202 | 243| 280 | 25957 | 147068 | 
| **-> 54 <-** | **21 lib/matplotlib/legend.py** | 1 | 49| 386 | 26343 | 147068 | 
| 55 | 22 examples/pyplots/pyplot_text.py | 1 | 42| 242 | 26585 | 147310 | 
| 56 | 22 lib/matplotlib/legend_handler.py | 266 | 302| 326 | 26911 | 147310 | 
| 57 | 23 examples/text_labels_and_annotations/accented_text.py | 1 | 36| 376 | 27287 | 147686 | 
| 58 | 23 lib/matplotlib/contour.py | 240 | 249| 147 | 27434 | 147686 | 
| 59 | 23 lib/matplotlib/legend_handler.py | 518 | 538| 177 | 27611 | 147686 | 
| 60 | 24 examples/text_labels_and_annotations/multiline.py | 1 | 46| 386 | 27997 | 148072 | 
| 61 | **24 lib/matplotlib/text.py** | 671 | 739| 630 | 28627 | 148072 | 
| 62 | 24 tutorials/text/text_intro.py | 262 | 329| 833 | 29460 | 148072 | 
| 63 | 25 examples/subplots_axes_and_figures/figure_title.py | 1 | 65| 544 | 30004 | 148616 | 
| 64 | 26 examples/showcase/anatomy.py | 90 | 162| 554 | 30558 | 149930 | 
| 65 | 26 lib/matplotlib/pyplot.py | 2445 | 2456| 146 | 30704 | 149930 | 
| 66 | 27 examples/text_labels_and_annotations/annotation_demo.py | 365 | 389| 264 | 30968 | 153899 | 
| 67 | 27 lib/matplotlib/figure.py | 421 | 429| 137 | 31105 | 153899 | 
| 68 | 27 lib/mpl_toolkits/axisartist/axis_artist.py | 949 | 963| 141 | 31246 | 153899 | 
| 69 | 27 lib/matplotlib/pyplot.py | 2404 | 2417| 158 | 31404 | 153899 | 
| 70 | 27 lib/mpl_toolkits/axisartist/axis_artist.py | 965 | 992| 222 | 31626 | 153899 | 
| 71 | 27 lib/matplotlib/figure.py | 412 | 419| 129 | 31755 | 153899 | 
| 72 | 28 examples/userdemo/simple_legend02.py | 1 | 24| 136 | 31891 | 154035 | 
| 73 | 28 tutorials/text/text_intro.py | 1 | 83| 733 | 32624 | 154035 | 
| 74 | 28 lib/matplotlib/axis.py | 1171 | 1201| 307 | 32931 | 154035 | 
| 75 | 29 examples/text_labels_and_annotations/demo_text_rotation_mode.py | 79 | 94| 102 | 33033 | 154886 | 
| 76 | 29 lib/matplotlib/pyplot.py | 2256 | 2331| 767 | 33800 | 154886 | 
| 77 | 30 examples/text_labels_and_annotations/date.py | 1 | 64| 658 | 34458 | 155544 | 
| 78 | **30 lib/matplotlib/text.py** | 372 | 445| 731 | 35189 | 155544 | 
| 79 | 31 tutorials/intermediate/artists.py | 616 | 727| 1004 | 36193 | 163248 | 
| 80 | **31 lib/matplotlib/legend.py** | 605 | 634| 238 | 36431 | 163248 | 
| 81 | 32 lib/matplotlib/artist.py | 1072 | 1109| 215 | 36646 | 176124 | 
| 82 | **32 lib/matplotlib/text.py** | 1792 | 1829| 388 | 37034 | 176124 | 
| 83 | 33 examples/mplot3d/text3d.py | 1 | 48| 433 | 37467 | 176557 | 
| 84 | 33 lib/matplotlib/contour.py | 561 | 620| 475 | 37942 | 176557 | 
| 85 | 33 lib/mpl_toolkits/axisartist/axis_artist.py | 265 | 282| 163 | 38105 | 176557 | 
| 86 | 34 examples/text_labels_and_annotations/engineering_formatter.py | 1 | 45| 367 | 38472 | 176924 | 
| 87 | 34 lib/matplotlib/figure.py | 403 | 410| 132 | 38604 | 176924 | 
| 88 | 34 lib/matplotlib/legend_handler.py | 476 | 494| 178 | 38782 | 176924 | 
| 89 | 35 lib/mpl_toolkits/mplot3d/axis3d.py | 265 | 343| 826 | 39608 | 182126 | 
| 90 | 35 lib/matplotlib/contour.py | 1 | 44| 289 | 39897 | 182126 | 
| 91 | 36 examples/text_labels_and_annotations/text_rotation.py | 24 | 35| 212 | 40109 | 182610 | 
| 92 | 37 examples/images_contours_and_fields/contour_label_demo.py | 1 | 87| 613 | 40722 | 183223 | 
| **-> 93 <-** | **37 lib/matplotlib/legend.py** | 705 | 785| 812 | 41534 | 183223 | 
| 94 | 37 lib/matplotlib/legend_handler.py | 391 | 407| 134 | 41668 | 183223 | 
| 95 | 38 examples/pyplots/align_ylabels.py | 1 | 37| 288 | 41956 | 183857 | 
| 96 | 38 lib/mpl_toolkits/axisartist/axis_artist.py | 994 | 1030| 271 | 42227 | 183857 | 
| 97 | 38 lib/matplotlib/contour.py | 76 | 165| 821 | 43048 | 183857 | 
| 98 | 38 examples/text_labels_and_annotations/angle_annotation.py | 254 | 267| 162 | 43210 | 183857 | 
| 99 | 39 tutorials/introductory/lifecycle.py | 105 | 198| 827 | 44037 | 186158 | 
| 100 | 40 examples/text_labels_and_annotations/fancytextbox_demo.py | 1 | 27| 184 | 44221 | 186342 | 
| 101 | 40 tutorials/intermediate/legend_guide.py | 285 | 304| 193 | 44414 | 186342 | 
| 102 | **40 lib/matplotlib/text.py** | 1968 | 1985| 187 | 44601 | 186342 | 
| 103 | 40 lib/matplotlib/legend_handler.py | 166 | 178| 158 | 44759 | 186342 | 
| 104 | 40 lib/matplotlib/contour.py | 299 | 317| 164 | 44923 | 186342 | 
| 105 | 40 lib/matplotlib/legend_handler.py | 47 | 91| 383 | 45306 | 186342 | 
| 106 | 41 lib/matplotlib/axes/_subplots.py | 131 | 145| 182 | 45488 | 187980 | 
| 107 | 41 tutorials/text/text_intro.py | 330 | 423| 764 | 46252 | 187980 | 
| 108 | 42 examples/pyplots/fig_axes_labels_simple.py | 1 | 43| 289 | 46541 | 188269 | 
| 109 | 43 examples/text_labels_and_annotations/mathtext_demo.py | 1 | 27| 219 | 46760 | 188488 | 
| 110 | 44 examples/text_labels_and_annotations/arrow_demo.py | 141 | 160| 238 | 46998 | 190047 | 
| 111 | 45 examples/text_labels_and_annotations/titles_demo.py | 1 | 60| 367 | 47365 | 190414 | 
| 112 | 46 examples/misc/logos2.py | 91 | 106| 154 | 47519 | 191742 | 


### Hint

```
This is an imprecision in the API. Technically, every `Artist` can have a label. But note every `Artist` has a legend handler (which creates the handle to show in the legend, see also https://matplotlib.org/3.3.3/api/legend_handler_api.html#module-matplotlib.legend_handler).

In particular `Text` does not have a legend handler. Also I wouldn't know what should be displayed there - what would you have expected for the text?

I'd tent to say that `Text` just cannot appear in legends and it's an imprecision that it accepts a `label` keyword argument. Maybe we should warn on that, OTOH you *could* write your own legend handler for `Text`, in which case that warning would be a bit annoying.
People can also query an artists label if they want to keep track of it somehow, so labels are not something we should just automatically assume labels are just for legends.
> Technically, every Artist can have a label. But note every Artist has a legend handler

What's confusing is that a `Patch` without a legend handler still appears, as a `Rectangle`, in the legend. I expected a legend entry for the `Text`, not blank output.

> In particular Text does not have a legend handler. Also I wouldn't know what should be displayed there - what would you have expected for the text?

In the non-MWE code I use alphabet letters as "markers". So I expected "A    \<label text\>" to appear in the legend.

> Maybe we should warn on that, OTOH you could write your own legend handler for Text

This is what I did as a workaround.
[Artist.set_label](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.artist.Artist.set_label.html) explicitly specifies 

> Set a label that will be displayed in the legend.

So while you could use it for something else, IMHO it's not in the intended scope and we would not have to care for that.

But thinking about it a bit more: In the current design, Artists don't know if handlers exist for them, so they cannot reasonably warn about that. There's a bit more issues underneath the surface. Overall, while it's a bit annoying as is, we cannot make this better without internal and possibly public API changes.

> In the non-MWE code I use alphabet letters as "markers". So I expected "A <label text>" to appear in the legend.

I see. Given that this only makes sense for special usecases where texts are one or a few characters, I don't think that we can add a reasonable general legend handler for `Text`s. You solution to write your own handler seems the right one for this kind of problem. I'll therefore close the issue (please report back if you think that there should be some better solution, and you have an idea how that can reasonably work for arbitrary texts). Anyway, thanks for opening the issue. 
BTW you can use arbitrary latex strings as markers with `plt.scatter`, something like

\`\`\`python
plt.scatter(.5, .9, marker="$a$", label="the letter a")
plt.legend()
\`\`\`

might give what you want.
> Artists don't know if handlers exist for them, so they cannot reasonably warn about that. There's a bit more issues underneath the surface. Overall, while it's a bit annoying as is, we cannot make this better without internal and possibly public API changes.

We could warn when collecting all artists that have handlers (in `_get_legend_handles`) if `has_handler` returns False.  I make no judgment as to whether we want to do that, though.
> We could warn when collecting all artists that have handlers (in `_get_legend_handles`) if `has_handler` returns False.  I make no judgment as to whether we want to do that, though.

Seems cleaner to me. It may be considered an error if a label is set, but that Artist cannot occur in a legend.

This requires looping through all artists instead of https://github.com/matplotlib/matplotlib/blob/93649f830c4ae428701d4f02ecd64d19da1d5a06/lib/matplotlib/legend.py#L1117.
> (please report back if you think that there should be some better solution, and you have an idea how that can reasonably work for arbitrary texts)

This is my custom class:

\`\`\`python
class HandlerText:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        handle_text = Text(x=x0, y=y0, text=orig_handle.get_text())
        handlebox.add_artist(handle_text)
        return handle_text
\`\`\`

Seems to me that it should work for arbitrary text. Here's a [gist](https://gist.github.com/kdpenner/a16d249ae24ed6496e6f5915e4540b4b). Note that I have to add the `Text` handle and label manually.

> We could warn when collecting all artists that have handlers (in _get_legend_handles) if has_handler returns False.

Yes x 1000. It's bewildering to create an `Ellipse` and get a `Rectangle` in the legend. Before I descended into the legend mine, my first thought was not "The `Ellipse` must not have a handler", it was "matplotlib must have a bug". And then it's bewildering again to have `Text` dropped from the legend instead of having a placeholder like in the patches case.
Also I'm willing to work on addition of the warning and/or handler after y'all decide what's best to do.
> Also I'm willing to work on addition of the warning and/or handler.

Great! :+1: 

Please give me a bit of time to think about what exactly should be done.
I'm revisiting my backlog of issues...any more thoughts?
### No default legend handler for Text.
The proposed implmentation is rather tautological, that's something only reasonable in very special cases. I also don't see any other good way to visualize a legend entry for a text.

#### Recommendation: Users should implement their own handler if needed.

### Warn on legend entries without handler.
> We could warn when collecting all artists that have handlers (in _get_legend_handles) if has_handler returns False.

What are the use cases and how would a warning affect them?

1) When using `legend(handles=artists)`, it would be awkward if something in `artists` is silently not rendered because it has no handler. --> warning is reasonable
2) When using `plt.text(..., label='a'), plt.legend()` it deprends:
   a) is the label setting done with the intention of legend? --> then reasonable
   b) is the label only used as a generic identifier? --> then a warning would be unhelpful.

Overall, since the label parameter is bound to taking part in legends, we can dismiss scenario 2b). (With the logic of 2b we'd also get undesired entries in the legend for artists that have a handler.

#### Recommendation: Implement the warning.


```

## Patch

```diff
diff --git a/lib/matplotlib/legend.py b/lib/matplotlib/legend.py
--- a/lib/matplotlib/legend.py
+++ b/lib/matplotlib/legend.py
@@ -38,6 +38,7 @@
 from matplotlib.collections import (
     Collection, CircleCollection, LineCollection, PathCollection,
     PolyCollection, RegularPolyCollection)
+from matplotlib.text import Text
 from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
 from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
 from matplotlib.offsetbox import (
@@ -740,11 +741,12 @@ def _init_legend_box(self, handles, labels, markerfirst=True):
             handler = self.get_legend_handler(legend_handler_map, orig_handle)
             if handler is None:
                 _api.warn_external(
-                    "Legend does not support {!r} instances.\nA proxy artist "
-                    "may be used instead.\nSee: "
-                    "https://matplotlib.org/users/legend_guide.html"
-                    "#creating-artists-specifically-for-adding-to-the-legend-"
-                    "aka-proxy-artists".format(orig_handle))
+                             "Legend does not support handles for {0} "
+                             "instances.\nA proxy artist may be used "
+                             "instead.\nSee: https://matplotlib.org/"
+                             "stable/tutorials/intermediate/legend_guide.html"
+                             "#controlling-the-legend-entries".format(
+                                 type(orig_handle).__name__))
                 # No handle for this artist, so we just defer to None.
                 handle_list.append(None)
             else:
@@ -1074,14 +1076,14 @@ def _get_legend_handles(axs, legend_handler_map=None):
     for ax in axs:
         handles_original += [
             *(a for a in ax._children
-              if isinstance(a, (Line2D, Patch, Collection))),
+              if isinstance(a, (Line2D, Patch, Collection, Text))),
             *ax.containers]
         # support parasite axes:
         if hasattr(ax, 'parasites'):
             for axx in ax.parasites:
                 handles_original += [
                     *(a for a in axx._children
-                      if isinstance(a, (Line2D, Patch, Collection))),
+                      if isinstance(a, (Line2D, Patch, Collection, Text))),
                     *axx.containers]
 
     handler_map = {**Legend.get_default_handler_map(),
@@ -1091,6 +1093,15 @@ def _get_legend_handles(axs, legend_handler_map=None):
         label = handle.get_label()
         if label != '_nolegend_' and has_handler(handler_map, handle):
             yield handle
+        elif (label not in ['_nolegend_', ''] and
+                not has_handler(handler_map, handle)):
+            _api.warn_external(
+                             "Legend does not support handles for {0} "
+                             "instances.\nSee: https://matplotlib.org/stable/"
+                             "tutorials/intermediate/legend_guide.html"
+                             "#implementing-a-custom-legend-handler".format(
+                                 type(handle).__name__))
+            continue
 
 
 def _get_legend_handles_labels(axs, legend_handler_map=None):
diff --git a/lib/matplotlib/text.py b/lib/matplotlib/text.py
--- a/lib/matplotlib/text.py
+++ b/lib/matplotlib/text.py
@@ -132,6 +132,9 @@ def __init__(self,
         """
         Create a `.Text` instance at *x*, *y* with string *text*.
 
+        While Text accepts the 'label' keyword argument, by default it is not
+        added to the handles of a legend.
+
         Valid keyword arguments are:
 
         %(Text:kwdoc)s

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_legend.py b/lib/matplotlib/tests/test_legend.py
--- a/lib/matplotlib/tests/test_legend.py
+++ b/lib/matplotlib/tests/test_legend.py
@@ -493,6 +493,15 @@ def test_handler_numpoints():
     ax.legend(numpoints=0.5)
 
 
+def test_text_nohandler_warning():
+    """Test that Text artists with labels raise a warning"""
+    fig, ax = plt.subplots()
+    ax.text(x=0, y=0, s="text", label="label")
+    with pytest.warns(UserWarning) as record:
+        ax.legend()
+    assert len(record) == 1
+
+
 def test_empty_bar_chart_with_legend():
     """Test legend when bar chart is empty with a label."""
     # related to issue #13003. Calling plt.legend() should not

```


## Code snippets

### 1 - lib/matplotlib/contour.py:

Start line: 465, End line: 482

```python
class ContourLabeler:

    def _add_label(self, t, x, y, lev, cvalue):
        color = self.labelMappable.to_rgba(cvalue, alpha=self.alpha)

        _text = self.get_text(lev, self.labelFmt)
        self.set_label_props(t, _text, color)
        self.labelTexts.append(t)
        self.labelCValues.append(cvalue)
        self.labelXYs.append((x, y))

        # Add label to plot here - useful for manual mode label selection
        self.axes.add_artist(t)

    def add_label(self, x, y, rotation, lev, cvalue):
        """
        Add contour label using :class:`~matplotlib.text.Text` class.
        """
        t = self._get_label_text(x, y, rotation)
        self._add_label(t, x, y, lev, cvalue)
```
### 2 - examples/text_labels_and_annotations/legend_demo.py:

Start line: 76, End line: 116

```python
middle_ax.errorbar([0, 1, 2], [3, 2, 4], yerr=0.3, fmt="o", label="test 2")
middle_ax.errorbar([0, 1, 2], [1, 1, 3], xerr=0.4, yerr=0.3, fmt="^",
                   label="test 3")
middle_ax.legend()

bottom_ax.stem([0.3, 1.5, 2.7], [1, 3.6, 2.7], label="stem test")
bottom_ax.legend()

plt.show()

###############################################################################
# Now we'll showcase legend entries with more than one legend key.

fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

# First plot: two legend keys for a single entry
p1 = ax1.scatter([1], [5], c='r', marker='s', s=100)
p2 = ax1.scatter([3], [2], c='b', marker='o', s=100)
# `plot` returns a list, but we want the handle - thus the comma on the left
p3, = ax1.plot([1, 5], [4, 4], 'm-d')

# Assign two of the handles to the same legend entry by putting them in a tuple
# and using a generic handler map (which would be used for any additional
# tuples of handles like (p1, p3)).
l = ax1.legend([(p1, p3), p2], ['two keys', 'one key'], scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

# Second plot: plot two bar charts on top of each other and change the padding
# between the legend keys
x_left = [1, 2, 3]
y_pos = [1, 3, 2]
y_neg = [2, 1, 4]

rneg = ax2.bar(x_left, y_neg, width=0.5, color='w', hatch='///', label='-1')
rpos = ax2.bar(x_left, y_pos, width=0.5, color='k', label='+1')

# Treat each legend entry differently by using specific `HandlerTuple`s
l = ax2.legend([(rpos, rneg), (rneg, rpos)], ['pad!=0', 'pad=0'],
               handler_map={(rpos, rneg): HandlerTuple(ndivide=None),
                            (rneg, rpos): HandlerTuple(ndivide=None, pad=0.)})
plt.show()
```
### 3 - lib/matplotlib/legend.py:

Start line: 787, End line: 802

```python
class Legend(Artist):

    def _init_legend_box(self, handles, labels, markerfirst=True):
        # ... other code

        mode = "expand" if self._mode == "expand" else "fixed"
        sep = self.columnspacing * fontsize
        self._legend_handle_box = HPacker(pad=0,
                                          sep=sep, align="baseline",
                                          mode=mode,
                                          children=columnbox)
        self._legend_title_box = TextArea("")
        self._legend_box = VPacker(pad=self.borderpad * fontsize,
                                   sep=self.labelspacing * fontsize,
                                   align="center",
                                   children=[self._legend_title_box,
                                             self._legend_handle_box])
        self._legend_box.set_figure(self.figure)
        self._legend_box.axes = self.axes
        self.texts = text_list
        self.legendHandles = handle_list
```
### 4 - examples/text_labels_and_annotations/line_with_text.py:

Start line: 51, End line: 87

```python
# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()
x, y = np.random.rand(2, 20)
line = MyLine(x, y, mfc='red', ms=12, label='line label')
line.text.set_color('red')
line.text.set_fontsize(16)

ax.add_line(line)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.lines`
#    - `matplotlib.lines.Line2D`
#    - `matplotlib.lines.Line2D.set_data`
#    - `matplotlib.artist`
#    - `matplotlib.artist.Artist`
#    - `matplotlib.artist.Artist.draw`
#    - `matplotlib.artist.Artist.set_transform`
#    - `matplotlib.text`
#    - `matplotlib.text.Text`
#    - `matplotlib.text.Text.set_color`
#    - `matplotlib.text.Text.set_fontsize`
#    - `matplotlib.text.Text.set_position`
#    - `matplotlib.axes.Axes.add_line`
#    - `matplotlib.transforms`
#    - `matplotlib.transforms.Affine2D`
```
### 5 - examples/text_labels_and_annotations/label_subplots.py:

Start line: 1, End line: 71

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
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

plt.show()

##############################################################################
# We may prefer the labels outside the axes, but still aligned
# with each other, in which case we use a slightly different transform:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')

plt.show()

##############################################################################
# If we want it aligned with the title, either incorporate in the title or
# use the *loc* keyword argument:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              constrained_layout=True)

for label, ax in axs.items():
    ax.set_title('Normal Title', fontstyle='italic')
    ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

plt.show()

#############################################################################
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
### 6 - lib/matplotlib/contour.py:

Start line: 443, End line: 463

```python
class ContourLabeler:

    def _get_label_text(self, x, y, rotation):
        dx, dy = self.axes.transData.inverted().transform((x, y))
        t = text.Text(dx, dy, rotation=rotation,
                      horizontalalignment='center',
                      verticalalignment='center', zorder=self._clabel_zorder)
        return t

    def _get_label_clabeltext(self, x, y, rotation):
        # x, y, rotation is given in pixel coordinate. Convert them to
        # the data coordinate and create a label using ClabelText
        # class. This way, the rotation of the clabel is along the
        # contour line always.
        transDataInv = self.axes.transData.inverted()
        dx, dy = transDataInv.transform((x, y))
        drotation = transDataInv.transform_angles(np.array([rotation]),
                                                  np.array([[x, y]]))
        t = ClabelText(dx, dy, rotation=drotation[0],
                       horizontalalignment='center',
                       verticalalignment='center', zorder=self._clabel_zorder)

        return t
```
### 7 - lib/matplotlib/contour.py:

Start line: 484, End line: 493

```python
class ContourLabeler:

    def add_label_clabeltext(self, x, y, rotation, lev, cvalue):
        """
        Add contour label using :class:`ClabelText` class.
        """
        # x, y, rotation is given in pixel coordinate. Convert them to
        # the data coordinate and create a label using ClabelText
        # class. This way, the rotation of the clabel is along the
        # contour line always.
        t = self._get_label_clabeltext(x, y, rotation)
        self._add_label(t, x, y, lev, cvalue)
```
### 8 - lib/matplotlib/legend_handler.py:

Start line: 540, End line: 622

```python
class HandlerErrorbar(HandlerLine2D):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        plotlines, caplines, barlinecols = orig_handle

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata)

        xdata_marker = np.asarray(xdata_marker)
        ydata_marker = np.asarray(ydata[:len(xdata_marker)])

        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
                                                 width, height, fontsize)

        legline_marker = Line2D(xdata_marker, ydata_marker)

        # when plotlines are None (only errorbars are drawn), we just
        # make legline invisible.
        if plotlines is None:
            legline.set_visible(False)
            legline_marker.set_visible(False)
        else:
            self.update_prop(legline, plotlines, legend)

            legline.set_drawstyle('default')
            legline.set_marker('none')

            self.update_prop(legline_marker, plotlines, legend)
            legline_marker.set_linestyle('None')

            if legend.markerscale != 1:
                newsz = legline_marker.get_markersize() * legend.markerscale
                legline_marker.set_markersize(newsz)

        handle_barlinecols = []
        handle_caplines = []

        if orig_handle.has_xerr:
            verts = [((x - xerr_size, y), (x + xerr_size, y))
                     for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker - xerr_size, ydata_marker)
                capline_right = Line2D(xdata_marker + xerr_size, ydata_marker)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker("|")
                capline_right.set_marker("|")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        if orig_handle.has_yerr:
            verts = [((x, y - yerr_size), (x, y + yerr_size))
                     for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker, ydata_marker - yerr_size)
                capline_right = Line2D(xdata_marker, ydata_marker + yerr_size)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker("_")
                capline_right.set_marker("_")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        artists = [
            *handle_barlinecols, *handle_caplines, legline, legline_marker,
        ]
        for artist in artists:
            artist.set_transform(trans)
        return artists
```
### 9 - lib/matplotlib/legend_handler.py:

Start line: 736, End line: 767

```python
class HandlerTuple(HandlerBase):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        handler_map = legend.get_legend_handler_map()

        if self._ndivide is None:
            ndivide = len(orig_handle)
        else:
            ndivide = self._ndivide

        if self._pad is None:
            pad = legend.borderpad * fontsize
        else:
            pad = self._pad * fontsize

        if ndivide > 1:
            width = (width - pad * (ndivide - 1)) / ndivide

        xds_cycle = cycle(xdescent - (width + pad) * np.arange(ndivide))

        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(
                legend, handle1,
                next(xds_cycle), ydescent, width, height, fontsize, trans)
            if isinstance(_a_list, _Line2DHandleList):
                _a_list = [_a_list[0]]
            a_list.extend(_a_list)

        return a_list
```
### 10 - examples/text_labels_and_annotations/legend_demo.py:

Start line: 1, End line: 75

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
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D
import numpy as np

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


###############################################################################
# Next we'll demonstrate plotting more complex labels.

x = np.linspace(0, 1)

fig, (ax0, ax1) = plt.subplots(2, 1)

# Plot the lines y=x**n for n=1..4.
for n in range(1, 5):
    ax0.plot(x, x**n, label="n={0}".format(n))
leg = ax0.legend(loc="upper left", bbox_to_anchor=[0, 1],
                 ncol=2, shadow=True, title="Legend", fancybox=True)
leg.get_title().set_color("red")

# Demonstrate some more complex labels.
ax1.plot(x, x**2, label="multi\nline")
half_pi = np.linspace(0, np.pi / 2)
ax1.plot(np.sin(half_pi), np.cos(half_pi), label=r"$\frac{1}{2}\pi$")
ax1.plot(x, 2**(x**2), label="$2^{x^2}$")
ax1.legend(shadow=True, fancybox=True)

plt.show()


###############################################################################
# Here we attach legends to more complex plots.

fig, axs = plt.subplots(3, 1, constrained_layout=True)
top_ax, middle_ax, bottom_ax = axs

top_ax.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4, label="Bar 1",
           align="center")
top_ax.bar([0.5, 1.5, 2.5], [0.3, 0.2, 0.2], color="red", width=0.4,
           label="Bar 2", align="center")
top_ax.legend()

middle_ax.errorbar([0, 1, 2], [2, 3, 1], xerr=0.4, fmt="s", label="test 1")
```
### 13 - lib/matplotlib/legend.py:

Start line: 1108, End line: 1204

```python
def _parse_legend_args(axs, *args, handles=None, labels=None, **kwargs):
    """
    Get the handles and labels from the calls to either ``figure.legend``
    or ``axes.legend``.

    The parser is a bit involved because we support::

        legend()
        legend(labels)
        legend(handles, labels)
        legend(labels=labels)
        legend(handles=handles)
        legend(handles=handles, labels=labels)

    The behavior for a mixture of positional and keyword handles and labels
    is undefined and issues a warning.

    Parameters
    ----------
    axs : list of `.Axes`
        If handles are not given explicitly, the artists in these Axes are
        used as handles.
    *args : tuple
        Positional parameters passed to ``legend()``.
    handles
        The value of the keyword argument ``legend(handles=...)``, or *None*
        if that keyword argument was not used.
    labels
        The value of the keyword argument ``legend(labels=...)``, or *None*
        if that keyword argument was not used.
    **kwargs
        All other keyword arguments passed to ``legend()``.

    Returns
    -------
    handles : list of `.Artist`
        The legend handles.
    labels : list of str
        The legend labels.
    extra_args : tuple
        *args* with positional handles and labels removed.
    kwargs : dict
        *kwargs* with keywords handles and labels removed.

    """
    log = logging.getLogger(__name__)

    handlers = kwargs.get('handler_map')
    extra_args = ()

    if (handles is not None or labels is not None) and args:
        _api.warn_external("You have mixed positional and keyword arguments, "
                           "some input may be discarded.")

    # if got both handles and labels as kwargs, make same length
    if handles and labels:
        handles, labels = zip(*zip(handles, labels))

    elif handles is not None and labels is None:
        labels = [handle.get_label() for handle in handles]

    elif labels is not None and handles is None:
        # Get as many handles as there are labels.
        handles = [handle for handle, label
                   in zip(_get_legend_handles(axs, handlers), labels)]

    # No arguments - automatically detect labels and handles.
    elif len(args) == 0:
        handles, labels = _get_legend_handles_labels(axs, handlers)
        if not handles:
            log.warning(
                "No artists with labels found to put in legend.  Note that "
                "artists whose label start with an underscore are ignored "
                "when legend() is called with no argument.")

    # One argument. User defined labels - automatic handle detection.
    elif len(args) == 1:
        labels, = args
        if any(isinstance(l, Artist) for l in labels):
            raise TypeError("A single argument passed to legend() must be a "
                            "list of labels, but found an Artist in there.")

        # Get as many handles as there are labels.
        handles = [handle for handle, label
                   in zip(_get_legend_handles(axs, handlers), labels)]

    # Two arguments:
    #   * user defined handles and labels
    elif len(args) >= 2:
        handles, labels = args[:2]
        extra_args = args[2:]

    else:
        raise TypeError('Invalid arguments to legend.')

    return handles, labels, extra_args, kwargs
```
### 19 - lib/matplotlib/legend.py:

Start line: 470, End line: 561

```python
class Legend(Artist):

    @docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
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

        ncol=1,     # number of columns
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
    ):
        # ... other code

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
        if isinstance(labelcolor, str) and labelcolor in color_getters:
            getter_names = color_getters[labelcolor]
            for handle, text in zip(self.legendHandles, self.texts):
                for getter_name in getter_names:
                    try:
                        color = getattr(handle, getter_name)()
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
            raise ValueError("Invalid argument for labelcolor : %s" %
                             str(labelcolor))
```
### 25 - lib/matplotlib/legend.py:

Start line: 1069, End line: 1105

```python
# Helper functions to parse legend arguments for both `figure.legend` and
# `axes.legend`:
def _get_legend_handles(axs, legend_handler_map=None):
    """Yield artists that can be used as handles in a legend."""
    handles_original = []
    for ax in axs:
        handles_original += [
            *(a for a in ax._children
              if isinstance(a, (Line2D, Patch, Collection))),
            *ax.containers]
        # support parasite axes:
        if hasattr(ax, 'parasites'):
            for axx in ax.parasites:
                handles_original += [
                    *(a for a in axx._children
                      if isinstance(a, (Line2D, Patch, Collection))),
                    *axx.containers]

    handler_map = {**Legend.get_default_handler_map(),
                   **(legend_handler_map or {})}
    has_handler = Legend.get_legend_handler
    for handle in handles_original:
        label = handle.get_label()
        if label != '_nolegend_' and has_handler(handler_map, handle):
            yield handle


def _get_legend_handles_labels(axs, legend_handler_map=None):
    """Return handles and labels for legend."""
    handles = []
    labels = []
    for handle in _get_legend_handles(axs, legend_handler_map):
        label = handle.get_label()
        if label and not label.startswith('_'):
            handles.append(handle)
            labels.append(label)
    return handles, labels
```
### 27 - lib/matplotlib/legend.py:

Start line: 285, End line: 468

```python
class Legend(Artist):
    """
    Place a legend on the axes at location loc.
    """

    # 'best' is only implemented for axes legends
    codes = {'best': 0, **AnchoredOffsetbox.codes}
    zorder = 5

    def __str__(self):
        return "Legend"

    @docstring.dedent_interpd
    def __init__(
        self, parent, handles, labels,
        loc=None,
        numpoints=None,      # number of points in the legend line
        markerscale=None,    # relative size of legend markers vs. original
        markerfirst=True,    # left/right ordering of legend marker and label
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

        ncol=1,     # number of columns
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
            to the smaller of both lengths.

        Other Parameters
        ----------------
        %(_legend_kw_doc)s

        Notes
        -----
        Users can specify any arbitrary location for the legend using the
        *bbox_to_anchor* keyword argument. *bbox_to_anchor* can be a
        `.BboxBase` (or derived there from) or a tuple of 2 or 4 floats.
        See `set_bbox_to_anchor` for more detail.

        The legend location can be specified by setting *loc* with a tuple of
        2 floats, which is interpreted as the lower-left corner of the legend
        in the normalized axes coordinate.
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
        self.legendHandles = []
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
                _api.warn_external('The handle {!r} has a label of {!r} '
                                   'which cannot be automatically added to'
                                   ' the legend.'.format(handle, label))
            else:
                _lab.append(label)
                _hand.append(handle)
        labels, handles = _lab, _hand

        handles = list(handles)
        if len(handles) < 2:
            ncol = 1
        self._ncol = ncol

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

        self._loc_used_default = loc is None
        if loc is None:
            loc = mpl.rcParams["legend.loc"]
            if not self.isaxes and loc in [0, 'best']:
                loc = 'upper right'
        if isinstance(loc, str):
            loc = _api.check_getitem(self.codes, loc=loc)
        if not self.isaxes and loc == 0:
            raise ValueError(
                "Automatic legend placement (loc='best') not implemented for "
                "figure legend")

        self._mode = mode
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)

        # We use FancyBboxPatch to draw a legend frame. The location
        # and size of the box will be updated during the drawing time.
        # ... other code
```
### 30 - lib/matplotlib/text.py:

Start line: 91, End line: 160

```python
@docstring.interpd
@cbook._define_aliases({
    "color": ["c"],
    "fontfamily": ["family"],
    "fontproperties": ["font", "font_properties"],
    "horizontalalignment": ["ha"],
    "multialignment": ["ma"],
    "fontname": ["name"],
    "fontsize": ["size"],
    "fontstretch": ["stretch"],
    "fontstyle": ["style"],
    "fontvariant": ["variant"],
    "verticalalignment": ["va"],
    "fontweight": ["weight"],
})
class Text(Artist):
    """Handle storing and drawing of text in window or data coordinates."""

    zorder = 3
    _cached = cbook.maxdict(50)

    def __repr__(self):
        return "Text(%s, %s, %s)" % (self._x, self._y, repr(self._text))

    def __init__(self,
                 x=0, y=0, text='',
                 color=None,           # defaults to rc params
                 verticalalignment='baseline',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None,  # defaults to FontProperties()
                 rotation=None,
                 linespacing=None,
                 rotation_mode=None,
                 usetex=None,          # defaults to rcParams['text.usetex']
                 wrap=False,
                 transform_rotates_text=False,
                 *,
                 parse_math=True,
                 **kwargs
                 ):
        """
        Create a `.Text` instance at *x*, *y* with string *text*.

        Valid keyword arguments are:

        %(Text:kwdoc)s
        """
        super().__init__()
        self._x, self._y = x, y
        self._text = ''
        self.set_text(text)
        self.set_color(
            color if color is not None else mpl.rcParams["text.color"])
        self.set_fontproperties(fontproperties)
        self.set_usetex(usetex)
        self.set_parse_math(parse_math)
        self.set_wrap(wrap)
        self.set_verticalalignment(verticalalignment)
        self.set_horizontalalignment(horizontalalignment)
        self._multialignment = multialignment
        self._rotation = rotation
        self._transform_rotates_text = transform_rotates_text
        self._bbox_patch = None  # a FancyBboxPatch instance
        self._renderer = None
        if linespacing is None:
            linespacing = 1.2   # Maybe use rcParam later.
        self._linespacing = linespacing
        self.set_rotation_mode(rotation_mode)
        self.update(kwargs)
```
### 32 - lib/matplotlib/legend.py:

Start line: 877, End line: 913

```python
class Legend(Artist):

    def get_title(self):
        """Return the `.Text` instance for the legend title."""
        return self._legend_title_box._text

    def get_window_extent(self, renderer=None):
        # docstring inherited
        if renderer is None:
            renderer = self.figure._cachedRenderer
        return self._legend_box.get_window_extent(renderer=renderer)

    def get_tightbbox(self, renderer):
        # docstring inherited
        return self._legend_box.get_window_extent(renderer)

    def get_frame_on(self):
        """Get whether the legend box patch is drawn."""
        return self.legendPatch.get_visible()

    def set_frame_on(self, b):
        """
        Set whether the legend box patch is drawn.

        Parameters
        ----------
        b : bool
        """
        self.legendPatch.set_visible(b)
        self.stale = True

    draw_frame = set_frame_on  # Backcompat alias.

    def get_bbox_to_anchor(self):
        """Return the bbox that the legend will be anchored to."""
        if self._bbox_to_anchor is None:
            return self.parent.bbox
        else:
            return self._bbox_to_anchor
```
### 54 - lib/matplotlib/legend.py:

Start line: 1, End line: 49

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
from matplotlib import _api, docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
                                StepPatch)
from matplotlib.collections import (
    Collection, CircleCollection, LineCollection, PathCollection,
    PolyCollection, RegularPolyCollection)
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
### 61 - lib/matplotlib/text.py:

Start line: 671, End line: 739

```python
@docstring.interpd
@cbook._define_aliases({
    "color": ["c"],
    "fontfamily": ["family"],
    "fontproperties": ["font", "font_properties"],
    "horizontalalignment": ["ha"],
    "multialignment": ["ma"],
    "fontname": ["name"],
    "fontsize": ["size"],
    "fontstretch": ["stretch"],
    "fontstyle": ["style"],
    "fontvariant": ["variant"],
    "verticalalignment": ["va"],
    "fontweight": ["weight"],
})
class Text(Artist):

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible():
            return
        if self.get_text() == '':
            return

        renderer.open_group('text', self.get_gid())

        with self._cm_set(text=self._get_wrapped_text()):
            bbox, info, descent = self._get_layout(renderer)
            trans = self.get_transform()

            # don't use self.get_position here, which refers to text
            # position in Text:
            posx = float(self.convert_xunits(self._x))
            posy = float(self.convert_yunits(self._y))
            posx, posy = trans.transform((posx, posy))
            if not np.isfinite(posx) or not np.isfinite(posy):
                _log.warning("posx and posy should be finite values")
                return
            canvasw, canvash = renderer.get_canvas_width_height()

            # Update the location and size of the bbox
            # (`.patches.FancyBboxPatch`), and draw it.
            if self._bbox_patch:
                self.update_bbox_position_size(renderer)
                self._bbox_patch.draw(renderer)

            gc = renderer.new_gc()
            gc.set_foreground(self.get_color())
            gc.set_alpha(self.get_alpha())
            gc.set_url(self._url)
            self._set_gc_clip(gc)

            angle = self.get_rotation()

            for line, wh, x, y in info:

                mtext = self if len(info) == 1 else None
                x = x + posx
                y = y + posy
                if renderer.flipy():
                    y = canvash - y
                clean_line, ismath = self._preprocess_math(line)

                if self.get_path_effects():
                    from matplotlib.patheffects import PathEffectRenderer
                    textrenderer = PathEffectRenderer(
                        self.get_path_effects(), renderer)
                else:
                    textrenderer = renderer

                if self.get_usetex():
                    textrenderer.draw_tex(gc, x, y, clean_line,
                                          self._fontproperties, angle,
                                          mtext=mtext)
                else:
                    textrenderer.draw_text(gc, x, y, clean_line,
                                           self._fontproperties, angle,
                                           ismath=ismath, mtext=mtext)

        gc.restore()
        renderer.close_group('text')
        self.stale = False
```
### 78 - lib/matplotlib/text.py:

Start line: 372, End line: 445

```python
@docstring.interpd
@cbook._define_aliases({
    "color": ["c"],
    "fontfamily": ["family"],
    "fontproperties": ["font", "font_properties"],
    "horizontalalignment": ["ha"],
    "multialignment": ["ma"],
    "fontname": ["name"],
    "fontsize": ["size"],
    "fontstretch": ["stretch"],
    "fontstyle": ["style"],
    "fontvariant": ["variant"],
    "verticalalignment": ["va"],
    "fontweight": ["weight"],
})
class Text(Artist):

    def _get_layout(self, renderer):
        # ... other code
        corners_horiz = np.array(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

        # now rotate the bbox
        corners_rotated = M.transform(corners_horiz)
        # compute the bounds of the rotated box
        xmin = corners_rotated[:, 0].min()
        xmax = corners_rotated[:, 0].max()
        ymin = corners_rotated[:, 1].min()
        ymax = corners_rotated[:, 1].max()
        width = xmax - xmin
        height = ymax - ymin

        # Now move the box to the target position offset the display
        # bbox by alignment
        halign = self._horizontalalignment
        valign = self._verticalalignment

        rotation_mode = self.get_rotation_mode()
        if rotation_mode != "anchor":
            # compute the text location in display coords and the offsets
            # necessary to align the bbox with that location
            if halign == 'center':
                offsetx = (xmin + xmax) / 2
            elif halign == 'right':
                offsetx = xmax
            else:
                offsetx = xmin

            if valign == 'center':
                offsety = (ymin + ymax) / 2
            elif valign == 'top':
                offsety = ymax
            elif valign == 'baseline':
                offsety = ymin + descent
            elif valign == 'center_baseline':
                offsety = ymin + height - baseline / 2.0
            else:
                offsety = ymin
        else:
            xmin1, ymin1 = corners_horiz[0]
            xmax1, ymax1 = corners_horiz[2]

            if halign == 'center':
                offsetx = (xmin1 + xmax1) / 2.0
            elif halign == 'right':
                offsetx = xmax1
            else:
                offsetx = xmin1

            if valign == 'center':
                offsety = (ymin1 + ymax1) / 2.0
            elif valign == 'top':
                offsety = ymax1
            elif valign == 'baseline':
                offsety = ymax1 - baseline
            elif valign == 'center_baseline':
                offsety = ymax1 - baseline / 2.0
            else:
                offsety = ymin1

            offsetx, offsety = M.transform((offsetx, offsety))

        xmin -= offsetx
        ymin -= offsety

        bbox = Bbox.from_bounds(xmin, ymin, width, height)

        # now rotate the positions around the first (x, y) position
        xys = M.transform(offset_layout) - (offsetx, offsety)

        ret = bbox, list(zip(lines, zip(ws, hs), *xys.T)), descent
        self._cached[key] = ret
        return ret
```
### 80 - lib/matplotlib/legend.py:

Start line: 605, End line: 634

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
### 82 - lib/matplotlib/text.py:

Start line: 1792, End line: 1829

```python
class Annotation(Text, _AnnotationBase):

    def __init__(self, text, xy,
                 xytext=None,
                 xycoords='data',
                 textcoords=None,
                 arrowprops=None,
                 annotation_clip=None,
                 **kwargs):
        _AnnotationBase.__init__(self,
                                 xy,
                                 xycoords=xycoords,
                                 annotation_clip=annotation_clip)
        # warn about wonky input data
        if (xytext is None and
                textcoords is not None and
                textcoords != xycoords):
            _api.warn_external("You have used the `textcoords` kwarg, but "
                               "not the `xytext` kwarg.  This can lead to "
                               "surprising results.")

        # clean up textcoords and assign default
        if textcoords is None:
            textcoords = self.xycoords
        self._textcoords = textcoords

        # cleanup xytext defaults
        if xytext is None:
            xytext = self.xy
        x, y = xytext

        self.arrowprops = arrowprops
        if arrowprops is not None:
            arrowprops = arrowprops.copy()
            if "arrowstyle" in arrowprops:
                self._arrow_relpos = arrowprops.pop("relpos", (0.5, 0.5))
            else:
                # modified YAArrow API to be used with FancyArrowPatch
                for key in [
                        'width', 'headwidth', 'headlength', 'shrink', 'frac']:
                    arrowprops.pop(key, None)
            self.arrow_patch = FancyArrowPatch((0, 0), (1, 1), **arrowprops)
        else:
            self.arrow_patch = None

        # Must come last, as some kwargs may be propagated to arrow_patch.
        Text.__init__(self, x, y, text, **kwargs)
```
### 93 - lib/matplotlib/legend.py:

Start line: 705, End line: 785

```python
class Legend(Artist):

    def _init_legend_box(self, handles, labels, markerfirst=True):
        """
        Initialize the legend_box. The legend_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """

        fontsize = self._fontsize

        # legend_box is a HPacker, horizontally packed with columns.
        # Each column is a VPacker, vertically packed with legend items.
        # Each legend item is a HPacker packed with:
        # - handlebox: a DrawingArea which contains the legend handle.
        # - labelbox: a TextArea which contains the legend text.

        text_list = []  # the list of text instances
        handle_list = []  # the list of handle instances
        handles_and_labels = []

        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.
        descent = 0.35 * fontsize * (self.handleheight - 0.7)  # heuristic.
        height = fontsize * self.handleheight - descent
        # each handle needs to be drawn inside a box of (x, y, w, h) =
        # (0, -descent, width, height).  And their coordinates should
        # be given in the display coordinates.

        # The transformation of each handle will be automatically set
        # to self.get_transform(). If the artist does not use its
        # default transform (e.g., Collections), you need to
        # manually set their transform to the self.get_transform().
        legend_handler_map = self.get_legend_handler_map()

        for orig_handle, label in zip(handles, labels):
            handler = self.get_legend_handler(legend_handler_map, orig_handle)
            if handler is None:
                _api.warn_external(
                    "Legend does not support {!r} instances.\nA proxy artist "
                    "may be used instead.\nSee: "
                    "https://matplotlib.org/users/legend_guide.html"
                    "#creating-artists-specifically-for-adding-to-the-legend-"
                    "aka-proxy-artists".format(orig_handle))
                # No handle for this artist, so we just defer to None.
                handle_list.append(None)
            else:
                textbox = TextArea(label, multilinebaseline=True,
                                   textprops=dict(
                                       verticalalignment='baseline',
                                       horizontalalignment='left',
                                       fontproperties=self.prop))
                handlebox = DrawingArea(width=self.handlelength * fontsize,
                                        height=height,
                                        xdescent=0., ydescent=descent)

                text_list.append(textbox._text)
                # Create the artist for the legend which represents the
                # original artist/handle.
                handle_list.append(handler.legend_artist(self, orig_handle,
                                                         fontsize, handlebox))
                handles_and_labels.append((handlebox, textbox))

        columnbox = []
        # array_split splits n handles_and_labels into ncol columns, with the
        # first n%ncol columns having an extra entry.  filter(len, ...) handles
        # the case where n < ncol: the last ncol-n columns are empty and get
        # filtered out.
        for handles_and_labels_column \
                in filter(len, np.array_split(handles_and_labels, self._ncol)):
            # pack handlebox and labelbox into itembox
            itemboxes = [HPacker(pad=0,
                                 sep=self.handletextpad * fontsize,
                                 children=[h, t] if markerfirst else [t, h],
                                 align="baseline")
                         for h, t in handles_and_labels_column]
            # pack columnbox
            alignment = "baseline" if markerfirst else "right"
            columnbox.append(VPacker(pad=0,
                                     sep=self.labelspacing * fontsize,
                                     align=alignment,
                                     children=itemboxes))
        # ... other code
```
### 102 - lib/matplotlib/text.py:

Start line: 1968, End line: 1985

```python
class Annotation(Text, _AnnotationBase):

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible() or not self._check_xy(renderer):
            return
        # Update text positions before `Text.draw` would, so that the
        # FancyArrowPatch is correctly positioned.
        self.update_positions(renderer)
        self.update_bbox_position_size(renderer)
        if self.arrow_patch is not None:   # FancyArrowPatch
            if self.arrow_patch.figure is None and self.figure is not None:
                self.arrow_patch.figure = self.figure
            self.arrow_patch.draw(renderer)
        # Draw text, including FancyBboxPatch, after FancyArrowPatch.
        # Otherwise, a wedge arrowstyle can land partly on top of the Bbox.
        Text.draw(self, renderer)
```
