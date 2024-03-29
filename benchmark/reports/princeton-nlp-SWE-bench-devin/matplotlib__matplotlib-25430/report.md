# matplotlib__matplotlib-25430

| **matplotlib/matplotlib** | `7eafdd8af3c523c1c77b027d378fb337dd489f18` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 1430 |
| **Any found context length** | 1430 |
| **Avg pos** | 2.3333333333333335 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/backends/backend_agg.py b/lib/matplotlib/backends/backend_agg.py
--- a/lib/matplotlib/backends/backend_agg.py
+++ b/lib/matplotlib/backends/backend_agg.py
@@ -441,7 +441,9 @@ def buffer_rgba(self):
         """
         return self.renderer.buffer_rgba()
 
-    def print_raw(self, filename_or_obj):
+    def print_raw(self, filename_or_obj, *, metadata=None):
+        if metadata is not None:
+            raise ValueError("metadata not supported for raw/rgba")
         FigureCanvasAgg.draw(self)
         renderer = self.get_renderer()
         with cbook.open_file_cm(filename_or_obj, "wb") as fh:
@@ -518,22 +520,22 @@ def print_to_buffer(self):
     # print_figure(), and the latter ensures that `self.figure.dpi` already
     # matches the dpi kwarg (if any).
 
-    def print_jpg(self, filename_or_obj, *, pil_kwargs=None):
+    def print_jpg(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
         # savefig() has already applied savefig.facecolor; we now set it to
         # white to make imsave() blend semi-transparent figures against an
         # assumed white background.
         with mpl.rc_context({"savefig.facecolor": "white"}):
-            self._print_pil(filename_or_obj, "jpeg", pil_kwargs)
+            self._print_pil(filename_or_obj, "jpeg", pil_kwargs, metadata)
 
     print_jpeg = print_jpg
 
-    def print_tif(self, filename_or_obj, *, pil_kwargs=None):
-        self._print_pil(filename_or_obj, "tiff", pil_kwargs)
+    def print_tif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
+        self._print_pil(filename_or_obj, "tiff", pil_kwargs, metadata)
 
     print_tiff = print_tif
 
-    def print_webp(self, filename_or_obj, *, pil_kwargs=None):
-        self._print_pil(filename_or_obj, "webp", pil_kwargs)
+    def print_webp(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
+        self._print_pil(filename_or_obj, "webp", pil_kwargs, metadata)
 
     print_jpg.__doc__, print_tif.__doc__, print_webp.__doc__ = map(
         """
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -3259,6 +3259,11 @@ def savefig(self, fname, *, transparent=None, **kwargs):
               `~.FigureCanvasSVG.print_svg`.
             - 'eps' and 'ps' with PS backend: Only 'Creator' is supported.
 
+            Not supported for 'pgf', 'raw', and 'rgba' as those formats do not support
+            embedding metadata.
+            Does not currently support 'jpg', 'tiff', or 'webp', but may include
+            embedding EXIF metadata in the future.
+
         bbox_inches : str or `.Bbox`, default: :rc:`savefig.bbox`
             Bounding box in inches: only the given portion of the figure is
             saved.  If 'tight', try to figure out the tight bbox of the figure.
diff --git a/lib/matplotlib/image.py b/lib/matplotlib/image.py
--- a/lib/matplotlib/image.py
+++ b/lib/matplotlib/image.py
@@ -1610,6 +1610,7 @@ def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
         Metadata in the image file.  The supported keys depend on the output
         format, see the documentation of the respective backends for more
         information.
+        Currently only supported for "png", "pdf", "ps", "eps", and "svg".
     pil_kwargs : dict, optional
         Keyword arguments passed to `PIL.Image.Image.save`.  If the 'pnginfo'
         key is present, it completely overrides *metadata*, including the
@@ -1674,6 +1675,8 @@ def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
                 for k, v in metadata.items():
                     if v is not None:
                         pnginfo.add_text(k, v)
+        elif metadata is not None:
+            raise ValueError(f"metadata not supported for format {format!r}")
         if format in ["jpg", "jpeg"]:
             format = "jpeg"  # Pillow doesn't recognize "jpg".
             facecolor = mpl.rcParams["savefig.facecolor"]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/backends/backend_agg.py | 444 | 444 | 1 | 1 | 1430
| lib/matplotlib/backends/backend_agg.py | 521 | 536 | 1 | 1 | 1430
| lib/matplotlib/figure.py | 3262 | 3262 | 3 | 3 | 3289
| lib/matplotlib/image.py | 1613 | 1613 | - | 2 | -
| lib/matplotlib/image.py | 1677 | 1677 | 2 | 2 | 2175


## Problem Statement

```
[Bug]: savefig + jpg + metadata fails with inscrutable error message
### Bug summary

If we call `savefig` with a `filename` with a `.jpg` extension, with the `metadata` kwarg specified, the error message is inscrutable.

### Code for reproduction

\`\`\`python
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
x = np.linspace(0, 10, 100)
y = 4 + 2 * np.sin(2 * x)

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.savefig("sin.jpg", metadata={})
\`\`\`


### Actual outcome

\`\`\`
Traceback (most recent call last):
  File "/private/tmp/./reproduce.py", line 19, in <module>
    plt.savefig("sin.jpg", metadata={})
  File "/private/tmp/lib/python3.11/site-packages/matplotlib/pyplot.py", line 1023, in savefig
    res = fig.savefig(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/lib/python3.11/site-packages/matplotlib/figure.py", line 3343, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "/private/tmp/lib/python3.11/site-packages/matplotlib/backend_bases.py", line 2366, in print_figure
    result = print_method(
             ^^^^^^^^^^^^^
  File "/private/tmp/lib/python3.11/site-packages/matplotlib/backend_bases.py", line 2232, in <lambda>
    print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
                                                                 ^^^^^
TypeError: FigureCanvasAgg.print_jpg() got an unexpected keyword argument 'metadata'
\`\`\`

### Expected outcome

Either metadata should be added, the argument ignored, or a more informative error message.

### Additional information

_No response_

### Operating system

OS/X

### Matplotlib Version

3.7.1

### Matplotlib Backend

MacOSX

### Python version

Python 3.11.2

### Jupyter version

_No response_

### Installation

pip

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 lib/matplotlib/backends/backend_agg.py** | 379 | 557| 1430 | 1430 | 4793 | 
| **-> 2 <-** | **2 lib/matplotlib/image.py** | 1624 | 1688| 745 | 2175 | 21959 | 
| **-> 3 <-** | **3 lib/matplotlib/figure.py** | 3208 | 3320| 1114 | 3289 | 51435 | 
| 4 | **3 lib/matplotlib/figure.py** | 3322 | 3359| 363 | 3652 | 51435 | 
| 5 | 4 lib/matplotlib/pyplot.py | 1019 | 1036| 148 | 3800 | 80145 | 
| 6 | 4 lib/matplotlib/pyplot.py | 2286 | 2296| 137 | 3937 | 80145 | 
| 7 | 5 lib/matplotlib/backends/backend_svg.py | 1289 | 1368| 721 | 4658 | 91422 | 
| 8 | 6 lib/matplotlib/backend_bases.py | 1650 | 2409| 6325 | 10983 | 120461 | 
| 9 | 7 lib/matplotlib/backends/backend_pgf.py | 128 | 163| 352 | 11335 | 130159 | 
| 10 | 8 lib/matplotlib/backends/backend_cairo.py | 447 | 501| 496 | 11831 | 134439 | 
| 11 | **8 lib/matplotlib/figure.py** | 1 | 69| 493 | 12324 | 134439 | 
| 12 | 9 lib/matplotlib/sphinxext/plot_directive.py | 517 | 625| 789 | 13113 | 140708 | 
| 13 | 10 lib/matplotlib/backends/backend_wx.py | 10 | 60| 343 | 13456 | 152823 | 
| 14 | 10 lib/matplotlib/pyplot.py | 1 | 89| 670 | 14126 | 152823 | 
| 15 | 10 lib/matplotlib/backends/backend_wx.py | 888 | 899| 151 | 14277 | 152823 | 
| 16 | 11 galleries/tutorials/introductory/lifecycle.py | 188 | 278| 821 | 15098 | 155185 | 
| 17 | **11 lib/matplotlib/figure.py** | 2362 | 2532| 1511 | 16609 | 155185 | 
| 18 | 12 lib/matplotlib/backends/backend_ps.py | 908 | 965| 593 | 17202 | 167244 | 
| 19 | 13 galleries/tutorials/text/usetex.py | 1 | 175| 1778 | 18980 | 169022 | 
| 20 | 13 lib/matplotlib/backends/backend_ps.py | 802 | 849| 439 | 19419 | 169022 | 
| 21 | 13 lib/matplotlib/backends/backend_ps.py | 967 | 986| 281 | 19700 | 169022 | 
| 22 | 13 lib/matplotlib/sphinxext/plot_directive.py | 330 | 410| 514 | 20214 | 169022 | 
| 23 | 13 lib/matplotlib/backends/backend_svg.py | 1 | 104| 477 | 20691 | 169022 | 
| 24 | 14 galleries/tutorials/introductory/images.py | 1 | 125| 1182 | 21873 | 171277 | 
| 25 | 14 lib/matplotlib/backends/backend_pgf.py | 1 | 28| 145 | 22018 | 171277 | 
| 26 | 15 galleries/examples/subplots_axes_and_figures/mosaic.py | 56 | 167| 799 | 22817 | 173970 | 
| 27 | **15 lib/matplotlib/image.py** | 1396 | 1416| 255 | 23072 | 173970 | 
| 28 | 16 galleries/examples/subplots_axes_and_figures/figure_size_units.py | 1 | 83| 629 | 23701 | 174599 | 
| 29 | **16 lib/matplotlib/figure.py** | 2534 | 2554| 268 | 23969 | 174599 | 
| 30 | 16 lib/matplotlib/backends/backend_pgf.py | 725 | 787| 573 | 24542 | 174599 | 
| 31 | 17 galleries/examples/misc/rasterization_demo.py | 75 | 95| 166 | 24708 | 175528 | 
| 32 | 17 galleries/tutorials/introductory/images.py | 127 | 230| 949 | 25657 | 175528 | 
| 33 | 18 galleries/tutorials/introductory/animation_tutorial.py | 94 | 244| 1492 | 27149 | 177966 | 
| 34 | 19 doc/conf.py | 216 | 309| 739 | 27888 | 184168 | 
| 35 | 19 doc/conf.py | 150 | 197| 481 | 28369 | 184168 | 
| 36 | 19 lib/matplotlib/pyplot.py | 802 | 868| 666 | 29035 | 184168 | 
| 37 | 20 galleries/examples/statistics/errorbars_and_boxes.py | 64 | 82| 117 | 29152 | 184814 | 
| 38 | 20 lib/matplotlib/backends/backend_cairo.py | 428 | 445| 181 | 29333 | 184814 | 
| 39 | 21 galleries/examples/user_interfaces/canvasagg.py | 1 | 72| 554 | 29887 | 185368 | 
| 40 | 22 lib/matplotlib/backends/backend_gtk3.py | 354 | 400| 430 | 30317 | 190362 | 


### Hint

```
Ultimately the jpg writer does not support metadata. If it could, that would be the most ideal solution for at least the narrow case. (Though this is not at all specific to jpg, while many of the most common formats such as png, pdf, or svg do accept metadata, not all do)

I could see an argument for failing a bit earlier/catching and raising earlier in the stack, but honestly an error that says "unexpected keyword argument 'metadata'" with a call stack that shows `**kwargs` an every step back to what you typed is relatively informative. I would lean towards not making changes for this, but not super strong in that opinion.

I do not think that silently ignoring metadata is actually a good answer, as that would lead to users who expect their metadata to be in the saved file to be mistaken (perhaps in not super recoverable/immediately discovered ways), thus I think an error message is warranted (this is less true in the specific case of passing `metadata={}`, but not interested in doing value inspection here)

I do think the docs could be a little more explicit (they _do_ mention that which keys are supported is a function of backend and output format, but don't explicitly state that some combinations do not accept metadata at all).
Some further investigations:

Adding `metadata` as a kwarg to `print_jpg` (and then passing to `_print_pil` which ultimately calls `mpl.image.imsave`) does silence the exception, however any metadata in that dictionary is silently dropped.

In fact, the `metadata` argument to `imsave` is _only_ used in handling for `png` (some formats, such as pdf, svg, and eps/ps also handle metadata, but do not go via `imsave` as they are vector formats)
I may suggest at least a warning for passing (non-empty?, just do `if metadata:`) for non-png formats there, if not some more extensive adding metadata. I'd be satisfied by such a warning combined with passing through the `metadata` argument so its not an exception before it gets there. (And that would be an appropriate warning for anybody calling `imsave` directly, too) 

JPEG, unlike PNG, does not have an arbitrary string-keyed key-value metadata store (at least exposed by Pillow, not like I went and read the JPEG standard to come to that determination)

There does exist EXIF tags that could in principle be added to JPEG (and are not in the `savefig` pipeline/`imsave` in particular). But EXIF is not arbitrary string keys (it is actually something more like integer keyed, with an external mapping of integers to the key names, so the keys are pretty static, not arbitrary)

I would not be opposed to adding supported tags from metadata (and warning or erroring on invalid tag, not sure which) but working with Pillow's EXIF datatypes is not as easy (or well documented) as I'd like.

Pillow has an `image.info` dict-like attribute, but _most_ of that is simply ignored on writing for most file formats (silently) and behavior varies by output format significantly.


Summary of current behavior (using `agg` based backends, calling `savefig`):

- PNG (filters to `imsave`/Pillow): basically fully works, arbitrary string keys
- PDF (filters to PDF backend): Mostly works, but warns on unrecognized keys, but still puts them there
- JPG, WebP, Tiff (filters to `imsave`/Pillow): Error similar to above if any metadata is passed, possibly could add EXIF fields (not yet implemented)
- SVG (filters to backend_svg): Limited keys, errors on unrecognized keys
- eps/ps (filters to backend_ps): Only accept `"Creator"`, all other keys silently ignored (As documented in `savefig` docstring)
- raw/rgba: no fields for metadata, currently error similar to JPEG, but no path to add metadata as the file is literally just the rgba image bytes
- pgf (filters to backend_pgf): currently errors similar to JPEG, in principle supports EXIF, I think, but does _not_ go through Pillow, so solution would be separately implemented

```

## Patch

```diff
diff --git a/lib/matplotlib/backends/backend_agg.py b/lib/matplotlib/backends/backend_agg.py
--- a/lib/matplotlib/backends/backend_agg.py
+++ b/lib/matplotlib/backends/backend_agg.py
@@ -441,7 +441,9 @@ def buffer_rgba(self):
         """
         return self.renderer.buffer_rgba()
 
-    def print_raw(self, filename_or_obj):
+    def print_raw(self, filename_or_obj, *, metadata=None):
+        if metadata is not None:
+            raise ValueError("metadata not supported for raw/rgba")
         FigureCanvasAgg.draw(self)
         renderer = self.get_renderer()
         with cbook.open_file_cm(filename_or_obj, "wb") as fh:
@@ -518,22 +520,22 @@ def print_to_buffer(self):
     # print_figure(), and the latter ensures that `self.figure.dpi` already
     # matches the dpi kwarg (if any).
 
-    def print_jpg(self, filename_or_obj, *, pil_kwargs=None):
+    def print_jpg(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
         # savefig() has already applied savefig.facecolor; we now set it to
         # white to make imsave() blend semi-transparent figures against an
         # assumed white background.
         with mpl.rc_context({"savefig.facecolor": "white"}):
-            self._print_pil(filename_or_obj, "jpeg", pil_kwargs)
+            self._print_pil(filename_or_obj, "jpeg", pil_kwargs, metadata)
 
     print_jpeg = print_jpg
 
-    def print_tif(self, filename_or_obj, *, pil_kwargs=None):
-        self._print_pil(filename_or_obj, "tiff", pil_kwargs)
+    def print_tif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
+        self._print_pil(filename_or_obj, "tiff", pil_kwargs, metadata)
 
     print_tiff = print_tif
 
-    def print_webp(self, filename_or_obj, *, pil_kwargs=None):
-        self._print_pil(filename_or_obj, "webp", pil_kwargs)
+    def print_webp(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
+        self._print_pil(filename_or_obj, "webp", pil_kwargs, metadata)
 
     print_jpg.__doc__, print_tif.__doc__, print_webp.__doc__ = map(
         """
diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py
--- a/lib/matplotlib/figure.py
+++ b/lib/matplotlib/figure.py
@@ -3259,6 +3259,11 @@ def savefig(self, fname, *, transparent=None, **kwargs):
               `~.FigureCanvasSVG.print_svg`.
             - 'eps' and 'ps' with PS backend: Only 'Creator' is supported.
 
+            Not supported for 'pgf', 'raw', and 'rgba' as those formats do not support
+            embedding metadata.
+            Does not currently support 'jpg', 'tiff', or 'webp', but may include
+            embedding EXIF metadata in the future.
+
         bbox_inches : str or `.Bbox`, default: :rc:`savefig.bbox`
             Bounding box in inches: only the given portion of the figure is
             saved.  If 'tight', try to figure out the tight bbox of the figure.
diff --git a/lib/matplotlib/image.py b/lib/matplotlib/image.py
--- a/lib/matplotlib/image.py
+++ b/lib/matplotlib/image.py
@@ -1610,6 +1610,7 @@ def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
         Metadata in the image file.  The supported keys depend on the output
         format, see the documentation of the respective backends for more
         information.
+        Currently only supported for "png", "pdf", "ps", "eps", and "svg".
     pil_kwargs : dict, optional
         Keyword arguments passed to `PIL.Image.Image.save`.  If the 'pnginfo'
         key is present, it completely overrides *metadata*, including the
@@ -1674,6 +1675,8 @@ def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
                 for k, v in metadata.items():
                     if v is not None:
                         pnginfo.add_text(k, v)
+        elif metadata is not None:
+            raise ValueError(f"metadata not supported for format {format!r}")
         if format in ["jpg", "jpeg"]:
             format = "jpeg"  # Pillow doesn't recognize "jpg".
             facecolor = mpl.rcParams["savefig.facecolor"]

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_figure.py b/lib/matplotlib/tests/test_figure.py
--- a/lib/matplotlib/tests/test_figure.py
+++ b/lib/matplotlib/tests/test_figure.py
@@ -1548,3 +1548,14 @@ def test_gridspec_no_mutate_input():
     plt.subplots(1, 2, width_ratios=[1, 2], gridspec_kw=gs)
     assert gs == gs_orig
     plt.subplot_mosaic('AB', width_ratios=[1, 2], gridspec_kw=gs)
+
+
+@pytest.mark.parametrize('fmt', ['eps', 'pdf', 'png', 'ps', 'svg', 'svgz'])
+def test_savefig_metadata(fmt):
+    Figure().savefig(io.BytesIO(), format=fmt, metadata={})
+
+
+@pytest.mark.parametrize('fmt', ['jpeg', 'jpg', 'tif', 'tiff', 'webp', "raw", "rgba"])
+def test_savefig_metadata_error(fmt):
+    with pytest.raises(ValueError, match="metadata not supported"):
+        Figure().savefig(io.BytesIO(), format=fmt, metadata={})

```


## Code snippets

### 1 - lib/matplotlib/backends/backend_agg.py:

Start line: 379, End line: 557

```python
class FigureCanvasAgg(FigureCanvasBase):
    # docstring inherited

    _lastKey = None  # Overwritten per-instance on the first draw.

    def copy_from_bbox(self, bbox):
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region, bbox=None, xy=None):
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        # docstring inherited
        self.renderer = self.get_renderer()
        self.renderer.clear()
        # Acquire a lock on the shared font cache.
        with RendererAgg.lock, \
             (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):
            self.figure.draw(self.renderer)
            # A GUI class may be need to update a window using this draw, so
            # don't forget to call the superclass.
            super().draw()

    @_api.delete_parameter("3.6", "cleared", alternative="renderer.clear()")
    def get_renderer(self, cleared=False):
        w, h = self.figure.bbox.size
        key = w, h, self.figure.dpi
        reuse_renderer = (self._lastKey == key)
        if not reuse_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        elif cleared:
            self.renderer.clear()
        return self.renderer

    def tostring_rgb(self):
        """
        Get the image as RGB `bytes`.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        """
        Get the image as ARGB `bytes`.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        """
        Get the image as a `memoryview` to the renderer's buffer.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.buffer_rgba()

    def print_raw(self, filename_or_obj):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        with cbook.open_file_cm(filename_or_obj, "wb") as fh:
            fh.write(renderer.buffer_rgba())

    print_rgba = print_raw

    def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
        """
        Draw the canvas, then save it using `.image.imsave` (to which
        *pil_kwargs* and *metadata* are forwarded).
        """
        FigureCanvasAgg.draw(self)
        mpl.image.imsave(
            filename_or_obj, self.buffer_rgba(), format=fmt, origin="upper",
            dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)

    def print_png(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """
        Write the figure to a PNG file.

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            The file to write to.

        metadata : dict, optional
            Metadata in the PNG file as key-value pairs of bytes or latin-1
            encodable strings.
            According to the PNG specification, keys must be shorter than 79
            chars.

            The `PNG specification`_ defines some common keywords that may be
            used as appropriate:

            - Title: Short (one line) title or caption for image.
            - Author: Name of image's creator.
            - Description: Description of image (possibly long).
            - Copyright: Copyright notice.
            - Creation Time: Time of original image creation
              (usually RFC 1123 format).
            - Software: Software used to create the image.
            - Disclaimer: Legal disclaimer.
            - Warning: Warning of nature of content.
            - Source: Device used to create the image.
            - Comment: Miscellaneous comment;
              conversion from other image format.

            Other keywords may be invented for other purposes.

            If 'Software' is not given, an autogenerated value for Matplotlib
            will be used.  This can be removed by setting it to *None*.

            For more details see the `PNG specification`_.

            .. _PNG specification: \
                https://www.w3.org/TR/2003/REC-PNG-20031110/#11keywords

        pil_kwargs : dict, optional
            Keyword arguments passed to `PIL.Image.Image.save`.

            If the 'pnginfo' key is present, it completely overrides
            *metadata*, including the default 'Software' key.
        """
        self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)

    def print_to_buffer(self):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        return (bytes(renderer.buffer_rgba()),
                (int(renderer.width), int(renderer.height)))

    # Note that these methods should typically be called via savefig() and
    # print_figure(), and the latter ensures that `self.figure.dpi` already
    # matches the dpi kwarg (if any).

    def print_jpg(self, filename_or_obj, *, pil_kwargs=None):
        # savefig() has already applied savefig.facecolor; we now set it to
        # white to make imsave() blend semi-transparent figures against an
        # assumed white background.
        with mpl.rc_context({"savefig.facecolor": "white"}):
            self._print_pil(filename_or_obj, "jpeg", pil_kwargs)

    print_jpeg = print_jpg

    def print_tif(self, filename_or_obj, *, pil_kwargs=None):
        self._print_pil(filename_or_obj, "tiff", pil_kwargs)

    print_tiff = print_tif

    def print_webp(self, filename_or_obj, *, pil_kwargs=None):
        self._print_pil(filename_or_obj, "webp", pil_kwargs)

    print_jpg.__doc__, print_tif.__doc__, print_webp.__doc__ = map(
        """
        Write the figure to a {} file.

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            The file to write to.
        pil_kwargs : dict, optional
            Additional keyword arguments that are passed to
            `PIL.Image.Image.save` when saving the figure.
        """.format, ["JPEG", "TIFF", "WebP"])


@_Backend.export
class _BackendAgg(_Backend):
    backend_version = 'v2.2'
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase
```
### 2 - lib/matplotlib/image.py:

Start line: 1624, End line: 1688

```python
def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
           origin=None, dpi=100, *, metadata=None, pil_kwargs=None):
    # ... other code
    if format in ["pdf", "ps", "eps", "svg"]:
        # Vector formats that are not handled by PIL.
        if pil_kwargs is not None:
            raise ValueError(
                f"Cannot use 'pil_kwargs' when saving to {format}")
        fig = Figure(dpi=dpi, frameon=False)
        fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin,
                     resize=True)
        fig.savefig(fname, dpi=dpi, format=format, transparent=True,
                    metadata=metadata)
    else:
        # Don't bother creating an image; this avoids rounding errors on the
        # size when dividing and then multiplying by dpi.
        if origin is None:
            origin = mpl.rcParams["image.origin"]
        if origin == "lower":
            arr = arr[::-1]
        if (isinstance(arr, memoryview) and arr.format == "B"
                and arr.ndim == 3 and arr.shape[-1] == 4):
            # Such an ``arr`` would also be handled fine by sm.to_rgba below
            # (after casting with asarray), but it is useful to special-case it
            # because that's what backend_agg passes, and can be in fact used
            # as is, saving a few operations.
            rgba = arr
        else:
            sm = cm.ScalarMappable(cmap=cmap)
            sm.set_clim(vmin, vmax)
            rgba = sm.to_rgba(arr, bytes=True)
        if pil_kwargs is None:
            pil_kwargs = {}
        else:
            # we modify this below, so make a copy (don't modify caller's dict)
            pil_kwargs = pil_kwargs.copy()
        pil_shape = (rgba.shape[1], rgba.shape[0])
        image = PIL.Image.frombuffer(
            "RGBA", pil_shape, rgba, "raw", "RGBA", 0, 1)
        if format == "png":
            # Only use the metadata kwarg if pnginfo is not set, because the
            # semantics of duplicate keys in pnginfo is unclear.
            if "pnginfo" in pil_kwargs:
                if metadata:
                    _api.warn_external("'metadata' is overridden by the "
                                       "'pnginfo' entry in 'pil_kwargs'.")
            else:
                metadata = {
                    "Software": (f"Matplotlib version{mpl.__version__}, "
                                 f"https://matplotlib.org/"),
                    **(metadata if metadata is not None else {}),
                }
                pil_kwargs["pnginfo"] = pnginfo = PIL.PngImagePlugin.PngInfo()
                for k, v in metadata.items():
                    if v is not None:
                        pnginfo.add_text(k, v)
        if format in ["jpg", "jpeg"]:
            format = "jpeg"  # Pillow doesn't recognize "jpg".
            facecolor = mpl.rcParams["savefig.facecolor"]
            if cbook._str_equal(facecolor, "auto"):
                facecolor = mpl.rcParams["figure.facecolor"]
            color = tuple(int(x * 255) for x in mcolors.to_rgb(facecolor))
            background = PIL.Image.new("RGB", pil_shape, color)
            background.paste(image, image)
            image = background
        pil_kwargs.setdefault("format", format)
        pil_kwargs.setdefault("dpi", (dpi, dpi))
        image.save(fname, **pil_kwargs)
```
### 3 - lib/matplotlib/figure.py:

Start line: 3208, End line: 3320

```python
@_docstring.interpd
class Figure(FigureBase):

    def savefig(self, fname, *, transparent=None, **kwargs):
        """
        Save the current figure.

        Call signature::

          savefig(fname, *, dpi='figure', format=None, metadata=None,
                  bbox_inches=None, pad_inches=0.1,
                  facecolor='auto', edgecolor='auto',
                  backend=None, **kwargs
                 )

        The available output formats depend on the backend being used.

        Parameters
        ----------
        fname : str or path-like or binary file-like
            A path, or a Python file-like object, or
            possibly some backend-dependent object such as
            `matplotlib.backends.backend_pdf.PdfPages`.

            If *format* is set, it determines the output format, and the file
            is saved as *fname*.  Note that *fname* is used verbatim, and there
            is no attempt to make the extension, if any, of *fname* match
            *format*, and no extension is appended.

            If *format* is not set, then the format is inferred from the
            extension of *fname*, if there is one.  If *format* is not
            set and *fname* has no extension, then the file is saved with
            :rc:`savefig.format` and the appropriate extension is appended to
            *fname*.

        Other Parameters
        ----------------
        dpi : float or 'figure', default: :rc:`savefig.dpi`
            The resolution in dots per inch.  If 'figure', use the figure's
            dpi value.

        format : str
            The file format, e.g. 'png', 'pdf', 'svg', ... The behavior when
            this is unset is documented under *fname*.

        metadata : dict, optional
            Key/value pairs to store in the image metadata. The supported keys
            and defaults depend on the image format and backend:

            - 'png' with Agg backend: See the parameter ``metadata`` of
              `~.FigureCanvasAgg.print_png`.
            - 'pdf' with pdf backend: See the parameter ``metadata`` of
              `~.backend_pdf.PdfPages`.
            - 'svg' with svg backend: See the parameter ``metadata`` of
              `~.FigureCanvasSVG.print_svg`.
            - 'eps' and 'ps' with PS backend: Only 'Creator' is supported.

        bbox_inches : str or `.Bbox`, default: :rc:`savefig.bbox`
            Bounding box in inches: only the given portion of the figure is
            saved.  If 'tight', try to figure out the tight bbox of the figure.

        pad_inches : float or 'layout', default: :rc:`savefig.pad_inches`
            Amount of padding in inches around the figure when bbox_inches is
            'tight'. If 'layout' use the padding from the constrained or
            compressed layout engine; ignored if one of those engines is not in
            use.

        facecolor : color or 'auto', default: :rc:`savefig.facecolor`
            The facecolor of the figure.  If 'auto', use the current figure
            facecolor.

        edgecolor : color or 'auto', default: :rc:`savefig.edgecolor`
            The edgecolor of the figure.  If 'auto', use the current figure
            edgecolor.

        backend : str, optional
            Use a non-default backend to render the file, e.g. to render a
            png file with the "cairo" backend rather than the default "agg",
            or a pdf file with the "pgf" backend rather than the default
            "pdf".  Note that the default backend is normally sufficient.  See
            :ref:`the-builtin-backends` for a list of valid backends for each
            file format.  Custom backends can be referenced as "module://...".

        orientation : {'landscape', 'portrait'}
            Currently only supported by the postscript backend.

        papertype : str
            One of 'letter', 'legal', 'executive', 'ledger', 'a0' through
            'a10', 'b0' through 'b10'. Only supported for postscript
            output.

        transparent : bool
            If *True*, the Axes patches will all be transparent; the
            Figure patch will also be transparent unless *facecolor*
            and/or *edgecolor* are specified via kwargs.

            If *False* has no effect and the color of the Axes and
            Figure patches are unchanged (unless the Figure patch
            is specified via the *facecolor* and/or *edgecolor* keyword
            arguments in which case those colors are used).

            The transparency of these patches will be restored to their
            original values upon exit of this function.

            This is useful, for example, for displaying
            a plot on top of a colored background on a web page.

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        pil_kwargs : dict, optional
            Additional keyword arguments that are passed to
            `PIL.Image.Image.save` when saving the figure.

        """
        # ... other code
```
### 4 - lib/matplotlib/figure.py:

Start line: 3322, End line: 3359

```python
@_docstring.interpd
class Figure(FigureBase):

    def savefig(self, fname, *, transparent=None, **kwargs):

        kwargs.setdefault('dpi', mpl.rcParams['savefig.dpi'])
        if transparent is None:
            transparent = mpl.rcParams['savefig.transparent']

        with ExitStack() as stack:
            if transparent:
                def _recursively_make_subfig_transparent(exit_stack, subfig):
                    exit_stack.enter_context(
                        subfig.patch._cm_set(
                            facecolor="none", edgecolor="none"))
                    for ax in subfig.axes:
                        exit_stack.enter_context(
                            ax.patch._cm_set(
                                facecolor="none", edgecolor="none"))
                    for sub_subfig in subfig.subfigs:
                        _recursively_make_subfig_transparent(
                            exit_stack, sub_subfig)

                def _recursively_make_axes_transparent(exit_stack, ax):
                    exit_stack.enter_context(
                        ax.patch._cm_set(facecolor="none", edgecolor="none"))
                    for child_ax in ax.child_axes:
                        exit_stack.enter_context(
                            child_ax.patch._cm_set(
                                facecolor="none", edgecolor="none"))
                    for child_childax in ax.child_axes:
                        _recursively_make_axes_transparent(
                            exit_stack, child_childax)

                kwargs.setdefault('facecolor', 'none')
                kwargs.setdefault('edgecolor', 'none')
                # set subfigure to appear transparent in printed image
                for subfig in self.subfigs:
                    _recursively_make_subfig_transparent(stack, subfig)
                # set axes to be transparent
                for ax in self.axes:
                    _recursively_make_axes_transparent(stack, ax)
            self.canvas.print_figure(fname, **kwargs)
```
### 5 - lib/matplotlib/pyplot.py:

Start line: 1019, End line: 1036

```python
@_copy_docstring_and_deprecators(Figure.savefig)
def savefig(*args, **kwargs):
    fig = gcf()
    res = fig.savefig(*args, **kwargs)
    fig.canvas.draw_idle()  # Need this if 'transparent=True', to reset colors.
    return res


## Putting things in figures ##


def figlegend(*args, **kwargs):
    return gcf().legend(*args, **kwargs)
if Figure.legend.__doc__:
    figlegend.__doc__ = Figure.legend.__doc__ \
        .replace(" legend(", " figlegend(") \
        .replace("fig.legend(", "plt.figlegend(") \
        .replace("ax.plot(", "plt.plot(")
```
### 6 - lib/matplotlib/pyplot.py:

Start line: 2286, End line: 2296

```python
################# REMAINING CONTENT GENERATED BY boilerplate.py ##############


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Figure.figimage)
def figimage(
        X, xo=0, yo=0, alpha=None, norm=None, cmap=None, vmin=None,
        vmax=None, origin=None, resize=False, **kwargs):
    return gcf().figimage(
        X, xo=xo, yo=yo, alpha=alpha, norm=norm, cmap=cmap, vmin=vmin,
        vmax=vmax, origin=origin, resize=resize, **kwargs)
```
### 7 - lib/matplotlib/backends/backend_svg.py:

Start line: 1289, End line: 1368

```python
class FigureCanvasSVG(FigureCanvasBase):
    filetypes = {'svg': 'Scalable Vector Graphics',
                 'svgz': 'Scalable Vector Graphics'}

    fixed_dpi = 72

    def print_svg(self, filename, *, bbox_inches_restore=None, metadata=None):
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict[str, Any], optional
            Metadata in the SVG file defined as key-value pairs of strings,
            datetimes, or lists of strings, e.g., ``{'Creator': 'My software',
            'Contributor': ['Me', 'My Friend'], 'Title': 'Awesome'}``.

            The standard keys and their value types are:

            * *str*: ``'Coverage'``, ``'Description'``, ``'Format'``,
              ``'Identifier'``, ``'Language'``, ``'Relation'``, ``'Source'``,
              ``'Title'``, and ``'Type'``.
            * *str* or *list of str*: ``'Contributor'``, ``'Creator'``,
              ``'Keywords'``, ``'Publisher'``, and ``'Rights'``.
            * *str*, *date*, *datetime*, or *tuple* of same: ``'Date'``. If a
              non-*str*, then it will be formatted as ISO 8601.

            Values have been predefined for ``'Creator'``, ``'Date'``,
            ``'Format'``, and ``'Type'``. They can be removed by setting them
            to `None`.

            Information is encoded as `Dublin Core Metadata`__.

            .. _DC: https://www.dublincore.org/specifications/dublin-core/

            __ DC_
        """
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as fh:
            if not cbook.file_requires_unicode(fh):
                fh = codecs.getwriter('utf-8')(fh)
            dpi = self.figure.dpi
            self.figure.dpi = 72
            width, height = self.figure.get_size_inches()
            w, h = width * 72, height * 72
            renderer = MixedModeRenderer(
                self.figure, width, height, dpi,
                RendererSVG(w, h, fh, image_dpi=dpi, metadata=metadata),
                bbox_inches_restore=bbox_inches_restore)
            self.figure.draw(renderer)
            renderer.finalize()

    def print_svgz(self, filename, **kwargs):
        with cbook.open_file_cm(filename, "wb") as fh, \
                gzip.GzipFile(mode='w', fileobj=fh) as gzipwriter:
            return self.print_svg(gzipwriter, **kwargs)

    def get_default_filetype(self):
        return 'svg'

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()


FigureManagerSVG = FigureManagerBase


svgProlog = """\
<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
"""


@_Backend.export
class _BackendSVG(_Backend):
    backend_version = mpl.__version__
    FigureCanvas = FigureCanvasSVG
```
### 8 - lib/matplotlib/backend_bases.py:

Start line: 1650, End line: 2409

```python
class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level figure instance.
    """

    # Set to one of {"qt", "gtk3", "gtk4", "wx", "tk", "macosx"} if an
    # interactive framework is required, or None otherwise.
    required_interactive_framework = None

    # The manager class instantiated by new_manager.
    # (This is defined as a classproperty because the manager class is
    # currently defined *after* the canvas class, but one could also assign
    # ``FigureCanvasBase.manager_class = FigureManagerBase``
    # after defining both classes.)
    manager_class = _api.classproperty(lambda cls: FigureManagerBase)

    events = [
        'resize_event',
        'draw_event',
        'key_press_event',
        'key_release_event',
        'button_press_event',
        'button_release_event',
        'scroll_event',
        'motion_notify_event',
        'pick_event',
        'figure_enter_event',
        'figure_leave_event',
        'axes_enter_event',
        'axes_leave_event',
        'close_event'
    ]

    fixed_dpi = None

    filetypes = _default_filetypes

    @_api.classproperty
    def supports_blit(cls):
        """If this Canvas sub-class supports blitting."""
        return (hasattr(cls, "copy_from_bbox")
                and hasattr(cls, "restore_region"))

    def __init__(self, figure=None):
        from matplotlib.figure import Figure
        self._fix_ipython_backend2gui()
        self._is_idle_drawing = True
        self._is_saving = False
        if figure is None:
            figure = Figure()
        figure.set_canvas(self)
        self.figure = figure
        self.manager = None
        self.widgetlock = widgets.LockDraw()
        self._button = None  # the button pressed
        self._key = None  # the key pressed
        self._lastx, self._lasty = None, None
        self.mouse_grabber = None  # the Axes currently grabbing mouse
        self.toolbar = None  # NavigationToolbar2 will set me
        self._is_idle_drawing = False
        # We don't want to scale up the figure DPI more than once.
        figure._original_dpi = figure.dpi
        self._device_pixel_ratio = 1
        super().__init__()  # Typically the GUI widget init (if any).

    callbacks = property(lambda self: self.figure._canvas_callbacks)
    button_pick_id = property(lambda self: self.figure._button_pick_id)
    scroll_pick_id = property(lambda self: self.figure._scroll_pick_id)

    @classmethod
    @functools.cache
    def _fix_ipython_backend2gui(cls):
        # Fix hard-coded module -> toolkit mapping in IPython (used for
        # `ipython --auto`).  This cannot be done at import time due to
        # ordering issues, so we do it when creating a canvas, and should only
        # be done once per class (hence the `cache`).
        if sys.modules.get("IPython") is None:
            return
        import IPython
        ip = IPython.get_ipython()
        if not ip:
            return
        from IPython.core import pylabtools as pt
        if (not hasattr(pt, "backend2gui")
                or not hasattr(ip, "enable_matplotlib")):
            # In case we ever move the patch to IPython and remove these APIs,
            # don't break on our side.
            return
        backend2gui_rif = {
            "qt": "qt",
            "gtk3": "gtk3",
            "gtk4": "gtk4",
            "wx": "wx",
            "macosx": "osx",
        }.get(cls.required_interactive_framework)
        if backend2gui_rif:
            if _is_non_interactive_terminal_ipython(ip):
                ip.enable_gui(backend2gui_rif)

    @classmethod
    def new_manager(cls, figure, num):
        """
        Create a new figure manager for *figure*, using this canvas class.

        Notes
        -----
        This method should not be reimplemented in subclasses.  If
        custom manager creation logic is needed, please reimplement
        ``FigureManager.create_with_canvas``.
        """
        return cls.manager_class.create_with_canvas(cls, figure, num)

    @contextmanager
    def _idle_draw_cntx(self):
        self._is_idle_drawing = True
        try:
            yield
        finally:
            self._is_idle_drawing = False

    def is_saving(self):
        """
        Return whether the renderer is in the process of saving
        to a file, rather than rendering for an on-screen buffer.
        """
        return self._is_saving

    @_api.deprecated("3.6", alternative="canvas.figure.pick")
    def pick(self, mouseevent):
        if not self.widgetlock.locked():
            self.figure.pick(mouseevent)

    def blit(self, bbox=None):
        """Blit the canvas in bbox (default entire canvas)."""

    def resize(self, w, h):
        """
        UNUSED: Set the canvas size in pixels.

        Certain backends may implement a similar method internally, but this is
        not a requirement of, nor is it used by, Matplotlib itself.
        """
        # The entire method is actually deprecated, but we allow pass-through
        # to a parent class to support e.g. QWidget.resize.
        if hasattr(super(), "resize"):
            return super().resize(w, h)
        else:
            _api.warn_deprecated("3.6", name="resize", obj_type="method",
                                 alternative="FigureManagerBase.resize")

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('draw_event', DrawEvent(...))"))
    def draw_event(self, renderer):
        """Pass a `DrawEvent` to all functions connected to ``draw_event``."""
        s = 'draw_event'
        event = DrawEvent(s, self, renderer)
        self.callbacks.process(s, event)

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('resize_event', ResizeEvent(...))"))
    def resize_event(self):
        """
        Pass a `ResizeEvent` to all functions connected to ``resize_event``.
        """
        s = 'resize_event'
        event = ResizeEvent(s, self)
        self.callbacks.process(s, event)
        self.draw_idle()

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('close_event', CloseEvent(...))"))
    def close_event(self, guiEvent=None):
        """
        Pass a `CloseEvent` to all functions connected to ``close_event``.
        """
        s = 'close_event'
        try:
            event = CloseEvent(s, self, guiEvent=guiEvent)
            self.callbacks.process(s, event)
        except (TypeError, AttributeError):
            pass
            # Suppress the TypeError when the python session is being killed.
            # It may be that a better solution would be a mechanism to
            # disconnect all callbacks upon shutdown.
            # AttributeError occurs on OSX with qt4agg upon exiting
            # with an open window; 'callbacks' attribute no longer exists.

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('key_press_event', KeyEvent(...))"))
    def key_press_event(self, key, guiEvent=None):
        """
        Pass a `KeyEvent` to all functions connected to ``key_press_event``.
        """
        self._key = key
        s = 'key_press_event'
        event = KeyEvent(
            s, self, key, self._lastx, self._lasty, guiEvent=guiEvent)
        self.callbacks.process(s, event)

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('key_release_event', KeyEvent(...))"))
    def key_release_event(self, key, guiEvent=None):
        """
        Pass a `KeyEvent` to all functions connected to ``key_release_event``.
        """
        s = 'key_release_event'
        event = KeyEvent(
            s, self, key, self._lastx, self._lasty, guiEvent=guiEvent)
        self.callbacks.process(s, event)
        self._key = None

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('pick_event', PickEvent(...))"))
    def pick_event(self, mouseevent, artist, **kwargs):
        """
        Callback processing for pick events.

        This method will be called by artists who are picked and will
        fire off `PickEvent` callbacks registered listeners.

        Note that artists are not pickable by default (see
        `.Artist.set_picker`).
        """
        s = 'pick_event'
        event = PickEvent(s, self, mouseevent, artist,
                          guiEvent=mouseevent.guiEvent,
                          **kwargs)
        self.callbacks.process(s, event)

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('scroll_event', MouseEvent(...))"))
    def scroll_event(self, x, y, step, guiEvent=None):
        """
        Callback processing for scroll events.

        Backend derived classes should call this function on any
        scroll wheel event.  (*x*, *y*) are the canvas coords ((0, 0) is lower
        left).  button and key are as defined in `MouseEvent`.

        This method will call all functions connected to the 'scroll_event'
        with a `MouseEvent` instance.
        """
        if step >= 0:
            self._button = 'up'
        else:
            self._button = 'down'
        s = 'scroll_event'
        mouseevent = MouseEvent(s, self, x, y, self._button, self._key,
                                step=step, guiEvent=guiEvent)
        self.callbacks.process(s, mouseevent)

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('button_press_event', MouseEvent(...))"))
    def button_press_event(self, x, y, button, dblclick=False, guiEvent=None):
        """
        Callback processing for mouse button press events.

        Backend derived classes should call this function on any mouse
        button press.  (*x*, *y*) are the canvas coords ((0, 0) is lower left).
        button and key are as defined in `MouseEvent`.

        This method will call all functions connected to the
        'button_press_event' with a `MouseEvent` instance.
        """
        self._button = button
        s = 'button_press_event'
        mouseevent = MouseEvent(s, self, x, y, button, self._key,
                                dblclick=dblclick, guiEvent=guiEvent)
        self.callbacks.process(s, mouseevent)

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('button_release_event', MouseEvent(...))"))
    def button_release_event(self, x, y, button, guiEvent=None):
        """
        Callback processing for mouse button release events.

        Backend derived classes should call this function on any mouse
        button release.

        This method will call all functions connected to the
        'button_release_event' with a `MouseEvent` instance.

        Parameters
        ----------
        x : float
            The canvas coordinates where 0=left.
        y : float
            The canvas coordinates where 0=bottom.
        guiEvent
            The native UI event that generated the Matplotlib event.
        """
        s = 'button_release_event'
        event = MouseEvent(s, self, x, y, button, self._key, guiEvent=guiEvent)
        self.callbacks.process(s, event)
        self._button = None

    # Also remove _lastx, _lasty when this goes away.
    @_api.deprecated("3.6", alternative=(
        "callbacks.process('motion_notify_event', MouseEvent(...))"))
    def motion_notify_event(self, x, y, guiEvent=None):
        """
        Callback processing for mouse movement events.

        Backend derived classes should call this function on any
        motion-notify-event.

        This method will call all functions connected to the
        'motion_notify_event' with a `MouseEvent` instance.

        Parameters
        ----------
        x : float
            The canvas coordinates where 0=left.
        y : float
            The canvas coordinates where 0=bottom.
        guiEvent
            The native UI event that generated the Matplotlib event.
        """
        self._lastx, self._lasty = x, y
        s = 'motion_notify_event'
        event = MouseEvent(s, self, x, y, self._button, self._key,
                           guiEvent=guiEvent)
        self.callbacks.process(s, event)

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('leave_notify_event', LocationEvent(...))"))
    def leave_notify_event(self, guiEvent=None):
        """
        Callback processing for the mouse cursor leaving the canvas.

        Backend derived classes should call this function when leaving
        canvas.

        Parameters
        ----------
        guiEvent
            The native UI event that generated the Matplotlib event.
        """
        self.callbacks.process('figure_leave_event', LocationEvent._lastevent)
        LocationEvent._lastevent = None
        self._lastx, self._lasty = None, None

    @_api.deprecated("3.6", alternative=(
        "callbacks.process('enter_notify_event', LocationEvent(...))"))
    def enter_notify_event(self, guiEvent=None, *, xy):
        """
        Callback processing for the mouse cursor entering the canvas.

        Backend derived classes should call this function when entering
        canvas.

        Parameters
        ----------
        guiEvent
            The native UI event that generated the Matplotlib event.
        xy : (float, float)
            The coordinate location of the pointer when the canvas is entered.
        """
        self._lastx, self._lasty = x, y = xy
        event = LocationEvent('figure_enter_event', self, x, y, guiEvent)
        self.callbacks.process('figure_enter_event', event)

    def inaxes(self, xy):
        """
        Return the topmost visible `~.axes.Axes` containing the point *xy*.

        Parameters
        ----------
        xy : (float, float)
            (x, y) pixel positions from left/bottom of the canvas.

        Returns
        -------
        `~matplotlib.axes.Axes` or None
            The topmost visible Axes containing the point, or None if there
            is no Axes at the point.
        """
        axes_list = [a for a in self.figure.get_axes()
                     if a.patch.contains_point(xy) and a.get_visible()]
        if axes_list:
            axes = cbook._topmost_artist(axes_list)
        else:
            axes = None

        return axes

    def grab_mouse(self, ax):
        """
        Set the child `~.axes.Axes` which is grabbing the mouse events.

        Usually called by the widgets themselves. It is an error to call this
        if the mouse is already grabbed by another Axes.
        """
        if self.mouse_grabber not in (None, ax):
            raise RuntimeError("Another Axes already grabs mouse input")
        self.mouse_grabber = ax

    def release_mouse(self, ax):
        """
        Release the mouse grab held by the `~.axes.Axes` *ax*.

        Usually called by the widgets. It is ok to call this even if *ax*
        doesn't have the mouse grab currently.
        """
        if self.mouse_grabber is ax:
            self.mouse_grabber = None

    def set_cursor(self, cursor):
        """
        Set the current cursor.

        This may have no effect if the backend does not display anything.

        If required by the backend, this method should trigger an update in
        the backend event loop after the cursor is set, as this method may be
        called e.g. before a long-running task during which the GUI is not
        updated.

        Parameters
        ----------
        cursor : `.Cursors`
            The cursor to display over the canvas. Note: some backends may
            change the cursor for the entire window.
        """

    def draw(self, *args, **kwargs):
        """
        Render the `.Figure`.

        This method must walk the artist tree, even if no output is produced,
        because it triggers deferred work that users may want to access
        before saving output to disk. For example computing limits,
        auto-limits, and tick values.
        """

    def draw_idle(self, *args, **kwargs):
        """
        Request a widget redraw once control returns to the GUI event loop.

        Even if multiple calls to `draw_idle` occur before control returns
        to the GUI event loop, the figure will only be rendered once.

        Notes
        -----
        Backends may choose to override the method and implement their own
        strategy to prevent multiple renderings.

        """
        if not self._is_idle_drawing:
            with self._idle_draw_cntx():
                self.draw(*args, **kwargs)

    @property
    def device_pixel_ratio(self):
        """
        The ratio of physical to logical pixels used for the canvas on screen.

        By default, this is 1, meaning physical and logical pixels are the same
        size. Subclasses that support High DPI screens may set this property to
        indicate that said ratio is different. All Matplotlib interaction,
        unless working directly with the canvas, remains in logical pixels.

        """
        return self._device_pixel_ratio

    def _set_device_pixel_ratio(self, ratio):
        """
        Set the ratio of physical to logical pixels used for the canvas.

        Subclasses that support High DPI screens can set this property to
        indicate that said ratio is different. The canvas itself will be
        created at the physical size, while the client side will use the
        logical size. Thus the DPI of the Figure will change to be scaled by
        this ratio. Implementations that support High DPI screens should use
        physical pixels for events so that transforms back to Axes space are
        correct.

        By default, this is 1, meaning physical and logical pixels are the same
        size.

        Parameters
        ----------
        ratio : float
            The ratio of logical to physical pixels used for the canvas.

        Returns
        -------
        bool
            Whether the ratio has changed. Backends may interpret this as a
            signal to resize the window, repaint the canvas, or change any
            other relevant properties.
        """
        if self._device_pixel_ratio == ratio:
            return False
        # In cases with mixed resolution displays, we need to be careful if the
        # device pixel ratio changes - in this case we need to resize the
        # canvas accordingly. Some backends provide events that indicate a
        # change in DPI, but those that don't will update this before drawing.
        dpi = ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)
        self._device_pixel_ratio = ratio
        return True

    def get_width_height(self, *, physical=False):
        """
        Return the figure width and height in integral points or pixels.

        When the figure is used on High DPI screens (and the backend supports
        it), the truncation to integers occurs after scaling by the device
        pixel ratio.

        Parameters
        ----------
        physical : bool, default: False
            Whether to return true physical pixels or logical pixels. Physical
            pixels may be used by backends that support HiDPI, but still
            configure the canvas using its actual size.

        Returns
        -------
        width, height : int
            The size of the figure, in points or pixels, depending on the
            backend.
        """
        return tuple(int(size / (1 if physical else self.device_pixel_ratio))
                     for size in self.figure.bbox.max)

    @classmethod
    def get_supported_filetypes(cls):
        """Return dict of savefig file formats supported by this backend."""
        return cls.filetypes

    @classmethod
    def get_supported_filetypes_grouped(cls):
        """
        Return a dict of savefig file formats supported by this backend,
        where the keys are a file type name, such as 'Joint Photographic
        Experts Group', and the values are a list of filename extensions used
        for that filetype, such as ['jpg', 'jpeg'].
        """
        groupings = {}
        for ext, name in cls.filetypes.items():
            groupings.setdefault(name, []).append(ext)
            groupings[name].sort()
        return groupings

    @contextmanager
    def _switch_canvas_and_return_print_method(self, fmt, backend=None):
        """
        Context manager temporarily setting the canvas for saving the figure::

            with canvas._switch_canvas_and_return_print_method(fmt, backend) \\
                    as print_method:
                # ``print_method`` is a suitable ``print_{fmt}`` method, and
                # the figure's canvas is temporarily switched to the method's
                # canvas within the with... block.  ``print_method`` is also
                # wrapped to suppress extra kwargs passed by ``print_figure``.

        Parameters
        ----------
        fmt : str
            If *backend* is None, then determine a suitable canvas class for
            saving to format *fmt* -- either the current canvas class, if it
            supports *fmt*, or whatever `get_registered_canvas_class` returns;
            switch the figure canvas to that canvas class.
        backend : str or None, default: None
            If not None, switch the figure canvas to the ``FigureCanvas`` class
            of the given backend.
        """
        canvas = None
        if backend is not None:
            # Return a specific canvas class, if requested.
            canvas_class = (
                importlib.import_module(cbook._backend_module_name(backend))
                .FigureCanvas)
            if not hasattr(canvas_class, f"print_{fmt}"):
                raise ValueError(
                    f"The {backend!r} backend does not support {fmt} output")
        elif hasattr(self, f"print_{fmt}"):
            # Return the current canvas if it supports the requested format.
            canvas = self
            canvas_class = None  # Skip call to switch_backends.
        else:
            # Return a default canvas for the requested format, if it exists.
            canvas_class = get_registered_canvas_class(fmt)
        if canvas_class:
            canvas = self.switch_backends(canvas_class)
        if canvas is None:
            raise ValueError(
                "Format {!r} is not supported (supported formats: {})".format(
                    fmt, ", ".join(sorted(self.get_supported_filetypes()))))
        meth = getattr(canvas, f"print_{fmt}")
        mod = (meth.func.__module__
               if hasattr(meth, "func")  # partialmethod, e.g. backend_wx.
               else meth.__module__)
        if mod.startswith(("matplotlib.", "mpl_toolkits.")):
            optional_kws = {  # Passed by print_figure for other renderers.
                "dpi", "facecolor", "edgecolor", "orientation",
                "bbox_inches_restore"}
            skip = optional_kws - {*inspect.signature(meth).parameters}
            print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
                *args, **{k: v for k, v in kwargs.items() if k not in skip}))
        else:  # Let third-parties do as they see fit.
            print_method = meth
        try:
            yield print_method
        finally:
            self.figure.canvas = self

    def print_figure(
            self, filename, dpi=None, facecolor=None, edgecolor=None,
            orientation='portrait', format=None, *,
            bbox_inches=None, pad_inches=None, bbox_extra_artists=None,
            backend=None, **kwargs):
        """
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you'll probably want to override this on
        hardcopy.

        Parameters
        ----------
        filename : str or path-like or file-like
            The file where the figure is saved.

        dpi : float, default: :rc:`savefig.dpi`
            The dots per inch to save the figure in.

        facecolor : color or 'auto', default: :rc:`savefig.facecolor`
            The facecolor of the figure.  If 'auto', use the current figure
            facecolor.

        edgecolor : color or 'auto', default: :rc:`savefig.edgecolor`
            The edgecolor of the figure.  If 'auto', use the current figure
            edgecolor.

        orientation : {'landscape', 'portrait'}, default: 'portrait'
            Only currently applies to PostScript printing.

        format : str, optional
            Force a specific file format. If not given, the format is inferred
            from the *filename* extension, and if that fails from
            :rc:`savefig.format`.

        bbox_inches : 'tight' or `.Bbox`, default: :rc:`savefig.bbox`
            Bounding box in inches: only the given portion of the figure is
            saved.  If 'tight', try to figure out the tight bbox of the figure.

        pad_inches : float or 'layout', default: :rc:`savefig.pad_inches`
            Amount of padding in inches around the figure when bbox_inches is
            'tight'. If 'layout' use the padding from the constrained or
            compressed layout engine; ignored if one of those engines is not in
            use.

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        backend : str, optional
            Use a non-default backend to render the file, e.g. to render a
            png file with the "cairo" backend rather than the default "agg",
            or a pdf file with the "pgf" backend rather than the default
            "pdf".  Note that the default backend is normally sufficient.  See
            :ref:`the-builtin-backends` for a list of valid backends for each
            file format.  Custom backends can be referenced as "module://...".
        """
        if format is None:
            # get format from filename, or from backend's default filetype
            if isinstance(filename, os.PathLike):
                filename = os.fspath(filename)
            if isinstance(filename, str):
                format = os.path.splitext(filename)[1][1:]
            if format is None or format == '':
                format = self.get_default_filetype()
                if isinstance(filename, str):
                    filename = filename.rstrip('.') + '.' + format
        format = format.lower()

        if dpi is None:
            dpi = rcParams['savefig.dpi']
        if dpi == 'figure':
            dpi = getattr(self.figure, '_original_dpi', self.figure.dpi)

        # Remove the figure manager, if any, to avoid resizing the GUI widget.
        with cbook._setattr_cm(self, manager=None), \
             self._switch_canvas_and_return_print_method(format, backend) \
                 as print_method, \
             cbook._setattr_cm(self.figure, dpi=dpi), \
             cbook._setattr_cm(self.figure.canvas, _device_pixel_ratio=1), \
             cbook._setattr_cm(self.figure.canvas, _is_saving=True), \
             ExitStack() as stack:

            for prop in ["facecolor", "edgecolor"]:
                color = locals()[prop]
                if color is None:
                    color = rcParams[f"savefig.{prop}"]
                if not cbook._str_equal(color, "auto"):
                    stack.enter_context(self.figure._cm_set(**{prop: color}))

            if bbox_inches is None:
                bbox_inches = rcParams['savefig.bbox']

            layout_engine = self.figure.get_layout_engine()
            if layout_engine is not None or bbox_inches == "tight":
                # we need to trigger a draw before printing to make sure
                # CL works.  "tight" also needs a draw to get the right
                # locations:
                renderer = _get_renderer(
                    self.figure,
                    functools.partial(
                        print_method, orientation=orientation)
                )
                with getattr(renderer, "_draw_disabled", nullcontext)():
                    self.figure.draw(renderer)

            if bbox_inches:
                if bbox_inches == "tight":
                    bbox_inches = self.figure.get_tightbbox(
                        renderer, bbox_extra_artists=bbox_extra_artists)
                    if (isinstance(layout_engine, ConstrainedLayoutEngine) and
                            pad_inches == "layout"):
                        h_pad = layout_engine.get()["h_pad"]
                        w_pad = layout_engine.get()["w_pad"]
                    else:
                        if pad_inches in [None, "layout"]:
                            pad_inches = rcParams['savefig.pad_inches']
                        h_pad = w_pad = pad_inches
                    bbox_inches = bbox_inches.padded(w_pad, h_pad)

                # call adjust_bbox to save only the given area
                restore_bbox = _tight_bbox.adjust_bbox(
                    self.figure, bbox_inches, self.figure.canvas.fixed_dpi)

                _bbox_inches_restore = (bbox_inches, restore_bbox)
            else:
                _bbox_inches_restore = None

            # we have already done layout above, so turn it off:
            stack.enter_context(self.figure._cm_set(layout_engine='none'))
            try:
                # _get_renderer may change the figure dpi (as vector formats
                # force the figure dpi to 72), so we need to set it again here.
                with cbook._setattr_cm(self.figure, dpi=dpi):
                    result = print_method(
                        filename,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        orientation=orientation,
                        bbox_inches_restore=_bbox_inches_restore,
                        **kwargs)
            finally:
                if bbox_inches and restore_bbox:
                    restore_bbox()

            return result
```
### 9 - lib/matplotlib/backends/backend_pgf.py:

Start line: 128, End line: 163

```python
def _metadata_to_str(key, value):
    """Convert metadata key/value to a form that hyperref accepts."""
    if isinstance(value, datetime.datetime):
        value = _datetime_to_pdf(value)
    elif key == 'Trapped':
        value = value.name.decode('ascii')
    else:
        value = str(value)
    return f'{key}={{{value}}}'


def make_pdf_to_png_converter():
    """Return a function that converts a pdf file to a png file."""
    try:
        mpl._get_executable_info("pdftocairo")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(
            ["pdftocairo", "-singlefile", "-transp", "-png", "-r", "%d" % dpi,
             pdffile, os.path.splitext(pngfile)[0]],
            stderr=subprocess.STDOUT)
    try:
        gs_info = mpl._get_executable_info("gs")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(
            [gs_info.executable,
             '-dQUIET', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dNOPROMPT',
             '-dUseCIEColor', '-dTextAlphaBits=4',
             '-dGraphicsAlphaBits=4', '-dDOINTERPOLATE',
             '-sDEVICE=pngalpha', '-sOutputFile=%s' % pngfile,
             '-r%d' % dpi, pdffile],
            stderr=subprocess.STDOUT)
    raise RuntimeError("No suitable pdf to png renderer found.")
```
### 10 - lib/matplotlib/backends/backend_cairo.py:

Start line: 447, End line: 501

```python
class FigureCanvasCairo(FigureCanvasBase):

    def _save(self, fmt, fobj, *, orientation='portrait'):
        # save PDF/PS/SVG

        dpi = 72
        self.figure.dpi = dpi
        w_in, h_in = self.figure.get_size_inches()
        width_in_points, height_in_points = w_in * dpi, h_in * dpi

        if orientation == 'landscape':
            width_in_points, height_in_points = (
                height_in_points, width_in_points)

        if fmt == 'ps':
            if not hasattr(cairo, 'PSSurface'):
                raise RuntimeError('cairo has not been compiled with PS '
                                   'support enabled')
            surface = cairo.PSSurface(fobj, width_in_points, height_in_points)
        elif fmt == 'pdf':
            if not hasattr(cairo, 'PDFSurface'):
                raise RuntimeError('cairo has not been compiled with PDF '
                                   'support enabled')
            surface = cairo.PDFSurface(fobj, width_in_points, height_in_points)
        elif fmt in ('svg', 'svgz'):
            if not hasattr(cairo, 'SVGSurface'):
                raise RuntimeError('cairo has not been compiled with SVG '
                                   'support enabled')
            if fmt == 'svgz':
                if isinstance(fobj, str):
                    fobj = gzip.GzipFile(fobj, 'wb')
                else:
                    fobj = gzip.GzipFile(None, 'wb', fileobj=fobj)
            surface = cairo.SVGSurface(fobj, width_in_points, height_in_points)
        else:
            raise ValueError(f"Unknown format: {fmt!r}")

        self._renderer.dpi = self.figure.dpi
        self._renderer.set_context(cairo.Context(surface))
        ctx = self._renderer.gc.ctx

        if orientation == 'landscape':
            ctx.rotate(np.pi / 2)
            ctx.translate(0, -height_in_points)
            # Perhaps add an '%%Orientation: Landscape' comment?

        self.figure.draw(self._renderer)

        ctx.show_page()
        surface.finish()
        if fmt == 'svgz':
            fobj.close()

    print_pdf = functools.partialmethod(_save, "pdf")
    print_ps = functools.partialmethod(_save, "ps")
    print_svg = functools.partialmethod(_save, "svg")
    print_svgz = functools.partialmethod(_save, "svgz")
```
### 11 - lib/matplotlib/figure.py:

Start line: 1, End line: 69

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

Figures are typically created using pyplot methods `~.pyplot.figure`,
`~.pyplot.subplots`, and `~.pyplot.subplot_mosaic`.

.. plot::
    :include-source:

    fig, ax = plt.subplots(figsize=(2, 2), facecolor='lightskyblue',
                           layout='constrained')
    fig.suptitle('Figure')
    ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')

Some situations call for directly instantiating a `~.figure.Figure` class,
usually inside an application of some sort (see :ref:`user_interfaces` for a
list of examples) .  More information about Figures can be found at
:ref:`figure_explanation`.
"""

from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
    Artist, allow_rasterization, _finalize_rasterization)
from matplotlib.backend_bases import (
    DrawEvent, FigureCanvasBase, NonGuiException, MouseButton, _get_renderer)
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
    ConstrainedLayoutEngine, TightLayoutEngine, LayoutEngine,
    PlaceHolderLayoutEngine
)
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)

_log = logging.getLogger(__name__)


def _stale_figure_callback(self, val):
    if self.figure:
        self.figure.stale = val
```
### 17 - lib/matplotlib/figure.py:

Start line: 2362, End line: 2532

```python
@_docstring.interpd
class Figure(FigureBase):

    def __init__(self,
                 figsize=None,
                 dpi=None,
                 *,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 subplotpars=None,  # rc figure.subplot.*
                 tight_layout=None,  # rc figure.autolayout
                 constrained_layout=None,  # rc figure.constrained_layout.use
                 layout=None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        figsize : 2-tuple of floats, default: :rc:`figure.figsize`
            Figure dimension ``(width, height)`` in inches.

        dpi : float, default: :rc:`figure.dpi`
            Dots per inch.

        facecolor : default: :rc:`figure.facecolor`
            The figure patch facecolor.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure background patch.

        subplotpars : `SubplotParams`
            Subplot parameters. If not given, the default subplot
            parameters :rc:`figure.subplot.*` are used.

        tight_layout : bool or dict, default: :rc:`figure.autolayout`
            Whether to use the tight layout mechanism. See `.set_tight_layout`.

            .. admonition:: Discouraged

                The use of this parameter is discouraged. Please use
                ``layout='tight'`` instead for the common case of
                ``tight_layout=True`` and use `.set_tight_layout` otherwise.

        constrained_layout : bool, default: :rc:`figure.constrained_layout.use`
            This is equal to ``layout='constrained'``.

            .. admonition:: Discouraged

                The use of this parameter is discouraged. Please use
                ``layout='constrained'`` instead.

        layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, \
        efault: None
            The layout mechanism for positioning of plot elements to avoid
            overlapping Axes decorations (labels, ticks, etc). Note that
            layout managers can have significant performance penalties.

            - 'constrained': The constrained layout solver adjusts axes sizes
              to avoid overlapping axes decorations.  Can handle complex plot
              layouts and colorbars, and is thus recommended.

              See :doc:`/tutorials/intermediate/constrainedlayout_guide`
              for examples.

            - 'compressed': uses the same algorithm as 'constrained', but
              removes extra space between fixed-aspect-ratio Axes.  Best for
              simple grids of axes.

            - 'tight': Use the tight layout mechanism. This is a relatively
              simple algorithm that adjusts the subplot parameters so that
              decorations do not overlap. See `.Figure.set_tight_layout` for
              further details.

            - 'none': Do not use a layout engine.

            - A `.LayoutEngine` instance. Builtin layout classes are
              `.ConstrainedLayoutEngine` and `.TightLayoutEngine`, more easily
              accessible by 'constrained' and 'tight'.  Passing an instance
              allows third parties to provide their own layout engine.

            If not given, fall back to using the parameters *tight_layout* and
            *constrained_layout*, including their config defaults
            :rc:`figure.autolayout` and :rc:`figure.constrained_layout.use`.

        Other Parameters
        ----------------
        **kwargs : `.Figure` properties, optional

            %(Figure:kwdoc)s
        """
        super().__init__(**kwargs)
        self._layout_engine = None

        if layout is not None:
            if (tight_layout is not None):
                _api.warn_external(
                    "The Figure parameters 'layout' and 'tight_layout' cannot "
                    "be used together. Please use 'layout' only.")
            if (constrained_layout is not None):
                _api.warn_external(
                    "The Figure parameters 'layout' and 'constrained_layout' "
                    "cannot be used together. Please use 'layout' only.")
            self.set_layout_engine(layout=layout)
        elif tight_layout is not None:
            if constrained_layout is not None:
                _api.warn_external(
                    "The Figure parameters 'tight_layout' and "
                    "'constrained_layout' cannot be used together. Please use "
                    "'layout' parameter")
            self.set_layout_engine(layout='tight')
            if isinstance(tight_layout, dict):
                self.get_layout_engine().set(**tight_layout)
        elif constrained_layout is not None:
            if isinstance(constrained_layout, dict):
                self.set_layout_engine(layout='constrained')
                self.get_layout_engine().set(**constrained_layout)
            elif constrained_layout:
                self.set_layout_engine(layout='constrained')

        else:
            # everything is None, so use default:
            self.set_layout_engine(layout=layout)

        self._fig_callbacks = cbook.CallbackRegistry(signals=["dpi_changed"])
        # Callbacks traditionally associated with the canvas (and exposed with
        # a proxy property), but that actually need to be on the figure for
        # pickling.
        self._canvas_callbacks = cbook.CallbackRegistry(
            signals=FigureCanvasBase.events)
        connect = self._canvas_callbacks._connect_picklable
        self._mouse_key_ids = [
            connect('key_press_event', backend_bases._key_handler),
            connect('key_release_event', backend_bases._key_handler),
            connect('key_release_event', backend_bases._key_handler),
            connect('button_press_event', backend_bases._mouse_handler),
            connect('button_release_event', backend_bases._mouse_handler),
            connect('scroll_event', backend_bases._mouse_handler),
            connect('motion_notify_event', backend_bases._mouse_handler),
        ]
        self._button_pick_id = connect('button_press_event', self.pick)
        self._scroll_pick_id = connect('scroll_event', self.pick)

        if figsize is None:
            figsize = mpl.rcParams['figure.figsize']
        if dpi is None:
            dpi = mpl.rcParams['figure.dpi']
        if facecolor is None:
            facecolor = mpl.rcParams['figure.facecolor']
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        if not np.isfinite(figsize).all() or (np.array(figsize) < 0).any():
            raise ValueError('figure size must be positive finite not '
                             f'{figsize}')
        self.bbox_inches = Bbox.from_bounds(0, 0, *figsize)

        self.dpi_scale_trans = Affine2D().scale(dpi)
        # do not use property as it will trigger
        self._dpi = dpi
        self.bbox = TransformedBbox(self.bbox_inches, self.dpi_scale_trans)
        self.figbbox = self.bbox
        self.transFigure = BboxTransformTo(self.bbox)
        self.transSubfigure = self.transFigure
        # ... other code
```
### 27 - lib/matplotlib/image.py:

Start line: 1396, End line: 1416

```python
class FigureImage(_ImageBase):

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        fac = renderer.dpi/self.figure.dpi
        # fac here is to account for pdf, eps, svg backends where
        # figure.dpi is set to 72.  This means we need to scale the
        # image (using magnification) and offset it appropriately.
        bbox = Bbox([[self.ox/fac, self.oy/fac],
                     [(self.ox/fac + self._A.shape[1]),
                     (self.oy/fac + self._A.shape[0])]])
        width, height = self.figure.get_size_inches()
        width *= renderer.dpi
        height *= renderer.dpi
        clip = Bbox([[0, 0], [width, height]])
        return self._make_image(
            self._A, bbox, bbox, clip, magnification=magnification / fac,
            unsampled=unsampled, round_to_pixel_border=False)

    def set_data(self, A):
        """Set the image array."""
        cm.ScalarMappable.set_array(self, A)
        self.stale = True
```
### 29 - lib/matplotlib/figure.py:

Start line: 2534, End line: 2554

```python
@_docstring.interpd
class Figure(FigureBase):

    def __init__(self,
                 figsize=None,
                 dpi=None,
                 *,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 subplotpars=None,  # rc figure.subplot.*
                 tight_layout=None,  # rc figure.autolayout
                 constrained_layout=None,  # rc figure.constrained_layout.use
                 layout=None,
                 **kwargs
                 ):
        # ... other code

        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1, visible=frameon,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            # Don't let the figure patch influence bbox calculation.
            in_layout=False)
        self._set_artist_props(self.patch)
        self.patch.set_antialiased(False)

        FigureCanvasBase(self)  # Set self.canvas.

        if subplotpars is None:
            subplotpars = SubplotParams()

        self.subplotpars = subplotpars

        self._axstack = _AxesStack()  # track all figure axes and current axes
        self.clear()

    def pick(self, mouseevent):
        if not self.canvas.widgetlock.locked():
            super().pick(mouseevent)
```
