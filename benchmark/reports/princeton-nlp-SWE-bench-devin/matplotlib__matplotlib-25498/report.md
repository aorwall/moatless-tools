# matplotlib__matplotlib-25498

| **matplotlib/matplotlib** | `78bf53caacbb5ce0dc7aa73f07a74c99f1ed919b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 16627 |
| **Any found context length** | 12116 |
| **Avg pos** | 64.0 |
| **Min pos** | 27 |
| **Max pos** | 37 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/colorbar.py b/lib/matplotlib/colorbar.py
--- a/lib/matplotlib/colorbar.py
+++ b/lib/matplotlib/colorbar.py
@@ -301,11 +301,6 @@ def __init__(self, ax, mappable=None, *, cmap=None,
         if mappable is None:
             mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
 
-        # Ensure the given mappable's norm has appropriate vmin and vmax
-        # set even if mappable.draw has not yet been called.
-        if mappable.get_array() is not None:
-            mappable.autoscale_None()
-
         self.mappable = mappable
         cmap = mappable.cmap
         norm = mappable.norm
@@ -1101,7 +1096,10 @@ def _process_values(self):
             b = np.hstack((b, b[-1] + 1))
 
         # transform from 0-1 to vmin-vmax:
+        if self.mappable.get_array() is not None:
+            self.mappable.autoscale_None()
         if not self.norm.scaled():
+            # If we still aren't scaled after autoscaling, use 0, 1 as default
             self.norm.vmin = 0
             self.norm.vmax = 1
         self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/colorbar.py | 304 | 308 | 37 | 1 | 16627
| lib/matplotlib/colorbar.py | 1104 | 1104 | 27 | 1 | 12116


## Problem Statement

```
Update colorbar after changing mappable.norm
How can I update a colorbar, after I changed the norm instance of the colorbar?

`colorbar.update_normal(mappable)` has now effect and `colorbar.update_bruteforce(mappable)` throws a `ZeroDivsionError`-Exception.

Consider this example:

\`\`\` python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

img = 10**np.random.normal(1, 1, size=(50, 50))

fig, ax = plt.subplots(1, 1)
plot = ax.imshow(img, cmap='gray')
cb = fig.colorbar(plot, ax=ax)
plot.norm = LogNorm()
cb.update_normal(plot)  # no effect
cb.update_bruteforce(plot)  # throws ZeroDivisionError
plt.show()
\`\`\`

Output for `cb.update_bruteforce(plot)`:

\`\`\`
Traceback (most recent call last):
  File "test_norm.py", line 12, in <module>
    cb.update_bruteforce(plot)
  File "/home/maxnoe/.local/anaconda3/lib/python3.4/site-packages/matplotlib/colorbar.py", line 967, in update_bruteforce
    self.draw_all()
  File "/home/maxnoe/.local/anaconda3/lib/python3.4/site-packages/matplotlib/colorbar.py", line 342, in draw_all
    self._process_values()
  File "/home/maxnoe/.local/anaconda3/lib/python3.4/site-packages/matplotlib/colorbar.py", line 664, in _process_values
    b = self.norm.inverse(self._uniform_y(self.cmap.N + 1))
  File "/home/maxnoe/.local/anaconda3/lib/python3.4/site-packages/matplotlib/colors.py", line 1011, in inverse
    return vmin * ma.power((vmax / vmin), val)
ZeroDivisionError: division by zero
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/colorbar.py** | 493 | 527| 310 | 310 | 14311 | 
| 2 | 2 galleries/users_explain/colors/colormapnorms.py | 1 | 95| 905 | 1215 | 18006 | 
| 3 | 2 galleries/users_explain/colors/colormapnorms.py | 307 | 336| 316 | 1531 | 18006 | 
| 4 | 3 galleries/examples/images_contours_and_fields/colormap_normalizations.py | 1 | 75| 733 | 2264 | 19481 | 
| 5 | 3 galleries/users_explain/colors/colormapnorms.py | 233 | 306| 761 | 3025 | 19481 | 
| 6 | 3 galleries/users_explain/colors/colormapnorms.py | 173 | 232| 750 | 3775 | 19481 | 
| 7 | 4 galleries/examples/images_contours_and_fields/colormap_normalizations_symlognorm.py | 1 | 86| 822 | 4597 | 20303 | 
| 8 | 4 galleries/users_explain/colors/colormapnorms.py | 96 | 172| 833 | 5430 | 20303 | 
| 9 | 5 lib/matplotlib/cm.py | 592 | 619| 209 | 5639 | 25968 | 
| 10 | 5 galleries/examples/images_contours_and_fields/colormap_normalizations.py | 107 | 145| 429 | 6068 | 25968 | 
| 11 | 6 lib/matplotlib/colors.py | 1898 | 1937| 328 | 6396 | 50077 | 
| 12 | 6 galleries/users_explain/colors/colormapnorms.py | 339 | 350| 128 | 6524 | 50077 | 
| 13 | 6 lib/matplotlib/cm.py | 621 | 665| 319 | 6843 | 50077 | 
| 14 | 6 galleries/examples/images_contours_and_fields/colormap_normalizations.py | 91 | 104| 152 | 6995 | 50077 | 
| 15 | 6 lib/matplotlib/colors.py | 2015 | 2065| 547 | 7542 | 50077 | 
| 16 | 6 lib/matplotlib/cm.py | 668 | 695| 360 | 7902 | 50077 | 
| 17 | 6 lib/matplotlib/colors.py | 1565 | 1586| 192 | 8094 | 50077 | 
| 18 | 7 galleries/users_explain/colors/colorbar_only.py | 91 | 134| 358 | 8452 | 51183 | 
| 19 | 7 galleries/examples/images_contours_and_fields/colormap_normalizations.py | 77 | 89| 160 | 8612 | 51183 | 
| 20 | 7 lib/matplotlib/colors.py | 1469 | 1477| 121 | 8733 | 51183 | 
| 21 | 7 galleries/users_explain/colors/colorbar_only.py | 1 | 89| 748 | 9481 | 51183 | 
| 22 | **7 lib/matplotlib/colorbar.py** | 1167 | 1199| 339 | 9820 | 51183 | 
| 23 | 7 lib/matplotlib/colors.py | 1690 | 1709| 281 | 10101 | 51183 | 
| 24 | 7 lib/matplotlib/colors.py | 1711 | 1744| 391 | 10492 | 51183 | 
| 25 | 7 lib/matplotlib/colors.py | 2068 | 2128| 556 | 11048 | 51183 | 
| 26 | **7 lib/matplotlib/colorbar.py** | 529 | 579| 484 | 11532 | 51183 | 
| **-> 27 <-** | **7 lib/matplotlib/colorbar.py** | 1065 | 1116| 584 | 12116 | 51183 | 
| 28 | 7 lib/matplotlib/colors.py | 1676 | 1688| 184 | 12300 | 51183 | 
| 29 | 8 galleries/examples/scales/power_norm.py | 1 | 50| 340 | 12640 | 51523 | 
| 30 | 8 lib/matplotlib/colors.py | 1864 | 1896| 287 | 12927 | 51523 | 
| 31 | 8 lib/matplotlib/colors.py | 1385 | 1435| 474 | 13401 | 51523 | 
| 32 | 9 galleries/examples/images_contours_and_fields/colormap_interactive_adjustment.py | 1 | 34| 306 | 13707 | 51829 | 
| 33 | 9 lib/matplotlib/colors.py | 1197 | 1268| 509 | 14216 | 51829 | 
| 34 | 9 lib/matplotlib/cm.py | 538 | 562| 182 | 14398 | 51829 | 
| 35 | 9 lib/matplotlib/colors.py | 1480 | 1563| 753 | 15151 | 51829 | 
| 36 | 9 lib/matplotlib/colors.py | 1832 | 1861| 208 | 15359 | 51829 | 
| **-> 37 <-** | **9 lib/matplotlib/colorbar.py** | 281 | 441| 1268 | 16627 | 51829 | 
| 38 | 9 lib/matplotlib/colors.py | 1940 | 2013| 766 | 17393 | 51829 | 
| 39 | 9 lib/matplotlib/colors.py | 1656 | 1674| 236 | 17629 | 51829 | 
| 40 | 10 galleries/examples/color/colorbar_basics.py | 1 | 59| 506 | 18135 | 52335 | 
| 41 | **10 lib/matplotlib/colorbar.py** | 195 | 279| 796 | 18931 | 52335 | 
| 42 | 11 galleries/users_explain/axes/colorbar_placement.py | 1 | 85| 764 | 19695 | 53230 | 
| 43 | 12 lib/matplotlib/pyplot.py | 2124 | 2137| 138 | 19833 | 81926 | 
| 44 | 12 lib/matplotlib/colors.py | 1351 | 1382| 292 | 20125 | 81926 | 
| 45 | 12 lib/matplotlib/cm.py | 404 | 424| 191 | 20316 | 81926 | 
| 46 | 12 lib/matplotlib/colors.py | 1747 | 1791| 382 | 20698 | 81926 | 
| 47 | **12 lib/matplotlib/colorbar.py** | 603 | 625| 246 | 20944 | 81926 | 
| 48 | 12 lib/matplotlib/colors.py | 1305 | 1349| 377 | 21321 | 81926 | 
| 49 | 12 lib/matplotlib/cm.py | 345 | 370| 245 | 21566 | 81926 | 
| 50 | **12 lib/matplotlib/colorbar.py** | 1330 | 1351| 248 | 21814 | 81926 | 
| 51 | 13 galleries/examples/specialty_plots/advanced_hillshading.py | 34 | 54| 231 | 22045 | 82536 | 
| 52 | 14 galleries/users_explain/colors/colormaps.py | 207 | 264| 655 | 22700 | 87335 | 
| 53 | **14 lib/matplotlib/colorbar.py** | 1150 | 1165| 191 | 22891 | 87335 | 
| 54 | 15 galleries/examples/images_contours_and_fields/pcolor_demo.py | 1 | 97| 793 | 23684 | 88385 | 
| 55 | 15 galleries/users_explain/colors/colormaps.py | 408 | 439| 435 | 24119 | 88385 | 
| 56 | 16 lib/matplotlib/image.py | 420 | 552| 1709 | 25828 | 105629 | 
| 57 | 17 galleries/examples/color/custom_cmap.py | 99 | 166| 637 | 26465 | 108507 | 
| 58 | 17 lib/matplotlib/colors.py | 1794 | 1829| 330 | 26795 | 108507 | 
| 59 | **17 lib/matplotlib/colorbar.py** | 1 | 27| 206 | 27001 | 108507 | 
| 60 | 17 lib/matplotlib/colors.py | 752 | 842| 772 | 27773 | 108507 | 
| 61 | 17 galleries/examples/color/custom_cmap.py | 242 | 283| 381 | 28154 | 108507 | 
| 62 | 17 galleries/users_explain/colors/colormaps.py | 372 | 405| 387 | 28541 | 108507 | 
| 63 | 17 lib/matplotlib/colors.py | 1451 | 1467| 200 | 28741 | 108507 | 
| 64 | 17 lib/matplotlib/colors.py | 2639 | 2694| 490 | 29231 | 108507 | 
| 65 | 17 galleries/users_explain/colors/colormaps.py | 265 | 371| 1204 | 30435 | 108507 | 
| 66 | 18 galleries/examples/specialty_plots/leftventricle_bullseye.py | 94 | 155| 697 | 31132 | 110112 | 
| 67 | **18 lib/matplotlib/colorbar.py** | 714 | 738| 283 | 31415 | 110112 | 
| 68 | 18 galleries/users_explain/colors/colormaps.py | 98 | 121| 286 | 31701 | 110112 | 
| 69 | **18 lib/matplotlib/colorbar.py** | 29 | 114| 961 | 32662 | 110112 | 
| 70 | **18 lib/matplotlib/colorbar.py** | 1298 | 1328| 267 | 32929 | 110112 | 
| 71 | **18 lib/matplotlib/colorbar.py** | 1351 | 1383| 385 | 33314 | 110112 | 
| 72 | 18 galleries/users_explain/colors/colormaps.py | 1 | 96| 778 | 34092 | 110112 | 
| 73 | 19 galleries/examples/images_contours_and_fields/contourf_log.py | 1 | 63| 494 | 34586 | 110606 | 
| 74 | 19 galleries/examples/images_contours_and_fields/pcolor_demo.py | 98 | 125| 257 | 34843 | 110606 | 
| 75 | 19 galleries/examples/color/custom_cmap.py | 168 | 241| 740 | 35583 | 110606 | 
| 76 | 19 lib/matplotlib/image.py | 1172 | 1206| 257 | 35840 | 110606 | 
| 77 | 19 lib/matplotlib/cm.py | 564 | 590| 144 | 35984 | 110606 | 
| 78 | 20 lib/matplotlib/figure.py | 1264 | 1289| 358 | 36342 | 140094 | 
| 79 | 21 galleries/users_explain/colors/colormap-manipulation.py | 1 | 83| 736 | 37078 | 143229 | 
| 80 | 21 galleries/users_explain/axes/colorbar_placement.py | 86 | 100| 131 | 37209 | 143229 | 
| 81 | **21 lib/matplotlib/colorbar.py** | 627 | 644| 234 | 37443 | 143229 | 
| 82 | **21 lib/matplotlib/colorbar.py** | 1201 | 1233| 259 | 37702 | 143229 | 
| 83 | 22 galleries/examples/images_contours_and_fields/pcolormesh_levels.py | 1 | 85| 793 | 38495 | 144439 | 
| 84 | 22 lib/matplotlib/colors.py | 1589 | 1635| 457 | 38952 | 144439 | 
| 85 | 22 lib/matplotlib/colors.py | 1270 | 1303| 291 | 39243 | 144439 | 
| 86 | 22 galleries/users_explain/colors/colormap-manipulation.py | 214 | 293| 847 | 40090 | 144439 | 
| 87 | 22 galleries/users_explain/colors/colormap-manipulation.py | 294 | 319| 176 | 40266 | 144439 | 
| 88 | 22 galleries/users_explain/colors/colormaps.py | 124 | 205| 1051 | 41317 | 144439 | 
| 89 | 22 lib/matplotlib/colors.py | 1638 | 1654| 163 | 41480 | 144439 | 
| 90 | 22 galleries/users_explain/colors/colormap-manipulation.py | 113 | 187| 794 | 42274 | 144439 | 
| 91 | 23 galleries/examples/ticks/colorbar_tick_labelling_demo.py | 1 | 48| 305 | 42579 | 144744 | 
| 92 | 24 galleries/users_explain/quick_start.py | 489 | 575| 1016 | 43595 | 150751 | 
| 93 | 25 lib/matplotlib/_cm.py | 843 | 868| 522 | 44117 | 179194 | 
| 94 | 25 lib/matplotlib/cm.py | 373 | 402| 294 | 44411 | 179194 | 
| 95 | 25 lib/matplotlib/colors.py | 998 | 1015| 216 | 44627 | 179194 | 
| 96 | 26 galleries/examples/animation/bayes_update.py | 44 | 67| 193 | 44820 | 179645 | 
| 97 | 26 galleries/examples/images_contours_and_fields/pcolormesh_levels.py | 87 | 133| 417 | 45237 | 179645 | 
| 98 | 27 galleries/examples/lines_bars_and_markers/multicolored_line.py | 1 | 50| 460 | 45697 | 180105 | 
| 99 | 28 galleries/examples/images_contours_and_fields/image_masked.py | 1 | 83| 739 | 46436 | 180844 | 
| 100 | 28 lib/matplotlib/colors.py | 1437 | 1449| 135 | 46571 | 180844 | 
| 101 | **28 lib/matplotlib/colorbar.py** | 1235 | 1271| 414 | 46985 | 180844 | 
| 102 | 29 galleries/examples/axes_grid1/simple_colorbar.py | 1 | 23| 135 | 47120 | 180979 | 
| 103 | 30 galleries/examples/color/colormap_reference.py | 48 | 64| 233 | 47353 | 181971 | 
| 104 | 30 lib/matplotlib/colors.py | 2438 | 2451| 175 | 47528 | 181971 | 
| 105 | 30 lib/matplotlib/_cm.py | 1 | 49| 726 | 48254 | 181971 | 
| 106 | **30 lib/matplotlib/colorbar.py** | 1028 | 1063| 263 | 48517 | 181971 | 
| 107 | 30 galleries/users_explain/colors/colormap-manipulation.py | 201 | 212| 160 | 48677 | 181971 | 
| 108 | 30 lib/matplotlib/_cm.py | 1213 | 1243| 510 | 49187 | 181971 | 
| 109 | 30 galleries/examples/animation/bayes_update.py | 27 | 42| 138 | 49325 | 181971 | 
| 110 | **30 lib/matplotlib/colorbar.py** | 117 | 143| 254 | 49579 | 181971 | 
| 111 | **30 lib/matplotlib/colorbar.py** | 820 | 876| 498 | 50077 | 181971 | 


### Hint

```
You have run into a big bug in imshow, not colorbar.  As a workaround, after setting `plot.norm`, call `plot.autoscale()`.  Then the `update_bruteforce` will work.
When the norm is changed, it should pick up the vmax, vmin values from the autoscaling; but this is not happening.  Actually, it's worse than that; it fails even if the norm is set as a kwarg in the call to imshow. I haven't looked beyond that to see why.  I've confirmed the problem with master.

In ipython using `%matplotlib` setting the norm the first time works, changing it back later to
`Normalize()` or something other blows up:

\`\`\`
--> 199         self.pixels.autoscale()
    200         self.update(force=True)
    201 

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cm.py in autoscale(self)
    323             raise TypeError('You must first set_array for mappable')
    324         self.norm.autoscale(self._A)
--> 325         self.changed()
    326 
    327     def autoscale_None(self):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cm.py in changed(self)
    357         callbackSM listeners to the 'changed' signal
    358         """
--> 359         self.callbacksSM.process('changed', self)
    360 
    361         for key in self.update_dict:

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cbook.py in process(self, s, *args, **kwargs)
    560             for cid, proxy in list(six.iteritems(self.callbacks[s])):
    561                 try:
--> 562                     proxy(*args, **kwargs)
    563                 except ReferenceError:
    564                     self._remove_proxy(proxy)

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cbook.py in __call__(self, *args, **kwargs)
    427             mtd = self.func
    428         # invoke the callable and return the result
--> 429         return mtd(*args, **kwargs)
    430 
    431     def __eq__(self, other):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in on_mappable_changed(self, mappable)
    915         self.set_cmap(mappable.get_cmap())
    916         self.set_clim(mappable.get_clim())
--> 917         self.update_normal(mappable)
    918 
    919     def add_lines(self, CS, erase=True):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in update_normal(self, mappable)
    946         or contour plot to which this colorbar belongs is changed.
    947         '''
--> 948         self.draw_all()
    949         if isinstance(self.mappable, contour.ContourSet):
    950             CS = self.mappable

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in draw_all(self)
    346         X, Y = self._mesh()
    347         C = self._values[:, np.newaxis]
--> 348         self._config_axes(X, Y)
    349         if self.filled:
    350             self._add_solids(X, Y, C)

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in _config_axes(self, X, Y)
    442         ax.add_artist(self.patch)
    443 
--> 444         self.update_ticks()
    445 
    446     def _set_label(self):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in update_ticks(self)
    371         """
    372         ax = self.ax
--> 373         ticks, ticklabels, offset_string = self._ticker()
    374         if self.orientation == 'vertical':
    375             ax.yaxis.set_ticks(ticks)

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in _ticker(self)
    592         formatter.set_data_interval(*intv)
    593 
--> 594         b = np.array(locator())
    595         if isinstance(locator, ticker.LogLocator):
    596             eps = 1e-10

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/ticker.py in __call__(self)
   1533         'Return the locations of the ticks'
   1534         vmin, vmax = self.axis.get_view_interval()
-> 1535         return self.tick_values(vmin, vmax)
   1536 
   1537     def tick_values(self, vmin, vmax):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/ticker.py in tick_values(self, vmin, vmax)
   1551             if vmin <= 0.0 or not np.isfinite(vmin):
   1552                 raise ValueError(
-> 1553                     "Data has no positive values, and therefore can not be "
   1554                     "log-scaled.")
   1555 

ValueError: Data has no positive values, and therefore can not be log-scaled.
\`\`\`

Any news on this? Why does setting the norm back to a linear norm blow up if there are negative values?

\`\`\` python
In [2]: %matplotlib
Using matplotlib backend: Qt4Agg

In [3]: # %load minimal_norm.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm


x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
z = np.random.normal(0, 5, size=x.shape)

fig = plt.figure()
img = plt.pcolor(x, y, z, cmap='viridis')
cbar = plt.colorbar(img)
   ...: 

In [4]: img.norm = LogNorm()

In [5]: img.autoscale()

In [7]: cbar.update_bruteforce(img)

In [8]: img.norm = Normalize()

In [9]: img.autoscale()
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-9-e26279d12b00> in <module>()
----> 1 img.autoscale()

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cm.py in autoscale(self)
    323             raise TypeError('You must first set_array for mappable')
    324         self.norm.autoscale(self._A)
--> 325         self.changed()
    326 
    327     def autoscale_None(self):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cm.py in changed(self)
    357         callbackSM listeners to the 'changed' signal
    358         """
--> 359         self.callbacksSM.process('changed', self)
    360 
    361         for key in self.update_dict:

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cbook.py in process(self, s, *args, **kwargs)
    561             for cid, proxy in list(six.iteritems(self.callbacks[s])):
    562                 try:
--> 563                     proxy(*args, **kwargs)
    564                 except ReferenceError:
    565                     self._remove_proxy(proxy)

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/cbook.py in __call__(self, *args, **kwargs)
    428             mtd = self.func
    429         # invoke the callable and return the result
--> 430         return mtd(*args, **kwargs)
    431 
    432     def __eq__(self, other):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in on_mappable_changed(self, mappable)
    915         self.set_cmap(mappable.get_cmap())
    916         self.set_clim(mappable.get_clim())
--> 917         self.update_normal(mappable)
    918 
    919     def add_lines(self, CS, erase=True):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in update_normal(self, mappable)
    946         or contour plot to which this colorbar belongs is changed.
    947         '''
--> 948         self.draw_all()
    949         if isinstance(self.mappable, contour.ContourSet):
    950             CS = self.mappable

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in draw_all(self)
    346         X, Y = self._mesh()
    347         C = self._values[:, np.newaxis]
--> 348         self._config_axes(X, Y)
    349         if self.filled:
    350             self._add_solids(X, Y, C)

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in _config_axes(self, X, Y)
    442         ax.add_artist(self.patch)
    443 
--> 444         self.update_ticks()
    445 
    446     def _set_label(self):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in update_ticks(self)
    371         """
    372         ax = self.ax
--> 373         ticks, ticklabels, offset_string = self._ticker()
    374         if self.orientation == 'vertical':
    375             ax.yaxis.set_ticks(ticks)

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/colorbar.py in _ticker(self)
    592         formatter.set_data_interval(*intv)
    593 
--> 594         b = np.array(locator())
    595         if isinstance(locator, ticker.LogLocator):
    596             eps = 1e-10

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/ticker.py in __call__(self)
   1536         'Return the locations of the ticks'
   1537         vmin, vmax = self.axis.get_view_interval()
-> 1538         return self.tick_values(vmin, vmax)
   1539 
   1540     def tick_values(self, vmin, vmax):

/home/maxnoe/.local/anaconda3/envs/ctapipe/lib/python3.5/site-packages/matplotlib/ticker.py in tick_values(self, vmin, vmax)
   1554             if vmin <= 0.0 or not np.isfinite(vmin):
   1555                 raise ValueError(
-> 1556                     "Data has no positive values, and therefore can not be "
   1557                     "log-scaled.")
   1558 

ValueError: Data has no positive values, and therefore can not be log-scaled
\`\`\`

This issue has been marked "inactive" because it has been 365 days since the last comment. If this issue is still present in recent Matplotlib releases, or the feature request is still wanted, please leave a comment and this label will be removed. If there are no updates in another 30 days, this issue will be automatically closed, but you are free to re-open or create a new issue if needed. We value issue reports, and this procedure is meant to help us resurface and prioritize issues that have not been addressed yet, not make them disappear.  Thanks for your help!
```

## Patch

```diff
diff --git a/lib/matplotlib/colorbar.py b/lib/matplotlib/colorbar.py
--- a/lib/matplotlib/colorbar.py
+++ b/lib/matplotlib/colorbar.py
@@ -301,11 +301,6 @@ def __init__(self, ax, mappable=None, *, cmap=None,
         if mappable is None:
             mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
 
-        # Ensure the given mappable's norm has appropriate vmin and vmax
-        # set even if mappable.draw has not yet been called.
-        if mappable.get_array() is not None:
-            mappable.autoscale_None()
-
         self.mappable = mappable
         cmap = mappable.cmap
         norm = mappable.norm
@@ -1101,7 +1096,10 @@ def _process_values(self):
             b = np.hstack((b, b[-1] + 1))
 
         # transform from 0-1 to vmin-vmax:
+        if self.mappable.get_array() is not None:
+            self.mappable.autoscale_None()
         if not self.norm.scaled():
+            # If we still aren't scaled after autoscaling, use 0, 1 as default
             self.norm.vmin = 0
             self.norm.vmax = 1
         self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_colorbar.py b/lib/matplotlib/tests/test_colorbar.py
--- a/lib/matplotlib/tests/test_colorbar.py
+++ b/lib/matplotlib/tests/test_colorbar.py
@@ -657,6 +657,12 @@ def test_colorbar_scale_reset():
 
     assert cbar.outline.get_edgecolor() == mcolors.to_rgba('red')
 
+    # log scale with no vmin/vmax set should scale to the data if there
+    # is a mappable already associated with the colorbar, not (0, 1)
+    pcm.norm = LogNorm()
+    assert pcm.norm.vmin == z.min()
+    assert pcm.norm.vmax == z.max()
+
 
 def test_colorbar_get_ticks_2():
     plt.rcParams['_internal.classic_mode'] = False

```


## Code snippets

### 1 - lib/matplotlib/colorbar.py:

Start line: 493, End line: 527

```python
@_docstring.interpd
class Colorbar:

    def update_normal(self, mappable):
        """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
        _log.debug('colorbar update normal %r %r', mappable.norm, self.norm)
        self.mappable = mappable
        self.set_alpha(mappable.get_alpha())
        self.cmap = mappable.cmap
        if mappable.norm != self.norm:
            self.norm = mappable.norm
            self._reset_locator_formatter_scale()

        self._draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
        self.stale = True

    @_api.deprecated("3.6", alternative="fig.draw_without_rendering()")
    def draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
        self._draw_all()
```
### 2 - galleries/users_explain/colors/colormapnorms.py:

Start line: 1, End line: 95

```python
"""

.. redirect-from:: /tutorials/colors/colormapnorms

.. _colormapnorms:

Colormap Normalization
======================

Objects that use colormaps by default linearly map the colors in the
colormap from data values *vmin* to *vmax*.  For example::

    pcm = ax.pcolormesh(x, y, Z, vmin=-1., vmax=1., cmap='RdBu_r')

will map the data in *Z* linearly from -1 to +1, so *Z=0* will
give a color at the center of the colormap *RdBu_r* (white in this
case).

Matplotlib does this mapping in two steps, with a normalization from
the input data to [0, 1] occurring first, and then mapping onto the
indices in the colormap.  Normalizations are classes defined in the
:func:`matplotlib.colors` module.  The default, linear normalization
is :func:`matplotlib.colors.Normalize`.

Artists that map data to color pass the arguments *vmin* and *vmax* to
construct a :func:`matplotlib.colors.Normalize` instance, then call it:

.. code-block:: pycon

   >>> import matplotlib as mpl
   >>> norm = mpl.colors.Normalize(vmin=-1, vmax=1)
   >>> norm(0)
   0.5

However, there are sometimes cases where it is useful to map data to
colormaps in a non-linear fashion.

Logarithmic
-----------

One of the most common transformations is to plot data by taking its logarithm
(to the base-10).  This transformation is useful to display changes across
disparate scales.  Using `.colors.LogNorm` normalizes the data via
:math:`log_{10}`.  In the example below, there are two bumps, one much smaller
than the other. Using `.colors.LogNorm`, the shape and location of each bump
can clearly be seen:

"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors

N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]

# A low hump with a spike coming out of the top right.  Needs to have
# z/colour axis on a log scale, so we see both hump and spike. A linear
# scale only shows the spike.
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
Z = Z1 + 50 * Z2

fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolor(X, Y, Z,
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='max')
plt.show()

# %%
# Centered
# --------
#
# In many cases, data is symmetrical around a center, for example, positive and
# negative anomalies around a center 0. In this case, we would like the center
# to be mapped to 0.5 and the datapoint with the largest deviation from the
# center to be mapped to 1.0, if its value is greater than the center, or 0.0
# otherwise. The norm `.colors.CenteredNorm` creates such a mapping
# automatically. It is well suited to be combined with a divergent colormap
# which uses different colors edges that meet in the center at an unsaturated
# color.
#
# If the center of symmetry is different from 0, it can be set with the
# *vcenter* argument. For logarithmic scaling on both sides of the center, see
# `.colors.SymLogNorm` below; to apply a different mapping above and below the
# center, use `.colors.TwoSlopeNorm` below.

delta = 0.1
```
### 3 - galleries/users_explain/colors/colormapnorms.py:

Start line: 307, End line: 336

```python
pcm = ax.pcolormesh(X, Y, Z1, norm=norm, cmap='PuBu_r', shading='auto')
ax.set_title('FuncNorm(x)')
fig.colorbar(pcm, shrink=0.6)
plt.show()

# %%
# Custom normalization: Manually implement two linear ranges
# ----------------------------------------------------------
#
# The `.TwoSlopeNorm` described above makes a useful example for
# defining your own norm.  Note for the colorbar to work, you must
# define an inverse for your norm:


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
```
### 4 - galleries/examples/images_contours_and_fields/colormap_normalizations.py:

Start line: 1, End line: 75

```python
"""
=======================
Colormap normalizations
=======================

Demonstration of using norm to map colormaps onto data in non-linear ways.

.. redirect-from:: /gallery/userdemo/colormap_normalizations
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors

# %%
# Lognorm: Instead of pcolor log10(Z1) you can have colorbars that have
# the exponential labels using a norm.

N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]

# A low hump with a spike coming out of the top.  Needs to have
# z/colour axis on a log scale, so we see both hump and spike.
# A linear scale only shows the spike.

Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
Z = Z1 + 50 * Z2

fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolor(X, Y, Z,
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='max')


# %%
# PowerNorm: Here a power-law trend in X partially obscures a rectified
# sine wave in Y. We can remove the power law using a PowerNorm.

X, Y = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.)) * X**2

fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolormesh(X, Y, Z1, norm=colors.PowerNorm(gamma=1. / 2.),
                       cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolormesh(X, Y, Z1, cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='max')

# %%
# SymLogNorm: two humps, one negative and one positive, The positive
# with 5-times the amplitude. Linearly, you cannot see detail in the
# negative hump.  Here we logarithmically scale the positive and
# negative data separately.
#
# Note that colorbar labels do not come out looking very good.

X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z = 5 * np.exp(-X**2 - Y**2)

fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=-1.0, vmax=1.0, base=10),
                       cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both')
```
### 5 - galleries/users_explain/colors/colormapnorms.py:

Start line: 233, End line: 306

```python
pcm = ax[2].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical')
ax[2].set_title('BoundaryNorm: nonuniform')

# With out-of-bounds colors:
bounds = np.linspace(-1.5, 1.5, 7)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
pcm = ax[3].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
# The colorbar inherits the "extend" argument from BoundaryNorm.
fig.colorbar(pcm, ax=ax[3], orientation='vertical')
ax[3].set_title('BoundaryNorm: extend="both"')
plt.show()

# %%
# TwoSlopeNorm: Different mapping on either side of a center
# ----------------------------------------------------------
#
# Sometimes we want to have a different colormap on either side of a
# conceptual center point, and we want those two colormaps to have
# different linear scales.  An example is a topographic map where the land
# and ocean have a center at zero, but land typically has a greater
# elevation range than the water has depth range, and they are often
# represented by a different colormap.

dem = cbook.get_sample_data('topobathy.npz')
topo = dem['topo']
longitude = dem['longitude']
latitude = dem['latitude']

fig, ax = plt.subplots()
# make a colormap that has land and ocean clearly delineated and of the
# same length (256 + 256)
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = colors.LinearSegmentedColormap.from_list(
    'terrain_map', all_colors)

# make the norm:  Note the center is offset so that the land has more
# dynamic range:
divnorm = colors.TwoSlopeNorm(vmin=-500., vcenter=0, vmax=4000)

pcm = ax.pcolormesh(longitude, latitude, topo, rasterized=True, norm=divnorm,
                    cmap=terrain_map, shading='auto')
# Simple geographic plot, set aspect ratio because distance between lines of
# longitude depends on latitude.
ax.set_aspect(1 / np.cos(np.deg2rad(49)))
ax.set_title('TwoSlopeNorm(x)')
cb = fig.colorbar(pcm, shrink=0.6)
cb.set_ticks([-500, 0, 1000, 2000, 3000, 4000])
plt.show()


# %%
# FuncNorm: Arbitrary function normalization
# ------------------------------------------
#
# If the above norms do not provide the normalization you want, you can use
# `~.colors.FuncNorm` to define your own.  Note that this example is the same
# as `~.colors.PowerNorm` with a power of 0.5:

def _forward(x):
    return np.sqrt(x)


def _inverse(x):
    return x**2

N = 100
X, Y = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.)) * X**2
fig, ax = plt.subplots()

norm = colors.FuncNorm((_forward, _inverse), vmin=0, vmax=20)
```
### 6 - galleries/users_explain/colors/colormapnorms.py:

Start line: 173, End line: 232

```python
X, Y = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.)) * X**2

fig, ax = plt.subplots(2, 1, layout='constrained')

pcm = ax[0].pcolormesh(X, Y, Z1, norm=colors.PowerNorm(gamma=0.5),
                       cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='max')
ax[0].set_title('PowerNorm()')

pcm = ax[1].pcolormesh(X, Y, Z1, cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='max')
ax[1].set_title('Normalize()')
plt.show()

# %%
# Discrete bounds
# ---------------
#
# Another normalization that comes with Matplotlib is `.colors.BoundaryNorm`.
# In addition to *vmin* and *vmax*, this takes as arguments boundaries between
# which data is to be mapped.  The colors are then linearly distributed between
# these "bounds".  It can also take an *extend* argument to add upper and/or
# lower out-of-bounds values to the range over which the colors are
# distributed. For instance:
#
# .. code-block:: pycon
#
#   >>> import matplotlib.colors as colors
#   >>> bounds = np.array([-0.25, -0.125, 0, 0.5, 1])
#   >>> norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)
#   >>> print(norm([-0.2, -0.15, -0.02, 0.3, 0.8, 0.99]))
#   [0 0 1 2 3 3]
#
# Note: Unlike the other norms, this norm returns values from 0 to *ncolors*-1.

N = 100
X, Y = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = ((Z1 - Z2) * 2)[:-1, :-1]

fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
ax = ax.flatten()

# Default norm:
pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[0], orientation='vertical')
ax[0].set_title('Default norm')

# Even bounds give a contour-like effect:
bounds = np.linspace(-1.5, 1.5, 7)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[1].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')
ax[1].set_title('BoundaryNorm: 7 boundaries')

# Bounds may be unevenly spaced:
bounds = np.array([-0.2, -0.1, 0, 0.5, 1])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
```
### 7 - galleries/examples/images_contours_and_fields/colormap_normalizations_symlognorm.py:

Start line: 1, End line: 86

```python
"""
==================================
Colormap normalizations SymLogNorm
==================================

Demonstration of using norm to map colormaps onto data in non-linear ways.

.. redirect-from:: /gallery/userdemo/colormap_normalization_symlognorm
"""

# %%
# Synthetic dataset consisting of two humps, one negative and one positive,
# the positive with 8-times the amplitude.
# Linearly, the negative hump is almost invisible,
# and it is very difficult to see any detail of its profile.
# With the logarithmic scaling applied to both positive and negative values,
# it is much easier to see the shape of each hump.
#
# See `~.colors.SymLogNorm`.

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors


def rbf(x, y):
    return 1.0 / (1 + 5 * ((x ** 2) + (y ** 2)))

N = 200
gain = 8
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = rbf(X + 0.5, Y + 0.5)
Z2 = rbf(X - 0.5, Y - 0.5)
Z = gain * Z1 - Z2

shadeopts = {'cmap': 'PRGn', 'shading': 'gouraud'}
colormap = 'PRGn'
lnrwidth = 0.5

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=lnrwidth, linscale=1,
                                              vmin=-gain, vmax=gain, base=10),
                       **shadeopts)
fig.colorbar(pcm, ax=ax[0], extend='both')
ax[0].text(-2.5, 1.5, 'symlog')

pcm = ax[1].pcolormesh(X, Y, Z, vmin=-gain, vmax=gain,
                       **shadeopts)
fig.colorbar(pcm, ax=ax[1], extend='both')
ax[1].text(-2.5, 1.5, 'linear')


# %%
# In order to find the best visualization for any particular dataset,
# it may be necessary to experiment with multiple different color scales.
# As well as the `~.colors.SymLogNorm` scaling, there is also
# the option of using `~.colors.AsinhNorm` (experimental), which has a smoother
# transition between the linear and logarithmic regions of the transformation
# applied to the data values, "Z".
# In the plots below, it may be possible to see contour-like artifacts
# around each hump despite there being no sharp features
# in the dataset itself. The ``asinh`` scaling shows a smoother shading
# of each hump.

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=lnrwidth, linscale=1,
                                              vmin=-gain, vmax=gain, base=10),
                       **shadeopts)
fig.colorbar(pcm, ax=ax[0], extend='both')
ax[0].text(-2.5, 1.5, 'symlog')

pcm = ax[1].pcolormesh(X, Y, Z,
                       norm=colors.AsinhNorm(linear_width=lnrwidth,
                                             vmin=-gain, vmax=gain),
                       **shadeopts)
fig.colorbar(pcm, ax=ax[1], extend='both')
ax[1].text(-2.5, 1.5, 'asinh')


plt.show()
```
### 8 - galleries/users_explain/colors/colormapnorms.py:

Start line: 96, End line: 172

```python
x = np.arange(-3.0, 4.001, delta)
y = np.arange(-4.0, 3.001, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (0.9*Z1 - 0.5*Z2) * 2

# select a divergent colormap
cmap = cm.coolwarm

fig, (ax1, ax2) = plt.subplots(ncols=2)
pc = ax1.pcolormesh(Z, cmap=cmap)
fig.colorbar(pc, ax=ax1)
ax1.set_title('Normalize()')

pc = ax2.pcolormesh(Z, norm=colors.CenteredNorm(), cmap=cmap)
fig.colorbar(pc, ax=ax2)
ax2.set_title('CenteredNorm()')

plt.show()

# %%
# Symmetric logarithmic
# ---------------------
#
# Similarly, it sometimes happens that there is data that is positive
# and negative, but we would still like a logarithmic scaling applied to
# both.  In this case, the negative numbers are also scaled
# logarithmically, and mapped to smaller numbers; e.g., if ``vmin=-vmax``,
# then the negative numbers are mapped from 0 to 0.5 and the
# positive from 0.5 to 1.
#
# Since the logarithm of values close to zero tends toward infinity, a
# small range around zero needs to be mapped linearly.  The parameter
# *linthresh* allows the user to specify the size of this range
# (-*linthresh*, *linthresh*).  The size of this range in the colormap is
# set by *linscale*.  When *linscale* == 1.0 (the default), the space used
# for the positive and negative halves of the linear range will be equal
# to one decade in the logarithmic range.

N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=-1.0, vmax=1.0, base=10),
                       cmap='RdBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='both')

pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z), shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='both')
plt.show()

# %%
# Power-law
# ---------
#
# Sometimes it is useful to remap the colors onto a power-law
# relationship (i.e. :math:`y=x^{\gamma}`, where :math:`\gamma` is the
# power).  For this we use the `.colors.PowerNorm`.  It takes as an
# argument *gamma* (*gamma* == 1.0 will just yield the default linear
# normalization):
#
# .. note::
#
#    There should probably be a good reason for plotting the data using
#    this type of transformation.  Technical viewers are used to linear
#    and logarithmic axes and data transformations.  Power laws are less
#    common, and viewers should explicitly be made aware that they have
#    been used.

N = 100
```
### 9 - lib/matplotlib/cm.py:

Start line: 592, End line: 619

```python
class ScalarMappable:

    @norm.setter
    def norm(self, norm):
        _api.check_isinstance((colors.Normalize, str, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()
        elif isinstance(norm, str):
            try:
                scale_cls = scale._scale_mapping[norm]
            except KeyError:
                raise ValueError(
                    "Invalid norm str name; the following values are "
                    f"supported: {', '.join(scale._scale_mapping)}"
                ) from None
            norm = _auto_norm_from_scale(scale_cls)()

        if norm is self.norm:
            # We aren't updating anything
            return

        in_init = self.norm is None
        # Remove the current callback and connect to the new one
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        self._norm = norm
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        if not in_init:
            self.changed()
```
### 10 - galleries/examples/images_contours_and_fields/colormap_normalizations.py:

Start line: 107, End line: 145

```python
# %%
fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=MidpointNormalize(midpoint=0.),
                       cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both')

pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z),
                       shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='both')

# %%
# BoundaryNorm: For this one you provide the boundaries for your colors,
# and the Norm puts the first color in between the first pair, the
# second color between the second pair, etc.

fig, ax = plt.subplots(3, 1, figsize=(8, 8))
ax = ax.flatten()
# even bounds gives a contour-like effect
bounds = np.linspace(-1, 1, 10)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=norm,
                       cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')

# uneven bounds changes the colormapping:
bounds = np.array([-0.25, -0.125, 0, 0.5, 1])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[1].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')

pcm = ax[2].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z1),
                       shading='nearest')
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical')

plt.show()
```
### 22 - lib/matplotlib/colorbar.py:

Start line: 1167, End line: 1199

```python
@_docstring.interpd
class Colorbar:

    def _reset_locator_formatter_scale(self):
        """
        Reset the locator et al to defaults.  Any user-hardcoded changes
        need to be re-entered if this gets called (either at init, or when
        the mappable normal gets changed: Colorbar.update_normal)
        """
        self._process_values()
        self._locator = None
        self._minorlocator = None
        self._formatter = None
        self._minorformatter = None
        if (isinstance(self.mappable, contour.ContourSet) and
                isinstance(self.norm, colors.LogNorm)):
            # if contours have lognorm, give them a log scale...
            self._set_scale('log')
        elif (self.boundaries is not None or
                isinstance(self.norm, colors.BoundaryNorm)):
            if self.spacing == 'uniform':
                funcs = (self._forward_boundaries, self._inverse_boundaries)
                self._set_scale('function', functions=funcs)
            elif self.spacing == 'proportional':
                self._set_scale('linear')
        elif getattr(self.norm, '_scale', None):
            # use the norm's scale (if it exists and is not None):
            self._set_scale(self.norm._scale)
        elif type(self.norm) is colors.Normalize:
            # plain Normalize:
            self._set_scale('linear')
        else:
            # norm._scale is None or not an attr: derive the scale from
            # the Norm:
            funcs = (self.norm, self.norm.inverse)
            self._set_scale('function', functions=funcs)
```
### 26 - lib/matplotlib/colorbar.py:

Start line: 529, End line: 579

```python
@_docstring.interpd
class Colorbar:

    def _draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
        if self.orientation == 'vertical':
            if mpl.rcParams['ytick.minor.visible']:
                self.minorticks_on()
        else:
            if mpl.rcParams['xtick.minor.visible']:
                self.minorticks_on()
        self._long_axis().set(label_position=self.ticklocation,
                              ticks_position=self.ticklocation)
        self._short_axis().set_ticks([])
        self._short_axis().set_ticks([], minor=True)

        # Set self._boundaries and self._values, including extensions.
        # self._boundaries are the edges of each square of color, and
        # self._values are the value to map into the norm to get the
        # color:
        self._process_values()
        # Set self.vmin and self.vmax to first and last boundary, excluding
        # extensions:
        self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]
        # Compute the X/Y mesh.
        X, Y = self._mesh()
        # draw the extend triangles, and shrink the inner axes to accommodate.
        # also adds the outline path to self.outline spine:
        self._do_extends()
        lower, upper = self.vmin, self.vmax
        if self._long_axis().get_inverted():
            # If the axis is inverted, we need to swap the vmin/vmax
            lower, upper = upper, lower
        if self.orientation == 'vertical':
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(lower, upper)
        else:
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(lower, upper)

        # set up the tick locators and formatters.  A bit complicated because
        # boundary norms + uniform spacing requires a manual locator.
        self.update_ticks()

        if self._filled:
            ind = np.arange(len(self._values))
            if self._extend_lower():
                ind = ind[1:]
            if self._extend_upper():
                ind = ind[:-1]
            self._add_solids(X, Y, self._values[ind, np.newaxis])
```
### 27 - lib/matplotlib/colorbar.py:

Start line: 1065, End line: 1116

```python
@_docstring.interpd
class Colorbar:

    def _process_values(self):
        """
        Set `_boundaries` and `_values` based on the self.boundaries and
        self.values if not None, or based on the size of the colormap and
        the vmin/vmax of the norm.
        """
        if self.values is not None:
            # set self._boundaries from the values...
            self._values = np.array(self.values)
            if self.boundaries is None:
                # bracket values by 1/2 dv:
                b = np.zeros(len(self.values) + 1)
                b[1:-1] = 0.5 * (self._values[:-1] + self._values[1:])
                b[0] = 2.0 * b[1] - b[2]
                b[-1] = 2.0 * b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = np.array(self.boundaries)
            return

        # otherwise values are set from the boundaries
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
        elif isinstance(self.norm, colors.NoNorm):
            # NoNorm has N blocks, so N+1 boundaries, centered on integers:
            b = np.arange(self.cmap.N + 1) - .5
        elif self.boundaries is not None:
            b = self.boundaries
        else:
            # otherwise make the boundaries from the size of the cmap:
            N = self.cmap.N + 1
            b, _ = self._uniform_y(N)
        # add extra boundaries if needed:
        if self._extend_lower():
            b = np.hstack((b[0] - 1, b))
        if self._extend_upper():
            b = np.hstack((b, b[-1] + 1))

        # transform from 0-1 to vmin-vmax:
        if not self.norm.scaled():
            self.norm.vmin = 0
            self.norm.vmax = 1
        self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(
            self.norm.vmin, self.norm.vmax, expander=0.1)
        if (not isinstance(self.norm, colors.BoundaryNorm) and
                (self.boundaries is None)):
            b = self.norm.inverse(b)

        self._boundaries = np.asarray(b, dtype=float)
        self._values = 0.5 * (self._boundaries[:-1] + self._boundaries[1:])
        if isinstance(self.norm, colors.NoNorm):
            self._values = (self._values + 0.00001).astype(np.int16)
```
### 37 - lib/matplotlib/colorbar.py:

Start line: 281, End line: 441

```python
@_docstring.interpd
class Colorbar:

    @_api.delete_parameter("3.6", "filled")
    def __init__(self, ax, mappable=None, *, cmap=None,
                 norm=None,
                 alpha=None,
                 values=None,
                 boundaries=None,
                 orientation=None,
                 ticklocation='auto',
                 extend=None,
                 spacing='uniform',  # uniform or proportional
                 ticks=None,
                 format=None,
                 drawedges=False,
                 filled=True,
                 extendfrac=None,
                 extendrect=False,
                 label='',
                 location=None,
                 ):

        if mappable is None:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        # Ensure the given mappable's norm has appropriate vmin and vmax
        # set even if mappable.draw has not yet been called.
        if mappable.get_array() is not None:
            mappable.autoscale_None()

        self.mappable = mappable
        cmap = mappable.cmap
        norm = mappable.norm

        if isinstance(mappable, contour.ContourSet):
            cs = mappable
            alpha = cs.get_alpha()
            boundaries = cs._levels
            values = cs.cvalues
            extend = cs.extend
            filled = cs.filled
            if ticks is None:
                ticks = ticker.FixedLocator(cs.levels, nbins=10)
        elif isinstance(mappable, martist.Artist):
            alpha = mappable.get_alpha()

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacks.connect(
            'changed', self.update_normal)

        location_orientation = _get_orientation_from_location(location)

        _api.check_in_list(
            [None, 'vertical', 'horizontal'], orientation=orientation)
        _api.check_in_list(
            ['auto', 'left', 'right', 'top', 'bottom'],
            ticklocation=ticklocation)
        _api.check_in_list(
            ['uniform', 'proportional'], spacing=spacing)

        if location_orientation is not None and orientation is not None:
            if location_orientation != orientation:
                raise TypeError(
                    "location and orientation are mutually exclusive")
        else:
            orientation = orientation or location_orientation or "vertical"

        self.ax = ax
        self.ax._axes_locator = _ColorbarAxesLocator(self)

        if extend is None:
            if (not isinstance(mappable, contour.ContourSet)
                    and getattr(cmap, 'colorbar_extend', False) is not False):
                extend = cmap.colorbar_extend
            elif hasattr(norm, 'extend'):
                extend = norm.extend
            else:
                extend = 'neither'
        self.alpha = None
        # Call set_alpha to handle array-like alphas properly
        self.set_alpha(alpha)
        self.cmap = cmap
        self.norm = norm
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = _api.check_getitem(
            {'neither': slice(0, None), 'both': slice(1, -1),
             'min': slice(1, None), 'max': slice(0, -1)},
            extend=extend)
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self._filled = filled
        self.extendfrac = extendfrac
        self.extendrect = extendrect
        self._extend_patches = []
        self.solids = None
        self.solids_patches = []
        self.lines = []

        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)

        self.dividers = collections.LineCollection(
            [],
            colors=[mpl.rcParams['axes.edgecolor']],
            linewidths=[0.5 * mpl.rcParams['axes.linewidth']],
            clip_on=False)
        self.ax.add_collection(self.dividers)

        self._locator = None
        self._minorlocator = None
        self._formatter = None
        self._minorformatter = None

        if ticklocation == 'auto':
            ticklocation = _get_ticklocation_from_orientation(
                orientation) if location is None else location
        self.ticklocation = ticklocation

        self.set_label(label)
        self._reset_locator_formatter_scale()

        if np.iterable(ticks):
            self._locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self._locator = ticks

        if isinstance(format, str):
            # Check format between FormatStrFormatter and StrMethodFormatter
            try:
                self._formatter = ticker.FormatStrFormatter(format)
                _ = self._formatter(0)
            except TypeError:
                self._formatter = ticker.StrMethodFormatter(format)
        else:
            self._formatter = format  # Assume it is a Formatter or None
        self._draw_all()

        if isinstance(mappable, contour.ContourSet) and not mappable.filled:
            self.add_lines(mappable)

        # Link the Axes and Colorbar for interactive use
        self.ax._colorbar = self
        # Don't navigate on any of these types of mappables
        if (isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm)) or
                isinstance(self.mappable, contour.ContourSet)):
            self.ax.set_navigate(False)

        # These are the functions that set up interactivity on this colorbar
        self._interactive_funcs = ["_get_view", "_set_view",
                                   "_set_view_from_bbox", "drag_pan"]
        for x in self._interactive_funcs:
            setattr(self.ax, x, getattr(self, x))
        # Set the cla function to the cbar's method to override it
        self.ax.cla = self._cbar_cla
        # Callbacks for the extend calculations to handle inverting the axis
        self._extend_cid1 = self.ax.callbacks.connect(
            "xlim_changed", self._do_extends)
        self._extend_cid2 = self.ax.callbacks.connect(
            "ylim_changed", self._do_extends)
```
### 41 - lib/matplotlib/colorbar.py:

Start line: 195, End line: 279

```python
@_docstring.interpd
class Colorbar:
    r"""
    Draw a colorbar in an existing axes.

    Typically, colorbars are created using `.Figure.colorbar` or
    `.pyplot.colorbar` and associated with `.ScalarMappable`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    In order to draw a colorbar not associated with other elements in the
    figure, e.g. when showing a colormap by itself, one can create an empty
    `.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
    to `Colorbar`.

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.

    mappable : `.ScalarMappable`
        The mappable whose colormap and norm will be used.

        To show the under- and over- value colors, the mappable's norm should
        be specified as ::

            norm = colors.Normalize(clip=False)

        To show the colors versus index instead of on a 0-1 scale, use::

            norm=colors.NoNorm()

    cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The colormap to use.  This parameter is ignored, unless *mappable* is
        None.

    norm : `~matplotlib.colors.Normalize`
        The normalization to use.  This parameter is ignored, unless *mappable*
        is None.

    alpha : float
        The colorbar transparency between 0 (transparent) and 1 (opaque).

    orientation : None or {'vertical', 'horizontal'}
        If None, use the value determined by *location*. If both
        *orientation* and *location* are None then defaults to 'vertical'.

    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}
        The location of the colorbar ticks. The *ticklocation* must match
        *orientation*. For example, a horizontal colorbar can only have ticks
        at the top or the bottom. If 'auto', the ticks will be the same as
        *location*, so a colorbar to the left will have ticks to the left. If
        *location* is None, the ticks will be at the bottom for a horizontal
        colorbar and at the right for a vertical.

    drawedges : bool
        Whether to draw lines at color boundaries.

    filled : bool

    %(_colormap_kw_doc)s

    location : None or {'left', 'right', 'top', 'bottom'}
        Set the *orientation* and *ticklocation* of the colorbar using a
        single argument. Colorbars on the left and right are vertical,
        colorbars at the top and bottom are horizontal. The *ticklocation* is
        the same as *location*, so if *location* is 'top', the ticks are on
        the top. *orientation* and/or *ticklocation* can be provided as well
        and overrides the value set by *location*, but there will be an error
        for incompatible combinations.

        .. versionadded:: 3.7
    """

    n_rasterize = 50  # rasterize solids if number of colors >= n_rasterize
```
### 47 - lib/matplotlib/colorbar.py:

Start line: 603, End line: 625

```python
@_docstring.interpd
class Colorbar:

    def _update_dividers(self):
        if not self.drawedges:
            self.dividers.set_segments([])
            return
        # Place all *internal* dividers.
        if self.orientation == 'vertical':
            lims = self.ax.get_ylim()
            bounds = (lims[0] < self._y) & (self._y < lims[1])
        else:
            lims = self.ax.get_xlim()
            bounds = (lims[0] < self._y) & (self._y < lims[1])
        y = self._y[bounds]
        # And then add outer dividers if extensions are on.
        if self._extend_lower():
            y = np.insert(y, 0, lims[0])
        if self._extend_upper():
            y = np.append(y, lims[1])
        X, Y = np.meshgrid([0, 1], y)
        if self.orientation == 'vertical':
            segments = np.dstack([X, Y])
        else:
            segments = np.dstack([Y, X])
        self.dividers.set_segments(segments)
```
### 50 - lib/matplotlib/colorbar.py:

Start line: 1330, End line: 1351

```python
@_docstring.interpd
class Colorbar:

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        # docstring inherited
        # For colorbars, we use the zoom bbox to scale the norm's vmin/vmax
        new_xbound, new_ybound = self.ax._prepare_view_from_bbox(
            bbox, direction=direction, mode=mode, twinx=twinx, twiny=twiny)
        if self.orientation == 'horizontal':
            self.norm.vmin, self.norm.vmax = new_xbound
        elif self.orientation == 'vertical':
            self.norm.vmin, self.norm.vmax = new_ybound

    def drag_pan(self, button, key, x, y):
        # docstring inherited
        points = self.ax._get_pan_points(button, key, x, y)
        if points is not None:
            if self.orientation == 'horizontal':
                self.norm.vmin, self.norm.vmax = points[:, 0]
            elif self.orientation == 'vertical':
                self.norm.vmin, self.norm.vmax = points[:, 1]


ColorbarBase = Colorbar
```
### 53 - lib/matplotlib/colorbar.py:

Start line: 1150, End line: 1165

```python
@_docstring.interpd
class Colorbar:

    def _forward_boundaries(self, x):
        # map boundaries equally between 0 and 1...
        b = self._boundaries
        y = np.interp(x, b, np.linspace(0, 1, len(b)))
        # the following avoids ticks in the extends:
        eps = (b[-1] - b[0]) * 1e-6
        # map these _well_ out of bounds to keep any ticks out
        # of the extends region...
        y[x < b[0]-eps] = -1
        y[x > b[-1]+eps] = 2
        return y

    def _inverse_boundaries(self, x):
        # invert the above...
        b = self._boundaries
        return np.interp(x, np.linspace(0, 1, len(b)), b)
```
### 59 - lib/matplotlib/colorbar.py:

Start line: 1, End line: 27

```python
"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which internally use `.Colorbar` together with
   `.make_axes_gridspec` (for `.GridSpec`-positioned axes) or `.make_axes` (for
   non-`.GridSpec`-positioned axes).

   End-users most likely won't need to directly use this module's API.
"""

import logging

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring

_log = logging.getLogger(__name__)
```
### 67 - lib/matplotlib/colorbar.py:

Start line: 714, End line: 738

```python
@_docstring.interpd
class Colorbar:

    def _do_extends(self, ax=None):
        # ... other code
        if self._extend_upper():
            if not self.extendrect:
                # triangle
                xy = np.array([[0, 1], [0.5, top], [1, 1]])
            else:
                # rectangle
                xy = np.array([[0, 1], [0, top], [1, top], [1, 1]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            val = 0 if self._long_axis().get_inverted() else -1
            color = self.cmap(self.norm(self._values[val]))
            hatch_idx = len(self._y) - 1
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color, alpha=self.alpha,
                linewidth=0, antialiased=False,
                transform=self.ax.transAxes, hatch=hatches[hatch_idx],
                clip_on=False,
                # Place it right behind the standard patches, which is
                # needed if we updated the extends
                zorder=np.nextafter(self.ax.patch.zorder, -np.inf))
            self.ax.add_patch(patch)
            self._extend_patches.append(patch)

        self._update_dividers()
```
### 69 - lib/matplotlib/colorbar.py:

Start line: 29, End line: 114

```python
_docstring.interpd.update(
    _make_axes_kw_doc="""
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent axes, where the colorbar axes
    is created.  It also determines the *orientation* of the colorbar
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.

orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbar.  It is preferable to set the *location*
    of the colorbar, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.

fraction : float, default: 0.15
    Fraction of original axes to use for colorbar.

shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.

aspect : float, default: 20
    Ratio of long to short dimensions.

pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original axes between colorbar and new image axes.

anchor : (float, float), optional
    The anchor point of the colorbar axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.

panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.""",
    _colormap_kw_doc="""
extend : {'neither', 'both', 'min', 'max'}
    Make pointed end(s) for out-of-range values (unless 'neither').  These are
    set for a given colormap using the colormap set_under and set_over methods.

extendfrac : {*None*, 'auto', length, lengths}
    If set to *None*, both the minimum and maximum triangular colorbar
    extensions will have a length of 5% of the interior colorbar length (this
    is the default setting).

    If set to 'auto', makes the triangular colorbar extensions the same lengths
    as the interior boxes (when *spacing* is set to 'uniform') or the same
    lengths as the respective adjacent interior boxes (when *spacing* is set to
    'proportional').

    If a scalar, indicates the length of both the minimum and maximum
    triangular colorbar extensions as a fraction of the interior colorbar
    length.  A two-element sequence of fractions may also be given, indicating
    the lengths of the minimum and maximum colorbar extensions respectively as
    a fraction of the interior colorbar length.

extendrect : bool
    If *False* the minimum and maximum colorbar extensions will be triangular
    (the default).  If *True* the extensions will be rectangular.

spacing : {'uniform', 'proportional'}
    For discrete colorbars (`.BoundaryNorm` or contours), 'uniform' gives each
    color the same space; 'proportional' makes the space proportional to the
    data interval.

ticks : None or list of ticks or Locator
    If None, ticks are determined automatically from the input.

format : None or str or Formatter
    If None, `~.ticker.ScalarFormatter` is used.
    Format strings, e.g., ``"%4.2e"`` or ``"{x:.2e}"``, are supported.
    An alternative `~.ticker.Formatter` may be given instead.

drawedges : bool
    Whether to draw lines at color boundaries.

label : str
    The label on the colorbar's long axis.

boundaries, values : None or a sequence
    If unset, the colormap will be displayed on a 0-1 scale.
    If sequences, *values* must have a length 1 less than *boundaries*.  For
    each region delimited by adjacent entries in *boundaries*, the color mapped
    to the corresponding value in values will be used.
    Normally only useful for indexed colors (i.e. ``norm=NoNorm()``) or other
    unusual circumstances.""")
```
### 70 - lib/matplotlib/colorbar.py:

Start line: 1298, End line: 1328

```python
@_docstring.interpd
class Colorbar:

    def _extend_lower(self):
        """Return whether the lower limit is open ended."""
        minmax = "max" if self._long_axis().get_inverted() else "min"
        return self.extend in ('both', minmax)

    def _extend_upper(self):
        """Return whether the upper limit is open ended."""
        minmax = "min" if self._long_axis().get_inverted() else "max"
        return self.extend in ('both', minmax)

    def _long_axis(self):
        """Return the long axis"""
        if self.orientation == 'vertical':
            return self.ax.yaxis
        return self.ax.xaxis

    def _short_axis(self):
        """Return the short axis"""
        if self.orientation == 'vertical':
            return self.ax.xaxis
        return self.ax.yaxis

    def _get_view(self):
        # docstring inherited
        # An interactive view for a colorbar is the norm's vmin/vmax
        return self.norm.vmin, self.norm.vmax

    def _set_view(self, view):
        # docstring inherited
        # An interactive view for a colorbar is the norm's vmin/vmax
        self.norm.vmin, self.norm.vmax = view
```
### 71 - lib/matplotlib/colorbar.py:

Start line: 1351, End line: 1383

```python
# Backcompat API


def _normalize_location_orientation(location, orientation):
    if location is None:
        location = _get_ticklocation_from_orientation(orientation)
    loc_settings = _api.check_getitem({
        "left":   {"location": "left", "anchor": (1.0, 0.5),
                   "panchor": (0.0, 0.5), "pad": 0.10},
        "right":  {"location": "right", "anchor": (0.0, 0.5),
                   "panchor": (1.0, 0.5), "pad": 0.05},
        "top":    {"location": "top", "anchor": (0.5, 0.0),
                   "panchor": (0.5, 1.0), "pad": 0.05},
        "bottom": {"location": "bottom", "anchor": (0.5, 1.0),
                   "panchor": (0.5, 0.0), "pad": 0.15},
    }, location=location)
    loc_settings["orientation"] = _get_orientation_from_location(location)
    if orientation is not None and orientation != loc_settings["orientation"]:
        # Allow the user to pass both if they are consistent.
        raise TypeError("location and orientation are mutually exclusive")
    return loc_settings


def _get_orientation_from_location(location):
    return _api.check_getitem(
        {None: None, "left": "vertical", "right": "vertical",
         "top": "horizontal", "bottom": "horizontal"}, location=location)


def _get_ticklocation_from_orientation(orientation):
    return _api.check_getitem(
        {None: "right", "vertical": "right", "horizontal": "bottom"},
        orientation=orientation)
```
### 81 - lib/matplotlib/colorbar.py:

Start line: 627, End line: 644

```python
@_docstring.interpd
class Colorbar:

    def _add_solids_patches(self, X, Y, C, mappable):
        hatches = mappable.hatches * (len(C) + 1)  # Have enough hatches.
        if self._extend_lower():
            # remove first hatch that goes into the extend patch
            hatches = hatches[1:]
        patches = []
        for i in range(len(X) - 1):
            xy = np.array([[X[i, 0], Y[i, 1]],
                           [X[i, 1], Y[i, 0]],
                           [X[i + 1, 1], Y[i + 1, 0]],
                           [X[i + 1, 0], Y[i + 1, 1]]])
            patch = mpatches.PathPatch(mpath.Path(xy),
                                       facecolor=self.cmap(self.norm(C[i][0])),
                                       hatch=hatches[i], linewidth=0,
                                       antialiased=False, alpha=self.alpha)
            self.ax.add_patch(patch)
            patches.append(patch)
        self.solids_patches = patches
```
### 82 - lib/matplotlib/colorbar.py:

Start line: 1201, End line: 1233

```python
@_docstring.interpd
class Colorbar:

    def _locate(self, x):
        """
        Given a set of color data values, return their
        corresponding colorbar data coordinates.
        """
        if isinstance(self.norm, (colors.NoNorm, colors.BoundaryNorm)):
            b = self._boundaries
            xn = x
        else:
            # Do calculations using normalized coordinates so
            # as to make the interpolation more accurate.
            b = self.norm(self._boundaries, clip=False).filled()
            xn = self.norm(x, clip=False).filled()

        bunique = b[self._inside]
        yunique = self._y

        z = np.interp(xn, bunique, yunique)
        return z

    # trivial helpers

    def _uniform_y(self, N):
        """
        Return colorbar data coordinates for *N* uniformly
        spaced boundaries, plus extension lengths if required.
        """
        automin = automax = 1. / (N - 1.)
        extendlength = self._get_extension_lengths(self.extendfrac,
                                                   automin, automax,
                                                   default=0.05)
        y = np.linspace(0, 1, N)
        return y, extendlength
```
### 101 - lib/matplotlib/colorbar.py:

Start line: 1235, End line: 1271

```python
@_docstring.interpd
class Colorbar:

    def _proportional_y(self):
        """
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar, plus extension lengths if required:
        """
        if (isinstance(self.norm, colors.BoundaryNorm) or
                self.boundaries is not None):
            y = (self._boundaries - self._boundaries[self._inside][0])
            y = y / (self._boundaries[self._inside][-1] -
                     self._boundaries[self._inside][0])
            # need yscaled the same as the axes scale to get
            # the extend lengths.
            if self.spacing == 'uniform':
                yscaled = self._forward_boundaries(self._boundaries)
            else:
                yscaled = y
        else:
            y = self.norm(self._boundaries.copy())
            y = np.ma.filled(y, np.nan)
            # the norm and the scale should be the same...
            yscaled = y
        y = y[self._inside]
        yscaled = yscaled[self._inside]
        # normalize from 0..1:
        norm = colors.Normalize(y[0], y[-1])
        y = np.ma.filled(norm(y), np.nan)
        norm = colors.Normalize(yscaled[0], yscaled[-1])
        yscaled = np.ma.filled(norm(yscaled), np.nan)
        # make the lower and upper extend lengths proportional to the lengths
        # of the first and last boundary spacing (if extendfrac='auto'):
        automin = yscaled[1] - yscaled[0]
        automax = yscaled[-1] - yscaled[-2]
        extendlength = [0, 0]
        if self._extend_lower() or self._extend_upper():
            extendlength = self._get_extension_lengths(
                    self.extendfrac, automin, automax, default=0.05)
        return y, extendlength
```
### 106 - lib/matplotlib/colorbar.py:

Start line: 1028, End line: 1063

```python
@_docstring.interpd
class Colorbar:

    def remove(self):
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
        if hasattr(self.ax, '_colorbar_info'):
            parents = self.ax._colorbar_info['parents']
            for a in parents:
                if self.ax in a._colorbars:
                    a._colorbars.remove(self.ax)

        self.ax.remove()

        self.mappable.callbacks.disconnect(self.mappable.colorbar_cid)
        self.mappable.colorbar = None
        self.mappable.colorbar_cid = None
        # Remove the extension callbacks
        self.ax.callbacks.disconnect(self._extend_cid1)
        self.ax.callbacks.disconnect(self._extend_cid2)

        try:
            ax = self.mappable.axes
        except AttributeError:
            return
        try:
            gs = ax.get_subplotspec().get_gridspec()
            subplotspec = gs.get_topmost_subplotspec()
        except AttributeError:
            # use_gridspec was False
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:
            # use_gridspec was True
            ax.set_subplotspec(subplotspec)
```
### 110 - lib/matplotlib/colorbar.py:

Start line: 117, End line: 143

```python
def _set_ticks_on_axis_warn(*args, **kwargs):
    # a top level function which gets put in at the axes'
    # set_xticks and set_yticks by Colorbar.__init__.
    _api.warn_external("Use the colorbar set_ticks() method instead.")


class _ColorbarSpine(mspines.Spine):
    def __init__(self, axes):
        self._ax = axes
        super().__init__(axes, 'colorbar', mpath.Path(np.empty((0, 2))))
        mpatches.Patch.set_transform(self, axes.transAxes)

    def get_window_extent(self, renderer=None):
        # This Spine has no Axis associated with it, and doesn't need to adjust
        # its location, so we can directly get the window extent from the
        # super-super-class.
        return mpatches.Patch.get_window_extent(self, renderer=renderer)

    def set_xy(self, xy):
        self._path = mpath.Path(xy, closed=True)
        self._xy = xy
        self.stale = True

    def draw(self, renderer):
        ret = mpatches.Patch.draw(self, renderer)
        self.stale = False
        return ret
```
### 111 - lib/matplotlib/colorbar.py:

Start line: 820, End line: 876

```python
@_docstring.interpd
class Colorbar:

    def update_ticks(self):
        """
        Set up the ticks and ticklabels. This should not be needed by users.
        """
        # Get the locator and formatter; defaults to self._locator if not None.
        self._get_ticker_locator_formatter()
        self._long_axis().set_major_locator(self._locator)
        self._long_axis().set_minor_locator(self._minorlocator)
        self._long_axis().set_major_formatter(self._formatter)

    def _get_ticker_locator_formatter(self):
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
        locator = self._locator
        formatter = self._formatter
        minorlocator = self._minorlocator
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
            if minorlocator is None:
                minorlocator = ticker.FixedLocator(b)
        elif isinstance(self.norm, colors.NoNorm):
            if locator is None:
                # put ticks on integers between the boundaries of NoNorm
                nv = len(self._values)
                base = 1 + int(nv / 10)
                locator = ticker.IndexLocator(base=base, offset=.5)
        elif self.boundaries is not None:
            b = self._boundaries[self._inside]
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        else:  # most cases:
            if locator is None:
                # we haven't set the locator explicitly, so use the default
                # for this axis:
                locator = self._long_axis().get_major_locator()
            if minorlocator is None:
                minorlocator = self._long_axis().get_minor_locator()

        if minorlocator is None:
            minorlocator = ticker.NullLocator()

        if formatter is None:
            formatter = self._long_axis().get_major_formatter()

        self._locator = locator
        self._formatter = formatter
        self._minorlocator = minorlocator
        _log.debug('locator: %r', locator)
```
