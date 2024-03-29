# matplotlib__matplotlib-14471

| **matplotlib/matplotlib** | `ddb891751d797517e28b9f74d1fffc98716f8c7d` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 38592 |
| **Any found context length** | 26753 |
| **Avg pos** | 100.0 |
| **Min pos** | 55 |
| **Max pos** | 90 |
| **Top file pos** | 4 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/__init__.py b/lib/matplotlib/__init__.py
--- a/lib/matplotlib/__init__.py
+++ b/lib/matplotlib/__init__.py
@@ -1105,6 +1105,10 @@ def use(backend, *, force=True):
     """
     Select the backend used for rendering and GUI integration.
 
+    If pyplot is already imported, `~matplotlib.pyplot.switch_backend` is used
+    and if the new backend is different than the current backend, all Figures
+    will be closed.
+
     Parameters
     ----------
     backend : str
@@ -1135,6 +1139,8 @@ def use(backend, *, force=True):
     --------
     :ref:`backends`
     matplotlib.get_backend
+    matplotlib.pyplot.switch_backend
+
     """
     name = validate_backend(backend)
     # don't (prematurely) resolve the "auto" backend setting
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -209,21 +209,24 @@ def _get_backend_mod():
 
 def switch_backend(newbackend):
     """
-    Close all open figures and set the Matplotlib backend.
+    Set the pyplot backend.
 
-    The argument is case-insensitive.  Switching to an interactive backend is
-    possible only if no event loop for another interactive backend has started.
-    Switching to and from non-interactive backends is always possible.
+    Switching to an interactive backend is possible only if no event loop for
+    another interactive backend has started.  Switching to and from
+    non-interactive backends is always possible.
+
+    If the new backend is different than the current backend then all open
+    Figures will be closed via ``plt.close('all')``.
 
     Parameters
     ----------
     newbackend : str
-        The name of the backend to use.
+        The case-insensitive name of the backend to use.
+
     """
     global _backend_mod
     # make sure the init is pulled up so we can assign to it later
     import matplotlib.backends
-    close("all")
 
     if newbackend is rcsetup._auto_backend_sentinel:
         current_framework = cbook._get_running_interactive_framework()
@@ -260,6 +263,8 @@ def switch_backend(newbackend):
             switch_backend("agg")
             rcParamsOrig["backend"] = "agg"
             return
+    # have to escape the switch on access logic
+    old_backend = dict.__getitem__(rcParams, 'backend')
 
     backend_mod = importlib.import_module(
         cbook._backend_module_name(newbackend))
@@ -323,6 +328,8 @@ def draw_if_interactive():
     # Need to keep a global reference to the backend for compatibility reasons.
     # See https://github.com/matplotlib/matplotlib/issues/6092
     matplotlib.backends.backend = newbackend
+    if not cbook._str_equal(old_backend, newbackend):
+        close("all")
 
     # make sure the repl display hook is installed in case we become
     # interactive

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/__init__.py | 1108 | 1108 | - | 30 | -
| lib/matplotlib/__init__.py | 1138 | 1138 | - | 30 | -
| lib/matplotlib/pyplot.py | 212 | 226 | 55 | 4 | 26753
| lib/matplotlib/pyplot.py | 263 | 263 | 55 | 4 | 26753
| lib/matplotlib/pyplot.py | 326 | 326 | 90 | 4 | 38592


## Problem Statement

```
Existing FigureCanvasQT objects destroyed by call to plt.figure
### Bug report

**Bug summary**

For a number of years, I have been maintaining an interactive application that embeds subclassed FigureCanvasQT objects within a PyQt application. Up until Matplotlib v3.0.3., it was possible to create standard Matplotlib PyQt figures, i.e., using `plt.figure` within an embedded IPython shell, that coexist with the subclassed canvas objects. Now in Matplotlib v3.1.0, a call to `plt.figure()` destroys all the other canvas objects. 

Unfortunately, I cannot debug this within Visual Studio since I am currently unable to install Matplotlib from the source. By going through the `new_figure_manager` code line by line, I can confirm that the windows are destroyed when calling `FigureCanvasQT.__init__(figure)`, but not when calling `FigureCanvasBase.__init__(figure)`, but I can't work out what triggers the destruction of the other windows. I presume the subclassed canvasses are not being registered somewhere, and `pyplot` assumes they are no longer active, but I am hoping someone will show me where to look. This may not be a Matplotlib bug, but it's certainly an unwelcome side effect in my application.

**Code for reproduction**
If you have `nexpy` installed (`pip install nexpy`) and launch it, you can reproduce this from within the embedded IPython shell with the following, which creates a new subclassed window and then attempts to open a regular `pyplot` window.:

\`\`\`
In [1]: new_window=NXPlotView()
In [2]: plt.get_fignums()
Out[2]: [1, 2]
In [3]: plt.figure()
\`\`\`
There are two figure numbers, because NeXpy automatically creates a window with a figure number of 1 when it is launched.

**Actual outcome**

A new window with an updated figure number is created but all other windows not created by `pyplot`  are destroyed.

\`\`\`
In [4]: plt.get_fignums()
Out[4]: [3]
\`\`\`

**Expected outcome**

In Matplotlib v3.0.3, a new `pyplot` window is created by the PyQt5 backend without destroying anything else.

\`\`\`
In [4]: plt.get_fignums()
Out[4]: [1, 2, 3]
\`\`\`

**Matplotlib version**
  * Operating system: Mac OS v10.14.5
  * Matplotlib version: 3.1.0
  * Matplotlib backend: Qt5Agg
  * Python version: 3.7.2
  * Jupyter version (if applicable): 1.0.0
  * Other libraries: 

<!--Please tell us how you installed matplotlib and python e.g., from source, pip, conda-->
<!--If you installed from conda, please specify which channel you used if not the default-->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 lib/matplotlib/_pylab_helpers.py | 69 | 136| 470 | 470 | 920 | 
| 2 | 2 lib/matplotlib/backends/backend_qt.py | 439 | 459| 203 | 673 | 9452 | 
| 3 | 2 lib/matplotlib/backends/backend_qt.py | 192 | 248| 476 | 1149 | 9452 | 
| 4 | 3 lib/matplotlib/backends/_backend_tk.py | 525 | 559| 332 | 1481 | 18746 | 
| 5 | 3 lib/matplotlib/backends/backend_qt.py | 272 | 307| 278 | 1759 | 18746 | 
| 6 | **4 lib/matplotlib/pyplot.py** | 734 | 793| 624 | 2383 | 46713 | 
| 7 | 4 lib/matplotlib/backends/backend_qt.py | 397 | 425| 251 | 2634 | 46713 | 
| 8 | 4 lib/matplotlib/backends/backend_qt.py | 336 | 360| 222 | 2856 | 46713 | 
| 9 | 4 lib/matplotlib/backends/backend_qt.py | 461 | 500| 358 | 3214 | 46713 | 
| 10 | 4 lib/matplotlib/backends/backend_qt.py | 503 | 609| 742 | 3956 | 46713 | 
| 11 | 5 lib/matplotlib/backends/backend_qtcairo.py | 1 | 48| 433 | 4389 | 47147 | 
| 12 | 6 examples/user_interfaces/embedding_in_qt_sgskip.py | 23 | 72| 412 | 4801 | 47681 | 
| 13 | 7 lib/matplotlib/backend_bases.py | 810 | 1660| 6378 | 11179 | 75658 | 
| 14 | 7 lib/matplotlib/backends/backend_qt.py | 962 | 983| 180 | 11359 | 75658 | 
| 15 | 8 lib/matplotlib/figure.py | 1 | 53| 348 | 11707 | 104352 | 
| 16 | **8 lib/matplotlib/pyplot.py** | 1 | 89| 669 | 12376 | 104352 | 
| 17 | **8 lib/matplotlib/pyplot.py** | 332 | 360| 250 | 12626 | 104352 | 
| 18 | 9 lib/matplotlib/backends/backend_gtk3.py | 73 | 129| 533 | 13159 | 109393 | 
| 19 | 10 lib/matplotlib/backends/backend_macosx.py | 145 | 184| 258 | 13417 | 110911 | 
| 20 | 11 lib/matplotlib/backends/backend_wx.py | 634 | 663| 262 | 13679 | 123093 | 
| 21 | 11 lib/matplotlib/backends/backend_qt.py | 309 | 334| 232 | 13911 | 123093 | 
| 22 | 11 lib/matplotlib/backends/backend_qt.py | 362 | 395| 344 | 14255 | 123093 | 
| 23 | 11 lib/matplotlib/backends/_backend_tk.py | 452 | 487| 359 | 14614 | 123093 | 
| 24 | 11 lib/matplotlib/backends/_backend_tk.py | 164 | 215| 592 | 15206 | 123093 | 
| 25 | 11 lib/matplotlib/backends/backend_gtk3.py | 153 | 202| 461 | 15667 | 123093 | 
| 26 | 11 lib/matplotlib/backends/backend_wx.py | 902 | 963| 585 | 16252 | 123093 | 
| 27 | 12 lib/matplotlib/backends/backend_gtk4.py | 31 | 89| 443 | 16695 | 127596 | 
| 28 | 13 examples/user_interfaces/mpl_with_glade3_sgskip.py | 1 | 52| 309 | 17004 | 127905 | 
| 29 | 13 examples/user_interfaces/embedding_in_qt_sgskip.py | 1 | 20| 121 | 17125 | 127905 | 
| 30 | 13 lib/matplotlib/figure.py | 3326 | 3368| 432 | 17557 | 127905 | 
| 31 | 13 lib/matplotlib/backends/_backend_tk.py | 489 | 523| 365 | 17922 | 127905 | 
| 32 | 13 lib/matplotlib/backends/_backend_tk.py | 992 | 1035| 292 | 18214 | 127905 | 
| 33 | 13 lib/matplotlib/backends/backend_qt.py | 997 | 1024| 194 | 18408 | 127905 | 
| 34 | 13 lib/matplotlib/figure.py | 2481 | 2501| 283 | 18691 | 127905 | 
| 35 | 13 lib/matplotlib/figure.py | 881 | 897| 247 | 18938 | 127905 | 
| 36 | 13 lib/matplotlib/figure.py | 3116 | 3147| 299 | 19237 | 127905 | 
| 37 | 13 lib/matplotlib/backends/backend_wx.py | 567 | 585| 182 | 19419 | 127905 | 
| 38 | 14 lib/matplotlib/backends/backend_gtk3agg.py | 50 | 75| 246 | 19665 | 128501 | 
| 39 | 15 lib/matplotlib/backends/_backend_gtk.py | 121 | 198| 602 | 20267 | 130978 | 
| 40 | 15 lib/matplotlib/backends/backend_wx.py | 888 | 899| 151 | 20418 | 130978 | 
| 41 | 15 lib/matplotlib/backends/backend_gtk4.py | 113 | 160| 440 | 20858 | 130978 | 
| 42 | 15 lib/matplotlib/figure.py | 2002 | 2037| 387 | 21245 | 130978 | 
| 43 | 15 lib/matplotlib/backends/backend_gtk3.py | 248 | 276| 282 | 21527 | 130978 | 
| 44 | 16 lib/matplotlib/backends/backend_tkcairo.py | 1 | 27| 224 | 21751 | 131203 | 
| 45 | 16 lib/matplotlib/backends/backend_wx.py | 433 | 491| 682 | 22433 | 131203 | 
| 46 | 17 lib/matplotlib/backends/backend_nbagg.py | 64 | 90| 197 | 22630 | 132973 | 
| 47 | 17 lib/matplotlib/backends/backend_qt.py | 427 | 437| 153 | 22783 | 132973 | 
| 48 | 17 lib/matplotlib/backends/_backend_tk.py | 371 | 412| 316 | 23099 | 132973 | 
| 49 | 17 lib/matplotlib/backends/backend_qt.py | 102 | 160| 532 | 23631 | 132973 | 
| 50 | 18 examples/subplots_axes_and_figures/subfigures.py | 1 | 83| 756 | 24387 | 134396 | 
| 51 | 18 lib/matplotlib/backends/backend_gtk4.py | 194 | 229| 337 | 24724 | 134396 | 
| 52 | 18 lib/matplotlib/backends/backend_wx.py | 965 | 979| 162 | 24886 | 134396 | 
| 53 | **18 lib/matplotlib/pyplot.py** | 1217 | 1281| 686 | 25572 | 134396 | 
| 54 | 18 lib/matplotlib/backends/backend_qt.py | 796 | 844| 431 | 26003 | 134396 | 
| **-> 55 <-** | **18 lib/matplotlib/pyplot.py** | 210 | 294| 750 | 26753 | 134396 | 
| 56 | 19 lib/matplotlib/backend_tools.py | 384 | 401| 124 | 26877 | 141820 | 
| 57 | 20 lib/matplotlib/backends/backend_ps.py | 929 | 986| 593 | 27470 | 154097 | 
| 58 | 21 lib/matplotlib/backends/backend_qt5.py | 1 | 29| 224 | 27694 | 154321 | 
| 59 | 21 lib/matplotlib/backends/backend_wx.py | 982 | 1045| 472 | 28166 | 154321 | 
| 60 | 21 lib/matplotlib/figure.py | 600 | 617| 190 | 28356 | 154321 | 
| 61 | 22 examples/event_handling/figure_axes_enter_leave.py | 1 | 53| 323 | 28679 | 154644 | 
| 62 | 22 lib/matplotlib/backends/backend_gtk3.py | 278 | 303| 172 | 28851 | 154644 | 
| 63 | 22 lib/matplotlib/figure.py | 2039 | 2060| 240 | 29091 | 154644 | 
| 64 | 22 lib/matplotlib/backends/backend_qt.py | 877 | 891| 128 | 29219 | 154644 | 
| 65 | 22 lib/matplotlib/backends/backend_macosx.py | 20 | 102| 749 | 29968 | 154644 | 
| 66 | 23 examples/pyplots/auto_subplots_adjust.py | 66 | 85| 140 | 30108 | 155346 | 
| 67 | 23 lib/matplotlib/backends/backend_gtk4.py | 231 | 256| 172 | 30280 | 155346 | 
| 68 | 24 tutorials/introductory/pyplot.py | 252 | 328| 910 | 31190 | 159822 | 
| 69 | 24 lib/matplotlib/backends/backend_qt.py | 1 | 81| 727 | 31917 | 159822 | 
| 70 | 25 examples/user_interfaces/web_application_server_sgskip.py | 1 | 73| 521 | 32438 | 160343 | 
| 71 | 25 lib/matplotlib/backends/_backend_tk.py | 1 | 56| 368 | 32806 | 160343 | 
| 72 | 25 lib/matplotlib/backends/backend_wx.py | 1282 | 1299| 159 | 32965 | 160343 | 
| 73 | **25 lib/matplotlib/pyplot.py** | 2203 | 2213| 137 | 33102 | 160343 | 
| 74 | 26 examples/user_interfaces/embedding_in_wx5_sgskip.py | 32 | 63| 288 | 33390 | 160838 | 
| 75 | 27 lib/matplotlib/backends/backend_cairo.py | 480 | 534| 496 | 33886 | 165378 | 
| 76 | 27 examples/subplots_axes_and_figures/subfigures.py | 84 | 149| 667 | 34553 | 165378 | 
| 77 | 27 lib/matplotlib/backends/_backend_tk.py | 229 | 244| 157 | 34710 | 165378 | 
| 78 | 28 examples/user_interfaces/embedding_in_wx4_sgskip.py | 42 | 81| 296 | 35006 | 165975 | 
| 79 | 29 lib/matplotlib/backends/backend_gtk4agg.py | 1 | 42| 305 | 35311 | 166281 | 
| 80 | 29 lib/matplotlib/backends/backend_wx.py | 419 | 431| 129 | 35440 | 166281 | 
| 81 | 29 lib/matplotlib/backends/backend_ps.py | 988 | 1007| 281 | 35721 | 166281 | 
| 82 | **30 lib/matplotlib/__init__.py** | 1 | 119| 742 | 36463 | 178095 | 
| 83 | 30 lib/matplotlib/figure.py | 2602 | 2637| 302 | 36765 | 178095 | 
| 84 | 30 lib/matplotlib/backends/backend_cairo.py | 428 | 445| 243 | 37008 | 178095 | 
| 85 | 30 lib/matplotlib/figure.py | 1869 | 1886| 260 | 37268 | 178095 | 
| 86 | 30 lib/matplotlib/backends/backend_gtk3.py | 221 | 246| 254 | 37522 | 178095 | 
| 87 | 30 lib/matplotlib/figure.py | 373 | 381| 154 | 37676 | 178095 | 
| 88 | 31 lib/matplotlib/backends/backend_wxcairo.py | 17 | 41| 225 | 37901 | 178471 | 
| 89 | 31 lib/matplotlib/backends/backend_macosx.py | 105 | 142| 379 | 38280 | 178471 | 
| **-> 90 <-** | **31 lib/matplotlib/pyplot.py** | 295 | 329| 312 | 38592 | 178471 | 
| 91 | 32 lib/matplotlib/backends/backend_tkagg.py | 1 | 21| 160 | 38752 | 178632 | 
| 92 | 32 lib/matplotlib/figure.py | 2311 | 2479| 1518 | 40270 | 178632 | 
| 93 | 33 examples/user_interfaces/embedding_in_wx3_sgskip.py | 93 | 146| 380 | 40650 | 179766 | 
| 94 | 33 lib/matplotlib/backends/_backend_tk.py | 301 | 317| 187 | 40837 | 179766 | 
| 95 | 33 examples/user_interfaces/embedding_in_wx3_sgskip.py | 24 | 54| 249 | 41086 | 179766 | 
| 96 | 34 lib/matplotlib/backends/backend_webagg_core.py | 157 | 200| 387 | 41473 | 183812 | 
| 97 | 35 examples/user_interfaces/embedding_in_gtk3_sgskip.py | 1 | 41| 246 | 41719 | 184058 | 
| 98 | 36 examples/user_interfaces/mathtext_wx_sgskip.py | 43 | 82| 304 | 42023 | 185118 | 
| 99 | 36 lib/matplotlib/backends/_backend_tk.py | 415 | 450| 309 | 42332 | 185118 | 
| 100 | 37 examples/user_interfaces/embedding_in_tk_sgskip.py | 1 | 68| 466 | 42798 | 185584 | 
| 101 | 37 lib/matplotlib/backends/backend_gtk3agg.py | 1 | 48| 375 | 43173 | 185584 | 
| 102 | 37 lib/matplotlib/backends/backend_cairo.py | 447 | 459| 149 | 43322 | 185584 | 
| 103 | 37 lib/matplotlib/backends/backend_nbagg.py | 109 | 160| 342 | 43664 | 185584 | 
| 104 | 37 lib/matplotlib/figure.py | 2255 | 2276| 149 | 43813 | 185584 | 
| 105 | 37 lib/matplotlib/figure.py | 1254 | 1281| 323 | 44136 | 185584 | 
| 106 | 37 lib/matplotlib/figure.py | 3049 | 3079| 209 | 44345 | 185584 | 
| 107 | 37 lib/matplotlib/figure.py | 383 | 391| 150 | 44495 | 185584 | 
| 108 | **37 lib/matplotlib/pyplot.py** | 919 | 936| 135 | 44630 | 185584 | 
| 109 | 37 examples/user_interfaces/embedding_in_wx5_sgskip.py | 1 | 29| 207 | 44837 | 185584 | 
| 110 | 37 lib/matplotlib/backends/backend_qt.py | 986 | 994| 108 | 44945 | 185584 | 
| 111 | 37 lib/matplotlib/backends/_backend_tk.py | 217 | 227| 167 | 45112 | 185584 | 
| 112 | 37 lib/matplotlib/figure.py | 722 | 760| 442 | 45554 | 185584 | 
| 113 | 38 examples/user_interfaces/embedding_in_gtk4_sgskip.py | 1 | 46| 300 | 45854 | 185884 | 
| 114 | 39 lib/matplotlib/backends/qt_compat.py | 193 | 270| 779 | 46633 | 188284 | 
| 115 | 39 lib/matplotlib/backends/backend_webagg_core.py | 429 | 520| 641 | 47274 | 188284 | 
| 116 | 39 lib/matplotlib/_pylab_helpers.py | 1 | 42| 277 | 47551 | 188284 | 
| 117 | 39 lib/matplotlib/backends/_backend_gtk.py | 200 | 213| 154 | 47705 | 188284 | 
| 118 | 39 lib/matplotlib/backends/_backend_tk.py | 290 | 299| 128 | 47833 | 188284 | 
| 119 | 40 examples/user_interfaces/pylab_with_gtk3_sgskip.py | 1 | 61| 388 | 48221 | 188672 | 
| 120 | 40 lib/matplotlib/backends/qt_compat.py | 71 | 100| 305 | 48526 | 188672 | 
| 121 | 40 lib/matplotlib/figure.py | 1102 | 1124| 242 | 48768 | 188672 | 
| 122 | 40 lib/matplotlib/backend_bases.py | 2387 | 2527| 908 | 49676 | 188672 | 


### Hint

```
This bisects to #12637, and is essentially due to the fact that we now initialize ipython/matplotlib support when the first canvas is created (here, by `plt.figure()`), that during initialization, ipython calls `switch_backend`, that `switch_backend` starts by calling `close("all")`, and that NXPlotView() is registered with pyplot and thus gets closed at that point.

I think we can actually remove the `close("all")` (well, modulo backcompat, yada yada)?  If there are conflicting event loops, closing figures won't help, if there aren't (like in this case), we don't need to close them.

In the meantime you can probably get away with creating a figure and immediately closing it -- we only initialize ipython/matplotlib support once.
Thanks for the information. I am subclassing FigureCanvasQT, so I would prefer to preempt the `close("all")` call, perhaps by calling`switch_backend` myself for the first instance, but I can't see how it gets called during the first `plt.figure()` call. I'll look again tomorrow, but if you have any tips, I would appreciate them. I wondered if it was hidden in the `switch_backend_decorator`, but I don't see that being used.
`switch_backend` is called via `ip.enable_matplotlib()`, which is called in `FigureCanvasBase._fix_ipython_backend2gui`, which is called in the FigureCanvasBase constructor.
I think removing that `close("all")` might be the only solution for me. I have tried endless permutations of preemptively setting mpl.rcParamsOrig('backend'), calling `pt.enable_matplotlib` or `pt.activate_matplotlib`, adding a dummy `_fix_ipython_backend2gui` to my FigureCanvasQT subclass, even defining my own subclass of _BackendQT5 using the `_Backend.export` decorator (which produces unhelpful side effects) , but none of them stop IPython from calling `switch_backend`, whose first line is `close("all")`. I need to make a new release of NeXpy in the next couple of days, but I am going to have to make Matplotlib v3.0.3 the highest allowed version, unless anyone has other tips for me to try.
It looks like the following workaround does suffice: add a call to `FigureCanvas(Figure())` at the toplevel of plotview.py (note that because the figure is not registered with pyplot, it will be immediately garbage collected anyways, but that'll arrange to call _fix_ipython_backend2gui properly).

(I still do think that we should not call `close("all")`, but that may take longer to change...)
Thanks for the suggestion, which I think would work. In the past few minutes, I have just found an alternative solution. I just have to monkey-patch the IPython InteractiveShell class to do nothing when there is a call to `enable_matplotlib`. Everything else is already initialized so the call is redundant. It also seems to work. 

I haven't thought about this issue nearly as long as you, but I get the impression that the ultimate solution would have to come from IPython allowing alternative backends to be registered in some well-documented way. 
The way we are registering backends with IPython definitely borders API abuse, sorry for breaking things on your side.  On the other hand, I would rather have as much of that code as possible living on Matplotlib's side, as cross-project coordination is a much harder problem...
I remilestone to 3.2 as we have a workaround now, although I do think the API needs to be revisited on our side.
I think the simplest short-term fix would be to add an optional keyword argument to the FigureCanvasBase to allow instances to skip the call to `_fix_ipython_backend2gui` if it's unnecessary. If you are loath to change call signatures, the same thing could be achieved by adding a private class variable (`_fix_ipython=True`) that a subclass could override. 
That doesn't seem much better than just documenting "call `FigureCanvas(Figure())` early"?
I am a little rushed at the moment, so I may be wrong about this, but I think the problem with calling `FigureCanvas(Figure())` in NeXpy is that the IPython shell doesn't exist when I initialize the plotting windows, so the call to `IPython.get_ipython()` would fail. I could probably reconfigure the initialization process I would prefer not to refactor my code if possible, and I think Matplotlib should allow for situations where the IPython fix is completely unnecessary.
As it is written right now, `FigureCanvas(Figure())` will call `_fix_ipython_backend2gui` regardless of whether IPython was already initialized, and `_fix_ipython_backend2gui` will *not* be called a second time (due to the `functools.lru_cache()` decorator) even if IPython is later initialized (I didn't think about the embedding case at all when I wrote this).  So the fix works (... at least for me) even if you call `FigureCanvas(Figure())` at the toplevel of the module.
I must admit that I had been puzzled about how subsequent calls were suppressed. I didn't know what the `functools.lru_cache()` decorator did. I am happy to leave this to real Matplotlib experts such as you now that I have my own solution and you are aware of the issue. If you reference this issue when any relevant changes are implemented, I should be able to keep my own code compatible. This isn't the first bit of monkey-patching I've had to do - it's one of the hazards of embedding other packages.
```

## Patch

```diff
diff --git a/lib/matplotlib/__init__.py b/lib/matplotlib/__init__.py
--- a/lib/matplotlib/__init__.py
+++ b/lib/matplotlib/__init__.py
@@ -1105,6 +1105,10 @@ def use(backend, *, force=True):
     """
     Select the backend used for rendering and GUI integration.
 
+    If pyplot is already imported, `~matplotlib.pyplot.switch_backend` is used
+    and if the new backend is different than the current backend, all Figures
+    will be closed.
+
     Parameters
     ----------
     backend : str
@@ -1135,6 +1139,8 @@ def use(backend, *, force=True):
     --------
     :ref:`backends`
     matplotlib.get_backend
+    matplotlib.pyplot.switch_backend
+
     """
     name = validate_backend(backend)
     # don't (prematurely) resolve the "auto" backend setting
diff --git a/lib/matplotlib/pyplot.py b/lib/matplotlib/pyplot.py
--- a/lib/matplotlib/pyplot.py
+++ b/lib/matplotlib/pyplot.py
@@ -209,21 +209,24 @@ def _get_backend_mod():
 
 def switch_backend(newbackend):
     """
-    Close all open figures and set the Matplotlib backend.
+    Set the pyplot backend.
 
-    The argument is case-insensitive.  Switching to an interactive backend is
-    possible only if no event loop for another interactive backend has started.
-    Switching to and from non-interactive backends is always possible.
+    Switching to an interactive backend is possible only if no event loop for
+    another interactive backend has started.  Switching to and from
+    non-interactive backends is always possible.
+
+    If the new backend is different than the current backend then all open
+    Figures will be closed via ``plt.close('all')``.
 
     Parameters
     ----------
     newbackend : str
-        The name of the backend to use.
+        The case-insensitive name of the backend to use.
+
     """
     global _backend_mod
     # make sure the init is pulled up so we can assign to it later
     import matplotlib.backends
-    close("all")
 
     if newbackend is rcsetup._auto_backend_sentinel:
         current_framework = cbook._get_running_interactive_framework()
@@ -260,6 +263,8 @@ def switch_backend(newbackend):
             switch_backend("agg")
             rcParamsOrig["backend"] = "agg"
             return
+    # have to escape the switch on access logic
+    old_backend = dict.__getitem__(rcParams, 'backend')
 
     backend_mod = importlib.import_module(
         cbook._backend_module_name(newbackend))
@@ -323,6 +328,8 @@ def draw_if_interactive():
     # Need to keep a global reference to the backend for compatibility reasons.
     # See https://github.com/matplotlib/matplotlib/issues/6092
     matplotlib.backends.backend = newbackend
+    if not cbook._str_equal(old_backend, newbackend):
+        close("all")
 
     # make sure the repl display hook is installed in case we become
     # interactive

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_pyplot.py b/lib/matplotlib/tests/test_pyplot.py
--- a/lib/matplotlib/tests/test_pyplot.py
+++ b/lib/matplotlib/tests/test_pyplot.py
@@ -398,3 +398,14 @@ def test_minor_ticks():
     tick_labels = ax.get_yticklabels(minor=True)
     assert np.all(tick_pos == np.array([3.5, 6.5]))
     assert [l.get_text() for l in tick_labels] == ['a', 'b']
+
+
+def test_switch_backend_no_close():
+    plt.switch_backend('agg')
+    fig = plt.figure()
+    fig = plt.figure()
+    assert len(plt.get_fignums()) == 2
+    plt.switch_backend('agg')
+    assert len(plt.get_fignums()) == 2
+    plt.switch_backend('svg')
+    assert len(plt.get_fignums()) == 0

```


## Code snippets

### 1 - lib/matplotlib/_pylab_helpers.py:

Start line: 69, End line: 136

```python
class Gcf:

    @classmethod
    def destroy_fig(cls, fig):
        """Destroy figure *fig*."""
        num = next((manager.num for manager in cls.figs.values()
                    if manager.canvas.figure == fig), None)
        if num is not None:
            cls.destroy(num)

    @classmethod
    def destroy_all(cls):
        """Destroy all figures."""
        for manager in list(cls.figs.values()):
            manager.canvas.mpl_disconnect(manager._cidgcf)
            manager.destroy()
        cls.figs.clear()

    @classmethod
    def has_fignum(cls, num):
        """Return whether figure number *num* exists."""
        return num in cls.figs

    @classmethod
    def get_all_fig_managers(cls):
        """Return a list of figure managers."""
        return list(cls.figs.values())

    @classmethod
    def get_num_fig_managers(cls):
        """Return the number of figures being managed."""
        return len(cls.figs)

    @classmethod
    def get_active(cls):
        """Return the active manager, or *None* if there is no manager."""
        return next(reversed(cls.figs.values())) if cls.figs else None

    @classmethod
    def _set_new_active_manager(cls, manager):
        """Adopt *manager* into pyplot and make it the active manager."""
        if not hasattr(manager, "_cidgcf"):
            manager._cidgcf = manager.canvas.mpl_connect(
                "button_press_event", lambda event: cls.set_active(manager))
        fig = manager.canvas.figure
        fig.number = manager.num
        label = fig.get_label()
        if label:
            manager.set_window_title(label)
        cls.set_active(manager)

    @classmethod
    def set_active(cls, manager):
        """Make *manager* the active manager."""
        cls.figs[manager.num] = manager
        cls.figs.move_to_end(manager.num)

    @classmethod
    def draw_all(cls, force=False):
        """
        Redraw all stale managed figures, or, if *force* is True, all managed
        figures.
        """
        for manager in cls.get_all_fig_managers():
            if force or manager.canvas.figure.stale:
                manager.canvas.draw_idle()


atexit.register(Gcf.destroy_all)
```
### 2 - lib/matplotlib/backends/backend_qt.py:

Start line: 439, End line: 459

```python
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):

    def blit(self, bbox=None):
        # docstring inherited
        if bbox is None and self.figure:
            bbox = self.figure.bbox  # Blit the entire canvas if bbox is None.
        # repaint uses logical pixels, not physical pixels like the renderer.
        l, b, w, h = [int(pt / self.device_pixel_ratio) for pt in bbox.bounds]
        t = b + h
        self.repaint(l, self.rect().height() - t, w, h)

    def _draw_idle(self):
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.height() < 0 or self.width() < 0:
                return
            try:
                self.draw()
            except Exception:
                # Uncaught exceptions are fatal for PyQt5, so catch them.
                traceback.print_exc()
```
### 3 - lib/matplotlib/backends/backend_qt.py:

Start line: 192, End line: 248

```python
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):
    required_interactive_framework = "qt"
    _timer_cls = TimerQT
    manager_class = _api.classproperty(lambda cls: FigureManagerQT)

    buttond = {
        getattr(_enum("QtCore.Qt.MouseButton"), k): v for k, v in [
            ("LeftButton", MouseButton.LEFT),
            ("RightButton", MouseButton.RIGHT),
            ("MiddleButton", MouseButton.MIDDLE),
            ("XButton1", MouseButton.BACK),
            ("XButton2", MouseButton.FORWARD),
        ]
    }

    def __init__(self, figure=None):
        _create_qApp()
        super().__init__(figure=figure)

        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None
        self._in_resize_event = False

        self.setAttribute(
            _enum("QtCore.Qt.WidgetAttribute").WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.resize(*self.get_width_height())

        palette = QtGui.QPalette(QtGui.QColor("white"))
        self.setPalette(palette)

    def _update_pixel_ratio(self):
        if self._set_device_pixel_ratio(_devicePixelRatioF(self)):
            # The easiest way to resize the canvas is to emit a resizeEvent
            # since we implement all the logic for resizing the canvas for
            # that event.
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)

    def _update_screen(self, screen):
        # Handler for changes to a window's attached screen.
        self._update_pixel_ratio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._update_pixel_ratio)
            screen.logicalDotsPerInchChanged.connect(self._update_pixel_ratio)

    def showEvent(self, event):
        # Set up correct pixel ratio, and connect to any signal changes for it,
        # once the window is shown (and thus has these attributes).
        window = self.window().windowHandle()
        window.screenChanged.connect(self._update_screen)
        self._update_screen(window.screen())

    def set_cursor(self, cursor):
        # docstring inherited
        self.setCursor(_api.check_getitem(cursord, cursor=cursor))
```
### 4 - lib/matplotlib/backends/_backend_tk.py:

Start line: 525, End line: 559

```python
class FigureManagerTk(FigureManagerBase):

    def destroy(self, *args):
        if self.canvas._idle_draw_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._idle_draw_id)
        if self.canvas._event_loop_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._event_loop_id)
        if self._window_dpi_cbname:
            self._window_dpi.trace_remove('write', self._window_dpi_cbname)

        # NOTE: events need to be flushed before issuing destroy (GH #9956),
        # however, self.window.update() can break user code. An async callback
        # is the safest way to achieve a complete draining of the event queue,
        # but it leaks if no tk event loop is running. Therefore we explicitly
        # check for an event loop and choose our best guess.
        def delayed_destroy():
            self.window.destroy()

            if self._owns_mainloop and not Gcf.get_num_fig_managers():
                self.window.quit()

        if cbook._get_running_interactive_framework() == "tk":
            # "after idle after 0" avoids Tcl error/race (GH #19940)
            self.window.after_idle(self.window.after, 0, delayed_destroy)
        else:
            self.window.update()
            delayed_destroy()

    def get_window_title(self):
        return self.window.wm_title()

    def set_window_title(self, title):
        self.window.wm_title(title)

    def full_screen_toggle(self):
        is_fullscreen = bool(self.window.attributes('-fullscreen'))
        self.window.attributes('-fullscreen', not is_fullscreen)
```
### 5 - lib/matplotlib/backends/backend_qt.py:

Start line: 272, End line: 307

```python
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):

    def enterEvent(self, event):
        LocationEvent("figure_enter_event", self,
                      *self.mouseEventCoords(event),
                      guiEvent=event)._process()

    def leaveEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        LocationEvent("figure_leave_event", self,
                      *self.mouseEventCoords(),
                      guiEvent=event)._process()

    def mousePressEvent(self, event):
        button = self.buttond.get(event.button())
        if button is not None:
            MouseEvent("button_press_event", self,
                       *self.mouseEventCoords(event), button,
                       guiEvent=event)._process()

    def mouseDoubleClickEvent(self, event):
        button = self.buttond.get(event.button())
        if button is not None:
            MouseEvent("button_press_event", self,
                       *self.mouseEventCoords(event), button, dblclick=True,
                       guiEvent=event)._process()

    def mouseMoveEvent(self, event):
        MouseEvent("motion_notify_event", self,
                   *self.mouseEventCoords(event),
                   guiEvent=event)._process()

    def mouseReleaseEvent(self, event):
        button = self.buttond.get(event.button())
        if button is not None:
            MouseEvent("button_release_event", self,
                       *self.mouseEventCoords(event), button,
                       guiEvent=event)._process()
```
### 6 - lib/matplotlib/pyplot.py:

Start line: 734, End line: 793

```python
@_api.make_keyword_only("3.6", "facecolor")
def figure(num=None,  # autoincrement if None, else integer from 1-N
           figsize=None,  # defaults to rc figure.figsize
           dpi=None,  # defaults to rc figure.dpi
           facecolor=None,  # defaults to rc figure.facecolor
           edgecolor=None,  # defaults to rc figure.edgecolor
           frameon=True,
           FigureClass=Figure,
           clear=False,
           **kwargs
           ):
    if isinstance(num, FigureBase):
        if num.canvas.manager is None:
            raise ValueError("The passed figure is not managed by pyplot")
        _pylab_helpers.Gcf.set_active(num.canvas.manager)
        return num.figure

    allnums = get_fignums()
    next_num = max(allnums) + 1 if allnums else 1
    fig_label = ''
    if num is None:
        num = next_num
    elif isinstance(num, str):
        fig_label = num
        all_labels = get_figlabels()
        if fig_label not in all_labels:
            if fig_label == 'all':
                _api.warn_external("close('all') closes all existing figures.")
            num = next_num
        else:
            inum = all_labels.index(fig_label)
            num = allnums[inum]
    else:
        num = int(num)  # crude validation of num argument

    manager = _pylab_helpers.Gcf.get_fig_manager(num)
    if manager is None:
        max_open_warning = rcParams['figure.max_open_warning']
        if len(allnums) == max_open_warning >= 1:
            _api.warn_external(
                f"More than {max_open_warning} figures have been opened. "
                f"Figures created through the pyplot interface "
                f"(`matplotlib.pyplot.figure`) are retained until explicitly "
                f"closed and may consume too much memory. (To control this "
                f"warning, see the rcParam `figure.max_open_warning`). "
                f"Consider using `matplotlib.pyplot.close()`.",
                RuntimeWarning)

        manager = new_figure_manager(
            num, figsize=figsize, dpi=dpi,
            facecolor=facecolor, edgecolor=edgecolor, frameon=frameon,
            FigureClass=FigureClass, **kwargs)
        fig = manager.canvas.figure
        if fig_label:
            fig.set_label(fig_label)

        _pylab_helpers.Gcf._set_new_active_manager(manager)

        # make sure backends (inline) that we don't ship that expect this
        # to be called in plotting commands to make the figure call show
        # still work.  There is probably a better way to do this in the
        # FigureManager base class.
        draw_if_interactive()

        if _REPL_DISPLAYHOOK is _ReplDisplayHook.PLAIN:
            fig.stale_callback = _auto_draw_if_interactive

    if clear:
        manager.canvas.figure.clear()

    return manager.canvas.figure
```
### 7 - lib/matplotlib/backends/backend_qt.py:

Start line: 397, End line: 425

```python
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):

    def flush_events(self):
        # docstring inherited
        QtWidgets.QApplication.instance().processEvents()

    def start_event_loop(self, timeout=0):
        # docstring inherited
        if hasattr(self, "_event_loop") and self._event_loop.isRunning():
            raise RuntimeError("Event loop already running")
        self._event_loop = event_loop = QtCore.QEventLoop()
        if timeout > 0:
            _ = QtCore.QTimer.singleShot(int(timeout * 1000), event_loop.quit)

        with _maybe_allow_interrupt(event_loop):
            qt_compat._exec(event_loop)

    def stop_event_loop(self, event=None):
        # docstring inherited
        if hasattr(self, "_event_loop"):
            self._event_loop.quit()

    def draw(self):
        """Render the figure, and queue a request for a Qt draw."""
        # The renderer draw is done here; delaying causes problems with code
        # that uses the result of the draw() to update plot elements.
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()
```
### 8 - lib/matplotlib/backends/backend_qt.py:

Start line: 336, End line: 360

```python
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):

    def resizeEvent(self, event):
        if self._in_resize_event:  # Prevent PyQt6 recursion
            return
        self._in_resize_event = True
        try:
            w = event.size().width() * self.device_pixel_ratio
            h = event.size().height() * self.device_pixel_ratio
            dpival = self.figure.dpi
            winch = w / dpival
            hinch = h / dpival
            self.figure.set_size_inches(winch, hinch, forward=False)
            # pass back into Qt to let it finish
            QtWidgets.QWidget.resizeEvent(self, event)
            # emit our resize events
            ResizeEvent("resize_event", self)._process()
            self.draw_idle()
        finally:
            self._in_resize_event = False

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minumumSizeHint(self):
        return QtCore.QSize(10, 10)
```
### 9 - lib/matplotlib/backends/backend_qt.py:

Start line: 461, End line: 500

```python
class FigureCanvasQT(FigureCanvasBase, QtWidgets.QWidget):

    def drawRectangle(self, rect):
        # Draw the zoom rectangle to the QPainter.  _draw_rect_callback needs
        # to be called at the end of paintEvent.
        if rect is not None:
            x0, y0, w, h = [int(pt / self.device_pixel_ratio) for pt in rect]
            x1 = x0 + w
            y1 = y0 + h
            def _draw_rect_callback(painter):
                pen = QtGui.QPen(
                    QtGui.QColor("black"),
                    1 / self.device_pixel_ratio
                )

                pen.setDashPattern([3, 3])
                for color, offset in [
                        (QtGui.QColor("black"), 0),
                        (QtGui.QColor("white"), 3),
                ]:
                    pen.setDashOffset(offset)
                    pen.setColor(color)
                    painter.setPen(pen)
                    # Draw the lines from x0, y0 towards x1, y1 so that the
                    # dashes don't "jump" when moving the zoom box.
                    painter.drawLine(x0, y0, x0, y1)
                    painter.drawLine(x0, y0, x1, y0)
                    painter.drawLine(x0, y1, x1, y1)
                    painter.drawLine(x1, y0, x1, y1)
        else:
            def _draw_rect_callback(painter):
                return
        self._draw_rect_callback = _draw_rect_callback
        self.update()


class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.Signal()

    def closeEvent(self, event):
        self.closing.emit()
        super().closeEvent(event)
```
### 10 - lib/matplotlib/backends/backend_qt.py:

Start line: 503, End line: 609

```python
class FigureManagerQT(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : qt.QToolBar
        The qt.QToolBar
    window : qt.QMainWindow
        The qt.QMainWindow
    """

    def __init__(self, canvas, num):
        self.window = MainWindow()
        super().__init__(canvas, num)
        self.window.closing.connect(
            # The lambda prevents the event from being immediately gc'd.
            lambda: CloseEvent("close_event", self.canvas)._process())
        self.window.closing.connect(self._widgetclosed)

        if sys.platform != "darwin":
            image = str(cbook._get_data_path('images/matplotlib.svg'))
            icon = QtGui.QIcon(image)
            self.window.setWindowIcon(icon)

        self.window._destroying = False

        if self.toolbar:
            self.window.addToolBar(self.toolbar)
            tbs_height = self.toolbar.sizeHint().height()
        else:
            tbs_height = 0

        # resize the main window so it will display the canvas with the
        # requested size:
        cs = canvas.sizeHint()
        cs_height = cs.height()
        height = cs_height + tbs_height
        self.window.resize(cs.width(), height)

        self.window.setCentralWidget(self.canvas)

        if mpl.is_interactive():
            self.window.show()
            self.canvas.draw_idle()

        # Give the keyboard focus to the figure instead of the manager:
        # StrongFocus accepts both tab and click to focus and will enable the
        # canvas to process event without clicking.
        # https://doc.qt.io/qt-5/qt.html#FocusPolicy-enum
        self.canvas.setFocusPolicy(_enum("QtCore.Qt.FocusPolicy").StrongFocus)
        self.canvas.setFocus()

        self.window.raise_()

    def full_screen_toggle(self):
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    def _widgetclosed(self):
        if self.window._destroying:
            return
        self.window._destroying = True
        try:
            Gcf.destroy(self)
        except AttributeError:
            pass
            # It seems that when the python session is killed,
            # Gcf can get destroyed before the Gcf.destroy
            # line is run, leading to a useless AttributeError.

    def resize(self, width, height):
        # The Qt methods return sizes in 'virtual' pixels so we do need to
        # rescale from physical to logical pixels.
        width = int(width / self.canvas.device_pixel_ratio)
        height = int(height / self.canvas.device_pixel_ratio)
        extra_width = self.window.width() - self.canvas.width()
        extra_height = self.window.height() - self.canvas.height()
        self.canvas.resize(width, height)
        self.window.resize(width + extra_width, height + extra_height)

    def show(self):
        self.window.show()
        if mpl.rcParams['figure.raise_window']:
            self.window.activateWindow()
            self.window.raise_()

    def destroy(self, *args):
        # check for qApp first, as PySide deletes it in its atexit handler
        if QtWidgets.QApplication.instance() is None:
            return
        if self.window._destroying:
            return
        self.window._destroying = True
        if self.toolbar:
            self.toolbar.destroy()
        self.window.close()

    def get_window_title(self):
        return self.window.windowTitle()

    def set_window_title(self, title):
        self.window.setWindowTitle(title)
```
### 16 - lib/matplotlib/pyplot.py:

Start line: 1, End line: 89

```python
# Note: The first part of this file can be modified in place, but the latter
# part is autogenerated by the boilerplate.py script.

"""
`matplotlib.pyplot` is a state-based interface to matplotlib. It provides
an implicit,  MATLAB-like, way of plotting.  It also opens figures on your
screen, and acts as the figure GUI manager.

pyplot is mainly intended for interactive plots and simple cases of
programmatic plot generation::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

The explicit object-oriented API is recommended for complex plots, though
pyplot is still usually used to create the figure and often the axes in the
figure. See `.pyplot.figure`, `.pyplot.subplots`, and
`.pyplot.subplot_mosaic` to create figures, and
:doc:`Axes API </api/axes_api>` for the plotting methods on an Axes::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)


See :ref:`api_interfaces` for an explanation of the tradeoffs between the
implicit and explicit interfaces.
"""

from contextlib import ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
from numbers import Number
import re
import sys
import threading
import time

from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import rcsetup, style
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import FigureCanvasBase, MouseButton
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcParams, rcParamsDefault, get_backend, rcParamsOrig
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names

from matplotlib import cm
from matplotlib.cm import _colormaps as colormaps, register_cmap
from matplotlib.colors import _color_sequences as color_sequences

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget

from .ticker import (
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator)

_log = logging.getLogger(__name__)
```
### 17 - lib/matplotlib/pyplot.py:

Start line: 332, End line: 360

```python
def _warn_if_gui_out_of_main_thread():
    # This compares native thread ids because even if python-level Thread
    # objects match, the underlying OS thread (which is what really matters)
    # may be different on Python implementations with green threads.
    if (_get_required_interactive_framework(_get_backend_mod()) and
            threading.get_native_id() != threading.main_thread().native_id):
        _api.warn_external(
            "Starting a Matplotlib GUI outside of the main thread will likely "
            "fail.")


# This function's signature is rewritten upon backend-load by switch_backend.
def new_figure_manager(*args, **kwargs):
    """Create a new figure manager instance."""
    _warn_if_gui_out_of_main_thread()
    return _get_backend_mod().new_figure_manager(*args, **kwargs)


# This function's signature is rewritten upon backend-load by switch_backend.
def draw_if_interactive(*args, **kwargs):
    """
    Redraw the current figure if in interactive mode.

    .. warning::

        End users will typically not have to call this function because the
        the interactive mode takes care of this.
    """
    return _get_backend_mod().draw_if_interactive(*args, **kwargs)
```
### 53 - lib/matplotlib/pyplot.py:

Start line: 1217, End line: 1281

```python
@_docstring.dedent_interpd
def subplot(*args, **kwargs):
    # Here we will only normalize `polar=True` vs `projection='polar'` and let
    # downstream code deal with the rest.
    unset = object()
    projection = kwargs.get('projection', unset)
    polar = kwargs.pop('polar', unset)
    if polar is not unset and polar:
        # if we got mixed messages from the user, raise
        if projection is not unset and projection != 'polar':
            raise ValueError(
                f"polar={polar}, yet projection={projection!r}. "
                "Only one of these arguments should be supplied."
            )
        kwargs['projection'] = projection = 'polar'

    # if subplot called without arguments, create subplot(1, 1, 1)
    if len(args) == 0:
        args = (1, 1, 1)

    # This check was added because it is very easy to type subplot(1, 2, False)
    # when subplots(1, 2, False) was intended (sharex=False, that is). In most
    # cases, no error will ever occur, but mysterious behavior can result
    # because what was intended to be the sharex argument is instead treated as
    # a subplot index for subplot()
    if len(args) >= 3 and isinstance(args[2], bool):
        _api.warn_external("The subplot index argument to subplot() appears "
                           "to be a boolean. Did you intend to use "
                           "subplots()?")
    # Check for nrows and ncols, which are not valid subplot args:
    if 'nrows' in kwargs or 'ncols' in kwargs:
        raise TypeError("subplot() got an unexpected keyword argument 'ncols' "
                        "and/or 'nrows'.  Did you intend to call subplots()?")

    fig = gcf()

    # First, search for an existing subplot with a matching spec.
    key = SubplotSpec._from_subplot_args(fig, args)

    for ax in fig.axes:
        # if we found an Axes at the position sort out if we can re-use it
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() == key:
            # if the user passed no kwargs, re-use
            if kwargs == {}:
                break
            # if the axes class and kwargs are identical, reuse
            elif ax._projection_init == fig._process_projection_requirements(
                *args, **kwargs
            ):
                break
    else:
        # we have exhausted the known Axes and none match, make a new one!
        ax = fig.add_subplot(*args, **kwargs)

    fig.sca(ax)

    axes_to_delete = [other for other in fig.axes
                      if other != ax and ax.bbox.fully_overlaps(other.bbox)]
    if axes_to_delete:
        _api.warn_deprecated(
            "3.6", message="Auto-removal of overlapping axes is deprecated "
            "since %(since)s and will be removed %(removal)s; explicitly call "
            "ax.remove() as needed.")
    for ax_to_del in axes_to_delete:
        delaxes(ax_to_del)

    return ax
```
### 55 - lib/matplotlib/pyplot.py:

Start line: 210, End line: 294

```python
def switch_backend(newbackend):
    """
    Close all open figures and set the Matplotlib backend.

    The argument is case-insensitive.  Switching to an interactive backend is
    possible only if no event loop for another interactive backend has started.
    Switching to and from non-interactive backends is always possible.

    Parameters
    ----------
    newbackend : str
        The name of the backend to use.
    """
    global _backend_mod
    # make sure the init is pulled up so we can assign to it later
    import matplotlib.backends
    close("all")

    if newbackend is rcsetup._auto_backend_sentinel:
        current_framework = cbook._get_running_interactive_framework()
        mapping = {'qt': 'qtagg',
                   'gtk3': 'gtk3agg',
                   'gtk4': 'gtk4agg',
                   'wx': 'wxagg',
                   'tk': 'tkagg',
                   'macosx': 'macosx',
                   'headless': 'agg'}

        best_guess = mapping.get(current_framework, None)
        if best_guess is not None:
            candidates = [best_guess]
        else:
            candidates = []
        candidates += [
            "macosx", "qtagg", "gtk4agg", "gtk3agg", "tkagg", "wxagg"]

        # Don't try to fallback on the cairo-based backends as they each have
        # an additional dependency (pycairo) over the agg-based backend, and
        # are of worse quality.
        for candidate in candidates:
            try:
                switch_backend(candidate)
            except ImportError:
                continue
            else:
                rcParamsOrig['backend'] = candidate
                return
        else:
            # Switching to Agg should always succeed; if it doesn't, let the
            # exception propagate out.
            switch_backend("agg")
            rcParamsOrig["backend"] = "agg"
            return

    backend_mod = importlib.import_module(
        cbook._backend_module_name(newbackend))
    canvas_class = backend_mod.FigureCanvas

    required_framework = _get_required_interactive_framework(backend_mod)
    if required_framework is not None:
        current_framework = cbook._get_running_interactive_framework()
        if (current_framework and required_framework
                and current_framework != required_framework):
            raise ImportError(
                "Cannot load backend {!r} which requires the {!r} interactive "
                "framework, as {!r} is currently running".format(
                    newbackend, required_framework, current_framework))

    # Load the new_figure_manager() and show() functions from the backend.

    # Classically, backends can directly export these functions.  This should
    # keep working for backcompat.
    new_figure_manager = getattr(backend_mod, "new_figure_manager", None)
    # show = getattr(backend_mod, "show", None)
    # In that classical approach, backends are implemented as modules, but
    # "inherit" default method implementations from backend_bases._Backend.
    # This is achieved by creating a "class" that inherits from
    # backend_bases._Backend and whose body is filled with the module globals.
    class backend_mod(matplotlib.backend_bases._Backend):
        locals().update(vars(backend_mod))

    # However, the newer approach for defining new_figure_manager (and, in
    # the future, show) is to derive them from canvas methods.  In that case,
    # also update backend_mod accordingly; also, per-backend customization of
    # draw_if_interactive is disabled.
    # ... other code
```
### 73 - lib/matplotlib/pyplot.py:

Start line: 2203, End line: 2213

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
### 82 - lib/matplotlib/__init__.py:

Start line: 1, End line: 119

```python
"""
An object-oriented plotting library.

A procedural interface is provided by the companion pyplot module,
which may be imported directly, e.g.::

    import matplotlib.pyplot as plt

or using ipython::

    ipython

at your terminal, followed by::

    In [1]: %matplotlib
    In [2]: import matplotlib.pyplot as plt

at the ipython shell prompt.

For the most part, direct use of the explicit object-oriented library is
encouraged when programming; the implicit pyplot interface is primarily for
working interactively. The exceptions to this suggestion are the pyplot
functions `.pyplot.figure`, `.pyplot.subplot`, `.pyplot.subplots`, and
`.pyplot.savefig`, which can greatly simplify scripting.  See
:ref:`api_interfaces` for an explanation of the tradeoffs between the implicit
and explicit interfaces.

Modules include:

    :mod:`matplotlib.axes`
        The `~.axes.Axes` class.  Most pyplot functions are wrappers for
        `~.axes.Axes` methods.  The axes module is the highest level of OO
        access to the library.

    :mod:`matplotlib.figure`
        The `.Figure` class.

    :mod:`matplotlib.artist`
        The `.Artist` base class for all classes that draw things.

    :mod:`matplotlib.lines`
        The `.Line2D` class for drawing lines and markers.

    :mod:`matplotlib.patches`
        Classes for drawing polygons.

    :mod:`matplotlib.text`
        The `.Text` and `.Annotation` classes.

    :mod:`matplotlib.image`
        The `.AxesImage` and `.FigureImage` classes.

    :mod:`matplotlib.collections`
        Classes for efficient drawing of groups of lines or polygons.

    :mod:`matplotlib.colors`
        Color specifications and making colormaps.

    :mod:`matplotlib.cm`
        Colormaps, and the `.ScalarMappable` mixin class for providing color
        mapping functionality to other classes.

    :mod:`matplotlib.ticker`
        Calculation of tick mark locations and formatting of tick labels.

    :mod:`matplotlib.backends`
        A subpackage with modules for various GUI libraries and output formats.

The base matplotlib namespace includes:

    `~matplotlib.rcParams`
        Default configuration settings; their defaults may be overridden using
        a :file:`matplotlibrc` file.

    `~matplotlib.use`
        Setting the Matplotlib backend.  This should be called before any
        figure is created, because it is not possible to switch between
        different GUI backends after that.

Matplotlib was initially written by John D. Hunter (1968-2012) and is now
developed and maintained by a host of others.

Occasionally the internal documentation (python docstrings) will refer
to MATLAB®, a registered trademark of The MathWorks, Inc.

"""

import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy
from packaging.version import parse as parse_version

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler


_log = logging.getLogger(__name__)
```
### 90 - lib/matplotlib/pyplot.py:

Start line: 295, End line: 329

```python
def switch_backend(newbackend):
    # ... other code
    if new_figure_manager is None:
        def new_figure_manager_given_figure(num, figure):
            return canvas_class.new_manager(figure, num)

        def new_figure_manager(num, *args, FigureClass=Figure, **kwargs):
            fig = FigureClass(*args, **kwargs)
            return new_figure_manager_given_figure(num, fig)

        def draw_if_interactive():
            if matplotlib.is_interactive():
                manager = _pylab_helpers.Gcf.get_active()
                if manager:
                    manager.canvas.draw_idle()

        backend_mod.new_figure_manager_given_figure = \
            new_figure_manager_given_figure
        backend_mod.new_figure_manager = new_figure_manager
        backend_mod.draw_if_interactive = draw_if_interactive

    _log.debug("Loaded backend %s version %s.",
               newbackend, backend_mod.backend_version)

    rcParams['backend'] = rcParamsDefault['backend'] = newbackend
    _backend_mod = backend_mod
    for func_name in ["new_figure_manager", "draw_if_interactive", "show"]:
        globals()[func_name].__signature__ = inspect.signature(
            getattr(backend_mod, func_name))

    # Need to keep a global reference to the backend for compatibility reasons.
    # See https://github.com/matplotlib/matplotlib/issues/6092
    matplotlib.backends.backend = newbackend

    # make sure the repl display hook is installed in case we become
    # interactive
    install_repl_displayhook()
```
### 108 - lib/matplotlib/pyplot.py:

Start line: 919, End line: 936

```python
def clf():
    """Clear the current figure."""
    gcf().clear()


def draw():
    """
    Redraw the current figure.

    This is used to update a figure that has been altered, but not
    automatically re-drawn.  If interactive mode is on (via `.ion()`), this
    should be only rarely needed, but there may be ways to modify the state of
    a figure without marking it as "stale".  Please report these cases as bugs.

    This is equivalent to calling ``fig.canvas.draw_idle()``, where ``fig`` is
    the current figure.
    """
    gcf().canvas.draw_idle()
```
