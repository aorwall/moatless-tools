# matplotlib__matplotlib-25785

| **matplotlib/matplotlib** | `950d0db55ac04e663d523144882af0ec2d172420` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 5436 |
| **Avg pos** | 61.5 |
| **Min pos** | 12 |
| **Max pos** | 111 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/backends/backend_ps.py b/lib/matplotlib/backends/backend_ps.py
--- a/lib/matplotlib/backends/backend_ps.py
+++ b/lib/matplotlib/backends/backend_ps.py
@@ -867,18 +867,24 @@ def _print_figure(
         # find the appropriate papertype
         width, height = self.figure.get_size_inches()
         if papertype == 'auto':
-            papertype = _get_papertype(
-                *orientation.swap_if_landscape((width, height)))
-        paper_width, paper_height = orientation.swap_if_landscape(
-            papersize[papertype])
+            _api.warn_deprecated("3.8", name="papertype='auto'",
+                                 addendum="Pass an explicit paper type, or omit the "
+                                 "*papertype* argument entirely.")
+            papertype = _get_papertype(*orientation.swap_if_landscape((width, height)))
 
-        if mpl.rcParams['ps.usedistiller']:
-            # distillers improperly clip eps files if pagesize is too small
-            if width > paper_width or height > paper_height:
-                papertype = _get_papertype(
-                    *orientation.swap_if_landscape((width, height)))
-                paper_width, paper_height = orientation.swap_if_landscape(
-                    papersize[papertype])
+        if is_eps:
+            paper_width, paper_height = width, height
+        else:
+            paper_width, paper_height = orientation.swap_if_landscape(
+                papersize[papertype])
+
+            if mpl.rcParams['ps.usedistiller']:
+                # distillers improperly clip eps files if pagesize is too small
+                if width > paper_width or height > paper_height:
+                    papertype = _get_papertype(
+                        *orientation.swap_if_landscape((width, height)))
+                    paper_width, paper_height = orientation.swap_if_landscape(
+                        papersize[papertype])
 
         # center the figure on the paper
         xo = 72 * 0.5 * (paper_width - width)
@@ -1055,6 +1061,9 @@ def _print_figure_tex(
                     self.figure.get_size_inches())
             else:
                 if papertype == 'auto':
+                    _api.warn_deprecated("3.8", name="papertype='auto'",
+                                         addendum="Pass an explicit paper type, or "
+                                         "omit the *papertype* argument entirely.")
                     papertype = _get_papertype(width, height)
                 paper_width, paper_height = papersize[papertype]
 
diff --git a/lib/matplotlib/rcsetup.py b/lib/matplotlib/rcsetup.py
--- a/lib/matplotlib/rcsetup.py
+++ b/lib/matplotlib/rcsetup.py
@@ -438,6 +438,19 @@ def validate_ps_distiller(s):
         return ValidateInStrings('ps.usedistiller', ['ghostscript', 'xpdf'])(s)
 
 
+def _validate_papersize(s):
+    # Re-inline this validator when the 'auto' deprecation expires.
+    s = ValidateInStrings("ps.papersize",
+                          ["auto", "letter", "legal", "ledger",
+                           *[f"{ab}{i}" for ab in "ab" for i in range(11)]],
+                          ignorecase=True)(s)
+    if s == "auto":
+        _api.warn_deprecated("3.8", name="ps.papersize='auto'",
+                             addendum="Pass an explicit paper type, or omit the "
+                             "*ps.papersize* rcParam entirely.")
+    return s
+
+
 # A validator dedicated to the named line styles, based on the items in
 # ls_mapper, and a list of possible strings read from Line2D.set_linestyle
 _validate_named_linestyle = ValidateInStrings(
@@ -1180,9 +1193,7 @@ def _convert_validator_spec(key, conv):
     "tk.window_focus": validate_bool,  # Maintain shell focus for TkAgg
 
     # Set the papersize/type
-    "ps.papersize":       _ignorecase(["auto", "letter", "legal", "ledger",
-                                      *[f"{ab}{i}"
-                                        for ab in "ab" for i in range(11)]]),
+    "ps.papersize":       _validate_papersize,
     "ps.useafm":          validate_bool,
     # use ghostscript or xpdf to distill ps output
     "ps.usedistiller":    validate_ps_distiller,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/backends/backend_ps.py | 870 | 881 | - | 1 | -
| lib/matplotlib/backends/backend_ps.py | 1058 | 1058 | 12 | 1 | 5436
| lib/matplotlib/rcsetup.py | 441 | 441 | 111 | 19 | 49357
| lib/matplotlib/rcsetup.py | 1183 | 1185 | - | 19 | -


## Problem Statement

```
automatic papersize selection by ps backend is almost certainly broken
No minimal example, but the relevant chunk (`backend_ps.py`) is
\`\`\`python
papersize = {'letter': (8.5,11),
             'legal': (8.5,14),
             'ledger': (11,17),
             'a0': (33.11,46.81),
             'a1': (23.39,33.11),
             <elided>
             'a10': (1.02,1.457),
             'b0': (40.55,57.32),
             'b1': (28.66,40.55),
             <elided>
             'b10': (1.26,1.76)}

def _get_papertype(w, h):
    keys = list(six.iterkeys(papersize))
    keys.sort()
    keys.reverse()
    for key in keys:
        if key.startswith('l'): continue
        pw, ph = papersize[key]
        if (w < pw) and (h < ph): return key
    else:
        return 'a0'
\`\`\`

Note that the sorting is by name, which means that the size is the first one among "a9, a8, ..., a2, a10, a1, b9, b8, ..., b2, b10, b1" (in that order) that is larger than the requested size -- which makes no sense.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 lib/matplotlib/backends/backend_ps.py** | 1 | 88| 763 | 763 | 12045 | 
| 2 | **1 lib/matplotlib/backends/backend_ps.py** | 806 | 854| 443 | 1206 | 12045 | 
| 3 | 2 lib/matplotlib/backends/_backend_pdf_ps.py | 109 | 146| 383 | 1589 | 13107 | 
| 4 | **2 lib/matplotlib/backends/backend_ps.py** | 1090 | 1136| 589 | 2178 | 13107 | 
| 5 | **2 lib/matplotlib/backends/backend_ps.py** | 1289 | 1331| 397 | 2575 | 13107 | 
| 6 | **2 lib/matplotlib/backends/backend_ps.py** | 294 | 333| 426 | 3001 | 13107 | 
| 7 | **2 lib/matplotlib/backends/backend_ps.py** | 566 | 606| 449 | 3450 | 13107 | 
| 8 | **2 lib/matplotlib/backends/backend_ps.py** | 913 | 970| 601 | 4051 | 13107 | 
| 9 | 2 lib/matplotlib/backends/_backend_pdf_ps.py | 82 | 107| 188 | 4239 | 13107 | 
| 10 | **2 lib/matplotlib/backends/backend_ps.py** | 354 | 386| 231 | 4470 | 13107 | 
| 11 | **2 lib/matplotlib/backends/backend_ps.py** | 335 | 352| 205 | 4675 | 13107 | 
| **-> 12 <-** | **2 lib/matplotlib/backends/backend_ps.py** | 993 | 1087| 761 | 5436 | 13107 | 
| 13 | 2 lib/matplotlib/backends/_backend_pdf_ps.py | 1 | 57| 350 | 5786 | 13107 | 
| 14 | **2 lib/matplotlib/backends/backend_ps.py** | 1180 | 1208| 356 | 6142 | 13107 | 
| 15 | **2 lib/matplotlib/backends/backend_ps.py** | 972 | 991| 281 | 6423 | 13107 | 
| 16 | **2 lib/matplotlib/backends/backend_ps.py** | 608 | 665| 516 | 6939 | 13107 | 
| 17 | **2 lib/matplotlib/backends/backend_ps.py** | 699 | 745| 448 | 7387 | 13107 | 
| 18 | **2 lib/matplotlib/backends/backend_ps.py** | 747 | 803| 440 | 7827 | 13107 | 
| 19 | **2 lib/matplotlib/backends/backend_ps.py** | 427 | 466| 366 | 8193 | 13107 | 
| 20 | 3 galleries/users_explain/text/usetex.py | 1 | 178| 1786 | 9979 | 14893 | 
| 21 | **3 lib/matplotlib/backends/backend_ps.py** | 405 | 425| 182 | 10161 | 14893 | 
| 22 | 4 lib/matplotlib/backends/backend_pgf.py | 920 | 933| 165 | 10326 | 24601 | 
| 23 | 5 lib/matplotlib/_mathtext.py | 1164 | 1226| 577 | 10903 | 48904 | 
| 24 | 5 lib/matplotlib/backends/backend_pgf.py | 726 | 788| 573 | 11476 | 48904 | 
| 25 | 6 lib/matplotlib/backends/backend_pdf.py | 615 | 658| 703 | 12179 | 73610 | 
| 26 | 6 lib/matplotlib/backends/backend_pdf.py | 1047 | 1108| 534 | 12713 | 73610 | 
| 27 | **6 lib/matplotlib/backends/backend_ps.py** | 1139 | 1177| 365 | 13078 | 73610 | 
| 28 | 6 lib/matplotlib/backends/backend_pdf.py | 2243 | 2329| 1039 | 14117 | 73610 | 
| 29 | **6 lib/matplotlib/backends/backend_ps.py** | 468 | 519| 440 | 14557 | 73610 | 
| 30 | 6 lib/matplotlib/backends/backend_pdf.py | 798 | 821| 212 | 14769 | 73610 | 
| 31 | 6 lib/matplotlib/backends/backend_pdf.py | 2765 | 2811| 296 | 15065 | 73610 | 
| 32 | 6 lib/matplotlib/backends/backend_pdf.py | 2189 | 2241| 522 | 15587 | 73610 | 
| 33 | **6 lib/matplotlib/backends/backend_ps.py** | 521 | 564| 455 | 16042 | 73610 | 
| 34 | 7 lib/matplotlib/_constrained_layout.py | 300 | 335| 391 | 16433 | 81173 | 
| 35 | 7 lib/matplotlib/backends/backend_pdf.py | 7 | 46| 261 | 16694 | 81173 | 
| 36 | 7 lib/matplotlib/backends/backend_pdf.py | 2331 | 2429| 1068 | 17762 | 81173 | 
| 37 | 7 lib/matplotlib/backends/backend_pdf.py | 1944 | 1983| 372 | 18134 | 81173 | 
| 38 | 7 lib/matplotlib/backends/backend_pdf.py | 2488 | 2557| 561 | 18695 | 81173 | 
| 39 | 7 lib/matplotlib/backends/backend_pgf.py | 701 | 723| 230 | 18925 | 81173 | 
| 40 | 7 lib/matplotlib/backends/_backend_pdf_ps.py | 60 | 79| 168 | 19093 | 81173 | 
| 41 | 8 doc/conf.py | 509 | 618| 809 | 19902 | 87531 | 
| 42 | 8 lib/matplotlib/_mathtext.py | 404 | 446| 743 | 20645 | 87531 | 
| 43 | 8 lib/matplotlib/backends/backend_pgf.py | 266 | 293| 299 | 20944 | 87531 | 
| 44 | 8 lib/matplotlib/backends/backend_pdf.py | 2579 | 2591| 138 | 21082 | 87531 | 
| 45 | **8 lib/matplotlib/backends/backend_ps.py** | 111 | 189| 656 | 21738 | 87531 | 
| 46 | 8 lib/matplotlib/backends/backend_pgf.py | 953 | 962| 111 | 21849 | 87531 | 
| 47 | **8 lib/matplotlib/backends/backend_ps.py** | 280 | 292| 132 | 21981 | 87531 | 
| 48 | 9 galleries/examples/subplots_axes_and_figures/figure_size_units.py | 1 | 83| 629 | 22610 | 88160 | 
| 49 | 9 lib/matplotlib/backends/backend_pdf.py | 291 | 297| 120 | 22730 | 88160 | 
| 50 | 9 lib/matplotlib/backends/backend_pdf.py | 1520 | 1551| 224 | 22954 | 88160 | 
| 51 | 9 lib/matplotlib/backends/backend_pdf.py | 956 | 987| 310 | 23264 | 88160 | 
| 52 | **9 lib/matplotlib/backends/backend_ps.py** | 192 | 227| 331 | 23595 | 88160 | 
| 53 | 9 lib/matplotlib/backends/backend_pdf.py | 1553 | 1589| 364 | 23959 | 88160 | 
| 54 | 10 lib/matplotlib/_tight_layout.py | 96 | 157| 738 | 24697 | 91089 | 
| 55 | **10 lib/matplotlib/backends/backend_ps.py** | 1228 | 1286| 503 | 25200 | 91089 | 
| 56 | **10 lib/matplotlib/backends/backend_ps.py** | 230 | 278| 412 | 25612 | 91089 | 
| 57 | 11 lib/matplotlib/font_manager.py | 1 | 95| 744 | 26356 | 104055 | 
| 58 | 11 lib/matplotlib/backends/backend_pdf.py | 2139 | 2177| 357 | 26713 | 104055 | 
| 59 | 11 lib/matplotlib/backends/backend_pdf.py | 1985 | 2025| 368 | 27081 | 104055 | 
| 60 | 12 lib/matplotlib/animation.py | 51 | 92| 311 | 27392 | 119564 | 
| 61 | 12 lib/matplotlib/backends/backend_pdf.py | 2179 | 2187| 125 | 27517 | 119564 | 
| 62 | 12 lib/matplotlib/backends/backend_pdf.py | 2447 | 2471| 234 | 27751 | 119564 | 
| 63 | 12 lib/matplotlib/backends/backend_pgf.py | 62 | 82| 177 | 27928 | 119564 | 
| 64 | 12 lib/matplotlib/backends/backend_pdf.py | 1261 | 1347| 807 | 28735 | 119564 | 
| 65 | 12 lib/matplotlib/backends/backend_pdf.py | 1824 | 1844| 265 | 29000 | 119564 | 
| 66 | 13 galleries/examples/style_sheets/bmh.py | 1 | 34| 211 | 29211 | 119775 | 
| 67 | 13 lib/matplotlib/backends/backend_pdf.py | 769 | 796| 267 | 29478 | 119775 | 
| 68 | 13 lib/matplotlib/backends/backend_pdf.py | 2097 | 2137| 391 | 29869 | 119775 | 
| 69 | 13 lib/matplotlib/_mathtext.py | 2600 | 2630| 263 | 30132 | 119775 | 
| 70 | 13 lib/matplotlib/backends/backend_pdf.py | 1846 | 1873| 289 | 30421 | 119775 | 
| 71 | 13 lib/matplotlib/backends/backend_pdf.py | 1614 | 1656| 447 | 30868 | 119775 | 
| 72 | 13 lib/matplotlib/backends/backend_pdf.py | 1348 | 1403| 574 | 31442 | 119775 | 
| 73 | 14 lib/matplotlib/backends/backend_cairo.py | 243 | 276| 281 | 31723 | 123907 | 
| 74 | 14 lib/matplotlib/backends/backend_cairo.py | 439 | 501| 532 | 32255 | 123907 | 
| 75 | 14 doc/conf.py | 619 | 701| 761 | 33016 | 123907 | 
| 76 | 14 lib/matplotlib/_mathtext.py | 1150 | 1162| 147 | 33163 | 123907 | 
| 77 | 15 lib/matplotlib/_type1font.py | 777 | 878| 808 | 33971 | 130726 | 
| 78 | 15 lib/matplotlib/_type1font.py | 491 | 592| 853 | 34824 | 130726 | 
| 79 | 16 lib/matplotlib/offsetbox.py | 499 | 524| 260 | 35084 | 142969 | 
| 80 | **16 lib/matplotlib/backends/backend_ps.py** | 388 | 403| 171 | 35255 | 142969 | 
| 81 | 17 galleries/examples/lines_bars_and_markers/psd_demo.py | 84 | 158| 758 | 36013 | 144678 | 
| 82 | 17 lib/matplotlib/backends/backend_pdf.py | 2027 | 2095| 716 | 36729 | 144678 | 
| 83 | 17 lib/matplotlib/backends/backend_pdf.py | 1226 | 1259| 369 | 37098 | 144678 | 
| 84 | 17 lib/matplotlib/_mathtext.py | 448 | 461| 148 | 37246 | 144678 | 
| 85 | 17 lib/matplotlib/backends/backend_pgf.py | 31 | 59| 369 | 37615 | 144678 | 
| 86 | 18 lib/matplotlib/backends/backend_wx.py | 10 | 48| 264 | 37879 | 156372 | 
| 87 | 18 galleries/examples/lines_bars_and_markers/psd_demo.py | 1 | 83| 768 | 38647 | 156372 | 
| 88 | **19 lib/matplotlib/rcsetup.py** | 357 | 371| 130 | 38777 | 168676 | 
| 89 | 19 lib/matplotlib/_mathtext.py | 330 | 346| 226 | 39003 | 168676 | 
| 90 | 19 lib/matplotlib/backends/backend_pdf.py | 989 | 1045| 572 | 39575 | 168676 | 
| 91 | 19 lib/matplotlib/backends/backend_pdf.py | 1155 | 1225| 654 | 40229 | 168676 | 
| 92 | 19 lib/matplotlib/_tight_layout.py | 20 | 95| 797 | 41026 | 168676 | 
| 93 | 19 lib/matplotlib/backends/backend_pdf.py | 1405 | 1453| 490 | 41516 | 168676 | 
| 94 | 20 galleries/examples/userdemo/pgf_fonts.py | 1 | 31| 203 | 41719 | 168879 | 
| 95 | 21 galleries/users_explain/axes/tight_layout_guide.py | 1 | 108| 798 | 42517 | 171055 | 
| 96 | 21 lib/matplotlib/_mathtext.py | 2084 | 2105| 359 | 42876 | 171055 | 
| 97 | 21 lib/matplotlib/backends/backend_pgf.py | 295 | 324| 372 | 43248 | 171055 | 
| 98 | 21 lib/matplotlib/_type1font.py | 1 | 36| 212 | 43460 | 171055 | 
| 99 | 21 lib/matplotlib/_mathtext.py | 2285 | 2378| 849 | 44309 | 171055 | 
| 100 | 22 lib/matplotlib/backends/backend_agg.py | 207 | 226| 210 | 44519 | 175708 | 
| 101 | 23 galleries/users_explain/axes/constrainedlayout_guide.py | 184 | 268| 863 | 45382 | 182418 | 
| 102 | 24 lib/matplotlib/dviread.py | 1114 | 1150| 364 | 45746 | 192986 | 
| 103 | **24 lib/matplotlib/rcsetup.py** | 221 | 241| 148 | 45894 | 192986 | 
| 104 | 24 galleries/users_explain/axes/constrainedlayout_guide.py | 270 | 351| 845 | 46739 | 192986 | 
| 105 | 25 galleries/examples/userdemo/pgf_texsystem.py | 1 | 31| 196 | 46935 | 193182 | 
| 106 | 25 lib/matplotlib/_mathtext.py | 300 | 328| 267 | 47202 | 193182 | 
| 107 | 25 lib/matplotlib/backends/backend_pgf.py | 475 | 524| 539 | 47741 | 193182 | 
| 108 | 26 galleries/examples/subplots_axes_and_figures/demo_tight_layout.py | 1 | 116| 762 | 48503 | 194056 | 
| 109 | 27 galleries/examples/axes_grid1/demo_fixed_size_axes.py | 1 | 51| 355 | 48858 | 194411 | 
| 110 | 27 lib/matplotlib/_mathtext.py | 2509 | 2539| 291 | 49149 | 194411 | 
| **-> 111 <-** | **27 lib/matplotlib/rcsetup.py** | 421 | 446| 208 | 49357 | 194411 | 
| 112 | 27 lib/matplotlib/_tight_layout.py | 266 | 302| 362 | 49719 | 194411 | 
| 113 | 27 lib/matplotlib/_mathtext.py | 523 | 589| 630 | 50349 | 194411 | 
| 114 | 27 lib/matplotlib/backends/backend_pdf.py | 300 | 370| 631 | 50980 | 194411 | 
| 115 | 27 lib/matplotlib/_type1font.py | 655 | 692| 310 | 51290 | 194411 | 
| 116 | 27 lib/matplotlib/dviread.py | 912 | 1005| 903 | 52193 | 194411 | 
| 117 | 27 lib/matplotlib/backends/backend_pdf.py | 373 | 385| 111 | 52304 | 194411 | 
| 118 | 27 lib/matplotlib/backends/backend_pdf.py | 926 | 954| 292 | 52596 | 194411 | 
| 119 | 28 lib/matplotlib/typing.py | 1 | 60| 534 | 53130 | 194945 | 
| 120 | 29 galleries/examples/subplots_axes_and_figures/auto_subplots_adjust.py | 1 | 49| 409 | 53539 | 195645 | 


### Hint

```
Currently the code looks like:
https://github.com/matplotlib/matplotlib/blob/9caa261595267001d75334a00698da500b0e4eef/lib/matplotlib/backends/backend_ps.py#L80-L85
so slightly different sorting. I guess that
`sorted(papersize.items(), key=lambda v: v[1])` will be better as it gives:
\`\`\`
{'a10': (1.02, 1.46),
 'b10': (1.26, 1.76),
 'a9': (1.46, 2.05),
 'b9': (1.76, 2.51),
 'a8': (2.05, 2.91),
 'b8': (2.51, 3.58),
 'a7': (2.91, 4.13),
 'b7': (3.58, 5.04),
 'a6': (4.13, 5.83),
 'b6': (5.04, 7.16),
 'a5': (5.83, 8.27),
 'b5': (7.16, 10.11),
 'a4': (8.27, 11.69),
 'letter': (8.5, 11),
 'legal': (8.5, 14),
 'b4': (10.11, 14.33),
 'ledger': (11, 17),
 'a3': (11.69, 16.54),
 'b3': (14.33, 20.27),
 'a2': (16.54, 23.39),
 'b2': (20.27, 28.66),
 'a1': (23.39, 33.11),
 'b1': (28.66, 40.55),
 'a0': (33.11, 46.81),
 'b0': (40.55, 57.32)}
\`\`\`
This issue has been marked "inactive" because it has been 365 days since the last comment. If this issue is still present in recent Matplotlib releases, or the feature request is still wanted, please leave a comment and this label will be removed. If there are no updates in another 30 days, this issue will be automatically closed, but you are free to re-open or create a new issue if needed. We value issue reports, and this procedure is meant to help us resurface and prioritize issues that have not been addressed yet, not make them disappear.  Thanks for your help!
Based on the discussions in #22796 this is very hard to fix in a back compatible way. (But easy to fix as such.)

There were some discussions if we actually require ps, as most people probably use eps anyway. One solution is to introduce a pending deprecation for ps and see the reactions?
My preference would be to completely deprecate and then drop papersize, and make ps output at the size of the figure, like all other backends.  We *could* (if there's really demand for it) additionally support `figsize="a4"` (and similar), auto-translating these to the corresponding inches sizes (this would not be equivalent to papersize, as the axes would default to spanning the entire papersize minus the paddings).
Talked about this on the call, the consensus was to remove the "auto" feature.
```

## Patch

```diff
diff --git a/lib/matplotlib/backends/backend_ps.py b/lib/matplotlib/backends/backend_ps.py
--- a/lib/matplotlib/backends/backend_ps.py
+++ b/lib/matplotlib/backends/backend_ps.py
@@ -867,18 +867,24 @@ def _print_figure(
         # find the appropriate papertype
         width, height = self.figure.get_size_inches()
         if papertype == 'auto':
-            papertype = _get_papertype(
-                *orientation.swap_if_landscape((width, height)))
-        paper_width, paper_height = orientation.swap_if_landscape(
-            papersize[papertype])
+            _api.warn_deprecated("3.8", name="papertype='auto'",
+                                 addendum="Pass an explicit paper type, or omit the "
+                                 "*papertype* argument entirely.")
+            papertype = _get_papertype(*orientation.swap_if_landscape((width, height)))
 
-        if mpl.rcParams['ps.usedistiller']:
-            # distillers improperly clip eps files if pagesize is too small
-            if width > paper_width or height > paper_height:
-                papertype = _get_papertype(
-                    *orientation.swap_if_landscape((width, height)))
-                paper_width, paper_height = orientation.swap_if_landscape(
-                    papersize[papertype])
+        if is_eps:
+            paper_width, paper_height = width, height
+        else:
+            paper_width, paper_height = orientation.swap_if_landscape(
+                papersize[papertype])
+
+            if mpl.rcParams['ps.usedistiller']:
+                # distillers improperly clip eps files if pagesize is too small
+                if width > paper_width or height > paper_height:
+                    papertype = _get_papertype(
+                        *orientation.swap_if_landscape((width, height)))
+                    paper_width, paper_height = orientation.swap_if_landscape(
+                        papersize[papertype])
 
         # center the figure on the paper
         xo = 72 * 0.5 * (paper_width - width)
@@ -1055,6 +1061,9 @@ def _print_figure_tex(
                     self.figure.get_size_inches())
             else:
                 if papertype == 'auto':
+                    _api.warn_deprecated("3.8", name="papertype='auto'",
+                                         addendum="Pass an explicit paper type, or "
+                                         "omit the *papertype* argument entirely.")
                     papertype = _get_papertype(width, height)
                 paper_width, paper_height = papersize[papertype]
 
diff --git a/lib/matplotlib/rcsetup.py b/lib/matplotlib/rcsetup.py
--- a/lib/matplotlib/rcsetup.py
+++ b/lib/matplotlib/rcsetup.py
@@ -438,6 +438,19 @@ def validate_ps_distiller(s):
         return ValidateInStrings('ps.usedistiller', ['ghostscript', 'xpdf'])(s)
 
 
+def _validate_papersize(s):
+    # Re-inline this validator when the 'auto' deprecation expires.
+    s = ValidateInStrings("ps.papersize",
+                          ["auto", "letter", "legal", "ledger",
+                           *[f"{ab}{i}" for ab in "ab" for i in range(11)]],
+                          ignorecase=True)(s)
+    if s == "auto":
+        _api.warn_deprecated("3.8", name="ps.papersize='auto'",
+                             addendum="Pass an explicit paper type, or omit the "
+                             "*ps.papersize* rcParam entirely.")
+    return s
+
+
 # A validator dedicated to the named line styles, based on the items in
 # ls_mapper, and a list of possible strings read from Line2D.set_linestyle
 _validate_named_linestyle = ValidateInStrings(
@@ -1180,9 +1193,7 @@ def _convert_validator_spec(key, conv):
     "tk.window_focus": validate_bool,  # Maintain shell focus for TkAgg
 
     # Set the papersize/type
-    "ps.papersize":       _ignorecase(["auto", "letter", "legal", "ledger",
-                                      *[f"{ab}{i}"
-                                        for ab in "ab" for i in range(11)]]),
+    "ps.papersize":       _validate_papersize,
     "ps.useafm":          validate_bool,
     # use ghostscript or xpdf to distill ps output
     "ps.usedistiller":    validate_ps_distiller,

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_backend_ps.py b/lib/matplotlib/tests/test_backend_ps.py
--- a/lib/matplotlib/tests/test_backend_ps.py
+++ b/lib/matplotlib/tests/test_backend_ps.py
@@ -336,3 +336,12 @@ def test_colorbar_shift(tmp_path):
     norm = mcolors.BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
     plt.scatter([0, 1], [1, 1], c=[0, 1], cmap=cmap, norm=norm)
     plt.colorbar()
+
+
+def test_auto_papersize_deprecation():
+    fig = plt.figure()
+    with pytest.warns(mpl.MatplotlibDeprecationWarning):
+        fig.savefig(io.BytesIO(), format='eps', papertype='auto')
+
+    with pytest.warns(mpl.MatplotlibDeprecationWarning):
+        mpl.rcParams['ps.papersize'] = 'auto'

```


## Code snippets

### 1 - lib/matplotlib/backends/backend_ps.py:

Start line: 1, End line: 88

```python
"""
A PostScript backend, which can produce both PostScript .ps and .eps.
"""

import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps


_log = logging.getLogger(__name__)
debugPS = False


@_api.deprecated("3.7")
class PsBackendHelper:
    def __init__(self):
        self._cached = {}


@_api.caching_module_getattr
class __getattr__:
    # module-level deprecations
    ps_backend_helper = _api.deprecated("3.7", obj_type="")(
        property(lambda self: PsBackendHelper()))
    psDefs = _api.deprecated("3.8", obj_type="")(property(lambda self: _psDefs))


papersize = {'letter': (8.5, 11),
             'legal': (8.5, 14),
             'ledger': (11, 17),
             'a0': (33.11, 46.81),
             'a1': (23.39, 33.11),
             'a2': (16.54, 23.39),
             'a3': (11.69, 16.54),
             'a4': (8.27, 11.69),
             'a5': (5.83, 8.27),
             'a6': (4.13, 5.83),
             'a7': (2.91, 4.13),
             'a8': (2.05, 2.91),
             'a9': (1.46, 2.05),
             'a10': (1.02, 1.46),
             'b0': (40.55, 57.32),
             'b1': (28.66, 40.55),
             'b2': (20.27, 28.66),
             'b3': (14.33, 20.27),
             'b4': (10.11, 14.33),
             'b5': (7.16, 10.11),
             'b6': (5.04, 7.16),
             'b7': (3.58, 5.04),
             'b8': (2.51, 3.58),
             'b9': (1.76, 2.51),
             'b10': (1.26, 1.76)}


def _get_papertype(w, h):
    for key, (pw, ph) in sorted(papersize.items(), reverse=True):
        if key.startswith('l'):
            continue
        if w < pw and h < ph:
            return key
    return 'a0'
```
### 2 - lib/matplotlib/backends/backend_ps.py:

Start line: 806, End line: 854

```python
class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi = 72
    filetypes = {'ps': 'Postscript',
                 'eps': 'Encapsulated Postscript'}

    def get_default_filetype(self):
        return 'ps'

    def _print_ps(
            self, fmt, outfile, *,
            metadata=None, papertype=None, orientation='portrait',
            bbox_inches_restore=None, **kwargs):

        dpi = self.figure.dpi
        self.figure.dpi = 72  # Override the dpi kwarg

        dsc_comments = {}
        if isinstance(outfile, (str, os.PathLike)):
            filename = pathlib.Path(outfile).name
            dsc_comments["Title"] = \
                filename.encode("ascii", "replace").decode("ascii")
        dsc_comments["Creator"] = (metadata or {}).get(
            "Creator",
            f"Matplotlib v{mpl.__version__}, https://matplotlib.org/")
        # See https://reproducible-builds.org/specs/source-date-epoch/
        source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
        dsc_comments["CreationDate"] = (
            datetime.datetime.fromtimestamp(
                int(source_date_epoch),
                datetime.timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
            if source_date_epoch
            else time.ctime())
        dsc_comments = "\n".join(
            f"%%{k}: {v}" for k, v in dsc_comments.items())

        if papertype is None:
            papertype = mpl.rcParams['ps.papersize']
        papertype = papertype.lower()
        _api.check_in_list(['auto', *papersize], papertype=papertype)

        orientation = _api.check_getitem(
            _Orientation, orientation=orientation.lower())

        printer = (self._print_figure_tex
                   if mpl.rcParams['text.usetex'] else
                   self._print_figure)
        printer(fmt, outfile, dpi=dpi, dsc_comments=dsc_comments,
                orientation=orientation, papertype=papertype,
                bbox_inches_restore=bbox_inches_restore, **kwargs)
```
### 3 - lib/matplotlib/backends/_backend_pdf_ps.py:

Start line: 109, End line: 146

```python
class RendererPDFPSBase(RendererBase):
    # The following attributes must be defined by the subclasses:
    # - _afm_font_dir

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        if ismath == "TeX":
            return super().get_text_width_height_descent(s, prop, ismath)
        elif ismath:
            parse = self._text2path.mathtext_parser.parse(s, 72, prop)
            return parse.width, parse.height, parse.depth
        elif mpl.rcParams[self._use_afm_rc_name]:
            font = self._get_font_afm(prop)
            l, b, w, h, d = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000
            w *= scale
            h *= scale
            d *= scale
            return w, h, d
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=ft2font.LOAD_NO_HINTING)
            w, h = font.get_width_height()
            d = font.get_descent()
            scale = 1 / 64
            w *= scale
            h *= scale
            d *= scale
            return w, h, d

    def _get_font_afm(self, prop):
        fname = font_manager.findfont(
            prop, fontext="afm", directory=self._afm_font_dir)
        return _cached_get_afm_from_fname(fname)

    def _get_font_ttf(self, prop):
        fnames = font_manager.fontManager._find_fonts_by_props(prop)
        font = font_manager.get_font(fnames)
        font.clear()
        font.set_size(prop.get_size_in_points(), 72)
        return font
```
### 4 - lib/matplotlib/backends/backend_ps.py:

Start line: 1090, End line: 1136

```python
def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
    with mpl.rc_context({
            "text.latex.preamble":
            mpl.rcParams["text.latex.preamble"] +
            mpl.texmanager._usepackage_if_not_loaded("color") +
            mpl.texmanager._usepackage_if_not_loaded("graphicx") +
            mpl.texmanager._usepackage_if_not_loaded("psfrag") +
            r"\geometry{papersize={%(width)sin,%(height)sin},margin=0in}"
            % {"width": paper_width, "height": paper_height}
    }):
        dvifile = TexManager().make_dvi(
            "\n"
            r"\begin{figure}""\n"
            r"  \centering\leavevmode""\n"
            r"  %(psfrags)s""\n"
            r"  \includegraphics*[angle=%(angle)s]{%(epsfile)s}""\n"
            r"\end{figure}"
            % {
                "psfrags": "\n".join(psfrags),
                "angle": 90 if orientation == 'landscape' else 0,
                "epsfile": tmppath.resolve().as_posix(),
            },
            fontsize=10)  # tex's default fontsize.

    with TemporaryDirectory() as tmpdir:
        psfile = os.path.join(tmpdir, "tmp.ps")
        cbook._check_and_log_subprocess(
            ['dvips', '-q', '-R0', '-o', psfile, dvifile], _log)
        shutil.move(psfile, tmppath)

    # check if the dvips created a ps in landscape paper.  Somehow,
    # above latex+dvips results in a ps file in a landscape mode for a
    # certain figure sizes (e.g., 8.3in, 5.8in which is a5). And the
    # bounding box of the final output got messed up. We check see if
    # the generated ps file is in landscape and return this
    # information. The return value is used in pstoeps step to recover
    # the correct bounding box. 2010-06-05 JJL
    with open(tmppath) as fh:
        psfrag_rotated = "Landscape" in fh.read(1000)
    return psfrag_rotated
```
### 5 - lib/matplotlib/backends/backend_ps.py:

Start line: 1289, End line: 1331

```python
FigureManagerPS = FigureManagerBase


# The following Python dictionary psDefs contains the entries for the
# PostScript dictionary mpldict.  This dictionary implements most of
# the matplotlib primitives and some abbreviations.
#
# References:
# https://www.adobe.com/content/dam/acom/en/devnet/actionscript/articles/PLRM.pdf
# http://preserve.mactech.com/articles/mactech/Vol.09/09.04/PostscriptTutorial
# http://www.math.ubc.ca/people/faculty/cass/graphics/text/www/
#

# The usage comments use the notation of the operator summary
# in the PostScript Language reference manual.
_psDefs = [
    # name proc  *_d*  -
    # Note that this cannot be bound to /d, because when embedding a Type3 font
    # we may want to define a "d" glyph using "/d{...} d" which would locally
    # overwrite the definition.
    "/_d { bind def } bind def",
    # x y  *m*  -
    "/m { moveto } _d",
    # x y  *l*  -
    "/l { lineto } _d",
    # x y  *r*  -
    "/r { rlineto } _d",
    # x1 y1 x2 y2 x y *c*  -
    "/c { curveto } _d",
    # *cl*  -
    "/cl { closepath } _d",
    # *ce*  -
    "/ce { closepath eofill } _d",
    # wx wy llx lly urx ury  *setcachedevice*  -
    "/sc { setcachedevice } _d",
]


@_Backend.export
class _BackendPS(_Backend):
    backend_version = 'Level II'
    FigureCanvas = FigureCanvasPS
```
### 6 - lib/matplotlib/backends/backend_ps.py:

Start line: 294, End line: 333

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def set_color(self, r, g, b, store=True):
        if (r, g, b) != self.color:
            self._pswriter.write(f"{_nums_to_str(r)} setgray\n"
                                 if r == g == b else
                                 f"{_nums_to_str(r, g, b)} setrgbcolor\n")
            if store:
                self.color = (r, g, b)

    def set_linewidth(self, linewidth, store=True):
        linewidth = float(linewidth)
        if linewidth != self.linewidth:
            self._pswriter.write(f"{_nums_to_str(linewidth)} setlinewidth\n")
            if store:
                self.linewidth = linewidth

    @staticmethod
    def _linejoin_cmd(linejoin):
        # Support for directly passing integer values is for backcompat.
        linejoin = {'miter': 0, 'round': 1, 'bevel': 2, 0: 0, 1: 1, 2: 2}[
            linejoin]
        return f"{linejoin:d} setlinejoin\n"

    def set_linejoin(self, linejoin, store=True):
        if linejoin != self.linejoin:
            self._pswriter.write(self._linejoin_cmd(linejoin))
            if store:
                self.linejoin = linejoin

    @staticmethod
    def _linecap_cmd(linecap):
        # Support for directly passing integer values is for backcompat.
        linecap = {'butt': 0, 'round': 1, 'projecting': 2, 0: 0, 1: 1, 2: 2}[
            linecap]
        return f"{linecap:d} setlinecap\n"

    def set_linecap(self, linecap, store=True):
        if linecap != self.linecap:
            self._pswriter.write(self._linecap_cmd(linecap))
            if store:
                self.linecap = linecap
```
### 7 - lib/matplotlib/backends/backend_ps.py:

Start line: 566, End line: 606

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        if self._is_transparent(gc.get_rgb()):
            return  # Special handling for fully transparent.

        if not hasattr(self, "psfrag"):
            self._logwarn_once(
                "The PS backend determines usetex status solely based on "
                "rcParams['text.usetex'] and does not support having "
                "usetex=True only for some elements; this element will thus "
                "be rendered as if usetex=False.")
            self.draw_text(gc, x, y, s, prop, angle, False, mtext)
            return

        w, h, bl = self.get_text_width_height_descent(s, prop, ismath="TeX")
        fontsize = prop.get_size_in_points()
        thetext = 'psmarker%d' % self.textcnt
        color = _nums_to_str(*gc.get_rgb()[:3], sep=',')
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(
                       mpl.rcParams['font.family'][0], r'{\rmfamily %s}')
        s = fontcmd % s
        tex = r'\color[rgb]{%s} %s' % (color, s)

        # Stick to bottom-left alignment, so subtract descent from the text-normal
        # direction since text is normally positioned by its baseline.
        rangle = np.radians(angle + 90)
        pos = _nums_to_str(x - bl * np.cos(rangle), y - bl * np.sin(rangle))
        self.psfrag.append(
            r'\psfrag{%s}[bl][bl][1][%f]{\fontsize{%f}{%f}%s}' % (
                thetext, angle, fontsize, fontsize*1.25, tex))

        self._pswriter.write(f"""\

        veto
        t})



        self.textcnt += 1
```
### 8 - lib/matplotlib/backends/backend_ps.py:

Start line: 913, End line: 970

```python
class FigureCanvasPS(FigureCanvasBase):

    def _print_figure(
            self, fmt, outfile, *,
            dpi, dsc_comments, orientation, papertype,
            bbox_inches_restore=None):
        # ... other code

        def print_figure_impl(fh):
            # write the PostScript headers
            if is_eps:
                print("%!PS-Adobe-3.0 EPSF-3.0", file=fh)
            else:
                print(f"%!PS-Adobe-3.0\n"
                      f"%%DocumentPaperSizes: {papertype}\n"
                      f"%%Pages: 1\n",
                      end="", file=fh)
            print(f"%%LanguageLevel: 3\n"
                  f"{dsc_comments}\n"
                  f"%%Orientation: {orientation.name}\n"
                  f"{get_bbox_header(bbox)[0]}\n"
                  f"%%EndComments\n",
                  end="", file=fh)

            Ndict = len(_psDefs)
            print("%%BeginProlog", file=fh)
            if not mpl.rcParams['ps.useafm']:
                Ndict += len(ps_renderer._character_tracker.used)
            print("/mpldict %d dict def" % Ndict, file=fh)
            print("mpldict begin", file=fh)
            print("\n".join(_psDefs), file=fh)
            if not mpl.rcParams['ps.useafm']:
                for font_path, chars \
                        in ps_renderer._character_tracker.used.items():
                    if not chars:
                        continue
                    fonttype = mpl.rcParams['ps.fonttype']
                    # Can't use more than 255 chars from a single Type 3 font.
                    if len(chars) > 255:
                        fonttype = 42
                    fh.flush()
                    if fonttype == 3:
                        fh.write(_font_to_ps_type3(font_path, chars))
                    else:  # Type 42 only.
                        _font_to_ps_type42(font_path, chars, fh)
            print("end", file=fh)
            print("%%EndProlog", file=fh)

            if not is_eps:
                print("%%Page: 1 1", file=fh)
            print("mpldict begin", file=fh)

            print("%s translate" % _nums_to_str(xo, yo), file=fh)
            if rotation:
                print("%d rotate" % rotation, file=fh)
            print(f"0 0 {_nums_to_str(width*72, height*72)} rectclip", file=fh)

            # write the figure
            print(self._pswriter.getvalue(), file=fh)

            # write the trailer
            print("end", file=fh)
            print("showpage", file=fh)
            if not is_eps:
                print("%%EOF", file=fh)
            fh.flush()
        # ... other code
```
### 9 - lib/matplotlib/backends/_backend_pdf_ps.py:

Start line: 82, End line: 107

```python
class RendererPDFPSBase(RendererBase):
    # The following attributes must be defined by the subclasses:
    # - _afm_font_dir
    # - _use_afm_rc_name

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def flipy(self):
        # docstring inherited
        return False  # y increases from bottom to top.

    def option_scale_image(self):
        # docstring inherited
        return True  # PDF and PS support arbitrary image scaling.

    def option_image_nocomposite(self):
        # docstring inherited
        # Decide whether to composite image based on rcParam value.
        return not mpl.rcParams["image.composite_image"]

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width * 72.0, self.height * 72.0
```
### 10 - lib/matplotlib/backends/backend_ps.py:

Start line: 354, End line: 386

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def create_hatch(self, hatch):
        sidelen = 72
        if hatch in self._hatches:
            return self._hatches[hatch]
        name = 'H%d' % len(self._hatches)
        linewidth = mpl.rcParams['hatch.linewidth']
        pageheight = self.height * 72
        self._pswriter.write(f"""\
        tternType 1
        intType 2
        lingType 2
        ox[0 0 {sidelen:d} {sidelen:d}]
        tep {sidelen:d}
        tep {sidelen:d}

        intProc {{
        pop
        {linewidth:g} setlinewidth
        onvert_path(
        .hatch(hatch), Affine2D().scale(sidelen), simplify=False)}
        gsave
        fill
        grestore
        stroke
        bind

        x
        geheight:g} translate
        attern
        e} exch def

        self._hatches[hatch] = name
        return name
```
### 11 - lib/matplotlib/backends/backend_ps.py:

Start line: 335, End line: 352

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def set_linedash(self, offset, seq, store=True):
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if np.array_equal(seq, oldseq) and oldo == offset:
                return

        self._pswriter.write(f"[{_nums_to_str(*seq)}] {_nums_to_str(offset)} setdash\n"
                             if seq is not None and len(seq) else
                             "[] 0 setdash\n")
        if store:
            self.linedash = (offset, seq)

    def set_font(self, fontname, fontsize, store=True):
        if (fontname, fontsize) != (self.fontname, self.fontsize):
            self._pswriter.write(f"/{fontname} {fontsize:1.3f} selectfont\n")
            if store:
                self.fontname = fontname
                self.fontsize = fontsize
```
### 12 - lib/matplotlib/backends/backend_ps.py:

Start line: 993, End line: 1087

```python
class FigureCanvasPS(FigureCanvasBase):

    def _print_figure_tex(
            self, fmt, outfile, *,
            dpi, dsc_comments, orientation, papertype,
            bbox_inches_restore=None):
        """
        If :rc:`text.usetex` is True, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.

        The rest of the behavior is as for `._print_figure`.
        """
        is_eps = fmt == 'eps'

        width, height = self.figure.get_size_inches()
        xo = 0
        yo = 0

        llx = xo
        lly = yo
        urx = llx + self.figure.bbox.width
        ury = lly + self.figure.bbox.height
        bbox = (llx, lly, urx, ury)

        self._pswriter = StringIO()

        # mixed mode rendering
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure,
                                     width, height, dpi, ps_renderer,
                                     bbox_inches_restore=bbox_inches_restore)

        self.figure.draw(renderer)

        # write to a temp file, we'll move it to outfile when done
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir, "tmp.ps")
            tmppath.write_text(
                f"""\
            .0 EPSF-3.0
            vel: 3
            s}
            ader(bbox)[0]}
            s
            g
            n(_psDefs)} dict def
            n
            Defs)}


            n
            r(xo, yo)} translate
            o_str(width*72, height*72)} rectclip
            ter.getvalue()}



                encoding="latin-1")

            if orientation is _Orientation.landscape:  # now, ready to rotate
                width, height = height, width
                bbox = (lly, llx, ury, urx)

            # set the paper size to the figure size if is_eps. The
            # resulting ps file has the given size with correct bounding
            # box so that there is no need to call 'pstoeps'
            if is_eps:
                paper_width, paper_height = orientation.swap_if_landscape(
                    self.figure.get_size_inches())
            else:
                if papertype == 'auto':
                    papertype = _get_papertype(width, height)
                paper_width, paper_height = papersize[papertype]

            psfrag_rotated = _convert_psfrags(
                tmppath, ps_renderer.psfrag, paper_width, paper_height,
                orientation.name)

            if (mpl.rcParams['ps.usedistiller'] == 'ghostscript'
                    or mpl.rcParams['text.usetex']):
                _try_distill(gs_distill,
                             tmppath, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)
            elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                _try_distill(xpdf_distill,
                             tmppath, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)

            _move_path_to_path_or_stream(tmppath, outfile)

    print_ps = functools.partialmethod(_print_ps, "ps")
    print_eps = functools.partialmethod(_print_ps, "eps")

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()
```
### 14 - lib/matplotlib/backends/backend_ps.py:

Start line: 1180, End line: 1208

```python
def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
    mpl._get_executable_info("gs")  # Effectively checks for ps2pdf.
    mpl._get_executable_info("pdftops")

    with TemporaryDirectory() as tmpdir:
        tmppdf = pathlib.Path(tmpdir, "tmp.pdf")
        tmpps = pathlib.Path(tmpdir, "tmp.ps")
        # Pass options as `-foo#bar` instead of `-foo=bar` to keep Windows
        # happy (https://ghostscript.com/doc/9.56.1/Use.htm#MS_Windows).
        cbook._check_and_log_subprocess(
            ["ps2pdf",
             "-dAutoFilterColorImages#false",
             "-dAutoFilterGrayImages#false",
             "-sAutoRotatePages#None",
             "-sGrayImageFilter#FlateEncode",
             "-sColorImageFilter#FlateEncode",
             "-dEPSCrop" if eps else "-sPAPERSIZE#%s" % ptype,
             tmpfile, tmppdf], _log)
        cbook._check_and_log_subprocess(
            ["pdftops", "-paper", "match", "-level3", tmppdf, tmpps], _log)
        shutil.move(tmpps, tmpfile)
    if eps:
        pstoeps(tmpfile)
```
### 15 - lib/matplotlib/backends/backend_ps.py:

Start line: 972, End line: 991

```python
class FigureCanvasPS(FigureCanvasBase):

    def _print_figure(
            self, fmt, outfile, *,
            dpi, dsc_comments, orientation, papertype,
            bbox_inches_restore=None):
        # ... other code

        if mpl.rcParams['ps.usedistiller']:
            # We are going to use an external program to process the output.
            # Write to a temporary file.
            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, "tmp.ps")
                with open(tmpfile, 'w', encoding='latin-1') as fh:
                    print_figure_impl(fh)
                if mpl.rcParams['ps.usedistiller'] == 'ghostscript':
                    _try_distill(gs_distill,
                                 tmpfile, is_eps, ptype=papertype, bbox=bbox)
                elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                    _try_distill(xpdf_distill,
                                 tmpfile, is_eps, ptype=papertype, bbox=bbox)
                _move_path_to_path_or_stream(tmpfile, outfile)

        else:  # Write directly to outfile.
            with cbook.open_file_cm(outfile, "w", encoding="latin-1") as file:
                if not file_requires_unicode(file):
                    file = codecs.getwriter("latin-1")(file)
                print_figure_impl(file)
```
### 16 - lib/matplotlib/backends/backend_ps.py:

Start line: 608, End line: 665

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        if self._is_transparent(gc.get_rgb()):
            return  # Special handling for fully transparent.

        if ismath == 'TeX':
            return self.draw_tex(gc, x, y, s, prop, angle)

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        stream = []  # list of (ps_name, x, char_name)

        if mpl.rcParams['ps.useafm']:
            font = self._get_font_afm(prop)
            ps_name = (font.postscript_name.encode("ascii", "replace")
                        .decode("ascii"))
            scale = 0.001 * prop.get_size_in_points()
            thisx = 0
            last_name = None  # kerns returns 0 for None.
            for c in s:
                name = uni2type1.get(ord(c), f"uni{ord(c):04X}")
                try:
                    width = font.get_width_from_char_name(name)
                except KeyError:
                    name = 'question'
                    width = font.get_width_char('?')
                kern = font.get_kern_dist_from_name(last_name, name)
                last_name = name
                thisx += kern * scale
                stream.append((ps_name, thisx, name))
                thisx += width * scale

        else:
            font = self._get_font_ttf(prop)
            self._character_tracker.track(font, s)
            for item in _text_helpers.layout(s, font):
                ps_name = (item.ft_object.postscript_name
                           .encode("ascii", "replace").decode("ascii"))
                glyph_name = item.ft_object.get_glyph_name(item.glyph_idx)
                stream.append((ps_name, item.x, glyph_name))
        self.set_color(*gc.get_rgb())

        for ps_name, group in itertools. \
                groupby(stream, lambda entry: entry[0]):
            self.set_font(ps_name, prop.get_size_in_points(), False)
            thetext = "\n".join(f"{x:g} 0 m /{name:s} glyphshow"
                                for _, x, name in group)
            self._pswriter.write(f"""\

            lip_cmd(gc)}
            translate
            tate
```
### 17 - lib/matplotlib/backends/backend_ps.py:

Start line: 699, End line: 745

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4

        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        flat_points = trans.transform(flat_points)
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))

        xmin, ymin = points_min
        xmax, ymax = points_max

        data = np.empty(
            shape[0] * shape[1],
            dtype=[('flags', 'u1'), ('points', '2>u4'), ('colors', '3u1')])
        data['flags'] = 0
        data['points'] = (flat_points - points_min) * factor
        data['colors'] = flat_colors[:, :3] * 255.0
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        self._pswriter.write(f"""\

        ingType 4
        rSpace [/DeviceRGB]
        PerCoordinate 32
        PerComponent 8
        PerFlag 8
        Alias true
        de [ {xmin:g} {xmax:g} {ymin:g} {ymax:g} 0 1 0 1 0 1 ]
        Source <
        }
```
### 18 - lib/matplotlib/backends/backend_ps.py:

Start line: 747, End line: 803

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def _draw_ps(self, ps, gc, rgbFace, *, fill=True, stroke=True):
        """
        Emit the PostScript snippet *ps* with all the attributes from *gc*
        applied.  *ps* must consist of PostScript commands to construct a path.

        The *fill* and/or *stroke* kwargs can be set to False if the *ps*
        string already includes filling and/or stroking, in which case
        `_draw_ps` is just supplying properties and clipping.
        """
        write = self._pswriter.write
        mightstroke = (gc.get_linewidth() > 0
                       and not self._is_transparent(gc.get_rgb()))
        if not mightstroke:
            stroke = False
        if self._is_transparent(rgbFace):
            fill = False
        hatch = gc.get_hatch()

        if mightstroke:
            self.set_linewidth(gc.get_linewidth())
            self.set_linejoin(gc.get_joinstyle())
            self.set_linecap(gc.get_capstyle())
            self.set_linedash(*gc.get_dashes())
        if mightstroke or hatch:
            self.set_color(*gc.get_rgb()[:3])
        write('gsave\n')

        write(self._get_clip_cmd(gc))

        write(ps.strip())
        write("\n")

        if fill:
            if stroke or hatch:
                write("gsave\n")
            self.set_color(*rgbFace[:3], store=False)
            write("fill\n")
            if stroke or hatch:
                write("grestore\n")

        if hatch:
            hatch_name = self.create_hatch(hatch)
            write("gsave\n")
            write(_nums_to_str(*gc.get_hatch_color()[:3]))
            write(f" {hatch_name} setpattern fill grestore\n")

        if stroke:
            write("stroke\n")

        write("grestore\n")


class _Orientation(Enum):
    portrait, landscape = range(2)

    def swap_if_landscape(self, shape):
        return shape[::-1] if self.name == "landscape" else shape
```
### 19 - lib/matplotlib/backends/backend_ps.py:

Start line: 427, End line: 466

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]
        imagecmd = "false 3 colorimage"
        data = im[::-1, :, :3]  # Vertically flipped rgb values.
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        if transform is None:
            matrix = "1 0 0 1 0 0"
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = " ".join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0

        self._pswriter.write(f"""\

        et_clip_cmd(gc)}
        :g} translate
        }] concat
        g} {yscale:g} scale
        ing {w:d} string def
        :d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]

        ile DataString readhexstring pop
        {imagecmd}
        }



    @_log_if_debug_on
    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)
```
### 21 - lib/matplotlib/backends/backend_ps.py:

Start line: 405, End line: 425

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def _get_clip_cmd(self, gc):
        clip = []
        rect = gc.get_clip_rectangle()
        if rect is not None:
            clip.append(f"{_nums_to_str(*rect.p0, *rect.size)} rectclip\n")
        path, trf = gc.get_clip_path()
        if path is not None:
            key = (path, id(trf))
            custom_clip_cmd = self._clip_paths.get(key)
            if custom_clip_cmd is None:
                custom_clip_cmd = "c%d" % len(self._clip_paths)
                self._pswriter.write(f"""\
                d} {{
                ath(path, trf, simplify=False)}




                self._clip_paths[key] = custom_clip_cmd
            clip.append(f"{custom_clip_cmd}\n")
        return "".join(clip)
```
### 27 - lib/matplotlib/backends/backend_ps.py:

Start line: 1139, End line: 1177

```python
def _try_distill(func, tmppath, *args, **kwargs):
    try:
        func(str(tmppath), *args, **kwargs)
    except mpl.ExecutableNotFoundError as exc:
        _log.warning("%s.  Distillation step skipped.", exc)


def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """

    if eps:
        paper_option = "-dEPSCrop"
    else:
        paper_option = "-sPAPERSIZE=%s" % ptype

    psfile = tmpfile + '.ps'
    dpi = mpl.rcParams['ps.distiller.res']

    cbook._check_and_log_subprocess(
        [mpl._get_executable_info("gs").executable,
         "-dBATCH", "-dNOPAUSE", "-r%d" % dpi, "-sDEVICE=ps2write",
         paper_option, "-sOutputFile=%s" % psfile, tmpfile],
        _log)

    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)

    # While it is best if above steps preserve the original bounding
    # box, there seem to be cases when it is not. For those cases,
    # the original bbox can be restored during the pstoeps step.

    if eps:
        # For some versions of gs, above steps result in a ps file where the
        # original bbox is no more correct. Do not adjust bbox for now.
        pstoeps(tmpfile, bbox, rotated=rotated)
```
### 29 - lib/matplotlib/backends/backend_ps.py:

Start line: 468, End line: 519

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_markers(
            self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # docstring inherited

        ps_color = (
            None
            if self._is_transparent(rgbFace)
            else f'{_nums_to_str(rgbFace[0])} setgray'
            if rgbFace[0] == rgbFace[1] == rgbFace[2]
            else f'{_nums_to_str(*rgbFace[:3])} setrgbcolor')

        # construct the generic marker command:

        # don't want the translate to be global
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate']

        lw = gc.get_linewidth()
        alpha = (gc.get_alpha()
                 if gc.get_forced_alpha() or len(gc.get_rgb()) == 3
                 else gc.get_rgb()[3])
        stroke = lw > 0 and alpha > 0
        if stroke:
            ps_cmd.append('%.1f setlinewidth' % lw)
            ps_cmd.append(self._linejoin_cmd(gc.get_joinstyle()))
            ps_cmd.append(self._linecap_cmd(gc.get_capstyle()))

        ps_cmd.append(self._convert_path(marker_path, marker_trans,
                                         simplify=False))

        if rgbFace:
            if stroke:
                ps_cmd.append('gsave')
            if ps_color:
                ps_cmd.extend([ps_color, 'fill'])
            if stroke:
                ps_cmd.append('grestore')

        if stroke:
            ps_cmd.append('stroke')
        ps_cmd.extend(['grestore', '} bind def'])

        for vertices, code in path.iter_segments(
                trans,
                clip=(0, 0, self.width*72, self.height*72),
                simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                ps_cmd.append(f"{x:g} {y:g} o")

        ps = '\n'.join(ps_cmd)
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)
```
### 33 - lib/matplotlib/backends/backend_ps.py:

Start line: 521, End line: 564

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    @_log_if_debug_on
    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        # Is the optimization worth it? Rough calculation:
        # cost of emitting a path in-line is
        #     (len_path + 2) * uses_per_path
        # cost of definition+use is
        #     (len_path + 3) + 3 * uses_per_path
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = \
            len_path + 3 * uses_per_path + 3 < (len_path + 2) * uses_per_path
        if not should_do_optimization:
            return RendererBase.draw_path_collection(
                self, gc, master_transform, paths, all_transforms,
                offsets, offset_trans, facecolors, edgecolors,
                linewidths, linestyles, antialiaseds, urls,
                offset_position)

        path_codes = []
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
                master_transform, paths, all_transforms)):
            name = 'p%d_%d' % (self._path_collection_id, i)
            path_bytes = self._convert_path(path, transform, simplify=False)
            self._pswriter.write(f"""\






            path_codes.append(name)

        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, path_codes, offsets, offset_trans,
                facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            ps = f"{xo:g} {yo:g} {path_id}"
            self._draw_ps(ps, gc0, rgbFace)

        self._path_collection_id += 1
```
### 45 - lib/matplotlib/backends/backend_ps.py:

Start line: 111, End line: 189

```python
def _font_to_ps_type3(font_path, chars):
    """
    Subset *chars* from the font at *font_path* into a Type 3 font.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.

    Returns
    -------
    str
        The string representation of a Type 3 font, which can be included
        verbatim into a PostScript file.
    """
    font = get_font(font_path, hinting_factor=1)
    glyph_ids = [font.get_char_index(c) for c in chars]

    preamble = """\
%!PS-Adobe-3.0 Resource-Font
%%Creator: Converted from TrueType to Type 3 by Matplotlib.
10 dict begin
/FontName /{font_name} def
/PaintType 0 def
/FontMatrix [{inv_units_per_em} 0 0 {inv_units_per_em} 0 0] def
/FontBBox [{bbox}] def
/FontType 3 def
/Encoding [{encoding}] def
/CharStrings {num_glyphs} dict dup begin
/.notdef 0 def
""".format(font_name=font.postscript_name,
           inv_units_per_em=1 / font.units_per_EM,
           bbox=" ".join(map(str, font.bbox)),
           encoding=" ".join(f"/{font.get_glyph_name(glyph_id)}"
                             for glyph_id in glyph_ids),
           num_glyphs=len(glyph_ids) + 1)
    postamble = """
end readonly def

/BuildGlyph {
 exch begin
 CharStrings exch
 2 copy known not {pop /.notdef} if
 true 3 1 roll get exec
 end
} _d

/BuildChar {
 1 index /Encoding get exch get
 1 index /BuildGlyph get exec
} _d

FontName currentdict end definefont pop
"""

    entries = []
    for glyph_id in glyph_ids:
        g = font.load_glyph(glyph_id, LOAD_NO_SCALE)
        v, c = font.get_path()
        entries.append(
            "/%(name)s{%(bbox)s sc\n" % {
                "name": font.get_glyph_name(glyph_id),
                "bbox": " ".join(map(str, [g.horiAdvance, 0, *g.bbox])),
            }
            + _path.convert_to_string(
                # Convert back to TrueType's internal units (1/64's).
                # (Other dimensions are already in these units.)
                Path(v * 64, c), None, None, False, None, 0,
                # No code for quad Beziers triggers auto-conversion to cubics.
                # Drop intermediate closepolys (relying on the outline
                # decomposer always explicitly moving to the closing point
                # first).
                [b"m", b"l", b"", b"c", b""], True).decode("ascii")
            + "ce} _d"
        )

    return preamble + "\n".join(entries) + postamble
```
### 47 - lib/matplotlib/backends/backend_ps.py:

Start line: 280, End line: 292

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def _is_transparent(self, rgb_or_rgba):
        if rgb_or_rgba is None:
            return True  # Consistent with rgbFace semantics.
        elif len(rgb_or_rgba) == 4:
            if rgb_or_rgba[3] == 0:
                return True
            if rgb_or_rgba[3] != 1:
                self._logwarn_once(
                    "The PostScript backend does not support transparency; "
                    "partially transparent artists will be rendered opaque.")
            return False
        else:  # len() == 3.
            return False
```
### 52 - lib/matplotlib/backends/backend_ps.py:

Start line: 192, End line: 227

```python
def _font_to_ps_type42(font_path, chars, fh):
    """
    Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.
    fh : file-like
        Where to write the font.
    """
    subset_str = ''.join(chr(c) for c in chars)
    _log.debug("SUBSET %s characters: %s", font_path, subset_str)
    try:
        fontdata = _backend_pdf_ps.get_glyphs_subset(font_path, subset_str)
        _log.debug("SUBSET %s %d -> %d", font_path, os.stat(font_path).st_size,
                   fontdata.getbuffer().nbytes)

        # Give ttconv a subsetted font along with updated glyph_ids.
        font = FT2Font(fontdata)
        glyph_ids = [font.get_char_index(c) for c in chars]
        with TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmp.ttf")

            with open(tmpfile, 'wb') as tmp:
                tmp.write(fontdata.getvalue())

            # TODO: allow convert_ttf_to_ps to input file objects (BytesIO)
            convert_ttf_to_ps(os.fsencode(tmpfile), fh, 42, glyph_ids)
    except RuntimeError:
        _log.warning(
            "The PostScript backend does not currently "
            "support the selected font.")
        raise
```
### 55 - lib/matplotlib/backends/backend_ps.py:

Start line: 1228, End line: 1286

```python
def pstoeps(tmpfile, bbox=None, rotated=False):
    """
    Convert the postscript to encapsulated postscript.  The bbox of
    the eps file will be replaced with the given *bbox* argument. If
    None, original bbox will be used.
    """

    # if rotated==True, the output eps file need to be rotated
    if bbox:
        bbox_info, rotate = get_bbox_header(bbox, rotated=rotated)
    else:
        bbox_info, rotate = None, None

    epsfile = tmpfile + '.eps'
    with open(epsfile, 'wb') as epsh, open(tmpfile, 'rb') as tmph:
        write = epsh.write
        # Modify the header:
        for line in tmph:
            if line.startswith(b'%!PS'):
                write(b"%!PS-Adobe-3.0 EPSF-3.0\n")
                if bbox:
                    write(bbox_info.encode('ascii') + b'\n')
            elif line.startswith(b'%%EndComments'):
                write(line)
                write(b'%%BeginProlog\n'
                      b'save\n'
                      b'countdictstack\n'
                      b'mark\n'
                      b'newpath\n'
                      b'/showpage {} def\n'
                      b'/setpagedevice {pop} def\n'
                      b'%%EndProlog\n'
                      b'%%Page 1 1\n')
                if rotate:
                    write(rotate.encode('ascii') + b'\n')
                break
            elif bbox and line.startswith((b'%%Bound', b'%%HiResBound',
                                           b'%%DocumentMedia', b'%%Pages')):
                pass
            else:
                write(line)
        # Now rewrite the rest of the file, and modify the trailer.
        # This is done in a second loop such that the header of the embedded
        # eps file is not modified.
        for line in tmph:
            if line.startswith(b'%%EOF'):
                write(b'cleartomark\n'
                      b'countdictstack\n'
                      b'exch sub { end } repeat\n'
                      b'restore\n'
                      b'showpage\n'
                      b'%%EOF\n')
            elif line.startswith(b'%%PageBoundingBox'):
                pass
            else:
                write(line)

    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)
```
### 56 - lib/matplotlib/backends/backend_ps.py:

Start line: 230, End line: 278

```python
def _log_if_debug_on(meth):
    """
    Wrap `RendererPS` method *meth* to emit a PS comment with the method name,
    if the global flag `debugPS` is set.
    """
    @functools.wraps(meth)
    def wrapper(self, *args, **kwargs):
        if debugPS:
            self._pswriter.write(f"% {meth.__name__}\n")
        return meth(self, *args, **kwargs)

    return wrapper


class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    _afm_font_dir = cbook._get_data_path("fonts/afm")
    _use_afm_rc_name = "ps.useafm"

    def __init__(self, width, height, pswriter, imagedpi=72):
        # Although postscript itself is dpi independent, we need to inform the
        # image code about a requested dpi to generate high resolution images
        # and them scale them before embedding them.
        super().__init__(width, height)
        self._pswriter = pswriter
        if mpl.rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi

        # current renderer state (None=uninitialised)
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self._hatches = {}
        self.image_magnification = imagedpi / 72
        self._clip_paths = {}
        self._path_collection_id = 0

        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        self._logwarn_once = functools.cache(_log.warning)
```
### 80 - lib/matplotlib/backends/backend_ps.py:

Start line: 388, End line: 403

```python
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return self.image_magnification

    def _convert_path(self, path, transform, clip=False, simplify=None):
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(
            path, transform, clip, simplify, None,
            6, [b"m", b"l", b"", b"c", b"cl"], True).decode("ascii")
```
### 88 - lib/matplotlib/rcsetup.py:

Start line: 357, End line: 371

```python
def validate_fontsize(s):
    fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
                 'x-large', 'xx-large', 'smaller', 'larger']
    if isinstance(s, str):
        s = s.lower()
    if s in fontsizes:
        return s
    try:
        return float(s)
    except ValueError as e:
        raise ValueError("%s is not a valid font size. Valid font sizes "
                         "are %s." % (s, ", ".join(fontsizes))) from e


validate_fontsizelist = _listify_validator(validate_fontsize)
```
### 103 - lib/matplotlib/rcsetup.py:

Start line: 221, End line: 241

```python
def validate_fonttype(s):
    """
    Confirm that this is a Postscript or PDF font type that we know how to
    convert to.
    """
    fonttypes = {'type3':    3,
                 'truetype': 42}
    try:
        fonttype = validate_int(s)
    except ValueError:
        try:
            return fonttypes[s.lower()]
        except KeyError as e:
            raise ValueError('Supported Postscript/PDF font types are %s'
                             % list(fonttypes)) from e
    else:
        if fonttype not in fonttypes.values():
            raise ValueError(
                'Supported Postscript/PDF font types are %s' %
                list(fonttypes.values()))
        return fonttype
```
### 111 - lib/matplotlib/rcsetup.py:

Start line: 421, End line: 446

```python
def validate_whiskers(s):
    try:
        return _listify_validator(validate_float, n=2)(s)
    except (TypeError, ValueError):
        try:
            return float(s)
        except ValueError as e:
            raise ValueError("Not a valid whisker value [float, "
                             "(float, float)]") from e


def validate_ps_distiller(s):
    if isinstance(s, str):
        s = s.lower()
    if s in ('none', None, 'false', False):
        return None
    else:
        return ValidateInStrings('ps.usedistiller', ['ghostscript', 'xpdf'])(s)


# A validator dedicated to the named line styles, based on the items in
# ls_mapper, and a list of possible strings read from Line2D.set_linestyle
_validate_named_linestyle = ValidateInStrings(
    'linestyle',
    [*ls_mapper.keys(), *ls_mapper.values(), 'None', 'none', ' ', ''],
    ignorecase=True)
```
