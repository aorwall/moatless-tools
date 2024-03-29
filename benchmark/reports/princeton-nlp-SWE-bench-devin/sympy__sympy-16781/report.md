# sympy__sympy-16781

| **sympy/sympy** | `8dcb72f6abe5c7edf94ea722429c0bb9f7eef54d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 132 |
| **Avg pos** | 101.0 |
| **Min pos** | 1 |
| **Max pos** | 96 |
| **Top file pos** | 1 |
| **Missing snippets** | 8 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/dot.py b/sympy/printing/dot.py
--- a/sympy/printing/dot.py
+++ b/sympy/printing/dot.py
@@ -7,6 +7,7 @@
 from sympy.core.compatibility import default_sort_key
 from sympy.core.add import Add
 from sympy.core.mul import Mul
+from sympy.printing.repr import srepr
 
 __all__ = ['dotprint']
 
@@ -14,20 +15,24 @@
           (Expr,  {'color': 'black'}))
 
 
-sort_classes = (Add, Mul)
 slotClasses = (Symbol, Integer, Rational, Float)
-# XXX: Why not just use srepr()?
-def purestr(x):
+def purestr(x, with_args=False):
     """ A string that follows obj = type(obj)(*obj.args) exactly """
+    sargs = ()
     if not isinstance(x, Basic):
-        return str(x)
-    if type(x) in slotClasses:
-        args = [getattr(x, slot) for slot in x.__slots__]
-    elif type(x) in sort_classes:
-        args = sorted(x.args, key=default_sort_key)
+        rv = str(x)
+    elif not x.args:
+        rv = srepr(x)
     else:
         args = x.args
-    return "%s(%s)"%(type(x).__name__, ', '.join(map(purestr, args)))
+        if isinstance(x, Add) or \
+                isinstance(x, Mul) and x.is_commutative:
+            args = sorted(args, key=default_sort_key)
+        sargs = tuple(map(purestr, args))
+        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
+    if with_args:
+        rv = rv, sargs
+    return rv
 
 
 def styleof(expr, styles=default_styles):
@@ -54,6 +59,7 @@ def styleof(expr, styles=default_styles):
             style.update(sty)
     return style
 
+
 def attrprint(d, delimiter=', '):
     """ Print a dictionary of attributes
 
@@ -66,6 +72,7 @@ def attrprint(d, delimiter=', '):
     """
     return delimiter.join('"%s"="%s"'%item for item in sorted(d.items()))
 
+
 def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
     """ String defining a node
 
@@ -75,7 +82,7 @@ def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
     >>> from sympy.printing.dot import dotnode
     >>> from sympy.abc import x
     >>> print(dotnode(x))
-    "Symbol(x)_()" ["color"="black", "label"="x", "shape"="ellipse"];
+    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
     """
     style = styleof(expr, styles)
 
@@ -102,20 +109,19 @@ def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True)
     >>> from sympy.abc import x
     >>> for e in dotedges(x+2):
     ...     print(e)
-    "Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";
-    "Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
     """
+    from sympy.utilities.misc import func_name
     if atom(expr):
         return []
     else:
-        # TODO: This is quadratic in complexity (purestr(expr) already
-        # contains [purestr(arg) for arg in expr.args]).
-        expr_str = purestr(expr)
-        arg_strs = [purestr(arg) for arg in expr.args]
+        expr_str, arg_strs = purestr(expr, with_args=True)
         if repeat:
             expr_str += '_%s' % str(pos)
-            arg_strs = [arg_str + '_%s' % str(pos + (i,)) for i, arg_str in enumerate(arg_strs)]
-        return ['"%s" -> "%s";' % (expr_str, arg_str) for arg_str in arg_strs]
+            arg_strs = ['%s_%s' % (a, str(pos + (i,)))
+                for i, a in enumerate(arg_strs)]
+        return ['"%s" -> "%s";' % (expr_str, a) for a in arg_strs]
 
 template = \
 """digraph{
@@ -161,7 +167,7 @@ def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
           ``repeat=True``, it will have two nodes for ``x`` and with
           ``repeat=False``, it will have one (warning: even if it appears
           twice in the same object, like Pow(x, x), it will still only appear
-          only once.  Hence, with repeat=False, the number of arrows out of an
+          once.  Hence, with repeat=False, the number of arrows out of an
           object might not equal the number of args it has).
 
     ``labelfunc``: How to label leaf nodes.  The default is ``str``.  Another
@@ -187,16 +193,16 @@ def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
     # Nodes #
     #########
     <BLANKLINE>
-    "Add(Integer(2), Symbol(x))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
+    "Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
     "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
-    "Symbol(x)_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
+    "Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
     <BLANKLINE>
     #########
     # Edges #
     #########
     <BLANKLINE>
-    "Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";
-    "Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
     }
 
     """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/dot.py | 10 | 10 | 1 | 1 | 132
| sympy/printing/dot.py | 17 | 30 | - | 1 | -
| sympy/printing/dot.py | 57 | 57 | - | 1 | -
| sympy/printing/dot.py | 69 | 69 | - | 1 | -
| sympy/printing/dot.py | 78 | 78 | - | 1 | -
| sympy/printing/dot.py | 105 | 118 | 96 | 1 | 38041
| sympy/printing/dot.py | 164 | 164 | 2 | 1 | 869
| sympy/printing/dot.py | 190 | 199 | 2 | 1 | 869


## Problem Statement

```
dotprint doesn't use the correct order for x**2
The dot diagram in the tutorial is wrong (http://docs.sympy.org/dev/tutorial/manipulation.html). It shows 

\`\`\`
          Pow
          /  \
Integer(2)    Symbol('x')
\`\`\`

but it should show

\`\`\`
           Pow
           /  \
Symbol('x')    Integer(2)
\`\`\`

since it represents `x**2`, not `2**x`. 

I can't figure out how to make dot give the vertices in the right order. Whatever the fix is, we should fix this in the dot printer as well as the tutorial.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sympy/printing/dot.py** | 1 | 18| 132 | 132 | 1916 | 
| **-> 2 <-** | **1 sympy/printing/dot.py** | 141 | 210| 737 | 869 | 1916 | 
| 3 | **1 sympy/printing/dot.py** | 211 | 222| 181 | 1050 | 1916 | 
| 4 | 2 sympy/printing/pycode.py | 516 | 527| 141 | 1191 | 8561 | 
| 5 | 3 sympy/printing/pretty/pretty.py | 848 | 884| 330 | 1521 | 31143 | 
| 6 | 3 sympy/printing/pretty/pretty.py | 124 | 134| 134 | 1655 | 31143 | 
| 7 | 4 sympy/printing/julia.py | 194 | 211| 198 | 1853 | 36898 | 
| 8 | 4 sympy/printing/pretty/pretty.py | 1 | 29| 276 | 2129 | 36898 | 
| 9 | 5 sympy/printing/theanocode.py | 141 | 215| 743 | 2872 | 41129 | 
| 10 | 6 sympy/printing/latex.py | 518 | 559| 458 | 3330 | 67255 | 
| 11 | 7 sympy/printing/codeprinter.py | 439 | 488| 449 | 3779 | 71643 | 
| 12 | 8 sympy/printing/octave.py | 136 | 208| 700 | 4479 | 78172 | 
| 13 | 8 sympy/printing/codeprinter.py | 386 | 437| 507 | 4986 | 78172 | 
| 14 | 8 sympy/printing/octave.py | 211 | 228| 199 | 5185 | 78172 | 
| 15 | 9 sympy/printing/printer.py | 1 | 173| 1419 | 6604 | 80555 | 
| 16 | 9 sympy/printing/pretty/pretty.py | 1700 | 1744| 501 | 7105 | 80555 | 
| 17 | 9 sympy/printing/pretty/pretty.py | 136 | 200| 615 | 7720 | 80555 | 
| 18 | 9 sympy/printing/pretty/pretty.py | 1254 | 1336| 773 | 8493 | 80555 | 
| 19 | 9 sympy/printing/julia.py | 214 | 256| 290 | 8783 | 80555 | 
| 20 | 9 sympy/printing/pretty/pretty.py | 464 | 516| 428 | 9211 | 80555 | 
| 21 | 10 sympy/printing/str.py | 777 | 791| 132 | 9343 | 87733 | 
| 22 | 10 sympy/printing/latex.py | 389 | 415| 306 | 9649 | 87733 | 
| 23 | 10 sympy/printing/str.py | 322 | 370| 370 | 10019 | 87733 | 
| 24 | 10 sympy/printing/latex.py | 2125 | 2206| 791 | 10810 | 87733 | 
| 25 | 10 sympy/printing/julia.py | 119 | 191| 697 | 11507 | 87733 | 
| 26 | 11 sympy/printing/mathematica.py | 1 | 35| 432 | 11939 | 89778 | 
| 27 | 11 sympy/printing/octave.py | 381 | 473| 811 | 12750 | 89778 | 
| 28 | 11 sympy/printing/latex.py | 955 | 1010| 577 | 13327 | 89778 | 
| 29 | 12 sympy/printing/pretty/pretty_symbology.py | 302 | 331| 230 | 13557 | 95667 | 
| 30 | 12 sympy/printing/octave.py | 341 | 357| 183 | 13740 | 95667 | 
| 31 | 12 sympy/printing/pretty/pretty_symbology.py | 260 | 300| 603 | 14343 | 95667 | 
| 32 | 12 sympy/printing/pretty/pretty.py | 1206 | 1252| 403 | 14746 | 95667 | 
| 33 | 13 sympy/physics/vector/printing.py | 1 | 12| 134 | 14880 | 99074 | 
| 34 | 14 sympy/printing/fcode.py | 338 | 355| 171 | 15051 | 107057 | 
| 35 | 14 sympy/printing/pretty/pretty.py | 332 | 367| 287 | 15338 | 107057 | 
| 36 | 14 sympy/printing/pretty/pretty.py | 106 | 122| 199 | 15537 | 107057 | 
| 37 | 14 sympy/printing/pycode.py | 428 | 454| 272 | 15809 | 107057 | 
| 38 | 14 sympy/printing/codeprinter.py | 490 | 524| 389 | 16198 | 107057 | 
| 39 | 15 sympy/printing/repr.py | 164 | 175| 141 | 16339 | 109337 | 
| 40 | 15 sympy/printing/codeprinter.py | 202 | 237| 257 | 16596 | 109337 | 
| 41 | 15 sympy/printing/theanocode.py | 1 | 65| 591 | 17187 | 109337 | 
| 42 | 15 sympy/printing/pretty/pretty.py | 2343 | 2375| 272 | 17459 | 109337 | 
| 43 | 15 sympy/printing/julia.py | 353 | 369| 181 | 17640 | 109337 | 
| 44 | 15 sympy/printing/pretty/pretty.py | 264 | 330| 583 | 18223 | 109337 | 
| 45 | 15 sympy/printing/octave.py | 231 | 252| 172 | 18395 | 109337 | 
| 46 | 15 sympy/printing/latex.py | 1568 | 1604| 312 | 18707 | 109337 | 
| 47 | 15 sympy/printing/mathematica.py | 144 | 163| 173 | 18880 | 109337 | 
| 48 | 15 sympy/printing/str.py | 557 | 618| 474 | 19354 | 109337 | 
| 49 | 15 sympy/printing/str.py | 689 | 775| 725 | 20079 | 109337 | 
| 50 | 15 sympy/printing/pretty/pretty.py | 720 | 745| 284 | 20363 | 109337 | 
| 51 | 15 sympy/printing/str.py | 397 | 453| 541 | 20904 | 109337 | 
| 52 | 15 sympy/printing/pretty/pretty.py | 1097 | 1124| 212 | 21116 | 109337 | 
| 53 | 16 sympy/printing/rust.py | 428 | 465| 349 | 21465 | 114760 | 
| 54 | 16 sympy/printing/latex.py | 1319 | 1379| 753 | 22218 | 114760 | 
| 55 | 16 sympy/printing/str.py | 204 | 247| 474 | 22692 | 114760 | 
| 56 | 16 sympy/printing/pretty/pretty.py | 1473 | 1490| 184 | 22876 | 114760 | 
| 57 | 16 sympy/printing/str.py | 266 | 320| 501 | 23377 | 114760 | 
| 58 | 16 sympy/printing/pretty/pretty.py | 1834 | 1887| 388 | 23765 | 114760 | 
| 59 | 16 sympy/printing/pretty/pretty.py | 2325 | 2341| 184 | 23949 | 114760 | 
| 60 | 16 sympy/physics/vector/printing.py | 121 | 155| 352 | 24301 | 114760 | 
| 61 | 16 sympy/printing/latex.py | 2208 | 2236| 215 | 24516 | 114760 | 
| 62 | 16 sympy/printing/pretty/pretty.py | 794 | 817| 216 | 24732 | 114760 | 
| 63 | 16 sympy/printing/pretty/pretty.py | 2392 | 2456| 612 | 25344 | 114760 | 
| 64 | 16 sympy/printing/str.py | 793 | 824| 287 | 25631 | 114760 | 
| 65 | 16 sympy/printing/pretty/pretty.py | 1746 | 1761| 184 | 25815 | 114760 | 
| 66 | 16 sympy/printing/fcode.py | 299 | 320| 253 | 26068 | 114760 | 
| 67 | 17 examples/beginner/print_pretty.py | 1 | 51| 284 | 26352 | 115044 | 
| 68 | 17 sympy/printing/pretty/pretty.py | 1968 | 1991| 227 | 26579 | 115044 | 
| 69 | 17 sympy/printing/codeprinter.py | 123 | 200| 718 | 27297 | 115044 | 
| 70 | 18 sympy/printing/rcode.py | 144 | 154| 130 | 27427 | 118763 | 
| 71 | 18 sympy/printing/rust.py | 1 | 55| 566 | 27993 | 118763 | 
| 72 | 18 sympy/printing/pycode.py | 705 | 733| 283 | 28276 | 118763 | 
| 73 | 18 sympy/printing/pretty/pretty.py | 1421 | 1440| 236 | 28512 | 118763 | 
| 74 | 18 sympy/printing/pretty/pretty.py | 1126 | 1180| 420 | 28932 | 118763 | 
| 75 | 18 sympy/printing/pretty/pretty.py | 1643 | 1698| 564 | 29496 | 118763 | 
| 76 | 18 sympy/printing/pretty/pretty.py | 32 | 92| 535 | 30031 | 118763 | 
| 77 | 18 sympy/printing/str.py | 521 | 555| 415 | 30446 | 118763 | 
| 78 | 19 sympy/interactive/printing.py | 271 | 380| 1205 | 31651 | 122754 | 
| 79 | 19 sympy/printing/rust.py | 337 | 399| 468 | 32119 | 122754 | 
| 80 | 19 sympy/printing/repr.py | 47 | 98| 396 | 32515 | 122754 | 
| 81 | 19 sympy/printing/pretty/pretty.py | 2312 | 2323| 147 | 32662 | 122754 | 
| 82 | 20 sympy/printing/mathml.py | 903 | 955| 472 | 33134 | 138249 | 
| 83 | 20 sympy/printing/pycode.py | 556 | 614| 774 | 33908 | 138249 | 
| 84 | 21 sympy/physics/vector/dyadic.py | 312 | 343| 376 | 34284 | 142891 | 
| 85 | 21 sympy/printing/pretty/pretty.py | 94 | 104| 135 | 34419 | 142891 | 
| 86 | 21 sympy/printing/pretty/pretty.py | 2252 | 2268| 190 | 34609 | 142891 | 
| 87 | 21 sympy/printing/latex.py | 2238 | 2304| 734 | 35343 | 142891 | 
| 88 | 21 sympy/printing/latex.py | 1381 | 1395| 146 | 35489 | 142891 | 
| 89 | 21 sympy/printing/pycode.py | 1 | 75| 650 | 36139 | 142891 | 
| 90 | 21 sympy/printing/str.py | 455 | 519| 468 | 36607 | 142891 | 
| 91 | 21 sympy/printing/pycode.py | 268 | 327| 539 | 37146 | 142891 | 
| 92 | 22 sympy/printing/jscode.py | 100 | 110| 133 | 37279 | 145706 | 
| 93 | 22 sympy/printing/latex.py | 2328 | 2336| 129 | 37408 | 145706 | 
| 94 | 22 sympy/printing/theanocode.py | 234 | 252| 171 | 37579 | 145706 | 
| 95 | 22 sympy/printing/pretty/pretty.py | 836 | 846| 124 | 37703 | 145706 | 
| **-> 96 <-** | **22 sympy/printing/dot.py** | 93 | 139| 338 | 38041 | 145706 | 
| 97 | 22 sympy/printing/latex.py | 417 | 467| 385 | 38426 | 145706 | 
| 98 | 22 sympy/printing/pycode.py | 642 | 667| 254 | 38680 | 145706 | 
| 99 | 22 sympy/printing/pretty/pretty.py | 1513 | 1536| 258 | 38938 | 145706 | 
| 100 | 22 sympy/printing/latex.py | 2116 | 2123| 118 | 39056 | 145706 | 
| 101 | 22 sympy/printing/latex.py | 1122 | 1153| 308 | 39364 | 145706 | 
| 102 | 22 sympy/printing/latex.py | 84 | 119| 491 | 39855 | 145706 | 
| 103 | 22 sympy/printing/pretty/pretty.py | 1442 | 1455| 142 | 39997 | 145706 | 
| 104 | 22 sympy/printing/mathematica.py | 211 | 238| 321 | 40318 | 145706 | 
| 105 | 22 sympy/printing/octave.py | 476 | 523| 519 | 40837 | 145706 | 
| 106 | 22 sympy/printing/latex.py | 647 | 678| 274 | 41111 | 145706 | 
| 107 | 22 sympy/printing/fcode.py | 322 | 336| 131 | 41242 | 145706 | 
| 108 | 22 sympy/printing/latex.py | 2104 | 2114| 124 | 41366 | 145706 | 
| 109 | 22 sympy/printing/mathml.py | 423 | 445| 185 | 41551 | 145706 | 
| 110 | 22 sympy/printing/pretty/pretty.py | 1600 | 1641| 325 | 41876 | 145706 | 
| 111 | 22 sympy/printing/latex.py | 469 | 516| 529 | 42405 | 145706 | 
| 112 | 22 sympy/printing/latex.py | 1809 | 1829| 210 | 42615 | 145706 | 
| 113 | 22 sympy/printing/latex.py | 1691 | 1758| 571 | 43186 | 145706 | 
| 114 | 22 sympy/printing/pretty/pretty.py | 518 | 561| 479 | 43665 | 145706 | 
| 115 | 22 sympy/printing/latex.py | 1760 | 1768| 135 | 43800 | 145706 | 
| 116 | 22 sympy/printing/pretty/pretty.py | 1024 | 1057| 331 | 44131 | 145706 | 
| 117 | 22 sympy/printing/fcode.py | 256 | 297| 340 | 44471 | 145706 | 
| 118 | 22 sympy/printing/octave.py | 255 | 280| 282 | 44753 | 145706 | 
| 119 | 22 sympy/printing/pretty/pretty.py | 2377 | 2390| 129 | 44882 | 145706 | 
| 120 | 22 sympy/printing/pretty/pretty.py | 2458 | 2466| 119 | 45001 | 145706 | 
| 121 | 22 sympy/printing/jscode.py | 112 | 139| 215 | 45216 | 145706 | 
| 122 | 22 sympy/printing/latex.py | 847 | 927| 767 | 45983 | 145706 | 
| 123 | 22 sympy/printing/pretty/pretty.py | 1059 | 1095| 307 | 46290 | 145706 | 
| 124 | 22 sympy/printing/pretty/pretty.py | 2235 | 2250| 179 | 46469 | 145706 | 
| 125 | 22 sympy/printing/str.py | 70 | 158| 805 | 47274 | 145706 | 
| 126 | 22 sympy/printing/rcode.py | 229 | 263| 371 | 47645 | 145706 | 
| 127 | 22 sympy/printing/pretty/pretty.py | 1538 | 1551| 180 | 47825 | 145706 | 
| 128 | 22 sympy/printing/latex.py | 561 | 577| 199 | 48024 | 145706 | 
| 129 | 22 sympy/printing/latex.py | 1897 | 1959| 542 | 48566 | 145706 | 
| 130 | 22 sympy/printing/latex.py | 1770 | 1780| 141 | 48707 | 145706 | 
| 131 | 22 sympy/printing/pretty/pretty.py | 1932 | 1966| 311 | 49018 | 145706 | 
| 132 | 22 sympy/printing/latex.py | 2318 | 2326| 123 | 49141 | 145706 | 
| 133 | 22 sympy/printing/pycode.py | 456 | 475| 200 | 49341 | 145706 | 
| 134 | 22 sympy/printing/mathml.py | 397 | 421| 228 | 49569 | 145706 | 
| 135 | 22 sympy/printing/pretty/pretty.py | 770 | 792| 224 | 49793 | 145706 | 
| 136 | 22 sympy/printing/repr.py | 114 | 145| 256 | 50049 | 145706 | 
| 137 | 23 sympy/printing/llvmjitcode.py | 60 | 80| 242 | 50291 | 149741 | 
| 138 | 23 sympy/printing/octave.py | 1 | 59| 662 | 50953 | 149741 | 
| 139 | 24 sympy/printing/glsl.py | 284 | 303| 281 | 51234 | 154554 | 
| 140 | 24 sympy/printing/fcode.py | 149 | 164| 131 | 51365 | 154554 | 
| 141 | 24 sympy/printing/codeprinter.py | 363 | 384| 228 | 51593 | 154554 | 
| 142 | 24 sympy/printing/pycode.py | 735 | 753| 166 | 51759 | 154554 | 
| 143 | 24 sympy/printing/fcode.py | 357 | 391| 383 | 52142 | 154554 | 
| 144 | 24 sympy/printing/mathml.py | 210 | 236| 226 | 52368 | 154554 | 
| 145 | 24 sympy/printing/glsl.py | 247 | 282| 303 | 52671 | 154554 | 
| 146 | 24 sympy/printing/latex.py | 1263 | 1317| 730 | 53401 | 154554 | 
| 147 | 24 sympy/physics/vector/dyadic.py | 155 | 190| 401 | 53802 | 154554 | 
| 148 | 24 sympy/printing/pretty/pretty.py | 1374 | 1388| 194 | 53996 | 154554 | 
| 149 | 24 sympy/printing/latex.py | 2079 | 2089| 126 | 54122 | 154554 | 
| 150 | 24 sympy/printing/str.py | 46 | 68| 160 | 54282 | 154554 | 
| 151 | 24 sympy/printing/pretty/pretty.py | 1492 | 1511| 194 | 54476 | 154554 | 
| 152 | 24 sympy/printing/str.py | 372 | 395| 266 | 54742 | 154554 | 
| 153 | 24 sympy/printing/pretty/pretty.py | 2035 | 2085| 366 | 55108 | 154554 | 
| 154 | 24 sympy/printing/fcode.py | 166 | 189| 207 | 55315 | 154554 | 
| 155 | 24 sympy/printing/pretty/pretty.py | 563 | 621| 510 | 55825 | 154554 | 
| 156 | 24 sympy/printing/pretty/pretty.py | 2270 | 2310| 403 | 56228 | 154554 | 
| 157 | 24 sympy/printing/pretty/pretty_symbology.py | 169 | 218| 455 | 56683 | 154554 | 
| 158 | 25 sympy/printing/ccode.py | 275 | 289| 204 | 56887 | 162742 | 
| 159 | 25 sympy/printing/julia.py | 393 | 418| 238 | 57125 | 162742 | 
| 160 | 26 sympy/printing/tensorflow.py | 144 | 166| 184 | 57309 | 164831 | 
| 161 | 26 sympy/printing/str.py | 249 | 264| 146 | 57455 | 164831 | 
| 162 | 26 sympy/printing/pretty/pretty.py | 226 | 241| 165 | 57620 | 164831 | 
| 163 | 26 sympy/printing/latex.py | 1973 | 2020| 457 | 58077 | 164831 | 
| 164 | 26 sympy/printing/mathml.py | 265 | 305| 344 | 58421 | 164831 | 
| 165 | 26 sympy/printing/latex.py | 1012 | 1021| 126 | 58547 | 164831 | 
| 166 | 26 sympy/printing/codeprinter.py | 239 | 292| 429 | 58976 | 164831 | 
| 167 | 26 sympy/printing/latex.py | 1831 | 1847| 150 | 59126 | 164831 | 
| 168 | 26 sympy/printing/pycode.py | 329 | 347| 216 | 59342 | 164831 | 
| 169 | 26 sympy/printing/pycode.py | 213 | 227| 127 | 59469 | 164831 | 
| 170 | 26 sympy/printing/latex.py | 326 | 346| 148 | 59617 | 164831 | 
| 171 | 26 sympy/printing/pretty/pretty.py | 1182 | 1204| 180 | 59797 | 164831 | 
| 172 | 26 sympy/printing/latex.py | 690 | 722| 325 | 60122 | 164831 | 
| 173 | 26 sympy/printing/latex.py | 1558 | 1566| 120 | 60242 | 164831 | 
| 174 | 26 sympy/printing/str.py | 1 | 15| 108 | 60350 | 164831 | 
| 175 | 26 sympy/printing/latex.py | 1155 | 1170| 138 | 60488 | 164831 | 
| 176 | 26 sympy/printing/latex.py | 1172 | 1237| 668 | 61156 | 164831 | 
| 177 | 26 sympy/printing/codeprinter.py | 330 | 361| 257 | 61413 | 164831 | 
| 178 | 26 sympy/printing/latex.py | 1036 | 1120| 813 | 62226 | 164831 | 
| 179 | 26 sympy/printing/pycode.py | 350 | 363| 152 | 62378 | 164831 | 
| 180 | 26 sympy/printing/octave.py | 283 | 314| 175 | 62553 | 164831 | 
| 181 | 26 sympy/printing/latex.py | 1537 | 1556| 167 | 62720 | 164831 | 
| 182 | 26 sympy/physics/vector/printing.py | 277 | 305| 226 | 62946 | 164831 | 
| 183 | 26 sympy/printing/ccode.py | 373 | 401| 314 | 63260 | 164831 | 
| 184 | 26 sympy/printing/codeprinter.py | 294 | 328| 358 | 63618 | 164831 | 
| 185 | 26 sympy/printing/latex.py | 1514 | 1535| 215 | 63833 | 164831 | 
| 186 | 26 sympy/printing/str.py | 663 | 687| 247 | 64080 | 164831 | 
| 187 | 26 sympy/printing/fcode.py | 393 | 408| 124 | 64204 | 164831 | 
| 188 | 26 sympy/printing/pretty/pretty.py | 1338 | 1347| 131 | 64335 | 164831 | 
| 189 | 26 sympy/printing/pretty/pretty.py | 819 | 834| 136 | 64471 | 164831 | 
| 190 | **26 sympy/printing/dot.py** | 19 | 30| 124 | 64595 | 164831 | 
| 191 | 26 sympy/printing/pretty/pretty.py | 1784 | 1808| 228 | 64823 | 164831 | 
| 192 | 26 sympy/printing/pretty/pretty.py | 623 | 649| 261 | 65084 | 164831 | 
| 193 | 26 sympy/printing/mathml.py | 1401 | 1494| 838 | 65922 | 164831 | 
| 194 | 26 sympy/printing/mathml.py | 1656 | 1708| 445 | 66367 | 164831 | 
| 195 | 26 sympy/printing/repr.py | 177 | 210| 382 | 66749 | 164831 | 
| 196 | 26 sympy/printing/mathml.py | 957 | 1016| 430 | 67179 | 164831 | 
| 197 | 26 sympy/printing/latex.py | 1397 | 1431| 244 | 67423 | 164831 | 
| 198 | 26 sympy/printing/fcode.py | 410 | 426| 187 | 67610 | 164831 | 
| 199 | 26 sympy/printing/pretty/pretty.py | 986 | 1022| 378 | 67988 | 164831 | 
| 200 | 26 sympy/printing/latex.py | 1782 | 1791| 134 | 68122 | 164831 | 


## Patch

```diff
diff --git a/sympy/printing/dot.py b/sympy/printing/dot.py
--- a/sympy/printing/dot.py
+++ b/sympy/printing/dot.py
@@ -7,6 +7,7 @@
 from sympy.core.compatibility import default_sort_key
 from sympy.core.add import Add
 from sympy.core.mul import Mul
+from sympy.printing.repr import srepr
 
 __all__ = ['dotprint']
 
@@ -14,20 +15,24 @@
           (Expr,  {'color': 'black'}))
 
 
-sort_classes = (Add, Mul)
 slotClasses = (Symbol, Integer, Rational, Float)
-# XXX: Why not just use srepr()?
-def purestr(x):
+def purestr(x, with_args=False):
     """ A string that follows obj = type(obj)(*obj.args) exactly """
+    sargs = ()
     if not isinstance(x, Basic):
-        return str(x)
-    if type(x) in slotClasses:
-        args = [getattr(x, slot) for slot in x.__slots__]
-    elif type(x) in sort_classes:
-        args = sorted(x.args, key=default_sort_key)
+        rv = str(x)
+    elif not x.args:
+        rv = srepr(x)
     else:
         args = x.args
-    return "%s(%s)"%(type(x).__name__, ', '.join(map(purestr, args)))
+        if isinstance(x, Add) or \
+                isinstance(x, Mul) and x.is_commutative:
+            args = sorted(args, key=default_sort_key)
+        sargs = tuple(map(purestr, args))
+        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
+    if with_args:
+        rv = rv, sargs
+    return rv
 
 
 def styleof(expr, styles=default_styles):
@@ -54,6 +59,7 @@ def styleof(expr, styles=default_styles):
             style.update(sty)
     return style
 
+
 def attrprint(d, delimiter=', '):
     """ Print a dictionary of attributes
 
@@ -66,6 +72,7 @@ def attrprint(d, delimiter=', '):
     """
     return delimiter.join('"%s"="%s"'%item for item in sorted(d.items()))
 
+
 def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
     """ String defining a node
 
@@ -75,7 +82,7 @@ def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
     >>> from sympy.printing.dot import dotnode
     >>> from sympy.abc import x
     >>> print(dotnode(x))
-    "Symbol(x)_()" ["color"="black", "label"="x", "shape"="ellipse"];
+    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
     """
     style = styleof(expr, styles)
 
@@ -102,20 +109,19 @@ def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True)
     >>> from sympy.abc import x
     >>> for e in dotedges(x+2):
     ...     print(e)
-    "Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";
-    "Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
     """
+    from sympy.utilities.misc import func_name
     if atom(expr):
         return []
     else:
-        # TODO: This is quadratic in complexity (purestr(expr) already
-        # contains [purestr(arg) for arg in expr.args]).
-        expr_str = purestr(expr)
-        arg_strs = [purestr(arg) for arg in expr.args]
+        expr_str, arg_strs = purestr(expr, with_args=True)
         if repeat:
             expr_str += '_%s' % str(pos)
-            arg_strs = [arg_str + '_%s' % str(pos + (i,)) for i, arg_str in enumerate(arg_strs)]
-        return ['"%s" -> "%s";' % (expr_str, arg_str) for arg_str in arg_strs]
+            arg_strs = ['%s_%s' % (a, str(pos + (i,)))
+                for i, a in enumerate(arg_strs)]
+        return ['"%s" -> "%s";' % (expr_str, a) for a in arg_strs]
 
 template = \
 """digraph{
@@ -161,7 +167,7 @@ def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
           ``repeat=True``, it will have two nodes for ``x`` and with
           ``repeat=False``, it will have one (warning: even if it appears
           twice in the same object, like Pow(x, x), it will still only appear
-          only once.  Hence, with repeat=False, the number of arrows out of an
+          once.  Hence, with repeat=False, the number of arrows out of an
           object might not equal the number of args it has).
 
     ``labelfunc``: How to label leaf nodes.  The default is ``str``.  Another
@@ -187,16 +193,16 @@ def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
     # Nodes #
     #########
     <BLANKLINE>
-    "Add(Integer(2), Symbol(x))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
+    "Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
     "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
-    "Symbol(x)_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
+    "Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
     <BLANKLINE>
     #########
     # Edges #
     #########
     <BLANKLINE>
-    "Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";
-    "Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
+    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
     }
 
     """

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_dot.py b/sympy/printing/tests/test_dot.py
--- a/sympy/printing/tests/test_dot.py
+++ b/sympy/printing/tests/test_dot.py
@@ -1,11 +1,13 @@
 from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
         dotedges, dotprint)
-from sympy import Symbol, Integer, Basic, Expr, srepr
+from sympy import Symbol, Integer, Basic, Expr, srepr, Float, symbols
 from sympy.abc import x
 
+
 def test_purestr():
-    assert purestr(Symbol('x')) == "Symbol(x)"
+    assert purestr(Symbol('x')) == "Symbol('x')"
     assert purestr(Basic(1, 2)) == "Basic(1, 2)"
+    assert purestr(Float(2)) == "Float('2.0', precision=53)"
 
 
 def test_styleof():
@@ -15,6 +17,7 @@ def test_styleof():
 
     assert styleof(x + 1, styles) == {'color': 'black', 'shape': 'ellipse'}
 
+
 def test_attrprint():
     assert attrprint({'color': 'blue', 'shape': 'ellipse'}) == \
            '"color"="blue", "shape"="ellipse"'
@@ -22,23 +25,23 @@ def test_attrprint():
 def test_dotnode():
 
     assert dotnode(x, repeat=False) ==\
-            '"Symbol(x)" ["color"="black", "label"="x", "shape"="ellipse"];'
+            '"Symbol(\'x\')" ["color"="black", "label"="x", "shape"="ellipse"];'
     assert dotnode(x+2, repeat=False) == \
-            '"Add(Integer(2), Symbol(x))" ["color"="black", "label"="Add", "shape"="ellipse"];'
+            '"Add(Integer(2), Symbol(\'x\'))" ["color"="black", "label"="Add", "shape"="ellipse"];', dotnode(x+2,repeat=0)
 
     assert dotnode(x + x**2, repeat=False) == \
-        '"Add(Symbol(x), Pow(Symbol(x), Integer(2)))" ["color"="black", "label"="Add", "shape"="ellipse"];'
+        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))" ["color"="black", "label"="Add", "shape"="ellipse"];'
     assert dotnode(x + x**2, repeat=True) == \
-        '"Add(Symbol(x), Pow(Symbol(x), Integer(2)))_()" ["color"="black", "label"="Add", "shape"="ellipse"];'
+        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))_()" ["color"="black", "label"="Add", "shape"="ellipse"];'
 
 def test_dotedges():
     assert sorted(dotedges(x+2, repeat=False)) == [
-        '"Add(Integer(2), Symbol(x))" -> "Integer(2)";',
-        '"Add(Integer(2), Symbol(x))" -> "Symbol(x)";'
+        '"Add(Integer(2), Symbol(\'x\'))" -> "Integer(2)";',
+        '"Add(Integer(2), Symbol(\'x\'))" -> "Symbol(\'x\')";'
         ]
     assert sorted(dotedges(x + 2, repeat=True)) == [
-        '"Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";',
-        '"Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";'
+        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Integer(2)_(0,)";',
+        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Symbol(\'x\')_(1,)";'
     ]
 
 def test_dotprint():
@@ -74,3 +77,9 @@ def test_labelfunc():
     text = dotprint(x + 2, labelfunc=srepr)
     assert "Symbol('x')" in text
     assert "Integer(2)" in text
+
+
+def test_commutative():
+    x, y = symbols('x y', commutative=False)
+    assert dotprint(x + y) == dotprint(y + x)
+    assert dotprint(x*y) != dotprint(y*x)

```


## Code snippets

### 1 - sympy/printing/dot.py:

Start line: 1, End line: 18

```python
from __future__ import print_function, division

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.core.compatibility import default_sort_key
from sympy.core.add import Add
from sympy.core.mul import Mul

__all__ = ['dotprint']

default_styles = ((Basic, {'color': 'blue', 'shape': 'ellipse'}),
          (Expr,  {'color': 'black'}))


sort_classes = (Add, Mul)
slotClasses = (Symbol, Integer, Rational, Float)
```
### 2 - sympy/printing/dot.py:

Start line: 141, End line: 210

```python
def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
    Basic), maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    """
    DOT description of a SymPy expression tree

    Options are

    ``styles``: Styles for different classes.  The default is::

        [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
        (Expr, {'color': 'black'})]``

    ``atom``: Function used to determine if an arg is an atom.  The default is
          ``lambda x: not isinstance(x, Basic)``.  Another good choice is
          ``lambda x: not x.args``.

    ``maxdepth``: The maximum depth.  The default is None, meaning no limit.

    ``repeat``: Whether to different nodes for separate common subexpressions.
          The default is True.  For example, for ``x + x*y`` with
          ``repeat=True``, it will have two nodes for ``x`` and with
          ``repeat=False``, it will have one (warning: even if it appears
          twice in the same object, like Pow(x, x), it will still only appear
          only once.  Hence, with repeat=False, the number of arrows out of an
          object might not equal the number of args it has).

    ``labelfunc``: How to label leaf nodes.  The default is ``str``.  Another
          good option is ``srepr``. For example with ``str``, the leaf nodes
          of ``x + 1`` are labeled, ``x`` and ``1``.  With ``srepr``, they
          are labeled ``Symbol('x')`` and ``Integer(1)``.

    Additional keyword arguments are included as styles for the graph.

    Examples
    ========

    >>> from sympy.printing.dot import dotprint
    >>> from sympy.abc import x
    >>> print(dotprint(x+2)) # doctest: +NORMALIZE_WHITESPACE
    digraph{
    <BLANKLINE>
    # Graph style
    "ordering"="out"
    "rankdir"="TD"
    <BLANKLINE>
    #########
    # Nodes #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol(x))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
    "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
    "Symbol(x)_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
    <BLANKLINE>
    #########
    # Edges #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";
    }

    """
    # repeat works by adding a signature tuple to the end of each node for its
    # position in the graph. For example, for expr = Add(x, Pow(x, 2)), the x in the
    # Pow will have the tuple (1, 0), meaning it is expr.args[1].args[0].
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)

    nodes = []
    edges = []
    # ... other code
```
### 3 - sympy/printing/dot.py:

Start line: 211, End line: 222

```python
def dotprint(expr, styles=default_styles, atom=lambda x: not isinstance(x,
    Basic), maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    # ... other code
    def traverse(e, depth, pos=()):
        nodes.append(dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat))
        if maxdepth and depth >= maxdepth:
            return
        edges.extend(dotedges(e, atom=atom, pos=pos, repeat=repeat))
        [traverse(arg, depth+1, pos + (i,)) for i, arg in enumerate(e.args) if not atom(arg)]
    traverse(expr, 0)

    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}
```
### 4 - sympy/printing/pycode.py:

Start line: 516, End line: 527

```python
class NumPyPrinter(PythonCodePrinter):

    def _print_DotProduct(self, expr):
        # DotProduct allows any shape order, but numpy.dot does matrix
        # multiplication, so we have to make sure it gets 1 x n by n x 1.
        arg1, arg2 = expr.args
        if arg1.shape[0] != 1:
            arg1 = arg1.T
        if arg2.shape[1] != 1:
            arg2 = arg2.T

        return "%s(%s, %s)" % (self._module_format('numpy.dot'),
                               self._print(arg1),
                               self._print(arg2))
```
### 5 - sympy/printing/pretty/pretty.py:

Start line: 848, End line: 884

```python
class PrettyPrinter(Printer):

    def _print_DotProduct(self, expr):
        args = list(expr.args)

        for i, a in enumerate(args):
            args[i] = self._print(a)
        return prettyForm.__mul__(*args)

    def _print_MatPow(self, expr):
        pform = self._print(expr.base)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.base, MatrixSymbol):
            pform = prettyForm(*pform.parens())
        pform = pform**(self._print(expr.exp))
        return pform

    def _print_HadamardProduct(self, expr):
        from sympy import MatAdd, MatMul
        if self._use_unicode:
            delim = pretty_atom('Ring')
        else:
            delim = '.*'
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    def _print_KroneckerProduct(self, expr):
        from sympy import MatAdd, MatMul
        if self._use_unicode:
            delim = u' \N{N-ARY CIRCLED TIMES OPERATOR} '
        else:
            delim = ' x '
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    def _print_FunctionMatrix(self, X):
        D = self._print(X.lamda.expr)
        D = prettyForm(*D.parens('[', ']'))
        return D
```
### 6 - sympy/printing/pretty/pretty.py:

Start line: 124, End line: 134

```python
class PrettyPrinter(Printer):

    def _print_Dot(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform
```
### 7 - sympy/printing/julia.py:

Start line: 194, End line: 211

```python
class JuliaCodePrinter(CodePrinter):


    def _print_Pow(self, expr):
        powsymbol = '^' if all([x.is_number for x in expr.args]) else '.^'

        PREC = precedence(expr)

        if expr.exp == S.Half:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if expr.exp == -S.Half:
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            if expr.exp == -S.One:
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))
```
### 8 - sympy/printing/pretty/pretty.py:

Start line: 1, End line: 29

```python
from __future__ import print_function, division

import itertools

from sympy.core import S
from sympy.core.compatibility import range, string_types
from sympy.core.containers import Tuple
from sympy.core.function import _coeff_isneg
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer
from sympy.printing.str import sstr
from sympy.utilities import default_sort_key
from sympy.utilities.iterables import has_variety

from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import xstr, hobj, vobj, xobj, \
    xsym, pretty_symbol, pretty_atom, pretty_use_unicode, greek_unicode, U, \
    pretty_try_use_unicode,  annotated

# rename for usage from outside
pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode
```
### 9 - sympy/printing/theanocode.py:

Start line: 141, End line: 215

```python
class TheanoPrinter(Printer):

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        if not hasattr(tt, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Theano")

        return tt.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = tt.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = tt.dot(result, children[0])
        else:
            raise NotImplementedError('''Only non-negative integer
            owers of matrices can be handled by Theano at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return tt.join(0, *[tt.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return 3.141592653589793
```
### 10 - sympy/printing/latex.py:

Start line: 518, End line: 559

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1 \
                and self._settings['root_notation']:
            base = self._print(expr.base)
            expq = expr.exp.q

            if expq == 2:
                tex = r"\sqrt{%s}" % base
            elif self._settings['itex']:
                tex = r"\root{%d}{%s}" % (expq, base)
            else:
                tex = r"\sqrt[%d]{%s}" % (expq, base)

            if expr.exp.is_negative:
                return r"\frac{1}{%s}" % tex
            else:
                return tex
        elif self._settings['fold_frac_powers'] \
            and expr.exp.is_Rational \
                and expr.exp.q != 1:
            base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
            p, q = expr.exp.p, expr.exp.q
            # issue #12886: add parentheses for superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, exp="%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and \
                expr.base.is_commutative:
            # special case for 1^(-x), issue 9216
            if expr.base == 1:
                return r"%s^{%s}" % (expr.base, expr.exp)
            # things like 1/x
            return self._print_Mul(expr)
        else:
            if expr.base.is_Function:
                return self._print(expr.base, exp=self._print(expr.exp))
            else:
                tex = r"%s^{%s}"
                return self._helper_print_standard_power(expr, tex)
```
### 96 - sympy/printing/dot.py:

Start line: 93, End line: 139

```python
def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True):
    """ List of strings for all expr->expr.arg pairs

    See the docstring of dotprint for explanations of the options.

    Examples
    ========

    >>> from sympy.printing.dot import dotedges
    >>> from sympy.abc import x
    >>> for e in dotedges(x+2):
    ...     print(e)
    "Add(Integer(2), Symbol(x))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol(x))_()" -> "Symbol(x)_(1,)";
    """
    if atom(expr):
        return []
    else:
        # TODO: This is quadratic in complexity (purestr(expr) already
        # contains [purestr(arg) for arg in expr.args]).
        expr_str = purestr(expr)
        arg_strs = [purestr(arg) for arg in expr.args]
        if repeat:
            expr_str += '_%s' % str(pos)
            arg_strs = [arg_str + '_%s' % str(pos + (i,)) for i, arg_str in enumerate(arg_strs)]
        return ['"%s" -> "%s";' % (expr_str, arg_str) for arg_str in arg_strs]

template =
"""digraph{

# Graph style
%(graphstyle)s

#########
# Nodes #
#########

%(nodes)s

#########
# Edges #
#########

%(edges)s
}"""

_graphstyle = {'rankdir': 'TD', 'ordering': 'out'}
```
### 190 - sympy/printing/dot.py:

Start line: 19, End line: 30

```python
# XXX: Why not just use srepr()?
def purestr(x):
    """ A string that follows obj = type(obj)(*obj.args) exactly """
    if not isinstance(x, Basic):
        return str(x)
    if type(x) in slotClasses:
        args = [getattr(x, slot) for slot in x.__slots__]
    elif type(x) in sort_classes:
        args = sorted(x.args, key=default_sort_key)
    else:
        args = x.args
    return "%s(%s)"%(type(x).__name__, ', '.join(map(purestr, args)))
```
