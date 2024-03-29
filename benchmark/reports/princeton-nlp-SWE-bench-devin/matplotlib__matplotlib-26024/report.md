# matplotlib__matplotlib-26024

| **matplotlib/matplotlib** | `bfaa6eb677b9c56cafb6a99d6897c9d0cd9d4210` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 12416 |
| **Any found context length** | 12416 |
| **Avg pos** | 36.0 |
| **Min pos** | 18 |
| **Max pos** | 18 |
| **Top file pos** | 8 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/lib/matplotlib/_mathtext_data.py b/lib/matplotlib/_mathtext_data.py
--- a/lib/matplotlib/_mathtext_data.py
+++ b/lib/matplotlib/_mathtext_data.py
@@ -1008,8 +1008,6 @@
     'leftparen'                : 40,
     'rightparen'               : 41,
     'bigoplus'                 : 10753,
-    'leftangle'                : 10216,
-    'rightangle'               : 10217,
     'leftbrace'                : 124,
     'rightbrace'               : 125,
     'jmath'                    : 567,
@@ -1032,7 +1030,55 @@
     'guilsinglleft'            : 8249,
     'plus'                     : 43,
     'thorn'                    : 254,
-    'dagger'                   : 8224
+    'dagger'                   : 8224,
+    'increment'                : 8710,
+    'smallin'                  : 8714,
+    'notsmallowns'             : 8716,
+    'smallowns'                : 8717,
+    'QED'                      : 8718,
+    'rightangle'               : 8735,
+    'smallintclockwise'        : 8753,
+    'smallvarointclockwise'    : 8754,
+    'smallointctrcclockwise'   : 8755,
+    'ratio'                    : 8758,
+    'minuscolon'               : 8761,
+    'dotsminusdots'            : 8762,
+    'sinewave'                 : 8767,
+    'simneqq'                  : 8774,
+    'nlesssim'                 : 8820,
+    'ngtrsim'                  : 8821,
+    'nlessgtr'                 : 8824,
+    'ngtrless'                 : 8825,
+    'cupleftarrow'             : 8844,
+    'oequal'                   : 8860,
+    'rightassert'              : 8870,
+    'rightModels'              : 8875,
+    'hermitmatrix'             : 8889,
+    'barvee'                   : 8893,
+    'measuredrightangle'       : 8894,
+    'varlrtriangle'            : 8895,
+    'equalparallel'            : 8917,
+    'npreccurlyeq'             : 8928,
+    'nsucccurlyeq'             : 8929,
+    'nsqsubseteq'              : 8930,
+    'nsqsupseteq'              : 8931,
+    'sqsubsetneq'              : 8932,
+    'sqsupsetneq'              : 8933,
+    'disin'                    : 8946,
+    'varisins'                 : 8947,
+    'isins'                    : 8948,
+    'isindot'                  : 8949,
+    'varisinobar'              : 8950,
+    'isinobar'                 : 8951,
+    'isinvb'                   : 8952,
+    'isinE'                    : 8953,
+    'nisd'                     : 8954,
+    'varnis'                   : 8955,
+    'nis'                      : 8956,
+    'varniobar'                : 8957,
+    'niobar'                   : 8958,
+    'bagmember'                : 8959,
+    'triangle'                 : 9651
 }
 
 # Each element is a 4-tuple of the form:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/_mathtext_data.py | 1011 | 1012 | 18 | 8 | 12416
| lib/matplotlib/_mathtext_data.py | 1035 | 1035 | 18 | 8 | 12416


## Problem Statement

```
[ENH]: Missing mathematical operations
### Problem

Just browsed the available mathematical operators and compared with the ones defined.

(One can probably do a similar thing with other groups of symbols.)

### Proposed solution

The following are missing (as in not defined in `tex2uni` in `_mathtext_data.py`, in hex):

\`\`\`
2206 220a 220c 220d 220e 221b 221c 221f 2231 2232 2233 2236 2239
223a 223f 2246 226d 2274 2275 2278 2279 228c 229c 22a6 22ab 22b9
22bd 22be 22bf 22d5 22e0 22e1 22e2 22e3 22e4 22e5 22f2 22f3 22f4
22f5 22f6 22f7 22f8 22f9 22fa 22fb 22fc 22fd 22fe 22ff
\`\`\`

For the corresponding symbols, see: https://www.compart.com/en/unicode/block/U+2200

For LaTeX names, see: https://tug.ctan.org/info/symbols/comprehensive/symbols-a4.pdf

One should probably be a bit discriminate when adding these, but at least those in standard LaTeX (like `0x2206` = `\triangle`) and those from AMS should be supported.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 galleries/users_explain/text/mathtext.py | 1 | 377| 3408 | 3408 | 3408 | 
| 2 | 2 lib/matplotlib/_type1font.py | 777 | 878| 808 | 4216 | 10227 | 
| 3 | 3 lib/matplotlib/_mathtext.py | 515 | 581| 630 | 4846 | 33473 | 
| 4 | 3 lib/matplotlib/_mathtext.py | 1706 | 1778| 781 | 5627 | 33473 | 
| 5 | 3 lib/matplotlib/_mathtext.py | 2081 | 2124| 473 | 6100 | 33473 | 
| 6 | 3 lib/matplotlib/_mathtext.py | 1821 | 1975| 1406 | 7506 | 33473 | 
| 7 | 3 lib/matplotlib/_mathtext.py | 456 | 513| 568 | 8074 | 33473 | 
| 8 | 4 galleries/users_explain/text/usetex.py | 1 | 178| 1786 | 9860 | 35259 | 
| 9 | 5 doc/sphinxext/math_symbol_table.py | 132 | 169| 241 | 10101 | 37468 | 
| 10 | 5 lib/matplotlib/_mathtext.py | 1 | 31| 200 | 10301 | 37468 | 
| 11 | 6 galleries/examples/text_labels_and_annotations/mathtext_examples.py | 1 | 55| 669 | 10970 | 38737 | 
| 12 | 6 doc/sphinxext/math_symbol_table.py | 102 | 129| 259 | 11229 | 38737 | 
| 13 | 6 lib/matplotlib/_mathtext.py | 1681 | 1703| 223 | 11452 | 38737 | 
| 14 | 6 doc/sphinxext/math_symbol_table.py | 1 | 99| 29 | 11481 | 38737 | 
| 15 | 7 lib/matplotlib/mathtext.py | 1 | 30| 208 | 11689 | 39886 | 
| 16 | 7 lib/matplotlib/_mathtext.py | 2126 | 2152| 322 | 12011 | 39886 | 
| 17 | 7 lib/matplotlib/_mathtext.py | 2459 | 2481| 246 | 12257 | 39886 | 
| **-> 18 <-** | **8 lib/matplotlib/_mathtext_data.py** | 1 | 1236| 159 | 12416 | 56541 | 
| 19 | 8 lib/matplotlib/_mathtext.py | 747 | 765| 207 | 12623 | 56541 | 
| 20 | 9 galleries/examples/text_labels_and_annotations/tex_demo.py | 1 | 73| 769 | 13392 | 57585 | 
| 21 | 9 lib/matplotlib/_mathtext.py | 2180 | 2213| 330 | 13722 | 57585 | 
| 22 | 9 lib/matplotlib/_mathtext.py | 396 | 438| 743 | 14465 | 57585 | 
| 23 | 9 lib/matplotlib/_mathtext.py | 807 | 840| 299 | 14764 | 57585 | 
| 24 | 9 lib/matplotlib/_mathtext.py | 440 | 453| 148 | 14912 | 57585 | 
| 25 | 9 lib/matplotlib/_mathtext.py | 2058 | 2079| 359 | 15271 | 57585 | 
| 26 | 9 lib/matplotlib/_mathtext.py | 34 | 57| 176 | 15447 | 57585 | 
| 27 | 9 lib/matplotlib/_mathtext.py | 698 | 745| 393 | 15840 | 57585 | 
| 28 | 10 tools/subset.py | 173 | 242| 755 | 16595 | 62975 | 
| 29 | 10 lib/matplotlib/_mathtext.py | 2483 | 2513| 291 | 16886 | 62975 | 
| 30 | 10 lib/matplotlib/_mathtext.py | 2215 | 2257| 306 | 17192 | 62975 | 
| 31 | 10 lib/matplotlib/_mathtext.py | 608 | 621| 156 | 17348 | 62975 | 
| 32 | 10 lib/matplotlib/_mathtext.py | 1780 | 1790| 175 | 17523 | 62975 | 
| 33 | 11 lib/matplotlib/_text_helpers.py | 1 | 35| 438 | 17961 | 63711 | 
| 34 | 11 lib/matplotlib/_mathtext.py | 1792 | 1819| 230 | 18191 | 63711 | 
| 35 | 12 lib/matplotlib/backends/backend_pdf.py | 454 | 519| 442 | 18633 | 88320 | 
| 36 | 12 lib/matplotlib/_mathtext.py | 210 | 253| 383 | 19016 | 88320 | 
| 37 | 12 lib/matplotlib/_mathtext.py | 2259 | 2352| 849 | 19865 | 88320 | 
| 38 | 12 lib/matplotlib/_mathtext.py | 2154 | 2178| 219 | 20084 | 88320 | 
| 39 | 12 lib/matplotlib/_mathtext.py | 2353 | 2419| 615 | 20699 | 88320 | 
| 40 | 13 doc/conf.py | 517 | 616| 712 | 21411 | 94560 | 
| 41 | 13 lib/matplotlib/_mathtext.py | 843 | 911| 594 | 22005 | 94560 | 
| 42 | 14 lib/matplotlib/sphinxext/mathmpl.py | 1 | 106| 704 | 22709 | 96447 | 
| 43 | 15 galleries/examples/text_labels_and_annotations/mathtext_demo.py | 1 | 27| 219 | 22928 | 96666 | 
| 44 | 16 lib/matplotlib/backends/backend_pgf.py | 62 | 82| 177 | 23105 | 106373 | 
| 45 | 17 lib/matplotlib/cbook.py | 359 | 394| 250 | 23355 | 124891 | 
| 46 | 18 galleries/examples/text_labels_and_annotations/accented_text.py | 1 | 37| 386 | 23741 | 125277 | 
| 47 | 19 galleries/examples/text_labels_and_annotations/unicode_minus.py | 1 | 29| 252 | 23993 | 125529 | 
| 48 | 19 lib/matplotlib/_mathtext.py | 171 | 208| 394 | 24387 | 125529 | 
| 49 | 19 galleries/examples/text_labels_and_annotations/mathtext_examples.py | 102 | 121| 179 | 24566 | 125529 | 
| 50 | 20 lib/matplotlib/markers.py | 1 | 135| 1825 | 26391 | 134887 | 
| 51 | 20 lib/matplotlib/sphinxext/mathmpl.py | 199 | 240| 362 | 26753 | 134887 | 
| 52 | 20 lib/matplotlib/cbook.py | 1259 | 1280| 168 | 26921 | 134887 | 
| 53 | 20 lib/matplotlib/_type1font.py | 655 | 692| 310 | 27231 | 134887 | 
| 54 | 21 galleries/examples/units/basic_units.py | 253 | 278| 186 | 27417 | 137468 | 
| 55 | 22 galleries/users_explain/text/text_props.py | 69 | 273| 1839 | 29256 | 140076 | 
| 56 | 22 lib/matplotlib/_type1font.py | 491 | 592| 853 | 30109 | 140076 | 
| 57 | 22 lib/matplotlib/_mathtext.py | 2515 | 2553| 410 | 30519 | 140076 | 
| 58 | 22 galleries/examples/units/basic_units.py | 1 | 70| 411 | 30930 | 140076 | 
| 59 | 22 lib/matplotlib/_mathtext.py | 2421 | 2457| 347 | 31277 | 140076 | 
| 60 | 23 lib/matplotlib/_afm.py | 38 | 85| 333 | 31610 | 144381 | 
| 61 | 23 lib/matplotlib/sphinxext/mathmpl.py | 109 | 127| 151 | 31761 | 144381 | 
| 62 | 23 lib/matplotlib/_type1font.py | 273 | 318| 266 | 32027 | 144381 | 
| 63 | 23 lib/matplotlib/_mathtext.py | 292 | 320| 267 | 32294 | 144381 | 
| 64 | 23 lib/matplotlib/_mathtext.py | 380 | 394| 182 | 32476 | 144381 | 
| 65 | 24 lib/matplotlib/text.py | 1248 | 1270| 329 | 32805 | 159521 | 
| 66 | 25 lib/matplotlib/testing/jpl_units/__init__.py | 1 | 77| 678 | 33483 | 160199 | 
| 67 | 26 galleries/examples/user_interfaces/mathtext_wx_sgskip.py | 1 | 42| 368 | 33851 | 161264 | 
| 68 | 27 lib/matplotlib/backends/backend_ps.py | 665 | 695| 376 | 34227 | 173253 | 
| 69 | 27 lib/matplotlib/_mathtext.py | 139 | 169| 259 | 34486 | 173253 | 
| 70 | 28 galleries/examples/text_labels_and_annotations/usetex_fonteffects.py | 1 | 30| 214 | 34700 | 173467 | 
| 71 | 29 lib/matplotlib/texmanager.py | 1 | 36| 227 | 34927 | 176980 | 
| 72 | 29 lib/matplotlib/sphinxext/mathmpl.py | 130 | 139| 112 | 35039 | 176980 | 
| 73 | 29 lib/matplotlib/_mathtext.py | 1142 | 1154| 147 | 35186 | 176980 | 
| 74 | 29 lib/matplotlib/mathtext.py | 79 | 103| 222 | 35408 | 176980 | 
| 75 | 29 lib/matplotlib/_mathtext.py | 1042 | 1062| 181 | 35589 | 176980 | 
| 76 | 29 galleries/examples/units/basic_units.py | 192 | 250| 355 | 35944 | 176980 | 
| 77 | 29 lib/matplotlib/texmanager.py | 123 | 162| 396 | 36340 | 176980 | 
| 78 | 29 lib/matplotlib/sphinxext/mathmpl.py | 142 | 178| 360 | 36700 | 176980 | 
| 79 | 30 lib/matplotlib/typing.py | 1 | 64| 538 | 37238 | 177518 | 
| 80 | 30 lib/matplotlib/_mathtext.py | 2041 | 2056| 190 | 37428 | 177518 | 


## Patch

```diff
diff --git a/lib/matplotlib/_mathtext_data.py b/lib/matplotlib/_mathtext_data.py
--- a/lib/matplotlib/_mathtext_data.py
+++ b/lib/matplotlib/_mathtext_data.py
@@ -1008,8 +1008,6 @@
     'leftparen'                : 40,
     'rightparen'               : 41,
     'bigoplus'                 : 10753,
-    'leftangle'                : 10216,
-    'rightangle'               : 10217,
     'leftbrace'                : 124,
     'rightbrace'               : 125,
     'jmath'                    : 567,
@@ -1032,7 +1030,55 @@
     'guilsinglleft'            : 8249,
     'plus'                     : 43,
     'thorn'                    : 254,
-    'dagger'                   : 8224
+    'dagger'                   : 8224,
+    'increment'                : 8710,
+    'smallin'                  : 8714,
+    'notsmallowns'             : 8716,
+    'smallowns'                : 8717,
+    'QED'                      : 8718,
+    'rightangle'               : 8735,
+    'smallintclockwise'        : 8753,
+    'smallvarointclockwise'    : 8754,
+    'smallointctrcclockwise'   : 8755,
+    'ratio'                    : 8758,
+    'minuscolon'               : 8761,
+    'dotsminusdots'            : 8762,
+    'sinewave'                 : 8767,
+    'simneqq'                  : 8774,
+    'nlesssim'                 : 8820,
+    'ngtrsim'                  : 8821,
+    'nlessgtr'                 : 8824,
+    'ngtrless'                 : 8825,
+    'cupleftarrow'             : 8844,
+    'oequal'                   : 8860,
+    'rightassert'              : 8870,
+    'rightModels'              : 8875,
+    'hermitmatrix'             : 8889,
+    'barvee'                   : 8893,
+    'measuredrightangle'       : 8894,
+    'varlrtriangle'            : 8895,
+    'equalparallel'            : 8917,
+    'npreccurlyeq'             : 8928,
+    'nsucccurlyeq'             : 8929,
+    'nsqsubseteq'              : 8930,
+    'nsqsupseteq'              : 8931,
+    'sqsubsetneq'              : 8932,
+    'sqsupsetneq'              : 8933,
+    'disin'                    : 8946,
+    'varisins'                 : 8947,
+    'isins'                    : 8948,
+    'isindot'                  : 8949,
+    'varisinobar'              : 8950,
+    'isinobar'                 : 8951,
+    'isinvb'                   : 8952,
+    'isinE'                    : 8953,
+    'nisd'                     : 8954,
+    'varnis'                   : 8955,
+    'nis'                      : 8956,
+    'varniobar'                : 8957,
+    'niobar'                   : 8958,
+    'bagmember'                : 8959,
+    'triangle'                 : 9651
 }
 
 # Each element is a 4-tuple of the form:

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_mathtext.py b/lib/matplotlib/tests/test_mathtext.py
--- a/lib/matplotlib/tests/test_mathtext.py
+++ b/lib/matplotlib/tests/test_mathtext.py
@@ -510,3 +510,31 @@ def test_mathtext_cmr10_minus_sign():
     ax.plot(range(-1, 1), range(-1, 1))
     # draw to make sure we have no warnings
     fig.canvas.draw()
+
+
+def test_mathtext_operators():
+    test_str = r'''
+    \increment \smallin \notsmallowns
+    \smallowns \QED \rightangle
+    \smallintclockwise \smallvarointclockwise
+    \smallointctrcclockwise
+    \ratio \minuscolon \dotsminusdots
+    \sinewave \simneqq \nlesssim
+    \ngtrsim \nlessgtr \ngtrless
+    \cupleftarrow \oequal \rightassert
+    \rightModels \hermitmatrix \barvee
+    \measuredrightangle \varlrtriangle
+    \equalparallel \npreccurlyeq \nsucccurlyeq
+    \nsqsubseteq \nsqsupseteq \sqsubsetneq
+    \sqsupsetneq  \disin \varisins
+    \isins \isindot \varisinobar
+    \isinobar \isinvb \isinE
+    \nisd \varnis \nis
+    \varniobar \niobar \bagmember
+    \triangle'''.split()
+
+    fig = plt.figure()
+    for x, i in enumerate(test_str):
+        fig.text(0.5, (x + 0.5)/len(test_str), r'${%s}$' % i)
+
+    fig.draw_without_rendering()

```


## Code snippets

### 1 - galleries/users_explain/text/mathtext.py:

Start line: 1, End line: 377

```python
r"""

.. redirect-from:: /tutorials/text/mathtext

.. _mathtext:

Writing mathematical expressions
================================

You can use a subset of TeX markup in any Matplotlib text string by placing it
inside a pair of dollar signs ($).

Note that you do not need to have TeX installed, since Matplotlib ships
its own TeX expression parser, layout engine, and fonts.  The layout engine
is a fairly direct adaptation of the layout algorithms in Donald Knuth's
TeX, so the quality is quite good (Matplotlib also provides a ``usetex``
option for those who do want to call out to TeX to generate their text; see
:ref:`usetex`).

Any text element can use math text.  You should use raw strings (precede the
quotes with an ``'r'``), and surround the math text with dollar signs ($), as
in TeX. Regular text and mathtext can be interleaved within the same string.
Mathtext can use DejaVu Sans (default), DejaVu Serif, the Computer Modern fonts
(from (La)TeX), `STIX <http://www.stixfonts.org/>`_ fonts (which are designed
to blend well with Times), or a Unicode font that you provide.  The mathtext
font can be selected via :rc:`mathtext.fontset` (see
:ref:`customizing`)

Here is a simple example::

    # plain text
    plt.title('alpha > beta')

produces "alpha > beta".

Whereas this::

    # math text
    plt.title(r'$\alpha > \beta$')

produces ":mathmpl:`\alpha > \beta`".

.. note::
   Mathtext should be placed between a pair of dollar signs ($). To make it
   easy to display monetary values, e.g., "$100.00", if a single dollar sign
   is present in the entire string, it will be displayed verbatim as a dollar
   sign.  This is a small change from regular TeX, where the dollar sign in
   non-math text would have to be escaped ('\\\$').

.. note::
   While the syntax inside the pair of dollar signs ($) aims to be TeX-like,
   the text outside does not.  In particular, characters such as::

     # $ % & ~ _ ^ \ { } \( \) \[ \]

   have special meaning outside of math mode in TeX.  Therefore, these
   characters will behave differently depending on :rc:`text.usetex`.  See the
   :ref:`usetex tutorial <usetex>` for more information.

.. note::
   To generate html output in documentation that will exactly match the output
   generated by ``mathtext``, use the `matplotlib.sphinxext.mathmpl` Sphinx
   extension.

Subscripts and superscripts
---------------------------
To make subscripts and superscripts, use the ``'_'`` and ``'^'`` symbols::

    r'$\alpha_i > \beta_i$'

.. math::

    \alpha_i > \beta_i

To display multi-letter subscripts or superscripts correctly,
you should put them in curly braces ``{...}``::

    r'$\alpha^{ic} > \beta_{ic}$'

.. math::

    \alpha^{ic} > \beta_{ic}

Some symbols automatically put their sub/superscripts under and over the
operator.  For example, to write the sum of :mathmpl:`x_i` from :mathmpl:`0` to
:mathmpl:`\infty`, you could do::

    r'$\sum_{i=0}^\infty x_i$'

.. math::

    \sum_{i=0}^\infty x_i

Fractions, binomials, and stacked numbers
-----------------------------------------
Fractions, binomials, and stacked numbers can be created with the
``\frac{}{}``, ``\binom{}{}`` and ``\genfrac{}{}{}{}{}{}`` commands,
respectively::

    r'$\frac{3}{4} \binom{3}{4} \genfrac{}{}{0}{}{3}{4}$'

produces

.. math::

    \frac{3}{4} \binom{3}{4} \genfrac{}{}{0pt}{}{3}{4}

Fractions can be arbitrarily nested::

    r'$\frac{5 - \frac{1}{x}}{4}$'

produces

.. math::

    \frac{5 - \frac{1}{x}}{4}

Note that special care needs to be taken to place parentheses and brackets
around fractions.  Doing things the obvious way produces brackets that are too
small::

    r'$(\frac{5 - \frac{1}{x}}{4})$'

.. math::

    (\frac{5 - \frac{1}{x}}{4})

The solution is to precede the bracket with ``\left`` and ``\right`` to inform
the parser that those brackets encompass the entire object.::

    r'$\left(\frac{5 - \frac{1}{x}}{4}\right)$'

.. math::

    \left(\frac{5 - \frac{1}{x}}{4}\right)

Radicals
--------
Radicals can be produced with the ``\sqrt[]{}`` command.  For example::

    r'$\sqrt{2}$'

.. math::

    \sqrt{2}

Any base can (optionally) be provided inside square brackets.  Note that the
base must be a simple expression, and cannot contain layout commands such as
fractions or sub/superscripts::

    r'$\sqrt[3]{x}$'

.. math::

    \sqrt[3]{x}

.. _mathtext-fonts:

Fonts
-----
The default font is *italics* for mathematical symbols.

.. note::

   This default can be changed using :rc:`mathtext.default`.  This is
   useful, for example, to use the same font as regular non-math text for math
   text, by setting it to ``regular``.

To change fonts, e.g., to write "sin" in a Roman font, enclose the text in a
font command::

    r'$s(t) = \mathcal{A}\mathrm{sin}(2 \omega t)$'

.. math::

    s(t) = \mathcal{A}\mathrm{sin}(2 \omega t)

More conveniently, many commonly used function names that are typeset in
a Roman font have shortcuts.  So the expression above could be written as
follows::

    r'$s(t) = \mathcal{A}\sin(2 \omega t)$'

.. math::

    s(t) = \mathcal{A}\sin(2 \omega t)

Here "s" and "t" are variable in italics font (default), "sin" is in Roman
font, and the amplitude "A" is in calligraphy font.  Note in the example above
the calligraphy ``A`` is squished into the ``sin``.  You can use a spacing
command to add a little whitespace between them::

    r's(t) = \mathcal{A}\/\sin(2 \omega t)'

.. Here we cheat a bit: for HTML math rendering, Sphinx relies on MathJax which
   doesn't actually support the italic correction (\/); instead, use a thin
   space (\,) which is supported.

.. math::

    s(t) = \mathcal{A}\,\sin(2 \omega t)

The choices available with all fonts are:

    ========================= ================================
    Command                   Result
    ========================= ================================
    ``\mathrm{Roman}``        :mathmpl:`\mathrm{Roman}`
    ``\mathit{Italic}``       :mathmpl:`\mathit{Italic}`
    ``\mathtt{Typewriter}``   :mathmpl:`\mathtt{Typewriter}`
    ``\mathcal{CALLIGRAPHY}`` :mathmpl:`\mathcal{CALLIGRAPHY}`
    ========================= ================================

.. role:: math-stix(mathmpl)
   :fontset: stix

When using the `STIX <http://www.stixfonts.org/>`_ fonts, you also have the
choice of:

    ================================ =========================================
    Command                          Result
    ================================ =========================================
    ``\mathbb{blackboard}``          :math-stix:`\mathbb{blackboard}`
    ``\mathrm{\mathbb{blackboard}}`` :math-stix:`\mathrm{\mathbb{blackboard}}`
    ``\mathfrak{Fraktur}``           :math-stix:`\mathfrak{Fraktur}`
    ``\mathsf{sansserif}``           :math-stix:`\mathsf{sansserif}`
    ``\mathrm{\mathsf{sansserif}}``  :math-stix:`\mathrm{\mathsf{sansserif}}`
    ``\mathbfit{bolditalic}``        :math-stix:`\mathbfit{bolditalic}`
    ================================ =========================================

There are also five global "font sets" to choose from, which are
selected using the ``mathtext.fontset`` parameter in :ref:`matplotlibrc
<matplotlibrc-sample>`.

``dejavusans``: DejaVu Sans

    .. mathmpl::
       :fontset: dejavusans

       \mathcal{R} \prod_{i=\alpha}^{\infty} a_i \sin\left(2\pi fx_i\right)

``dejavuserif``: DejaVu Serif

    .. mathmpl::
       :fontset: dejavuserif

       \mathcal{R} \prod_{i=\alpha}^{\infty} a_i \sin\left(2\pi fx_i\right)

``cm``: Computer Modern (TeX)

    .. mathmpl::
       :fontset: cm

       \mathcal{R} \prod_{i=\alpha}^{\infty} a_i \sin\left(2\pi fx_i\right)

``stix``: STIX (designed to blend well with Times)

    .. mathmpl::
       :fontset: stix

       \mathcal{R} \prod_{i=\alpha}^{\infty} a_i \sin\left(2\pi fx_i\right)

``stixsans``: STIX sans-serif

    .. mathmpl::
       :fontset: stixsans

       \mathcal{R} \prod_{i=\alpha}^{\infty} a_i \sin\left(2\pi fx_i\right)

Additionally, you can use ``\mathdefault{...}`` or its alias
``\mathregular{...}`` to use the font used for regular text outside of
mathtext.  There are a number of limitations to this approach, most notably
that far fewer symbols will be available, but it can be useful to make math
expressions blend well with other text in the plot.

For compatibility with popular packages, ``\text{...}`` is available and uses the
``\mathrm{...}`` font, but otherwise retains spaces and renders - as a dash
(not minus).

Custom fonts
~~~~~~~~~~~~
mathtext also provides a way to use custom fonts for math.  This method is
fairly tricky to use, and should be considered an experimental feature for
patient users only.  By setting :rc:`mathtext.fontset` to ``custom``,
you can then set the following parameters, which control which font file to use
for a particular set of math characters.

    ============================== =================================
    Parameter                      Corresponds to
    ============================== =================================
    ``mathtext.it``                ``\mathit{}`` or default italic
    ``mathtext.rm``                ``\mathrm{}`` Roman (upright)
    ``mathtext.tt``                ``\mathtt{}`` Typewriter (monospace)
    ``mathtext.bf``                ``\mathbf{}`` bold
    ``mathtext.bfit``              ``\mathbfit{}`` bold italic
    ``mathtext.cal``               ``\mathcal{}`` calligraphic
    ``mathtext.sf``                ``\mathsf{}`` sans-serif
    ============================== =================================

Each parameter should be set to a fontconfig font descriptor (as defined in the
yet-to-be-written font chapter).

.. TODO: Link to font chapter

The fonts used should have a Unicode mapping in order to find any
non-Latin characters, such as Greek.  If you want to use a math symbol
that is not contained in your custom fonts, you can set
:rc:`mathtext.fallback` to either ``'cm'``, ``'stix'`` or ``'stixsans'``
which will cause the mathtext system to use
characters from an alternative font whenever a particular
character cannot be found in the custom font.

Note that the math glyphs specified in Unicode have evolved over time, and many
fonts may not have glyphs in the correct place for mathtext.

Accents
-------
An accent command may precede any symbol to add an accent above it.  There are
long and short forms for some of them.

    ============================== =================================
    Command                        Result
    ============================== =================================
    ``\acute a`` or ``\'a``        :mathmpl:`\acute a`
    ``\bar a``                     :mathmpl:`\bar a`
    ``\breve a``                   :mathmpl:`\breve a`
    ``\dot a`` or ``\.a``          :mathmpl:`\dot a`
    ``\ddot a`` or ``\''a``        :mathmpl:`\ddot a`
    ``\dddot a``                   :mathmpl:`\dddot a`
    ``\ddddot a``                  :mathmpl:`\ddddot a`
    ``\grave a`` or ``\`a``        :mathmpl:`\grave a`
    ``\hat a`` or ``\^a``          :mathmpl:`\hat a`
    ``\tilde a`` or ``\~a``        :mathmpl:`\tilde a`
    ``\vec a``                     :mathmpl:`\vec a`
    ``\overline{abc}``             :mathmpl:`\overline{abc}`
    ============================== =================================

In addition, there are two special accents that automatically adjust to the
width of the symbols below:

    ============================== =================================
    Command                        Result
    ============================== =================================
    ``\widehat{xyz}``              :mathmpl:`\widehat{xyz}`
    ``\widetilde{xyz}``            :mathmpl:`\widetilde{xyz}`
    ============================== =================================

Care should be taken when putting accents on lower-case i's and j's.  Note that
in the following ``\imath`` is used to avoid the extra dot over the i::

    r"$\hat i\ \ \hat \imath$"

.. math::

    \hat i\ \ \hat \imath

Symbols
-------
You can also use a large number of the TeX symbols, as in ``\infty``,
``\leftarrow``, ``\sum``, ``\int``.

.. math_symbol_table::

If a particular symbol does not have a name (as is true of many of the more
obscure symbols in the STIX fonts), Unicode characters can also be used::

   r'$\u23ce$'

Example
-------
Here is an example illustrating many of these features in context.

.. figure:: /gallery/text_labels_and_annotations/images/sphx_glr_mathtext_demo_001.png
   :target: /gallery/text_labels_and_annotations/mathtext_demo.html
   :align: center
"""
```
### 2 - lib/matplotlib/_type1font.py:

Start line: 777, End line: 878

```python
_StandardEncoding = {
    **{ord(letter): letter for letter in string.ascii_letters},
    0: '.notdef',
    32: 'space',
    33: 'exclam',
    34: 'quotedbl',
    35: 'numbersign',
    36: 'dollar',
    37: 'percent',
    38: 'ampersand',
    39: 'quoteright',
    40: 'parenleft',
    41: 'parenright',
    42: 'asterisk',
    43: 'plus',
    44: 'comma',
    45: 'hyphen',
    46: 'period',
    47: 'slash',
    48: 'zero',
    49: 'one',
    50: 'two',
    51: 'three',
    52: 'four',
    53: 'five',
    54: 'six',
    55: 'seven',
    56: 'eight',
    57: 'nine',
    58: 'colon',
    59: 'semicolon',
    60: 'less',
    61: 'equal',
    62: 'greater',
    63: 'question',
    64: 'at',
    91: 'bracketleft',
    92: 'backslash',
    93: 'bracketright',
    94: 'asciicircum',
    95: 'underscore',
    96: 'quoteleft',
    123: 'braceleft',
    124: 'bar',
    125: 'braceright',
    126: 'asciitilde',
    161: 'exclamdown',
    162: 'cent',
    163: 'sterling',
    164: 'fraction',
    165: 'yen',
    166: 'florin',
    167: 'section',
    168: 'currency',
    169: 'quotesingle',
    170: 'quotedblleft',
    171: 'guillemotleft',
    172: 'guilsinglleft',
    173: 'guilsinglright',
    174: 'fi',
    175: 'fl',
    177: 'endash',
    178: 'dagger',
    179: 'daggerdbl',
    180: 'periodcentered',
    182: 'paragraph',
    183: 'bullet',
    184: 'quotesinglbase',
    185: 'quotedblbase',
    186: 'quotedblright',
    187: 'guillemotright',
    188: 'ellipsis',
    189: 'perthousand',
    191: 'questiondown',
    193: 'grave',
    194: 'acute',
    195: 'circumflex',
    196: 'tilde',
    197: 'macron',
    198: 'breve',
    199: 'dotaccent',
    200: 'dieresis',
    202: 'ring',
    203: 'cedilla',
    205: 'hungarumlaut',
    206: 'ogonek',
    207: 'caron',
    208: 'emdash',
    225: 'AE',
    227: 'ordfeminine',
    232: 'Lslash',
    233: 'Oslash',
    234: 'OE',
    235: 'ordmasculine',
    241: 'ae',
    245: 'dotlessi',
    248: 'lslash',
    249: 'oslash',
    250: 'oe',
    251: 'germandbls',
}
```
### 3 - lib/matplotlib/_mathtext.py:

Start line: 515, End line: 581

```python
class UnicodeFonts(TruetypeFonts):

    def _get_glyph(self, fontname, font_class, sym):
        try:
            uniindex = get_unicode_index(sym)
            found_symbol = True
        except ValueError:
            uniindex = ord('?')
            found_symbol = False
            _log.warning("No TeX to Unicode mapping for %a.", sym)

        fontname, uniindex = self._map_virtual_font(
            fontname, font_class, uniindex)

        new_fontname = fontname

        # Only characters in the "Letter" class should be italicized in 'it'
        # mode.  Greek capital letters should be Roman.
        if found_symbol:
            if fontname == 'it' and uniindex < 0x10000:
                char = chr(uniindex)
                if (unicodedata.category(char)[0] != "L"
                        or unicodedata.name(char).startswith("GREEK CAPITAL")):
                    new_fontname = 'rm'

            slanted = (new_fontname == 'it') or sym in self._slanted_symbols
            found_symbol = False
            font = self._get_font(new_fontname)
            if font is not None:
                if (uniindex in self._cmr10_substitutions
                        and font.family_name == "cmr10"):
                    font = get_font(
                        cbook._get_data_path("fonts/ttf/cmsy10.ttf"))
                    uniindex = self._cmr10_substitutions[uniindex]
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    found_symbol = True

        if not found_symbol:
            if self._fallback_font:
                if (fontname in ('it', 'regular')
                        and isinstance(self._fallback_font, StixFonts)):
                    fontname = 'rm'

                g = self._fallback_font._get_glyph(fontname, font_class, sym)
                family = g[0].family_name
                if family in list(BakomaFonts._fontmap.values()):
                    family = "Computer Modern"
                _log.info("Substituting symbol %s from %s", sym, family)
                return g

            else:
                if (fontname in ('it', 'regular')
                        and isinstance(self, StixFonts)):
                    return self._get_glyph('rm', font_class, sym)
                _log.warning("Font %r does not have a glyph for %a [U+%x], "
                             "substituting with a dummy symbol.",
                             new_fontname, sym, uniindex)
                font = self._get_font('rm')
                uniindex = 0xA4  # currency char, for lack of anything better
                slanted = False

        return font, uniindex, slanted

    def get_sized_alternatives_for_symbol(self, fontname, sym):
        if self._fallback_font:
            return self._fallback_font.get_sized_alternatives_for_symbol(
                fontname, sym)
        return [(fontname, sym)]
```
### 4 - lib/matplotlib/_mathtext.py:

Start line: 1706, End line: 1778

```python
class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    class _MathStyle(enum.Enum):
        DISPLAYSTYLE = 0
        TEXTSTYLE = 1
        SCRIPTSTYLE = 2
        SCRIPTSCRIPTSTYLE = 3

    _binary_operators = set(
      '+ * - \N{MINUS SIGN}'
      r'''
      \pm             \sqcap                   \rhd
      \mp             \sqcup                   \unlhd
      \times          \vee                     \unrhd
      \div            \wedge                   \oplus
      \ast            \setminus                \ominus
      \star           \wr                      \otimes
      \circ           \diamond                 \oslash
      \bullet         \bigtriangleup           \odot
      \cdot           \bigtriangledown         \bigcirc
      \cap            \triangleleft            \dagger
      \cup            \triangleright           \ddagger
      \uplus          \lhd                     \amalg
      \dotplus        \dotminus'''.split())

    _relation_symbols = set(r'''
      = < > :
      \leq        \geq        \equiv   \models
      \prec       \succ       \sim     \perp
      \preceq     \succeq     \simeq   \mid
      \ll         \gg         \asymp   \parallel
      \subset     \supset     \approx  \bowtie
      \subseteq   \supseteq   \cong    \Join
      \sqsubset   \sqsupset   \neq     \smile
      \sqsubseteq \sqsupseteq \doteq   \frown
      \in         \ni         \propto  \vdash
      \dashv      \dots       \doteqdot'''.split())

    _arrow_symbols = set(r'''
      \leftarrow              \longleftarrow           \uparrow
      \Leftarrow              \Longleftarrow           \Uparrow
      \rightarrow             \longrightarrow          \downarrow
      \Rightarrow             \Longrightarrow          \Downarrow
      \leftrightarrow         \longleftrightarrow      \updownarrow
      \Leftrightarrow         \Longleftrightarrow      \Updownarrow
      \mapsto                 \longmapsto              \nearrow
      \hookleftarrow          \hookrightarrow          \searrow
      \leftharpoonup          \rightharpoonup          \swarrow
      \leftharpoondown        \rightharpoondown        \nwarrow
      \rightleftharpoons      \leadsto'''.split())

    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols

    _punctuation_symbols = set(r', ; . ! \ldotp \cdotp'.split())

    _overunder_symbols = set(r'''
       \sum \prod \coprod \bigcap \bigcup \bigsqcup \bigvee
       \bigwedge \bigodot \bigotimes \bigoplus \biguplus
       '''.split())

    _overunder_functions = set("lim liminf limsup sup max min".split())

    _dropsub_symbols = set(r'''\int \oint'''.split())

    _fontnames = set("rm cal it tt sf bf bfit "
                     "default bb frak scr regular".split())
```
### 5 - lib/matplotlib/_mathtext.py:

Start line: 2081, End line: 2124

```python
class Parser:

    def symbol(self, s, loc, toks):
        c = toks["sym"]
        if c == "-":
            # "U+2212 minus sign is the preferred representation of the unary
            # and binary minus sign rather than the ASCII-derived U+002D
            # hyphen-minus, because minus sign is unambiguous and because it
            # is rendered with a more desirable length, usually longer than a
            # hyphen." (https://www.unicode.org/reports/tr25/)
            c = "\N{MINUS SIGN}"
        try:
            char = Char(c, self.get_state())
        except ValueError as err:
            raise ParseFatalException(s, loc,
                                      "Unknown symbol: %s" % c) from err

        if c in self._spaced_symbols:
            # iterate until we find previous character, needed for cases
            # such as ${ -2}$, $ -2$, or $   -2$.
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            # Binary operators at start of string should not be spaced
            if (c in self._binary_operators and
                    (len(s[:loc].split()) == 0 or prev_char == '{' or
                     prev_char in self._left_delims)):
                return [char]
            else:
                return [Hlist([self._make_space(0.2),
                               char,
                               self._make_space(0.2)],
                              do_kern=True)]
        elif c in self._punctuation_symbols:
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            next_char = next((c for c in s[loc + 1:] if c != ' '), '')

            # Do not space commas between brackets
            if c == ',':
                if prev_char == '{' and next_char == '}':
                    return [char]

            # Do not space dots as decimal separators
            if c == '.' and prev_char.isdigit() and next_char.isdigit():
                return [char]
            else:
                return [Hlist([char, self._make_space(0.2)], do_kern=True)]
        return [char]
```
### 6 - lib/matplotlib/_mathtext.py:

Start line: 1821, End line: 1975

```python
class Parser:

    def __init__(self):
        # ... other code

        p.float_literal  = Regex(r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)")
        p.space          = oneOf(self._space_widths)("space")

        p.style_literal  = oneOf(
            [str(e.value) for e in self._MathStyle])("style_literal")

        p.symbol         = Regex(
            r"[a-zA-Z0-9 +\-*/<>=:,.;!\?&'@()\[\]|\U00000080-\U0001ffff]"
            r"|\\[%${}\[\]_|]"
            + r"|\\(?:{})(?![A-Za-z])".format(
                "|".join(map(re.escape, tex2uni)))
        )("sym").leaveWhitespace()
        p.unknown_symbol = Regex(r"\\[A-Za-z]*")("name")

        p.font           = csnames("font", self._fontnames)
        p.start_group    = (
            Optional(r"\math" + oneOf(self._fontnames)("font")) + "{")
        p.end_group      = Literal("}")

        p.delim          = oneOf(self._delims)

        set_names_and_parse_actions()  # for root definitions.

        # Mutually recursive definitions.  (Minimizing the number of Forward
        # elements is important for speed.)
        p.accent           = Forward()
        p.auto_delim       = Forward()
        p.binom            = Forward()
        p.customspace      = Forward()
        p.frac             = Forward()
        p.dfrac            = Forward()
        p.function         = Forward()
        p.genfrac          = Forward()
        p.group            = Forward()
        p.operatorname     = Forward()
        p.overline         = Forward()
        p.overset          = Forward()
        p.placeable        = Forward()
        p.required_group   = Forward()
        p.simple           = Forward()
        p.optional_group   = Forward()
        p.sqrt             = Forward()
        p.subsuper         = Forward()
        p.text             = Forward()
        p.token            = Forward()
        p.underset         = Forward()

        set_names_and_parse_actions()  # for mutually recursive definitions.

        p.customspace <<= cmd(r"\hspace", "{" + p.float_literal("space") + "}")

        p.accent <<= (
            csnames("accent", [*self._accent_map, *self._wide_accents])
            - p.placeable("sym"))

        p.function <<= csnames("name", self._function_names)
        p.operatorname <<= cmd(
            r"\operatorname", "{" + ZeroOrMore(p.simple)("name") + "}")

        p.group <<= p.start_group + ZeroOrMore(p.token)("group") + p.end_group

        p.optional_group <<= "{" + ZeroOrMore(p.token)("group") + "}"
        p.required_group <<= "{" + OneOrMore(p.token)("group") + "}"

        p.frac  <<= cmd(
            r"\frac", p.required_group("num") + p.required_group("den"))
        p.dfrac <<= cmd(
            r"\dfrac", p.required_group("num") + p.required_group("den"))
        p.binom <<= cmd(
            r"\binom", p.required_group("num") + p.required_group("den"))

        p.genfrac <<= cmd(
            r"\genfrac",
            "{" + Optional(p.delim)("ldelim") + "}"
            + "{" + Optional(p.delim)("rdelim") + "}"
            + "{" + p.float_literal("rulesize") + "}"
            + "{" + Optional(p.style_literal)("style") + "}"
            + p.required_group("num")
            + p.required_group("den"))

        p.sqrt <<= cmd(
            r"\sqrt{value}",
            Optional("[" + OneOrMore(NotAny("]") + p.token)("root") + "]")
            + p.required_group("value"))

        p.overline <<= cmd(r"\overline", p.required_group("body"))

        p.overset  <<= cmd(
            r"\overset",
            p.optional_group("annotation") + p.optional_group("body"))
        p.underset <<= cmd(
            r"\underset",
            p.optional_group("annotation") + p.optional_group("body"))

        p.text <<= cmd(r"\text", QuotedString('{', '\\', endQuoteChar="}"))

        p.placeable     <<= (
            p.accent     # Must be before symbol as all accents are symbols
            | p.symbol   # Must be second to catch all named symbols and single
                         # chars not in a group
            | p.function
            | p.operatorname
            | p.group
            | p.frac
            | p.dfrac
            | p.binom
            | p.genfrac
            | p.overset
            | p.underset
            | p.sqrt
            | p.overline
            | p.text
        )

        p.simple        <<= (
            p.space
            | p.customspace
            | p.font
            | p.subsuper
        )

        p.subsuper      <<= (
            (Optional(p.placeable)("nucleus")
             + OneOrMore(oneOf(["_", "^"]) - p.placeable)("subsuper")
             + Regex("'*")("apostrophes"))
            | Regex("'+")("apostrophes")
            | (p.placeable("nucleus") + Regex("'*")("apostrophes"))
        )

        p.token         <<= (
            p.simple
            | p.auto_delim
            | p.unknown_symbol  # Must be last
        )

        p.auto_delim    <<= (
            r"\left" - (p.delim("left") | Error("Expected a delimiter"))
            + ZeroOrMore(p.simple | p.auto_delim)("mid")
            + r"\right" - (p.delim("right") | Error("Expected a delimiter"))
        )

        # Leaf definitions.
        p.math          = OneOrMore(p.token)
        p.math_string   = QuotedString('$', '\\', unquoteResults=False)
        p.non_math      = Regex(r"(?:(?:\\[$])|[^$])*").leaveWhitespace()
        p.main          = (
            p.non_math + ZeroOrMore(p.math_string + p.non_math) + StringEnd()
        )
        set_names_and_parse_actions()  # for leaf definitions.

        self._expression = p.main
        self._math_expression = p.math

        # To add space to nucleus operators after sub/superscripts
        self._in_subscript_or_superscript = False
```
### 7 - lib/matplotlib/_mathtext.py:

Start line: 456, End line: 513

```python
class UnicodeFonts(TruetypeFonts):
    """
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I'm aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol cannot be found in the font.
    """

    # Some glyphs are not present in the `cmr10` font, and must be brought in
    # from `cmsy10`. Map the Unicode indices of those glyphs to the indices at
    # which they are found in `cmsy10`.
    _cmr10_substitutions = {
        0x00D7: 0x00A3,  # Multiplication sign.
        0x2212: 0x00A1,  # Minus sign.
    }

    def __init__(self, *args, **kwargs):
        # This must come first so the backend's owner is set correctly
        fallback_rc = mpl.rcParams['mathtext.fallback']
        font_cls = {'stix': StixFonts,
                    'stixsans': StixSansFonts,
                    'cm': BakomaFonts
                    }.get(fallback_rc)
        self._fallback_font = font_cls(*args, **kwargs) if font_cls else None

        super().__init__(*args, **kwargs)
        self.fontmap = {}
        for texfont in "cal rm tt it bf sf bfit".split():
            prop = mpl.rcParams['mathtext.' + texfont]
            font = findfont(prop)
            self.fontmap[texfont] = font
        prop = FontProperties('cmex10')
        font = findfont(prop)
        self.fontmap['ex'] = font

        # include STIX sized alternatives for glyphs if fallback is STIX
        if isinstance(self._fallback_font, StixFonts):
            stixsizedaltfonts = {
                 0: 'STIXGeneral',
                 1: 'STIXSizeOneSym',
                 2: 'STIXSizeTwoSym',
                 3: 'STIXSizeThreeSym',
                 4: 'STIXSizeFourSym',
                 5: 'STIXSizeFiveSym'}

            for size, name in stixsizedaltfonts.items():
                fullpath = findfont(name)
                self.fontmap[size] = fullpath
                self.fontmap[name] = fullpath

    _slanted_symbols = set(r"\int \oint".split())

    def _map_virtual_font(self, fontname, font_class, uniindex):
        return fontname, uniindex
```
### 8 - galleries/users_explain/text/usetex.py:

Start line: 1, End line: 178

```python
r"""
.. redirect-from:: /tutorials/text/usetex

.. _usetex:

*************************
Text rendering with LaTeX
*************************

Matplotlib can use LaTeX to render text.  This is activated by setting
``text.usetex : True`` in your rcParams, or by setting the ``usetex`` property
to True on individual `.Text` objects.  Text handling through LaTeX is slower
than Matplotlib's very capable :ref:`mathtext <mathtext>`, but
is more flexible, since different LaTeX packages (font packages, math packages,
etc.) can be used. The results can be striking, especially when you take care
to use the same fonts in your figures as in the main document.

Matplotlib's LaTeX support requires a working LaTeX_ installation.  For
the \*Agg backends, dvipng_ is additionally required; for the PS backend,
PSfrag_, dvips_ and Ghostscript_ are additionally required.  For the PDF
and SVG backends, if LuaTeX is present, it will be used to speed up some
post-processing steps, but note that it is not used to parse the TeX string
itself (only LaTeX is supported).  The executables for these external
dependencies must all be located on your :envvar:`PATH`.

Only a small number of font families (defined by the PSNFSS_ scheme) are
supported.  They are listed here, with the corresponding LaTeX font selection
commands and LaTeX packages, which are automatically used.

=========================== =================================================
generic family              fonts
=========================== =================================================
serif (``\rmfamily``)       Computer Modern Roman, Palatino (``mathpazo``),
                            Times (``mathptmx``),  Bookman (``bookman``),
                            New Century Schoolbook (``newcent``),
                            Charter (``charter``)

sans-serif (``\sffamily``)  Computer Modern Serif, Helvetica (``helvet``),
                            Avant Garde (``avant``)

cursive (``\rmfamily``)     Zapf Chancery (``chancery``)

monospace (``\ttfamily``)   Computer Modern Typewriter, Courier (``courier``)
=========================== =================================================

The default font family (which does not require loading any LaTeX package) is
Computer Modern.  All other families are Adobe fonts.  Times and Palatino each
have their own accompanying math fonts, while the other Adobe serif fonts make
use of the Computer Modern math fonts.

To enable LaTeX and select a font, use e.g.::

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

or equivalently, set your :ref:`matplotlibrc <customizing>` to::

    text.usetex : true
    font.family : Helvetica

It is also possible to instead set ``font.family`` to one of the generic family
names and then configure the corresponding generic family; e.g.::

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

(this was the required approach until Matplotlib 3.5).

Here is the standard example,
:doc:`/gallery/text_labels_and_annotations/tex_demo`:

.. figure:: /gallery/text_labels_and_annotations/images/sphx_glr_tex_demo_001.png
   :target: /gallery/text_labels_and_annotations/tex_demo.html
   :align: center

Note that display math mode (``$$ e=mc^2 $$``) is not supported, but adding the
command ``\displaystyle``, as in the above demo, will produce the same results.

Non-ASCII characters (e.g. the degree sign in the y-label above) are supported
to the extent that they are supported by inputenc_.

.. note::
   For consistency with the non-usetex case, Matplotlib special-cases newlines,
   so that single-newlines yield linebreaks (rather than being interpreted as
   whitespace in standard LaTeX).

   Matplotlib uses the underscore_ package so that underscores (``_``) are
   printed "as-is" in text mode (rather than causing an error as in standard
   LaTeX).  Underscores still introduce subscripts in math mode.

.. note::
   Certain characters require special escaping in TeX, such as::

     # $ % & ~ ^ \ { } \( \) \[ \]

   Therefore, these characters will behave differently depending on
   :rc:`text.usetex`.  As noted above, underscores (``_``) do not require
   escaping outside of math mode.

PostScript options
==================

In order to produce encapsulated PostScript (EPS) files that can be embedded
in a new LaTeX document, the default behavior of Matplotlib is to distill the
output, which removes some PostScript operators used by LaTeX that are illegal
in an EPS file. This step produces results which may be unacceptable to some
users, because the text is coarsely rasterized and converted to bitmaps, which
are not scalable like standard PostScript, and the text is not searchable. One
workaround is to set :rc:`ps.distiller.res` to a higher value (perhaps 6000)
in your rc settings, which will produce larger files but may look better and
scale reasonably. A better workaround, which requires Poppler_ or Xpdf_, can
be activated by changing :rc:`ps.usedistiller` to ``xpdf``. This alternative
produces PostScript without rasterizing text, so it scales properly, can be
edited in Adobe Illustrator, and searched text in pdf documents.

.. _usetex-hangups:

Possible hangups
================

* On Windows, the :envvar:`PATH` environment variable may need to be modified
  to include the directories containing the latex, dvipng and ghostscript
  executables. See :ref:`environment-variables` and
  :ref:`setting-windows-environment-variables` for details.

* Using MiKTeX with Computer Modern fonts, if you get odd \*Agg and PNG
  results, go to MiKTeX/Options and update your format files

* On Ubuntu and Gentoo, the base texlive install does not ship with
  the type1cm package. You may need to install some of the extra
  packages to get all the goodies that come bundled with other LaTeX
  distributions.

* Some progress has been made so Matplotlib uses the dvi files
  directly for text layout. This allows LaTeX to be used for text
  layout with the pdf and svg backends, as well as the \*Agg and PS
  backends. In the future, a LaTeX installation may be the only
  external dependency.

.. _usetex-troubleshooting:

Troubleshooting
===============

* Try deleting your :file:`.matplotlib/tex.cache` directory. If you don't know
  where to find :file:`.matplotlib`, see :ref:`locating-matplotlib-config-dir`.

* Make sure LaTeX, dvipng and ghostscript are each working and on your
  :envvar:`PATH`.

* Make sure what you are trying to do is possible in a LaTeX document,
  that your LaTeX syntax is valid and that you are using raw strings
  if necessary to avoid unintended escape sequences.

* :rc:`text.latex.preamble` is not officially supported. This
  option provides lots of flexibility, and lots of ways to cause
  problems. Please disable this option before reporting problems to
  the mailing list.

* If you still need help, please see :ref:`reporting-problems`.

.. _dvipng: http://www.nongnu.org/dvipng/
.. _dvips: https://tug.org/texinfohtml/dvips.html
.. _Ghostscript: https://ghostscript.com/
.. _inputenc: https://ctan.org/pkg/inputenc
.. _LaTeX: http://www.tug.org
.. _Poppler: https://poppler.freedesktop.org/
.. _PSNFSS: http://www.ctan.org/tex-archive/macros/latex/required/psnfss/psnfss2e.pdf
.. _PSfrag: https://ctan.org/pkg/psfrag
.. _underscore: https://ctan.org/pkg/underscore
.. _Xpdf: http://www.xpdfreader.com/
"""
```
### 9 - doc/sphinxext/math_symbol_table.py:

Start line: 132, End line: 169

```python
class MathSymbolTableDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        return run(self.state_machine)


def setup(app):
    app.add_directive("math_symbol_table", MathSymbolTableDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata


if __name__ == "__main__":
    # Do some verification of the tables

    print("SYMBOLS NOT IN STIX:")
    all_symbols = {}
    for category, columns, syms in symbols:
        if category == "Standard Function Names":
            continue
        syms = syms.split()
        for sym in syms:
            if len(sym) > 1:
                all_symbols[sym[1:]] = None
                if sym[1:] not in _mathtext_data.tex2uni:
                    print(sym)

    print("SYMBOLS NOT IN TABLE:")
    for sym in _mathtext_data.tex2uni:
        if sym not in all_symbols:
            print(sym)
```
### 10 - lib/matplotlib/_mathtext.py:

Start line: 1, End line: 31

```python
"""
Implementation details for :mod:`.mathtext`.
"""

import copy
from collections import namedtuple
import enum
import functools
import logging
import os
import re
import types
import unicodedata

import numpy as np
from pyparsing import (
    Empty, Forward, Literal, NotAny, oneOf, OneOrMore, Optional,
    ParseBaseException, ParseException, ParseExpression, ParseFatalException,
    ParserElement, ParseResults, QuotedString, Regex, StringEnd, ZeroOrMore,
    pyparsing_common)

import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
    latex_to_bakoma, stix_glyph_fixes, stix_virtual_fonts, tex2uni)
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Image, KERNING_DEFAULT


ParserElement.enablePackrat()
_log = logging.getLogger("matplotlib.mathtext")
```
### 18 - lib/matplotlib/_mathtext_data.py:

Start line: 1, End line: 1236

```python
"""
font data tables for truetype and afm computer modern fonts
"""

latex_to_bakoma =
 # ... other code

# Automatically generated.

type12uni =
 # ... other code

uni2type1 = {v: k for k, v in type12uni.items()}

tex2uni =
 # ... other code

# Each element is a 4-tuple of the form:
#   src_start, src_end, dst_font, dst_start
#
stix_virtual_fonts =
 # ... other code


# Fix some incorrect glyphs.
stix_glyph_fixes = {
    # Cap and Cup glyphs are swapped.
    0x22d2: 0x22d3,
    0x22d3: 0x22d2,
}
```
