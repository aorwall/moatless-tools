# sympy__sympy-15308

| **sympy/sympy** | `fb59d703e6863ed803c98177b59197b5513332e9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 42790 |
| **Any found context length** | 742 |
| **Avg pos** | 127.0 |
| **Min pos** | 3 |
| **Max pos** | 124 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -289,6 +289,10 @@ def _do_exponent(self, expr, exp):
         else:
             return expr
 
+    def _print_Basic(self, expr):
+        l = [self._print(o) for o in expr.args]
+        return self._deal_with_super_sub(expr.__class__.__name__) + r"\left(%s\right)" % ", ".join(l)
+
     def _print_bool(self, e):
         return r"\mathrm{%s}" % e
 
@@ -1462,6 +1466,10 @@ def _print_Transpose(self, expr):
         else:
             return "%s^T" % self._print(mat)
 
+    def _print_Trace(self, expr):
+        mat = expr.arg
+        return r"\mathrm{tr}\left (%s \right )" % self._print(mat)
+
     def _print_Adjoint(self, expr):
         mat = expr.arg
         from sympy.matrices import MatrixSymbol

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sympy/printing/latex.py | 292 | 292 | 124 | 1 | 42790
| sympy/printing/latex.py | 1465 | 1465 | 3 | 1 | 742


## Problem Statement

```
LaTeX printing for Matrix Expression
\`\`\`py
>>> A = MatrixSymbol("A", n, n)
>>> latex(trace(A**2))
'Trace(A**2)'
\`\`\`

The bad part is not only is Trace not recognized, but whatever printer is being used doesn't fallback to the LaTeX printer for the inner expression (it should be `A^2`). 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sympy/printing/latex.py** | 1406 | 1431| 271 | 271 | 24180 | 
| 2 | **1 sympy/printing/latex.py** | 1432 | 1452| 208 | 479 | 24180 | 
| **-> 3 <-** | **1 sympy/printing/latex.py** | 1454 | 1486| 263 | 742 | 24180 | 
| 4 | **1 sympy/printing/latex.py** | 1541 | 1588| 489 | 1231 | 24180 | 
| 5 | **1 sympy/printing/latex.py** | 1509 | 1539| 303 | 1534 | 24180 | 
| 6 | 2 sympy/matrices/expressions/trace.py | 1 | 35| 195 | 1729 | 24756 | 
| 7 | 2 sympy/matrices/expressions/trace.py | 76 | 92| 105 | 1834 | 24756 | 
| 8 | 3 sympy/printing/julia.py | 342 | 350| 139 | 1973 | 30515 | 
| 9 | 3 sympy/printing/julia.py | 327 | 339| 181 | 2154 | 30515 | 
| 10 | **3 sympy/printing/latex.py** | 489 | 539| 587 | 2741 | 30515 | 
| 11 | **3 sympy/printing/latex.py** | 1488 | 1500| 159 | 2900 | 30515 | 
| 12 | **3 sympy/printing/latex.py** | 365 | 387| 268 | 3168 | 30515 | 
| 13 | **3 sympy/printing/latex.py** | 2273 | 2431| 2158 | 5326 | 30515 | 
| 14 | **3 sympy/printing/latex.py** | 2432 | 2458| 275 | 5601 | 30515 | 
| 15 | **3 sympy/printing/latex.py** | 1345 | 1371| 210 | 5811 | 30515 | 
| 16 | **3 sympy/printing/latex.py** | 1198 | 1209| 187 | 5998 | 30515 | 
| 17 | 4 sympy/printing/pretty/pretty.py | 722 | 747| 284 | 6282 | 52074 | 
| 18 | **4 sympy/printing/latex.py** | 1 | 82| 713 | 6995 | 52074 | 
| 19 | **4 sympy/printing/latex.py** | 801 | 875| 684 | 7679 | 52074 | 
| 20 | 5 sympy/printing/glsl.py | 135 | 168| 339 | 8018 | 56899 | 
| 21 | 5 sympy/printing/pretty/pretty.py | 750 | 769| 207 | 8225 | 56899 | 
| 22 | **5 sympy/printing/latex.py** | 2127 | 2191| 754 | 8979 | 56899 | 
| 23 | **5 sympy/printing/latex.py** | 612 | 641| 272 | 9251 | 56899 | 
| 24 | **5 sympy/printing/latex.py** | 1187 | 1196| 138 | 9389 | 56899 | 
| 25 | **5 sympy/printing/latex.py** | 1120 | 1185| 672 | 10061 | 56899 | 
| 26 | 6 sympy/printing/repr.py | 88 | 100| 139 | 10200 | 59081 | 
| 27 | **6 sympy/printing/latex.py** | 2017 | 2060| 771 | 10971 | 59081 | 
| 28 | **6 sympy/printing/latex.py** | 1267 | 1327| 753 | 11724 | 59081 | 
| 29 | 6 sympy/printing/glsl.py | 107 | 133| 393 | 12117 | 59081 | 
| 30 | 7 sympy/printing/octave.py | 330 | 338| 140 | 12257 | 65594 | 
| 31 | **7 sympy/printing/latex.py** | 440 | 487| 529 | 12786 | 65594 | 
| 32 | **7 sympy/printing/latex.py** | 715 | 782| 658 | 13444 | 65594 | 
| 33 | **7 sympy/printing/latex.py** | 652 | 682| 329 | 13773 | 65594 | 
| 34 | **7 sympy/printing/latex.py** | 1622 | 1687| 562 | 14335 | 65594 | 
| 35 | **7 sympy/printing/latex.py** | 389 | 438| 383 | 14718 | 65594 | 
| 36 | **7 sympy/printing/latex.py** | 903 | 958| 577 | 15295 | 65594 | 
| 37 | **7 sympy/printing/latex.py** | 1329 | 1343| 147 | 15442 | 65594 | 
| 38 | **7 sympy/printing/latex.py** | 1502 | 1507| 116 | 15558 | 65594 | 
| 39 | 7 sympy/printing/julia.py | 372 | 390| 187 | 15745 | 65594 | 
| 40 | **7 sympy/printing/latex.py** | 324 | 340| 165 | 15910 | 65594 | 
| 41 | **7 sympy/printing/latex.py** | 1711 | 1720| 131 | 16041 | 65594 | 
| 42 | 8 sympy/physics/vector/printing.py | 45 | 120| 664 | 16705 | 68941 | 
| 43 | **8 sympy/printing/latex.py** | 1984 | 2006| 234 | 16939 | 68941 | 
| 44 | **8 sympy/printing/latex.py** | 2008 | 2015| 118 | 17057 | 68941 | 
| 45 | 8 sympy/printing/julia.py | 353 | 369| 181 | 17238 | 68941 | 
| 46 | **8 sympy/printing/latex.py** | 1699 | 1709| 137 | 17375 | 68941 | 
| 47 | 8 sympy/printing/octave.py | 317 | 327| 157 | 17532 | 68941 | 
| 48 | **8 sympy/printing/latex.py** | 1393 | 1404| 171 | 17703 | 68941 | 
| 49 | 9 sympy/printing/str.py | 253 | 268| 146 | 17849 | 76262 | 
| 50 | 10 sympy/printing/theanocode.py | 140 | 216| 745 | 18594 | 80614 | 
| 51 | **10 sympy/printing/latex.py** | 984 | 1068| 809 | 19403 | 80614 | 
| 52 | **10 sympy/printing/latex.py** | 563 | 580| 197 | 19600 | 80614 | 
| 53 | **10 sympy/printing/latex.py** | 541 | 561| 221 | 19821 | 80614 | 
| 54 | **10 sympy/printing/latex.py** | 1722 | 1758| 380 | 20201 | 80614 | 
| 55 | **10 sympy/printing/latex.py** | 960 | 969| 126 | 20327 | 80614 | 
| 56 | **10 sympy/printing/latex.py** | 2193 | 2203| 161 | 20488 | 80614 | 
| 57 | **10 sympy/printing/latex.py** | 784 | 799| 162 | 20650 | 80614 | 
| 58 | **10 sympy/printing/latex.py** | 1867 | 1914| 457 | 21107 | 80614 | 
| 59 | **10 sympy/printing/latex.py** | 1103 | 1118| 138 | 21245 | 80614 | 
| 60 | **10 sympy/printing/latex.py** | 302 | 322| 149 | 21394 | 80614 | 
| 61 | 10 sympy/physics/vector/printing.py | 337 | 374| 339 | 21733 | 80614 | 
| 62 | 11 sympy/printing/pycode.py | 534 | 554| 195 | 21928 | 85734 | 
| 63 | **11 sympy/printing/latex.py** | 2225 | 2235| 159 | 22087 | 85734 | 
| 64 | **11 sympy/printing/latex.py** | 1070 | 1101| 308 | 22395 | 85734 | 
| 65 | **11 sympy/printing/latex.py** | 582 | 610| 239 | 22634 | 85734 | 
| 66 | **11 sympy/printing/latex.py** | 1590 | 1620| 279 | 22913 | 85734 | 
| 67 | **11 sympy/printing/latex.py** | 1373 | 1391| 135 | 23048 | 85734 | 
| 68 | **11 sympy/printing/latex.py** | 1790 | 1852| 538 | 23586 | 85734 | 
| 69 | 12 sympy/printing/fcode.py | 254 | 295| 352 | 23938 | 93655 | 
| 70 | **12 sympy/printing/latex.py** | 1689 | 1697| 135 | 24073 | 93655 | 
| 71 | **12 sympy/printing/latex.py** | 684 | 696| 159 | 24232 | 93655 | 
| 72 | 12 sympy/matrices/expressions/trace.py | 51 | 73| 162 | 24394 | 93655 | 
| 73 | **12 sympy/printing/latex.py** | 1211 | 1265| 730 | 25124 | 93655 | 
| 74 | **12 sympy/printing/latex.py** | 971 | 982| 157 | 25281 | 93655 | 
| 75 | 12 sympy/printing/repr.py | 102 | 132| 243 | 25524 | 93655 | 
| 76 | 12 sympy/printing/pretty/pretty.py | 796 | 819| 216 | 25740 | 93655 | 
| 77 | 13 sympy/printing/rust.py | 429 | 466| 349 | 26089 | 99092 | 
| 78 | 13 sympy/matrices/expressions/trace.py | 37 | 49| 123 | 26212 | 99092 | 
| 79 | **13 sympy/printing/latex.py** | 877 | 886| 136 | 26348 | 99092 | 
| 80 | **13 sympy/printing/latex.py** | 1973 | 1982| 125 | 26473 | 99092 | 
| 81 | **13 sympy/printing/latex.py** | 342 | 363| 196 | 26669 | 99092 | 
| 82 | **13 sympy/printing/latex.py** | 2215 | 2223| 129 | 26798 | 99092 | 
| 83 | **13 sympy/printing/latex.py** | 2205 | 2213| 123 | 26921 | 99092 | 
| 84 | **13 sympy/printing/latex.py** | 2062 | 2107| 407 | 27328 | 99092 | 
| 85 | 13 sympy/printing/octave.py | 341 | 357| 183 | 27511 | 99092 | 
| 86 | 14 sympy/printing/mathml.py | 219 | 244| 211 | 27722 | 106352 | 
| 87 | 15 sympy/printing/codeprinter.py | 66 | 122| 477 | 28199 | 110789 | 
| 88 | **15 sympy/printing/latex.py** | 888 | 901| 139 | 28338 | 110789 | 
| 89 | **15 sympy/printing/latex.py** | 187 | 208| 219 | 28557 | 110789 | 
| 90 | **15 sympy/printing/latex.py** | 2238 | 2451| 294 | 28851 | 110789 | 
| 91 | 15 sympy/printing/octave.py | 360 | 378| 189 | 29040 | 110789 | 
| 92 | 15 sympy/printing/str.py | 207 | 251| 481 | 29521 | 110789 | 
| 93 | **15 sympy/printing/latex.py** | 1916 | 1971| 414 | 29935 | 110789 | 
| 94 | **15 sympy/printing/latex.py** | 1775 | 1788| 143 | 30078 | 110789 | 
| 95 | 16 sympy/matrices/expressions/matexpr.py | 645 | 701| 422 | 30500 | 116956 | 
| 96 | 16 sympy/printing/codeprinter.py | 295 | 329| 358 | 30858 | 116956 | 
| 97 | 16 sympy/printing/mathml.py | 575 | 601| 218 | 31076 | 116956 | 
| 98 | 16 sympy/printing/glsl.py | 318 | 496| 1932 | 33008 | 116956 | 
| 99 | 16 sympy/matrices/expressions/matexpr.py | 140 | 200| 443 | 33451 | 116956 | 
| 100 | 16 sympy/printing/pycode.py | 412 | 423| 141 | 33592 | 116956 | 
| 101 | **16 sympy/printing/latex.py** | 2109 | 2125| 135 | 33727 | 116956 | 
| 102 | **16 sympy/printing/latex.py** | 121 | 185| 536 | 34263 | 116956 | 
| 103 | **16 sympy/printing/latex.py** | 698 | 713| 161 | 34424 | 116956 | 
| 104 | 16 sympy/printing/julia.py | 214 | 256| 290 | 34714 | 116956 | 
| 105 | 16 sympy/printing/pycode.py | 452 | 505| 694 | 35408 | 116956 | 
| 106 | 16 sympy/printing/pretty/pretty.py | 772 | 794| 224 | 35632 | 116956 | 
| 107 | **16 sympy/printing/latex.py** | 643 | 650| 127 | 35759 | 116956 | 
| 108 | 16 sympy/printing/pretty/pretty.py | 838 | 888| 449 | 36208 | 116956 | 
| 109 | 16 sympy/matrices/expressions/matexpr.py | 202 | 246| 417 | 36625 | 116956 | 
| 110 | 17 sympy/matrices/expressions/blockmatrix.py | 1 | 20| 219 | 36844 | 120703 | 
| 111 | 17 sympy/printing/codeprinter.py | 124 | 201| 718 | 37562 | 120703 | 
| 112 | **17 sympy/printing/latex.py** | 1854 | 1865| 133 | 37695 | 120703 | 
| 113 | 18 sympy/printing/jscode.py | 173 | 204| 259 | 37954 | 123518 | 
| 114 | **18 sympy/printing/latex.py** | 83 | 118| 491 | 38445 | 123518 | 
| 115 | 18 sympy/physics/vector/printing.py | 122 | 153| 320 | 38765 | 123518 | 
| 116 | 18 sympy/printing/julia.py | 194 | 211| 198 | 38963 | 123518 | 
| 117 | 18 sympy/printing/pretty/pretty.py | 990 | 1022| 354 | 39317 | 123518 | 
| 118 | 19 sympy/matrices/expressions/funcmatrix.py | 1 | 52| 387 | 39704 | 123905 | 
| 119 | 20 sympy/printing/rcode.py | 230 | 264| 371 | 40075 | 127629 | 
| 120 | 20 sympy/printing/rcode.py | 181 | 216| 384 | 40459 | 127629 | 
| 121 | 20 sympy/printing/pretty/pretty.py | 653 | 720| 515 | 40974 | 127629 | 
| 122 | 21 sympy/printing/printer.py | 1 | 173| 1420 | 42394 | 130013 | 
| 123 | **21 sympy/printing/latex.py** | 1760 | 1773| 134 | 42528 | 130013 | 
| **-> 124 <-** | **21 sympy/printing/latex.py** | 262 | 299| 262 | 42790 | 130013 | 
| 125 | 22 sympy/matrices/expressions/__init__.py | 1 | 20| 187 | 42977 | 130201 | 
| 126 | 23 sympy/interactive/printing.py | 155 | 236| 698 | 43675 | 134083 | 
| 127 | 23 sympy/printing/str.py | 326 | 380| 419 | 44094 | 134083 | 
| 128 | 23 sympy/printing/str.py | 701 | 790| 754 | 44848 | 134083 | 
| 129 | 23 sympy/printing/mathml.py | 462 | 506| 353 | 45201 | 134083 | 
| 130 | 23 sympy/matrices/expressions/matexpr.py | 33 | 138| 782 | 45983 | 134083 | 
| 131 | 24 sympy/printing/llvmjitcode.py | 57 | 77| 242 | 46225 | 138099 | 
| 132 | 25 sympy/parsing/latex/_parse_latex_antlr.py | 1 | 59| 471 | 46696 | 142491 | 
| 133 | 26 sympy/matrices/expressions/matmul.py | 78 | 136| 501 | 47197 | 145250 | 
| 134 | 26 sympy/printing/julia.py | 491 | 633| 1597 | 48794 | 145250 | 
| 135 | 26 sympy/printing/codeprinter.py | 494 | 529| 395 | 49189 | 145250 | 
| 136 | 26 sympy/printing/codeprinter.py | 331 | 362| 257 | 49446 | 145250 | 
| 137 | 26 sympy/printing/julia.py | 259 | 284| 281 | 49727 | 145250 | 
| 138 | 26 sympy/printing/julia.py | 119 | 191| 701 | 50428 | 145250 | 
| 139 | 26 sympy/printing/mathml.py | 108 | 154| 359 | 50787 | 145250 | 
| 140 | 27 sympy/printing/mathematica.py | 1 | 35| 412 | 51199 | 146478 | 
| 141 | 27 sympy/printing/theanocode.py | 67 | 97| 320 | 51519 | 146478 | 
| 142 | 27 sympy/printing/pretty/pretty.py | 1210 | 1256| 403 | 51922 | 146478 | 
| 143 | 28 sympy/printing/lambdarepr.py | 148 | 241| 742 | 52664 | 148400 | 
| 144 | 29 sympy/printing/ccode.py | 740 | 876| 1518 | 54182 | 156397 | 
| 145 | 29 sympy/printing/octave.py | 566 | 710| 1626 | 55808 | 156397 | 
| 146 | 29 sympy/matrices/expressions/matexpr.py | 285 | 318| 355 | 56163 | 156397 | 
| 147 | 29 sympy/printing/ccode.py | 377 | 405| 314 | 56477 | 156397 | 
| 148 | 29 sympy/printing/mathematica.py | 38 | 117| 710 | 57187 | 156397 | 
| 149 | 29 sympy/printing/pycode.py | 193 | 272| 718 | 57905 | 156397 | 
| 150 | 29 sympy/printing/glsl.py | 248 | 283| 303 | 58208 | 156397 | 
| 151 | 29 sympy/printing/rcode.py | 307 | 420| 1239 | 59447 | 156397 | 
| 152 | 29 sympy/printing/lambdarepr.py | 129 | 145| 149 | 59596 | 156397 | 
| 153 | 30 sympy/physics/quantum/tensorproduct.py | 210 | 237| 285 | 59881 | 159913 | 
| 154 | 30 sympy/printing/octave.py | 255 | 280| 282 | 60163 | 159913 | 
| 155 | 30 sympy/matrices/expressions/blockmatrix.py | 22 | 127| 821 | 60984 | 159913 | 
| 156 | 30 sympy/printing/codeprinter.py | 390 | 441| 507 | 61491 | 159913 | 
| 157 | 30 sympy/matrices/expressions/blockmatrix.py | 266 | 301| 302 | 61793 | 159913 | 
| 158 | 30 sympy/printing/mathml.py | 555 | 573| 160 | 61953 | 159913 | 
| 159 | 30 sympy/printing/julia.py | 46 | 116| 498 | 62451 | 159913 | 
| 160 | 30 sympy/matrices/expressions/matexpr.py | 399 | 442| 418 | 62869 | 159913 | 
| 161 | 31 sympy/matrices/expressions/kronecker.py | 335 | 355| 160 | 63029 | 163222 | 
| 162 | 31 sympy/printing/octave.py | 62 | 133| 504 | 63533 | 163222 | 
| 163 | 31 sympy/printing/str.py | 569 | 630| 474 | 64007 | 163222 | 
| 164 | 31 sympy/printing/llvmjitcode.py | 79 | 107| 267 | 64274 | 163222 | 
| 165 | 31 sympy/printing/codeprinter.py | 203 | 238| 257 | 64531 | 163222 | 
| 166 | 32 sympy/physics/vector/vector.py | 216 | 250| 388 | 64919 | 169158 | 
| 167 | **32 sympy/printing/latex.py** | 210 | 231| 215 | 65134 | 169158 | 
| 168 | 32 sympy/printing/pycode.py | 387 | 410| 213 | 65347 | 169158 | 
| 169 | 32 sympy/printing/mathml.py | 342 | 377| 313 | 65660 | 169158 | 
| 170 | 32 sympy/printing/mathml.py | 508 | 553| 336 | 65996 | 169158 | 
| 171 | 32 sympy/printing/glsl.py | 215 | 246| 359 | 66355 | 169158 | 
| 172 | 32 sympy/printing/mathml.py | 191 | 217| 232 | 66587 | 169158 | 
| 173 | 33 sympy/matrices/expressions/transpose.py | 1 | 71| 455 | 67042 | 169756 | 
| 174 | 33 sympy/printing/codeprinter.py | 443 | 492| 453 | 67495 | 169756 | 
| 175 | 33 sympy/matrices/expressions/matexpr.py | 320 | 347| 168 | 67663 | 169756 | 
| 176 | 33 sympy/printing/mathml.py | 246 | 286| 345 | 68008 | 169756 | 
| 177 | 33 sympy/printing/julia.py | 393 | 418| 238 | 68246 | 169756 | 
| 178 | 33 sympy/printing/theanocode.py | 1 | 64| 573 | 68819 | 169756 | 
| 179 | 33 sympy/printing/glsl.py | 306 | 488| 139 | 68958 | 169756 | 
| 180 | 34 sympy/printing/dot.py | 199 | 210| 181 | 69139 | 171644 | 
| 181 | 34 sympy/printing/octave.py | 136 | 208| 704 | 69843 | 171644 | 
| 182 | 34 sympy/printing/lambdarepr.py | 107 | 127| 178 | 70021 | 171644 | 
| 183 | 34 sympy/printing/str.py | 73 | 161| 805 | 70826 | 171644 | 
| 184 | 34 sympy/printing/mathml.py | 898 | 946| 328 | 71154 | 171644 | 
| 185 | 34 sympy/printing/mathml.py | 750 | 785| 301 | 71455 | 171644 | 
| 186 | 34 sympy/printing/mathml.py | 625 | 659| 302 | 71757 | 171644 | 
| 187 | 35 sympy/parsing/latex/__init__.py | 1 | 32| 273 | 72030 | 171917 | 
| 188 | 35 sympy/printing/mathml.py | 156 | 189| 260 | 72290 | 171917 | 
| 189 | 35 sympy/printing/rcode.py | 78 | 128| 348 | 72638 | 171917 | 
| 190 | 36 sympy/matrices/expressions/matpow.py | 52 | 86| 317 | 72955 | 172653 | 
| 191 | 36 sympy/printing/jscode.py | 100 | 110| 133 | 73088 | 172653 | 
| 192 | 36 sympy/printing/pretty/pretty.py | 1258 | 1340| 773 | 73861 | 172653 | 
| 193 | 37 sympy/matrices/expressions/hadamard.py | 34 | 84| 396 | 74257 | 173294 | 
| 194 | 37 sympy/matrices/expressions/blockmatrix.py | 347 | 370| 206 | 74463 | 173294 | 
| 195 | 37 sympy/printing/pretty/pretty.py | 37 | 87| 385 | 74848 | 173294 | 
| 196 | 37 sympy/printing/fcode.py | 164 | 187| 207 | 75055 | 173294 | 
| 197 | 37 sympy/printing/str.py | 407 | 468| 607 | 75662 | 173294 | 
| 198 | 37 sympy/printing/octave.py | 478 | 522| 496 | 76158 | 173294 | 
| 199 | 37 sympy/printing/octave.py | 231 | 252| 172 | 76330 | 173294 | 
| 200 | 37 sympy/matrices/expressions/matexpr.py | 470 | 589| 1182 | 77512 | 173294 | 
| 201 | 37 sympy/printing/mathml.py | 379 | 398| 190 | 77702 | 173294 | 
| 202 | 37 sympy/printing/octave.py | 211 | 228| 199 | 77901 | 173294 | 
| 203 | 37 sympy/printing/rust.py | 511 | 626| 1226 | 79127 | 173294 | 
| 204 | 37 sympy/matrices/expressions/matmul.py | 16 | 46| 224 | 79351 | 173294 | 
| 205 | 37 sympy/printing/lambdarepr.py | 1 | 58| 402 | 79753 | 173294 | 
| 206 | 37 sympy/matrices/expressions/matexpr.py | 704 | 752| 261 | 80014 | 173294 | 
| 207 | 37 sympy/interactive/printing.py | 85 | 97| 189 | 80203 | 173294 | 
| 208 | 38 sympy/matrices/sparse.py | 1 | 17| 120 | 80323 | 183738 | 
| 209 | 38 sympy/printing/codeprinter.py | 364 | 388| 255 | 80578 | 183738 | 
| 210 | 38 sympy/printing/mathml.py | 1 | 54| 379 | 80957 | 183738 | 
| 211 | 38 sympy/printing/pretty/pretty.py | 1063 | 1099| 307 | 81264 | 183738 | 
| 212 | 38 sympy/matrices/expressions/kronecker.py | 212 | 233| 208 | 81472 | 183738 | 
| 213 | 38 sympy/matrices/expressions/matexpr.py | 592 | 616| 215 | 81687 | 183738 | 


### Hint

```
What is the correct way to print the trace? AFAIK there isn't one built in to Latex. One option is \`\`\`\mathrm{Tr}\`\`\`. Or \`\`\`\operatorname{Tr}\`\`\`.
What's the difference between the two. It looks like we use both in different parts of the latex printer. 
\operatorname puts a thin space after the operator.
```

## Patch

```diff
diff --git a/sympy/printing/latex.py b/sympy/printing/latex.py
--- a/sympy/printing/latex.py
+++ b/sympy/printing/latex.py
@@ -289,6 +289,10 @@ def _do_exponent(self, expr, exp):
         else:
             return expr
 
+    def _print_Basic(self, expr):
+        l = [self._print(o) for o in expr.args]
+        return self._deal_with_super_sub(expr.__class__.__name__) + r"\left(%s\right)" % ", ".join(l)
+
     def _print_bool(self, e):
         return r"\mathrm{%s}" % e
 
@@ -1462,6 +1466,10 @@ def _print_Transpose(self, expr):
         else:
             return "%s^T" % self._print(mat)
 
+    def _print_Trace(self, expr):
+        mat = expr.arg
+        return r"\mathrm{tr}\left (%s \right )" % self._print(mat)
+
     def _print_Adjoint(self, expr):
         mat = expr.arg
         from sympy.matrices import MatrixSymbol

```

## Test Patch

```diff
diff --git a/sympy/printing/tests/test_latex.py b/sympy/printing/tests/test_latex.py
--- a/sympy/printing/tests/test_latex.py
+++ b/sympy/printing/tests/test_latex.py
@@ -1866,3 +1866,35 @@ def test_latex_printer_tensor():
 
     expr = TensorElement(K(i,j,-k,-l), {i:3})
     assert latex(expr) == 'K{}^{i=3,j}{}_{kl}'
+
+
+def test_trace():
+    # Issue 15303
+    from sympy import trace
+    A = MatrixSymbol("A", 2, 2)
+    assert latex(trace(A)) == r"\mathrm{tr}\left (A \right )"
+    assert latex(trace(A**2)) == r"\mathrm{tr}\left (A^{2} \right )"
+
+
+def test_print_basic():
+    # Issue 15303
+    from sympy import Basic, Expr
+
+    # dummy class for testing printing where the function is not implemented in latex.py
+    class UnimplementedExpr(Expr):
+        def __new__(cls, e):
+            return Basic.__new__(cls, e)
+
+    # dummy function for testing
+    def unimplemented_expr(expr):
+        return UnimplementedExpr(expr).doit()
+
+    # override class name to use superscript / subscript
+    def unimplemented_expr_sup_sub(expr):
+        result = UnimplementedExpr(expr)
+        result.__class__.__name__ = 'UnimplementedExpr_x^1'
+        return result
+
+    assert latex(unimplemented_expr(x)) == r'UnimplementedExpr\left(x\right)'
+    assert latex(unimplemented_expr(x**2)) == r'UnimplementedExpr\left(x^{2}\right)'
+    assert latex(unimplemented_expr_sup_sub(x)) == r'UnimplementedExpr^{1}_{x}\left(x\right)'

```


## Code snippets

### 1 - sympy/printing/latex.py:

Start line: 1406, End line: 1431

```python
class LatexPrinter(Printer):

    def _print_MatrixBase(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" & ".join([ self._print(i) for i in expr[line, :] ]))

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.cols <= 10) is True:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        return out_str % r"\\".join(lines)
```
### 2 - sympy/printing/latex.py:

Start line: 1432, End line: 1452

```python
class LatexPrinter(Printer):
    _print_ImmutableMatrix = _print_ImmutableDenseMatrix \
                           = _print_Matrix \
                           = _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '_{%s, %s}' % (expr.i, expr.j)

    def _print_MatrixSlice(self, expr):
        def latexslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(self._print, x))
        return (self._print(expr.parent) + r'\left[' +
                latexslice(expr.rowslice) + ', ' +
                latexslice(expr.colslice) + r'\right]')
```
### 3 - sympy/printing/latex.py:

Start line: 1454, End line: 1486

```python
class LatexPrinter(Printer):

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^T" % self._print(mat)
        else:
            return "%s^T" % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"\left(%s\right)^\dagger" % self._print(mat)
        else:
            return r"%s^\dagger" % self._print(mat)

    def _print_MatAdd(self, expr):
        terms = [self._print(t) for t in expr.args]
        l = []
        for t in terms:
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)
```
### 4 - sympy/printing/latex.py:

Start line: 1541, End line: 1588

```python
class LatexPrinter(Printer):

    def _print_NDimArray(self, expr):

        if expr.rank() == 0:
            return self._print(expr[()])

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                      r'\right' + right_delim

        if expr.rank() == 0:
            return block_str % ""

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str
```
### 5 - sympy/printing/latex.py:

Start line: 1509, End line: 1539

```python
class LatexPrinter(Printer):

    def _print_HadamardProduct(self, expr):
        from sympy import Add, MatAdd, MatMul

        def parens(x):
            if isinstance(x, (Add, MatAdd, MatMul)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return r' \circ '.join(map(parens, expr.args))

    def _print_KroneckerProduct(self, expr):
        from sympy import Add, MatAdd, MatMul

        def parens(x):
            if isinstance(x, (Add, MatAdd, MatMul)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)
        return r' \otimes '.join(map(parens, expr.args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol):
            return r"\left(%s\right)^{%s}" % (self._print(base), self._print(exp))
        else:
            return "%s^{%s}" % (self._print(base), self._print(exp))

    def _print_ZeroMatrix(self, Z):
        return r"\mathbb{0}"

    def _print_Identity(self, I):
        return r"\mathbb{I}"
```
### 6 - sympy/matrices/expressions/trace.py:

Start line: 1, End line: 35

```python
from __future__ import print_function, division

from sympy import Basic, Expr, sympify
from sympy.matrices.matrices import MatrixBase
from .matexpr import ShapeError


class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)

    See Also:
        trace
    """
    is_Trace = True

    def __new__(cls, mat):
        mat = sympify(mat)

        if not mat.is_Matrix:
            raise TypeError("input to Trace, %s, is not a matrix" % str(mat))

        if not mat.is_square:
            raise ShapeError("Trace of a non-square matrix")

        return Basic.__new__(cls, mat)

    def _eval_transpose(self):
        return self
```
### 7 - sympy/matrices/expressions/trace.py:

Start line: 76, End line: 92

```python
def trace(expr):
    """ Trace of a Matrix.  Sum of the diagonal elements

    >>> from sympy import trace, Symbol, MatrixSymbol, pprint, eye
    >>> n = Symbol('n')
    >>> X = MatrixSymbol('X', n, n)  # A square matrix
    >>> trace(2*X)
    2*Trace(X)

    >>> trace(eye(3))
    3

    See Also:
        Trace
    """
    return Trace(expr).doit()
```
### 8 - sympy/printing/julia.py:

Start line: 342, End line: 350

```python
class JuliaCodePrinter(CodePrinter):


    def _print_SparseMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list();
        # make row vectors of the indices and entries
        I = Matrix([k[0] + 1 for k in L])
        J = Matrix([k[1] + 1 for k in L])
        AIJ = Matrix([k[2] for k in L])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)
```
### 9 - sympy/printing/julia.py:

Start line: 327, End line: 339

```python
class JuliaCodePrinter(CodePrinter):


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if A.rows == 0 or A.cols == 0:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            return "[%s]" % A[0, 0]
        elif A.rows == 1:
            return "[%s]" % A.table(self, rowstart='', rowend='', colsep=' ')
        elif A.cols == 1:
            # note .table would unnecessarily equispace the rows
            return "[%s]" % ", ".join([self._print(a) for a in A])
        return "[%s]" % A.table(self, rowstart='', rowend='',
                                rowsep=';\n', colsep=' ')
```
### 10 - sympy/printing/latex.py:

Start line: 489, End line: 539

```python
class LatexPrinter(Printer):

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1:
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
            base, p, q = self.parenthesize(expr.base, PRECEDENCE['Pow']), expr.exp.p, expr.exp.q
            # issue #12886: add parentheses for superscripts raised to powers
            if '^' in base and expr.base.is_Symbol:
                base = r"\left(%s\right)" % base
            if expr.base.is_Function:
                return self._print(expr.base, exp="%s/%s" % (p, q))
            return r"%s^{%s/%s}" % (base, p, q)
        elif expr.exp.is_Rational and expr.exp.is_negative and expr.base.is_commutative:
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
                exp = self._print(expr.exp)
                # issue #12886: add parentheses around superscripts raised to powers
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                if '^' in base and expr.base.is_Symbol:
                    base = r"\left(%s\right)" % base
                elif isinstance(expr.base, Derivative
                        ) and base.startswith(r'\left('
                        ) and re.match(r'\\left\(\\d?d?dot', base
                        ) and base.endswith(r'\right)'):
                    # don't use parentheses around dotted derivative
                    base = base[6: -7]  # remove outermost added parens

                return tex % (base, exp)
```
### 11 - sympy/printing/latex.py:

Start line: 1488, End line: 1500

```python
class LatexPrinter(Printer):

    def _print_MatMul(self, expr):
        from sympy import Add, MatAdd, HadamardProduct, MatMul, Mul

        def parens(x):
            if isinstance(x, (Add, MatAdd, HadamardProduct)):
                return r"\left(%s\right)" % self._print(x)
            return self._print(x)

        if isinstance(expr, MatMul) and expr.args[0].is_Number and expr.args[0]<0:
            expr = Mul(-1*expr.args[0], MatMul(*expr.args[1:]))
            return '-' + ' '.join(map(parens, expr.args))
        else:
            return ' '.join(map(parens, expr.args))
```
### 12 - sympy/printing/latex.py:

Start line: 365, End line: 387

```python
class LatexPrinter(Printer):

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(func, PRECEDENCE['Mul'])
```
### 13 - sympy/printing/latex.py:

Start line: 2273, End line: 2431

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``abbreviated``, ``full``, or ``power``. Defaults to ``abbreviated``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``[``, ``(``, or
        the empty string. Defaults to ``[``.
    mat_str : string, optional
        Which matrix environment string to emit. ``smallmatrix``, ``matrix``,
        ``array``, etc. Defaults to ``smallmatrix`` for inline mode, ``matrix``
        for matrices of no more than 10 columns, and ``array`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``plain``, ``inline``, ``equation`` or ``equation*``.  If ``mode``
        is set to ``plain``, then the resulting code will not be delimited at
        all (this is the default). If ``mode`` is set to ``inline`` then inline
        LaTeX ``$...$`` will be used. If ``mode`` is set to ``equation`` or
        ``equation*``, the resulting code will be enclosed in the ``equation``
        or ``equation*`` environment (remember to import ``amsmath`` for
        ``equation*``), unless the ``itex`` option is set. In the latter case,
        the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``, ``ldot``,
        ``dot``, or ``times``.
    order: string, optional
        Any of the supported monomial orderings (currently ``lex``, ``grlex``,
        or ``grevlex``), ``old``, and ``none``. This parameter does nothing for
        Mul objects. Setting order to ``old`` uses the compatibility ordering
        for Add defined in Printer. For very large expressions, set the
        ``order`` keyword to ``none`` if speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.

    Notes
    =====

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> from sympy import latex, Rational
    >>> from sympy.abc import tau
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Examples
    ========

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    Basic usage:

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    ``mode`` and ``itex`` options:

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    Fraction options:

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left (\frac{7}{2} \right )}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}
    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    Multiplication options:

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left (\frac{7}{2} \right )}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left (\frac{7}{2} \right )}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left (\frac{7}{2} \right )}

    Matrix options:

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    Custom printing of symbols:

    >>> print(latex(x**2, symbol_names={x: 'x_i'}))
    x_i^{2}

    Logarithms:

    >>> print(latex(log(10)))
    \log{\left (10 \right )}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left (10 \right )}

    ``latex()`` also supports the builtin container types list, tuple, and
    dictionary.

    >>> print(latex([2/x, y], mode='inline'))
    $\left [ 2 / x, \quad y\right ]$

    """
    # ... other code
```
### 14 - sympy/printing/latex.py:

Start line: 2432, End line: 2458

```python
def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None):
    if symbol_names is None:
        symbol_names = {}

    settings = {
        'fold_frac_powers' : fold_frac_powers,
        'fold_func_brackets' : fold_func_brackets,
        'fold_short_frac' : fold_short_frac,
        'inv_trig_style' : inv_trig_style,
        'itex' : itex,
        'ln_notation' : ln_notation,
        'long_frac_ratio' : long_frac_ratio,
        'mat_delim' : mat_delim,
        'mat_str' : mat_str,
        'mode' : mode,
        'mul_symbol' : mul_symbol,
        'order' : order,
        'symbol_names' : symbol_names,
    }

    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""
    print(latex(expr, **settings))
```
### 15 - sympy/printing/latex.py:

Start line: 1345, End line: 1371

```python
class LatexPrinter(Printer):

    def _print_Symbol(self, expr):
        if expr in self._settings['symbol_names']:
            return self._settings['symbol_names'][expr]

        return self._deal_with_super_sub(expr.name) if \
            '\\' not in expr.name else expr.name

    _print_RandomSymbol = _print_Symbol
    _print_MatrixSymbol = _print_Symbol

    def _deal_with_super_sub(self, string):
        if '{' in string:
            return string

        name, supers, subs = split_super_sub(string)

        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        # glue all items together:
        if len(supers) > 0:
            name += "^{%s}" % " ".join(supers)
        if len(subs) > 0:
            name += "_{%s}" % " ".join(subs)

        return name
```
### 16 - sympy/printing/latex.py:

Start line: 1198, End line: 1209

```python
class LatexPrinter(Printer):

    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, self._print(exp))
        return tex
```
### 18 - sympy/printing/latex.py:

Start line: 1, End line: 82

```python
"""
A Printer which converts an expression into its LaTeX equivalent.
"""

from __future__ import print_function, division

import itertools

from sympy.core import S, Add, Symbol, Mod
from sympy.core.sympify import SympifyError
from sympy.core.alphabets import greeks
from sympy.core.operations import AssocOp
from sympy.core.containers import Tuple
from sympy.logic.boolalg import true
from sympy.core.function import (_coeff_isneg,
    UndefinedFunction, AppliedUndef, Derivative)

## sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from .printer import Printer
from .conventions import split_super_sub, requires_partial
from .precedence import precedence, PRECEDENCE

import mpmath.libmp as mlib
from mpmath.libmp import prec_to_dps

from sympy.core.compatibility import default_sort_key, range
from sympy.utilities.iterables import has_variety

import re

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at http://www.mathjax.org/docs/1.1/tex.html#supported-latex-commands
# This variable only contains those functions which sympy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                    'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec', 'csc',
                    'cot', 'coth', 're', 'im', 'frac', 'root', 'arg',
                    ]

tex_greek_dictionary = {
    'Alpha': 'A',
    'Beta': 'B',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': 'E',
    'Zeta': 'Z',
    'Eta': 'H',
    'Theta': r'\Theta',
    'Iota': 'I',
    'Kappa': 'K',
    'Lambda': r'\Lambda',
    'Mu': 'M',
    'Nu': 'N',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': 'O',
    'Pi': r'\Pi',
    'Rho': 'P',
    'Sigma': r'\Sigma',
    'Tau': 'T',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': 'X',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'X',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

other_symbols = set(['aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp', ])

# Variable name modifiers
```
### 19 - sympy/printing/latex.py:

Start line: 801, End line: 875

```python
class LatexPrinter(Printer):

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        args = (symbols, self._print(expr))
        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _hprint_variadic_function(self, expr, exp=None):
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % (self._print((str(expr.func)).lower()), ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = r"\lfloor{%s}\rfloor" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\lceil{%s}\rceil" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings["ln_notation"]:
            tex = r"\log{\left (%s \right )}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left (%s \right )}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = r"\left|{%s}\right|" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
    _print_Determinant = _print_Abs

    def _print_re(self, expr, exp=None):
        tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Func'])

        return self._do_exponent(tex, exp)
```
### 22 - sympy/printing/latex.py:

Start line: 2127, End line: 2191

```python
class LatexPrinter(Printer):

    def _print_FreeModule(self, M):
        return '{%s}^{%s}' % (self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ %s \right]" % ",".join(
            '{' + self._print(x) + '}' for x in m)

    def _print_SubModule(self, m):
        return r"\left< %s \right>" % ",".join(
            '{' + self._print(x) + '}' for x in m.gens)

    def _print_ModuleImplementedIdeal(self, m):
        return r"\left< %s \right>" % ",".join(
            '{' + self._print(x) + '}' for [x] in m._module.gens)

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{%s}{%s}" % (self._print(R.ring), self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return r"{%s} + {%s}" % (self._print(x.data), self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return r"{%s} + {%s}" % (self._print(m.data),
                                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{%s}{%s}" % (self._print(M.base),
                                   self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{%s} : {%s} \to {%s}" % (self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_BaseScalarField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\boldsymbol{\mathrm{%s}}' % self._print(Symbol(string))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys._names[field._index]
        return r'\partial_{%s}' % self._print(Symbol(string))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys._names[field._index]
            return r'\mathrm{d}%s' % self._print(Symbol(string))
        else:
            return 'd(%s)' % self._print(field)
            string = self._print(field)
            return r'\mathrm{d}\left(%s\right)' % string

    def _print_Tr(self, p):
        #Todo: Handle indices
        contents = self._print(p.args[0])
        return r'\mbox{Tr}\left(%s\right)' % (contents)
```
### 23 - sympy/printing/latex.py:

Start line: 612, End line: 641

```python
class LatexPrinter(Printer):

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)
        tex = '{'+tex_base+'}'+'_{%s}' % ','.join(
            map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Derivative(self, expr):
        if requires_partial(expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol, self._print(x), num)

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, dim, tex)

        return r"%s %s" % (tex, self.parenthesize(expr.expr, PRECEDENCE["Mul"], strict=True))
```
### 24 - sympy/printing/latex.py:

Start line: 1187, End line: 1196

```python
class LatexPrinter(Printer):

    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, self._print(exp))
        return tex
```
### 25 - sympy/printing/latex.py:

Start line: 1120, End line: 1185

```python
class LatexPrinter(Printer):

    def _hprint_vec(self, vec):
        if len(vec) == 0:
            return ""
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation=""):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation=""):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')
```
### 27 - sympy/printing/latex.py:

Start line: 2017, End line: 2060

```python
class LatexPrinter(Printer):

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        return tex

    def _print_MellinTransform(self, expr):
        return r"\mathcal{M}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseMellinTransform(self, expr):
        return r"\mathcal{M}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_LaplaceTransform(self, expr):
        return r"\mathcal{L}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseLaplaceTransform(self, expr):
        return r"\mathcal{L}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_FourierTransform(self, expr):
        return r"\mathcal{F}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseFourierTransform(self, expr):
        return r"\mathcal{F}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_SineTransform(self, expr):
        return r"\mathcal{SIN}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseSineTransform(self, expr):
        return r"\mathcal{SIN}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_CosineTransform(self, expr):
        return r"\mathcal{COS}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_InverseCosineTransform(self, expr):
        return r"\mathcal{COS}^{-1}_{%s}\left[%s\right]\left(%s\right)" % (self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))
```
### 28 - sympy/printing/latex.py:

Start line: 1267, End line: 1327

```python
class LatexPrinter(Printer):

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            if self._settings['fold_short_frac']:
                return r"%s%d / %d" % (sign, p, expr.q)
            return r"%s\frac{%d}{%d}" % (sign, p, expr.q)
        else:
            return self._print(expr.p)
```
### 31 - sympy/printing/latex.py:

Start line: 440, End line: 487

```python
class LatexPrinter(Printer):

    def _print_Mul(self, expr):
        # ... other code

        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)

        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] \
                   and ldenom <= 2 and not "^" in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            elif ratio is not None and \
                    len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"\frac{1}{%s}%s\left(%s\right)" \
                        % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    # split a long numerator
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or \
                                len(convert(a*x).split()) > ratio*ldenom or \
                                (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"\frac{%s}{%s}%s\left(%s\right)" \
                            % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += r"\frac{%s}{%s}%s%s" \
                            % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += r"\frac{1}{%s}%s%s" % (sdenom, separator, snumer)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex
```
### 32 - sympy/printing/latex.py:

Start line: 715, End line: 782

```python
class LatexPrinter(Printer):

    def _print_Function(self, expr, exp=None):
        r'''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and \
            not isinstance(expr.func, UndefinedFunction):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [ str(self._print(arg)) for arg in expr.args ]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = ["asin", "acos", "atan", "acsc", "asec", "acot"]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    func = func
                elif inv_trig_style == "full":
                    func = "arc" + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        can_fold_brackets = False

            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = r"\%s^{-1}" % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func
            elif exp is not None:
                name = r'%s^{%s}' % (self._hprint_Function(func), exp)
            else:
                name = self._hprint_Function(func)

            if can_fold_brackets:
                if func in accepted_latex_functions:
                    # Wrap argument safely to avoid parse-time conflicts
                    # with the function name itself
                    name += r" {%s}"
                else:
                    name += r"%s"
            else:
                name += r"{\left (%s \right )}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)
```
### 33 - sympy/printing/latex.py:

Start line: 652, End line: 682

```python
class LatexPrinter(Printer):

    def _print_Integral(self, expr):
        tex, symbols = "", []

        # Only up to \iiiint exists
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            symbols = [r"\, d%s" % self._print(symbol[0])
                       for symbol in expr.limits]

        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"

                if len(lim) > 1:
                    if self._settings['mode'] in ['equation', 'equation*'] \
                            and not self._settings['itex']:
                        tex += r"\limits"

                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]),
                                               self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % (self._print(lim[1]))

                symbols.insert(0, r"\, d%s" % self._print(symbol))

        return r"%s %s%s" % (tex,
            self.parenthesize(expr.function, PRECEDENCE["Mul"], strict=True), "".join(symbols))
```
### 34 - sympy/printing/latex.py:

Start line: 1622, End line: 1687

```python
class LatexPrinter(Printer):

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices)

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )
        return self._print(expr.args[0])

    def _print_tuple(self, expr):
        return r"\left ( %s\right )" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        return r"\left [ %s\right ]" % \
            r", \quad ".join([ self._print(i) for i in expr ])

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left \{ %s\right \}" % r", \quad ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)
```
### 35 - sympy/printing/latex.py:

Start line: 389, End line: 438

```python
class LatexPrinter(Printer):

    def _print_Mul(self, expr):
        from sympy.core.power import Pow
        from sympy.physics.units import Quantity
        include_parens = False
        if _coeff_isneg(expr):
            expr = -expr
            tex = "- "
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        from sympy.simplify import fraction
        numer, denom = fraction(expr, exact=True)
        separator = self._settings['mul_symbol_latex']
        numbersep = self._settings['mul_symbol_latex_numbers']

        def convert(expr):
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                _tex = last_term_tex = ""

                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)

                # If quantities are present append them at the back
                args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                             (isinstance(x, Pow) and isinstance(x.base, Quantity)))

                for i, term in enumerate(args):
                    term_tex = self._print(term)

                    if self._needs_mul_brackets(term, first=(i == 0),
                                                last=(i == len(args) - 1)):
                        term_tex = r"\left(%s\right)" % term_tex

                    if _between_two_numbers_p[0].search(last_term_tex) and \
                            _between_two_numbers_p[1].match(term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator

                    _tex += term_tex
                    last_term_tex = term_tex
                return _tex
        # ... other code
```
### 36 - sympy/printing/latex.py:

Start line: 903, End line: 958

```python
class LatexPrinter(Printer):

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\wedge")

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\vee")

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\veebar")

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or r"\Rightarrow")

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    def _print_conjugate(self, expr, exp=None):
        tex = r"\overline{%s}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left (%s \right )}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (func, exp, arg)
        else:
            return r"%s%s" % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        tex = r"e^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex
```
### 37 - sympy/printing/latex.py:

Start line: 1329, End line: 1343

```python
class LatexPrinter(Printer):

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif len(expr.variables):
                s += self._print(expr.variables[0])
            s += r'\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return r"O\left(%s\right)" % s
```
### 38 - sympy/printing/latex.py:

Start line: 1502, End line: 1507

```python
class LatexPrinter(Printer):

    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s\bmod{%s}\right)^{%s}' % (self.parenthesize(expr.args[0],
                    PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]), self._print(exp))
        return r'%s\bmod{%s}' % (self.parenthesize(expr.args[0],
                PRECEDENCE['Mul'], strict=True), self._print(expr.args[1]))
```
### 40 - sympy/printing/latex.py:

Start line: 324, End line: 340

```python
class LatexPrinter(Printer):

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    _print_Permutation = _print_Cycle
```
### 41 - sympy/printing/latex.py:

Start line: 1711, End line: 1720

```python
class LatexPrinter(Printer):

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex
```
### 43 - sympy/printing/latex.py:

Start line: 1984, End line: 2006

```python
class LatexPrinter(Printer):

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls, ", ".join(args))

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)
```
### 44 - sympy/printing/latex.py:

Start line: 2008, End line: 2015

```python
class LatexPrinter(Printer):

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        if exp is not None:
            tex = r"%s^{%s}" % (tex, self._print(exp))
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex
```
### 46 - sympy/printing/latex.py:

Start line: 1699, End line: 1709

```python
class LatexPrinter(Printer):

    def _print_SingularityFunction(self, expr):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\langle %s \rangle}^{%s}" % (shift, power)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        tex = r"\theta\left(%s\right)" % self._print(expr.args[0])
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 48 - sympy/printing/latex.py:

Start line: 1393, End line: 1404

```python
class LatexPrinter(Printer):

    def _print_Piecewise(self, expr):
        ecpairs = [r"%s & \text{for}\: %s" % (self._print(e), self._print(c))
                   for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            ecpairs.append(r"%s & \text{otherwise}" %
                           self._print(expr.args[-1].expr))
        else:
            ecpairs.append(r"%s & \text{for}\: %s" %
                           (self._print(expr.args[-1].expr),
                            self._print(expr.args[-1].cond)))
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)
```
### 51 - sympy/printing/latex.py:

Start line: 984, End line: 1068

```python
class LatexPrinter(Printer):

    def _print_beta(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    def _hprint_one_arg_func(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
```
### 52 - sympy/printing/latex.py:

Start line: 563, End line: 580

```python
class LatexPrinter(Printer):

    def _print_Product(self, expr):
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 53 - sympy/printing/latex.py:

Start line: 541, End line: 561

```python
class LatexPrinter(Printer):

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex
```
### 54 - sympy/printing/latex.py:

Start line: 1722, End line: 1758

```python
class LatexPrinter(Printer):

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_ProductSet(self, p):
        if len(p.sets) > 1 and not has_variety(p.sets):
            return self._print(p.sets[0]) + "^%d" % len(p.sets)
        else:
            return r" \times ".join(self._print(set) for set in p.sets)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return 'Domain on ' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        items = ", ".join(map(self._print, items))
        return r"\left\{%s\right\}" % items

    _print_frozenset = _print_set
```
### 55 - sympy/printing/latex.py:

Start line: 960, End line: 969

```python
class LatexPrinter(Printer):

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex
```
### 56 - sympy/printing/latex.py:

Start line: 2193, End line: 2203

```python
class LatexPrinter(Printer):

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])
```
### 57 - sympy/printing/latex.py:

Start line: 784, End line: 799

```python
class LatexPrinter(Printer):

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}
```
### 58 - sympy/printing/latex.py:

Start line: 1867, End line: 1914

```python
class LatexPrinter(Printer):

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return r"\left\{%s\; |\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        return self._print_Add(s.truncate()) + self._print(r' + \ldots')

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left[%s\right]" % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left(%s\right)" % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        inv = ""
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)
```
### 59 - sympy/printing/latex.py:

Start line: 1103, End line: 1118

```python
class LatexPrinter(Printer):

    def _hprint_BesselBase(self, expr, exp, sym):
        tex = r"%s" % (sym)

        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, self._print(exp))
            else:
                need_exp = True

        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex
```
### 60 - sympy/printing/latex.py:

Start line: 302, End line: 322

```python
class LatexPrinter(Printer):


    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)

        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif _coeff_isneg(term):
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex

        return tex
```
### 63 - sympy/printing/latex.py:

Start line: 2225, End line: 2235

```python
class LatexPrinter(Printer):

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % (self._print(expr.args[0]),
                    self._print(exp))
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])
```
### 64 - sympy/printing/latex.py:

Start line: 1070, End line: 1101

```python
class LatexPrinter(Printer):

    def _print_factorial2(self, expr, exp=None):
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_binomial(self, expr, exp=None):
        tex = r"{\binom{%s}{%s}}" % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        return self._do_exponent(tex, exp)
```
### 65 - sympy/printing/latex.py:

Start line: 582, End line: 610

```python
class LatexPrinter(Printer):

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        o1 = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x:x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = '(' + LatexPrinter().doprint(v) + ')'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr
```
### 66 - sympy/printing/latex.py:

Start line: 1590, End line: 1620

```python
class LatexPrinter(Printer):

    _print_ImmutableDenseNDimArray = _print_NDimArray
    _print_ImmutableSparseNDimArray = _print_NDimArray
    _print_MutableDenseNDimArray = _print_NDimArray
    _print_MutableSparseNDimArray = _print_NDimArray

    def _printer_tensor_indices(self, name, indices, index_map={}):
        out_str = self._print(name)
        last_valence = None
        prev_map = None
        for index in indices:
            new_valence = index.is_up
            if ((index in index_map) or prev_map) and last_valence == new_valence:
                out_str += ","
            if last_valence != new_valence:
                if last_valence is not None:
                    out_str += "}"
                if index.is_up:
                    out_str += "{}^{"
                else:
                    out_str += "{}_{"
            out_str += self._print(index.args[0])
            if index in index_map:
                out_str += "="
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            last_valence = new_valence
        if last_valence is not None:
            out_str += "}"
        return out_str
```
### 67 - sympy/printing/latex.py:

Start line: 1373, End line: 1391

```python
class LatexPrinter(Printer):

    def _print_Relational(self, expr):
        if self._settings['itex']:
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
        }

        return "%s %s %s" % (self._print(expr.lhs),
            charmap[expr.rel_op], self._print(expr.rhs))
```
### 68 - sympy/printing/latex.py:

Start line: 1790, End line: 1852

```python
class LatexPrinter(Printer):

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return r"\left%s%s, %s\right%s" % \
                   (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        return r"\langle %s, %s\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        return r" \cup ".join([self._print(i) for i in u.args])

    def _print_Complement(self, u):
        return r" \setminus ".join([self._print(i) for i in u.args])

    def _print_Intersection(self, u):
        return r" \cap ".join([self._print(i) for i in u.args])

    def _print_SymmetricDifference(self, u):
        return r" \triangle ".join([self._print(i) for i in u.args])

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"

    def _print_ImageSet(self, s):
        sets = s.args[1:]
        varsets = [r"%s \in %s" % (self._print(var), self._print(setv))
            for var, setv in zip(s.lamda.variables, sets)]
        return r"\left\{%s\; |\; %s\right\}" % (
            self._print(s.lamda.expr),
            ', '.join(varsets))
```
### 70 - sympy/printing/latex.py:

Start line: 1689, End line: 1697

```python
class LatexPrinter(Printer):

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex
```
### 71 - sympy/printing/latex.py:

Start line: 684, End line: 696

```python
class LatexPrinter(Printer):

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))
```
### 73 - sympy/printing/latex.py:

Start line: 1211, End line: 1265

```python
class LatexPrinter(Printer):

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (self._print(exp), tex)
        return r"\eta%s" % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\zeta^{%s}%s" % (self._print(exp), tex)
        return r"\zeta%s" % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        if exp is None:
            return r"\Phi%s" % tex
        return r"\Phi^{%s}%s" % (self._print(exp), tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, self._print(exp), tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (self._print(exp))
        return tex
```
### 74 - sympy/printing/latex.py:

Start line: 971, End line: 982

```python
class LatexPrinter(Printer):

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]), \
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex
```
### 79 - sympy/printing/latex.py:

Start line: 877, End line: 886

```python
class LatexPrinter(Printer):

    def _print_Not(self, e):
        from sympy import Equivalent, Implies
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg (%s)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])
```
### 80 - sympy/printing/latex.py:

Start line: 1973, End line: 1982

```python
class LatexPrinter(Printer):

    def _print_ComplexRootOf(self, root):
        cls = root.__class__.__name__
        if cls == "ComplexRootOf":
            cls = "CRootOf"
        expr = self._print(root.expr)
        index = root.index
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s, %d\right)}" % (cls, expr, index)
        else:
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr, index)
```
### 81 - sympy/printing/latex.py:

Start line: 342, End line: 363

```python
class LatexPrinter(Printer):

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"\infty"
        elif str_real == "-inf":
            return r"- \infty"
        else:
            return str_real
```
### 82 - sympy/printing/latex.py:

Start line: 2215, End line: 2223

```python
class LatexPrinter(Printer):

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (self._print(exp), tex)
        return r"\sigma^*%s" % tex
```
### 83 - sympy/printing/latex.py:

Start line: 2205, End line: 2213

```python
class LatexPrinter(Printer):

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^{%s}%s" % (self._print(exp), tex)
        return r"\sigma%s" % tex
```
### 84 - sympy/printing/latex.py:

Start line: 2062, End line: 2107

```python
class LatexPrinter(Printer):

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return "%s:%s" % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):
        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return "\\mathbf{%s}" % self._print(Symbol(morphism.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        return latex_result
```
### 88 - sympy/printing/latex.py:

Start line: 888, End line: 901

```python
class LatexPrinter(Printer):

    def _print_LogOp(self, args, char):
        arg = args[0]
        if arg.is_Boolean and not arg.is_Not:
            tex = r"\left(%s\right)" % self._print(arg)
        else:
            tex = r"%s" % self._print(arg)

        for arg in args[1:]:
            if arg.is_Boolean and not arg.is_Not:
                tex += r" %s \left(%s\right)" % (char, self._print(arg))
            else:
                tex += r" %s %s" % (char, self._print(arg))

        return tex
```
### 89 - sympy/printing/latex.py:

Start line: 187, End line: 208

```python
class LatexPrinter(Printer):

    def doprint(self, expr):
        tex = Printer.doprint(self, expr)

        if self._settings['mode'] == 'plain':
            return tex
        elif self._settings['mode'] == 'inline':
            return r"$%s$" % tex
        elif self._settings['itex']:
            return r"$$%s$$" % tex
        else:
            env_str = self._settings['mode']
            return r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)

    def _needs_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))
```
### 90 - sympy/printing/latex.py:

Start line: 2238, End line: 2451

```python
def translate(s):
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # Process the rest
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    elif s in other_symbols:
        return "\\" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True):
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s


def latex(expr, fold_frac_powers=False, fold_func_brackets=False,
    fold_short_frac=None, inv_trig_style="abbreviated",
    itex=False, ln_notation=False, long_frac_ratio=None,
    mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
    order=None, symbol_names=None):
    # ... other code
```
### 93 - sympy/printing/latex.py:

Start line: 1916, End line: 1971

```python
class LatexPrinter(Printer):

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = r"\left(%s\right)" % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ['-', '+']:
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = "domain=%s" % self._print(poly.get_domain())

        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = r"\%s {\left (%s \right )}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        return tex
```
### 94 - sympy/printing/latex.py:

Start line: 1775, End line: 1788

```python
class LatexPrinter(Printer):

    def _print_SeqFormula(self, s):
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (r'\ldots', s.coeff(stop - 3), s.coeff(stop - 2),
                s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(r'\ldots')
        else:
            printset = tuple(s)

        return (r"\left["
              + r", ".join(self._print(el) for el in printset)
              + r"\right]")
```
### 101 - sympy/printing/latex.py:

Start line: 2109, End line: 2125

```python
class LatexPrinter(Printer):

    def _print_DiagramGrid(self, grid):
        latex_result = "\\begin{array}{%s}\n" % ("c" * grid.width)

        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    latex_result += latex(grid[i, j])
                latex_result += " "
                if j != grid.width - 1:
                    latex_result += "& "

            if i != grid.height - 1:
                latex_result += "\\\\"
            latex_result += "\n"

        latex_result += "\\end{array}\n"
        return latex_result
```
### 102 - sympy/printing/latex.py:

Start line: 121, End line: 185

```python
class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings = {
        "order": None,
        "mode": "plain",
        "itex": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "long_frac_ratio": None,
        "mul_symbol": None,
        "inv_trig_style": "abbreviated",
        "mat_str": None,
        "mat_delim": "[",
        "symbol_names": {},
        "ln_notation": False,
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation',
                           'equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', "
                    "'equation' or 'equation*'")

        if self._settings['fold_short_frac'] is None and \
                self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        mul_symbol_table = {
            None: r" ",
            "ldot": r" \,.\, ",
            "dot": r" \cdot ",
            "times": r" \times "
        }
        try:
            self._settings['mul_symbol_latex'] = \
                mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            self._settings['mul_symbol_latex'] = \
                self._settings['mul_symbol']
        try:
            self._settings['mul_symbol_latex_numbers'] = \
                mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            if (self._settings['mul_symbol'].strip() in
                    ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']):
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol']

        self._delim_dict = {'(': ')', '[': ']'}

    def parenthesize(self, item, level, strict=False):
        prec_val = precedence_traditional(item)
        if (prec_val < level) or ((not strict) and prec_val <= level):
            return r"\left(%s\right)" % self._print(item)
        else:
            return self._print(item)
```
### 103 - sympy/printing/latex.py:

Start line: 698, End line: 713

```python
class LatexPrinter(Printer):

    def _hprint_Function(self, func):
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        func = self._deal_with_super_sub(func)
        if func in accepted_latex_functions:
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\'):
            name = func
        else:
            name = r"\operatorname{%s}" % func
        return name
```
### 107 - sympy/printing/latex.py:

Start line: 643, End line: 650

```python
class LatexPrinter(Printer):

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr, latex_subs)
```
### 112 - sympy/printing/latex.py:

Start line: 1854, End line: 1865

```python
class LatexPrinter(Printer):

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        if s.base_set is S.UniversalSet:
            return r"\left\{%s \mid %s \right\}" % (
            vars_print,
            self._print(s.condition.as_expr()))

        return r"\left\{%s \mid %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition.as_expr()))
```
### 114 - sympy/printing/latex.py:

Start line: 83, End line: 118

```python
modifier_dict = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
    'ddot': lambda s: r'\ddot{'+s+r'}',
    'dot': lambda s: r'\dot{'+s+r'}',
    'check': lambda s: r'\check{'+s+r'}',
    'breve': lambda s: r'\breve{'+s+r'}',
    'acute': lambda s: r'\acute{'+s+r'}',
    'grave': lambda s: r'\grave{'+s+r'}',
    'tilde': lambda s: r'\tilde{'+s+r'}',
    'hat': lambda s: r'\hat{'+s+r'}',
    'bar': lambda s: r'\bar{'+s+r'}',
    'vec': lambda s: r'\vec{'+s+r'}',
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'\boldsymbol{'+s+r'}',
    'bm': lambda s: r'\boldsymbol{'+s+r'}',
    'cal': lambda s: r'\mathcal{'+s+r'}',
    'scr': lambda s: r'\mathscr{'+s+r'}',
    'frak': lambda s: r'\mathfrak{'+s+r'}',
    # Brackets
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',
    'abs': lambda s: r'\left|{'+s+r'}\right|',
    'mag': lambda s: r'\left|{'+s+r'}\right|',
}

greek_letters_set = frozenset(greeks)

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # search
    re.compile(r'[{ ]*[-+0-9]'),  # match
)
```
### 123 - sympy/printing/latex.py:

Start line: 1760, End line: 1773

```python
class LatexPrinter(Printer):

    def _print_Range(self, s):
        dots = r'\ldots'

        if s.start.is_infinite:
            printset = s.start, dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite or len(s) > 4:
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            printset = tuple(s)

        return (r"\left\{"
              + r", ".join(self._print(el) for el in printset)
              + r"\right\}")
```
### 124 - sympy/printing/latex.py:

Start line: 262, End line: 299

```python
class LatexPrinter(Printer):


    def _needs_add_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of an Add, False otherwise.  This is False for most
        things.
        """
        if expr.is_Relational:
            return True
        if any([expr.has(x) for x in (Mod,)]):
            return True
        if expr.is_Add:
            return True
        return False


    def _mul_is_clean(self, expr):
        for arg in expr.args:
            if arg.is_Function:
                return False
        return True

    def _pow_is_clean(self, expr):
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr, exp):
        if exp is not None:
            return r"\left(%s\right)^{%s}" % (expr, exp)
        else:
            return expr

    def _print_bool(self, e):
        return r"\mathrm{%s}" % e

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        return r"\mathrm{%s}" % e
```
### 167 - sympy/printing/latex.py:

Start line: 210, End line: 231

```python
class LatexPrinter(Printer):

    def _needs_function_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        if not self._needs_brackets(expr):
            return False
        else:
            # Muls of the form a*b*c... can be folded
            if expr.is_Mul and not self._mul_is_clean(expr):
                return True
            # Pows which don't need brackets can be folded
            elif expr.is_Pow and not self._pow_is_clean(expr):
                return True
            # Add and Function always need brackets
            elif expr.is_Add or expr.is_Function:
                return True
            else:
                return False
```
