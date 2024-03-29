# sphinx-doc__sphinx-10451

| **sphinx-doc/sphinx** | `195e911f1dab04b8ddeacbe04b7d214aaf81bb0b` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 19855 |
| **Avg pos** | 41.0 |
| **Min pos** | 41 |
| **Max pos** | 41 |
| **Top file pos** | 9 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/typehints.py b/sphinx/ext/autodoc/typehints.py
--- a/sphinx/ext/autodoc/typehints.py
+++ b/sphinx/ext/autodoc/typehints.py
@@ -115,7 +115,15 @@ def modify_field_list(node: nodes.field_list, annotations: Dict[str, str],
         if name == 'return':
             continue
 
-        arg = arguments.get(name, {})
+        if '*' + name in arguments:
+            name = '*' + name
+            arguments.get(name)
+        elif '**' + name in arguments:
+            name = '**' + name
+            arguments.get(name)
+        else:
+            arg = arguments.get(name, {})
+
         if not arg.get('type'):
             field = nodes.field()
             field += nodes.field_name('', 'type ' + name)
@@ -167,13 +175,19 @@ def augment_descriptions_with_types(
             has_type.add('return')
 
     # Add 'type' for parameters with a description but no declared type.
-    for name in annotations:
+    for name, annotation in annotations.items():
         if name in ('return', 'returns'):
             continue
+
+        if '*' + name in has_description:
+            name = '*' + name
+        elif '**' + name in has_description:
+            name = '**' + name
+
         if name in has_description and name not in has_type:
             field = nodes.field()
             field += nodes.field_name('', 'type ' + name)
-            field += nodes.field_body('', nodes.paragraph('', annotations[name]))
+            field += nodes.field_body('', nodes.paragraph('', annotation))
             node += field
 
     # Add 'rtype' if 'return' is present and 'rtype' isn't.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/typehints.py | 118 | 118 | - | 9 | -
| sphinx/ext/autodoc/typehints.py | 170 | 176 | 41 | 9 | 19855


## Problem Statement

```
Fix duplicated *args and **kwargs with autodoc_typehints
Fix duplicated *args and **kwargs with autodoc_typehints

### Bugfix
- Bugfix

### Detail
Consider this
\`\`\`python
class _ClassWithDocumentedInitAndStarArgs:
    """Class docstring."""

    def __init__(self, x: int, *args: int, **kwargs: int) -> None:
        """Init docstring.

        :param x: Some integer
        :param *args: Some integer
        :param **kwargs: Some integer
        """
\`\`\`
when using the autodoc extension and the setting `autodoc_typehints = "description"`.

WIth sphinx 4.2.0, the current output is
\`\`\`
Class docstring.

   Parameters:
      * **x** (*int*) --

      * **args** (*int*) --

      * **kwargs** (*int*) --

   Return type:
      None

   __init__(x, *args, **kwargs)

      Init docstring.

      Parameters:
         * **x** (*int*) -- Some integer

         * ***args** --

           Some integer

         * ****kwargs** --

           Some integer

         * **args** (*int*) --

         * **kwargs** (*int*) --

      Return type:
         None
\`\`\`
where the *args and **kwargs are duplicated and incomplete.

The expected output is
\`\`\`
  Class docstring.

   Parameters:
      * **x** (*int*) --

      * ***args** (*int*) --

      * ****kwargs** (*int*) --

   Return type:
      None

   __init__(x, *args, **kwargs)

      Init docstring.

      Parameters:
         * **x** (*int*) -- Some integer

         * ***args** (*int*) --

           Some integer

         * ****kwargs** (*int*) --

           Some integer

      Return type:
         None

\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/__init__.py | 1411 | 1790| 3400 | 3400 | 23659 | 
| 2 | 2 doc/usage/extensions/example_numpy.py | 221 | 314| 589 | 3989 | 25739 | 
| 3 | 3 doc/usage/extensions/example_google.py | 176 | 255| 563 | 4552 | 27824 | 
| 4 | 3 sphinx/ext/autodoc/__init__.py | 2663 | 2687| 295 | 4847 | 27824 | 
| 5 | 3 sphinx/ext/autodoc/__init__.py | 1216 | 1232| 177 | 5024 | 27824 | 
| 6 | 3 sphinx/ext/autodoc/__init__.py | 2099 | 2309| 1904 | 6928 | 27824 | 
| 7 | 3 sphinx/ext/autodoc/__init__.py | 2616 | 2644| 332 | 7260 | 27824 | 
| 8 | 3 sphinx/ext/autodoc/__init__.py | 2011 | 2040| 305 | 7565 | 27824 | 
| 9 | 3 sphinx/ext/autodoc/__init__.py | 682 | 792| 907 | 8472 | 27824 | 
| 10 | 3 sphinx/ext/autodoc/__init__.py | 558 | 570| 139 | 8611 | 27824 | 
| 11 | 3 sphinx/ext/autodoc/__init__.py | 1380 | 1790| 188 | 8799 | 27824 | 
| 12 | 3 sphinx/ext/autodoc/__init__.py | 1 | 100| 748 | 9547 | 27824 | 
| 13 | 3 sphinx/ext/autodoc/__init__.py | 274 | 350| 751 | 10298 | 27824 | 
| 14 | 3 sphinx/ext/autodoc/__init__.py | 2591 | 2614| 229 | 10527 | 27824 | 
| 15 | 3 sphinx/ext/autodoc/__init__.py | 2797 | 2837| 535 | 11062 | 27824 | 
| 16 | 4 sphinx/application.py | 1108 | 1129| 263 | 11325 | 39633 | 
| 17 | 4 sphinx/ext/autodoc/__init__.py | 2042 | 2078| 296 | 11621 | 39633 | 
| 18 | 4 sphinx/ext/autodoc/__init__.py | 1793 | 1828| 245 | 11866 | 39633 | 
| 19 | 4 sphinx/ext/autodoc/__init__.py | 2646 | 2661| 182 | 12048 | 39633 | 
| 20 | 4 sphinx/ext/autodoc/__init__.py | 1124 | 1154| 287 | 12335 | 39633 | 
| 21 | 4 sphinx/ext/autodoc/__init__.py | 2372 | 2395| 203 | 12538 | 39633 | 
| 22 | 5 sphinx/ext/napoleon/docstring.py | 1 | 59| 603 | 13141 | 50622 | 
| 23 | 5 sphinx/ext/autodoc/__init__.py | 1253 | 1377| 1051 | 14192 | 50622 | 
| 24 | 5 sphinx/ext/autodoc/__init__.py | 1057 | 1081| 209 | 14401 | 50622 | 
| 25 | 5 sphinx/ext/autodoc/__init__.py | 2690 | 2767| 655 | 15056 | 50622 | 
| 26 | 5 sphinx/ext/napoleon/docstring.py | 818 | 848| 240 | 15296 | 50622 | 
| 27 | 5 sphinx/ext/autodoc/__init__.py | 1962 | 2009| 365 | 15661 | 50622 | 
| 28 | 5 sphinx/ext/autodoc/__init__.py | 2569 | 2589| 231 | 15892 | 50622 | 
| 29 | 5 sphinx/ext/autodoc/__init__.py | 1157 | 1214| 431 | 16323 | 50622 | 
| 30 | 6 sphinx/ext/autodoc/directive.py | 111 | 161| 453 | 16776 | 51979 | 
| 31 | 6 sphinx/ext/autodoc/__init__.py | 864 | 955| 857 | 17633 | 51979 | 
| 32 | 6 sphinx/ext/autodoc/__init__.py | 1017 | 1028| 126 | 17759 | 51979 | 
| 33 | 6 sphinx/ext/autodoc/__init__.py | 543 | 556| 131 | 17890 | 51979 | 
| 34 | 6 doc/usage/extensions/example_google.py | 294 | 311| 125 | 18015 | 51979 | 
| 35 | 6 sphinx/ext/autodoc/__init__.py | 1003 | 1015| 124 | 18139 | 51979 | 
| 36 | 7 sphinx/directives/__init__.py | 41 | 63| 187 | 18326 | 54124 | 
| 37 | 7 sphinx/ext/autodoc/__init__.py | 1896 | 1918| 225 | 18551 | 54124 | 
| 38 | 7 sphinx/ext/autodoc/__init__.py | 794 | 837| 451 | 19002 | 54124 | 
| 39 | 8 doc/development/tutorials/examples/autodoc_intenum.py | 30 | 55| 199 | 19201 | 54512 | 
| 40 | 8 sphinx/ext/autodoc/__init__.py | 1235 | 1250| 166 | 19367 | 54512 | 
| **-> 41 <-** | **9 sphinx/ext/autodoc/typehints.py** | 141 | 199| 488 | 19855 | 56087 | 
| 42 | 9 sphinx/ext/autodoc/__init__.py | 1083 | 1100| 182 | 20037 | 56087 | 
| 43 | 9 sphinx/ext/autodoc/__init__.py | 1921 | 1959| 300 | 20337 | 56087 | 
| 44 | 10 sphinx/util/inspect.py | 807 | 858| 402 | 20739 | 62669 | 
| 45 | 10 sphinx/ext/autodoc/__init__.py | 2533 | 2567| 291 | 21030 | 62669 | 
| 46 | 10 sphinx/ext/autodoc/__init__.py | 572 | 609| 355 | 21385 | 62669 | 
| 47 | 11 sphinx/ext/autodoc/mock.py | 65 | 90| 217 | 21602 | 64086 | 
| 48 | 11 sphinx/ext/autodoc/__init__.py | 423 | 471| 359 | 21961 | 64086 | 
| 49 | **11 sphinx/ext/autodoc/typehints.py** | 1 | 34| 256 | 22217 | 64086 | 
| 50 | 12 sphinx/ext/doctest.py | 58 | 144| 847 | 23064 | 69061 | 
| 51 | 12 sphinx/ext/autodoc/__init__.py | 1103 | 1121| 179 | 23243 | 69061 | 
| 52 | 12 sphinx/ext/autodoc/__init__.py | 1854 | 1874| 169 | 23412 | 69061 | 
| 53 | 12 sphinx/ext/autodoc/__init__.py | 2081 | 2309| 136 | 23548 | 69061 | 
| 54 | 12 sphinx/ext/autodoc/__init__.py | 2312 | 2340| 241 | 23789 | 69061 | 
| 55 | 12 sphinx/ext/napoleon/docstring.py | 850 | 863| 153 | 23942 | 69061 | 
| 56 | 12 doc/development/tutorials/examples/autodoc_intenum.py | 1 | 28| 198 | 24140 | 69061 | 
| 57 | 13 sphinx/ext/autosummary/generate.py | 455 | 472| 210 | 24350 | 74278 | 
| 58 | 13 sphinx/ext/napoleon/docstring.py | 470 | 511| 345 | 24695 | 74278 | 
| 59 | 13 sphinx/ext/autodoc/__init__.py | 2770 | 2794| 208 | 24903 | 74278 | 
| 60 | 13 sphinx/ext/autodoc/__init__.py | 2478 | 2495| 134 | 25037 | 74278 | 
| 61 | 13 sphinx/ext/autodoc/__init__.py | 958 | 1001| 387 | 25424 | 74278 | 
| 62 | 13 sphinx/ext/napoleon/docstring.py | 324 | 364| 329 | 25753 | 74278 | 
| 63 | 13 sphinx/ext/napoleon/docstring.py | 422 | 449| 258 | 26011 | 74278 | 
| 64 | 13 sphinx/ext/autodoc/mock.py | 1 | 62| 475 | 26486 | 74278 | 
| 65 | 13 doc/usage/extensions/example_google.py | 257 | 271| 109 | 26595 | 74278 | 
| 66 | 13 sphinx/ext/napoleon/docstring.py | 251 | 275| 254 | 26849 | 74278 | 
| 67 | 13 sphinx/ext/napoleon/docstring.py | 686 | 727| 418 | 27267 | 74278 | 
| 68 | 13 sphinx/ext/autodoc/__init__.py | 2422 | 2438| 151 | 27418 | 74278 | 
| 69 | 13 sphinx/ext/napoleon/docstring.py | 574 | 606| 265 | 27683 | 74278 | 
| 70 | 13 sphinx/ext/napoleon/docstring.py | 300 | 322| 199 | 27882 | 74278 | 
| 71 | 13 doc/usage/extensions/example_numpy.py | 316 | 330| 109 | 27991 | 74278 | 
| 72 | 14 sphinx/domains/cpp.py | 7133 | 7837| 5982 | 33973 | 142623 | 
| 73 | 14 sphinx/ext/autosummary/generate.py | 79 | 97| 210 | 34183 | 142623 | 
| 74 | 14 sphinx/ext/autodoc/__init__.py | 473 | 506| 286 | 34469 | 142623 | 
| 75 | 14 sphinx/ext/napoleon/docstring.py | 1159 | 1185| 259 | 34728 | 142623 | 
| 76 | 14 doc/usage/extensions/example_google.py | 149 | 173| 175 | 34903 | 142623 | 
| 77 | 14 sphinx/ext/napoleon/docstring.py | 1137 | 1157| 157 | 35060 | 142623 | 
| 78 | 14 sphinx/ext/autosummary/generate.py | 207 | 242| 343 | 35403 | 142623 | 
| 79 | 14 sphinx/ext/napoleon/docstring.py | 786 | 816| 283 | 35686 | 142623 | 
| 80 | 15 sphinx/domains/python.py | 827 | 864| 231 | 35917 | 155080 | 
| 81 | 15 doc/usage/extensions/example_numpy.py | 187 | 218| 188 | 36105 | 155080 | 
| 82 | 15 sphinx/ext/autodoc/__init__.py | 508 | 528| 235 | 36340 | 155080 | 
| 83 | 15 sphinx/ext/napoleon/docstring.py | 729 | 745| 195 | 36535 | 155080 | 
| 84 | 15 sphinx/ext/autodoc/__init__.py | 165 | 192| 278 | 36813 | 155080 | 
| 85 | 15 sphinx/ext/napoleon/docstring.py | 405 | 420| 180 | 36993 | 155080 | 
| 86 | 16 sphinx/pycode/parser.py | 74 | 116| 317 | 37310 | 159652 | 
| 87 | 16 sphinx/ext/napoleon/docstring.py | 379 | 403| 235 | 37545 | 159652 | 
| 88 | 16 sphinx/ext/autodoc/__init__.py | 1831 | 1851| 162 | 37707 | 159652 | 
| 89 | 17 sphinx/cmd/quickstart.py | 1 | 111| 767 | 38474 | 165185 | 
| 90 | 17 sphinx/ext/autodoc/directive.py | 1 | 39| 307 | 38781 | 165185 | 
| 91 | 18 sphinx/ext/autodoc/importer.py | 68 | 138| 649 | 39430 | 167421 | 
| 92 | 18 sphinx/ext/autodoc/__init__.py | 103 | 146| 300 | 39730 | 167421 | 
| 93 | 18 sphinx/ext/autodoc/__init__.py | 2398 | 2420| 171 | 39901 | 167421 | 
| 94 | 18 sphinx/ext/autodoc/__init__.py | 363 | 398| 317 | 40218 | 167421 | 
| 95 | 19 sphinx/util/docutils.py | 364 | 393| 229 | 40447 | 172121 | 
| 96 | 19 sphinx/ext/napoleon/docstring.py | 513 | 528| 130 | 40577 | 172121 | 
| 97 | 19 sphinx/ext/napoleon/docstring.py | 1038 | 1135| 745 | 41322 | 172121 | 
| 98 | 19 sphinx/util/docutils.py | 396 | 412| 181 | 41503 | 172121 | 
| 99 | 19 sphinx/ext/autodoc/__init__.py | 2343 | 2370| 198 | 41701 | 172121 | 
| 100 | 19 sphinx/domains/python.py | 734 | 761| 227 | 41928 | 172121 | 
| 101 | 19 doc/usage/extensions/example_google.py | 1 | 37| 254 | 42182 | 172121 | 
| 102 | 19 sphinx/ext/napoleon/docstring.py | 552 | 572| 233 | 42415 | 172121 | 
| 103 | 19 sphinx/ext/napoleon/docstring.py | 290 | 298| 121 | 42536 | 172121 | 
| 104 | 19 sphinx/ext/napoleon/docstring.py | 451 | 468| 178 | 42714 | 172121 | 
| 105 | 20 sphinx/util/typing.py | 259 | 330| 876 | 43590 | 177588 | 
| 106 | 20 sphinx/ext/napoleon/docstring.py | 646 | 671| 267 | 43857 | 177588 | 
| 107 | 20 sphinx/ext/napoleon/docstring.py | 366 | 377| 119 | 43976 | 177588 | 
| 108 | 21 sphinx/ext/napoleon/__init__.py | 345 | 392| 450 | 44426 | 181561 | 
| 109 | 21 sphinx/ext/autodoc/__init__.py | 530 | 541| 120 | 44546 | 181561 | 
| 110 | 21 sphinx/ext/napoleon/docstring.py | 760 | 784| 232 | 44778 | 181561 | 
| 111 | 21 sphinx/ext/napoleon/docstring.py | 620 | 644| 239 | 45017 | 181561 | 
| 112 | 21 sphinx/ext/napoleon/__init__.py | 395 | 478| 730 | 45747 | 181561 | 
| 113 | 21 sphinx/ext/napoleon/docstring.py | 62 | 126| 599 | 46346 | 181561 | 
| 114 | 21 sphinx/ext/autodoc/__init__.py | 1877 | 1894| 142 | 46488 | 181561 | 
| 115 | 21 sphinx/util/inspect.py | 530 | 561| 249 | 46737 | 181561 | 
| 116 | 21 sphinx/ext/napoleon/docstring.py | 1187 | 1224| 349 | 47086 | 181561 | 
| 117 | 21 sphinx/ext/autosummary/generate.py | 135 | 154| 176 | 47262 | 181561 | 


### Hint

```
I noticed this docstring causes warnings because `*` and `**` are considered as mark-up symbols:

\`\`\`
    def __init__(self, x: int, *args: int, **kwargs: int) -> None:
        """Init docstring.

        :param x: Some integer
        :param *args: Some integer
        :param **kwargs: Some integer
        """
\`\`\`

Here are warnings:
\`\`\`
/Users/tkomiya/work/tmp/doc/example.py:docstring of example.ClassWithDocumentedInitAndStarArgs:6: WARNING: Inline emphasis start-string without end-string.
/Users/tkomiya/work/tmp/doc/example.py:docstring of example.ClassWithDocumentedInitAndStarArgs:7: WARNING: Inline strong start-string without end-string.
\`\`\`

It will work fine if we escape `*` character like the following. But it's not officially recommended way, I believe.

\`\`\`
    def __init__(self, x: int, *args: int, **kwargs: int) -> None:
        """Init docstring.

        :param x: Some integer
        :param \*args: Some integer
        :param \*\*kwargs: Some integer
        """
\`\`\`

I'm not sure this feature is really needed?
> I noticed this docstring causes warnings because `*` and `**` are considered as mark-up symbols:
> 
> \`\`\`
>     def __init__(self, x: int, *args: int, **kwargs: int) -> None:
>         """Init docstring.
> 
>         :param x: Some integer
>         :param *args: Some integer
>         :param **kwargs: Some integer
>         """
> \`\`\`
> 
> Here are warnings:
> 
> \`\`\`
> /Users/tkomiya/work/tmp/doc/example.py:docstring of example.ClassWithDocumentedInitAndStarArgs:6: WARNING: Inline emphasis start-string without end-string.
> /Users/tkomiya/work/tmp/doc/example.py:docstring of example.ClassWithDocumentedInitAndStarArgs:7: WARNING: Inline strong start-string without end-string.
> \`\`\`
> 
> It will work fine if we escape `*` character like the following. But it's not officially recommended way, I believe.
> 
> \`\`\`
>     def __init__(self, x: int, *args: int, **kwargs: int) -> None:
>         """Init docstring.
> 
>         :param x: Some integer
>         :param \*args: Some integer
>         :param \*\*kwargs: Some integer
>         """
> \`\`\`
> 
> I'm not sure this feature is really needed?

This is needed for the Numpy and Google docstring formats, which napoleon converts to `:param:`s.

Oh, I missed numpydoc format. Indeed, it recommends prepending stars.
https://numpydoc.readthedocs.io/en/latest/format.html#parameters
```

## Patch

```diff
diff --git a/sphinx/ext/autodoc/typehints.py b/sphinx/ext/autodoc/typehints.py
--- a/sphinx/ext/autodoc/typehints.py
+++ b/sphinx/ext/autodoc/typehints.py
@@ -115,7 +115,15 @@ def modify_field_list(node: nodes.field_list, annotations: Dict[str, str],
         if name == 'return':
             continue
 
-        arg = arguments.get(name, {})
+        if '*' + name in arguments:
+            name = '*' + name
+            arguments.get(name)
+        elif '**' + name in arguments:
+            name = '**' + name
+            arguments.get(name)
+        else:
+            arg = arguments.get(name, {})
+
         if not arg.get('type'):
             field = nodes.field()
             field += nodes.field_name('', 'type ' + name)
@@ -167,13 +175,19 @@ def augment_descriptions_with_types(
             has_type.add('return')
 
     # Add 'type' for parameters with a description but no declared type.
-    for name in annotations:
+    for name, annotation in annotations.items():
         if name in ('return', 'returns'):
             continue
+
+        if '*' + name in has_description:
+            name = '*' + name
+        elif '**' + name in has_description:
+            name = '**' + name
+
         if name in has_description and name not in has_type:
             field = nodes.field()
             field += nodes.field_name('', 'type ' + name)
-            field += nodes.field_body('', nodes.paragraph('', annotations[name]))
+            field += nodes.field_body('', nodes.paragraph('', annotation))
             node += field
 
     # Add 'rtype' if 'return' is present and 'rtype' isn't.

```

## Test Patch

```diff
diff --git a/tests/roots/test-ext-autodoc/target/typehints.py b/tests/roots/test-ext-autodoc/target/typehints.py
--- a/tests/roots/test-ext-autodoc/target/typehints.py
+++ b/tests/roots/test-ext-autodoc/target/typehints.py
@@ -94,8 +94,10 @@ def missing_attr(c,
 class _ClassWithDocumentedInit:
     """Class docstring."""
 
-    def __init__(self, x: int) -> None:
+    def __init__(self, x: int, *args: int, **kwargs: int) -> None:
         """Init docstring.
 
         :param x: Some integer
+        :param args: Some integer
+        :param kwargs: Some integer
         """
diff --git a/tests/roots/test-ext-napoleon/conf.py b/tests/roots/test-ext-napoleon/conf.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-ext-napoleon/conf.py
@@ -0,0 +1,5 @@
+import os
+import sys
+
+sys.path.insert(0, os.path.abspath('.'))
+extensions = ['sphinx.ext.napoleon']
diff --git a/tests/roots/test-ext-napoleon/index.rst b/tests/roots/test-ext-napoleon/index.rst
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-ext-napoleon/index.rst
@@ -0,0 +1,6 @@
+test-ext-napoleon
+=================
+
+.. toctree::
+
+   typehints
diff --git a/tests/roots/test-ext-napoleon/mypackage/__init__.py b/tests/roots/test-ext-napoleon/mypackage/__init__.py
new file mode 100644
diff --git a/tests/roots/test-ext-napoleon/mypackage/typehints.py b/tests/roots/test-ext-napoleon/mypackage/typehints.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-ext-napoleon/mypackage/typehints.py
@@ -0,0 +1,11 @@
+def hello(x: int, *args: int, **kwargs: int) -> None:
+    """
+    Parameters
+    ----------
+    x
+        X
+    *args
+        Additional arguments.
+    **kwargs
+        Extra arguments.
+    """
diff --git a/tests/roots/test-ext-napoleon/typehints.rst b/tests/roots/test-ext-napoleon/typehints.rst
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-ext-napoleon/typehints.rst
@@ -0,0 +1,5 @@
+typehints
+=========
+
+.. automodule:: mypackage.typehints
+   :members:
diff --git a/tests/test_ext_autodoc_configs.py b/tests/test_ext_autodoc_configs.py
--- a/tests/test_ext_autodoc_configs.py
+++ b/tests/test_ext_autodoc_configs.py
@@ -1034,19 +1034,27 @@ def test_autodoc_typehints_description_with_documented_init(app):
     )
     app.build()
     context = (app.outdir / 'index.txt').read_text(encoding='utf8')
-    assert ('class target.typehints._ClassWithDocumentedInit(x)\n'
+    assert ('class target.typehints._ClassWithDocumentedInit(x, *args, **kwargs)\n'
             '\n'
             '   Class docstring.\n'
             '\n'
             '   Parameters:\n'
-            '      **x** (*int*) --\n'
+            '      * **x** (*int*) --\n'
             '\n'
-            '   __init__(x)\n'
+            '      * **args** (*int*) --\n'
+            '\n'
+            '      * **kwargs** (*int*) --\n'
+            '\n'
+            '   __init__(x, *args, **kwargs)\n'
             '\n'
             '      Init docstring.\n'
             '\n'
             '      Parameters:\n'
-            '         **x** (*int*) -- Some integer\n'
+            '         * **x** (*int*) -- Some integer\n'
+            '\n'
+            '         * **args** (*int*) -- Some integer\n'
+            '\n'
+            '         * **kwargs** (*int*) -- Some integer\n'
             '\n'
             '      Return type:\n'
             '         None\n' == context)
@@ -1063,16 +1071,20 @@ def test_autodoc_typehints_description_with_documented_init_no_undoc(app):
     )
     app.build()
     context = (app.outdir / 'index.txt').read_text(encoding='utf8')
-    assert ('class target.typehints._ClassWithDocumentedInit(x)\n'
+    assert ('class target.typehints._ClassWithDocumentedInit(x, *args, **kwargs)\n'
             '\n'
             '   Class docstring.\n'
             '\n'
-            '   __init__(x)\n'
+            '   __init__(x, *args, **kwargs)\n'
             '\n'
             '      Init docstring.\n'
             '\n'
             '      Parameters:\n'
-            '         **x** (*int*) -- Some integer\n' == context)
+            '         * **x** (*int*) -- Some integer\n'
+            '\n'
+            '         * **args** (*int*) -- Some integer\n'
+            '\n'
+            '         * **kwargs** (*int*) -- Some integer\n' == context)
 
 
 @pytest.mark.sphinx('text', testroot='ext-autodoc',
@@ -1089,16 +1101,20 @@ def test_autodoc_typehints_description_with_documented_init_no_undoc_doc_rtype(a
     )
     app.build()
     context = (app.outdir / 'index.txt').read_text(encoding='utf8')
-    assert ('class target.typehints._ClassWithDocumentedInit(x)\n'
+    assert ('class target.typehints._ClassWithDocumentedInit(x, *args, **kwargs)\n'
             '\n'
             '   Class docstring.\n'
             '\n'
-            '   __init__(x)\n'
+            '   __init__(x, *args, **kwargs)\n'
             '\n'
             '      Init docstring.\n'
             '\n'
             '      Parameters:\n'
-            '         **x** (*int*) -- Some integer\n' == context)
+            '         * **x** (*int*) -- Some integer\n'
+            '\n'
+            '         * **args** (*int*) -- Some integer\n'
+            '\n'
+            '         * **kwargs** (*int*) -- Some integer\n' == context)
 
 
 @pytest.mark.sphinx('text', testroot='ext-autodoc',
diff --git a/tests/test_ext_napoleon_docstring.py b/tests/test_ext_napoleon_docstring.py
--- a/tests/test_ext_napoleon_docstring.py
+++ b/tests/test_ext_napoleon_docstring.py
@@ -2593,3 +2593,48 @@ def test_pep526_annotations(self):
 """
         print(actual)
         assert expected == actual
+
+
+@pytest.mark.sphinx('text', testroot='ext-napoleon',
+                    confoverrides={'autodoc_typehints': 'description',
+                                   'autodoc_typehints_description_target': 'all'})
+def test_napoleon_and_autodoc_typehints_description_all(app, status, warning):
+    app.build()
+    content = (app.outdir / 'typehints.txt').read_text(encoding='utf-8')
+    assert content == (
+        'typehints\n'
+        '*********\n'
+        '\n'
+        'mypackage.typehints.hello(x, *args, **kwargs)\n'
+        '\n'
+        '   Parameters:\n'
+        '      * **x** (*int*) -- X\n'
+        '\n'
+        '      * ***args** (*int*) -- Additional arguments.\n'
+        '\n'
+        '      * ****kwargs** (*int*) -- Extra arguments.\n'
+        '\n'
+        '   Return type:\n'
+        '      None\n'
+    )
+
+
+@pytest.mark.sphinx('text', testroot='ext-napoleon',
+                    confoverrides={'autodoc_typehints': 'description',
+                                   'autodoc_typehints_description_target': 'documented_params'})
+def test_napoleon_and_autodoc_typehints_description_documented_params(app, status, warning):
+    app.build()
+    content = (app.outdir / 'typehints.txt').read_text(encoding='utf-8')
+    assert content == (
+        'typehints\n'
+        '*********\n'
+        '\n'
+        'mypackage.typehints.hello(x, *args, **kwargs)\n'
+        '\n'
+        '   Parameters:\n'
+        '      * **x** (*int*) -- X\n'
+        '\n'
+        '      * ***args** (*int*) -- Additional arguments.\n'
+        '\n'
+        '      * ****kwargs** (*int*) -- Extra arguments.\n'
+    )

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 1411, End line: 1790

```python
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec: OptionSpec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'member-order': member_order_option,
        'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
        'class-doc-from': class_doc_from_option,
    }

    _signature_class: Any = None
    _signature_method_name: str = None

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

        if self.config.autodoc_class_signature == 'separated':
            self.options = self.options.copy()

            # show __init__() method
            if self.options.special_members is None:
                self.options['special-members'] = ['__new__', '__init__']
            else:
                self.options.special_members.append('__new__')
                self.options.special_members.append('__init__')

        merge_members_option(self.options)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, type)

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        # if the class is documented under another name, document it
        # as data/attribute
        if ret:
            if hasattr(self.object, '__name__'):
                self.doc_as_attr = (self.objpath[-1] != self.object.__name__)
            else:
                self.doc_as_attr = True
        return ret

    def _get_signature(self) -> Tuple[Optional[Any], Optional[str], Optional[Signature]]:
        def get_user_defined_function_or_method(obj: Any, attr: str) -> Any:
            """ Get the `attr` function or method from `obj`, if it is user-defined. """
            if inspect.is_builtin_class_method(obj, attr):
                return None
            attr = self.get_attr(obj, attr, None)
            if not (inspect.ismethod(attr) or inspect.isfunction(attr)):
                return None
            return attr

        # This sequence is copied from inspect._signature_from_callable.
        # ValueError means that no signature could be found, so we keep going.

        # First, we check the obj has a __signature__ attribute
        if (hasattr(self.object, '__signature__') and
                isinstance(self.object.__signature__, Signature)):
            return None, None, self.object.__signature__

        # Next, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = get_user_defined_function_or_method(type(self.object), '__call__')

        if call is not None:
            if "{0.__module__}.{0.__qualname__}".format(call) in _METACLASS_CALL_BLACKLIST:
                call = None

        if call is not None:
            self.env.app.emit('autodoc-before-process-signature', call, True)
            try:
                sig = inspect.signature(call, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return type(self.object), '__call__', sig
            except ValueError:
                pass

        # Now we check if the 'obj' class has a '__new__' method
        new = get_user_defined_function_or_method(self.object, '__new__')

        if new is not None:
            if "{0.__module__}.{0.__qualname__}".format(new) in _CLASS_NEW_BLACKLIST:
                new = None

        if new is not None:
            self.env.app.emit('autodoc-before-process-signature', new, True)
            try:
                sig = inspect.signature(new, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return self.object, '__new__', sig
            except ValueError:
                pass

        # Finally, we should have at least __init__ implemented
        init = get_user_defined_function_or_method(self.object, '__init__')
        if init is not None:
            self.env.app.emit('autodoc-before-process-signature', init, True)
            try:
                sig = inspect.signature(init, bound_method=True,
                                        type_aliases=self.config.autodoc_type_aliases)
                return self.object, '__init__', sig
            except ValueError:
                pass

        # None of the attributes are user-defined, so fall back to let inspect
        # handle it.
        # We don't know the exact method that inspect.signature will read
        # the signature from, so just pass the object itself to our hook.
        self.env.app.emit('autodoc-before-process-signature', self.object, False)
        try:
            sig = inspect.signature(self.object, bound_method=False,
                                    type_aliases=self.config.autodoc_type_aliases)
            return None, None, sig
        except ValueError:
            pass

        # Still no signature: happens e.g. for old-style classes
        # with __init__ in C and no `__text_signature__`.
        return None, None, None

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        try:
            self._signature_class, self._signature_method_name, sig = self._get_signature()
        except TypeError as exc:
            # __signature__ attribute contained junk
            logger.warning(__("Failed to get a constructor signature for %s: %s"),
                           self.fullname, exc)
            return None

        if sig is None:
            return None

        return stringify_signature(sig, show_return_annotation=False, **kwargs)

    def _find_signature(self) -> Tuple[str, str]:
        result = super()._find_signature()
        if result is not None:
            # Strip a return value from signature of constructor in docstring (first entry)
            result = (result[0], None)

        for i, sig in enumerate(self._signatures):
            if sig.endswith(' -> None'):
                # Strip a return value from signatures of constructor in docstring (subsequent
                # entries)
                self._signatures[i] = sig[:-8]

        return result

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''
        if self.config.autodoc_class_signature == 'separated':
            # do not show signatures
            return ''

        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        sig = super().format_signature()
        sigs = []

        overloads = self.get_overloaded_signatures()
        if overloads and self.config.autodoc_typehints != 'none':
            # Use signatures for overloaded methods instead of the implementation method.
            method = safe_getattr(self._signature_class, self._signature_method_name, None)
            __globals__ = safe_getattr(method, '__globals__', {})
            for overload in overloads:
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                parameters = list(overload.parameters.values())
                overload = overload.replace(parameters=parameters[1:],
                                            return_annotation=Parameter.empty)
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)
        else:
            sigs.append(sig)

        return "\n".join(sigs)

    def get_overloaded_signatures(self) -> List[Signature]:
        if self._signature_class and self._signature_method_name:
            for cls in self._signature_class.__mro__:
                try:
                    analyzer = ModuleAnalyzer.for_module(cls.__module__)
                    analyzer.analyze()
                    qualname = '.'.join([cls.__qualname__, self._signature_method_name])
                    if qualname in analyzer.overloads:
                        return analyzer.overloads.get(qualname)
                    elif qualname in analyzer.tagorder:
                        # the constructor is defined in the class, but not overridden.
                        return []
                except PycodeError:
                    pass

        return []

    def get_canonical_fullname(self) -> Optional[str]:
        __modname__ = safe_getattr(self.object, '__module__', self.modname)
        __qualname__ = safe_getattr(self.object, '__qualname__', None)
        if __qualname__ is None:
            __qualname__ = safe_getattr(self.object, '__name__', None)
        if __qualname__ and '<locals>' in __qualname__:
            # No valid qualname found if the object is defined as locals
            __qualname__ = None

        if __modname__ and __qualname__:
            return '.'.join([__modname__, __qualname__])
        else:
            return None

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

        canonical_fullname = self.get_canonical_fullname()
        if not self.doc_as_attr and canonical_fullname and self.fullname != canonical_fullname:
            self.add_line('   :canonical: %s' % canonical_fullname, sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            if inspect.getorigbases(self.object):
                # A subclass of generic types
                # refs: PEP-560 <https://peps.python.org/pep-0560/>
                bases = list(self.object.__orig_bases__)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                # A normal class
                bases = list(self.object.__bases__)
            else:
                bases = []

            self.env.events.emit('autodoc-process-bases',
                                 self.fullname, self.object, self.options, bases)

            if self.config.autodoc_typehints_format == "short":
                base_classes = [restify(cls, "smart") for cls in bases]
            else:
                base_classes = [restify(cls) for cls in bases]

            sourcename = self.get_sourcename()
            self.add_line('', sourcename)
            self.add_line('   ' + _('Bases: %s') % ', '.join(base_classes), sourcename)

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = get_class_members(self.object, self.objpath, self.get_attr)
        if not want_all:
            if not self.options.members:
                return False, []  # type: ignore
            # specific members given
            selected = []
            for name in self.options.members:  # type: str
                if name in members:
                    selected.append(members[name])
                else:
                    logger.warning(__('missing attribute %s in object %s') %
                                   (name, self.fullname), type='autodoc')
            return False, selected
        elif self.options.inherited_members:
            return False, list(members.values())
        else:
            return False, [m for m in members.values() if m.class_ == self.object]

    def get_doc(self) -> Optional[List[List[str]]]:
        if self.doc_as_attr:
            # Don't show the docstring of the class when it is an alias.
            comment = self.get_variable_comment()
            if comment:
                return []
            else:
                return None

        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        classdoc_from = self.options.get('class-doc-from', self.config.autoclass_content)

        docstrings = []
        attrdocstring = getdoc(self.object, self.get_attr)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if classdoc_from in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.config.autodoc_inherit_docstrings,
                                   self.object, '__init__')
            # for new-style classes, no __init__ means default __init__
            if (initdocstring is not None and
                (initdocstring == object.__init__.__doc__ or  # for pypy
                 initdocstring.strip() == object.__init__.__doc__)):  # for !pypy
                initdocstring = None
            if not initdocstring:
                # try __new__
                __new__ = self.get_attr(self.object, '__new__', None)
                initdocstring = getdoc(__new__, self.get_attr,
                                       self.config.autodoc_inherit_docstrings,
                                       self.object, '__new__')
                # for new-style classes, no __new__ means default __new__
                if (initdocstring is not None and
                    (initdocstring == object.__new__.__doc__ or  # for pypy
                     initdocstring.strip() == object.__new__.__doc__)):  # for !pypy
                    initdocstring = None
            if initdocstring:
                if classdoc_from == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, tab_width) for docstring in docstrings]

    def get_variable_comment(self) -> Optional[List[str]]:
        try:
            key = ('', '.'.join(self.objpath))
            if self.doc_as_attr:
                analyzer = ModuleAnalyzer.for_module(self.modname)
            else:
                analyzer = ModuleAnalyzer.for_module(self.get_real_modname())
            analyzer.analyze()
            return list(analyzer.attr_docs.get(key, []))
        except PycodeError:
            return None

    def add_content(self, more_content: Optional[StringList]) -> None:
        if self.doc_as_attr and self.modname != self.get_real_modname():
            try:
                # override analyzer to obtain doccomment around its definition.
                self.analyzer = ModuleAnalyzer.for_module(self.modname)
                self.analyzer.analyze()
            except PycodeError:
                pass

        if self.doc_as_attr and not self.get_variable_comment():
            try:
                if self.config.autodoc_typehints_format == "short":
                    alias = restify(self.object, "smart")
                else:
                    alias = restify(self.object)
                more_content = StringList([_('alias of %s') % alias], source='')
            except AttributeError:
                pass  # Invalid class object is passed.

        super().add_content(more_content)

    def document_members(self, all_members: bool = False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Optional[StringList] = None, real_modname: str = None,
                 check_module: bool = False, all_members: bool = False) -> None:
        # Do not pass real_modname and use the name from the __module__
        # attribute of the class.
        # If a class gets imported into the module real_modname
        # the analyzer won't find the source of the class, if
        # it looks in real_modname.
        return super().generate(more_content=more_content,
                                check_module=check_module,
                                all_members=all_members)
```
### 2 - doc/usage/extensions/example_numpy.py:

Start line: 221, End line: 314

```python
class ExampleClass:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    attr1 : str
        Description of `attr1`.
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : list(str)
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list(str): Doc comment *before* attribute, with type specified
        self.attr4 = ["attr4"]

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return "readonly_property"

    @property
    def readwrite_property(self):
        """list(str): Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ["readwrite_property"]

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        return True
```
### 3 - doc/usage/extensions/example_google.py:

Start line: 176, End line: 255

```python
class ExampleClass:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (list(str)): Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list(str): Doc comment *before* attribute, with type specified
        self.attr4 = ['attr4']

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return 'readonly_property'

    @property
    def readwrite_property(self):
        """list(str): Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True
```
### 4 - sphinx/ext/autodoc/__init__.py:

Start line: 2663, End line: 2687

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_doc(self) -> Optional[List[List[str]]]:
        # Check the attribute has a docstring-comment
        comment = self.get_attribute_comment(self.parent, self.objpath[-1])
        if comment:
            return [comment]

        try:
            # Disable `autodoc_inherit_docstring` temporarily to avoid to obtain
            # a docstring from the value which descriptor returns unexpectedly.
            # ref: https://github.com/sphinx-doc/sphinx/issues/7805
            orig = self.config.autodoc_inherit_docstrings
            self.config.autodoc_inherit_docstrings = False  # type: ignore
            return super().get_doc()
        finally:
            self.config.autodoc_inherit_docstrings = orig  # type: ignore

    def add_content(self, more_content: Optional[StringList]) -> None:
        # Disable analyzing attribute comment on Documenter.add_content() to control it on
        # AttributeDocumenter.add_content()
        self.analyzer = None

        if more_content is None:
            more_content = StringList()
        self.update_content(more_content)
        super().add_content(more_content)
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 1216, End line: 1232

```python
class DocstringSignatureMixin:

    def get_doc(self) -> List[List[str]]:
        if self._new_docstrings is not None:
            return self._new_docstrings
        return super().get_doc()  # type: ignore

    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.config.autodoc_docstring_signature:  # type: ignore
            # only act if a signature is not explicitly given already, and if
            # the feature is enabled
            result = self._find_signature()
            if result is not None:
                self.args, self.retann = result
        sig = super().format_signature(**kwargs)  # type: ignore
        if self._signatures:
            return "\n".join([sig] + self._signatures)
        else:
            return sig
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 2099, End line: 2309

```python
class MethodDocumenter(DocstringSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for methods (normal, static and class).
    """
    objtype = 'method'
    directivetype = 'method'
    member_order = 50
    priority = 1  # must be more than FunctionDocumenter

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return inspect.isroutine(member) and not isinstance(parent, ModuleDocumenter)

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if not ret:
            return ret

        # to distinguish classmethod/staticmethod
        obj = self.parent.__dict__.get(self.object_name)
        if obj is None:
            obj = self.object

        if (inspect.isclassmethod(obj) or
                inspect.isstaticmethod(obj, cls=self.parent, name=self.object_name)):
            # document class and static members before ordinary ones
            self.member_order = self.member_order - 1

        return ret

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        try:
            if self.object == object.__init__ and self.parent != object:
                # Classes not having own __init__() method are shown as no arguments.
                #
                # Note: The signature of object.__init__() is (self, /, *args, **kwargs).
                #       But it makes users confused.
                args = '()'
            else:
                if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                    self.env.app.emit('autodoc-before-process-signature', self.object, False)
                    sig = inspect.signature(self.object, bound_method=False,
                                            type_aliases=self.config.autodoc_type_aliases)
                else:
                    self.env.app.emit('autodoc-before-process-signature', self.object, True)
                    sig = inspect.signature(self.object, bound_method=True,
                                            type_aliases=self.config.autodoc_type_aliases)
                args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

        sourcename = self.get_sourcename()
        obj = self.parent.__dict__.get(self.object_name, self.object)
        if inspect.isabstractmethod(obj):
            self.add_line('   :abstractmethod:', sourcename)
        if inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj):
            self.add_line('   :async:', sourcename)
        if inspect.isclassmethod(obj):
            self.add_line('   :classmethod:', sourcename)
        if inspect.isstaticmethod(obj, cls=self.parent, name=self.object_name):
            self.add_line('   :staticmethod:', sourcename)
        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

    def document_members(self, all_members: bool = False) -> None:
        pass

    def format_signature(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints_format == "short":
            kwargs.setdefault('unqualified_typehints', True)

        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints != 'none'):
            # Use signatures for overloaded methods instead of the implementation method.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        meth = self.parent.__dict__.get(self.objpath[-1])
        if inspect.is_singledispatch_method(meth):
            # append signature of singledispatch'ed functions
            for typ, func in meth.dispatcher.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    dispatchmeth = self.annotate_to_first_argument(func, typ)
                    if dispatchmeth:
                        documenter = MethodDocumenter(self.directive, '')
                        documenter.parent = self.parent
                        documenter.object = dispatchmeth
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
                actual = inspect.signature(self.object, bound_method=False,
                                           type_aliases=self.config.autodoc_type_aliases)
            else:
                actual = inspect.signature(self.object, bound_method=True,
                                           type_aliases=self.config.autodoc_type_aliases)

            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

                if not inspect.isstaticmethod(self.object, cls=self.parent,
                                              name=self.object_name):
                    parameters = list(overload.parameters.values())
                    overload = overload.replace(parameters=parameters[1:])
                sig = stringify_signature(overload, **kwargs)
                sigs.append(sig)

        return "\n".join(sigs)

    def merge_default_value(self, actual: Signature, overload: Signature) -> Signature:
        """Merge default values of actual implementation to the overload variants."""
        parameters = list(overload.parameters.values())
        for i, param in enumerate(parameters):
            actual_param = actual.parameters.get(param.name)
            if actual_param and param.default == '...':
                parameters[i] = param.replace(default=actual_param.default)

        return overload.replace(parameters=parameters)

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
        """Annotate type hint to the first argument of function if needed."""
        try:
            sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
        except TypeError as exc:
            logger.warning(__("Failed to get a method signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 1:
            return None

        def dummy():
            pass

        params = list(sig.parameters.values())
        if params[1].annotation is Parameter.empty:
            params[1] = params[1].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)  # type: ignore
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None

        return func

    def get_doc(self) -> Optional[List[List[str]]]:
        if self._new_docstrings is not None:
            # docstring already returned previously, then modified by
            # `DocstringSignatureMixin`.  Just return the previously-computed
            # result, so that we don't lose the processing done by
            # `DocstringSignatureMixin`.
            return self._new_docstrings
        if self.objpath[-1] == '__init__':
            docstring = getdoc(self.object, self.get_attr,
                               self.config.autodoc_inherit_docstrings,
                               self.parent, self.object_name)
            if (docstring is not None and
                (docstring == object.__init__.__doc__ or  # for pypy
                 docstring.strip() == object.__init__.__doc__)):  # for !pypy
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        elif self.objpath[-1] == '__new__':
            docstring = getdoc(self.object, self.get_attr,
                               self.config.autodoc_inherit_docstrings,
                               self.parent, self.object_name)
            if (docstring is not None and
                (docstring == object.__new__.__doc__ or  # for pypy
                 docstring.strip() == object.__new__.__doc__)):  # for !pypy
                docstring = None
            if docstring:
                tab_width = self.directive.state.document.settings.tab_width
                return [prepare_docstring(docstring, tabsize=tab_width)]
            else:
                return []
        else:
            return super().get_doc()
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 2616, End line: 2644

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)
        else:
            if self.config.autodoc_typehints != 'none':
                # obtain type annotation for this attribute
                annotations = get_type_hints(self.parent, None,
                                             self.config.autodoc_type_aliases)
                if self.objpath[-1] in annotations:
                    if self.config.autodoc_typehints_format == "short":
                        objrepr = stringify_typehint(annotations.get(self.objpath[-1]),
                                                     "smart")
                    else:
                        objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                    self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if (self.options.no_value or self.should_suppress_value_header() or
                        ismock(self.object)):
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass
```
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 2011, End line: 2040

```python
class DataDocumenter(GenericAliasMixin, NewTypeMixin, TypeVarMixin,
                     UninitializedGlobalVariableMixin, ModuleLevelDocumenter):

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation,
                          sourcename)
        else:
            if self.config.autodoc_typehints != 'none':
                # obtain annotation for this data
                annotations = get_type_hints(self.parent, None,
                                             self.config.autodoc_type_aliases)
                if self.objpath[-1] in annotations:
                    if self.config.autodoc_typehints_format == "short":
                        objrepr = stringify_typehint(annotations.get(self.objpath[-1]),
                                                     "smart")
                    else:
                        objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                    self.add_line('   :type: ' + objrepr, sourcename)

            try:
                if (self.options.no_value or self.should_suppress_value_header() or
                        ismock(self.object)):
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass
```
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 682, End line: 792

```python
class Documenter:

    def filter_members(self, members: ObjectMembers, want_all: bool
                       ) -> List[Tuple[str, Any, bool]]:
        # ... other code
        for obj in members:
            try:
                membername, member = obj
                # if isattr is True, the member is documented as an attribute
                if member is INSTANCEATTR:
                    isattr = True
                elif (namespace, membername) in attr_docs:
                    isattr = True
                else:
                    isattr = False

                doc = getdoc(member, self.get_attr, self.config.autodoc_inherit_docstrings,
                             self.object, membername)
                if not isinstance(doc, str):
                    # Ignore non-string __doc__
                    doc = None

                # if the member __doc__ is the same as self's __doc__, it's just
                # inherited and therefore not the member's doc
                cls = self.get_attr(member, '__class__', None)
                if cls:
                    cls_doc = self.get_attr(cls, '__doc__', None)
                    if cls_doc == doc:
                        doc = None

                if isinstance(obj, ObjectMember) and obj.docstring:
                    # hack for ClassDocumenter to inject docstring via ObjectMember
                    doc = obj.docstring

                doc, metadata = separate_metadata(doc)
                has_doc = bool(doc)

                if 'private' in metadata:
                    # consider a member private if docstring has "private" metadata
                    isprivate = True
                elif 'public' in metadata:
                    # consider a member public if docstring has "public" metadata
                    isprivate = False
                else:
                    isprivate = membername.startswith('_')

                keep = False
                if ismock(member) and (namespace, membername) not in attr_docs:
                    # mocked module or object
                    pass
                elif (self.options.exclude_members and
                      membername in self.options.exclude_members):
                    # remove members given by exclude-members
                    keep = False
                elif want_all and special_member_re.match(membername):
                    # special __methods__
                    if (self.options.special_members and
                            membername in self.options.special_members):
                        if membername == '__doc__':
                            keep = False
                        elif is_filtered_inherited_member(membername, obj):
                            keep = False
                        else:
                            keep = has_doc or self.options.undoc_members
                    else:
                        keep = False
                elif (namespace, membername) in attr_docs:
                    if want_all and isprivate:
                        if self.options.private_members is None:
                            keep = False
                        else:
                            keep = membername in self.options.private_members
                    else:
                        # keep documented attributes
                        keep = True
                elif want_all and isprivate:
                    if has_doc or self.options.undoc_members:
                        if self.options.private_members is None:
                            keep = False
                        elif is_filtered_inherited_member(membername, obj):
                            keep = False
                        else:
                            keep = membername in self.options.private_members
                    else:
                        keep = False
                else:
                    if (self.options.members is ALL and
                            is_filtered_inherited_member(membername, obj)):
                        keep = False
                    else:
                        # ignore undocumented members if :undoc-members: is not given
                        keep = has_doc or self.options.undoc_members

                if isinstance(obj, ObjectMember) and obj.skipped:
                    # forcedly skipped member (ex. a module attribute not defined in __all__)
                    keep = False

                # give the user a chance to decide whether this member
                # should be skipped
                if self.env.app:
                    # let extensions preprocess docstrings
                    skip_user = self.env.app.emit_firstresult(
                        'autodoc-skip-member', self.objtype, membername, member,
                        not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
            except Exception as exc:
                logger.warning(__('autodoc: failed to determine %s.%s (%r) to be documented, '
                                  'the following exception was raised:\n%s'),
                               self.name, membername, member, exc, type='autodoc')
                keep = False

            if keep:
                ret.append((membername, member, isattr))

        return ret
```
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 558, End line: 570

```python
class Documenter:

    def get_sourcename(self) -> str:
        if (inspect.safe_getattr(self.object, '__module__', None) and
                inspect.safe_getattr(self.object, '__qualname__', None)):
            # Get the correct location of docstring from self.object
            # to support inherited methods
            fullname = '%s.%s' % (self.object.__module__, self.object.__qualname__)
        else:
            fullname = self.fullname

        if self.analyzer:
            return '%s:docstring of %s' % (self.analyzer.srcname, fullname)
        else:
            return 'docstring of %s' % fullname
```
### 41 - sphinx/ext/autodoc/typehints.py:

Start line: 141, End line: 199

```python
def augment_descriptions_with_types(
    node: nodes.field_list,
    annotations: Dict[str, str],
    force_rtype: bool
) -> None:
    fields = cast(Iterable[nodes.field], node)
    has_description = set()  # type: Set[str]
    has_type = set()  # type: Set[str]
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                # :param xxx:
                has_description.add(parts[1])
            elif len(parts) > 2:
                # :param xxx yyy:
                name = ' '.join(parts[2:])
                has_description.add(name)
                has_type.add(name)
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            has_type.add(name)
        elif parts[0] in ('return', 'returns'):
            has_description.add('return')
        elif parts[0] == 'rtype':
            has_type.add('return')

    # Add 'type' for parameters with a description but no declared type.
    for name in annotations:
        if name in ('return', 'returns'):
            continue
        if name in has_description and name not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotations[name]))
            node += field

    # Add 'rtype' if 'return' is present and 'rtype' isn't.
    if 'return' in annotations:
        rtype = annotations['return']
        if 'return' not in has_type and ('return' in has_description or
                                         (force_rtype and rtype != "None")):
            field = nodes.field()
            field += nodes.field_name('', 'rtype')
            field += nodes.field_body('', nodes.paragraph('', rtype))
            node += field


def setup(app: Sphinx) -> Dict[str, Any]:
    app.connect('autodoc-process-signature', record_typehints)
    app.connect('object-description-transform', merge_typehints)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 49 - sphinx/ext/autodoc/typehints.py:

Start line: 1, End line: 34

```python
"""Generating content for autodoc using typehints"""

import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast

from docutils import nodes
from docutils.nodes import Element

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing


def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
                     options: Dict, args: str, retann: str) -> None:
    """Record type hints to env object."""
    if app.config.autodoc_typehints_format == 'short':
        mode = 'smart'
    else:
        mode = 'fully-qualified'

    try:
        if callable(obj):
            annotations = app.env.temp_data.setdefault('annotations', {})
            annotation = annotations.setdefault(name, OrderedDict())
            sig = inspect.signature(obj, type_aliases=app.config.autodoc_type_aliases)
            for param in sig.parameters.values():
                if param.annotation is not param.empty:
                    annotation[param.name] = typing.stringify(param.annotation, mode)
            if sig.return_annotation is not sig.empty:
                annotation['return'] = typing.stringify(sig.return_annotation, mode)
    except (TypeError, ValueError):
        pass
```
