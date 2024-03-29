# sphinx-doc__sphinx-9230

| **sphinx-doc/sphinx** | `567ff22716ac258b9edd2c1711d766b440ac0b11` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 13 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/util/docfields.py b/sphinx/util/docfields.py
--- a/sphinx/util/docfields.py
+++ b/sphinx/util/docfields.py
@@ -298,7 +298,7 @@ def transform(self, node: nodes.field_list) -> None:
             # also support syntax like ``:param type name:``
             if typedesc.is_typed:
                 try:
-                    argtype, argname = fieldarg.split(None, 1)
+                    argtype, argname = fieldarg.rsplit(None, 1)
                 except ValueError:
                     pass
                 else:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/util/docfields.py | 301 | 301 | - | 13 | -


## Problem Statement

```
Doc rendering is incorrect when :param has datatype dict(str,str)
**Describe the bug**
I have a parameter defined under docstring of a method as:-
:param dict(str, str) opc_meta: (optional)

Which is being incorrectly rendered in the generated docs as:-
str) opc_meta (dict(str,) –(optional) 

**To Reproduce**
Create any method with the docstring containg the above param

**Expected behavior**
The param should be rendered in the generated docs as:-
opc_meta (dict(str,str)) – (optional) 

**Your project**
[sphinxTest.zip](https://github.com/sphinx-doc/sphinx/files/6468074/sphinxTest.zip)


**Screenshots**
<img width="612" alt="Screen Shot 2021-05-12 at 12 30 50 PM" src="https://user-images.githubusercontent.com/8617566/118020143-5f59a280-b31f-11eb-8dc2-5280d5c4896b.png">
<img width="681" alt="Screen Shot 2021-05-12 at 12 32 25 PM" src="https://user-images.githubusercontent.com/8617566/118020154-62549300-b31f-11eb-953d-9287f9cc27ff.png">


**Environment info**
- OS: Mac
- Python version: 3.9.0
- Sphinx version: 4.0.1
- Sphinx extensions:  ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.intersphinx", "autodocsumm"]
- Extra tools: Browser Firefox.

**Additional context**
N/A



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/__init__.py | 2711 | 2754| 528 | 528 | 23212 | 
| 2 | 1 sphinx/ext/autodoc/__init__.py | 2592 | 2617| 316 | 844 | 23212 | 
| 3 | 1 sphinx/ext/autodoc/__init__.py | 2527 | 2550| 229 | 1073 | 23212 | 
| 4 | 2 sphinx/ext/autodoc/deprecated.py | 32 | 47| 140 | 1213 | 24172 | 
| 5 | 3 sphinx/writers/html5.py | 153 | 209| 521 | 1734 | 31248 | 
| 6 | 4 sphinx/writers/html.py | 182 | 238| 516 | 2250 | 38792 | 
| 7 | 4 sphinx/ext/autodoc/__init__.py | 2058 | 2259| 1807 | 4057 | 38792 | 
| 8 | 5 sphinx/ext/autodoc/typehints.py | 130 | 185| 454 | 4511 | 40240 | 
| 9 | 5 sphinx/ext/autodoc/__init__.py | 1426 | 1766| 3118 | 7629 | 40240 | 
| 10 | 5 sphinx/ext/autodoc/__init__.py | 1273 | 1392| 995 | 8624 | 40240 | 
| 11 | 6 sphinx/ext/napoleon/docstring.py | 695 | 736| 418 | 9042 | 51240 | 
| 12 | 6 sphinx/ext/autodoc/__init__.py | 1976 | 1998| 245 | 9287 | 51240 | 
| 13 | 7 sphinx/ext/autosummary/__init__.py | 165 | 179| 113 | 9400 | 57751 | 
| 14 | 7 sphinx/ext/autodoc/__init__.py | 1236 | 1252| 184 | 9584 | 57751 | 
| 15 | 7 sphinx/ext/napoleon/docstring.py | 413 | 428| 180 | 9764 | 57751 | 
| 16 | 7 sphinx/ext/napoleon/docstring.py | 1168 | 1194| 259 | 10023 | 57751 | 
| 17 | 7 sphinx/ext/autodoc/__init__.py | 584 | 596| 133 | 10156 | 57751 | 
| 18 | 7 sphinx/ext/autodoc/__init__.py | 2000 | 2037| 320 | 10476 | 57751 | 
| 19 | 7 sphinx/ext/autodoc/__init__.py | 1927 | 1974| 365 | 10841 | 57751 | 
| 20 | 7 sphinx/ext/autodoc/__init__.py | 2575 | 2590| 182 | 11023 | 57751 | 
| 21 | 7 sphinx/ext/napoleon/docstring.py | 430 | 457| 258 | 11281 | 57751 | 
| 22 | 7 sphinx/ext/napoleon/docstring.py | 13 | 67| 578 | 11859 | 57751 | 
| 23 | 7 sphinx/ext/autodoc/__init__.py | 2323 | 2337| 152 | 12011 | 57751 | 
| 24 | 7 sphinx/ext/autodoc/__init__.py | 2552 | 2573| 272 | 12283 | 57751 | 
| 25 | 7 sphinx/ext/autodoc/__init__.py | 710 | 818| 879 | 13162 | 57751 | 
| 26 | 7 sphinx/ext/autodoc/__init__.py | 1395 | 1766| 189 | 13351 | 57751 | 
| 27 | 7 sphinx/ext/napoleon/docstring.py | 1146 | 1166| 157 | 13508 | 57751 | 
| 28 | 7 sphinx/ext/autodoc/__init__.py | 1868 | 1883| 160 | 13668 | 57751 | 
| 29 | 7 sphinx/ext/autodoc/__init__.py | 13 | 114| 788 | 14456 | 57751 | 
| 30 | 7 sphinx/ext/napoleon/docstring.py | 1196 | 1233| 349 | 14805 | 57751 | 
| 31 | 8 sphinx/ext/autosummary/generate.py | 84 | 96| 163 | 14968 | 63050 | 
| 32 | 8 sphinx/ext/autodoc/deprecated.py | 114 | 127| 114 | 15082 | 63050 | 
| 33 | 8 sphinx/ext/autodoc/__init__.py | 2505 | 2525| 231 | 15313 | 63050 | 
| 34 | 9 sphinx/util/smartypants.py | 380 | 392| 137 | 15450 | 67197 | 
| 35 | 10 doc/conf.py | 141 | 162| 255 | 15705 | 68661 | 
| 36 | 10 sphinx/ext/napoleon/docstring.py | 374 | 385| 119 | 15824 | 68661 | 
| 37 | 10 sphinx/ext/napoleon/docstring.py | 655 | 680| 267 | 16091 | 68661 | 
| 38 | 10 sphinx/ext/autosummary/generate.py | 20 | 53| 257 | 16348 | 68661 | 
| 39 | 10 sphinx/ext/napoleon/docstring.py | 332 | 372| 329 | 16677 | 68661 | 
| 40 | 10 sphinx/ext/napoleon/docstring.py | 738 | 754| 195 | 16872 | 68661 | 
| 41 | 10 sphinx/ext/napoleon/docstring.py | 259 | 283| 254 | 17126 | 68661 | 
| 42 | 10 sphinx/ext/autosummary/__init__.py | 286 | 307| 200 | 17326 | 68661 | 
| 43 | 10 sphinx/ext/autodoc/deprecated.py | 50 | 62| 113 | 17439 | 68661 | 
| 44 | 10 sphinx/ext/autodoc/deprecated.py | 65 | 75| 108 | 17547 | 68661 | 
| 45 | 10 sphinx/ext/napoleon/docstring.py | 478 | 519| 345 | 17892 | 68661 | 
| 46 | 11 doc/usage/extensions/example_numpy.py | 1 | 47| 301 | 18193 | 70769 | 
| 47 | 12 doc/development/tutorials/examples/autodoc_intenum.py | 27 | 53| 200 | 18393 | 71143 | 
| 48 | 12 sphinx/ext/napoleon/docstring.py | 308 | 330| 199 | 18592 | 71143 | 
| 49 | 12 sphinx/ext/napoleon/docstring.py | 859 | 872| 153 | 18745 | 71143 | 
| 50 | 12 sphinx/ext/autosummary/__init__.py | 182 | 214| 279 | 19024 | 71143 | 
| 51 | **13 sphinx/util/docfields.py** | 144 | 170| 259 | 19283 | 74251 | 
| 52 | 13 sphinx/ext/napoleon/docstring.py | 628 | 653| 255 | 19538 | 74251 | 
| 53 | 13 sphinx/ext/napoleon/docstring.py | 829 | 857| 212 | 19750 | 74251 | 
| 54 | 13 sphinx/ext/autodoc/__init__.py | 2262 | 2290| 248 | 19998 | 74251 | 
| 55 | 13 doc/conf.py | 1 | 82| 731 | 20729 | 74251 | 
| 56 | 14 sphinx/util/typing.py | 210 | 281| 834 | 21563 | 78593 | 
| 57 | 14 sphinx/ext/napoleon/docstring.py | 769 | 795| 223 | 21786 | 78593 | 
| 58 | 14 sphinx/ext/napoleon/docstring.py | 1047 | 1144| 745 | 22531 | 78593 | 
| 59 | 15 sphinx/ext/autodoc/preserve_defaults.py | 52 | 89| 350 | 22881 | 79287 | 
| 60 | 15 sphinx/ext/napoleon/docstring.py | 459 | 476| 178 | 23059 | 79287 | 
| 61 | 15 sphinx/ext/autodoc/__init__.py | 296 | 372| 751 | 23810 | 79287 | 
| 62 | 16 sphinx/ext/autodoc/directive.py | 50 | 79| 251 | 24061 | 80737 | 
| 63 | 17 sphinx/ext/autodoc/importer.py | 77 | 147| 649 | 24710 | 83199 | 
| 64 | 17 sphinx/ext/napoleon/docstring.py | 1235 | 1339| 728 | 25438 | 83199 | 
| 65 | 17 sphinx/ext/napoleon/docstring.py | 298 | 306| 121 | 25559 | 83199 | 
| 66 | 17 sphinx/util/typing.py | 383 | 446| 638 | 26197 | 83199 | 
| 67 | 17 sphinx/ext/napoleon/docstring.py | 582 | 614| 265 | 26462 | 83199 | 
| 68 | 17 sphinx/ext/napoleon/docstring.py | 560 | 580| 233 | 26695 | 83199 | 
| 69 | 17 sphinx/ext/autodoc/__init__.py | 1023 | 1035| 124 | 26819 | 83199 | 
| 70 | 17 sphinx/ext/autodoc/deprecated.py | 78 | 93| 139 | 26958 | 83199 | 
| 71 | 18 sphinx/ext/autodoc/mock.py | 11 | 69| 452 | 27410 | 84547 | 
| 72 | 18 sphinx/ext/autodoc/importer.py | 11 | 40| 211 | 27621 | 84547 | 
| 73 | 18 sphinx/ext/autosummary/__init__.py | 55 | 103| 372 | 27993 | 84547 | 
| 74 | 19 sphinx/writers/manpage.py | 192 | 256| 502 | 28495 | 88034 | 
| 75 | 19 sphinx/ext/autodoc/__init__.py | 1103 | 1120| 182 | 28677 | 88034 | 
| 76 | 19 sphinx/ext/napoleon/docstring.py | 521 | 536| 130 | 28807 | 88034 | 
| 77 | 19 sphinx/ext/autosummary/generate.py | 225 | 260| 339 | 29146 | 88034 | 
| 78 | 20 sphinx/directives/__init__.py | 74 | 86| 132 | 29278 | 90286 | 
| 79 | 20 sphinx/ext/autodoc/__init__.py | 1769 | 1804| 245 | 29523 | 90286 | 
| 80 | 20 sphinx/ext/napoleon/docstring.py | 70 | 134| 599 | 30122 | 90286 | 
| 81 | 20 sphinx/ext/autodoc/__init__.py | 1077 | 1101| 209 | 30331 | 90286 | 
| 82 | 20 sphinx/ext/autodoc/__init__.py | 890 | 976| 806 | 31137 | 90286 | 
| 83 | 20 doc/usage/extensions/example_numpy.py | 225 | 318| 589 | 31726 | 90286 | 
| 84 | 21 sphinx/ext/autodoc/type_comment.py | 115 | 140| 257 | 31983 | 91510 | 
| 85 | 21 sphinx/ext/napoleon/docstring.py | 797 | 827| 283 | 32266 | 91510 | 
| 86 | 21 sphinx/ext/napoleon/docstring.py | 1011 | 1044| 302 | 32568 | 91510 | 
| 87 | 21 sphinx/ext/autodoc/__init__.py | 495 | 528| 286 | 32854 | 91510 | 
| 88 | 22 sphinx/domains/python.py | 11 | 79| 507 | 33361 | 103401 | 
| 89 | 23 sphinx/util/inspect.py | 827 | 854| 230 | 33591 | 109890 | 
| 90 | 23 sphinx/ext/autodoc/type_comment.py | 11 | 35| 239 | 33830 | 109890 | 
| 91 | 23 sphinx/ext/napoleon/docstring.py | 756 | 767| 147 | 33977 | 109890 | 
| 92 | 23 sphinx/ext/autodoc/__init__.py | 820 | 863| 451 | 34428 | 109890 | 
| 93 | 23 sphinx/ext/autodoc/directive.py | 82 | 105| 255 | 34683 | 109890 | 
| 94 | 23 sphinx/ext/autodoc/directive.py | 9 | 47| 310 | 34993 | 109890 | 
| 95 | 23 sphinx/ext/autodoc/deprecated.py | 11 | 29| 155 | 35148 | 109890 | 
| 96 | 23 sphinx/ext/napoleon/docstring.py | 682 | 693| 124 | 35272 | 109890 | 
| 97 | 24 doc/usage/extensions/example_google.py | 180 | 259| 563 | 35835 | 112000 | 
| 98 | 24 sphinx/ext/autosummary/__init__.py | 759 | 786| 378 | 36213 | 112000 | 
| 99 | 24 sphinx/ext/autodoc/__init__.py | 2681 | 2711| 363 | 36576 | 112000 | 
| 100 | 24 doc/usage/extensions/example_numpy.py | 320 | 334| 109 | 36685 | 112000 | 
| 101 | 24 sphinx/ext/autodoc/__init__.py | 1177 | 1234| 434 | 37119 | 112000 | 
| 102 | 24 sphinx/ext/napoleon/docstring.py | 387 | 411| 235 | 37354 | 112000 | 
| 103 | 25 sphinx/writers/texinfo.py | 858 | 962| 788 | 38142 | 124314 | 
| 104 | 25 sphinx/ext/autodoc/__init__.py | 2448 | 2482| 292 | 38434 | 124314 | 
| 105 | 25 sphinx/ext/autosummary/generate.py | 317 | 365| 516 | 38950 | 124314 | 
| 106 | 25 sphinx/writers/html5.py | 52 | 151| 789 | 39739 | 124314 | 
| 107 | 25 sphinx/ext/autodoc/__init__.py | 2620 | 2660| 351 | 40090 | 124314 | 
| 108 | 25 sphinx/ext/napoleon/docstring.py | 285 | 296| 139 | 40229 | 124314 | 
| 109 | 25 sphinx/ext/autodoc/__init__.py | 2040 | 2259| 136 | 40365 | 124314 | 
| 110 | 25 sphinx/ext/autodoc/__init__.py | 445 | 493| 359 | 40724 | 124314 | 
| 111 | 25 sphinx/util/inspect.py | 11 | 47| 290 | 41014 | 124314 | 
| 112 | 25 sphinx/ext/autodoc/__init__.py | 1037 | 1048| 126 | 41140 | 124314 | 
| 113 | 25 sphinx/ext/autodoc/__init__.py | 1255 | 1270| 166 | 41306 | 124314 | 
| 114 | 25 sphinx/writers/texinfo.py | 1296 | 1384| 699 | 42005 | 124314 | 
| 115 | 26 sphinx/ext/apidoc.py | 303 | 368| 751 | 42756 | 128544 | 
| 116 | 26 sphinx/ext/autodoc/deprecated.py | 96 | 111| 133 | 42889 | 128544 | 
| 117 | 27 sphinx/writers/latex.py | 785 | 857| 605 | 43494 | 147879 | 
| 118 | 27 doc/usage/extensions/example_google.py | 261 | 275| 109 | 43603 | 147879 | 
| 119 | 28 sphinx/ext/doctest.py | 246 | 273| 277 | 43880 | 152897 | 
| 120 | 28 sphinx/ext/autosummary/__init__.py | 498 | 548| 399 | 44279 | 152897 | 
| 121 | **28 sphinx/util/docfields.py** | 172 | 204| 356 | 44635 | 152897 | 
| 122 | 28 doc/usage/extensions/example_google.py | 38 | 75| 245 | 44880 | 152897 | 
| 123 | 28 sphinx/ext/autodoc/__init__.py | 552 | 567| 178 | 45058 | 152897 | 
| 124 | 28 sphinx/ext/napoleon/docstring.py | 538 | 558| 195 | 45253 | 152897 | 
| 125 | 29 sphinx/util/docutils.py | 361 | 381| 174 | 45427 | 157039 | 
| 126 | 29 sphinx/ext/autosummary/__init__.py | 424 | 438| 183 | 45610 | 157039 | 
| 127 | 29 doc/development/tutorials/examples/autodoc_intenum.py | 1 | 25| 183 | 45793 | 157039 | 
| 128 | 29 doc/conf.py | 83 | 138| 476 | 46269 | 157039 | 
| 129 | 29 sphinx/ext/autodoc/__init__.py | 1807 | 1822| 129 | 46398 | 157039 | 
| 130 | 29 sphinx/writers/texinfo.py | 1083 | 1182| 839 | 47237 | 157039 | 
| 131 | 29 sphinx/writers/texinfo.py | 1490 | 1561| 519 | 47756 | 157039 | 
| 132 | 29 sphinx/ext/autodoc/__init__.py | 1144 | 1174| 287 | 48043 | 157039 | 
| 133 | 29 sphinx/ext/autosummary/__init__.py | 721 | 756| 325 | 48368 | 157039 | 
| 134 | 29 sphinx/ext/autodoc/__init__.py | 256 | 293| 311 | 48679 | 157039 | 
| 135 | 30 sphinx/util/cfamily.py | 79 | 116| 271 | 48950 | 160344 | 
| 136 | 30 sphinx/writers/html5.py | 653 | 739| 686 | 49636 | 160344 | 


## Patch

```diff
diff --git a/sphinx/util/docfields.py b/sphinx/util/docfields.py
--- a/sphinx/util/docfields.py
+++ b/sphinx/util/docfields.py
@@ -298,7 +298,7 @@ def transform(self, node: nodes.field_list) -> None:
             # also support syntax like ``:param type name:``
             if typedesc.is_typed:
                 try:
-                    argtype, argname = fieldarg.split(None, 1)
+                    argtype, argname = fieldarg.rsplit(None, 1)
                 except ValueError:
                     pass
                 else:

```

## Test Patch

```diff
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -922,7 +922,8 @@ def test_info_field_list(app):
             "   :param age: blah blah\n"
             "   :type age: int\n"
             "   :param items: blah blah\n"
-            "   :type items: Tuple[str, ...]\n")
+            "   :type items: Tuple[str, ...]\n"
+            "   :param Dict[str, str] params: blah blah\n")
     doctree = restructuredtext.parse(app, text)
     print(doctree)
 
@@ -936,6 +937,7 @@ def test_info_field_list(app):
     assert_node(doctree[3][1][0][0],
                 ([nodes.field_name, "Parameters"],
                  [nodes.field_body, nodes.bullet_list, ([nodes.list_item, nodes.paragraph],
+                                                        [nodes.list_item, nodes.paragraph],
                                                         [nodes.list_item, nodes.paragraph],
                                                         [nodes.list_item, nodes.paragraph])]))
 
@@ -983,6 +985,29 @@ def test_info_field_list(app):
                 refdomain="py", reftype="class", reftarget="str",
                 **{"py:module": "example", "py:class": "Class"})
 
+    # :param Dict[str, str] params:
+    assert_node(doctree[3][1][0][0][1][0][3][0],
+                ([addnodes.literal_strong, "params"],
+                 " (",
+                 [pending_xref, addnodes.literal_emphasis, "Dict"],
+                 [addnodes.literal_emphasis, "["],
+                 [pending_xref, addnodes.literal_emphasis, "str"],
+                 [addnodes.literal_emphasis, ", "],
+                 [pending_xref, addnodes.literal_emphasis, "str"],
+                 [addnodes.literal_emphasis, "]"],
+                 ")",
+                 " -- ",
+                 "blah blah"))
+    assert_node(doctree[3][1][0][0][1][0][3][0][2], pending_xref,
+                refdomain="py", reftype="class", reftarget="Dict",
+                **{"py:module": "example", "py:class": "Class"})
+    assert_node(doctree[3][1][0][0][1][0][3][0][4], pending_xref,
+                refdomain="py", reftype="class", reftarget="str",
+                **{"py:module": "example", "py:class": "Class"})
+    assert_node(doctree[3][1][0][0][1][0][3][0][6], pending_xref,
+                refdomain="py", reftype="class", reftarget="str",
+                **{"py:module": "example", "py:class": "Class"})
+
 
 def test_info_field_list_var(app):
     text = (".. py:class:: Class\n"

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 2711, End line: 2754

```python
# NOQA


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_autodocumenter(ModuleDocumenter)
    app.add_autodocumenter(ClassDocumenter)
    app.add_autodocumenter(ExceptionDocumenter)
    app.add_autodocumenter(DataDocumenter)
    app.add_autodocumenter(NewTypeDataDocumenter)
    app.add_autodocumenter(FunctionDocumenter)
    app.add_autodocumenter(DecoratorDocumenter)
    app.add_autodocumenter(MethodDocumenter)
    app.add_autodocumenter(AttributeDocumenter)
    app.add_autodocumenter(PropertyDocumenter)
    app.add_autodocumenter(NewTypeAttributeDocumenter)

    app.add_config_value('autoclass_content', 'class', True, ENUM('both', 'class', 'init'))
    app.add_config_value('autodoc_member_order', 'alphabetical', True,
                         ENUM('alphabetic', 'alphabetical', 'bysource', 'groupwise'))
    app.add_config_value('autodoc_class_signature', 'mixed', True, ENUM('mixed', 'separated'))
    app.add_config_value('autodoc_default_options', {}, True)
    app.add_config_value('autodoc_docstring_signature', True, True)
    app.add_config_value('autodoc_mock_imports', [], True)
    app.add_config_value('autodoc_typehints', "signature", True,
                         ENUM("signature", "description", "none", "both"))
    app.add_config_value('autodoc_typehints_description_target', 'all', True,
                         ENUM('all', 'documented'))
    app.add_config_value('autodoc_type_aliases', {}, True)
    app.add_config_value('autodoc_warningiserror', True, True)
    app.add_config_value('autodoc_inherit_docstrings', True, True)
    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')
    app.add_event('autodoc-process-bases')

    app.connect('config-inited', migrate_autodoc_member_order, priority=800)

    app.setup_extension('sphinx.ext.autodoc.preserve_defaults')
    app.setup_extension('sphinx.ext.autodoc.type_comment')
    app.setup_extension('sphinx.ext.autodoc.typehints')

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 2592, End line: 2617

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
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
            return super().get_doc(ignore)
        finally:
            self.config.autodoc_inherit_docstrings = orig  # type: ignore

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        # Disable analyzing attribute comment on Documenter.add_content() to control it on
        # AttributeDocumenter.add_content()
        self.analyzer = None

        if more_content is None:
            more_content = StringList()
        self.update_content(more_content)
        super().add_content(more_content, no_docstring)
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 2527, End line: 2550

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        if inspect.isenumattribute(self.object):
            self.object = self.object.value
        if self.parent:
            self.update_annotations(self.parent)

        return ret

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            if doc:
                docstring, metadata = separate_metadata('\n'.join(sum(doc, [])))
                if 'hide-value' in metadata:
                    return True

        return False
```
### 4 - sphinx/ext/autodoc/deprecated.py:

Start line: 32, End line: 47

```python
class DataDeclarationDocumenter(DataDocumenter):
    """
    Specialized Documenter subclass for data that cannot be imported
    because they are declared without initial value (refs: PEP-526).
    """
    objtype = 'datadecl'
    directivetype = 'data'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("%s is deprecated." % self.__class__.__name__,
                      RemovedInSphinx50Warning, stacklevel=2)
        super().__init__(*args, **kwargs)
```
### 5 - sphinx/writers/html5.py:

Start line: 153, End line: 209

```python
class HTML5Translator(SphinxTranslator, BaseTranslator):

    def visit_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">(</span>')
        self.first_param = 1
        self.optional_param_level = 0
        # How many required parameters are left.
        self.required_params_left = sum([isinstance(c, addnodes.desc_parameter)
                                         for c in node.children])
        self.param_separator = node.child_text_separator

    def depart_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">)</span>')

    # If required parameters are still to come, then put the comma after
    # the parameter.  Otherwise, put the comma before.  This ensures that
    # signatures like the following render correctly (see issue #1001):
    #
    #     foo([a, ]b, c[, d])
    #
    def visit_desc_parameter(self, node: Element) -> None:
        if self.first_param:
            self.first_param = 0
        elif not self.required_params_left:
            self.body.append(self.param_separator)
        if self.optional_param_level == 0:
            self.required_params_left -= 1
        if not node.hasattr('noemph'):
            self.body.append('<em class="sig-param">')

    def depart_desc_parameter(self, node: Element) -> None:
        if not node.hasattr('noemph'):
            self.body.append('</em>')
        if self.required_params_left:
            self.body.append(self.param_separator)

    def visit_desc_optional(self, node: Element) -> None:
        self.optional_param_level += 1
        self.body.append('<span class="optional">[</span>')

    def depart_desc_optional(self, node: Element) -> None:
        self.optional_param_level -= 1
        self.body.append('<span class="optional">]</span>')

    def visit_desc_annotation(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'em', '', CLASS='property'))

    def depart_desc_annotation(self, node: Element) -> None:
        self.body.append('</em>')

    ##############################################

    def visit_versionmodified(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'div', CLASS=node['type']))

    def depart_versionmodified(self, node: Element) -> None:
        self.body.append('</div>\n')

    # overwritten
```
### 6 - sphinx/writers/html.py:

Start line: 182, End line: 238

```python
class HTMLTranslator(SphinxTranslator, BaseTranslator):

    def visit_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">(</span>')
        self.first_param = 1
        self.optional_param_level = 0
        # How many required parameters are left.
        self.required_params_left = sum([isinstance(c, addnodes.desc_parameter)
                                         for c in node.children])
        self.param_separator = node.child_text_separator

    def depart_desc_parameterlist(self, node: Element) -> None:
        self.body.append('<span class="sig-paren">)</span>')

    # If required parameters are still to come, then put the comma after
    # the parameter.  Otherwise, put the comma before.  This ensures that
    # signatures like the following render correctly (see issue #1001):
    #
    #     foo([a, ]b, c[, d])
    #
    def visit_desc_parameter(self, node: Element) -> None:
        if self.first_param:
            self.first_param = 0
        elif not self.required_params_left:
            self.body.append(self.param_separator)
        if self.optional_param_level == 0:
            self.required_params_left -= 1
        if not node.hasattr('noemph'):
            self.body.append('<em>')

    def depart_desc_parameter(self, node: Element) -> None:
        if not node.hasattr('noemph'):
            self.body.append('</em>')
        if self.required_params_left:
            self.body.append(self.param_separator)

    def visit_desc_optional(self, node: Element) -> None:
        self.optional_param_level += 1
        self.body.append('<span class="optional">[</span>')

    def depart_desc_optional(self, node: Element) -> None:
        self.optional_param_level -= 1
        self.body.append('<span class="optional">]</span>')

    def visit_desc_annotation(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'em', '', CLASS='property'))

    def depart_desc_annotation(self, node: Element) -> None:
        self.body.append('</em>')

    ##############################################

    def visit_versionmodified(self, node: Element) -> None:
        self.body.append(self.starttag(node, 'div', CLASS=node['type']))

    def depart_versionmodified(self, node: Element) -> None:
        self.body.append('</div>\n')

    # overwritten
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 2058, End line: 2259

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
        if inspect.iscoroutinefunction(obj):
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
        else:
            return None

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
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
            __new__ = self.get_attr(self.object, '__new__', None)
            if __new__:
                docstring = getdoc(__new__, self.get_attr,
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
### 8 - sphinx/ext/autodoc/typehints.py:

Start line: 130, End line: 185

```python
def augment_descriptions_with_types(
    node: nodes.field_list,
    annotations: Dict[str, str],
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
        elif parts[0] == 'return':
            has_description.add('return')
        elif parts[0] == 'rtype':
            has_type.add('return')

    # Add 'type' for parameters with a description but no declared type.
    for name in annotations:
        if name == 'return':
            continue
        if name in has_description and name not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotations[name]))
            node += field

    # Add 'rtype' if 'return' is present and 'rtype' isn't.
    if 'return' in annotations:
        if 'return' in has_description and 'return' not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'rtype')
            field += nodes.field_body('', nodes.paragraph('', annotations['return']))
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
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 1426, End line: 1766

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
            # show __init__() method
            if self.options.special_members is None:
                self.options['special-members'] = {'__new__', '__init__'}
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

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''
        if self.config.autodoc_class_signature == 'separated':
            # do not show signatures
            return ''

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
                        # the constructor is defined in the class, but not overrided.
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
            if hasattr(self.object, '__orig_bases__') and len(self.object.__orig_bases__):
                # A subclass of generic types
                # refs: PEP-560 <https://www.python.org/dev/peps/pep-0560/>
                bases = list(self.object.__orig_bases__)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                # A normal class
                bases = list(self.object.__bases__)
            else:
                bases = []

            self.env.events.emit('autodoc-process-bases',
                                 self.fullname, self.object, self.options, bases)

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

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
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
        attrdocstring = self.get_attr(self.object, '__doc__', None)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if classdoc_from in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.config.autodoc_inherit_docstrings,
                                   self.parent, self.object_name)
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
                                       self.parent, self.object_name)
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
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def get_variable_comment(self) -> Optional[List[str]]:
        try:
            key = ('', '.'.join(self.objpath))
            analyzer = ModuleAnalyzer.for_module(self.get_real_modname())
            analyzer.analyze()
            return list(self.analyzer.attr_docs.get(key, []))
        except PycodeError:
            return None

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        if self.doc_as_attr and not self.get_variable_comment():
            try:
                more_content = StringList([_('alias of %s') % restify(self.object)], source='')
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
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 1273, End line: 1392

```python
class FunctionDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for functions.
    """
    objtype = 'function'
    member_order = 30

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        # supports functions, builtins and bound methods exported at the module level
        return (inspect.isfunction(member) or inspect.isbuiltin(member) or
                (inspect.isroutine(member) and isinstance(parent, ModuleDocumenter)))

    def format_args(self, **kwargs: Any) -> str:
        if self.config.autodoc_typehints in ('none', 'description'):
            kwargs.setdefault('show_annotation', False)

        try:
            self.env.app.emit('autodoc-before-process-signature', self.object, False)
            sig = inspect.signature(self.object, type_aliases=self.config.autodoc_type_aliases)
            args = stringify_signature(sig, **kwargs)
        except TypeError as exc:
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            args = ''

        if self.config.strip_signature_backslash:
            # escape backslashes for reST
            args = args.replace('\\', '\\\\')
        return args

    def document_members(self, all_members: bool = False) -> None:
        pass

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if inspect.iscoroutinefunction(self.object):
            self.add_line('   :async:', sourcename)

    def format_signature(self, **kwargs: Any) -> str:
        sigs = []
        if (self.analyzer and
                '.'.join(self.objpath) in self.analyzer.overloads and
                self.config.autodoc_typehints != 'none'):
            # Use signatures for overloaded functions instead of the implementation function.
            overloaded = True
        else:
            overloaded = False
            sig = super().format_signature(**kwargs)
            sigs.append(sig)

        if inspect.is_singledispatch_function(self.object):
            # append signature of singledispatch'ed functions
            for typ, func in self.object.registry.items():
                if typ is object:
                    pass  # default implementation. skipped.
                else:
                    dispatchfunc = self.annotate_to_first_argument(func, typ)
                    if dispatchfunc:
                        documenter = FunctionDocumenter(self.directive, '')
                        documenter.object = dispatchfunc
                        documenter.objpath = [None]
                        sigs.append(documenter.format_signature())
        if overloaded:
            actual = inspect.signature(self.object,
                                       type_aliases=self.config.autodoc_type_aliases)
            __globals__ = safe_getattr(self.object, '__globals__', {})
            for overload in self.analyzer.overloads.get('.'.join(self.objpath)):
                overload = self.merge_default_value(actual, overload)
                overload = evaluate_signature(overload, __globals__,
                                              self.config.autodoc_type_aliases)

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
            logger.warning(__("Failed to get a function signature for %s: %s"),
                           self.fullname, exc)
            return None
        except ValueError:
            return None

        if len(sig.parameters) == 0:
            return None

        def dummy():
            pass

        params = list(sig.parameters.values())
        if params[0].annotation is Parameter.empty:
            params[0] = params[0].replace(annotation=typ)
            try:
                dummy.__signature__ = sig.replace(parameters=params)  # type: ignore
                return dummy
            except (AttributeError, TypeError):
                # failed to update signature (ex. built-in or extension types)
                return None
        else:
            return None
```
### 51 - sphinx/util/docfields.py:

Start line: 144, End line: 170

```python
class TypedField(GroupedField):
    """
    A doc field that is grouped and has type information for the arguments.  It
    always has an argument.  The argument can be linked using the given
    *rolename*, the type using the given *typerolename*.

    Two uses are possible: either parameter and type description are given
    separately, using a field from *names* and one from *typenames*,
    respectively, or both are given using a field from *names*, see the example.

    Example::

       :param foo: description of parameter foo
       :type foo:  SomeClass

       -- or --

       :param SomeClass foo: description of parameter foo
    """
    is_typed = True

    def __init__(self, name: str, names: Tuple[str, ...] = (), typenames: Tuple[str, ...] = (),
                 label: str = None, rolename: str = None, typerolename: str = None,
                 can_collapse: bool = False) -> None:
        super().__init__(name, names, label, rolename, can_collapse)
        self.typenames = typenames
        self.typerolename = typerolename
```
### 121 - sphinx/util/docfields.py:

Start line: 172, End line: 204

```python
class TypedField(GroupedField):

    def make_field(self, types: Dict[str, List[Node]], domain: str,
                   items: Tuple, env: BuildEnvironment = None) -> nodes.field:
        def handle_item(fieldarg: str, content: str) -> nodes.paragraph:
            par = nodes.paragraph()
            par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
                                       addnodes.literal_strong, env=env))
            if fieldarg in types:
                par += nodes.Text(' (')
                # NOTE: using .pop() here to prevent a single type node to be
                # inserted twice into the doctree, which leads to
                # inconsistencies later when references are resolved
                fieldtype = types.pop(fieldarg)
                if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                    typename = fieldtype[0].astext()
                    par.extend(self.make_xrefs(self.typerolename, domain, typename,
                                               addnodes.literal_emphasis, env=env))
                else:
                    par += fieldtype
                par += nodes.Text(')')
            par += nodes.Text(' -- ')
            par += content
            return par

        fieldname = nodes.field_name('', self.label)
        if len(items) == 1 and self.can_collapse:
            fieldarg, content = items[0]
            bodynode: Node = handle_item(fieldarg, content)
        else:
            bodynode = self.list_type()
            for fieldarg, content in items:
                bodynode += nodes.list_item('', handle_item(fieldarg, content))
        fieldbody = nodes.field_body('', bodynode)
        return nodes.field('', fieldname, fieldbody)
```
