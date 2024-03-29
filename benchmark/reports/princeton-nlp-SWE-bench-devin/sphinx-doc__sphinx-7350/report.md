# sphinx-doc__sphinx-7350

| **sphinx-doc/sphinx** | `c75470f9b79046f6d32344be5eacf60a4e1c1b7d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1770 |
| **Any found context length** | 1460 |
| **Avg pos** | 15.0 |
| **Min pos** | 7 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/napoleon/docstring.py b/sphinx/ext/napoleon/docstring.py
--- a/sphinx/ext/napoleon/docstring.py
+++ b/sphinx/ext/napoleon/docstring.py
@@ -583,7 +583,11 @@ def _parse_attributes_section(self, section: str) -> List[str]:
                 if _type:
                     lines.append(':vartype %s: %s' % (_name, _type))
             else:
-                lines.extend(['.. attribute:: ' + _name, ''])
+                lines.append('.. attribute:: ' + _name)
+                if self._opt and 'noindex' in self._opt:
+                    lines.append('   :noindex:')
+                lines.append('')
+
                 fields = self._format_field('', '', _desc)
                 lines.extend(self._indent(fields, 3))
                 if _type:
@@ -641,6 +645,8 @@ def _parse_methods_section(self, section: str) -> List[str]:
         lines = []  # type: List[str]
         for _name, _type, _desc in self._consume_fields(parse_type=False):
             lines.append('.. method:: %s' % _name)
+            if self._opt and 'noindex' in self._opt:
+                lines.append('   :noindex:')
             if _desc:
                 lines.extend([''] + self._indent(_desc, 3))
             lines.append('')

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/napoleon/docstring.py | 586 | 586 | 7 | 1 | 1460
| sphinx/ext/napoleon/docstring.py | 644 | 644 | 8 | 1 | 1770


## Problem Statement

```
Napoleon's Attributes directive ignores :noindex: option.
**Description of the bug**
Sphinxcontrib-napoleon's `Attributes:` directive appears to ignore the `:noindex:` option. 

The following reST code produces an index that includes the `Attributes:` directives found in `example_google.py` but leaves out all other directives:

\`\`\`reST
Google Example
==============

.. automodule:: example_google
   :members:
   :noindex:

:ref:`genindex`
\`\`\`


**Expected behavior**
The above example should produce an empty document index.


**Environment info**
I am using the Sphinx packages that are provided by Ubuntu 18.04 and installed Napoleon with pip3 afterwards:

\`\`\`
apt install make python3-sphinx python3-pip
pip3 install sphinxcontrib-napoleon
\`\`\`

The file `example_google.py` is from https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

I used `sphinx-quickstart` to configure my directory, edited `conf.py` to include `sphinxcontrib-napoleon` and set the Python path, then typed `make html`.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sphinx/ext/napoleon/docstring.py** | 13 | 38| 311 | 311 | 8743 | 
| 2 | 2 doc/usage/extensions/example_google.py | 277 | 297| 120 | 431 | 10728 | 
| 3 | 3 sphinx/ext/napoleon/__init__.py | 252 | 272| 258 | 689 | 14491 | 
| 4 | 3 sphinx/ext/napoleon/__init__.py | 11 | 272| 50 | 739 | 14491 | 
| 5 | **3 sphinx/ext/napoleon/docstring.py** | 287 | 327| 315 | 1054 | 14491 | 
| 6 | **3 sphinx/ext/napoleon/docstring.py** | 597 | 615| 195 | 1249 | 14491 | 
| **-> 7 <-** | **3 sphinx/ext/napoleon/docstring.py** | 576 | 595| 211 | 1460 | 14491 | 
| **-> 8 <-** | **3 sphinx/ext/napoleon/docstring.py** | 630 | 661| 310 | 1770 | 14491 | 
| 9 | **3 sphinx/ext/napoleon/docstring.py** | 268 | 285| 148 | 1918 | 14491 | 
| 10 | 3 sphinx/ext/napoleon/__init__.py | 19 | 251| 1791 | 3709 | 14491 | 
| 11 | **3 sphinx/ext/napoleon/docstring.py** | 476 | 491| 130 | 3839 | 14491 | 
| 12 | **3 sphinx/ext/napoleon/docstring.py** | 742 | 770| 213 | 4052 | 14491 | 
| 13 | **3 sphinx/ext/napoleon/docstring.py** | 433 | 474| 345 | 4397 | 14491 | 
| 14 | 3 doc/usage/extensions/example_google.py | 261 | 275| 109 | 4506 | 14491 | 
| 15 | **3 sphinx/ext/napoleon/docstring.py** | 617 | 628| 124 | 4630 | 14491 | 
| 16 | 4 doc/conf.py | 1 | 80| 767 | 5397 | 16027 | 
| 17 | **4 sphinx/ext/napoleon/docstring.py** | 564 | 574| 124 | 5521 | 16027 | 
| 18 | **4 sphinx/ext/napoleon/docstring.py** | 530 | 562| 268 | 5789 | 16027 | 
| 19 | **4 sphinx/ext/napoleon/docstring.py** | 710 | 740| 277 | 6066 | 16027 | 
| 20 | **4 sphinx/ext/napoleon/docstring.py** | 248 | 266| 222 | 6288 | 16027 | 
| 21 | **4 sphinx/ext/napoleon/docstring.py** | 663 | 680| 217 | 6505 | 16027 | 
| 22 | **4 sphinx/ext/napoleon/docstring.py** | 889 | 925| 340 | 6845 | 16027 | 
| 23 | 5 doc/usage/extensions/example_numpy.py | 336 | 356| 120 | 6965 | 18135 | 
| 24 | **5 sphinx/ext/napoleon/docstring.py** | 226 | 246| 215 | 7180 | 18135 | 
| 25 | **5 sphinx/ext/napoleon/docstring.py** | 329 | 340| 119 | 7299 | 18135 | 
| 26 | **5 sphinx/ext/napoleon/docstring.py** | 493 | 513| 195 | 7494 | 18135 | 
| 27 | 6 sphinx/directives/__init__.py | 259 | 297| 273 | 7767 | 20558 | 
| 28 | 6 doc/usage/extensions/example_numpy.py | 320 | 334| 109 | 7876 | 20558 | 
| 29 | **6 sphinx/ext/napoleon/docstring.py** | 342 | 366| 235 | 8111 | 20558 | 
| 30 | **6 sphinx/ext/napoleon/docstring.py** | 368 | 383| 180 | 8291 | 20558 | 
| 31 | **6 sphinx/ext/napoleon/docstring.py** | 201 | 224| 201 | 8492 | 20558 | 
| 32 | **6 sphinx/ext/napoleon/docstring.py** | 515 | 528| 164 | 8656 | 20558 | 
| 33 | **6 sphinx/ext/napoleon/docstring.py** | 872 | 887| 192 | 8848 | 20558 | 
| 34 | **6 sphinx/ext/napoleon/docstring.py** | 682 | 708| 226 | 9074 | 20558 | 
| 35 | **6 sphinx/ext/napoleon/docstring.py** | 385 | 412| 258 | 9332 | 20558 | 
| 36 | 7 sphinx/directives/other.py | 9 | 40| 238 | 9570 | 23715 | 
| 37 | **7 sphinx/ext/napoleon/docstring.py** | 107 | 199| 805 | 10375 | 23715 | 
| 38 | **7 sphinx/ext/napoleon/docstring.py** | 41 | 105| 599 | 10974 | 23715 | 
| 39 | 7 sphinx/ext/napoleon/__init__.py | 381 | 464| 738 | 11712 | 23715 | 
| 40 | 7 sphinx/ext/napoleon/__init__.py | 312 | 328| 158 | 11870 | 23715 | 
| 41 | 8 sphinx/domains/rst.py | 141 | 172| 356 | 12226 | 26185 | 
| 42 | 9 sphinx/domains/index.py | 96 | 130| 281 | 12507 | 27119 | 
| 43 | 9 doc/conf.py | 81 | 140| 513 | 13020 | 27119 | 
| 44 | 10 sphinx/directives/code.py | 417 | 481| 658 | 13678 | 31016 | 
| 45 | **10 sphinx/ext/napoleon/docstring.py** | 773 | 870| 745 | 14423 | 31016 | 
| 46 | **10 sphinx/ext/napoleon/docstring.py** | 414 | 431| 181 | 14604 | 31016 | 
| 47 | 10 sphinx/directives/code.py | 9 | 31| 146 | 14750 | 31016 | 
| 48 | 11 sphinx/directives/patches.py | 9 | 25| 124 | 14874 | 32629 | 
| 49 | 12 sphinx/transforms/__init__.py | 11 | 44| 226 | 15100 | 35776 | 
| 50 | 13 sphinx/domains/python.py | 779 | 791| 132 | 15232 | 46617 | 
| 51 | 14 setup.py | 171 | 246| 626 | 15858 | 48311 | 
| 52 | **14 sphinx/ext/napoleon/docstring.py** | 927 | 1033| 762 | 16620 | 48311 | 
| 53 | 14 sphinx/domains/python.py | 879 | 897| 143 | 16763 | 48311 | 
| 54 | 14 sphinx/directives/patches.py | 54 | 68| 143 | 16906 | 48311 | 
| 55 | 15 sphinx/ext/apidoc.py | 347 | 412| 751 | 17657 | 53012 | 
| 56 | 15 sphinx/directives/other.py | 88 | 150| 602 | 18259 | 53012 | 
| 57 | 15 sphinx/directives/other.py | 153 | 184| 213 | 18472 | 53012 | 
| 58 | 15 sphinx/directives/other.py | 347 | 371| 186 | 18658 | 53012 | 
| 59 | 16 sphinx/domains/std.py | 11 | 48| 323 | 18981 | 62964 | 
| 60 | 16 sphinx/directives/other.py | 374 | 398| 228 | 19209 | 62964 | 
| 61 | 16 sphinx/domains/python.py | 11 | 67| 432 | 19641 | 62964 | 
| 62 | 16 sphinx/domains/index.py | 11 | 30| 127 | 19768 | 62964 | 
| 63 | 17 sphinx/ext/doctest.py | 157 | 191| 209 | 19977 | 67914 | 
| 64 | 18 sphinx/builders/html/__init__.py | 867 | 885| 225 | 20202 | 79088 | 
| 65 | 18 sphinx/domains/index.py | 64 | 93| 230 | 20432 | 79088 | 
| 66 | 18 sphinx/domains/rst.py | 11 | 33| 161 | 20593 | 79088 | 
| 67 | 18 sphinx/domains/rst.py | 260 | 286| 274 | 20867 | 79088 | 
| 68 | 19 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 21165 | 80344 | 
| 69 | 20 sphinx/ext/autosummary/__init__.py | 716 | 749| 313 | 21478 | 86716 | 
| 70 | 21 sphinx/domains/javascript.py | 11 | 34| 190 | 21668 | 90731 | 
| 71 | 22 sphinx/search/__init__.py | 10 | 30| 135 | 21803 | 94739 | 
| 72 | 22 doc/usage/extensions/example_google.py | 1 | 37| 254 | 22057 | 94739 | 
| 73 | 22 sphinx/search/__init__.py | 322 | 357| 346 | 22403 | 94739 | 
| 74 | 22 sphinx/domains/std.py | 1045 | 1081| 323 | 22726 | 94739 | 
| 75 | 22 sphinx/directives/other.py | 289 | 344| 454 | 23180 | 94739 | 
| 76 | 23 sphinx/transforms/references.py | 11 | 68| 362 | 23542 | 95153 | 
| 77 | 24 sphinx/setup_command.py | 14 | 91| 419 | 23961 | 96835 | 
| 78 | 24 sphinx/builders/html/__init__.py | 11 | 67| 457 | 24418 | 96835 | 
| 79 | 24 sphinx/ext/napoleon/__init__.py | 331 | 378| 453 | 24871 | 96835 | 
| 80 | 24 sphinx/domains/std.py | 243 | 260| 125 | 24996 | 96835 | 
| 81 | 24 sphinx/ext/apidoc.py | 267 | 344| 738 | 25734 | 96835 | 
| 82 | 25 sphinx/cmd/quickstart.py | 550 | 617| 491 | 26225 | 102379 | 
| 83 | 26 sphinx/builders/latex/constants.py | 70 | 119| 529 | 26754 | 104465 | 
| 84 | 26 sphinx/domains/python.py | 167 | 185| 214 | 26968 | 104465 | 
| 85 | 27 sphinx/parsers.py | 71 | 119| 360 | 27328 | 105328 | 
| 86 | 27 sphinx/directives/other.py | 64 | 86| 272 | 27600 | 105328 | 
| 87 | 27 sphinx/directives/__init__.py | 11 | 49| 254 | 27854 | 105328 | 
| 88 | 27 sphinx/search/__init__.py | 385 | 401| 172 | 28026 | 105328 | 
| 89 | 28 sphinx/environment/__init__.py | 11 | 82| 489 | 28515 | 111151 | 
| 90 | 29 sphinx/roles.py | 576 | 597| 218 | 28733 | 116813 | 
| 91 | 30 sphinx/testing/util.py | 141 | 156| 195 | 28928 | 118576 | 
| 92 | 30 sphinx/directives/code.py | 191 | 220| 248 | 29176 | 118576 | 
| 93 | 30 sphinx/domains/python.py | 602 | 656| 530 | 29706 | 118576 | 
| 94 | 31 sphinx/builders/_epub_base.py | 11 | 102| 662 | 30368 | 125311 | 
| 95 | 32 utils/doclinter.py | 11 | 81| 467 | 30835 | 125830 | 
| 96 | 32 sphinx/domains/rst.py | 92 | 115| 202 | 31037 | 125830 | 
| 97 | 32 sphinx/builders/html/__init__.py | 658 | 685| 270 | 31307 | 125830 | 
| 98 | 32 setup.py | 1 | 74| 438 | 31745 | 125830 | 
| 99 | 33 sphinx/project.py | 11 | 26| 111 | 31856 | 126551 | 
| 100 | 33 sphinx/domains/python.py | 938 | 1008| 583 | 32439 | 126551 | 
| 101 | 33 sphinx/ext/doctest.py | 12 | 51| 281 | 32720 | 126551 | 
| 102 | 33 sphinx/search/__init__.py | 160 | 189| 194 | 32914 | 126551 | 
| 103 | 33 sphinx/domains/rst.py | 230 | 236| 113 | 33027 | 126551 | 
| 104 | 33 sphinx/domains/rst.py | 249 | 258| 134 | 33161 | 126551 | 
| 105 | 33 sphinx/directives/__init__.py | 300 | 315| 122 | 33283 | 126551 | 
| 106 | 34 sphinx/search/no.py | 198 | 216| 116 | 33399 | 131661 | 
| 107 | 35 sphinx/builders/manpage.py | 11 | 31| 156 | 33555 | 132569 | 
| 108 | 36 sphinx/builders/epub3.py | 12 | 53| 276 | 33831 | 135258 | 
| 109 | 37 sphinx/util/docutils.py | 11 | 112| 796 | 34627 | 139386 | 
| 110 | 38 sphinx/registry.py | 11 | 50| 307 | 34934 | 143876 | 
| 111 | 38 sphinx/directives/patches.py | 209 | 223| 119 | 35053 | 143876 | 
| 112 | 38 sphinx/domains/rst.py | 118 | 139| 186 | 35239 | 143876 | 
| 113 | 38 sphinx/builders/html/__init__.py | 1184 | 1251| 815 | 36054 | 143876 | 
| 114 | 38 sphinx/testing/util.py | 10 | 49| 297 | 36351 | 143876 | 
| 115 | 38 sphinx/search/__init__.py | 436 | 455| 178 | 36529 | 143876 | 
| 116 | 38 sphinx/domains/std.py | 214 | 240| 274 | 36803 | 143876 | 
| 117 | 38 sphinx/domains/python.py | 900 | 920| 248 | 37051 | 143876 | 
| 118 | 38 sphinx/domains/python.py | 469 | 479| 145 | 37196 | 143876 | 
| 119 | 39 sphinx/application.py | 63 | 125| 494 | 37690 | 154498 | 
| 120 | 40 sphinx/config.py | 98 | 155| 709 | 38399 | 158863 | 
| 121 | 40 sphinx/util/docutils.py | 228 | 238| 163 | 38562 | 158863 | 
| 122 | 41 sphinx/io.py | 10 | 46| 274 | 38836 | 160554 | 
| 123 | 41 sphinx/roles.py | 544 | 573| 328 | 39164 | 160554 | 
| 124 | 41 sphinx/ext/apidoc.py | 236 | 264| 249 | 39413 | 160554 | 
| 125 | 41 sphinx/testing/util.py | 90 | 139| 443 | 39856 | 160554 | 
| 126 | 41 sphinx/ext/autosummary/__init__.py | 55 | 106| 362 | 40218 | 160554 | 
| 127 | 42 doc/development/tutorials/examples/todo.py | 1 | 27| 111 | 40329 | 161392 | 
| 128 | 42 doc/development/tutorials/examples/todo.py | 30 | 61| 230 | 40559 | 161392 | 
| 129 | 42 sphinx/directives/patches.py | 28 | 51| 187 | 40746 | 161392 | 
| 130 | 42 sphinx/domains/rst.py | 174 | 200| 223 | 40969 | 161392 | 
| 131 | 42 sphinx/application.py | 13 | 60| 378 | 41347 | 161392 | 
| 132 | 43 sphinx/domains/citation.py | 11 | 31| 126 | 41473 | 162672 | 
| 133 | 43 sphinx/application.py | 337 | 386| 410 | 41883 | 162672 | 
| 134 | 43 sphinx/directives/other.py | 43 | 62| 151 | 42034 | 162672 | 
| 135 | 44 sphinx/ext/autodoc/__init__.py | 628 | 692| 643 | 42677 | 178101 | 
| 136 | 45 sphinx/environment/adapters/indexentries.py | 57 | 102| 583 | 43260 | 179761 | 
| 137 | 46 doc/development/tutorials/examples/recipe.py | 1 | 33| 218 | 43478 | 180882 | 
| 138 | 47 sphinx/writers/manpage.py | 313 | 411| 757 | 44235 | 184393 | 
| 139 | 48 sphinx/writers/html.py | 681 | 780| 803 | 45038 | 191656 | 
| 140 | 48 sphinx/cmd/quickstart.py | 467 | 534| 755 | 45793 | 191656 | 
| 141 | 49 utils/checks.py | 33 | 109| 545 | 46338 | 192562 | 
| 142 | 49 sphinx/directives/__init__.py | 114 | 134| 151 | 46489 | 192562 | 
| 143 | 49 sphinx/ext/autodoc/__init__.py | 1731 | 1766| 437 | 46926 | 192562 | 
| 144 | 49 sphinx/roles.py | 11 | 47| 284 | 47210 | 192562 | 
| 145 | 49 sphinx/directives/__init__.py | 237 | 257| 165 | 47375 | 192562 | 
| 146 | 49 sphinx/directives/code.py | 382 | 415| 284 | 47659 | 192562 | 
| 147 | 49 sphinx/ext/apidoc.py | 413 | 441| 374 | 48033 | 192562 | 
| 148 | 50 sphinx/ext/inheritance_diagram.py | 38 | 66| 212 | 48245 | 196415 | 


### Hint

```
I'm hitting this bug as well, but I believe this bug is in `autodoc` not `napoleon`, since all `napoleon` does is convert Google/Numpy style docstrings to valid reST. As far as I can tell from the code, it doesn't set `:noindex:` for any other option so I don't see how Attributes is handled any differently.
```

## Patch

```diff
diff --git a/sphinx/ext/napoleon/docstring.py b/sphinx/ext/napoleon/docstring.py
--- a/sphinx/ext/napoleon/docstring.py
+++ b/sphinx/ext/napoleon/docstring.py
@@ -583,7 +583,11 @@ def _parse_attributes_section(self, section: str) -> List[str]:
                 if _type:
                     lines.append(':vartype %s: %s' % (_name, _type))
             else:
-                lines.extend(['.. attribute:: ' + _name, ''])
+                lines.append('.. attribute:: ' + _name)
+                if self._opt and 'noindex' in self._opt:
+                    lines.append('   :noindex:')
+                lines.append('')
+
                 fields = self._format_field('', '', _desc)
                 lines.extend(self._indent(fields, 3))
                 if _type:
@@ -641,6 +645,8 @@ def _parse_methods_section(self, section: str) -> List[str]:
         lines = []  # type: List[str]
         for _name, _type, _desc in self._consume_fields(parse_type=False):
             lines.append('.. method:: %s' % _name)
+            if self._opt and 'noindex' in self._opt:
+                lines.append('   :noindex:')
             if _desc:
                 lines.extend([''] + self._indent(_desc, 3))
             lines.append('')

```

## Test Patch

```diff
diff --git a/tests/test_ext_napoleon_docstring.py b/tests/test_ext_napoleon_docstring.py
--- a/tests/test_ext_napoleon_docstring.py
+++ b/tests/test_ext_napoleon_docstring.py
@@ -1020,6 +1020,34 @@ def test_custom_generic_sections(self):
             actual = str(GoogleDocstring(docstring, testConfig))
             self.assertEqual(expected, actual)
 
+    def test_noindex(self):
+        docstring = """
+Attributes:
+    arg
+        description
+
+Methods:
+    func(i, j)
+        description
+"""
+
+        expected = """
+.. attribute:: arg
+   :noindex:
+
+   description
+
+.. method:: func(i, j)
+   :noindex:
+
+   
+   description
+"""
+        config = Config()
+        actual = str(GoogleDocstring(docstring, config=config, app=None, what='module',
+                                     options={'noindex': True}))
+        self.assertEqual(expected, actual)
+
 
 class NumpyDocstringTest(BaseDocstringTest):
     docstrings = [(

```


## Code snippets

### 1 - sphinx/ext/napoleon/docstring.py:

Start line: 13, End line: 38

```python
import inspect
import re
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.iterators import modify_iter
from sphinx.locale import _

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


_directive_regex = re.compile(r'\.\. \S+::')
_google_section_regex = re.compile(r'^(\s|\w)+:\s*$')
_google_typed_arg_regex = re.compile(r'\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)')
_numpy_section_regex = re.compile(r'^[=\-`:\'"~^_*+#<>]{2,}\s*$')
_single_colon_regex = re.compile(r'(?<!:):(?!:)')
_xref_regex = re.compile(r'(:(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)')
_bullet_list_regex = re.compile(r'^(\*|\+|\-)(\s+\S|\s*$)')
_enumerated_list_regex = re.compile(
    r'^(?P<paren>\()?'
    r'(\d+|#|[ivxlcdm]+|[IVXLCDM]+|[a-zA-Z])'
    r'(?(paren)\)|\.)(\s+\S|\s*$)')
```
### 2 - doc/usage/extensions/example_google.py:

Start line: 277, End line: 297

```python
class ExampleClass:

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self):
        pass
```
### 3 - sphinx/ext/napoleon/__init__.py:

Start line: 252, End line: 272

```python
class Config:
    _config_values = {
        'napoleon_google_docstring': (True, 'env'),
        'napoleon_numpy_docstring': (True, 'env'),
        'napoleon_include_init_with_doc': (False, 'env'),
        'napoleon_include_private_with_doc': (False, 'env'),
        'napoleon_include_special_with_doc': (False, 'env'),
        'napoleon_use_admonition_for_examples': (False, 'env'),
        'napoleon_use_admonition_for_notes': (False, 'env'),
        'napoleon_use_admonition_for_references': (False, 'env'),
        'napoleon_use_ivar': (False, 'env'),
        'napoleon_use_param': (True, 'env'),
        'napoleon_use_rtype': (True, 'env'),
        'napoleon_use_keyword': (True, 'env'),
        'napoleon_custom_sections': (None, 'env')
    }

    def __init__(self, **settings: Any) -> None:
        for name, (default, rebuild) in self._config_values.items():
            setattr(self, name, default)
        for name, value in settings.items():
            setattr(self, name, value)
```
### 4 - sphinx/ext/napoleon/__init__.py:

Start line: 11, End line: 272

```python
from typing import Any, Dict, List

from sphinx import __display_version__ as __version__
from sphinx.application import Sphinx
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring


class Config:
```
### 5 - sphinx/ext/napoleon/docstring.py:

Start line: 287, End line: 327

```python
class GoogleDocstring:

    def _consume_usage_section(self) -> List[str]:
        lines = self._dedent(self._consume_to_next_section())
        return lines

    def _consume_section_header(self) -> str:
        section = next(self._line_iter)
        stripped_section = section.strip(':')
        if stripped_section.lower() in self._sections:
            section = stripped_section
        return section

    def _consume_to_end(self) -> List[str]:
        lines = []
        while self._line_iter.has_next():
            lines.append(next(self._line_iter))
        return lines

    def _consume_to_next_section(self) -> List[str]:
        self._consume_empty()
        lines = []
        while not self._is_section_break():
            lines.append(next(self._line_iter))
        return lines + self._consume_empty()

    def _dedent(self, lines: List[str], full: bool = False) -> List[str]:
        if full:
            return [line.lstrip() for line in lines]
        else:
            min_indent = self._get_min_indent(lines)
            return [line[min_indent:] for line in lines]

    def _escape_args_and_kwargs(self, name: str) -> str:
        if name.endswith('_'):
            name = name[:-1] + r'\_'

        if name[:2] == '**':
            return r'\*\*' + name[2:]
        elif name[:1] == '*':
            return r'\*' + name[1:]
        else:
            return name
```
### 6 - sphinx/ext/napoleon/docstring.py:

Start line: 597, End line: 615

```python
class GoogleDocstring:

    def _parse_examples_section(self, section: str) -> List[str]:
        labels = {
            'example': _('Example'),
            'examples': _('Examples'),
        }
        use_admonition = self._config.napoleon_use_admonition_for_examples
        label = labels.get(section.lower(), section)
        return self._parse_generic_section(label, use_admonition)

    def _parse_custom_generic_section(self, section: str) -> List[str]:
        # for now, no admonition for simple custom sections
        return self._parse_generic_section(section, False)

    def _parse_usage_section(self, section: str) -> List[str]:
        header = ['.. rubric:: Usage:', '']
        block = ['.. code-block:: python', '']
        lines = self._consume_usage_section()
        lines = self._indent(lines, 3)
        return header + block + lines + ['']
```
### 7 - sphinx/ext/napoleon/docstring.py:

Start line: 576, End line: 595

```python
class GoogleDocstring:

    def _parse_attributes_section(self, section: str) -> List[str]:
        lines = []
        for _name, _type, _desc in self._consume_fields():
            if self._config.napoleon_use_ivar:
                _name = self._qualify_name(_name, self._obj)
                field = ':ivar %s: ' % _name
                lines.extend(self._format_block(field, _desc))
                if _type:
                    lines.append(':vartype %s: %s' % (_name, _type))
            else:
                lines.extend(['.. attribute:: ' + _name, ''])
                fields = self._format_field('', '', _desc)
                lines.extend(self._indent(fields, 3))
                if _type:
                    lines.append('')
                    lines.extend(self._indent([':type: %s' % _type], 3))
                lines.append('')
        if self._config.napoleon_use_ivar:
            lines.append('')
        return lines
```
### 8 - sphinx/ext/napoleon/docstring.py:

Start line: 630, End line: 661

```python
class GoogleDocstring:

    def _parse_keyword_arguments_section(self, section: str) -> List[str]:
        fields = self._consume_fields()
        if self._config.napoleon_use_keyword:
            return self._format_docutils_params(
                fields,
                field_role="keyword",
                type_role="kwtype")
        else:
            return self._format_fields(_('Keyword Arguments'), fields)

    def _parse_methods_section(self, section: str) -> List[str]:
        lines = []  # type: List[str]
        for _name, _type, _desc in self._consume_fields(parse_type=False):
            lines.append('.. method:: %s' % _name)
            if _desc:
                lines.extend([''] + self._indent(_desc, 3))
            lines.append('')
        return lines

    def _parse_notes_section(self, section: str) -> List[str]:
        use_admonition = self._config.napoleon_use_admonition_for_notes
        return self._parse_generic_section(_('Notes'), use_admonition)

    def _parse_other_parameters_section(self, section: str) -> List[str]:
        return self._format_fields(_('Other Parameters'), self._consume_fields())

    def _parse_parameters_section(self, section: str) -> List[str]:
        fields = self._consume_fields()
        if self._config.napoleon_use_param:
            return self._format_docutils_params(fields)
        else:
            return self._format_fields(_('Parameters'), fields)
```
### 9 - sphinx/ext/napoleon/docstring.py:

Start line: 268, End line: 285

```python
class GoogleDocstring:

    def _consume_returns_section(self) -> List[Tuple[str, str, List[str]]]:
        lines = self._dedent(self._consume_to_next_section())
        if lines:
            before, colon, after = self._partition_field_on_colon(lines[0])
            _name, _type, _desc = '', '', lines

            if colon:
                if after:
                    _desc = [after] + lines[1:]
                else:
                    _desc = lines[1:]

                _type = before

            _desc = self.__class__(_desc, self._config).lines()
            return [(_name, _type, _desc,)]
        else:
            return []
```
### 10 - sphinx/ext/napoleon/__init__.py:

Start line: 19, End line: 251

```python
class Config:
    """Sphinx napoleon extension settings in `conf.py`.

    Listed below are all the settings used by napoleon and their default
    values. These settings can be changed in the Sphinx `conf.py` file. Make
    sure that "sphinx.ext.napoleon" is enabled in `conf.py`::

        # conf.py

        # Add any Sphinx extension module names here, as strings
        extensions = ['sphinx.ext.napoleon']

        # Napoleon settings
        napoleon_google_docstring = True
        napoleon_numpy_docstring = True
        napoleon_include_init_with_doc = False
        napoleon_include_private_with_doc = False
        napoleon_include_special_with_doc = False
        napoleon_use_admonition_for_examples = False
        napoleon_use_admonition_for_notes = False
        napoleon_use_admonition_for_references = False
        napoleon_use_ivar = False
        napoleon_use_param = True
        napoleon_use_rtype = True
        napoleon_use_keyword = True
        napoleon_custom_sections = None

    .. _Google style:
       https://google.github.io/styleguide/pyguide.html
    .. _NumPy style:
       https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

    Attributes
    ----------
    napoleon_google_docstring : :obj:`bool` (Defaults to True)
        True to parse `Google style`_ docstrings. False to disable support
        for Google style docstrings.
    napoleon_numpy_docstring : :obj:`bool` (Defaults to True)
        True to parse `NumPy style`_ docstrings. False to disable support
        for NumPy style docstrings.
    napoleon_include_init_with_doc : :obj:`bool` (Defaults to False)
        True to list ``__init___`` docstrings separately from the class
        docstring. False to fall back to Sphinx's default behavior, which
        considers the ``__init___`` docstring as part of the class
        documentation.

        **If True**::

            def __init__(self):
                \"\"\"
                This will be included in the docs because it has a docstring
                \"\"\"

            def __init__(self):
                # This will NOT be included in the docs

    napoleon_include_private_with_doc : :obj:`bool` (Defaults to False)
        True to include private members (like ``_membername``) with docstrings
        in the documentation. False to fall back to Sphinx's default behavior.

        **If True**::

            def _included(self):
                \"\"\"
                This will be included in the docs because it has a docstring
                \"\"\"
                pass

            def _skipped(self):
                # This will NOT be included in the docs
                pass

    napoleon_include_special_with_doc : :obj:`bool` (Defaults to False)
        True to include special members (like ``__membername__``) with
        docstrings in the documentation. False to fall back to Sphinx's
        default behavior.

        **If True**::

            def __str__(self):
                \"\"\"
                This will be included in the docs because it has a docstring
                \"\"\"
                return unicode(self).encode('utf-8')

            def __unicode__(self):
                # This will NOT be included in the docs
                return unicode(self.__class__.__name__)

    napoleon_use_admonition_for_examples : :obj:`bool` (Defaults to False)
        True to use the ``.. admonition::`` directive for the **Example** and
        **Examples** sections. False to use the ``.. rubric::`` directive
        instead. One may look better than the other depending on what HTML
        theme is used.

        This `NumPy style`_ snippet will be converted as follows::

            Example
            -------
            This is just a quick example

        **If True**::

            .. admonition:: Example

               This is just a quick example

        **If False**::

            .. rubric:: Example

            This is just a quick example

    napoleon_use_admonition_for_notes : :obj:`bool` (Defaults to False)
        True to use the ``.. admonition::`` directive for **Notes** sections.
        False to use the ``.. rubric::`` directive instead.

        Note
        ----
        The singular **Note** section will always be converted to a
        ``.. note::`` directive.

        See Also
        --------
        :attr:`napoleon_use_admonition_for_examples`

    napoleon_use_admonition_for_references : :obj:`bool` (Defaults to False)
        True to use the ``.. admonition::`` directive for **References**
        sections. False to use the ``.. rubric::`` directive instead.

        See Also
        --------
        :attr:`napoleon_use_admonition_for_examples`

    napoleon_use_ivar : :obj:`bool` (Defaults to False)
        True to use the ``:ivar:`` role for instance variables. False to use
        the ``.. attribute::`` directive instead.

        This `NumPy style`_ snippet will be converted as follows::

            Attributes
            ----------
            attr1 : int
                Description of `attr1`

        **If True**::

            :ivar attr1: Description of `attr1`
            :vartype attr1: int

        **If False**::

            .. attribute:: attr1

               Description of `attr1`

               :type: int

    napoleon_use_param : :obj:`bool` (Defaults to True)
        True to use a ``:param:`` role for each function parameter. False to
        use a single ``:parameters:`` role for all the parameters.

        This `NumPy style`_ snippet will be converted as follows::

            Parameters
            ----------
            arg1 : str
                Description of `arg1`
            arg2 : int, optional
                Description of `arg2`, defaults to 0

        **If True**::

            :param arg1: Description of `arg1`
            :type arg1: str
            :param arg2: Description of `arg2`, defaults to 0
            :type arg2: int, optional

        **If False**::

            :parameters: * **arg1** (*str*) --
                           Description of `arg1`
                         * **arg2** (*int, optional*) --
                           Description of `arg2`, defaults to 0

    napoleon_use_keyword : :obj:`bool` (Defaults to True)
        True to use a ``:keyword:`` role for each function keyword argument.
        False to use a single ``:keyword arguments:`` role for all the
        keywords.

        This behaves similarly to  :attr:`napoleon_use_param`. Note unlike
        docutils, ``:keyword:`` and ``:param:`` will not be treated the same
        way - there will be a separate "Keyword Arguments" section, rendered
        in the same fashion as "Parameters" section (type links created if
        possible)

        See Also
        --------
        :attr:`napoleon_use_param`

    napoleon_use_rtype : :obj:`bool` (Defaults to True)
        True to use the ``:rtype:`` role for the return type. False to output
        the return type inline with the description.

        This `NumPy style`_ snippet will be converted as follows::

            Returns
            -------
            bool
                True if successful, False otherwise

        **If True**::

            :returns: True if successful, False otherwise
            :rtype: bool

        **If False**::

            :returns: *bool* -- True if successful, False otherwise

    napoleon_custom_sections : :obj:`list` (Defaults to None)
        Add a list of custom sections to include, expanding the list of parsed sections.

        The entries can either be strings or tuples, depending on the intention:
          * To create a custom "generic" section, just pass a string.
          * To create an alias for an existing section, pass a tuple containing the
            alias name and the original, in that order.

        If an entry is just a string, it is interpreted as a header for a generic
        section. If the entry is a tuple/list/indexed container, the first entry
        is the name of the section, the second is the section key to emulate.


    """
```
### 11 - sphinx/ext/napoleon/docstring.py:

Start line: 476, End line: 491

```python
class GoogleDocstring:

    def _is_list(self, lines: List[str]) -> bool:
        if not lines:
            return False
        if _bullet_list_regex.match(lines[0]):
            return True
        if _enumerated_list_regex.match(lines[0]):
            return True
        if len(lines) < 2 or lines[0].endswith('::'):
            return False
        indent = self._get_indent(lines[0])
        next_indent = indent
        for line in lines[1:]:
            if line:
                next_indent = self._get_indent(line)
                break
        return next_indent > indent
```
### 12 - sphinx/ext/napoleon/docstring.py:

Start line: 742, End line: 770

```python
class GoogleDocstring:

    def _qualify_name(self, attr_name: str, klass: "Type") -> str:
        if klass and '.' not in attr_name:
            if attr_name.startswith('~'):
                attr_name = attr_name[1:]
            try:
                q = klass.__qualname__
            except AttributeError:
                q = klass.__name__
            return '~%s.%s' % (q, attr_name)
        return attr_name

    def _strip_empty(self, lines: List[str]) -> List[str]:
        if lines:
            start = -1
            for i, line in enumerate(lines):
                if line:
                    start = i
                    break
            if start == -1:
                lines = []
            end = -1
            for i in reversed(range(len(lines))):
                line = lines[i]
                if line:
                    end = i
                    break
            if start > 0 or end + 1 < len(lines):
                lines = lines[start:end + 1]
        return lines
```
### 13 - sphinx/ext/napoleon/docstring.py:

Start line: 433, End line: 474

```python
class GoogleDocstring:

    def _get_current_indent(self, peek_ahead: int = 0) -> int:
        line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        while line != self._line_iter.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        return 0

    def _get_indent(self, line: str) -> int:
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return len(line)

    def _get_initial_indent(self, lines: List[str]) -> int:
        for line in lines:
            if line:
                return self._get_indent(line)
        return 0

    def _get_min_indent(self, lines: List[str]) -> int:
        min_indent = None
        for line in lines:
            if line:
                indent = self._get_indent(line)
                if min_indent is None:
                    min_indent = indent
                elif indent < min_indent:
                    min_indent = indent
        return min_indent or 0

    def _indent(self, lines: List[str], n: int = 4) -> List[str]:
        return [(' ' * n) + line for line in lines]

    def _is_indented(self, line: str, indent: int = 1) -> bool:
        for i, s in enumerate(line):
            if i >= indent:
                return True
            elif not s.isspace():
                return False
        return False
```
### 15 - sphinx/ext/napoleon/docstring.py:

Start line: 617, End line: 628

```python
class GoogleDocstring:

    def _parse_generic_section(self, section: str, use_admonition: bool) -> List[str]:
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        if use_admonition:
            header = '.. admonition:: %s' % section
            lines = self._indent(lines, 3)
        else:
            header = '.. rubric:: %s' % section
        if lines:
            return [header, ''] + lines + ['']
        else:
            return [header, '']
```
### 17 - sphinx/ext/napoleon/docstring.py:

Start line: 564, End line: 574

```python
class GoogleDocstring:

    def _parse_admonition(self, admonition: str, section: str) -> List[str]:
        # type (str, str) -> List[str]
        lines = self._consume_to_next_section()
        return self._format_admonition(admonition, lines)

    def _parse_attribute_docstring(self) -> List[str]:
        _type, _desc = self._consume_inline_attribute()
        lines = self._format_field('', '', _desc)
        if _type:
            lines.extend(['', ':type: %s' % _type])
        return lines
```
### 18 - sphinx/ext/napoleon/docstring.py:

Start line: 530, End line: 562

```python
class GoogleDocstring:

    def _parse(self) -> None:
        self._parsed_lines = self._consume_empty()

        if self._name and self._what in ('attribute', 'data', 'property'):
            # Implicit stop using StopIteration no longer allowed in
            # Python 3.7; see PEP 479
            res = []  # type: List[str]
            try:
                res = self._parse_attribute_docstring()
            except StopIteration:
                pass
            self._parsed_lines.extend(res)
            return

        while self._line_iter.has_next():
            if self._is_section_header():
                try:
                    section = self._consume_section_header()
                    self._is_in_section = True
                    self._section_indent = self._get_current_indent()
                    if _directive_regex.match(section):
                        lines = [section] + self._consume_to_next_section()
                    else:
                        lines = self._sections[section.lower()](section)
                finally:
                    self._is_in_section = False
                    self._section_indent = 0
            else:
                if not self._parsed_lines:
                    lines = self._consume_contiguous() + self._consume_empty()
                else:
                    lines = self._consume_to_next_section()
            self._parsed_lines.extend(lines)
```
### 19 - sphinx/ext/napoleon/docstring.py:

Start line: 710, End line: 740

```python
class GoogleDocstring:

    def _parse_see_also_section(self, section: str) -> List[str]:
        return self._parse_admonition('seealso', section)

    def _parse_warns_section(self, section: str) -> List[str]:
        return self._format_fields(_('Warns'), self._consume_fields())

    def _parse_yields_section(self, section: str) -> List[str]:
        fields = self._consume_returns_section()
        return self._format_fields(_('Yields'), fields)

    def _partition_field_on_colon(self, line: str) -> Tuple[str, str, str]:
        before_colon = []
        after_colon = []
        colon = ''
        found_colon = False
        for i, source in enumerate(_xref_regex.split(line)):
            if found_colon:
                after_colon.append(source)
            else:
                m = _single_colon_regex.search(source)
                if (i % 2) == 0 and m:
                    found_colon = True
                    colon = source[m.start(): m.end()]
                    before_colon.append(source[:m.start()])
                    after_colon.append(source[m.end():])
                else:
                    before_colon.append(source)

        return ("".join(before_colon).strip(),
                colon,
                "".join(after_colon).strip())
```
### 20 - sphinx/ext/napoleon/docstring.py:

Start line: 248, End line: 266

```python
class GoogleDocstring:

    def _consume_fields(self, parse_type: bool = True, prefer_type: bool = False
                        ) -> List[Tuple[str, str, List[str]]]:
        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = self._consume_field(parse_type, prefer_type)
            if _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields

    def _consume_inline_attribute(self) -> Tuple[str, List[str]]:
        line = next(self._line_iter)
        _type, colon, _desc = self._partition_field_on_colon(line)
        if not colon or not _desc:
            _type, _desc = _desc, _type
            _desc += colon
        _descs = [_desc] + self._dedent(self._consume_to_end())
        _descs = self.__class__(_descs, self._config).lines()
        return _type, _descs
```
### 21 - sphinx/ext/napoleon/docstring.py:

Start line: 663, End line: 680

```python
class GoogleDocstring:

    def _parse_raises_section(self, section: str) -> List[str]:
        fields = self._consume_fields(parse_type=False, prefer_type=True)
        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            m = self._name_rgx.match(_type)
            if m and m.group('name'):
                _type = m.group('name')
            _type = ' ' + _type if _type else ''
            _desc = self._strip_empty(_desc)
            _descs = ' ' + '\n    '.join(_desc) if any(_desc) else ''
            lines.append(':raises%s:%s' % (_type, _descs))
        if lines:
            lines.append('')
        return lines

    def _parse_references_section(self, section: str) -> List[str]:
        use_admonition = self._config.napoleon_use_admonition_for_references
        return self._parse_generic_section(_('References'), use_admonition)
```
### 22 - sphinx/ext/napoleon/docstring.py:

Start line: 889, End line: 925

```python
class NumpyDocstring(GoogleDocstring):

    def _consume_returns_section(self) -> List[Tuple[str, str, List[str]]]:
        return self._consume_fields(prefer_type=True)

    def _consume_section_header(self) -> str:
        section = next(self._line_iter)
        if not _directive_regex.match(section):
            # Consume the header underline
            next(self._line_iter)
        return section

    def _is_section_break(self) -> bool:
        line1, line2 = self._line_iter.peek(2)
        return (not self._line_iter.has_next() or
                self._is_section_header() or
                ['', ''] == [line1, line2] or
                (self._is_in_section and
                    line1 and
                    not self._is_indented(line1, self._section_indent)))

    def _is_section_header(self) -> bool:
        section, underline = self._line_iter.peek(2)
        section = section.lower()
        if section in self._sections and isinstance(underline, str):
            return bool(_numpy_section_regex.match(underline))
        elif self._directive_sections:
            if _directive_regex.match(section):
                for directive_section in self._directive_sections:
                    if section.startswith(directive_section):
                        return True
        return False

    def _parse_see_also_section(self, section: str) -> List[str]:
        lines = self._consume_to_next_section()
        try:
            return self._parse_numpydoc_see_also_section(lines)
        except ValueError:
            return self._format_admonition('seealso', lines)
```
### 24 - sphinx/ext/napoleon/docstring.py:

Start line: 226, End line: 246

```python
class GoogleDocstring:

    def _consume_field(self, parse_type: bool = True, prefer_type: bool = False
                       ) -> Tuple[str, str, List[str]]:
        line = next(self._line_iter)

        before, colon, after = self._partition_field_on_colon(line)
        _name, _type, _desc = before, '', after

        if parse_type:
            match = _google_typed_arg_regex.match(before)
            if match:
                _name = match.group(1)
                _type = match.group(2)

        _name = self._escape_args_and_kwargs(_name)

        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = self._get_indent(line) + 1
        _descs = [_desc] + self._dedent(self._consume_indented_block(indent))
        _descs = self.__class__(_descs, self._config).lines()
        return _name, _type, _descs
```
### 25 - sphinx/ext/napoleon/docstring.py:

Start line: 329, End line: 340

```python
class GoogleDocstring:

    def _fix_field_desc(self, desc: List[str]) -> List[str]:
        if self._is_list(desc):
            desc = [''] + desc
        elif desc[0].endswith('::'):
            desc_block = desc[1:]
            indent = self._get_indent(desc[0])
            block_indent = self._get_initial_indent(desc_block)
            if block_indent > indent:
                desc = [''] + desc
            else:
                desc = ['', desc[0]] + self._indent(desc_block, 4)
        return desc
```
### 26 - sphinx/ext/napoleon/docstring.py:

Start line: 493, End line: 513

```python
class GoogleDocstring:

    def _is_section_header(self) -> bool:
        section = self._line_iter.peek().lower()
        match = _google_section_regex.match(section)
        if match and section.strip(':') in self._sections:
            header_indent = self._get_indent(section)
            section_indent = self._get_current_indent(peek_ahead=1)
            return section_indent > header_indent
        elif self._directive_sections:
            if _directive_regex.match(section):
                for directive_section in self._directive_sections:
                    if section.startswith(directive_section):
                        return True
        return False

    def _is_section_break(self) -> bool:
        line = self._line_iter.peek()
        return (not self._line_iter.has_next() or
                self._is_section_header() or
                (self._is_in_section and
                    line and
                    not self._is_indented(line, self._section_indent)))
```
### 29 - sphinx/ext/napoleon/docstring.py:

Start line: 342, End line: 366

```python
class GoogleDocstring:

    def _format_admonition(self, admonition: str, lines: List[str]) -> List[str]:
        lines = self._strip_empty(lines)
        if len(lines) == 1:
            return ['.. %s:: %s' % (admonition, lines[0].strip()), '']
        elif lines:
            lines = self._indent(self._dedent(lines), 3)
            return ['.. %s::' % admonition, ''] + lines + ['']
        else:
            return ['.. %s::' % admonition, '']

    def _format_block(self, prefix: str, lines: List[str], padding: str = None) -> List[str]:
        if lines:
            if padding is None:
                padding = ' ' * len(prefix)
            result_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    result_lines.append((prefix + line).rstrip())
                elif line:
                    result_lines.append(padding + line)
                else:
                    result_lines.append('')
            return result_lines
        else:
            return [prefix]
```
### 30 - sphinx/ext/napoleon/docstring.py:

Start line: 368, End line: 383

```python
class GoogleDocstring:

    def _format_docutils_params(self, fields: List[Tuple[str, str, List[str]]],
                                field_role: str = 'param', type_role: str = 'type'
                                ) -> List[str]:
        lines = []
        for _name, _type, _desc in fields:
            _desc = self._strip_empty(_desc)
            if any(_desc):
                _desc = self._fix_field_desc(_desc)
                field = ':%s %s: ' % (field_role, _name)
                lines.extend(self._format_block(field, _desc))
            else:
                lines.append(':%s %s:' % (field_role, _name))

            if _type:
                lines.append(':%s %s: %s' % (type_role, _name, _type))
        return lines + ['']
```
### 31 - sphinx/ext/napoleon/docstring.py:

Start line: 201, End line: 224

```python
class GoogleDocstring:

    def _consume_indented_block(self, indent: int = 1) -> List[str]:
        lines = []
        line = self._line_iter.peek()
        while(not self._is_section_break() and
              (not line or self._is_indented(line, indent))):
            lines.append(next(self._line_iter))
            line = self._line_iter.peek()
        return lines

    def _consume_contiguous(self) -> List[str]:
        lines = []
        while (self._line_iter.has_next() and
               self._line_iter.peek() and
               not self._is_section_header()):
            lines.append(next(self._line_iter))
        return lines

    def _consume_empty(self) -> List[str]:
        lines = []
        line = self._line_iter.peek()
        while self._line_iter.has_next() and not line:
            lines.append(next(self._line_iter))
            line = self._line_iter.peek()
        return lines
```
### 32 - sphinx/ext/napoleon/docstring.py:

Start line: 515, End line: 528

```python
class GoogleDocstring:

    def _load_custom_sections(self) -> None:
        if self._config.napoleon_custom_sections is not None:
            for entry in self._config.napoleon_custom_sections:
                if isinstance(entry, str):
                    # if entry is just a label, add to sections list,
                    # using generic section logic.
                    self._sections[entry.lower()] = self._parse_custom_generic_section
                else:
                    # otherwise, assume entry is container;
                    # [0] is new section, [1] is the section to alias.
                    # in the case of key mismatch, just handle as generic section.
                    self._sections[entry[0].lower()] = \
                        self._sections.get(entry[1].lower(),
                                           self._parse_custom_generic_section)
```
### 33 - sphinx/ext/napoleon/docstring.py:

Start line: 872, End line: 887

```python
class NumpyDocstring(GoogleDocstring):

    def _consume_field(self, parse_type: bool = True, prefer_type: bool = False
                       ) -> Tuple[str, str, List[str]]:
        line = next(self._line_iter)
        if parse_type:
            _name, _, _type = self._partition_field_on_colon(line)
        else:
            _name, _type = line, ''
        _name, _type = _name.strip(), _type.strip()
        _name = self._escape_args_and_kwargs(_name)

        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = self._get_indent(line) + 1
        _desc = self._dedent(self._consume_indented_block(indent))
        _desc = self.__class__(_desc, self._config).lines()
        return _name, _type, _desc
```
### 34 - sphinx/ext/napoleon/docstring.py:

Start line: 682, End line: 708

```python
class GoogleDocstring:

    def _parse_returns_section(self, section: str) -> List[str]:
        fields = self._consume_returns_section()
        multi = len(fields) > 1
        if multi:
            use_rtype = False
        else:
            use_rtype = self._config.napoleon_use_rtype

        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            if use_rtype:
                field = self._format_field(_name, '', _desc)
            else:
                field = self._format_field(_name, _type, _desc)

            if multi:
                if lines:
                    lines.extend(self._format_block('          * ', field))
                else:
                    lines.extend(self._format_block(':returns: * ', field))
            else:
                lines.extend(self._format_block(':returns: ', field))
                if _type and use_rtype:
                    lines.extend([':rtype: %s' % _type, ''])
        if lines and lines[-1]:
            lines.append('')
        return lines
```
### 35 - sphinx/ext/napoleon/docstring.py:

Start line: 385, End line: 412

```python
class GoogleDocstring:

    def _format_field(self, _name: str, _type: str, _desc: List[str]) -> List[str]:
        _desc = self._strip_empty(_desc)
        has_desc = any(_desc)
        separator = ' -- ' if has_desc else ''
        if _name:
            if _type:
                if '`' in _type:
                    field = '**%s** (%s)%s' % (_name, _type, separator)
                else:
                    field = '**%s** (*%s*)%s' % (_name, _type, separator)
            else:
                field = '**%s**%s' % (_name, separator)
        elif _type:
            if '`' in _type:
                field = '%s%s' % (_type, separator)
            else:
                field = '*%s*%s' % (_type, separator)
        else:
            field = ''

        if has_desc:
            _desc = self._fix_field_desc(_desc)
            if _desc[0]:
                return [field + _desc[0]] + _desc[1:]
            else:
                return [field] + _desc
        else:
            return [field]
```
### 37 - sphinx/ext/napoleon/docstring.py:

Start line: 107, End line: 199

```python
class GoogleDocstring:

    def __init__(self, docstring: Union[str, List[str]], config: SphinxConfig = None,
                 app: Sphinx = None, what: str = '', name: str = '',
                 obj: Any = None, options: Any = None) -> None:
        self._config = config
        self._app = app

        if not self._config:
            from sphinx.ext.napoleon import Config
            self._config = self._app.config if self._app else Config()  # type: ignore

        if not what:
            if inspect.isclass(obj):
                what = 'class'
            elif inspect.ismodule(obj):
                what = 'module'
            elif callable(obj):
                what = 'function'
            else:
                what = 'object'

        self._what = what
        self._name = name
        self._obj = obj
        self._opt = options
        if isinstance(docstring, str):
            lines = docstring.splitlines()
        else:
            lines = docstring
        self._line_iter = modify_iter(lines, modifier=lambda s: s.rstrip())
        self._parsed_lines = []  # type: List[str]
        self._is_in_section = False
        self._section_indent = 0
        if not hasattr(self, '_directive_sections'):
            self._directive_sections = []  # type: List[str]
        if not hasattr(self, '_sections'):
            self._sections = {
                'args': self._parse_parameters_section,
                'arguments': self._parse_parameters_section,
                'attention': partial(self._parse_admonition, 'attention'),
                'attributes': self._parse_attributes_section,
                'caution': partial(self._parse_admonition, 'caution'),
                'danger': partial(self._parse_admonition, 'danger'),
                'error': partial(self._parse_admonition, 'error'),
                'example': self._parse_examples_section,
                'examples': self._parse_examples_section,
                'hint': partial(self._parse_admonition, 'hint'),
                'important': partial(self._parse_admonition, 'important'),
                'keyword args': self._parse_keyword_arguments_section,
                'keyword arguments': self._parse_keyword_arguments_section,
                'methods': self._parse_methods_section,
                'note': partial(self._parse_admonition, 'note'),
                'notes': self._parse_notes_section,
                'other parameters': self._parse_other_parameters_section,
                'parameters': self._parse_parameters_section,
                'return': self._parse_returns_section,
                'returns': self._parse_returns_section,
                'raises': self._parse_raises_section,
                'references': self._parse_references_section,
                'see also': self._parse_see_also_section,
                'tip': partial(self._parse_admonition, 'tip'),
                'todo': partial(self._parse_admonition, 'todo'),
                'warning': partial(self._parse_admonition, 'warning'),
                'warnings': partial(self._parse_admonition, 'warning'),
                'warns': self._parse_warns_section,
                'yield': self._parse_yields_section,
                'yields': self._parse_yields_section,
            }  # type: Dict[str, Callable]

        self._load_custom_sections()

        self._parse()

    def __str__(self) -> str:
        """Return the parsed docstring in reStructuredText format.

        Returns
        -------
        unicode
            Unicode version of the docstring.

        """
        return '\n'.join(self.lines())

    def lines(self) -> List[str]:
        """Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

        """
        return self._parsed_lines
```
### 38 - sphinx/ext/napoleon/docstring.py:

Start line: 41, End line: 105

```python
class GoogleDocstring:
    """Convert Google style docstrings to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    config: :obj:`sphinx.ext.napoleon.Config` or :obj:`sphinx.config.Config`
        The configuration settings to use. If not given, defaults to the
        config object on `app`; or if `app` is not given defaults to the
        a new :class:`sphinx.ext.napoleon.Config` object.


    Other Parameters
    ----------------
    app : :class:`sphinx.application.Sphinx`, optional
        Application object representing the Sphinx process.
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.


    Example
    -------
    >>> from sphinx.ext.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Args:
    ...   arg1(int): Description of `arg1`
    ...   arg2(str): Description of `arg2`
    ... Returns:
    ...   str: Description of return value.
    ... '''
    >>> print(GoogleDocstring(docstring, config))
    One line summary.
    <BLANKLINE>
    Extended description.
    <BLANKLINE>
    :param arg1: Description of `arg1`
    :type arg1: int
    :param arg2: Description of `arg2`
    :type arg2: str
    <BLANKLINE>
    :returns: Description of return value.
    :rtype: str
    <BLANKLINE>

    """

    _name_rgx = re.compile(r"^\s*((?::(?P<role>\S+):)?`(?P<name>~?[a-zA-Z0-9_.-]+)`|"
                           r" (?P<name2>~?[a-zA-Z0-9_.-]+))\s*", re.X)
```
### 45 - sphinx/ext/napoleon/docstring.py:

Start line: 773, End line: 870

```python
class NumpyDocstring(GoogleDocstring):
    """Convert NumPy style docstrings to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    config: :obj:`sphinx.ext.napoleon.Config` or :obj:`sphinx.config.Config`
        The configuration settings to use. If not given, defaults to the
        config object on `app`; or if `app` is not given defaults to the
        a new :class:`sphinx.ext.napoleon.Config` object.


    Other Parameters
    ----------------
    app : :class:`sphinx.application.Sphinx`, optional
        Application object representing the Sphinx process.
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.


    Example
    -------
    >>> from sphinx.ext.napoleon import Config
    >>> config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    >>> docstring = '''One line summary.
    ...
    ... Extended description.
    ...
    ... Parameters
    ... ----------
    ... arg1 : int
    ...     Description of `arg1`
    ... arg2 : str
    ...     Description of `arg2`
    ... Returns
    ... -------
    ... str
    ...     Description of return value.
    ... '''
    >>> print(NumpyDocstring(docstring, config))
    One line summary.
    <BLANKLINE>
    Extended description.
    <BLANKLINE>
    :param arg1: Description of `arg1`
    :type arg1: int
    :param arg2: Description of `arg2`
    :type arg2: str
    <BLANKLINE>
    :returns: Description of return value.
    :rtype: str
    <BLANKLINE>

    Methods
    -------
    __str__()
        Return the parsed docstring in reStructuredText format.

        Returns
        -------
        str
            UTF-8 encoded version of the docstring.

    __unicode__()
        Return the parsed docstring in reStructuredText format.

        Returns
        -------
        unicode
            Unicode version of the docstring.

    lines()
        Return the parsed lines of the docstring in reStructuredText format.

        Returns
        -------
        list(str)
            The lines of the docstring in a list.

    """
    def __init__(self, docstring: Union[str, List[str]], config: SphinxConfig = None,
                 app: Sphinx = None, what: str = '', name: str = '',
                 obj: Any = None, options: Any = None) -> None:
        self._directive_sections = ['.. index::']
        super().__init__(docstring, config, app, what, name, obj, options)
```
### 46 - sphinx/ext/napoleon/docstring.py:

Start line: 414, End line: 431

```python
class GoogleDocstring:

    def _format_fields(self, field_type: str, fields: List[Tuple[str, str, List[str]]]
                       ) -> List[str]:
        field_type = ':%s:' % field_type.strip()
        padding = ' ' * len(field_type)
        multi = len(fields) > 1
        lines = []  # type: List[str]
        for _name, _type, _desc in fields:
            field = self._format_field(_name, _type, _desc)
            if multi:
                if lines:
                    lines.extend(self._format_block(padding + ' * ', field))
                else:
                    lines.extend(self._format_block(field_type + ' * ', field))
            else:
                lines.extend(self._format_block(field_type + ' ', field))
        if lines and lines[-1]:
            lines.append('')
        return lines
```
### 52 - sphinx/ext/napoleon/docstring.py:

Start line: 927, End line: 1033

```python
class NumpyDocstring(GoogleDocstring):

    def _parse_numpydoc_see_also_section(self, content: List[str]) -> List[str]:
        """
        Derived from the NumpyDoc implementation of _parse_see_also.

        See Also
        --------
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
        items = []

        def parse_item_name(text: str) -> Tuple[str, str]:
            """Match ':role:`name`' or 'name'"""
            m = self._name_rgx.match(text)
            if m:
                g = m.groups()
                if g[1] is None:
                    return g[3], None
                else:
                    return g[2], g[1]
            raise ValueError("%s is not a item name" % text)

        def push_item(name: str, rest: List[str]) -> None:
            if not name:
                return
            name, role = parse_item_name(name)
            items.append((name, list(rest), role))
            del rest[:]

        current_func = None
        rest = []  # type: List[str]

        for line in content:
            if not line.strip():
                continue

            m = self._name_rgx.match(line)
            if m and line[m.end():].strip().startswith(':'):
                push_item(current_func, rest)
                current_func, line = line[:m.end()], line[m.end():]
                rest = [line.split(':', 1)[1].strip()]
                if not rest[0]:
                    rest = []
            elif not line.startswith(' '):
                push_item(current_func, rest)
                current_func = None
                if ',' in line:
                    for func in line.split(','):
                        if func.strip():
                            push_item(func, [])
                elif line.strip():
                    current_func = line
            elif current_func is not None:
                rest.append(line.strip())
        push_item(current_func, rest)

        if not items:
            return []

        roles = {
            'method': 'meth',
            'meth': 'meth',
            'function': 'func',
            'func': 'func',
            'class': 'class',
            'exception': 'exc',
            'exc': 'exc',
            'object': 'obj',
            'obj': 'obj',
            'module': 'mod',
            'mod': 'mod',
            'data': 'data',
            'constant': 'const',
            'const': 'const',
            'attribute': 'attr',
            'attr': 'attr'
        }
        if self._what is None:
            func_role = 'obj'
        else:
            func_role = roles.get(self._what, '')
        lines = []  # type: List[str]
        last_had_desc = True
        for func, desc, role in items:
            if role:
                link = ':%s:`%s`' % (role, func)
            elif func_role:
                link = ':%s:`%s`' % (func_role, func)
            else:
                link = "`%s`_" % func
            if desc or last_had_desc:
                lines += ['']
                lines += [link]
            else:
                lines[-1] += ", %s" % link
            if desc:
                lines += self._indent([' '.join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
        lines += ['']

        return self._format_admonition('seealso', lines)
```
