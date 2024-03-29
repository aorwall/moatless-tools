# sphinx-doc__sphinx-7268

| **sphinx-doc/sphinx** | `a73617c51b9e29d7f059a2794f4574bb80cfcf57` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 28970 |
| **Any found context length** | 119 |
| **Avg pos** | 48.5 |
| **Min pos** | 1 |
| **Max pos** | 85 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1004,7 +1004,7 @@ def can_document_member(cls, member: Any, membername: str, isattr: bool, parent:
                 (inspect.isroutine(member) and isinstance(parent, ModuleDocumenter)))
 
     def format_args(self, **kwargs: Any) -> str:
-        if self.env.config.autodoc_typehints == 'none':
+        if self.env.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
 
         if inspect.isbuiltin(self.object) or inspect.ismethoddescriptor(self.object):
@@ -1744,7 +1744,8 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('autodoc_default_options', {}, True)
     app.add_config_value('autodoc_docstring_signature', True, True)
     app.add_config_value('autodoc_mock_imports', [], True)
-    app.add_config_value('autodoc_typehints', "signature", True, ENUM("signature", "none"))
+    app.add_config_value('autodoc_typehints', "signature", True,
+                         ENUM("signature", "description", "none"))
     app.add_config_value('autodoc_warningiserror', True, True)
     app.add_config_value('autodoc_inherit_docstrings', True, True)
     app.add_event('autodoc-before-process-signature')
@@ -1753,5 +1754,6 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_event('autodoc-skip-member')
 
     app.setup_extension('sphinx.ext.autodoc.type_comment')
+    app.setup_extension('sphinx.ext.autodoc.typehints')
 
     return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
diff --git a/sphinx/ext/autodoc/typehints.py b/sphinx/ext/autodoc/typehints.py
--- a/sphinx/ext/autodoc/typehints.py
+++ b/sphinx/ext/autodoc/typehints.py
@@ -18,21 +18,9 @@
 
 from sphinx import addnodes
 from sphinx.application import Sphinx
-from sphinx.config import Config, ENUM
 from sphinx.util import inspect, typing
 
 
-def config_inited(app: Sphinx, config: Config) -> None:
-    if config.autodoc_typehints == 'description':
-        # HACK: override this to make autodoc suppressing typehints in signatures
-        config.autodoc_typehints = 'none'  # type: ignore
-
-        # preserve user settings
-        app._autodoc_typehints_description = True  # type: ignore
-    else:
-        app._autodoc_typehints_description = False  # type: ignore
-
-
 def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
                      options: Dict, args: str, retann: str) -> None:
     """Record type hints to env object."""
@@ -53,7 +41,7 @@ def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
 def merge_typehints(app: Sphinx, domain: str, objtype: str, contentnode: Element) -> None:
     if domain != 'py':
         return
-    if app._autodoc_typehints_description is False:  # type: ignore
+    if app.config.autodoc_typehints != 'description':
         return
 
     signature = cast(addnodes.desc_signature, contentnode.parent[0])
@@ -141,10 +129,6 @@ def modify_field_list(node: nodes.field_list, annotations: Dict[str, str]) -> No
 
 
 def setup(app: Sphinx) -> Dict[str, Any]:
-    app.setup_extension('sphinx.ext.autodoc')
-    app.config.values['autodoc_typehints'] = ('signature', True,
-                                              ENUM("signature", "description", "none"))
-    app.connect('config-inited', config_inited)
     app.connect('autodoc-process-signature', record_typehints)
     app.connect('object-description-transform', merge_typehints)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/__init__.py | 1007 | 1007 | 85 | 2 | 28970
| sphinx/ext/autodoc/__init__.py | 1747 | 1747 | 4 | 2 | 987
| sphinx/ext/autodoc/__init__.py | 1756 | 1756 | 4 | 2 | 987
| sphinx/ext/autodoc/typehints.py | 21 | 35 | - | 1 | -
| sphinx/ext/autodoc/typehints.py | 56 | 56 | 3 | 1 | 568
| sphinx/ext/autodoc/typehints.py | 144 | 147 | 1 | 1 | 119


## Problem Statement

```
autodoc: Load sphinx.ext.autodoc.typehints automatically
After typehints enough matured, it should be loaded automatically from autodoc extension.
refs: #6418 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/ext/autodoc/typehints.py** | 143 | 156| 119 | 119 | 1241 | 
| 2 | **1 sphinx/ext/autodoc/typehints.py** | 11 | 33| 170 | 289 | 1241 | 
| **-> 3 <-** | **1 sphinx/ext/autodoc/typehints.py** | 53 | 85| 279 | 568 | 1241 | 
| **-> 4 <-** | **2 sphinx/ext/autodoc/__init__.py** | 1725 | 1758| 419 | 987 | 16558 | 
| 5 | **2 sphinx/ext/autodoc/__init__.py** | 1480 | 1517| 336 | 1323 | 16558 | 
| 6 | **2 sphinx/ext/autodoc/typehints.py** | 36 | 50| 142 | 1465 | 16558 | 
| 7 | 3 sphinx/ext/autodoc/importer.py | 39 | 101| 582 | 2047 | 18082 | 
| 8 | **3 sphinx/ext/autodoc/__init__.py** | 824 | 857| 313 | 2360 | 18082 | 
| 9 | 4 sphinx/ext/autosummary/__init__.py | 752 | 778| 362 | 2722 | 24459 | 
| 10 | **4 sphinx/ext/autodoc/__init__.py** | 811 | 822| 126 | 2848 | 24459 | 
| 11 | 5 sphinx/ext/autodoc/mock.py | 70 | 91| 191 | 3039 | 25547 | 
| 12 | **5 sphinx/ext/autodoc/__init__.py** | 1077 | 1286| 421 | 3460 | 25547 | 
| 13 | **5 sphinx/ext/autodoc/__init__.py** | 548 | 625| 676 | 4136 | 25547 | 
| 14 | **5 sphinx/ext/autodoc/typehints.py** | 88 | 140| 468 | 4604 | 25547 | 
| 15 | 5 sphinx/ext/autodoc/mock.py | 94 | 106| 115 | 4719 | 25547 | 
| 16 | 5 sphinx/ext/autosummary/__init__.py | 55 | 106| 366 | 5085 | 25547 | 
| 17 | **5 sphinx/ext/autodoc/__init__.py** | 202 | 272| 750 | 5835 | 25547 | 
| 18 | **5 sphinx/ext/autodoc/__init__.py** | 1133 | 1286| 1545 | 7380 | 25547 | 
| 19 | 6 doc/conf.py | 81 | 139| 505 | 7885 | 27074 | 
| 20 | 7 sphinx/application.py | 992 | 1010| 233 | 8118 | 37647 | 
| 21 | 7 sphinx/application.py | 1012 | 1025| 174 | 8292 | 37647 | 
| 22 | 7 sphinx/ext/autodoc/mock.py | 11 | 67| 441 | 8733 | 37647 | 
| 23 | **7 sphinx/ext/autodoc/__init__.py** | 1711 | 1722| 116 | 8849 | 37647 | 
| 24 | **7 sphinx/ext/autodoc/__init__.py** | 455 | 483| 298 | 9147 | 37647 | 
| 25 | 7 sphinx/ext/autosummary/__init__.py | 402 | 428| 217 | 9364 | 37647 | 
| 26 | 8 sphinx/ext/autosummary/generate.py | 69 | 99| 279 | 9643 | 41595 | 
| 27 | 9 sphinx/ext/autodoc/directive.py | 9 | 49| 302 | 9945 | 42856 | 
| 28 | 9 sphinx/ext/autosummary/generate.py | 20 | 66| 317 | 10262 | 42856 | 
| 29 | **9 sphinx/ext/autodoc/__init__.py** | 285 | 319| 303 | 10565 | 42856 | 
| 30 | 10 sphinx/registry.py | 11 | 50| 307 | 10872 | 47346 | 
| 31 | 10 sphinx/ext/autosummary/__init__.py | 180 | 212| 282 | 11154 | 47346 | 
| 32 | **10 sphinx/ext/autodoc/__init__.py** | 440 | 453| 151 | 11305 | 47346 | 
| 33 | 10 sphinx/application.py | 63 | 125| 494 | 11799 | 47346 | 
| 34 | **10 sphinx/ext/autodoc/__init__.py** | 1354 | 1388| 263 | 12062 | 47346 | 
| 35 | 11 sphinx/ext/autodoc/type_comment.py | 117 | 139| 226 | 12288 | 48547 | 
| 36 | 11 sphinx/registry.py | 388 | 438| 460 | 12748 | 48547 | 
| 37 | 11 doc/conf.py | 142 | 163| 255 | 13003 | 48547 | 
| 38 | 11 sphinx/ext/autosummary/generate.py | 224 | 306| 654 | 13657 | 48547 | 
| 39 | **11 sphinx/ext/autodoc/__init__.py** | 412 | 425| 184 | 13841 | 48547 | 
| 40 | **11 sphinx/ext/autodoc/__init__.py** | 321 | 337| 166 | 14007 | 48547 | 
| 41 | 11 sphinx/ext/autodoc/directive.py | 109 | 159| 454 | 14461 | 48547 | 
| 42 | 11 doc/conf.py | 1 | 80| 766 | 15227 | 48547 | 
| 43 | 12 sphinx/transforms/__init__.py | 215 | 229| 157 | 15384 | 51694 | 
| 44 | 13 sphinx/directives/__init__.py | 76 | 88| 128 | 15512 | 54089 | 
| 45 | 14 sphinx/directives/code.py | 9 | 31| 146 | 15658 | 57986 | 
| 46 | **14 sphinx/ext/autodoc/__init__.py** | 1682 | 1708| 261 | 15919 | 57986 | 
| 47 | **14 sphinx/ext/autodoc/__init__.py** | 954 | 971| 213 | 16132 | 57986 | 
| 48 | **14 sphinx/ext/autodoc/__init__.py** | 693 | 765| 697 | 16829 | 57986 | 
| 49 | 14 sphinx/registry.py | 209 | 231| 283 | 17112 | 57986 | 
| 50 | 14 sphinx/ext/autosummary/generate.py | 135 | 156| 211 | 17323 | 57986 | 
| 51 | 15 sphinx/ext/apidoc.py | 412 | 440| 374 | 17697 | 62678 | 
| 52 | 16 sphinx/ext/autosectionlabel.py | 11 | 71| 458 | 18155 | 63196 | 
| 53 | 17 sphinx/directives/patches.py | 9 | 25| 124 | 18279 | 64809 | 
| 54 | 18 sphinx/domains/python.py | 11 | 67| 432 | 18711 | 75430 | 
| 55 | **18 sphinx/ext/autodoc/__init__.py** | 627 | 691| 643 | 19354 | 75430 | 
| 56 | 19 sphinx/extension.py | 11 | 40| 232 | 19586 | 75932 | 
| 57 | **19 sphinx/ext/autodoc/__init__.py** | 1391 | 1457| 587 | 20173 | 75932 | 
| 58 | 20 sphinx/ext/inheritance_diagram.py | 461 | 476| 173 | 20346 | 79785 | 
| 59 | 20 sphinx/ext/autosummary/generate.py | 473 | 490| 138 | 20484 | 79785 | 
| 60 | 20 sphinx/ext/autosummary/__init__.py | 716 | 749| 313 | 20797 | 79785 | 
| 61 | 20 sphinx/registry.py | 233 | 251| 240 | 21037 | 79785 | 
| 62 | 21 sphinx/util/docutils.py | 228 | 238| 163 | 21200 | 83863 | 
| 63 | 22 sphinx/domains/std.py | 11 | 48| 323 | 21523 | 93814 | 
| 64 | 23 sphinx/domains/javascript.py | 11 | 34| 190 | 21713 | 97829 | 
| 65 | 23 sphinx/ext/autodoc/mock.py | 109 | 148| 284 | 21997 | 97829 | 
| 66 | **23 sphinx/ext/autodoc/__init__.py** | 427 | 438| 147 | 22144 | 97829 | 
| 67 | **23 sphinx/ext/autodoc/__init__.py** | 1632 | 1661| 240 | 22384 | 97829 | 
| 68 | 24 sphinx/transforms/post_transforms/__init__.py | 152 | 175| 279 | 22663 | 99559 | 
| 69 | 24 sphinx/transforms/__init__.py | 11 | 44| 226 | 22889 | 99559 | 
| 70 | 25 sphinx/search/__init__.py | 10 | 30| 135 | 23024 | 103462 | 
| 71 | **25 sphinx/ext/autodoc/__init__.py** | 1520 | 1598| 682 | 23706 | 103462 | 
| 72 | 26 sphinx/environment/__init__.py | 11 | 82| 489 | 24195 | 109283 | 
| 73 | **26 sphinx/ext/autodoc/__init__.py** | 13 | 122| 791 | 24986 | 109283 | 
| 74 | **26 sphinx/ext/autodoc/__init__.py** | 860 | 878| 179 | 25165 | 109283 | 
| 75 | 27 sphinx/io.py | 10 | 46| 274 | 25439 | 110974 | 
| 76 | 27 sphinx/directives/code.py | 417 | 481| 658 | 26097 | 110974 | 
| 77 | 27 sphinx/ext/apidoc.py | 346 | 411| 751 | 26848 | 110974 | 
| 78 | 27 sphinx/ext/autosummary/__init__.py | 672 | 696| 230 | 27078 | 110974 | 
| 79 | 27 sphinx/ext/autosummary/generate.py | 431 | 470| 360 | 27438 | 110974 | 
| 80 | 28 sphinx/ext/todo.py | 321 | 344| 214 | 27652 | 113670 | 
| 81 | 28 sphinx/directives/__init__.py | 258 | 296| 273 | 27925 | 113670 | 
| 82 | 29 sphinx/util/compat.py | 11 | 36| 189 | 28114 | 114119 | 
| 83 | 30 doc/development/tutorials/examples/todo.py | 1 | 27| 111 | 28225 | 114937 | 
| 84 | 30 sphinx/ext/autodoc/directive.py | 78 | 89| 125 | 28350 | 114937 | 
| **-> 85 <-** | **30 sphinx/ext/autodoc/__init__.py** | 992 | 1057| 620 | 28970 | 114937 | 
| 86 | 31 sphinx/builders/devhelp.py | 13 | 39| 153 | 29123 | 115164 | 
| 87 | 32 sphinx/builders/texinfo.py | 130 | 170| 455 | 29578 | 117191 | 
| 88 | 33 sphinx/parsers.py | 11 | 28| 134 | 29712 | 118054 | 
| 89 | 34 sphinx/ext/viewcode.py | 86 | 118| 340 | 30052 | 120292 | 
| 90 | 34 sphinx/application.py | 13 | 60| 378 | 30430 | 120292 | 
| 91 | 35 sphinx/domains/citation.py | 11 | 31| 126 | 30556 | 121572 | 
| 92 | **35 sphinx/ext/autodoc/__init__.py** | 1664 | 1680| 126 | 30682 | 121572 | 
| 93 | 36 sphinx/domains/__init__.py | 12 | 30| 140 | 30822 | 125083 | 
| 94 | 36 sphinx/registry.py | 253 | 302| 466 | 31288 | 125083 | 
| 95 | 36 sphinx/ext/autosummary/__init__.py | 391 | 400| 152 | 31440 | 125083 | 
| 96 | 37 sphinx/util/typing.py | 11 | 39| 200 | 31640 | 126898 | 
| 97 | 37 sphinx/ext/autosummary/__init__.py | 215 | 274| 488 | 32128 | 126898 | 
| 98 | **37 sphinx/ext/autodoc/__init__.py** | 1060 | 1075| 139 | 32267 | 126898 | 
| 99 | 38 sphinx/builders/manpage.py | 11 | 31| 156 | 32423 | 127806 | 
| 100 | 39 sphinx/testing/util.py | 141 | 156| 195 | 32618 | 129569 | 
| 101 | 39 sphinx/ext/inheritance_diagram.py | 38 | 66| 212 | 32830 | 129569 | 
| 102 | 39 sphinx/ext/autosummary/__init__.py | 643 | 669| 294 | 33124 | 129569 | 
| 103 | 39 sphinx/ext/autosummary/generate.py | 324 | 341| 182 | 33306 | 129569 | 
| 104 | 39 sphinx/extension.py | 43 | 70| 221 | 33527 | 129569 | 
| 105 | 39 sphinx/ext/viewcode.py | 48 | 84| 313 | 33840 | 129569 | 
| 106 | 40 sphinx/ext/doctest.py | 309 | 324| 130 | 33970 | 134519 | 
| 107 | 41 sphinx/util/docfields.py | 68 | 78| 146 | 34116 | 137873 | 
| 108 | 42 sphinx/directives/other.py | 88 | 150| 602 | 34718 | 141030 | 
| 109 | 42 sphinx/ext/todo.py | 14 | 44| 207 | 34925 | 141030 | 
| 110 | 43 sphinx/builders/latex/__init__.py | 11 | 42| 329 | 35254 | 146657 | 
| 111 | 44 sphinx/domains/index.py | 11 | 30| 127 | 35381 | 147591 | 
| 112 | 44 sphinx/ext/autodoc/type_comment.py | 11 | 37| 247 | 35628 | 147591 | 
| 113 | **44 sphinx/ext/autodoc/__init__.py** | 881 | 911| 287 | 35915 | 147591 | 
| 114 | 45 sphinx/domains/c.py | 310 | 336| 248 | 36163 | 150565 | 
| 115 | 45 sphinx/ext/autosummary/__init__.py | 374 | 389| 182 | 36345 | 150565 | 
| 116 | 45 sphinx/directives/patches.py | 54 | 68| 143 | 36488 | 150565 | 
| 117 | **45 sphinx/ext/autodoc/__init__.py** | 768 | 809| 371 | 36859 | 150565 | 
| 118 | 46 setup.py | 171 | 246| 626 | 37485 | 152266 | 
| 119 | 46 sphinx/directives/other.py | 374 | 398| 228 | 37713 | 152266 | 
| 120 | 47 sphinx/addnodes.py | 321 | 363| 301 | 38014 | 154543 | 
| 121 | 48 sphinx/writers/texinfo.py | 968 | 1085| 846 | 38860 | 166717 | 
| 122 | 49 sphinx/util/__init__.py | 11 | 75| 505 | 39365 | 172494 | 
| 123 | 50 sphinx/setup_command.py | 159 | 208| 415 | 39780 | 174176 | 
| 124 | 50 sphinx/ext/doctest.py | 550 | 567| 215 | 39995 | 174176 | 
| 125 | 50 sphinx/setup_command.py | 93 | 110| 145 | 40140 | 174176 | 
| 126 | 50 sphinx/registry.py | 174 | 184| 134 | 40274 | 174176 | 
| 127 | 51 sphinx/util/smartypants.py | 376 | 388| 137 | 40411 | 178287 | 
| 128 | 51 sphinx/directives/__init__.py | 299 | 313| 106 | 40517 | 178287 | 
| 129 | 52 sphinx/util/logging.py | 11 | 56| 279 | 40796 | 181924 | 
| 130 | 52 doc/development/tutorials/examples/todo.py | 106 | 125| 148 | 40944 | 181924 | 
| 131 | 52 sphinx/ext/autosummary/generate.py | 158 | 183| 274 | 41218 | 181924 | 
| 132 | 52 sphinx/ext/todo.py | 80 | 104| 206 | 41424 | 181924 | 
| 133 | 52 sphinx/builders/texinfo.py | 11 | 38| 228 | 41652 | 181924 | 
| 134 | 53 sphinx/transforms/references.py | 11 | 68| 362 | 42014 | 182338 | 
| 135 | 53 sphinx/directives/other.py | 43 | 62| 151 | 42165 | 182338 | 
| 136 | 53 sphinx/util/docfields.py | 80 | 86| 121 | 42286 | 182338 | 
| 137 | 54 sphinx/util/jsdump.py | 107 | 201| 587 | 42873 | 183753 | 
| 138 | 54 sphinx/ext/autodoc/importer.py | 104 | 125| 165 | 43038 | 183753 | 
| 139 | 54 sphinx/ext/autodoc/directive.py | 92 | 106| 134 | 43172 | 183753 | 
| 140 | 54 sphinx/domains/python.py | 858 | 876| 143 | 43315 | 183753 | 
| 141 | 54 sphinx/registry.py | 198 | 207| 125 | 43440 | 183753 | 
| 142 | 54 doc/development/tutorials/examples/todo.py | 30 | 61| 230 | 43670 | 183753 | 
| 143 | 55 sphinx/builders/html/__init__.py | 11 | 67| 457 | 44127 | 194776 | 
| 144 | 55 sphinx/domains/python.py | 159 | 177| 214 | 44341 | 194776 | 
| 145 | 56 sphinx/writers/manpage.py | 313 | 411| 757 | 45098 | 198287 | 


## Patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -1004,7 +1004,7 @@ def can_document_member(cls, member: Any, membername: str, isattr: bool, parent:
                 (inspect.isroutine(member) and isinstance(parent, ModuleDocumenter)))
 
     def format_args(self, **kwargs: Any) -> str:
-        if self.env.config.autodoc_typehints == 'none':
+        if self.env.config.autodoc_typehints in ('none', 'description'):
             kwargs.setdefault('show_annotation', False)
 
         if inspect.isbuiltin(self.object) or inspect.ismethoddescriptor(self.object):
@@ -1744,7 +1744,8 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_config_value('autodoc_default_options', {}, True)
     app.add_config_value('autodoc_docstring_signature', True, True)
     app.add_config_value('autodoc_mock_imports', [], True)
-    app.add_config_value('autodoc_typehints', "signature", True, ENUM("signature", "none"))
+    app.add_config_value('autodoc_typehints', "signature", True,
+                         ENUM("signature", "description", "none"))
     app.add_config_value('autodoc_warningiserror', True, True)
     app.add_config_value('autodoc_inherit_docstrings', True, True)
     app.add_event('autodoc-before-process-signature')
@@ -1753,5 +1754,6 @@ def setup(app: Sphinx) -> Dict[str, Any]:
     app.add_event('autodoc-skip-member')
 
     app.setup_extension('sphinx.ext.autodoc.type_comment')
+    app.setup_extension('sphinx.ext.autodoc.typehints')
 
     return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
diff --git a/sphinx/ext/autodoc/typehints.py b/sphinx/ext/autodoc/typehints.py
--- a/sphinx/ext/autodoc/typehints.py
+++ b/sphinx/ext/autodoc/typehints.py
@@ -18,21 +18,9 @@
 
 from sphinx import addnodes
 from sphinx.application import Sphinx
-from sphinx.config import Config, ENUM
 from sphinx.util import inspect, typing
 
 
-def config_inited(app: Sphinx, config: Config) -> None:
-    if config.autodoc_typehints == 'description':
-        # HACK: override this to make autodoc suppressing typehints in signatures
-        config.autodoc_typehints = 'none'  # type: ignore
-
-        # preserve user settings
-        app._autodoc_typehints_description = True  # type: ignore
-    else:
-        app._autodoc_typehints_description = False  # type: ignore
-
-
 def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
                      options: Dict, args: str, retann: str) -> None:
     """Record type hints to env object."""
@@ -53,7 +41,7 @@ def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
 def merge_typehints(app: Sphinx, domain: str, objtype: str, contentnode: Element) -> None:
     if domain != 'py':
         return
-    if app._autodoc_typehints_description is False:  # type: ignore
+    if app.config.autodoc_typehints != 'description':
         return
 
     signature = cast(addnodes.desc_signature, contentnode.parent[0])
@@ -141,10 +129,6 @@ def modify_field_list(node: nodes.field_list, annotations: Dict[str, str]) -> No
 
 
 def setup(app: Sphinx) -> Dict[str, Any]:
-    app.setup_extension('sphinx.ext.autodoc')
-    app.config.values['autodoc_typehints'] = ('signature', True,
-                                              ENUM("signature", "description", "none"))
-    app.connect('config-inited', config_inited)
     app.connect('autodoc-process-signature', record_typehints)
     app.connect('object-description-transform', merge_typehints)
 

```

## Test Patch

```diff
diff --git a/tests/test_ext_autodoc_configs.py b/tests/test_ext_autodoc_configs.py
--- a/tests/test_ext_autodoc_configs.py
+++ b/tests/test_ext_autodoc_configs.py
@@ -462,10 +462,9 @@ def test_mocked_module_imports(app, warning):
     assert warning.getvalue() == ''
 
 
-@pytest.mark.sphinx('html', testroot='ext-autodoc')
+@pytest.mark.sphinx('html', testroot='ext-autodoc',
+                    confoverrides={'autodoc_typehints': "signature"})
 def test_autodoc_typehints_signature(app):
-    app.config.autodoc_typehints = "signature"
-
     options = {"members": None,
                "undoc-members": True}
     actual = do_autodoc(app, 'module', 'target.typehints', options)
@@ -513,10 +512,9 @@ def test_autodoc_typehints_signature(app):
     ]
 
 
-@pytest.mark.sphinx('html', testroot='ext-autodoc')
+@pytest.mark.sphinx('html', testroot='ext-autodoc',
+                    confoverrides={'autodoc_typehints': "none"})
 def test_autodoc_typehints_none(app):
-    app.config.autodoc_typehints = "none"
-
     options = {"members": None,
                "undoc-members": True}
     actual = do_autodoc(app, 'module', 'target.typehints', options)
@@ -564,8 +562,7 @@ def test_autodoc_typehints_none(app):
 
 
 @pytest.mark.sphinx('text', testroot='ext-autodoc',
-                    confoverrides={'extensions': ['sphinx.ext.autodoc.typehints'],
-                                   'autodoc_typehints': 'description'})
+                    confoverrides={'autodoc_typehints': "description"})
 def test_autodoc_typehints_description(app):
     app.build()
     context = (app.outdir / 'index.txt').read_text()

```


## Code snippets

### 1 - sphinx/ext/autodoc/typehints.py:

Start line: 143, End line: 156

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.setup_extension('sphinx.ext.autodoc')
    app.config.values['autodoc_typehints'] = ('signature', True,
                                              ENUM("signature", "description", "none"))
    app.connect('config-inited', config_inited)
    app.connect('autodoc-process-signature', record_typehints)
    app.connect('object-description-transform', merge_typehints)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 2 - sphinx/ext/autodoc/typehints.py:

Start line: 11, End line: 33

```python
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable
from typing import cast

from docutils import nodes
from docutils.nodes import Element

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config, ENUM
from sphinx.util import inspect, typing


def config_inited(app: Sphinx, config: Config) -> None:
    if config.autodoc_typehints == 'description':
        # HACK: override this to make autodoc suppressing typehints in signatures
        config.autodoc_typehints = 'none'  # type: ignore

        # preserve user settings
        app._autodoc_typehints_description = True  # type: ignore
    else:
        app._autodoc_typehints_description = False  # type: ignore
```
### 3 - sphinx/ext/autodoc/typehints.py:

Start line: 53, End line: 85

```python
def merge_typehints(app: Sphinx, domain: str, objtype: str, contentnode: Element) -> None:
    if domain != 'py':
        return
    if app._autodoc_typehints_description is False:  # type: ignore
        return

    signature = cast(addnodes.desc_signature, contentnode.parent[0])
    if signature['module']:
        fullname = '.'.join([signature['module'], signature['fullname']])
    else:
        fullname = signature['fullname']
    annotations = app.env.temp_data.get('annotations', {})
    if annotations.get(fullname, {}):
        field_lists = [n for n in contentnode if isinstance(n, nodes.field_list)]
        if field_lists == []:
            field_list = insert_field_list(contentnode)
            field_lists.append(field_list)

        for field_list in field_lists:
            modify_field_list(field_list, annotations[fullname])


def insert_field_list(node: Element) -> nodes.field_list:
    field_list = nodes.field_list()
    desc = [n for n in node if isinstance(n, addnodes.desc)]
    if desc:
        # insert just before sub object descriptions (ex. methods, nested classes, etc.)
        index = node.index(desc[0])
        node.insert(index - 1, [field_list])
    else:
        node += field_list

    return field_list
```
### 4 - sphinx/ext/autodoc/__init__.py:

Start line: 1725, End line: 1758

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_autodocumenter(ModuleDocumenter)
    app.add_autodocumenter(ClassDocumenter)
    app.add_autodocumenter(ExceptionDocumenter)
    app.add_autodocumenter(DataDocumenter)
    app.add_autodocumenter(DataDeclarationDocumenter)
    app.add_autodocumenter(FunctionDocumenter)
    app.add_autodocumenter(SingledispatchFunctionDocumenter)
    app.add_autodocumenter(DecoratorDocumenter)
    app.add_autodocumenter(MethodDocumenter)
    app.add_autodocumenter(SingledispatchMethodDocumenter)
    app.add_autodocumenter(AttributeDocumenter)
    app.add_autodocumenter(PropertyDocumenter)
    app.add_autodocumenter(InstanceAttributeDocumenter)
    app.add_autodocumenter(SlotsAttributeDocumenter)

    app.add_config_value('autoclass_content', 'class', True)
    app.add_config_value('autodoc_member_order', 'alphabetic', True)
    app.add_config_value('autodoc_default_flags', [], True)
    app.add_config_value('autodoc_default_options', {}, True)
    app.add_config_value('autodoc_docstring_signature', True, True)
    app.add_config_value('autodoc_mock_imports', [], True)
    app.add_config_value('autodoc_typehints', "signature", True, ENUM("signature", "none"))
    app.add_config_value('autodoc_warningiserror', True, True)
    app.add_config_value('autodoc_inherit_docstrings', True, True)
    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')

    app.setup_extension('sphinx.ext.autodoc.type_comment')

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 1480, End line: 1517

```python
class SingledispatchMethodDocumenter(MethodDocumenter):

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        # intercept generated directive headers
        # TODO: It is very hacky to use mock to intercept header generation
        with patch.object(self, 'add_line') as add_line:
            super().add_directive_header(sig)

        # output first line of header
        self.add_line(*add_line.call_args_list[0][0])

        # inserts signature of singledispatch'ed functions
        meth = self.parent.__dict__.get(self.objpath[-1])
        for typ, func in meth.dispatcher.registry.items():
            if typ is object:
                pass  # default implementation. skipped.
            else:
                self.annotate_to_first_argument(func, typ)

                documenter = MethodDocumenter(self.directive, '')
                documenter.object = func
                self.add_line('   %s%s' % (self.format_name(),
                                           documenter.format_signature()),
                              sourcename)

        # output remains of directive header
        for call in add_line.call_args_list[1:]:
            self.add_line(*call[0])

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
        """Annotate type hint to the first argument of function if needed."""
        sig = inspect.signature(func, bound_method=True)
        if len(sig.parameters) == 0:
            return

        name = list(sig.parameters)[0]
        if name not in func.__annotations__:
            func.__annotations__[name] = typ
```
### 6 - sphinx/ext/autodoc/typehints.py:

Start line: 36, End line: 50

```python
def record_typehints(app: Sphinx, objtype: str, name: str, obj: Any,
                     options: Dict, args: str, retann: str) -> None:
    """Record type hints to env object."""
    try:
        if callable(obj):
            annotations = app.env.temp_data.setdefault('annotations', {})
            annotation = annotations.setdefault(name, OrderedDict())
            sig = inspect.signature(obj)
            for param in sig.parameters.values():
                if param.annotation is not param.empty:
                    annotation[param.name] = typing.stringify(param.annotation)
            if sig.return_annotation is not sig.empty:
                annotation['return'] = typing.stringify(sig.return_annotation)
    except (TypeError, ValueError):
        pass
```
### 7 - sphinx/ext/autodoc/importer.py:

Start line: 39, End line: 101

```python
def import_object(modname: str, objpath: List[str], objtype: str = '',
                  attrgetter: Callable[[Any, str], Any] = safe_getattr,
                  warningiserror: bool = False) -> Any:
    if objpath:
        logger.debug('[autodoc] from %s import %s', modname, '.'.join(objpath))
    else:
        logger.debug('[autodoc] import %s', modname)

    try:
        module = None
        exc_on_importing = None
        objpath = list(objpath)
        while module is None:
            try:
                module = import_module(modname, warningiserror=warningiserror)
                logger.debug('[autodoc] import %s => %r', modname, module)
            except ImportError as exc:
                logger.debug('[autodoc] import %s => failed', modname)
                exc_on_importing = exc
                if '.' in modname:
                    # retry with parent module
                    modname, name = modname.rsplit('.', 1)
                    objpath.insert(0, name)
                else:
                    raise

        obj = module
        parent = None
        object_name = None
        for attrname in objpath:
            parent = obj
            logger.debug('[autodoc] getattr(_, %r)', attrname)
            obj = attrgetter(obj, attrname)
            logger.debug('[autodoc] => %r', obj)
            object_name = attrname
        return [module, parent, object_name, obj]
    except (AttributeError, ImportError) as exc:
        if isinstance(exc, AttributeError) and exc_on_importing:
            # restore ImportError
            exc = exc_on_importing

        if objpath:
            errmsg = ('autodoc: failed to import %s %r from module %r' %
                      (objtype, '.'.join(objpath), modname))
        else:
            errmsg = 'autodoc: failed to import %s %r' % (objtype, modname)

        if isinstance(exc, ImportError):
            # import_module() raises ImportError having real exception obj and
            # traceback
            real_exc, traceback_msg = exc.args
            if isinstance(real_exc, SystemExit):
                errmsg += ('; the module executes module level statement '
                           'and it might call sys.exit().')
            elif isinstance(real_exc, ImportError) and real_exc.args:
                errmsg += '; the following exception was raised:\n%s' % real_exc.args[0]
            else:
                errmsg += '; the following exception was raised:\n%s' % traceback_msg
        else:
            errmsg += '; the following exception was raised:\n%s' % traceback.format_exc()

        logger.debug(errmsg)
        raise ImportError(errmsg)
```
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 824, End line: 857

```python
class ModuleDocumenter(Documenter):

    def get_object_members(self, want_all: bool) -> Tuple[bool, List[Tuple[str, object]]]:
        if want_all:
            if (self.options.ignore_module_all or not
                    hasattr(self.object, '__all__')):
                # for implicit module members, check __module__ to avoid
                # documenting imported objects
                return True, get_module_members(self.object)
            else:
                memberlist = self.object.__all__
                # Sometimes __all__ is broken...
                if not isinstance(memberlist, (list, tuple)) or not \
                   all(isinstance(entry, str) for entry in memberlist):
                    logger.warning(
                        __('__all__ should be a list of strings, not %r '
                           '(in module %s) -- ignoring __all__') %
                        (memberlist, self.fullname),
                        type='autodoc'
                    )
                    # fall back to all members
                    return True, get_module_members(self.object)
        else:
            memberlist = self.options.members or []
        ret = []
        for mname in memberlist:
            try:
                ret.append((mname, safe_getattr(self.object, mname)))
            except AttributeError:
                logger.warning(
                    __('missing attribute mentioned in :members: or __all__: '
                       'module %s, attribute %s') %
                    (safe_getattr(self.object, '__name__', '???'), mname),
                    type='autodoc'
                )
        return False, ret
```
### 9 - sphinx/ext/autosummary/__init__.py:

Start line: 752, End line: 778

```python
def setup(app: Sphinx) -> Dict[str, Any]:
    # I need autodoc
    app.setup_extension('sphinx.ext.autodoc')
    app.add_node(autosummary_toc,
                 html=(autosummary_toc_visit_html, autosummary_noop),
                 latex=(autosummary_noop, autosummary_noop),
                 text=(autosummary_noop, autosummary_noop),
                 man=(autosummary_noop, autosummary_noop),
                 texinfo=(autosummary_noop, autosummary_noop))
    app.add_node(autosummary_table,
                 html=(autosummary_table_visit_html, autosummary_noop),
                 latex=(autosummary_noop, autosummary_noop),
                 text=(autosummary_noop, autosummary_noop),
                 man=(autosummary_noop, autosummary_noop),
                 texinfo=(autosummary_noop, autosummary_noop))
    app.add_directive('autosummary', Autosummary)
    app.add_role('autolink', AutoLink())
    app.connect('doctree-read', process_autosummary_toc)
    app.connect('builder-inited', process_generate_options)
    app.add_config_value('autosummary_generate', [], True, [bool])
    app.add_config_value('autosummary_generate_overwrite', True, False)
    app.add_config_value('autosummary_mock_imports',
                         lambda config: config.autodoc_mock_imports, 'env')
    app.add_config_value('autosummary_imported_members', [], False, [bool])

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
```
### 10 - sphinx/ext/autodoc/__init__.py:

Start line: 811, End line: 822

```python
class ModuleDocumenter(Documenter):

    def add_directive_header(self, sig: str) -> None:
        Documenter.add_directive_header(self, sig)

        sourcename = self.get_sourcename()

        # add some module-specific options
        if self.options.synopsis:
            self.add_line('   :synopsis: ' + self.options.synopsis, sourcename)
        if self.options.platform:
            self.add_line('   :platform: ' + self.options.platform, sourcename)
        if self.options.deprecated:
            self.add_line('   :deprecated:', sourcename)
```
### 12 - sphinx/ext/autodoc/__init__.py:

Start line: 1077, End line: 1286

```python
class SingledispatchFunctionDocumenter(FunctionDocumenter):

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        # intercept generated directive headers
        # TODO: It is very hacky to use mock to intercept header generation
        with patch.object(self, 'add_line') as add_line:
            super().add_directive_header(sig)

        # output first line of header
        self.add_line(*add_line.call_args_list[0][0])

        # inserts signature of singledispatch'ed functions
        for typ, func in self.object.registry.items():
            if typ is object:
                pass  # default implementation. skipped.
            else:
                self.annotate_to_first_argument(func, typ)

                documenter = FunctionDocumenter(self.directive, '')
                documenter.object = func
                self.add_line('   %s%s' % (self.format_name(),
                                           documenter.format_signature()),
                              sourcename)

        # output remains of directive header
        for call in add_line.call_args_list[1:]:
            self.add_line(*call[0])

    def annotate_to_first_argument(self, func: Callable, typ: Type) -> None:
        """Annotate type hint to the first argument of function if needed."""
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            return

        name = list(sig.parameters)[0]
        if name not in func.__annotations__:
            func.__annotations__[name] = typ


class DecoratorDocumenter(FunctionDocumenter):
    """
    Specialized Documenter subclass for decorator functions.
    """
    objtype = 'decorator'

    # must be lower than FunctionDocumenter
    priority = -1

    def format_args(self, **kwargs: Any) -> Any:
        args = super().format_args(**kwargs)
        if ',' in args:
            return args
        else:
            return None


class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):
```
### 13 - sphinx/ext/autodoc/__init__.py:

Start line: 548, End line: 625

```python
class Documenter:

    def filter_members(self, members: List[Tuple[str, Any]], want_all: bool
                       ) -> List[Tuple[str, Any, bool]]:
        # ... other code
        for (membername, member) in members:
            # if isattr is True, the member is documented as an attribute
            if member is INSTANCEATTR:
                isattr = True
            else:
                isattr = False

            doc = getdoc(member, self.get_attr, self.env.config.autodoc_inherit_docstrings)

            # if the member __doc__ is the same as self's __doc__, it's just
            # inherited and therefore not the member's doc
            cls = self.get_attr(member, '__class__', None)
            if cls:
                cls_doc = self.get_attr(cls, '__doc__', None)
                if cls_doc == doc:
                    doc = None
            has_doc = bool(doc)

            metadata = extract_metadata(doc)
            if 'private' in metadata:
                # consider a member private if docstring has "private" metadata
                isprivate = True
            else:
                isprivate = membername.startswith('_')

            keep = False
            if want_all and membername.startswith('__') and \
                    membername.endswith('__') and len(membername) > 4:
                # special __methods__
                if self.options.special_members is ALL:
                    if membername == '__doc__':
                        keep = False
                    elif is_filtered_inherited_member(membername):
                        keep = False
                    else:
                        keep = has_doc or self.options.undoc_members
                elif self.options.special_members:
                    if membername in self.options.special_members:
                        keep = has_doc or self.options.undoc_members
            elif (namespace, membername) in attr_docs:
                if want_all and isprivate:
                    # ignore members whose name starts with _ by default
                    keep = self.options.private_members
                else:
                    # keep documented attributes
                    keep = True
                isattr = True
            elif want_all and isprivate:
                # ignore members whose name starts with _ by default
                keep = self.options.private_members and \
                    (has_doc or self.options.undoc_members)
            else:
                if self.options.members is ALL and is_filtered_inherited_member(membername):
                    keep = False
                else:
                    # ignore undocumented members if :undoc-members: is not given
                    keep = has_doc or self.options.undoc_members

            # give the user a chance to decide whether this member
            # should be skipped
            if self.env.app:
                # let extensions preprocess docstrings
                try:
                    skip_user = self.env.app.emit_firstresult(
                        'autodoc-skip-member', self.objtype, membername, member,
                        not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
                except Exception as exc:
                    logger.warning(__('autodoc: failed to determine %r to be documented.'
                                      'the following exception was raised:\n%s'),
                                   member, exc, type='autodoc')
                    keep = False

            if keep:
                ret.append((membername, member, isattr))

        return ret
```
### 14 - sphinx/ext/autodoc/typehints.py:

Start line: 88, End line: 140

```python
def modify_field_list(node: nodes.field_list, annotations: Dict[str, str]) -> None:
    arguments = {}  # type: Dict[str, Dict[str, bool]]
    fields = cast(Iterable[nodes.field], node)
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                # :param xxx:
                arg = arguments.setdefault(parts[1], {})
                arg['param'] = True
            elif len(parts) > 2:
                # :param xxx yyy:
                name = ' '.join(parts[2:])
                arg = arguments.setdefault(name, {})
                arg['param'] = True
                arg['type'] = True
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            arg = arguments.setdefault(name, {})
            arg['type'] = True
        elif parts[0] == 'rtype':
            arguments['return'] = {'type': True}

    for name, annotation in annotations.items():
        if name == 'return':
            continue

        arg = arguments.get(name, {})
        field = nodes.field()
        if arg.get('param') and arg.get('type'):
            # both param and type are already filled manually
            continue
        elif arg.get('param'):
            # only param: fill type field
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotation))
        elif arg.get('type'):
            # only type: It's odd...
            field += nodes.field_name('', 'param ' + name)
            field += nodes.field_body('', nodes.paragraph('', ''))
        else:
            # both param and type are not found
            field += nodes.field_name('', 'param ' + annotation + ' ' + name)
            field += nodes.field_body('', nodes.paragraph('', ''))

        node += field

    if 'return' in annotations and 'return' not in arguments:
        field = nodes.field()
        field += nodes.field_name('', 'rtype')
        field += nodes.field_body('', nodes.paragraph('', annotation))
        node += field
```
### 17 - sphinx/ext/autodoc/__init__.py:

Start line: 202, End line: 272

```python
class Documenter:
    """
    A Documenter knows how to autodocument a single object type.  When
    registered with the AutoDirective, it will be used to document objects
    of that type when needed by autodoc.

    Its *objtype* attribute selects what auto directive it is assigned to
    (the directive name is 'auto' + objtype), and what directive it generates
    by default, though that can be overridden by an attribute called
    *directivetype*.

    A Documenter has an *option_spec* that works like a docutils directive's;
    in fact, it will be used to parse an auto directive's options that matches
    the documenter.
    """
    #: name by which the directive is called (auto...) and the default
    #: generated directive name
    objtype = 'object'
    #: indentation by which to indent the directive content
    content_indent = '   '
    #: priority if multiple documenters return True from can_document_member
    priority = 0
    #: order if autodoc_member_order is set to 'groupwise'
    member_order = 0
    #: true if the generated content may contain titles
    titles_allowed = False

    option_spec = {'noindex': bool_option}  # type: Dict[str, Callable]

    def get_attr(self, obj: Any, name: str, *defargs: Any) -> Any:
        """getattr() override for types such as Zope interfaces."""
        return autodoc_attrgetter(self.env.app, obj, name, *defargs)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """Called to see if a member can be documented by this documenter."""
        raise NotImplementedError('must be implemented in subclasses')

    def __init__(self, directive: "DocumenterBridge", name: str, indent: str = '') -> None:
        self.directive = directive
        self.env = directive.env    # type: BuildEnvironment
        self.options = directive.genopt
        self.name = name
        self.indent = indent
        # the module and object path within the module, and the fully
        # qualified name (all set after resolve_name succeeds)
        self.modname = None         # type: str
        self.module = None          # type: ModuleType
        self.objpath = None         # type: List[str]
        self.fullname = None        # type: str
        # extra signature items (arguments and return annotation,
        # also set after resolve_name succeeds)
        self.args = None            # type: str
        self.retann = None          # type: str
        # the object to document (set after import_object succeeds)
        self.object = None          # type: Any
        self.object_name = None     # type: str
        # the parent/owner of the object to document
        self.parent = None          # type: Any
        # the module analyzer to get at attribute docs, or None
        self.analyzer = None        # type: ModuleAnalyzer

    @property
    def documenters(self) -> Dict[str, "Type[Documenter]"]:
        """Returns registered Documenter classes"""
        return get_documenters(self.env.app)

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        """Append one line of generated reST to the output."""
        self.directive.result.append(self.indent + line, source, *lineno)
```
### 18 - sphinx/ext/autodoc/__init__.py:

Start line: 1133, End line: 1286

```python
class ClassDocumenter(DocstringSignatureMixin, ModuleLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for classes.
    """
    objtype = 'class'
    member_order = 20
    option_spec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'member-order': identity,
        'exclude-members': members_set_option,
        'private-members': bool_option, 'special-members': members_option,
    }  # type: Dict[str, Callable]

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        merge_special_members_option(self.options)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return isinstance(member, type)

    def import_object(self) -> Any:
        ret = super().import_object()
        # if the class is documented under another name, document it
        # as data/attribute
        if ret:
            if hasattr(self.object, '__name__'):
                self.doc_as_attr = (self.objpath[-1] != self.object.__name__)
            else:
                self.doc_as_attr = True
        return ret

    def format_args(self, **kwargs: Any) -> str:
        if self.env.config.autodoc_typehints == 'none':
            kwargs.setdefault('show_annotation', False)

        # for classes, the relevant signature is the __init__ method's
        initmeth = self.get_attr(self.object, '__init__', None)
        # classes without __init__ method, default __init__ or
        # __init__ written in C?
        if initmeth is None or \
                inspect.is_builtin_class_method(self.object, '__init__') or \
                not(inspect.ismethod(initmeth) or inspect.isfunction(initmeth)):
            return None
        try:
            self.env.app.emit('autodoc-before-process-signature', initmeth, True)
            sig = inspect.signature(initmeth, bound_method=True)
            return stringify_signature(sig, show_return_annotation=False, **kwargs)
        except TypeError:
            # still not possible: happens e.g. for old-style classes
            # with __init__ in C
            return None

    def format_signature(self, **kwargs: Any) -> str:
        if self.doc_as_attr:
            return ''

        return super().format_signature(**kwargs)

    def add_directive_header(self, sig: str) -> None:
        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            sourcename = self.get_sourcename()
            self.add_line('', sourcename)
            if hasattr(self.object, '__bases__') and len(self.object.__bases__):
                bases = [':class:`%s`' % b.__name__
                         if b.__module__ in ('__builtin__', 'builtins')
                         else ':class:`%s.%s`' % (b.__module__, b.__qualname__)
                         for b in self.object.__bases__]
                self.add_line('   ' + _('Bases: %s') % ', '.join(bases),
                              sourcename)

    def get_doc(self, encoding: str = None, ignore: int = 1) -> List[List[str]]:
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning)
        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        content = self.env.config.autoclass_content

        docstrings = []
        attrdocstring = self.get_attr(self.object, '__doc__', None)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if content in ('both', 'init'):
            __init__ = self.get_attr(self.object, '__init__', None)
            initdocstring = getdoc(__init__, self.get_attr,
                                   self.env.config.autodoc_inherit_docstrings)
            # for new-style classes, no __init__ means default __init__
            if (initdocstring is not None and
                (initdocstring == object.__init__.__doc__ or  # for pypy
                 initdocstring.strip() == object.__init__.__doc__)):  # for !pypy
                initdocstring = None
            if not initdocstring:
                # try __new__
                __new__ = self.get_attr(self.object, '__new__', None)
                initdocstring = getdoc(__new__, self.get_attr,
                                       self.env.config.autodoc_inherit_docstrings)
                # for new-style classes, no __new__ means default __new__
                if (initdocstring is not None and
                    (initdocstring == object.__new__.__doc__ or  # for pypy
                     initdocstring.strip() == object.__new__.__doc__)):  # for !pypy
                    initdocstring = None
            if initdocstring:
                if content == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        if self.doc_as_attr:
            classname = safe_getattr(self.object, '__qualname__', None)
            if not classname:
                classname = safe_getattr(self.object, '__name__', None)
            if classname:
                module = safe_getattr(self.object, '__module__', None)
                parentmodule = safe_getattr(self.parent, '__module__', None)
                if module and module != parentmodule:
                    classname = str(module) + '.' + str(classname)
                content = StringList([_('alias of :class:`%s`') % classname], source='')
                super().add_content(content, no_docstring=True)
        else:
            super().add_content(more_content)

    def document_members(self, all_members: bool = False) -> None:
        if self.doc_as_attr:
            return
        super().document_members(all_members)

    def generate(self, more_content: Any = None, real_modname: str = None,
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
### 23 - sphinx/ext/autodoc/__init__.py:

Start line: 1711, End line: 1722

```python
def get_documenters(app: Sphinx) -> Dict[str, "Type[Documenter]"]:
    """Returns registered Documenter classes"""
    return app.registry.documenters


def autodoc_attrgetter(app: Sphinx, obj: Any, name: str, *defargs: Any) -> Any:
    """Alternative getattr() for types"""
    for typ, func in app.registry.autodoc_attrgettrs.items():
        if isinstance(obj, typ):
            return func(obj, name, *defargs)

    return safe_getattr(obj, name, *defargs)
```
### 24 - sphinx/ext/autodoc/__init__.py:

Start line: 455, End line: 483

```python
class Documenter:

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Add content from docstrings, attribute documentation and user."""
        # set sourcename and add content from attribute documentation
        sourcename = self.get_sourcename()
        if self.analyzer:
            attr_docs = self.analyzer.find_attr_docs()
            if self.objpath:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if key in attr_docs:
                    no_docstring = True
                    docstrings = [attr_docs[key]]
                    for i, line in enumerate(self.process_doc(docstrings)):
                        self.add_line(line, sourcename, i)

        # add content from docstrings
        if not no_docstring:
            docstrings = self.get_doc()
            if not docstrings:
                # append at least a dummy docstring, so that the event
                # autodoc-process-docstring is fired and can add some
                # content if desired
                docstrings.append([])
            for i, line in enumerate(self.process_doc(docstrings)):
                self.add_line(line, sourcename, i)

        # add additional content (e.g. from document), if present
        if more_content:
            for line, src in zip(more_content.data, more_content.items):
                self.add_line(line, src[0], src[1])
```
### 29 - sphinx/ext/autodoc/__init__.py:

Start line: 285, End line: 319

```python
class Documenter:

    def parse_name(self) -> bool:
        """Determine what module to import and what attribute to document.

        Returns True and sets *self.modname*, *self.objpath*, *self.fullname*,
        *self.args* and *self.retann* if parsing and resolving was successful.
        """
        # first, parse the definition -- auto directives for classes and
        # functions can contain a signature which is then used instead of
        # an autogenerated one
        try:
            explicit_modname, path, base, args, retann = \
                py_ext_sig_re.match(self.name).groups()
        except AttributeError:
            logger.warning(__('invalid signature for auto%s (%r)') % (self.objtype, self.name),
                           type='autodoc')
            return False

        # support explicit module and class name separation via ::
        if explicit_modname is not None:
            modname = explicit_modname[:-2]
            parents = path.rstrip('.').split('.') if path else []
        else:
            modname = None
            parents = []

        self.modname, self.objpath = self.resolve_name(modname, parents, path, base)

        if not self.modname:
            return False

        self.args = args
        self.retann = retann
        self.fullname = (self.modname or '') + \
                        ('.' + '.'.join(self.objpath) if self.objpath else '')
        return True
```
### 32 - sphinx/ext/autodoc/__init__.py:

Start line: 440, End line: 453

```python
class Documenter:

    def process_doc(self, docstrings: List[List[str]]) -> Iterator[str]:
        """Let the user process the docstrings before adding them."""
        for docstringlines in docstrings:
            if self.env.app:
                # let extensions preprocess docstrings
                self.env.app.emit('autodoc-process-docstring',
                                  self.objtype, self.fullname, self.object,
                                  self.options, docstringlines)
            yield from docstringlines

    def get_sourcename(self) -> str:
        if self.analyzer:
            return '%s:docstring of %s' % (self.analyzer.srcname, self.fullname)
        return 'docstring of %s' % self.fullname
```
### 34 - sphinx/ext/autodoc/__init__.py:

Start line: 1354, End line: 1388

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

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only INSTANCEATTR members."""
        return (isinstance(parent, ModuleDocumenter) and
                isattr and
                member is INSTANCEATTR)

    def import_object(self) -> bool:
        """Never import anything."""
        # disguise as a data
        self.objtype = 'data'
        try:
            # import module to obtain type annotation
            self.parent = importlib.import_module(self.modname)
        except ImportError:
            pass

        return True

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Never try to get a docstring from the object."""
        super().add_content(more_content, no_docstring=True)
```
### 39 - sphinx/ext/autodoc/__init__.py:

Start line: 412, End line: 425

```python
class Documenter:

    def add_directive_header(self, sig: str) -> None:
        """Add the directive header and options to the generated content."""
        domain = getattr(self, 'domain', 'py')
        directive = getattr(self, 'directivetype', self.objtype)
        name = self.format_name()
        sourcename = self.get_sourcename()
        self.add_line('.. %s:%s:: %s%s' % (domain, directive, name, sig),
                      sourcename)
        if self.options.noindex:
            self.add_line('   :noindex:', sourcename)
        if self.objpath:
            # Be explicit about the module, this is necessary since .. class::
            # etc. don't support a prepended module name
            self.add_line('   :module: %s' % self.modname, sourcename)
```
### 40 - sphinx/ext/autodoc/__init__.py:

Start line: 321, End line: 337

```python
class Documenter:

    def import_object(self) -> bool:
        """Import the object given by *self.modname* and *self.objpath* and set
        it as *self.object*.

        Returns True if successful, False if an error occurred.
        """
        with mock(self.env.config.autodoc_mock_imports):
            try:
                ret = import_object(self.modname, self.objpath, self.objtype,
                                    attrgetter=self.get_attr,
                                    warningiserror=self.env.config.autodoc_warningiserror)
                self.module, self.parent, self.object_name, self.object = ret
                return True
            except ImportError as exc:
                logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                self.env.note_reread()
                return False
```
### 46 - sphinx/ext/autodoc/__init__.py:

Start line: 1682, End line: 1708

```python
class SlotsAttributeDocumenter(AttributeDocumenter):

    def import_object(self) -> Any:
        """Never import anything."""
        # disguise as an attribute
        self.objtype = 'attribute'
        self._datadescriptor = True

        with mock(self.env.config.autodoc_mock_imports):
            try:
                ret = import_object(self.modname, self.objpath[:-1], 'class',
                                    attrgetter=self.get_attr,
                                    warningiserror=self.env.config.autodoc_warningiserror)
                self.module, _, _, self.parent = ret
                return True
            except ImportError as exc:
                logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                self.env.note_reread()
                return False

    def get_doc(self, encoding: str = None, ignore: int = 1) -> List[List[str]]:
        """Decode and return lines of the docstring(s) for the object."""
        name = self.objpath[-1]
        __slots__ = safe_getattr(self.parent, '__slots__', [])
        if isinstance(__slots__, dict) and isinstance(__slots__.get(name), str):
            docstring = prepare_docstring(__slots__[name])
            return [docstring]
        else:
            return []
```
### 47 - sphinx/ext/autodoc/__init__.py:

Start line: 954, End line: 971

```python
class DocstringSignatureMixin:

    def get_doc(self, encoding: str = None, ignore: int = 1) -> List[List[str]]:
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning)
        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines
        return super().get_doc(None, ignore)  # type: ignore

    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.env.config.autodoc_docstring_signature:  # type: ignore
            # only act if a signature is not explicitly given already, and if
            # the feature is enabled
            result = self._find_signature()
            if result is not None:
                self.args, self.retann = result
        return super().format_signature(**kwargs)  # type: ignore
```
### 48 - sphinx/ext/autodoc/__init__.py:

Start line: 693, End line: 765

```python
class Documenter:

    def generate(self, more_content: Any = None, real_modname: str = None,
                 check_module: bool = False, all_members: bool = False) -> None:
        """Generate reST for the object given by *self.name*, and possibly for
        its members.

        If *more_content* is given, include that content. If *real_modname* is
        given, use that module name to find attribute docs. If *check_module* is
        True, only generate if the object is defined in the module name it is
        imported from. If *all_members* is True, document all members.
        """
        if not self.parse_name():
            # need a module to import
            logger.warning(
                __('don\'t know which module to import for autodocumenting '
                   '%r (try placing a "module" or "currentmodule" directive '
                   'in the document, or giving an explicit module name)') %
                self.name, type='autodoc')
            return

        # now, import the module and get object to document
        if not self.import_object():
            return

        # If there is no real module defined, figure out which to use.
        # The real module is used in the module analyzer to look up the module
        # where the attribute documentation would actually be found in.
        # This is used for situations where you have a module that collects the
        # functions and classes of internal submodules.
        self.real_modname = real_modname or self.get_real_modname()  # type: str

        # try to also get a source code analyzer for attribute docs
        try:
            self.analyzer = ModuleAnalyzer.for_module(self.real_modname)
            # parse right now, to get PycodeErrors on parsing (results will
            # be cached anyway)
            self.analyzer.find_attr_docs()
        except PycodeError as err:
            logger.debug('[autodoc] module analyzer failed: %s', err)
            # no source file -- e.g. for builtin and C modules
            self.analyzer = None
            # at least add the module.__file__ as a dependency
            if hasattr(self.module, '__file__') and self.module.__file__:
                self.directive.filename_set.add(self.module.__file__)
        else:
            self.directive.filename_set.add(self.analyzer.srcname)

        # check __module__ of object (for members not given explicitly)
        if check_module:
            if not self.check_module():
                return

        sourcename = self.get_sourcename()

        # make sure that the result starts with an empty line.  This is
        # necessary for some situations where another directive preprocesses
        # reST and no starting newline is present
        self.add_line('', sourcename)

        # format the object's signature, if any
        sig = self.format_signature()

        # generate the directive header and options, if applicable
        self.add_directive_header(sig)
        self.add_line('', sourcename)

        # e.g. the module directive doesn't have content
        self.indent += self.content_indent

        # add all content (from docstrings, attribute docs etc.)
        self.add_content(more_content)

        # document members, if possible
        self.document_members(all_members)
```
### 55 - sphinx/ext/autodoc/__init__.py:

Start line: 627, End line: 691

```python
class Documenter:

    def document_members(self, all_members: bool = False) -> None:
        """Generate reST for member documentation.

        If *all_members* is True, do all members, else those given by
        *self.options.members*.
        """
        # set current namespace for finding members
        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]

        want_all = all_members or self.options.inherited_members or \
            self.options.members is ALL
        # find out which members are documentable
        members_check_module, members = self.get_object_members(want_all)

        # remove members given by exclude-members
        if self.options.exclude_members:
            members = [
                (membername, member) for (membername, member) in members
                if (
                    self.options.exclude_members is ALL or
                    membername not in self.options.exclude_members
                )
            ]

        # document non-skipped members
        memberdocumenters = []  # type: List[Tuple[Documenter, bool]]
        for (mname, member, isattr) in self.filter_members(members, want_all):
            classes = [cls for cls in self.documenters.values()
                       if cls.can_document_member(member, mname, isattr, self)]
            if not classes:
                # don't know how to document this member
                continue
            # prefer the documenter with the highest priority
            classes.sort(key=lambda cls: cls.priority)
            # give explicitly separated module name, so that members
            # of inner classes can be documented
            full_mname = self.modname + '::' + \
                '.'.join(self.objpath + [mname])
            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter, isattr))
        member_order = self.options.member_order or \
            self.env.config.autodoc_member_order
        if member_order == 'groupwise':
            # sort by group; relies on stable sort to keep items in the
            # same group sorted alphabetically
            memberdocumenters.sort(key=lambda e: e[0].member_order)
        elif member_order == 'bysource' and self.analyzer:
            # sort by source order, by virtue of the module analyzer
            tagorder = self.analyzer.tagorder

            def keyfunc(entry: Tuple[Documenter, bool]) -> int:
                fullname = entry[0].name.split('::')[1]
                return tagorder.get(fullname, len(tagorder))
            memberdocumenters.sort(key=keyfunc)

        for documenter, isattr in memberdocumenters:
            documenter.generate(
                all_members=True, real_modname=self.real_modname,
                check_module=members_check_module and not isattr)

        # reset current objects
        self.env.temp_data['autodoc:module'] = None
        self.env.temp_data['autodoc:class'] = None
```
### 57 - sphinx/ext/autodoc/__init__.py:

Start line: 1391, End line: 1457

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
        return inspect.isroutine(member) and \
            not isinstance(parent, ModuleDocumenter)

    def import_object(self) -> Any:
        ret = super().import_object()
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
        if self.env.config.autodoc_typehints == 'none':
            kwargs.setdefault('show_annotation', False)

        if inspect.isbuiltin(self.object) or inspect.ismethoddescriptor(self.object):
            # can never get arguments of a C function or method
            return None
        if inspect.isstaticmethod(self.object, cls=self.parent, name=self.object_name):
            self.env.app.emit('autodoc-before-process-signature', self.object, False)
            sig = inspect.signature(self.object, bound_method=False)
        else:
            self.env.app.emit('autodoc-before-process-signature', self.object, True)
            sig = inspect.signature(self.object, bound_method=True)
        args = stringify_signature(sig, **kwargs)

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

    def document_members(self, all_members: bool = False) -> None:
        pass
```
### 66 - sphinx/ext/autodoc/__init__.py:

Start line: 427, End line: 438

```python
class Documenter:

    def get_doc(self, encoding: str = None, ignore: int = 1) -> List[List[str]]:
        """Decode and return lines of the docstring(s) for the object."""
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning)
        docstring = getdoc(self.object, self.get_attr,
                           self.env.config.autodoc_inherit_docstrings)
        if docstring:
            tab_width = self.directive.state.document.settings.tab_width
            return [prepare_docstring(docstring, ignore, tab_width)]
        return []
```
### 67 - sphinx/ext/autodoc/__init__.py:

Start line: 1632, End line: 1661

```python
class InstanceAttributeDocumenter(AttributeDocumenter):
    """
    Specialized Documenter subclass for attributes that cannot be imported
    because they are instance attributes (e.g. assigned in __init__).
    """
    objtype = 'instanceattribute'
    directivetype = 'attribute'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only INSTANCEATTR members."""
        return (not isinstance(parent, ModuleDocumenter) and
                isattr and
                member is INSTANCEATTR)

    def import_object(self) -> bool:
        """Never import anything."""
        # disguise as an attribute
        self.objtype = 'attribute'
        self._datadescriptor = False
        return True

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        """Never try to get a docstring from the object."""
        super().add_content(more_content, no_docstring=True)
```
### 71 - sphinx/ext/autodoc/__init__.py:

Start line: 1520, End line: 1598

```python
class AttributeDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):  # type: ignore
    """
    Specialized Documenter subclass for attributes.
    """
    objtype = 'attribute'
    member_order = 60
    option_spec = dict(ModuleLevelDocumenter.option_spec)
    option_spec["annotation"] = annotation_option

    # must be higher than the MethodDocumenter, else it will recognize
    # some non-data descriptors as methods
    priority = 10

    @staticmethod
    def is_function_or_method(obj: Any) -> bool:
        return inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        if inspect.isattributedescriptor(member):
            return True
        elif (not isinstance(parent, ModuleDocumenter) and
              not inspect.isroutine(member) and
              not isinstance(member, type)):
            return True
        else:
            return False

    def document_members(self, all_members: bool = False) -> None:
        pass

    def import_object(self) -> Any:
        ret = super().import_object()
        if inspect.isenumattribute(self.object):
            self.object = self.object.value
        if inspect.isattributedescriptor(self.object):
            self._datadescriptor = True
        else:
            # if it's not a data descriptor
            self._datadescriptor = False
        return ret

    def get_real_modname(self) -> str:
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if not self.options.annotation:
            if not self._datadescriptor:
                # obtain annotation for this attribute
                annotations = getattr(self.parent, '__annotations__', {})
                if annotations and self.objpath[-1] in annotations:
                    objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                    self.add_line('   :type: ' + objrepr, sourcename)
                else:
                    key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                    if self.analyzer and key in self.analyzer.annotations:
                        self.add_line('   :type: ' + self.analyzer.annotations[key],
                                      sourcename)

                try:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
                except ValueError:
                    pass
        elif self.options.annotation is SUPPRESS:
            pass
        else:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)

    def add_content(self, more_content: Any, no_docstring: bool = False) -> None:
        if not self._datadescriptor:
            # if it's not a data descriptor, its docstring is very probably the
            # wrong thing to display
            no_docstring = True
        super().add_content(more_content, no_docstring)
```
### 73 - sphinx/ext/autodoc/__init__.py:

Start line: 13, End line: 122

```python
import importlib
import re
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Sequence, Set, Tuple, Type, Union
from unittest.mock import patch

from docutils.statemachine import StringList

import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import import_object, get_module_members, get_object_members
from sphinx.ext.autodoc.mock import mock
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect
from sphinx.util import logging
from sphinx.util import rpartition
from sphinx.util.docstrings import extract_metadata, prepare_docstring
from sphinx.util.inspect import getdoc, object_description, safe_getattr, stringify_signature
from sphinx.util.typing import stringify as stringify_typehint

if False:
    # For type annotation
    from typing import Type  # NOQA # for python3.5.1
    from sphinx.ext.autodoc.directive import DocumenterBridge


logger = logging.getLogger(__name__)


# This type isn't exposed directly in any modules, but can be found
# here in most Python versions
MethodDescriptorType = type(type.__subclasses__)


#: extended signature RE: with explicit module name separated by ::
py_ext_sig_re = re.compile(
    r'''^ ([\w.]+::)?            # explicit module name
          ([\w.]+\.)?            # module and/or class name(s)
          (\w+)  \s*             # thing name
          (?: \((.*)\)           # optional: arguments
           (?:\s* -> \s* (.*))?  #           return annotation
          )? $                   # and nothing more
          ''', re.VERBOSE)


def identity(x: Any) -> Any:
    return x


ALL = object()
INSTANCEATTR = object()
SLOTSATTR = object()


def members_option(arg: Any) -> Union[object, List[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg is None or arg is True:
        return ALL
    return [x.strip() for x in arg.split(',') if x.strip()]


def members_set_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg is None:
        return ALL
    return {x.strip() for x in arg.split(',') if x.strip()}


def inherited_members_option(arg: Any) -> Union[object, Set[str]]:
    """Used to convert the :members: option to auto directives."""
    if arg is None:
        return 'object'
    else:
        return arg


SUPPRESS = object()


def annotation_option(arg: Any) -> Any:
    if arg is None:
        # suppress showing the representation of the object
        return SUPPRESS
    else:
        return arg


def bool_option(arg: Any) -> bool:
    """Used to convert flag options to auto directives.  (Instead of
    directives.flag(), which returns None).
    """
    return True


def merge_special_members_option(options: Dict) -> None:
    """Merge :special-members: option to :members: option."""
    if 'special-members' in options and options['special-members'] is not ALL:
        if options.get('members') is ALL:
            pass
        elif options.get('members'):
            for member in options['special-members']:
                if member not in options['members']:
                    options['members'].append(member)
        else:
            options['members'] = options['special-members']
```
### 74 - sphinx/ext/autodoc/__init__.py:

Start line: 860, End line: 878

```python
class ModuleLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on module level (functions,
    classes, data/constants).
    """
    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is None:
            if path:
                modname = path.rstrip('.')
            else:
                # if documenting a toplevel object without explicit module,
                # it can be contained in another auto directive ...
                modname = self.env.temp_data.get('autodoc:module')
                # ... or in the scope of a module directive
                if not modname:
                    modname = self.env.ref_context.get('py:module')
                # ... else, it stays None, which means invalid
        return modname, parents + [base]
```
### 85 - sphinx/ext/autodoc/__init__.py:

Start line: 992, End line: 1057

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
        if self.env.config.autodoc_typehints == 'none':
            kwargs.setdefault('show_annotation', False)

        if inspect.isbuiltin(self.object) or inspect.ismethoddescriptor(self.object):
            # cannot introspect arguments of a C function or method
            return None
        try:
            if (not inspect.isfunction(self.object) and
                    not inspect.ismethod(self.object) and
                    not inspect.isbuiltin(self.object) and
                    not inspect.isclass(self.object) and
                    hasattr(self.object, '__call__')):
                self.env.app.emit('autodoc-before-process-signature',
                                  self.object.__call__, False)
                sig = inspect.signature(self.object.__call__)
            else:
                self.env.app.emit('autodoc-before-process-signature', self.object, False)
                sig = inspect.signature(self.object)
            args = stringify_signature(sig, **kwargs)
        except TypeError:
            if (inspect.is_builtin_class_method(self.object, '__new__') and
               inspect.is_builtin_class_method(self.object, '__init__')):
                raise TypeError('%r is a builtin class' % self.object)

            # if a class should be documented as function (yay duck
            # typing) we try to use the constructor signature as function
            # signature without the first argument.
            try:
                self.env.app.emit('autodoc-before-process-signature',
                                  self.object.__new__, True)
                sig = inspect.signature(self.object.__new__, bound_method=True)
                args = stringify_signature(sig, show_return_annotation=False, **kwargs)
            except TypeError:
                self.env.app.emit('autodoc-before-process-signature',
                                  self.object.__init__, True)
                sig = inspect.signature(self.object.__init__, bound_method=True)
                args = stringify_signature(sig, show_return_annotation=False, **kwargs)

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
```
### 92 - sphinx/ext/autodoc/__init__.py:

Start line: 1664, End line: 1680

```python
class SlotsAttributeDocumenter(AttributeDocumenter):
    """
    Specialized Documenter subclass for attributes that cannot be imported
    because they are attributes in __slots__.
    """
    objtype = 'slotsattribute'
    directivetype = 'attribute'
    member_order = 60

    # must be higher than AttributeDocumenter
    priority = 11

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        """This documents only SLOTSATTR members."""
        return member is SLOTSATTR
```
### 98 - sphinx/ext/autodoc/__init__.py:

Start line: 1060, End line: 1075

```python
class SingledispatchFunctionDocumenter(FunctionDocumenter):
    """
    Specialized Documenter subclass for singledispatch'ed functions.
    """
    objtype = 'singledispatch_function'
    directivetype = 'function'
    member_order = 30

    # before FunctionDocumenter
    priority = FunctionDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        return (super().can_document_member(member, membername, isattr, parent) and
                inspect.is_singledispatch_function(member))
```
### 113 - sphinx/ext/autodoc/__init__.py:

Start line: 881, End line: 911

```python
class ClassLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on class level (methods,
    attributes).
    """
    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is None:
            if path:
                mod_cls = path.rstrip('.')
            else:
                mod_cls = None
                # if documenting a class-level object without path,
                # there must be a current class, either from a parent
                # auto directive ...
                mod_cls = self.env.temp_data.get('autodoc:class')
                # ... or from a class directive
                if mod_cls is None:
                    mod_cls = self.env.ref_context.get('py:class')
                # ... if still None, there's no way to know
                if mod_cls is None:
                    return None, []
            modname, cls = rpartition(mod_cls, '.')
            parents = [cls]
            # if the module name is still missing, get it like above
            if not modname:
                modname = self.env.temp_data.get('autodoc:module')
            if not modname:
                modname = self.env.ref_context.get('py:module')
            # ... else, it stays None, which means invalid
        return modname, parents + [base]
```
### 117 - sphinx/ext/autodoc/__init__.py:

Start line: 768, End line: 809

```python
class ModuleDocumenter(Documenter):
    """
    Specialized Documenter subclass for modules.
    """
    objtype = 'module'
    content_indent = ''
    titles_allowed = True

    option_spec = {
        'members': members_option, 'undoc-members': bool_option,
        'noindex': bool_option, 'inherited-members': inherited_members_option,
        'show-inheritance': bool_option, 'synopsis': identity,
        'platform': identity, 'deprecated': bool_option,
        'member-order': identity, 'exclude-members': members_set_option,
        'private-members': bool_option, 'special-members': members_option,
        'imported-members': bool_option, 'ignore-module-all': bool_option
    }  # type: Dict[str, Callable]

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        merge_special_members_option(self.options)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        # don't document submodules automatically
        return False

    def resolve_name(self, modname: str, parents: Any, path: str, base: Any
                     ) -> Tuple[str, List[str]]:
        if modname is not None:
            logger.warning(__('"::" in automodule name doesn\'t make sense'),
                           type='autodoc')
        return (path or '') + base, []

    def parse_name(self) -> bool:
        ret = super().parse_name()
        if self.args or self.retann:
            logger.warning(__('signature arguments or return annotation '
                              'given for automodule %s') % self.fullname,
                           type='autodoc')
        return ret
```
