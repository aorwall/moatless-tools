# sphinx-doc__sphinx-8801

| **sphinx-doc/sphinx** | `7ca279e33aebb60168d35e6be4ed059f4a68f2c1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 12308 |
| **Any found context length** | 12308 |
| **Avg pos** | 33.0 |
| **Min pos** | 33 |
| **Max pos** | 33 |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/ext/autodoc/importer.py b/sphinx/ext/autodoc/importer.py
--- a/sphinx/ext/autodoc/importer.py
+++ b/sphinx/ext/autodoc/importer.py
@@ -294,24 +294,35 @@ def get_class_members(subject: Any, objpath: List[str], attrgetter: Callable
 
     try:
         for cls in getmro(subject):
+            try:
+                modname = safe_getattr(cls, '__module__')
+                qualname = safe_getattr(cls, '__qualname__')
+                analyzer = ModuleAnalyzer.for_module(modname)
+                analyzer.analyze()
+            except AttributeError:
+                qualname = None
+                analyzer = None
+            except PycodeError:
+                analyzer = None
+
             # annotation only member (ex. attr: int)
             for name in getannotations(cls):
                 name = unmangle(cls, name)
                 if name and name not in members:
-                    members[name] = ObjectMember(name, INSTANCEATTR, class_=cls)
+                    if analyzer and (qualname, name) in analyzer.attr_docs:
+                        docstring = '\n'.join(analyzer.attr_docs[qualname, name])
+                    else:
+                        docstring = None
+
+                    members[name] = ObjectMember(name, INSTANCEATTR, class_=cls,
+                                                 docstring=docstring)
 
             # append instance attributes (cf. self.attr1) if analyzer knows
-            try:
-                modname = safe_getattr(cls, '__module__')
-                qualname = safe_getattr(cls, '__qualname__')
-                analyzer = ModuleAnalyzer.for_module(modname)
-                analyzer.analyze()
+            if analyzer:
                 for (ns, name), docstring in analyzer.attr_docs.items():
                     if ns == qualname and name not in members:
                         members[name] = ObjectMember(name, INSTANCEATTR, class_=cls,
                                                      docstring='\n'.join(docstring))
-            except (AttributeError, PycodeError):
-                pass
     except AttributeError:
         pass
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/autodoc/importer.py | 297 | 310 | 33 | 3 | 12308


## Problem Statement

```
autodoc: The annotation only member in superclass is treated as "undocumented"
**Describe the bug**
autodoc: The annotation only member in superclass is treated as "undocumented".

**To Reproduce**

\`\`\`
# example.py
class Foo:
    """docstring"""
    attr1: int  #: docstring


class Bar(Foo):
    """docstring"""
    attr2: str  #: docstring
\`\`\`
\`\`\`
# index.rst
.. autoclass:: example.Bar
   :members:
   :inherited-members:
\`\`\`

`Bar.attr1` is not documented. It will be shown if I give `:undoc-members:` option to the autoclass directive call. It seems the attribute is treated as undocumented.

**Expected behavior**
It should be shown.

**Your project**
No

**Screenshots**
No

**Environment info**
- OS: Mac
- Python version: 3.9.1
- Sphinx version: HEAD of 3.x
- Sphinx extensions: sphinx.ext.autodoc
- Extra tools: No

**Additional context**
No


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/ext/autodoc/__init__.py | 1433 | 1728| 2767 | 2767 | 22486 | 
| 2 | 1 sphinx/ext/autodoc/__init__.py | 705 | 812| 865 | 3632 | 22486 | 
| 3 | 1 sphinx/ext/autodoc/__init__.py | 1071 | 1095| 209 | 3841 | 22486 | 
| 4 | 1 sphinx/ext/autodoc/__init__.py | 2498 | 2513| 182 | 4023 | 22486 | 
| 5 | 1 sphinx/ext/autodoc/__init__.py | 2450 | 2473| 220 | 4243 | 22486 | 
| 6 | 1 sphinx/ext/autodoc/__init__.py | 2371 | 2405| 289 | 4532 | 22486 | 
| 7 | 1 sphinx/ext/autodoc/__init__.py | 2515 | 2540| 324 | 4856 | 22486 | 
| 8 | 1 sphinx/ext/autodoc/__init__.py | 1961 | 1998| 320 | 5176 | 22486 | 
| 9 | 1 sphinx/ext/autodoc/__init__.py | 2316 | 2333| 134 | 5310 | 22486 | 
| 10 | 2 sphinx/ext/autodoc/deprecated.py | 78 | 93| 139 | 5449 | 23446 | 
| 11 | 2 sphinx/ext/autodoc/deprecated.py | 32 | 47| 140 | 5589 | 23446 | 
| 12 | 2 sphinx/ext/autodoc/__init__.py | 2475 | 2496| 272 | 5861 | 23446 | 
| 13 | 2 sphinx/ext/autodoc/__init__.py | 13 | 115| 794 | 6655 | 23446 | 
| 14 | 2 sphinx/ext/autodoc/__init__.py | 1937 | 1959| 245 | 6900 | 23446 | 
| 15 | 2 sphinx/ext/autodoc/__init__.py | 289 | 363| 784 | 7684 | 23446 | 
| 16 | 2 sphinx/ext/autodoc/__init__.py | 2407 | 2426| 224 | 7908 | 23446 | 
| 17 | **3 sphinx/ext/autodoc/importer.py** | 167 | 181| 125 | 8033 | 26096 | 
| 18 | 3 sphinx/ext/autodoc/__init__.py | 1888 | 1935| 359 | 8392 | 26096 | 
| 19 | 3 sphinx/ext/autodoc/deprecated.py | 96 | 111| 133 | 8525 | 26096 | 
| 20 | 3 sphinx/ext/autodoc/__init__.py | 2335 | 2368| 297 | 8822 | 26096 | 
| 21 | 3 sphinx/ext/autodoc/__init__.py | 1847 | 1885| 315 | 9137 | 26096 | 
| 22 | 3 sphinx/ext/autodoc/__init__.py | 1017 | 1029| 124 | 9261 | 26096 | 
| 23 | 3 sphinx/ext/autodoc/__init__.py | 2622 | 2660| 445 | 9706 | 26096 | 
| 24 | 3 sphinx/ext/autodoc/__init__.py | 2592 | 2622| 365 | 10071 | 26096 | 
| 25 | 3 sphinx/ext/autodoc/__init__.py | 579 | 591| 133 | 10204 | 26096 | 
| 26 | 3 sphinx/ext/autodoc/__init__.py | 118 | 151| 216 | 10420 | 26096 | 
| 27 | 3 sphinx/ext/autodoc/__init__.py | 2185 | 2213| 256 | 10676 | 26096 | 
| 28 | 3 sphinx/ext/autodoc/__init__.py | 2428 | 2448| 227 | 10903 | 26096 | 
| 29 | 3 sphinx/ext/autodoc/__init__.py | 2246 | 2260| 160 | 11063 | 26096 | 
| 30 | 4 sphinx/ext/autodoc/mock.py | 72 | 96| 218 | 11281 | 27452 | 
| 31 | 4 sphinx/ext/autodoc/__init__.py | 1731 | 1766| 262 | 11543 | 27452 | 
| 32 | 4 sphinx/ext/autodoc/__init__.py | 1097 | 1114| 177 | 11720 | 27452 | 
| **-> 33 <-** | **4 sphinx/ext/autodoc/importer.py** | 245 | 318| 588 | 12308 | 27452 | 
| 34 | 4 sphinx/ext/autodoc/__init__.py | 1138 | 1168| 287 | 12595 | 27452 | 
| 35 | 4 sphinx/ext/autodoc/__init__.py | 249 | 286| 311 | 12906 | 27452 | 
| 36 | 5 doc/usage/extensions/example_numpy.py | 320 | 334| 109 | 13015 | 29560 | 
| 37 | 5 sphinx/ext/autodoc/__init__.py | 2019 | 2182| 1448 | 14463 | 29560 | 
| 38 | 5 sphinx/ext/autodoc/__init__.py | 1286 | 1399| 958 | 15421 | 29560 | 
| 39 | 5 doc/usage/extensions/example_numpy.py | 336 | 356| 120 | 15541 | 29560 | 
| 40 | 6 doc/usage/extensions/example_google.py | 277 | 296| 120 | 15661 | 31670 | 
| 41 | 6 sphinx/ext/autodoc/__init__.py | 1245 | 1252| 114 | 15775 | 31670 | 
| 42 | 6 sphinx/ext/autodoc/__init__.py | 2263 | 2283| 155 | 15930 | 31670 | 
| 43 | 6 doc/usage/extensions/example_google.py | 261 | 275| 109 | 16039 | 31670 | 
| 44 | 6 sphinx/ext/autodoc/__init__.py | 2543 | 2571| 247 | 16286 | 31670 | 
| 45 | 7 sphinx/domains/python.py | 729 | 783| 530 | 16816 | 43613 | 
| 46 | 7 sphinx/ext/autodoc/__init__.py | 1402 | 1728| 189 | 17005 | 43613 | 
| 47 | 7 sphinx/ext/autodoc/__init__.py | 1031 | 1042| 126 | 17131 | 43613 | 
| 48 | 7 sphinx/ext/autodoc/deprecated.py | 114 | 127| 114 | 17245 | 43613 | 
| 49 | 7 sphinx/ext/autodoc/__init__.py | 2216 | 2244| 209 | 17454 | 43613 | 
| 50 | 7 sphinx/ext/autodoc/__init__.py | 2574 | 2589| 126 | 17580 | 43613 | 
| 51 | 7 sphinx/ext/autodoc/__init__.py | 1117 | 1135| 179 | 17759 | 43613 | 
| 52 | 8 sphinx/ext/autosummary/generate.py | 229 | 244| 184 | 17943 | 48938 | 
| 53 | 8 sphinx/ext/autodoc/__init__.py | 1831 | 1844| 136 | 18079 | 48938 | 
| 54 | 8 sphinx/ext/autosummary/generate.py | 273 | 286| 202 | 18281 | 48938 | 
| 55 | 8 sphinx/ext/autodoc/__init__.py | 814 | 857| 453 | 18734 | 48938 | 
| 56 | 8 sphinx/ext/autodoc/mock.py | 11 | 69| 454 | 19188 | 48938 | 
| 57 | 8 sphinx/ext/autodoc/__init__.py | 1044 | 1069| 210 | 19398 | 48938 | 
| 58 | 8 sphinx/ext/autodoc/deprecated.py | 50 | 62| 113 | 19511 | 48938 | 
| 59 | 9 sphinx/directives/__init__.py | 52 | 74| 209 | 19720 | 51688 | 
| 60 | 10 sphinx/ext/autodoc/directive.py | 115 | 165| 453 | 20173 | 53001 | 
| 61 | 10 sphinx/ext/autodoc/__init__.py | 2001 | 2016| 120 | 20293 | 53001 | 
| 62 | 10 sphinx/ext/autodoc/__init__.py | 1769 | 1785| 135 | 20428 | 53001 | 
| 63 | 11 sphinx/application.py | 1124 | 1145| 265 | 20693 | 64868 | 
| 64 | 11 sphinx/ext/autodoc/directive.py | 9 | 49| 305 | 20998 | 64868 | 
| 65 | 11 sphinx/ext/autodoc/__init__.py | 973 | 1015| 388 | 21386 | 64868 | 
| 66 | 11 sphinx/domains/python.py | 701 | 727| 190 | 21576 | 64868 | 
| 67 | **11 sphinx/ext/autodoc/importer.py** | 78 | 141| 597 | 22173 | 64868 | 
| 68 | 11 sphinx/ext/autodoc/__init__.py | 2285 | 2313| 249 | 22422 | 64868 | 
| 69 | 12 sphinx/domains/c.py | 2577 | 3409| 6794 | 29216 | 96181 | 
| 70 | 12 sphinx/domains/python.py | 910 | 922| 132 | 29348 | 96181 | 
| 71 | 12 sphinx/ext/autodoc/__init__.py | 1268 | 1283| 166 | 29514 | 96181 | 
| 72 | 12 sphinx/domains/python.py | 1011 | 1029| 143 | 29657 | 96181 | 
| 73 | 12 sphinx/domains/python.py | 887 | 908| 158 | 29815 | 96181 | 
| 74 | 12 sphinx/ext/autosummary/generate.py | 246 | 271| 298 | 30113 | 96181 | 
| 75 | 12 sphinx/ext/autodoc/__init__.py | 376 | 411| 315 | 30428 | 96181 | 
| 76 | **12 sphinx/ext/autodoc/importer.py** | 184 | 242| 459 | 30887 | 96181 | 
| 77 | 12 sphinx/ext/autodoc/__init__.py | 636 | 662| 306 | 31193 | 96181 | 
| 78 | 12 sphinx/ext/autodoc/__init__.py | 664 | 704| 327 | 31520 | 96181 | 
| 79 | 12 sphinx/application.py | 1147 | 1160| 174 | 31694 | 96181 | 
| 80 | 12 sphinx/ext/autodoc/__init__.py | 436 | 484| 359 | 32053 | 96181 | 
| 81 | 12 sphinx/ext/autodoc/__init__.py | 1807 | 1829| 203 | 32256 | 96181 | 
| 82 | 12 sphinx/ext/autodoc/__init__.py | 1254 | 1265| 134 | 32390 | 96181 | 
| 83 | 12 sphinx/ext/autosummary/generate.py | 176 | 195| 176 | 32566 | 96181 | 
| 84 | **12 sphinx/ext/autodoc/importer.py** | 321 | 340| 180 | 32746 | 96181 | 
| 85 | 12 sphinx/ext/autodoc/__init__.py | 884 | 970| 804 | 33550 | 96181 | 
| 86 | 13 sphinx/ext/napoleon/__init__.py | 403 | 486| 740 | 34290 | 100214 | 
| 87 | 13 sphinx/ext/autodoc/__init__.py | 1788 | 1804| 137 | 34427 | 100214 | 
| 88 | 14 sphinx/util/cfamily.py | 179 | 230| 341 | 34768 | 103716 | 
| 89 | 14 sphinx/ext/autodoc/deprecated.py | 11 | 29| 155 | 34923 | 103716 | 
| 90 | 14 sphinx/ext/autosummary/generate.py | 477 | 498| 217 | 35140 | 103716 | 
| 91 | 14 doc/usage/extensions/example_numpy.py | 225 | 318| 589 | 35729 | 103716 | 
| 92 | 14 sphinx/ext/autodoc/__init__.py | 564 | 577| 131 | 35860 | 103716 | 
| 93 | 14 sphinx/ext/autodoc/deprecated.py | 65 | 75| 108 | 35968 | 103716 | 
| 94 | 14 doc/usage/extensions/example_google.py | 180 | 259| 563 | 36531 | 103716 | 
| 95 | **14 sphinx/ext/autodoc/importer.py** | 11 | 41| 225 | 36756 | 103716 | 
| 96 | 15 sphinx/ext/napoleon/docstring.py | 858 | 871| 153 | 36909 | 114711 | 
| 97 | 15 doc/usage/extensions/example_google.py | 298 | 314| 125 | 37034 | 114711 | 
| 98 | 15 sphinx/ext/autodoc/__init__.py | 859 | 882| 246 | 37280 | 114711 | 
| 99 | 15 sphinx/ext/autodoc/__init__.py | 593 | 634| 409 | 37689 | 114711 | 
| 100 | 15 sphinx/domains/python.py | 11 | 78| 526 | 38215 | 114711 | 
| 101 | 15 sphinx/ext/autosummary/generate.py | 88 | 100| 166 | 38381 | 114711 | 
| 102 | 16 sphinx/directives/other.py | 186 | 207| 141 | 38522 | 117896 | 
| 103 | 16 sphinx/ext/autosummary/generate.py | 20 | 58| 288 | 38810 | 117896 | 
| 104 | 17 sphinx/ext/autodoc/typehints.py | 82 | 138| 460 | 39270 | 118949 | 
| 105 | 18 sphinx/util/inspect.py | 892 | 919| 228 | 39498 | 125961 | 
| 106 | 18 sphinx/directives/__init__.py | 76 | 88| 132 | 39630 | 125961 | 
| 107 | **18 sphinx/ext/autodoc/importer.py** | 144 | 164| 164 | 39794 | 125961 | 
| 108 | 18 sphinx/util/cfamily.py | 126 | 156| 209 | 40003 | 125961 | 
| 109 | 19 sphinx/util/__init__.py | 480 | 493| 123 | 40126 | 132263 | 
| 110 | 19 sphinx/directives/__init__.py | 269 | 323| 517 | 40643 | 132263 | 
| 111 | 19 sphinx/domains/python.py | 337 | 382| 381 | 41024 | 132263 | 
| 112 | 19 sphinx/ext/autodoc/mock.py | 155 | 188| 227 | 41251 | 132263 | 
| 113 | 19 sphinx/ext/autosummary/generate.py | 303 | 351| 519 | 41770 | 132263 | 
| 114 | 19 sphinx/domains/c.py | 3412 | 3839| 3824 | 45594 | 132263 | 
| 115 | 20 sphinx/domains/cpp.py | 2402 | 2459| 494 | 46088 | 195420 | 
| 116 | 20 sphinx/domains/python.py | 553 | 572| 147 | 46235 | 195420 | 
| 117 | 20 sphinx/ext/autodoc/__init__.py | 413 | 434| 199 | 46434 | 195420 | 
| 118 | 21 sphinx/pycode/ast.py | 138 | 172| 410 | 46844 | 197414 | 
| 119 | 21 sphinx/ext/autodoc/__init__.py | 1171 | 1243| 541 | 47385 | 197414 | 
| 120 | 22 sphinx/ext/autodoc/type_comment.py | 11 | 35| 239 | 47624 | 198638 | 
| 121 | 22 sphinx/ext/autosummary/generate.py | 288 | 301| 201 | 47825 | 198638 | 
| 122 | 22 sphinx/ext/autodoc/__init__.py | 365 | 374| 121 | 47946 | 198638 | 
| 123 | 22 sphinx/domains/python.py | 786 | 823| 228 | 48174 | 198638 | 


## Patch

```diff
diff --git a/sphinx/ext/autodoc/importer.py b/sphinx/ext/autodoc/importer.py
--- a/sphinx/ext/autodoc/importer.py
+++ b/sphinx/ext/autodoc/importer.py
@@ -294,24 +294,35 @@ def get_class_members(subject: Any, objpath: List[str], attrgetter: Callable
 
     try:
         for cls in getmro(subject):
+            try:
+                modname = safe_getattr(cls, '__module__')
+                qualname = safe_getattr(cls, '__qualname__')
+                analyzer = ModuleAnalyzer.for_module(modname)
+                analyzer.analyze()
+            except AttributeError:
+                qualname = None
+                analyzer = None
+            except PycodeError:
+                analyzer = None
+
             # annotation only member (ex. attr: int)
             for name in getannotations(cls):
                 name = unmangle(cls, name)
                 if name and name not in members:
-                    members[name] = ObjectMember(name, INSTANCEATTR, class_=cls)
+                    if analyzer and (qualname, name) in analyzer.attr_docs:
+                        docstring = '\n'.join(analyzer.attr_docs[qualname, name])
+                    else:
+                        docstring = None
+
+                    members[name] = ObjectMember(name, INSTANCEATTR, class_=cls,
+                                                 docstring=docstring)
 
             # append instance attributes (cf. self.attr1) if analyzer knows
-            try:
-                modname = safe_getattr(cls, '__module__')
-                qualname = safe_getattr(cls, '__qualname__')
-                analyzer = ModuleAnalyzer.for_module(modname)
-                analyzer.analyze()
+            if analyzer:
                 for (ns, name), docstring in analyzer.attr_docs.items():
                     if ns == qualname and name not in members:
                         members[name] = ObjectMember(name, INSTANCEATTR, class_=cls,
                                                      docstring='\n'.join(docstring))
-            except (AttributeError, PycodeError):
-                pass
     except AttributeError:
         pass
 

```

## Test Patch

```diff
diff --git a/tests/roots/test-ext-autodoc/target/uninitialized_attributes.py b/tests/roots/test-ext-autodoc/target/uninitialized_attributes.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-ext-autodoc/target/uninitialized_attributes.py
@@ -0,0 +1,8 @@
+class Base:
+    attr1: int  #: docstring
+    attr2: str
+
+
+class Derived(Base):
+    attr3: int  #: docstring
+    attr4: str
diff --git a/tests/test_ext_autodoc_autoclass.py b/tests/test_ext_autodoc_autoclass.py
--- a/tests/test_ext_autodoc_autoclass.py
+++ b/tests/test_ext_autodoc_autoclass.py
@@ -106,6 +106,73 @@ def test_inherited_instance_variable(app):
     ]
 
 
+@pytest.mark.skipif(sys.version_info < (3, 6), reason='py36+ is available since python3.6.')
+@pytest.mark.sphinx('html', testroot='ext-autodoc')
+def test_uninitialized_attributes(app):
+    options = {"members": None,
+               "inherited-members": True}
+    actual = do_autodoc(app, 'class', 'target.uninitialized_attributes.Derived', options)
+    assert list(actual) == [
+        '',
+        '.. py:class:: Derived()',
+        '   :module: target.uninitialized_attributes',
+        '',
+        '',
+        '   .. py:attribute:: Derived.attr1',
+        '      :module: target.uninitialized_attributes',
+        '      :type: int',
+        '',
+        '      docstring',
+        '',
+        '',
+        '   .. py:attribute:: Derived.attr3',
+        '      :module: target.uninitialized_attributes',
+        '      :type: int',
+        '',
+        '      docstring',
+        '',
+    ]
+
+
+@pytest.mark.skipif(sys.version_info < (3, 6), reason='py36+ is available since python3.6.')
+@pytest.mark.sphinx('html', testroot='ext-autodoc')
+def test_undocumented_uninitialized_attributes(app):
+    options = {"members": None,
+               "inherited-members": True,
+               "undoc-members": True}
+    actual = do_autodoc(app, 'class', 'target.uninitialized_attributes.Derived', options)
+    assert list(actual) == [
+        '',
+        '.. py:class:: Derived()',
+        '   :module: target.uninitialized_attributes',
+        '',
+        '',
+        '   .. py:attribute:: Derived.attr1',
+        '      :module: target.uninitialized_attributes',
+        '      :type: int',
+        '',
+        '      docstring',
+        '',
+        '',
+        '   .. py:attribute:: Derived.attr2',
+        '      :module: target.uninitialized_attributes',
+        '      :type: str',
+        '',
+        '',
+        '   .. py:attribute:: Derived.attr3',
+        '      :module: target.uninitialized_attributes',
+        '      :type: int',
+        '',
+        '      docstring',
+        '',
+        '',
+        '   .. py:attribute:: Derived.attr4',
+        '      :module: target.uninitialized_attributes',
+        '      :type: str',
+        '',
+    ]
+
+
 def test_decorators(app):
     actual = do_autodoc(app, 'class', 'target.decorator.Baz')
     assert list(actual) == [

```


## Code snippets

### 1 - sphinx/ext/autodoc/__init__.py:

Start line: 1433, End line: 1728

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
        'show-inheritance': bool_option, 'member-order': member_order_option,
        'exclude-members': exclude_members_option,
        'private-members': members_option, 'special-members': members_option,
    }  # type: Dict[str, Callable]

    _signature_class = None  # type: Any
    _signature_method_name = None  # type: str

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
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

        sig = super().format_signature()
        sigs = []

        overloads = self.get_overloaded_signatures()
        if overloads and self.config.autodoc_typehints == 'signature':
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

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        super().add_directive_header(sig)

        if self.analyzer and '.'.join(self.objpath) in self.analyzer.finals:
            self.add_line('   :final:', sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            sourcename = self.get_sourcename()
            self.add_line('', sourcename)

            if hasattr(self.object, '__orig_bases__') and len(self.object.__orig_bases__):
                # A subclass of generic types
                # refs: PEP-560 <https://www.python.org/dev/peps/pep-0560/>
                bases = [restify(cls) for cls in self.object.__orig_bases__]
                self.add_line('   ' + _('Bases: %s') % ', '.join(bases), sourcename)
            elif hasattr(self.object, '__bases__') and len(self.object.__bases__):
                # A normal class
                bases = [restify(cls) for cls in self.object.__bases__]
                self.add_line('   ' + _('Bases: %s') % ', '.join(bases), sourcename)

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

    def get_doc(self, encoding: str = None, ignore: int = None) -> Optional[List[List[str]]]:
        if encoding is not None:
            warnings.warn("The 'encoding' argument to autodoc.%s.get_doc() is deprecated."
                          % self.__class__.__name__,
                          RemovedInSphinx40Warning, stacklevel=2)
        if self.doc_as_attr:
            # Don't show the docstring of the class when it is an alias.
            return None

        lines = getattr(self, '_new_docstrings', None)
        if lines is not None:
            return lines

        content = self.config.autoclass_content

        docstrings = []
        attrdocstring = self.get_attr(self.object, '__doc__', None)
        if attrdocstring:
            docstrings.append(attrdocstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if content in ('both', 'init'):
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
                if content == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(docstring, ignore, tab_width) for docstring in docstrings]

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        if self.doc_as_attr:
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
### 2 - sphinx/ext/autodoc/__init__.py:

Start line: 705, End line: 812

```python
class Documenter:

    def filter_members(self, members: ObjectMembers, want_all: bool
                       ) -> List[Tuple[str, Any, bool]]:
        # ... other code
        for obj in members:
            membername, member = obj
            # if isattr is True, the member is documented as an attribute
            if member is INSTANCEATTR:
                isattr = True
            else:
                isattr = False

            doc = getdoc(member, self.get_attr, self.config.autodoc_inherit_docstrings,
                         self.parent, self.object_name)
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

            has_doc = bool(doc)

            metadata = extract_metadata(doc)
            if 'private' in metadata:
                # consider a member private if docstring has "private" metadata
                isprivate = True
            elif 'public' in metadata:
                # consider a member public if docstring has "public" metadata
                isprivate = False
            else:
                isprivate = membername.startswith('_')

            keep = False
            if ismock(member):
                # mocked module or object
                pass
            elif self.options.exclude_members and membername in self.options.exclude_members:
                # remove members given by exclude-members
                keep = False
            elif want_all and special_member_re.match(membername):
                # special __methods__
                if self.options.special_members and membername in self.options.special_members:
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
                isattr = True
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
                try:
                    skip_user = self.env.app.emit_firstresult(
                        'autodoc-skip-member', self.objtype, membername, member,
                        not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
                except Exception as exc:
                    logger.warning(__('autodoc: failed to determine %r to be documented, '
                                      'the following exception was raised:\n%s'),
                                   member, exc, type='autodoc')
                    keep = False

            if keep:
                ret.append((membername, member, isattr))

        return ret
```
### 3 - sphinx/ext/autodoc/__init__.py:

Start line: 1071, End line: 1095

```python
class ModuleDocumenter(Documenter):

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        members = self.get_module_members()
        if want_all:
            if self.__all__ is None:
                # for implicit module members, check __module__ to avoid
                # documenting imported objects
                return True, list(members.values())
            else:
                for member in members.values():
                    if member.__name__ not in self.__all__:
                        member.skipped = True

                return False, list(members.values())
        else:
            memberlist = self.options.members or []
            ret = []
            for name in memberlist:
                if name in members:
                    ret.append(members[name])
                else:
                    logger.warning(__('missing attribute mentioned in :members: option: '
                                      'module %s, attribute %s') %
                                   (safe_getattr(self.object, '__name__', '???'), name),
                                   type='autodoc')
            return False, ret
```
### 4 - sphinx/ext/autodoc/__init__.py:

Start line: 2498, End line: 2513

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_attribute_comment(self, parent: Any, attrname: str) -> Optional[List[str]]:
        for cls in inspect.getmro(parent):
            try:
                module = safe_getattr(cls, '__module__')
                qualname = safe_getattr(cls, '__qualname__')

                analyzer = ModuleAnalyzer.for_module(module)
                analyzer.analyze()
                if qualname and self.objpath:
                    key = (qualname, attrname)
                    if key in analyzer.attr_docs:
                        return list(analyzer.attr_docs[key])
            except (AttributeError, PycodeError):
                pass

        return None
```
### 5 - sphinx/ext/autodoc/__init__.py:

Start line: 2450, End line: 2473

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
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            if doc:
                metadata = extract_metadata('\n'.join(sum(doc, [])))
                if 'hide-value' in metadata:
                    return True

        return False
```
### 6 - sphinx/ext/autodoc/__init__.py:

Start line: 2371, End line: 2405

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):
    """
    Specialized Documenter subclass for attributes.
    """
    objtype = 'attribute'
    member_order = 60
    option_spec = dict(ModuleLevelDocumenter.option_spec)
    option_spec["annotation"] = annotation_option
    option_spec["no-value"] = bool_option

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
```
### 7 - sphinx/ext/autodoc/__init__.py:

Start line: 2515, End line: 2540

```python
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin,  # type: ignore
                          TypeVarMixin, RuntimeInstanceAttributeMixin,
                          UninitializedInstanceAttributeMixin, NonDataDescriptorMixin,
                          DocstringStripSignatureMixin, ClassLevelDocumenter):

    def get_doc(self, encoding: str = None, ignore: int = None) -> Optional[List[List[str]]]:
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
            return super().get_doc(encoding, ignore)
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
### 8 - sphinx/ext/autodoc/__init__.py:

Start line: 1961, End line: 1998

```python
class DataDocumenter(GenericAliasMixin, NewTypeMixin, TypeVarMixin,
                     UninitializedGlobalVariableMixin, ModuleLevelDocumenter):

    def document_members(self, all_members: bool = False) -> None:
        pass

    def get_real_modname(self) -> str:
        return self.get_attr(self.parent or self.object, '__module__', None) \
            or self.modname

    def get_module_comment(self, attrname: str) -> Optional[List[str]]:
        try:
            analyzer = ModuleAnalyzer.for_module(self.modname)
            analyzer.analyze()
            key = ('', attrname)
            if key in analyzer.attr_docs:
                return list(analyzer.attr_docs[key])
        except PycodeError:
            pass

        return None

    def get_doc(self, encoding: str = None, ignore: int = None) -> List[List[str]]:
        # Check the variable has a docstring-comment
        comment = self.get_module_comment(self.objpath[-1])
        if comment:
            return [comment]
        else:
            return super().get_doc(encoding, ignore)

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False
                    ) -> None:
        # Disable analyzing variable comment on Documenter.add_content() to control it on
        # DataDocumenter.add_content()
        self.analyzer = None

        if not more_content:
            more_content = StringList()

        self.update_content(more_content)
        super().add_content(more_content, no_docstring=no_docstring)
```
### 9 - sphinx/ext/autodoc/__init__.py:

Start line: 2316, End line: 2333

```python
class UninitializedInstanceAttributeMixin(DataDocumenterMixinBase):
    """
    Mixin for AttributeDocumenter to provide the feature for supporting uninitialized
    instance attributes (PEP-526 styled, annotation only attributes).

    Example:

        class Foo:
            attr: int  #: This is a target of this mix-in.
    """

    def is_uninitialized_instance_attribute(self, parent: Any) -> bool:
        """Check the subject is an annotation only attribute."""
        annotations = get_type_hints(parent, None, self.config.autodoc_type_aliases)
        if self.objpath[-1] in annotations:
            return True
        else:
            return False
```
### 10 - sphinx/ext/autodoc/deprecated.py:

Start line: 78, End line: 93

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("%s is deprecated." % self.__class__.__name__,
                      RemovedInSphinx50Warning, stacklevel=2)
        super().__init__(*args, **kwargs)
```
### 17 - sphinx/ext/autodoc/importer.py:

Start line: 167, End line: 181

```python
Attribute = NamedTuple('Attribute', [('name', str),
                                     ('directly_defined', bool),
                                     ('value', Any)])


def _getmro(obj: Any) -> Tuple["Type", ...]:
    warnings.warn('sphinx.ext.autodoc.importer._getmro() is deprecated.',
                  RemovedInSphinx40Warning)
    return getmro(obj)


def _getannotations(obj: Any) -> Mapping[str, Any]:
    warnings.warn('sphinx.ext.autodoc.importer._getannotations() is deprecated.',
                  RemovedInSphinx40Warning)
    return getannotations(obj)
```
### 33 - sphinx/ext/autodoc/importer.py:

Start line: 245, End line: 318

```python
def get_class_members(subject: Any, objpath: List[str], attrgetter: Callable
                      ) -> Dict[str, "ObjectMember"]:
    """Get members and attributes of target class."""
    from sphinx.ext.autodoc import INSTANCEATTR, ObjectMember

    # the members directly defined in the class
    obj_dict = attrgetter(subject, '__dict__', {})

    members = {}  # type: Dict[str, ObjectMember]

    # enum members
    if isenumclass(subject):
        for name, value in subject.__members__.items():
            if name not in members:
                members[name] = ObjectMember(name, value, class_=subject)

        superclass = subject.__mro__[1]
        for name in obj_dict:
            if name not in superclass.__dict__:
                value = safe_getattr(subject, name)
                members[name] = ObjectMember(name, value, class_=subject)

    # members in __slots__
    try:
        __slots__ = getslots(subject)
        if __slots__:
            from sphinx.ext.autodoc import SLOTSATTR

            for name, docstring in __slots__.items():
                members[name] = ObjectMember(name, SLOTSATTR, class_=subject,
                                             docstring=docstring)
    except (TypeError, ValueError):
        pass

    # other members
    for name in dir(subject):
        try:
            value = attrgetter(subject, name)
            if ismock(value):
                value = undecorate(value)

            unmangled = unmangle(subject, name)
            if unmangled and unmangled not in members:
                if name in obj_dict:
                    members[unmangled] = ObjectMember(unmangled, value, class_=subject)
                else:
                    members[unmangled] = ObjectMember(unmangled, value)
        except AttributeError:
            continue

    try:
        for cls in getmro(subject):
            # annotation only member (ex. attr: int)
            for name in getannotations(cls):
                name = unmangle(cls, name)
                if name and name not in members:
                    members[name] = ObjectMember(name, INSTANCEATTR, class_=cls)

            # append instance attributes (cf. self.attr1) if analyzer knows
            try:
                modname = safe_getattr(cls, '__module__')
                qualname = safe_getattr(cls, '__qualname__')
                analyzer = ModuleAnalyzer.for_module(modname)
                analyzer.analyze()
                for (ns, name), docstring in analyzer.attr_docs.items():
                    if ns == qualname and name not in members:
                        members[name] = ObjectMember(name, INSTANCEATTR, class_=cls,
                                                     docstring='\n'.join(docstring))
            except (AttributeError, PycodeError):
                pass
    except AttributeError:
        pass

    return members
```
### 67 - sphinx/ext/autodoc/importer.py:

Start line: 78, End line: 141

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
            mangled_name = mangle(obj, attrname)
            obj = attrgetter(obj, mangled_name)
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
        raise ImportError(errmsg) from exc
```
### 76 - sphinx/ext/autodoc/importer.py:

Start line: 184, End line: 242

```python
def get_object_members(subject: Any, objpath: List[str], attrgetter: Callable,
                       analyzer: ModuleAnalyzer = None) -> Dict[str, Attribute]:
    """Get members and attributes of target object."""
    from sphinx.ext.autodoc import INSTANCEATTR

    # the members directly defined in the class
    obj_dict = attrgetter(subject, '__dict__', {})

    members = {}  # type: Dict[str, Attribute]

    # enum members
    if isenumclass(subject):
        for name, value in subject.__members__.items():
            if name not in members:
                members[name] = Attribute(name, True, value)

        superclass = subject.__mro__[1]
        for name in obj_dict:
            if name not in superclass.__dict__:
                value = safe_getattr(subject, name)
                members[name] = Attribute(name, True, value)

    # members in __slots__
    try:
        __slots__ = getslots(subject)
        if __slots__:
            from sphinx.ext.autodoc import SLOTSATTR

            for name in __slots__:
                members[name] = Attribute(name, True, SLOTSATTR)
    except (TypeError, ValueError):
        pass

    # other members
    for name in dir(subject):
        try:
            value = attrgetter(subject, name)
            directly_defined = name in obj_dict
            name = unmangle(subject, name)
            if name and name not in members:
                members[name] = Attribute(name, directly_defined, value)
        except AttributeError:
            continue

    # annotation only member (ex. attr: int)
    for i, cls in enumerate(getmro(subject)):
        for name in getannotations(cls):
            name = unmangle(cls, name)
            if name and name not in members:
                members[name] = Attribute(name, i == 0, INSTANCEATTR)

    if analyzer:
        # append instance attributes (cf. self.attr1) if analyzer knows
        namespace = '.'.join(objpath)
        for (ns, name) in analyzer.find_attr_docs():
            if namespace == ns and name not in members:
                members[name] = Attribute(name, True, INSTANCEATTR)

    return members
```
### 84 - sphinx/ext/autodoc/importer.py:

Start line: 321, End line: 340

```python
from sphinx.ext.autodoc.mock import (MockFinder, MockLoader, _MockModule, _MockObject,  # NOQA
                                     mock)

deprecated_alias('sphinx.ext.autodoc.importer',
                 {
                     '_MockModule': _MockModule,
                     '_MockObject': _MockObject,
                     'MockFinder': MockFinder,
                     'MockLoader': MockLoader,
                     'mock': mock,
                 },
                 RemovedInSphinx40Warning,
                 {
                     '_MockModule': 'sphinx.ext.autodoc.mock._MockModule',
                     '_MockObject': 'sphinx.ext.autodoc.mock._MockObject',
                     'MockFinder': 'sphinx.ext.autodoc.mock.MockFinder',
                     'MockLoader': 'sphinx.ext.autodoc.mock.MockLoader',
                     'mock': 'sphinx.ext.autodoc.mock.mock',
                 })
```
### 95 - sphinx/ext/autodoc/importer.py:

Start line: 11, End line: 41

```python
import importlib
import traceback
import warnings
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

from sphinx.deprecation import (RemovedInSphinx40Warning, RemovedInSphinx50Warning,
                                deprecated_alias)
from sphinx.ext.autodoc.mock import ismock, undecorate
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import logging
from sphinx.util.inspect import (getannotations, getmro, getslots, isclass, isenumclass,
                                 safe_getattr)

if False:
    # For type annotation
    from typing import Type  # NOQA

    from sphinx.ext.autodoc import ObjectMember

logger = logging.getLogger(__name__)


def mangle(subject: Any, name: str) -> str:
    """mangle the given name."""
    try:
        if isclass(subject) and name.startswith('__') and not name.endswith('__'):
            return "_%s%s" % (subject.__name__, name)
    except AttributeError:
        pass

    return name
```
### 107 - sphinx/ext/autodoc/importer.py:

Start line: 144, End line: 164

```python
def get_module_members(module: Any) -> List[Tuple[str, Any]]:
    """Get members of target module."""
    from sphinx.ext.autodoc import INSTANCEATTR

    warnings.warn('sphinx.ext.autodoc.importer.get_module_members() is deprecated.',
                  RemovedInSphinx50Warning)

    members = {}  # type: Dict[str, Tuple[str, Any]]
    for name in dir(module):
        try:
            value = safe_getattr(module, name, None)
            members[name] = (name, value)
        except AttributeError:
            continue

    # annotation only member (ex. attr: int)
    for name in getannotations(module):
        if name not in members:
            members[name] = (name, INSTANCEATTR)

    return sorted(list(members.values()))
```
