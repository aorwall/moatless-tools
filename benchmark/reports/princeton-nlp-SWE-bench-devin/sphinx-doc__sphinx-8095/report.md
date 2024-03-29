# sphinx-doc__sphinx-8095

| **sphinx-doc/sphinx** | `bf26080042fabf6e3aba22cfe05ad8d93bcad3e9` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 30 |
| **Missing snippets** | 4 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sphinx/ext/napoleon/__init__.py b/sphinx/ext/napoleon/__init__.py
--- a/sphinx/ext/napoleon/__init__.py
+++ b/sphinx/ext/napoleon/__init__.py
@@ -41,6 +41,7 @@ class Config:
         napoleon_use_param = True
         napoleon_use_rtype = True
         napoleon_use_keyword = True
+        napoleon_preprocess_types = False
         napoleon_type_aliases = None
         napoleon_custom_sections = None
 
@@ -237,9 +238,12 @@ def __unicode__(self):
 
             :returns: *bool* -- True if successful, False otherwise
 
+    napoleon_preprocess_types : :obj:`bool` (Defaults to False)
+        Enable the type preprocessor for numpy style docstrings.
+
     napoleon_type_aliases : :obj:`dict` (Defaults to None)
         Add a mapping of strings to string, translating types in numpy
-        style docstrings.
+        style docstrings. Only works if ``napoleon_preprocess_types = True``.
 
     napoleon_custom_sections : :obj:`list` (Defaults to None)
         Add a list of custom sections to include, expanding the list of parsed sections.
@@ -268,6 +272,7 @@ def __unicode__(self):
         'napoleon_use_param': (True, 'env'),
         'napoleon_use_rtype': (True, 'env'),
         'napoleon_use_keyword': (True, 'env'),
+        'napoleon_preprocess_types': (False, 'env'),
         'napoleon_type_aliases': (None, 'env'),
         'napoleon_custom_sections': (None, 'env')
     }
diff --git a/sphinx/ext/napoleon/docstring.py b/sphinx/ext/napoleon/docstring.py
--- a/sphinx/ext/napoleon/docstring.py
+++ b/sphinx/ext/napoleon/docstring.py
@@ -1104,11 +1104,12 @@ def _consume_field(self, parse_type: bool = True, prefer_type: bool = False
             _name, _type = line, ''
         _name, _type = _name.strip(), _type.strip()
         _name = self._escape_args_and_kwargs(_name)
-        _type = _convert_numpy_type_spec(
-            _type,
-            location=self._get_location(),
-            translations=self._config.napoleon_type_aliases or {},
-        )
+        if self._config.napoleon_preprocess_types:
+            _type = _convert_numpy_type_spec(
+                _type,
+                location=self._get_location(),
+                translations=self._config.napoleon_type_aliases or {},
+            )
 
         if prefer_type and not _type:
             _type, _name = _name, _type

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/ext/napoleon/__init__.py | 44 | 44 | - | - | -
| sphinx/ext/napoleon/__init__.py | 240 | 240 | - | - | -
| sphinx/ext/napoleon/__init__.py | 271 | 271 | - | - | -
| sphinx/ext/napoleon/docstring.py | 1107 | 1111 | - | 30 | -


## Problem Statement

```
Warning: Inline literal start-string without end-string in Numpy style Parameters section
**Describe the bug**
The following docstring generates a warning on the line of the timeout parameter. Removing the quote around `default` cause the warning to go away.
\`\`\`python
def lock(
        self,
        timeout: Union[float, Literal["default"]] = "default",
        requested_key: Optional[str] = None,
    ) -> str:
        """Establish a shared lock to the resource.

        Parameters
        ----------
        timeout : Union[float, Literal["default"]], optional
            Absolute time period (in milliseconds) that a resource waits to get
            unlocked by the locking session before returning an error.
            Defaults to "default" which means use self.timeout.
        requested_key : Optional[str], optional
            Access key used by another session with which you want your session
            to share a lock or None to generate a new shared access key.

        Returns
        -------
        str
            A new shared access key if requested_key is None, otherwise, same
            value as the requested_key

        """
\`\`\`

**To Reproduce**
Steps to reproduce the behavior:
\`\`\`
$ git clone https://github.com/pyvisa/pyvisa
$ git checkout pytest
$ cd pyvisa
$ pip install -e .
$ cd docs
$ sphinx-build source build -W -b html;
\`\`\`

**Expected behavior**
I do not expect to see a warning there and was not seeing any before 3.2

**Your project**
The project is build under the Documentation build action. https://github.com/pyvisa/pyvisa/pull/531

**Environment info**
- OS: Mac Os and Linux
- Python version: 3.8.2 and 3.8.5
- Sphinx version: 3.2.0
- Sphinx extensions: "sphinx.ext.autodoc", "sphinx.ext.doctest","sphinx.ext.intersphinx", "sphinx.ext.coverage", "sphinx.ext.viewcode", "sphinx.ext.mathjax",  "sphinx.ext.napoleon"



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sphinx/__init__.py | 14 | 65| 495 | 495 | 569 | 
| 2 | 2 sphinx/parsers.py | 11 | 28| 134 | 629 | 1437 | 
| 3 | 3 sphinx/deprecation.py | 11 | 36| 164 | 793 | 2149 | 
| 4 | 3 sphinx/deprecation.py | 67 | 94| 261 | 1054 | 2149 | 
| 5 | 4 sphinx/util/pycompat.py | 92 | 111| 158 | 1212 | 3017 | 
| 6 | 5 sphinx/util/inspect.py | 11 | 48| 276 | 1488 | 9374 | 
| 7 | 6 sphinx/ext/jsmath.py | 12 | 37| 145 | 1633 | 9581 | 
| 8 | 7 sphinx/builders/devhelp.py | 13 | 42| 173 | 1806 | 9828 | 
| 9 | 8 sphinx/domains/python.py | 1369 | 1407| 304 | 2110 | 21729 | 
| 10 | 9 sphinx/errors.py | 38 | 67| 213 | 2323 | 22471 | 
| 11 | 10 sphinx/builders/latex/__init__.py | 11 | 42| 329 | 2652 | 28334 | 
| 12 | 11 sphinx/builders/qthelp.py | 11 | 44| 212 | 2864 | 28601 | 
| 13 | 12 sphinx/builders/htmlhelp.py | 12 | 50| 298 | 3162 | 28964 | 
| 14 | 13 sphinx/util/__init__.py | 11 | 75| 509 | 3671 | 35269 | 
| 15 | 14 sphinx/ext/autosummary/generate.py | 20 | 60| 289 | 3960 | 40552 | 
| 16 | 15 sphinx/environment/__init__.py | 11 | 82| 489 | 4449 | 46415 | 
| 17 | 16 sphinx/util/requests.py | 11 | 62| 368 | 4817 | 47316 | 
| 18 | 17 sphinx/directives/code.py | 9 | 31| 146 | 4963 | 51224 | 
| 19 | 18 sphinx/cmd/quickstart.py | 11 | 119| 772 | 5735 | 56784 | 
| 20 | 18 sphinx/directives/code.py | 383 | 416| 284 | 6019 | 56784 | 
| 21 | 19 sphinx/util/smartypants.py | 376 | 388| 137 | 6156 | 60895 | 
| 22 | 20 sphinx/ext/doctest.py | 12 | 51| 286 | 6442 | 65983 | 
| 23 | 21 sphinx/util/logging.py | 11 | 56| 279 | 6721 | 69817 | 
| 24 | 22 sphinx/io.py | 10 | 47| 282 | 7003 | 71567 | 
| 25 | 23 sphinx/events.py | 13 | 54| 352 | 7355 | 72587 | 
| 26 | 23 sphinx/util/pycompat.py | 11 | 54| 357 | 7712 | 72587 | 
| 27 | 24 sphinx/ext/todo.py | 14 | 44| 207 | 7919 | 75304 | 
| 28 | 25 sphinx/builders/applehelp.py | 11 | 50| 269 | 8188 | 75623 | 
| 29 | 25 sphinx/deprecation.py | 39 | 64| 237 | 8425 | 75623 | 
| 30 | 26 sphinx/builders/latex/constants.py | 70 | 119| 523 | 8948 | 77703 | 
| 31 | 26 sphinx/util/logging.py | 426 | 464| 246 | 9194 | 77703 | 
| 32 | 27 sphinx/util/cfamily.py | 11 | 80| 766 | 9960 | 81206 | 
| 33 | 27 sphinx/domains/python.py | 11 | 78| 522 | 10482 | 81206 | 
| 34 | 28 sphinx/util/nodes.py | 11 | 41| 227 | 10709 | 86632 | 
| 35 | 29 sphinx/cmd/build.py | 11 | 30| 132 | 10841 | 89293 | 
| 36 | **30 sphinx/ext/napoleon/docstring.py** | 13 | 57| 496 | 11337 | 99620 | 
| 37 | 31 sphinx/directives/__init__.py | 269 | 329| 527 | 11864 | 102369 | 
| 38 | 31 sphinx/util/logging.py | 188 | 213| 135 | 11999 | 102369 | 
| 39 | 32 doc/conf.py | 1 | 58| 502 | 12501 | 103923 | 
| 40 | 33 sphinx/testing/util.py | 10 | 50| 300 | 12801 | 105745 | 
| 41 | 33 sphinx/util/logging.py | 396 | 423| 191 | 12992 | 105745 | 
| 42 | 34 sphinx/application.py | 13 | 60| 378 | 13370 | 116513 | 
| 43 | 34 sphinx/util/__init__.py | 397 | 414| 207 | 13577 | 116513 | 
| 44 | 35 sphinx/writers/manpage.py | 11 | 46| 232 | 13809 | 120024 | 
| 45 | 36 sphinx/util/i18n.py | 10 | 37| 166 | 13975 | 122922 | 
| 46 | 36 sphinx/directives/code.py | 418 | 482| 658 | 14633 | 122922 | 
| 47 | 37 sphinx/ext/autodoc/importer.py | 11 | 36| 172 | 14805 | 124940 | 
| 48 | 37 sphinx/errors.py | 70 | 127| 297 | 15102 | 124940 | 
| 49 | 38 sphinx/domains/math.py | 11 | 40| 207 | 15309 | 126450 | 
| 50 | 38 sphinx/util/logging.py | 265 | 282| 131 | 15440 | 126450 | 
| 51 | 39 sphinx/domains/std.py | 11 | 48| 323 | 15763 | 136786 | 
| 52 | 39 sphinx/builders/latex/__init__.py | 452 | 487| 255 | 16018 | 136786 | 
| 53 | 39 doc/conf.py | 59 | 126| 693 | 16711 | 136786 | 
| 54 | 39 sphinx/util/logging.py | 353 | 369| 121 | 16832 | 136786 | 
| 55 | 39 sphinx/cmd/quickstart.py | 207 | 282| 673 | 17505 | 136786 | 
| 56 | 40 sphinx/util/docutils.py | 241 | 264| 220 | 17725 | 140912 | 
| 57 | 41 sphinx/builders/singlehtml.py | 11 | 27| 133 | 17858 | 142794 | 
| 58 | 42 sphinx/setup_command.py | 114 | 136| 216 | 18074 | 144525 | 
| 59 | 43 sphinx/transforms/post_transforms/__init__.py | 154 | 177| 279 | 18353 | 146525 | 
| 60 | 43 sphinx/builders/latex/__init__.py | 139 | 150| 122 | 18475 | 146525 | 
| 61 | 43 sphinx/builders/latex/constants.py | 121 | 203| 925 | 19400 | 146525 | 
| 62 | 43 sphinx/ext/autodoc/importer.py | 240 | 260| 181 | 19581 | 146525 | 
| 63 | 44 sphinx/ext/autodoc/directive.py | 9 | 49| 298 | 19879 | 147786 | 
| 64 | 45 sphinx/transforms/__init__.py | 11 | 45| 234 | 20113 | 151035 | 
| 65 | 45 sphinx/application.py | 337 | 386| 410 | 20523 | 151035 | 
| 66 | 46 setup.py | 172 | 248| 638 | 21161 | 152760 | 
| 67 | 47 utils/checks.py | 33 | 109| 545 | 21706 | 153666 | 
| 68 | 48 sphinx/config.py | 98 | 155| 703 | 22409 | 158068 | 
| 69 | 48 sphinx/util/logging.py | 285 | 328| 248 | 22657 | 158068 | 
| 70 | 49 sphinx/ext/autodoc/__init__.py | 1147 | 1167| 242 | 22899 | 177041 | 
| 71 | 49 sphinx/testing/util.py | 180 | 209| 245 | 23144 | 177041 | 
| 72 | 49 sphinx/cmd/quickstart.py | 551 | 618| 491 | 23635 | 177041 | 
| 73 | 50 sphinx/writers/latex.py | 14 | 73| 453 | 24088 | 196948 | 


## Missing Patch Files

 * 1: sphinx/ext/napoleon/__init__.py
 * 2: sphinx/ext/napoleon/docstring.py

### Hint

```
@keewis Could you check this please? I think this is related to convert_numpy_type_spec.
`napoleon` converts the docstring to
\`\`\`rst
Establish a shared lock to the resource.

:Parameters: * **timeout** (:class:`Union[float`, :class:`Literal[\`\`\`"default"``:class:`]]`, *optional*) -- Absolute time period (in milliseconds) that a resource waits to get
               unlocked by the locking session before returning an error.
               Defaults to "default" which means use self.timeout.
             * **requested_key** (:class:`Optional[str]`, *optional*) -- Access key used by another session with which you want your session
               to share a lock or None to generate a new shared access key.

:returns: *str* -- A new shared access key if requested_key is None, otherwise, same
          value as the requested_key
\`\`\`
which I guess happens because I never considered typehints when I wrote the preprocessor. To be clear, type hints are not part of the format guide, but then again it also doesn't say they can't be used.

If we allow type hints, we probably want to link those types and thus should extend the preprocessor. Since that would be a new feature, I guess we shouldn't include that in a bugfix release.

For now, I suggest we fix this by introducing a setting that allows opting out of the type preprocessor (could also be opt-in).
Faced the same issue in our builds yesterday.

\`\`\`
Warning, treated as error:
/home/travis/build/microsoft/LightGBM/docs/../python-package/lightgbm/basic.py:docstring of lightgbm.Booster.dump_model:12:Inline literal start-string without end-string.
\`\`\`

`conf.py`: https://github.com/microsoft/LightGBM/blob/master/docs/conf.py
 Logs: https://travis-ci.org/github/microsoft/LightGBM/jobs/716228303

One of the "problem" docstrings: https://github.com/microsoft/LightGBM/blob/ee8ec182010c570c6371a5fc68ab9f4da9c6dc74/python-package/lightgbm/basic.py#L2762-L2782

that's a separate issue: you're using a unsupported notation for `default`. Supported are currently `default <obj>` and `default: <obj>`, while you are using `optional (default=<obj>)`. To be fair, this is currently not standardized, see numpy/numpydoc#289.

Edit: in particular, the type preprocessor chokes on something like `string, optional (default="split")`, which becomes:
\`\`\`rst
:class:`string`, :class:`optional (default=\`\`\`"split"``:class:`)`
\`\`\`
so it splits the default notation into `optional (default=`, `"split"`, and `)`

However, the temporary fix is the same: deactivate the type preprocessor using a new setting. For a long term fix we'd first need to update the `numpydoc` format guide.

@tk0miya, should I send in a PR that adds that setting?
@keewis Yes, please.

>If we allow type hints, we probably want to link those types and thus should extend the preprocessor. Since that would be a new feature, I guess we shouldn't include that in a bugfix release.

I think the new option is needed to keep compatibility for some users. So it must be released as a bugfix release. So could you send a PR to 3.2.x branch? I'm still debating which is better to enable or disable the numpy type feature by default. But it should be controlled via user settings.
```

## Patch

```diff
diff --git a/sphinx/ext/napoleon/__init__.py b/sphinx/ext/napoleon/__init__.py
--- a/sphinx/ext/napoleon/__init__.py
+++ b/sphinx/ext/napoleon/__init__.py
@@ -41,6 +41,7 @@ class Config:
         napoleon_use_param = True
         napoleon_use_rtype = True
         napoleon_use_keyword = True
+        napoleon_preprocess_types = False
         napoleon_type_aliases = None
         napoleon_custom_sections = None
 
@@ -237,9 +238,12 @@ def __unicode__(self):
 
             :returns: *bool* -- True if successful, False otherwise
 
+    napoleon_preprocess_types : :obj:`bool` (Defaults to False)
+        Enable the type preprocessor for numpy style docstrings.
+
     napoleon_type_aliases : :obj:`dict` (Defaults to None)
         Add a mapping of strings to string, translating types in numpy
-        style docstrings.
+        style docstrings. Only works if ``napoleon_preprocess_types = True``.
 
     napoleon_custom_sections : :obj:`list` (Defaults to None)
         Add a list of custom sections to include, expanding the list of parsed sections.
@@ -268,6 +272,7 @@ def __unicode__(self):
         'napoleon_use_param': (True, 'env'),
         'napoleon_use_rtype': (True, 'env'),
         'napoleon_use_keyword': (True, 'env'),
+        'napoleon_preprocess_types': (False, 'env'),
         'napoleon_type_aliases': (None, 'env'),
         'napoleon_custom_sections': (None, 'env')
     }
diff --git a/sphinx/ext/napoleon/docstring.py b/sphinx/ext/napoleon/docstring.py
--- a/sphinx/ext/napoleon/docstring.py
+++ b/sphinx/ext/napoleon/docstring.py
@@ -1104,11 +1104,12 @@ def _consume_field(self, parse_type: bool = True, prefer_type: bool = False
             _name, _type = line, ''
         _name, _type = _name.strip(), _type.strip()
         _name = self._escape_args_and_kwargs(_name)
-        _type = _convert_numpy_type_spec(
-            _type,
-            location=self._get_location(),
-            translations=self._config.napoleon_type_aliases or {},
-        )
+        if self._config.napoleon_preprocess_types:
+            _type = _convert_numpy_type_spec(
+                _type,
+                location=self._get_location(),
+                translations=self._config.napoleon_type_aliases or {},
+            )
 
         if prefer_type and not _type:
             _type, _name = _name, _type

```

## Test Patch

```diff
diff --git a/tests/test_ext_napoleon_docstring.py b/tests/test_ext_napoleon_docstring.py
--- a/tests/test_ext_napoleon_docstring.py
+++ b/tests/test_ext_napoleon_docstring.py
@@ -66,19 +66,19 @@ def test_attributes_docstring(self):
 
    Quick description of attr1
 
-   :type: :class:`Arbitrary type`
+   :type: Arbitrary type
 
 .. attribute:: attr2
 
    Quick description of attr2
 
-   :type: :class:`Another arbitrary type`
+   :type: Another arbitrary type
 
 .. attribute:: attr3
 
    Adds a newline after the type
 
-   :type: :class:`Type`
+   :type: Type
 """
 
         self.assertEqual(expected, actual)
@@ -1311,12 +1311,34 @@ def test_docstrings(self):
         config = Config(
             napoleon_use_param=False,
             napoleon_use_rtype=False,
-            napoleon_use_keyword=False)
+            napoleon_use_keyword=False,
+            napoleon_preprocess_types=True)
         for docstring, expected in self.docstrings:
             actual = str(NumpyDocstring(dedent(docstring), config))
             expected = dedent(expected)
             self.assertEqual(expected, actual)
 
+    def test_type_preprocessor(self):
+        docstring = dedent("""
+        Single line summary
+
+        Parameters
+        ----------
+        arg1:str
+            Extended
+            description of arg1
+        """)
+
+        config = Config(napoleon_preprocess_types=False, napoleon_use_param=False)
+        actual = str(NumpyDocstring(docstring, config))
+        expected = dedent("""
+        Single line summary
+
+        :Parameters: **arg1** (*str*) -- Extended
+                     description of arg1
+        """)
+        self.assertEqual(expected, actual)
+
     def test_parameters_with_class_reference(self):
         docstring = """\
 Parameters
@@ -1352,7 +1374,7 @@ def test_multiple_parameters(self):
         config = Config(napoleon_use_param=False)
         actual = str(NumpyDocstring(docstring, config))
         expected = """\
-:Parameters: **x1, x2** (:class:`array_like`) -- Input arrays, description of ``x1``, ``x2``.
+:Parameters: **x1, x2** (*array_like*) -- Input arrays, description of ``x1``, ``x2``.
 """
         self.assertEqual(expected, actual)
 
@@ -1360,9 +1382,9 @@ def test_multiple_parameters(self):
         actual = str(NumpyDocstring(dedent(docstring), config))
         expected = """\
 :param x1: Input arrays, description of ``x1``, ``x2``.
-:type x1: :class:`array_like`
+:type x1: array_like
 :param x2: Input arrays, description of ``x1``, ``x2``.
-:type x2: :class:`array_like`
+:type x2: array_like
 """
         self.assertEqual(expected, actual)
 
@@ -1377,7 +1399,7 @@ def test_parameters_without_class_reference(self):
         config = Config(napoleon_use_param=False)
         actual = str(NumpyDocstring(docstring, config))
         expected = """\
-:Parameters: **param1** (:class:`MyClass instance`)
+:Parameters: **param1** (*MyClass instance*)
 """
         self.assertEqual(expected, actual)
 
@@ -1385,7 +1407,7 @@ def test_parameters_without_class_reference(self):
         actual = str(NumpyDocstring(dedent(docstring), config))
         expected = """\
 :param param1:
-:type param1: :class:`MyClass instance`
+:type param1: MyClass instance
 """
         self.assertEqual(expected, actual)
 
@@ -1474,7 +1496,7 @@ def test_underscore_in_attribute(self):
 
         expected = """
 :ivar arg_: some description
-:vartype arg_: :class:`type`
+:vartype arg_: type
 """
 
         config = Config(napoleon_use_ivar=True)
@@ -1494,7 +1516,7 @@ def test_underscore_in_attribute_strip_signature_backslash(self):
 
         expected = """
 :ivar arg\\_: some description
-:vartype arg\\_: :class:`type`
+:vartype arg\\_: type
 """
 
         config = Config(napoleon_use_ivar=True)
@@ -1862,59 +1884,59 @@ def test_list_in_parameter_description(self):
         expected = """One line summary.
 
 :param no_list:
-:type no_list: :class:`int`
+:type no_list: int
 :param one_bullet_empty:
                          *
-:type one_bullet_empty: :class:`int`
+:type one_bullet_empty: int
 :param one_bullet_single_line:
                                - first line
-:type one_bullet_single_line: :class:`int`
+:type one_bullet_single_line: int
 :param one_bullet_two_lines:
                              +   first line
                                  continued
-:type one_bullet_two_lines: :class:`int`
+:type one_bullet_two_lines: int
 :param two_bullets_single_line:
                                 -  first line
                                 -  second line
-:type two_bullets_single_line: :class:`int`
+:type two_bullets_single_line: int
 :param two_bullets_two_lines:
                               * first line
                                 continued
                               * second line
                                 continued
-:type two_bullets_two_lines: :class:`int`
+:type two_bullets_two_lines: int
 :param one_enumeration_single_line:
                                     1.  first line
-:type one_enumeration_single_line: :class:`int`
+:type one_enumeration_single_line: int
 :param one_enumeration_two_lines:
                                   1)   first line
                                        continued
-:type one_enumeration_two_lines: :class:`int`
+:type one_enumeration_two_lines: int
 :param two_enumerations_one_line:
                                   (iii) first line
                                   (iv) second line
-:type two_enumerations_one_line: :class:`int`
+:type two_enumerations_one_line: int
 :param two_enumerations_two_lines:
                                    a. first line
                                       continued
                                    b. second line
                                       continued
-:type two_enumerations_two_lines: :class:`int`
+:type two_enumerations_two_lines: int
 :param one_definition_one_line:
                                 item 1
                                     first line
-:type one_definition_one_line: :class:`int`
+:type one_definition_one_line: int
 :param one_definition_two_lines:
                                  item 1
                                      first line
                                      continued
-:type one_definition_two_lines: :class:`int`
+:type one_definition_two_lines: int
 :param two_definitions_one_line:
                                  item 1
                                      first line
                                  item 2
                                      second line
-:type two_definitions_one_line: :class:`int`
+:type two_definitions_one_line: int
 :param two_definitions_two_lines:
                                   item 1
                                       first line
@@ -1922,14 +1944,14 @@ def test_list_in_parameter_description(self):
                                   item 2
                                       second line
                                       continued
-:type two_definitions_two_lines: :class:`int`
+:type two_definitions_two_lines: int
 :param one_definition_blank_line:
                                   item 1
 
                                       first line
 
                                       extra first line
-:type one_definition_blank_line: :class:`int`
+:type one_definition_blank_line: int
 :param two_definitions_blank_lines:
                                     item 1
 
@@ -1942,12 +1964,12 @@ def test_list_in_parameter_description(self):
                                         second line
 
                                         extra second line
-:type two_definitions_blank_lines: :class:`int`
+:type two_definitions_blank_lines: int
 :param definition_after_normal_text: text line
 
                                      item 1
                                          first line
-:type definition_after_normal_text: :class:`int`
+:type definition_after_normal_text: int
 """
         config = Config(napoleon_use_param=True)
         actual = str(NumpyDocstring(docstring, config))
@@ -2041,7 +2063,7 @@ def test_list_in_parameter_description(self):
                item 1
                    first line
 """
-        config = Config(napoleon_use_param=False)
+        config = Config(napoleon_use_param=False, napoleon_preprocess_types=True)
         actual = str(NumpyDocstring(docstring, config))
         self.assertEqual(expected, actual)
 
@@ -2222,6 +2244,7 @@ def test_parameter_types(self):
         config = Config(
             napoleon_use_param=True,
             napoleon_use_rtype=True,
+            napoleon_preprocess_types=True,
             napoleon_type_aliases=translations,
         )
         actual = str(NumpyDocstring(docstring, config))

```


## Code snippets

### 1 - sphinx/__init__.py:

Start line: 14, End line: 65

```python
import os
import subprocess
import warnings
from os import path
from subprocess import PIPE

from .deprecation import RemovedInNextVersionWarning

if False:
    # For type annotation
    from typing import Any  # NOQA


# by default, all DeprecationWarning under sphinx package will be emit.
# Users can avoid this by using environment variable: PYTHONWARNINGS=
if 'PYTHONWARNINGS' not in os.environ:
    warnings.filterwarnings('default', category=RemovedInNextVersionWarning)
# docutils.io using mode='rU' for open
warnings.filterwarnings('ignore', "'U' mode is deprecated",
                        DeprecationWarning, module='docutils.io')

__version__ = '3.3.0+'
__released__ = '3.3.0'  # used when Sphinx builds its own docs

#: Version info for better programmatic use.
#:
#: A tuple of five elements; for Sphinx version 1.2.1 beta 3 this would be
#: ``(1, 2, 1, 'beta', 3)``. The fourth element can be one of: ``alpha``,
#: ``beta``, ``rc``, ``final``. ``final`` always has 0 as the last element.
#:
#: .. versionadded:: 1.2
#:    Before version 1.2, check the string ``sphinx.__version__``.
version_info = (3, 3, 0, 'beta', 0)

package_dir = path.abspath(path.dirname(__file__))

__display_version__ = __version__  # used for command line version
if __version__.endswith('+'):
    # try to find out the commit hash if checked out from git, and append
    # it to __version__ (since we use this value from setup.py, it gets
    # automatically propagated to an installed copy as well)
    __display_version__ = __version__
    __version__ = __version__[:-1]  # remove '+' for PEP-440 version spec.
    try:
        ret = subprocess.run(['git', 'show', '-s', '--pretty=format:%h'],
                             cwd=package_dir,
                             stdout=PIPE, stderr=PIPE)
        if ret.stdout:
            __display_version__ += '/' + ret.stdout.decode('ascii').strip()
    except Exception:
        pass
```
### 2 - sphinx/parsers.py:

Start line: 11, End line: 28

```python
import warnings
from typing import Any, Dict, List, Union

import docutils.parsers
import docutils.parsers.rst
from docutils import nodes
from docutils.parsers.rst import states
from docutils.statemachine import StringList
from docutils.transforms.universal import SmartQuotes

from sphinx.deprecation import RemovedInSphinx50Warning
from sphinx.util.rst import append_epilog, prepend_prolog

if False:
    # For type annotation
    from docutils.transforms import Transform  # NOQA
    from typing import Type  # NOQA # for python3.5.1
    from sphinx.application import Sphinx
```
### 3 - sphinx/deprecation.py:

Start line: 11, End line: 36

```python
import sys
import warnings
from importlib import import_module
from typing import Any, Dict

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


class RemovedInSphinx40Warning(DeprecationWarning):
    pass


class RemovedInSphinx50Warning(PendingDeprecationWarning):
    pass


RemovedInNextVersionWarning = RemovedInSphinx40Warning


def deprecated_alias(modname: str, objects: Dict[str, object],
                     warning: "Type[Warning]", names: Dict[str, str] = None) -> None:
    module = import_module(modname)
    sys.modules[modname] = _ModuleWrapper(  # type: ignore
        module, modname, objects, warning, names)
```
### 4 - sphinx/deprecation.py:

Start line: 67, End line: 94

```python
class DeprecatedDict(dict):
    """A deprecated dict which warns on each access."""

    def __init__(self, data: Dict, message: str, warning: "Type[Warning]") -> None:
        self.message = message
        self.warning = warning
        super().__init__(data)

    def __setitem__(self, key: str, value: Any) -> None:
        warnings.warn(self.message, self.warning, stacklevel=2)
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: Any = None) -> Any:
        warnings.warn(self.message, self.warning, stacklevel=2)
        return super().setdefault(key, default)

    def __getitem__(self, key: str) -> None:
        warnings.warn(self.message, self.warning, stacklevel=2)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        warnings.warn(self.message, self.warning, stacklevel=2)
        return super().get(key, default)

    def update(self, other: Dict) -> None:  # type: ignore
        warnings.warn(self.message, self.warning, stacklevel=2)
        super().update(other)
```
### 5 - sphinx/util/pycompat.py:

Start line: 92, End line: 111

```python
deprecated_alias('sphinx.util.pycompat',
                 {
                     'NoneType': NoneType,
                     'TextIOWrapper': io.TextIOWrapper,
                     'htmlescape': html.escape,
                     'indent': textwrap.indent,
                     'terminal_safe': terminal_safe,
                     'sys_encoding': sys.getdefaultencoding(),
                     'u': '',
                 },
                 RemovedInSphinx40Warning,
                 {
                     'NoneType': 'sphinx.util.typing.NoneType',
                     'TextIOWrapper': 'io.TextIOWrapper',
                     'htmlescape': 'html.escape',
                     'indent': 'textwrap.indent',
                     'terminal_safe': 'sphinx.util.console.terminal_safe',
                     'sys_encoding': 'sys.getdefaultencoding',
                 })
```
### 6 - sphinx/util/inspect.py:

Start line: 11, End line: 48

```python
import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
import warnings
from functools import partial, partialmethod
from inspect import (  # NOQA
    Parameter, isclass, ismethod, ismethoddescriptor, ismodule
)
from io import StringIO
from typing import Any, Callable, Dict, Mapping, List, Optional, Tuple
from typing import cast

from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.pycode.ast import ast  # for py35-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation

if sys.version_info > (3, 7):
    from types import (
        ClassMethodDescriptorType,
        MethodDescriptorType,
        WrapperDescriptorType
    )
else:
    ClassMethodDescriptorType = type(object.__init__)
    MethodDescriptorType = type(str.join)
    WrapperDescriptorType = type(dict.__dict__['fromkeys'])

logger = logging.getLogger(__name__)

memory_address_re = re.compile(r' at 0x[0-9a-f]{8,16}(?=>)', re.IGNORECASE)
```
### 7 - sphinx/ext/jsmath.py:

Start line: 12, End line: 37

```python
import warnings
from typing import Any, Dict

from sphinxcontrib.jsmath import (  # NOQA
    html_visit_math,
    html_visit_displaymath,
    install_jsmath,
)

import sphinx
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx40Warning


def setup(app: Sphinx) -> Dict[str, Any]:
    warnings.warn('sphinx.ext.jsmath has been moved to sphinxcontrib-jsmath.',
                  RemovedInSphinx40Warning)

    app.setup_extension('sphinxcontrib.jsmath')

    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 8 - sphinx/builders/devhelp.py:

Start line: 13, End line: 42

```python
import warnings
from typing import Any, Dict

from sphinxcontrib.devhelp import DevhelpBuilder

from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx40Warning, deprecated_alias


deprecated_alias('sphinx.builders.devhelp',
                 {
                     'DevhelpBuilder': DevhelpBuilder,
                 },
                 RemovedInSphinx40Warning,
                 {
                     'DevhelpBuilder': 'sphinxcontrib.devhelp.DevhelpBuilder'
                 })


def setup(app: Sphinx) -> Dict[str, Any]:
    warnings.warn('sphinx.builders.devhelp has been moved to sphinxcontrib-devhelp.',
                  RemovedInSphinx40Warning)
    app.setup_extension('sphinxcontrib.devhelp')

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 9 - sphinx/domains/python.py:

Start line: 1369, End line: 1407

```python
def builtin_resolver(app: Sphinx, env: BuildEnvironment,
                     node: pending_xref, contnode: Element) -> Element:
    """Do not emit nitpicky warnings for built-in types."""
    def istyping(s: str) -> bool:
        if s.startswith('typing.'):
            s = s.split('.', 1)[1]

        return s in typing.__all__  # type: ignore

    if node.get('refdomain') != 'py':
        return None
    elif node.get('reftype') in ('class', 'obj') and node.get('reftarget') == 'None':
        return contnode
    elif node.get('reftype') in ('class', 'exc'):
        reftarget = node.get('reftarget')
        if inspect.isclass(getattr(builtins, reftarget, None)):
            # built-in class
            return contnode
        elif istyping(reftarget):
            # typing class
            return contnode

    return None


def setup(app: Sphinx) -> Dict[str, Any]:
    app.setup_extension('sphinx.directives')

    app.add_domain(PythonDomain)
    app.connect('object-description-transform', filter_meta_fields)
    app.connect('missing-reference', builtin_resolver, priority=900)

    return {
        'version': 'builtin',
        'env_version': 2,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 10 - sphinx/errors.py:

Start line: 38, End line: 67

```python
class SphinxWarning(SphinxError):
    """Warning, treated as error."""
    category = 'Warning, treated as error'


class ApplicationError(SphinxError):
    """Application initialization error."""
    category = 'Application error'


class ExtensionError(SphinxError):
    """Extension error."""
    category = 'Extension error'

    def __init__(self, message: str, orig_exc: Exception = None) -> None:
        super().__init__(message)
        self.message = message
        self.orig_exc = orig_exc

    def __repr__(self) -> str:
        if self.orig_exc:
            return '%s(%r, %r)' % (self.__class__.__name__,
                                   self.message, self.orig_exc)
        return '%s(%r)' % (self.__class__.__name__, self.message)

    def __str__(self) -> str:
        parent_str = super().__str__()
        if self.orig_exc:
            return '%s (exception: %s)' % (parent_str, self.orig_exc)
        return parent_str
```
### 36 - sphinx/ext/napoleon/docstring.py:

Start line: 13, End line: 57

```python
import collections
import inspect
import re
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.iterators import modify_iter
from sphinx.locale import _, __
from sphinx.util import logging

if False:
    # For type annotation
    from typing import Type  # for python3.5.1


logger = logging.getLogger(__name__)

_directive_regex = re.compile(r'\.\. \S+::')
_google_section_regex = re.compile(r'^(\s|\w)+:\s*$')
_google_typed_arg_regex = re.compile(r'\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)')
_numpy_section_regex = re.compile(r'^[=\-`:\'"~^_*+#<>]{2,}\s*$')
_single_colon_regex = re.compile(r'(?<!:):(?!:)')
_xref_or_code_regex = re.compile(
    r'((?::(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:`.+?`)|'
    r'(?:``.+``))')
_xref_regex = re.compile(
    r'(?:(?::(?:[a-zA-Z0-9]+[\-_+:.])*[a-zA-Z0-9]+:)?`.+?`)'
)
_bullet_list_regex = re.compile(r'^(\*|\+|\-)(\s+\S|\s*$)')
_enumerated_list_regex = re.compile(
    r'^(?P<paren>\()?'
    r'(\d+|#|[ivxlcdm]+|[IVXLCDM]+|[a-zA-Z])'
    r'(?(paren)\)|\.)(\s+\S|\s*$)')
_token_regex = re.compile(
    r"(,\sor\s|\sor\s|\sof\s|:\s|\sto\s|,\sand\s|\sand\s|,\s"
    r"|[{]|[}]"
    r'|"(?:\\"|[^"])*"'
    r"|'(?:\\'|[^'])*')"
)
_default_regex = re.compile(
    r"^default[^_0-9A-Za-z].*$",
)
_SINGLETONS = ("None", "True", "False", "Ellipsis")
```
