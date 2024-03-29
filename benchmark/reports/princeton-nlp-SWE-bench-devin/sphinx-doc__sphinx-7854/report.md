# sphinx-doc__sphinx-7854

| **sphinx-doc/sphinx** | `66e55a02d125e7b65b211a7cf9b48506195e3bf4` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 12905 |
| **Any found context length** | 492 |
| **Avg pos** | 19.666666666666668 |
| **Min pos** | 1 |
| **Max pos** | 42 |
| **Top file pos** | 1 |
| **Missing snippets** | 9 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sphinx/domains/c.py b/sphinx/domains/c.py
--- a/sphinx/domains/c.py
+++ b/sphinx/domains/c.py
@@ -28,7 +28,8 @@
 from sphinx.roles import SphinxRole, XRefRole
 from sphinx.util import logging
 from sphinx.util.cfamily import (
-    NoOldIdError, ASTBaseBase, verify_description_mode, StringifyTransform,
+    NoOldIdError, ASTBaseBase, ASTBaseParenExprList,
+    verify_description_mode, StringifyTransform,
     BaseParser, DefinitionError, UnsupportedMultiCharacterCharLiteral,
     identifier_re, anon_identifier_re, integer_literal_re, octal_literal_re,
     hex_literal_re, binary_literal_re, integers_literal_suffix_re,
@@ -1053,7 +1054,7 @@ def describe_signature(self, signode: TextElement, mode: str,
 # Initializer
 ################################################################################
 
-class ASTParenExprList(ASTBase):
+class ASTParenExprList(ASTBaseParenExprList):
     def __init__(self, exprs: List[ASTExpression]) -> None:
         self.exprs = exprs
 
diff --git a/sphinx/domains/cpp.py b/sphinx/domains/cpp.py
--- a/sphinx/domains/cpp.py
+++ b/sphinx/domains/cpp.py
@@ -31,7 +31,8 @@
 from sphinx.transforms.post_transforms import ReferencesResolver
 from sphinx.util import logging
 from sphinx.util.cfamily import (
-    NoOldIdError, ASTBaseBase, ASTAttribute, verify_description_mode, StringifyTransform,
+    NoOldIdError, ASTBaseBase, ASTAttribute, ASTBaseParenExprList,
+    verify_description_mode, StringifyTransform,
     BaseParser, DefinitionError, UnsupportedMultiCharacterCharLiteral,
     identifier_re, anon_identifier_re, integer_literal_re, octal_literal_re,
     hex_literal_re, binary_literal_re, integers_literal_suffix_re,
@@ -2742,7 +2743,7 @@ def describe_signature(self, signode: TextElement, mode: str,
         signode += nodes.Text('...')
 
 
-class ASTParenExprList(ASTBase):
+class ASTParenExprList(ASTBaseParenExprList):
     def __init__(self, exprs: List[Union[ASTExpression, ASTBracedInitList]]) -> None:
         self.exprs = exprs
 
diff --git a/sphinx/util/cfamily.py b/sphinx/util/cfamily.py
--- a/sphinx/util/cfamily.py
+++ b/sphinx/util/cfamily.py
@@ -12,7 +12,7 @@
 import warnings
 from copy import deepcopy
 from typing import (
-    Any, Callable, List, Match, Pattern, Tuple, Union
+    Any, Callable, List, Match, Optional, Pattern, Tuple, Union
 )
 
 from docutils import nodes
@@ -148,16 +148,14 @@ def describe_signature(self, signode: TextElement) -> None:
 
 
 class ASTGnuAttribute(ASTBaseBase):
-    def __init__(self, name: str, args: Any) -> None:
+    def __init__(self, name: str, args: Optional["ASTBaseParenExprList"]) -> None:
         self.name = name
         self.args = args
 
     def _stringify(self, transform: StringifyTransform) -> str:
         res = [self.name]
         if self.args:
-            res.append('(')
             res.append(transform(self.args))
-            res.append(')')
         return ''.join(res)
 
 
@@ -211,6 +209,11 @@ def describe_signature(self, signode: TextElement) -> None:
 
 ################################################################################
 
+class ASTBaseParenExprList(ASTBaseBase):
+    pass
+
+
+################################################################################
 
 class UnsupportedMultiCharacterCharLiteral(Exception):
     @property
@@ -415,11 +418,8 @@ def _parse_attribute(self) -> ASTAttribute:
             while 1:
                 if self.match(identifier_re):
                     name = self.matched_text
-                    self.skip_ws()
-                    if self.skip_string_and_ws('('):
-                        self.fail('Parameterized GNU style attribute not yet supported.')
-                    attrs.append(ASTGnuAttribute(name, None))
-                    # TODO: parse arguments for the attribute
+                    exprs = self._parse_paren_expression_list()
+                    attrs.append(ASTGnuAttribute(name, exprs))
                 if self.skip_string_and_ws(','):
                     continue
                 elif self.skip_string_and_ws(')'):
@@ -447,3 +447,6 @@ def _parse_attribute(self) -> ASTAttribute:
             return ASTParenAttribute(id, arg)
 
         return None
+
+    def _parse_paren_expression_list(self) -> ASTBaseParenExprList:
+        raise NotImplementedError

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sphinx/domains/c.py | 31 | 31 | - | 8 | -
| sphinx/domains/c.py | 1056 | 1056 | - | 8 | -
| sphinx/domains/cpp.py | 34 | 34 | - | 2 | -
| sphinx/domains/cpp.py | 2745 | 2745 | 42 | 2 | 38034
| sphinx/util/cfamily.py | 15 | 15 | 10 | 1 | 12905
| sphinx/util/cfamily.py | 151 | 160 | 2 | 1 | 705
| sphinx/util/cfamily.py | 214 | 214 | 3 | 1 | 1030
| sphinx/util/cfamily.py | 418 | 422 | 1 | 1 | 492
| sphinx/util/cfamily.py | 450 | 450 | 1 | 1 | 492


## Problem Statement

```
Support for parameterized GNU style attributes on C++ code.
Hi folks.

My C++ codebase uses GNU attributes for code like 

`__attribute__ ((optimize(3))) void readMatrix(void)`

Unfortunately, it looks like Sphinx doesn't support them. 

\`\`\`
Exception occurred:
  File "/usr/local/lib/python3.7/site-packages/sphinx/domains/cpp.py", line 6099, in _parse_type
    raise self._make_multi_error(prevErrors, header)
sphinx.util.cfamily.DefinitionError: Error when parsing function declaration.
If the function has no return type:
  Invalid C++ declaration: Parameterized GNU style attribute not yet supported. [error at 25]
    __attribute__ ((optimize(3))) void readMatrix(void)
    -------------------------^
If the function has a return type:
  Invalid C++ declaration: Parameterized GNU style attribute not yet supported. [error at 25]
    __attribute__ ((optimize(3))) void readMatrix(void)
    -------------------------^
\`\`\`

I'm running Sphinx 3.1.1, though this functionality doesn't appear to have changed in 4.

I tried to get clever with the custom attribute support you offer, but can't seem to get that to work either.
\`\`\`
cpp_id_attributes = ["aligned","packed","weak","always_inline","noinline","no-unroll-loops","__attribute__((optimize(3)))"]
cpp_paren_attributes = ["optimize","__aligned__","section","deprecated"]
\`\`\`

Is there a right way to do this? I'd honestly be fine having the attributes stripped entirely for doc generation if there isn't another option.

Even though I'm bumping up against a sharp edge, I really appreciate Sphinx. Thanks so much for making a useful tool. 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sphinx/util/cfamily.py** | 392 | 450| 492 | 492 | 3500 | 
| **-> 2 <-** | **1 sphinx/util/cfamily.py** | 129 | 161| 213 | 705 | 3500 | 
| **-> 3 <-** | **1 sphinx/util/cfamily.py** | 184 | 230| 325 | 1030 | 3500 | 
| 4 | **1 sphinx/util/cfamily.py** | 164 | 181| 136 | 1166 | 3500 | 
| 5 | **2 sphinx/domains/cpp.py** | 7166 | 7517| 3246 | 4412 | 65926 | 
| 6 | **2 sphinx/domains/cpp.py** | 2388 | 2445| 494 | 4906 | 65926 | 
| 7 | 3 sphinx/directives/__init__.py | 269 | 307| 273 | 5179 | 68421 | 
| 8 | **3 sphinx/domains/cpp.py** | 6434 | 7163| 6230 | 11409 | 68421 | 
| 9 | 4 sphinx/pycode/ast.py | 132 | 206| 732 | 12141 | 70113 | 
| **-> 10 <-** | **4 sphinx/util/cfamily.py** | 11 | 80| 764 | 12905 | 70113 | 
| 11 | **4 sphinx/util/cfamily.py** | 293 | 370| 477 | 13382 | 70113 | 
| 12 | 5 sphinx/domains/python.py | 858 | 879| 155 | 13537 | 81760 | 
| 13 | 6 sphinx/pygments_styles.py | 38 | 96| 506 | 14043 | 82452 | 
| 14 | 6 sphinx/domains/python.py | 881 | 893| 132 | 14175 | 82452 | 
| 15 | 7 sphinx/ext/autodoc/__init__.py | 1982 | 2012| 247 | 14422 | 100715 | 
| 16 | **8 sphinx/domains/c.py** | 2699 | 3478| 6630 | 21052 | 129412 | 
| 17 | **8 sphinx/domains/c.py** | 3480 | 3549| 677 | 21729 | 129412 | 
| 18 | 9 sphinx/ext/napoleon/docstring.py | 13 | 40| 327 | 22056 | 138238 | 
| 19 | 9 sphinx/ext/napoleon/docstring.py | 578 | 601| 238 | 22294 | 138238 | 
| 20 | 10 sphinx/directives/code.py | 418 | 482| 658 | 22952 | 142146 | 
| 21 | 10 sphinx/ext/autodoc/__init__.py | 2015 | 2031| 126 | 23078 | 142146 | 
| 22 | **10 sphinx/domains/c.py** | 831 | 870| 304 | 23382 | 142146 | 
| 23 | 11 sphinx/transforms/post_transforms/code.py | 114 | 143| 208 | 23590 | 143174 | 
| 24 | 12 sphinx/transforms/__init__.py | 404 | 438| 245 | 23835 | 146380 | 
| 25 | 13 sphinx/io.py | 10 | 46| 275 | 24110 | 148073 | 
| 26 | 14 sphinx/util/__init__.py | 482 | 495| 123 | 24233 | 154454 | 
| 27 | 15 doc/conf.py | 59 | 126| 693 | 24926 | 156008 | 
| 28 | 15 sphinx/transforms/__init__.py | 11 | 45| 234 | 25160 | 156008 | 
| 29 | 15 sphinx/ext/napoleon/docstring.py | 250 | 268| 222 | 25382 | 156008 | 
| 30 | 15 sphinx/transforms/post_transforms/code.py | 87 | 112| 235 | 25617 | 156008 | 
| 31 | 16 sphinx/ext/autosummary/generate.py | 20 | 60| 283 | 25900 | 161146 | 
| 32 | 17 sphinx/util/inspect.py | 11 | 47| 264 | 26164 | 167026 | 
| 33 | 18 sphinx/highlighting.py | 11 | 54| 379 | 26543 | 168334 | 
| 34 | 18 sphinx/ext/napoleon/docstring.py | 566 | 576| 124 | 26667 | 168334 | 
| 35 | 18 sphinx/ext/autodoc/__init__.py | 1280 | 1526| 2333 | 29000 | 168334 | 
| 36 | 19 sphinx/addnodes.py | 254 | 353| 580 | 29580 | 170853 | 
| 37 | **19 sphinx/domains/cpp.py** | 1499 | 1531| 330 | 29910 | 170853 | 
| 38 | **19 sphinx/util/cfamily.py** | 272 | 291| 184 | 30094 | 170853 | 
| 39 | 20 utils/checks.py | 33 | 109| 545 | 30639 | 171759 | 
| 40 | 20 sphinx/ext/autosummary/generate.py | 273 | 286| 187 | 30826 | 171759 | 
| 41 | 21 sphinx/environment/__init__.py | 11 | 82| 489 | 31315 | 177592 | 
| **-> 42 <-** | **21 sphinx/domains/cpp.py** | 2448 | 3293| 6719 | 38034 | 177592 | 
| 43 | 22 sphinx/domains/std.py | 11 | 48| 323 | 38357 | 187801 | 
| 44 | 23 sphinx/writers/html.py | 171 | 231| 562 | 38919 | 195167 | 
| 45 | 23 sphinx/pygments_styles.py | 11 | 35| 135 | 39054 | 195167 | 


## Patch

```diff
diff --git a/sphinx/domains/c.py b/sphinx/domains/c.py
--- a/sphinx/domains/c.py
+++ b/sphinx/domains/c.py
@@ -28,7 +28,8 @@
 from sphinx.roles import SphinxRole, XRefRole
 from sphinx.util import logging
 from sphinx.util.cfamily import (
-    NoOldIdError, ASTBaseBase, verify_description_mode, StringifyTransform,
+    NoOldIdError, ASTBaseBase, ASTBaseParenExprList,
+    verify_description_mode, StringifyTransform,
     BaseParser, DefinitionError, UnsupportedMultiCharacterCharLiteral,
     identifier_re, anon_identifier_re, integer_literal_re, octal_literal_re,
     hex_literal_re, binary_literal_re, integers_literal_suffix_re,
@@ -1053,7 +1054,7 @@ def describe_signature(self, signode: TextElement, mode: str,
 # Initializer
 ################################################################################
 
-class ASTParenExprList(ASTBase):
+class ASTParenExprList(ASTBaseParenExprList):
     def __init__(self, exprs: List[ASTExpression]) -> None:
         self.exprs = exprs
 
diff --git a/sphinx/domains/cpp.py b/sphinx/domains/cpp.py
--- a/sphinx/domains/cpp.py
+++ b/sphinx/domains/cpp.py
@@ -31,7 +31,8 @@
 from sphinx.transforms.post_transforms import ReferencesResolver
 from sphinx.util import logging
 from sphinx.util.cfamily import (
-    NoOldIdError, ASTBaseBase, ASTAttribute, verify_description_mode, StringifyTransform,
+    NoOldIdError, ASTBaseBase, ASTAttribute, ASTBaseParenExprList,
+    verify_description_mode, StringifyTransform,
     BaseParser, DefinitionError, UnsupportedMultiCharacterCharLiteral,
     identifier_re, anon_identifier_re, integer_literal_re, octal_literal_re,
     hex_literal_re, binary_literal_re, integers_literal_suffix_re,
@@ -2742,7 +2743,7 @@ def describe_signature(self, signode: TextElement, mode: str,
         signode += nodes.Text('...')
 
 
-class ASTParenExprList(ASTBase):
+class ASTParenExprList(ASTBaseParenExprList):
     def __init__(self, exprs: List[Union[ASTExpression, ASTBracedInitList]]) -> None:
         self.exprs = exprs
 
diff --git a/sphinx/util/cfamily.py b/sphinx/util/cfamily.py
--- a/sphinx/util/cfamily.py
+++ b/sphinx/util/cfamily.py
@@ -12,7 +12,7 @@
 import warnings
 from copy import deepcopy
 from typing import (
-    Any, Callable, List, Match, Pattern, Tuple, Union
+    Any, Callable, List, Match, Optional, Pattern, Tuple, Union
 )
 
 from docutils import nodes
@@ -148,16 +148,14 @@ def describe_signature(self, signode: TextElement) -> None:
 
 
 class ASTGnuAttribute(ASTBaseBase):
-    def __init__(self, name: str, args: Any) -> None:
+    def __init__(self, name: str, args: Optional["ASTBaseParenExprList"]) -> None:
         self.name = name
         self.args = args
 
     def _stringify(self, transform: StringifyTransform) -> str:
         res = [self.name]
         if self.args:
-            res.append('(')
             res.append(transform(self.args))
-            res.append(')')
         return ''.join(res)
 
 
@@ -211,6 +209,11 @@ def describe_signature(self, signode: TextElement) -> None:
 
 ################################################################################
 
+class ASTBaseParenExprList(ASTBaseBase):
+    pass
+
+
+################################################################################
 
 class UnsupportedMultiCharacterCharLiteral(Exception):
     @property
@@ -415,11 +418,8 @@ def _parse_attribute(self) -> ASTAttribute:
             while 1:
                 if self.match(identifier_re):
                     name = self.matched_text
-                    self.skip_ws()
-                    if self.skip_string_and_ws('('):
-                        self.fail('Parameterized GNU style attribute not yet supported.')
-                    attrs.append(ASTGnuAttribute(name, None))
-                    # TODO: parse arguments for the attribute
+                    exprs = self._parse_paren_expression_list()
+                    attrs.append(ASTGnuAttribute(name, exprs))
                 if self.skip_string_and_ws(','):
                     continue
                 elif self.skip_string_and_ws(')'):
@@ -447,3 +447,6 @@ def _parse_attribute(self) -> ASTAttribute:
             return ASTParenAttribute(id, arg)
 
         return None
+
+    def _parse_paren_expression_list(self) -> ASTBaseParenExprList:
+        raise NotImplementedError

```

## Test Patch

```diff
diff --git a/tests/test_domain_c.py b/tests/test_domain_c.py
--- a/tests/test_domain_c.py
+++ b/tests/test_domain_c.py
@@ -469,6 +469,8 @@ def test_attributes():
     check('member', '__attribute__(()) int f', {1: 'f'})
     check('member', '__attribute__((a)) int f', {1: 'f'})
     check('member', '__attribute__((a, b)) int f', {1: 'f'})
+    check('member', '__attribute__((optimize(3))) int f', {1: 'f'})
+    check('member', '__attribute__((format(printf, 1, 2))) int f', {1: 'f'})
     # style: user-defined id
     check('member', 'id_attr int f', {1: 'f'})
     # style: user-defined paren
diff --git a/tests/test_domain_cpp.py b/tests/test_domain_cpp.py
--- a/tests/test_domain_cpp.py
+++ b/tests/test_domain_cpp.py
@@ -897,6 +897,8 @@ def test_attributes():
     check('member', '__attribute__(()) int f', {1: 'f__i', 2: '1f'})
     check('member', '__attribute__((a)) int f', {1: 'f__i', 2: '1f'})
     check('member', '__attribute__((a, b)) int f', {1: 'f__i', 2: '1f'})
+    check('member', '__attribute__((optimize(3))) int f', {1: 'f__i', 2: '1f'})
+    check('member', '__attribute__((format(printf, 1, 2))) int f', {1: 'f__i', 2: '1f'})
     # style: user-defined id
     check('member', 'id_attr int f', {1: 'f__i', 2: '1f'})
     # style: user-defined paren

```


## Code snippets

### 1 - sphinx/util/cfamily.py:

Start line: 392, End line: 450

```python
class BaseParser:

    def _parse_attribute(self) -> ASTAttribute:
        self.skip_ws()
        # try C++11 style
        startPos = self.pos
        if self.skip_string_and_ws('['):
            if not self.skip_string('['):
                self.pos = startPos
            else:
                # TODO: actually implement the correct grammar
                arg = self._parse_balanced_token_seq(end=[']'])
                if not self.skip_string_and_ws(']'):
                    self.fail("Expected ']' in end of attribute.")
                if not self.skip_string_and_ws(']'):
                    self.fail("Expected ']' in end of attribute after [[...]")
                return ASTCPPAttribute(arg)

        # try GNU style
        if self.skip_word_and_ws('__attribute__'):
            if not self.skip_string_and_ws('('):
                self.fail("Expected '(' after '__attribute__'.")
            if not self.skip_string_and_ws('('):
                self.fail("Expected '(' after '__attribute__('.")
            attrs = []
            while 1:
                if self.match(identifier_re):
                    name = self.matched_text
                    self.skip_ws()
                    if self.skip_string_and_ws('('):
                        self.fail('Parameterized GNU style attribute not yet supported.')
                    attrs.append(ASTGnuAttribute(name, None))
                    # TODO: parse arguments for the attribute
                if self.skip_string_and_ws(','):
                    continue
                elif self.skip_string_and_ws(')'):
                    break
                else:
                    self.fail("Expected identifier, ')', or ',' in __attribute__.")
            if not self.skip_string_and_ws(')'):
                self.fail("Expected ')' after '__attribute__((...)'")
            return ASTGnuAttributeList(attrs)

        # try the simple id attributes defined by the user
        for id in self.id_attributes:
            if self.skip_word_and_ws(id):
                return ASTIdAttribute(id)

        # try the paren attributes defined by the user
        for id in self.paren_attributes:
            if not self.skip_string_and_ws(id):
                continue
            if not self.skip_string('('):
                self.fail("Expected '(' after user-defined paren-attribute.")
            arg = self._parse_balanced_token_seq(end=[')'])
            if not self.skip_string(')'):
                self.fail("Expected ')' to end user-defined paren-attribute.")
            return ASTParenAttribute(id, arg)

        return None
```
### 2 - sphinx/util/cfamily.py:

Start line: 129, End line: 161

```python
################################################################################
# Attributes
################################################################################

class ASTAttribute(ASTBaseBase):
    def describe_signature(self, signode: TextElement) -> None:
        raise NotImplementedError(repr(self))


class ASTCPPAttribute(ASTAttribute):
    def __init__(self, arg: str) -> None:
        self.arg = arg

    def _stringify(self, transform: StringifyTransform) -> str:
        return "[[" + self.arg + "]]"

    def describe_signature(self, signode: TextElement) -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))


class ASTGnuAttribute(ASTBaseBase):
    def __init__(self, name: str, args: Any) -> None:
        self.name = name
        self.args = args

    def _stringify(self, transform: StringifyTransform) -> str:
        res = [self.name]
        if self.args:
            res.append('(')
            res.append(transform(self.args))
            res.append(')')
        return ''.join(res)
```
### 3 - sphinx/util/cfamily.py:

Start line: 184, End line: 230

```python
class ASTIdAttribute(ASTAttribute):
    """For simple attributes defined by the user."""

    def __init__(self, id: str) -> None:
        self.id = id

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.id

    def describe_signature(self, signode: TextElement) -> None:
        signode.append(nodes.Text(self.id, self.id))


class ASTParenAttribute(ASTAttribute):
    """For paren attributes defined by the user."""

    def __init__(self, id: str, arg: str) -> None:
        self.id = id
        self.arg = arg

    def _stringify(self, transform: StringifyTransform) -> str:
        return self.id + '(' + self.arg + ')'

    def describe_signature(self, signode: TextElement) -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))


################################################################################


class UnsupportedMultiCharacterCharLiteral(Exception):
    @property
    def decoded(self) -> str:
        warnings.warn('%s.decoded is deprecated. '
                      'Coerce the instance to a string instead.' % self.__class__.__name__,
                      RemovedInSphinx40Warning, stacklevel=2)
        return str(self)


class DefinitionError(Exception):
    @property
    def description(self) -> str:
        warnings.warn('%s.description is deprecated. '
                      'Coerce the instance to a string instead.' % self.__class__.__name__,
                      RemovedInSphinx40Warning, stacklevel=2)
        return str(self)
```
### 4 - sphinx/util/cfamily.py:

Start line: 164, End line: 181

```python
class ASTGnuAttributeList(ASTAttribute):
    def __init__(self, attrs: List[ASTGnuAttribute]) -> None:
        self.attrs = attrs

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['__attribute__((']
        first = True
        for attr in self.attrs:
            if not first:
                res.append(', ')
            first = False
            res.append(transform(attr))
        res.append('))')
        return ''.join(res)

    def describe_signature(self, signode: TextElement) -> None:
        txt = str(self)
        signode.append(nodes.Text(txt, txt))
```
### 5 - sphinx/domains/cpp.py:

Start line: 7166, End line: 7517

```python
class CPPDomain(Domain):
    """C++ language domain.

    There are two 'object type' attributes being used::

    - Each object created from directives gets an assigned .objtype from ObjectDescription.run.
      This is simply the directive name.
    - Each declaration (see the distinction in the directives dict below) has a nested .ast of
      type ASTDeclaration. That object has .objectType which corresponds to the keys in the
      object_types dict below. They are the core different types of declarations in C++ that
      one can document.
    """
    name = 'cpp'
    label = 'C++'
    object_types = {
        'class':      ObjType(_('class'),      'class',             'type', 'identifier'),
        'union':      ObjType(_('union'),      'union',             'type', 'identifier'),
        'function':   ObjType(_('function'),   'function',  'func', 'type', 'identifier'),
        'member':     ObjType(_('member'),     'member',    'var'),
        'type':       ObjType(_('type'),                            'type', 'identifier'),
        'concept':    ObjType(_('concept'),    'concept',                   'identifier'),
        'enum':       ObjType(_('enum'),       'enum',              'type', 'identifier'),
        'enumerator': ObjType(_('enumerator'), 'enumerator')
    }

    directives = {
        # declarations
        'class': CPPClassObject,
        'struct': CPPClassObject,
        'union': CPPUnionObject,
        'function': CPPFunctionObject,
        'member': CPPMemberObject,
        'var': CPPMemberObject,
        'type': CPPTypeObject,
        'concept': CPPConceptObject,
        'enum': CPPEnumObject,
        'enum-struct': CPPEnumObject,
        'enum-class': CPPEnumObject,
        'enumerator': CPPEnumeratorObject,
        # scope control
        'namespace': CPPNamespaceObject,
        'namespace-push': CPPNamespacePushObject,
        'namespace-pop': CPPNamespacePopObject,
        # other
        'alias': CPPAliasObject
    }
    roles = {
        'any': CPPXRefRole(),
        'class': CPPXRefRole(),
        'struct': CPPXRefRole(),
        'union': CPPXRefRole(),
        'func': CPPXRefRole(fix_parens=True),
        'member': CPPXRefRole(),
        'var': CPPXRefRole(),
        'type': CPPXRefRole(),
        'concept': CPPXRefRole(),
        'enum': CPPXRefRole(),
        'enumerator': CPPXRefRole(),
        'expr': CPPExprRole(asCode=True),
        'texpr': CPPExprRole(asCode=False)
    }
    initial_data = {
        'root_symbol': Symbol(None, None, None, None, None, None),
        'names': {}  # full name for indexing -> docname
    }

    def clear_doc(self, docname: str) -> None:
        if Symbol.debug_show_tree:
            print("clear_doc:", docname)
            print("\tbefore:")
            print(self.data['root_symbol'].dump(1))
            print("\tbefore end")

        rootSymbol = self.data['root_symbol']
        rootSymbol.clear_doc(docname)

        if Symbol.debug_show_tree:
            print("\tafter:")
            print(self.data['root_symbol'].dump(1))
            print("\tafter end")
            print("clear_doc end:", docname)
        for name, nDocname in list(self.data['names'].items()):
            if nDocname == docname:
                del self.data['names'][name]

    def process_doc(self, env: BuildEnvironment, docname: str,
                    document: nodes.document) -> None:
        if Symbol.debug_show_tree:
            print("process_doc:", docname)
            print(self.data['root_symbol'].dump(0))
            print("process_doc end:", docname)

    def process_field_xref(self, pnode: pending_xref) -> None:
        pnode.attributes.update(self.env.ref_context)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        if Symbol.debug_show_tree:
            print("merge_domaindata:")
            print("\tself:")
            print(self.data['root_symbol'].dump(1))
            print("\tself end")
            print("\tother:")
            print(otherdata['root_symbol'].dump(1))
            print("\tother end")

        self.data['root_symbol'].merge_with(otherdata['root_symbol'],
                                            docnames, self.env)
        ourNames = self.data['names']
        for name, docname in otherdata['names'].items():
            if docname in docnames:
                if name in ourNames:
                    msg = __("Duplicate declaration, also defined in '%s'.\n"
                             "Name of declaration is '%s'.")
                    msg = msg % (ourNames[name], name)
                    logger.warning(msg, location=docname)
                else:
                    ourNames[name] = docname
        if Symbol.debug_show_tree:
            print("\tresult:")
            print(self.data['root_symbol'].dump(1))
            print("\tresult end")
            print("merge_domaindata end")

    def _resolve_xref_inner(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                            typ: str, target: str, node: pending_xref,
                            contnode: Element) -> Tuple[Element, str]:
        # add parens again for those that could be functions
        if typ == 'any' or typ == 'func':
            target += '()'
        parser = DefinitionParser(target, location=node, config=env.config)
        try:
            ast, isShorthand = parser.parse_xref_object()
        except DefinitionError as e:
            # as arg to stop flake8 from complaining
            def findWarning(e: Exception) -> Tuple[str, Exception]:
                if typ != 'any' and typ != 'func':
                    return target, e
                # hax on top of the paren hax to try to get correct errors
                parser2 = DefinitionParser(target[:-2],
                                           location=node,
                                           config=env.config)
                try:
                    parser2.parse_xref_object()
                except DefinitionError as e2:
                    return target[:-2], e2
                # strange, that we don't get the error now, use the original
                return target, e
            t, ex = findWarning(e)
            logger.warning('Unparseable C++ cross-reference: %r\n%s', t, ex,
                           location=node)
            return None, None
        parentKey = node.get("cpp:parent_key", None)  # type: LookupKey
        rootSymbol = self.data['root_symbol']
        if parentKey:
            parentSymbol = rootSymbol.direct_lookup(parentKey)  # type: Symbol
            if not parentSymbol:
                print("Target: ", target)
                print("ParentKey: ", parentKey.data)
                print(rootSymbol.dump(1))
            assert parentSymbol  # should be there
        else:
            parentSymbol = rootSymbol

        if isShorthand:
            assert isinstance(ast, ASTNamespace)
            ns = ast
            name = ns.nestedName
            if ns.templatePrefix:
                templateDecls = ns.templatePrefix.templates
            else:
                templateDecls = []
            # let's be conservative with the sibling lookup for now
            searchInSiblings = (not name.rooted) and len(name.names) == 1
            symbols, failReason = parentSymbol.find_name(
                name, templateDecls, typ,
                templateShorthand=True,
                matchSelf=True, recurseInAnon=True,
                searchInSiblings=searchInSiblings)
            if symbols is None:
                if typ == 'identifier':
                    if failReason == 'templateParamInQualified':
                        # this is an xref we created as part of a signature,
                        # so don't warn for names nested in template parameters
                        raise NoUri(str(name), typ)
                s = None
            else:
                # just refer to the arbitrarily first symbol
                s = symbols[0]
        else:
            assert isinstance(ast, ASTDeclaration)
            decl = ast
            name = decl.name
            s = parentSymbol.find_declaration(decl, typ,
                                              templateShorthand=True,
                                              matchSelf=True, recurseInAnon=True)
        if s is None or s.declaration is None:
            txtName = str(name)
            if txtName.startswith('std::') or txtName == 'std':
                raise NoUri(txtName, typ)
            return None, None

        if typ.startswith('cpp:'):
            typ = typ[4:]
        origTyp = typ
        if typ == 'func':
            typ = 'function'
        if typ == 'struct':
            typ = 'class'
        declTyp = s.declaration.objectType

        def checkType() -> bool:
            if typ == 'any' or typ == 'identifier':
                return True
            if declTyp == 'templateParam':
                # TODO: perhaps this should be strengthened one day
                return True
            if declTyp == 'functionParam':
                if typ == 'var' or typ == 'member':
                    return True
            objtypes = self.objtypes_for_role(typ)
            if objtypes:
                return declTyp in objtypes
            print("Type is %s (originally: %s), declType is %s" % (typ, origTyp, declTyp))
            assert False
        if not checkType():
            logger.warning("cpp:%s targets a %s (%s).",
                           origTyp, s.declaration.objectType,
                           s.get_full_nested_name(),
                           location=node)

        declaration = s.declaration
        if isShorthand:
            fullNestedName = s.get_full_nested_name()
            displayName = fullNestedName.get_display_string().lstrip(':')
        else:
            displayName = decl.get_display_string()
        docname = s.docname
        assert docname

        # the non-identifier refs are cross-references, which should be processed:
        # - fix parenthesis due to operator() and add_function_parentheses
        if typ != "identifier":
            title = contnode.pop(0).astext()
            # If it's operator(), we need to add '()' if explicit function parens
            # are requested. Then the Sphinx machinery will add another pair.
            # Also, if it's an 'any' ref that resolves to a function, we need to add
            # parens as well.
            # However, if it's a non-shorthand function ref, for a function that
            # takes no arguments, then we may need to add parens again as well.
            addParen = 0
            if not node.get('refexplicit', False) and declaration.objectType == 'function':
                if isShorthand:
                    # this is just the normal haxing for 'any' roles
                    if env.config.add_function_parentheses and typ == 'any':
                        addParen += 1
                    # and now this stuff for operator()
                    if (env.config.add_function_parentheses and typ == 'function' and
                            title.endswith('operator()')):
                        addParen += 1
                    if ((typ == 'any' or typ == 'function') and
                            title.endswith('operator') and
                            displayName.endswith('operator()')):
                        addParen += 1
                else:
                    # our job here is to essentially nullify add_function_parentheses
                    if env.config.add_function_parentheses:
                        if typ == 'any' and displayName.endswith('()'):
                            addParen += 1
                        elif typ == 'function':
                            if title.endswith('()') and not displayName.endswith('()'):
                                title = title[:-2]
                    else:
                        if displayName.endswith('()'):
                            addParen += 1
            if addParen > 0:
                title += '()' * addParen
            # and reconstruct the title again
            contnode += nodes.Text(title)
        return make_refnode(builder, fromdocname, docname,
                            declaration.get_newest_id(), contnode, displayName
                            ), declaration.objectType

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                     typ: str, target: str, node: pending_xref, contnode: Element
                     ) -> Element:
        return self._resolve_xref_inner(env, fromdocname, builder, typ,
                                        target, node, contnode)[0]

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                         target: str, node: pending_xref, contnode: Element
                         ) -> List[Tuple[str, Element]]:
        with logging.suppress_logging():
            retnode, objtype = self._resolve_xref_inner(env, fromdocname, builder,
                                                        'any', target, node, contnode)
        if retnode:
            if objtype == 'templateParam':
                return [('cpp:templateParam', retnode)]
            else:
                return [('cpp:' + self.role_for_objtype(objtype), retnode)]
        return []

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        rootSymbol = self.data['root_symbol']
        for symbol in rootSymbol.get_all_symbols():
            if symbol.declaration is None:
                continue
            assert symbol.docname
            fullNestedName = symbol.get_full_nested_name()
            name = str(fullNestedName).lstrip(':')
            dispname = fullNestedName.get_display_string().lstrip(':')
            objectType = symbol.declaration.objectType
            docname = symbol.docname
            newestId = symbol.declaration.get_newest_id()
            yield (name, dispname, objectType, docname, newestId, 1)

    def get_full_qualified_name(self, node: Element) -> str:
        target = node.get('reftarget', None)
        if target is None:
            return None
        parentKey = node.get("cpp:parent_key", None)  # type: LookupKey
        if parentKey is None or len(parentKey.data) <= 0:
            return None

        rootSymbol = self.data['root_symbol']
        parentSymbol = rootSymbol.direct_lookup(parentKey)
        parentName = parentSymbol.get_full_nested_name()
        return '::'.join([str(parentName), target])


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_domain(CPPDomain)
    app.add_config_value("cpp_index_common_prefix", [], 'env')
    app.add_config_value("cpp_id_attributes", [], 'env')
    app.add_config_value("cpp_paren_attributes", [], 'env')
    app.add_post_transform(AliasTransform)

    # debug stuff
    app.add_config_value("cpp_debug_lookup", False, '')
    app.add_config_value("cpp_debug_show_tree", False, '')

    def setDebugFlags(app):
        Symbol.debug_lookup = app.config.cpp_debug_lookup
        Symbol.debug_show_tree = app.config.cpp_debug_show_tree
    app.connect("builder-inited", setDebugFlags)

    return {
        'version': 'builtin',
        'env_version': 3,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 6 - sphinx/domains/cpp.py:

Start line: 2388, End line: 2445

```python
class ASTDeclaratorPtr(ASTDeclarator):

    def get_modifiers_id(self, version: int) -> str:
        return self.next.get_modifiers_id(version)

    def get_param_id(self, version: int) -> str:
        return self.next.get_param_id(version)

    def get_ptr_suffix_id(self, version: int) -> str:
        if version == 1:
            res = ['P']
            if self.volatile:
                res.append('V')
            if self.const:
                res.append('C')
            res.append(self.next.get_ptr_suffix_id(version))
            return ''.join(res)

        res = [self.next.get_ptr_suffix_id(version)]
        res.append('P')
        if self.volatile:
            res.append('V')
        if self.const:
            res.append('C')
        return ''.join(res)

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        # ReturnType *next, so we are part of the return type of 'next
        res = ['P']
        if self.volatile:
            res.append('V')
        if self.const:
            res.append('C')
        res.append(returnTypeId)
        return self.next.get_type_id(version, returnTypeId=''.join(res))

    def is_function_type(self) -> bool:
        return self.next.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text("*")
        for a in self.attrs:
            a.describe_signature(signode)
        if len(self.attrs) > 0 and (self.volatile or self.const):
            signode += nodes.Text(' ')

        def _add_anno(signode: TextElement, text: str) -> None:
            signode += addnodes.desc_annotation(text, text)
        if self.volatile:
            _add_anno(signode, 'volatile')
        if self.const:
            if self.volatile:
                signode += nodes.Text(' ')
            _add_anno(signode, 'const')
        if self.const or self.volatile or len(self.attrs) > 0:
            if self.next.require_space_after_declSpecs():
                signode += nodes.Text(' ')
        self.next.describe_signature(signode, mode, env, symbol)
```
### 7 - sphinx/directives/__init__.py:

Start line: 269, End line: 307

```python
from sphinx.directives.code import (  # noqa
    Highlight, CodeBlock, LiteralInclude
)
from sphinx.directives.other import (  # noqa
    TocTree, Author, VersionChange, SeeAlso,
    TabularColumns, Centered, Acks, HList, Only, Include, Class
)
from sphinx.directives.patches import (  # noqa
    Figure, Meta
)
from sphinx.domains.index import IndexDirective  # noqa

deprecated_alias('sphinx.directives',
                 {
                     'Highlight': Highlight,
                     'CodeBlock': CodeBlock,
                     'LiteralInclude': LiteralInclude,
                     'TocTree': TocTree,
                     'Author': Author,
                     'Index': IndexDirective,
                     'VersionChange': VersionChange,
                     'SeeAlso': SeeAlso,
                     'TabularColumns': TabularColumns,
                     'Centered': Centered,
                     'Acks': Acks,
                     'HList': HList,
                     'Only': Only,
                     'Include': Include,
                     'Class': Class,
                     'Figure': Figure,
                     'Meta': Meta,
                 },
                 RemovedInSphinx40Warning)

deprecated_alias('sphinx.directives',
                 {
                     'DescDirective': ObjectDescription,
                 },
                 RemovedInSphinx50Warning)
```
### 8 - sphinx/domains/cpp.py:

Start line: 6434, End line: 7163

```python
class DefinitionParser(BaseParser):
    # those without signedness and size modifiers
    _simple_fundemental_types =
    # ... other code

    def _check_template_consistency(self, nestedName: ASTNestedName,
                                    templatePrefix: ASTTemplateDeclarationPrefix,
                                    fullSpecShorthand: bool, isMember: bool = False
                                    ) -> ASTTemplateDeclarationPrefix:
        numArgs = nestedName.num_templates()
        isMemberInstantiation = False
        if not templatePrefix:
            numParams = 0
        else:
            if isMember and templatePrefix.templates is None:
                numParams = 0
                isMemberInstantiation = True
            else:
                numParams = len(templatePrefix.templates)
        if numArgs + 1 < numParams:
            self.fail("Too few template argument lists comapred to parameter"
                      " lists. Argument lists: %d, Parameter lists: %d."
                      % (numArgs, numParams))
        if numArgs > numParams:
            numExtra = numArgs - numParams
            if not fullSpecShorthand and not isMemberInstantiation:
                msg = "Too many template argument lists compared to parameter" \
                    " lists. Argument lists: %d, Parameter lists: %d," \
                    " Extra empty parameters lists prepended: %d." \
                    % (numArgs, numParams, numExtra)
                msg += " Declaration:\n\t"
                if templatePrefix:
                    msg += "%s\n\t" % templatePrefix
                msg += str(nestedName)
                self.warn(msg)

            newTemplates = []  # type: List[Union[ASTTemplateParams, ASTTemplateIntroduction]]
            for i in range(numExtra):
                newTemplates.append(ASTTemplateParams([]))
            if templatePrefix and not isMemberInstantiation:
                newTemplates.extend(templatePrefix.templates)
            templatePrefix = ASTTemplateDeclarationPrefix(newTemplates)
        return templatePrefix

    def parse_declaration(self, objectType: str, directiveType: str) -> ASTDeclaration:
        if objectType not in ('class', 'union', 'function', 'member', 'type',
                              'concept', 'enum', 'enumerator'):
            raise Exception('Internal error, unknown objectType "%s".' % objectType)
        if directiveType not in ('class', 'struct', 'union', 'function', 'member', 'var',
                                 'type', 'concept',
                                 'enum', 'enum-struct', 'enum-class', 'enumerator'):
            raise Exception('Internal error, unknown directiveType "%s".' % directiveType)
        visibility = None
        templatePrefix = None
        requiresClause = None
        trailingRequiresClause = None
        declaration = None  # type: Any

        self.skip_ws()
        if self.match(_visibility_re):
            visibility = self.matched_text

        if objectType in ('type', 'concept', 'member', 'function', 'class'):
            templatePrefix = self._parse_template_declaration_prefix(objectType)
            if objectType == 'function' and templatePrefix is not None:
                requiresClause = self._parse_requires_clause()

        if objectType == 'type':
            prevErrors = []
            pos = self.pos
            try:
                if not templatePrefix:
                    declaration = self._parse_type(named=True, outer='type')
            except DefinitionError as e:
                prevErrors.append((e, "If typedef-like declaration"))
                self.pos = pos
            pos = self.pos
            try:
                if not declaration:
                    declaration = self._parse_type_using()
            except DefinitionError as e:
                self.pos = pos
                prevErrors.append((e, "If type alias or template alias"))
                header = "Error in type declaration."
                raise self._make_multi_error(prevErrors, header)
        elif objectType == 'concept':
            declaration = self._parse_concept()
        elif objectType == 'member':
            declaration = self._parse_type_with_init(named=True, outer='member')
        elif objectType == 'function':
            declaration = self._parse_type(named=True, outer='function')
            if templatePrefix is not None:
                trailingRequiresClause = self._parse_requires_clause()
        elif objectType == 'class':
            declaration = self._parse_class()
        elif objectType == 'union':
            declaration = self._parse_union()
        elif objectType == 'enum':
            declaration = self._parse_enum()
        elif objectType == 'enumerator':
            declaration = self._parse_enumerator()
        else:
            assert False
        templatePrefix = self._check_template_consistency(declaration.name,
                                                          templatePrefix,
                                                          fullSpecShorthand=False,
                                                          isMember=objectType == 'member')
        self.skip_ws()
        semicolon = self.skip_string(';')
        return ASTDeclaration(objectType, directiveType, visibility,
                              templatePrefix, requiresClause, declaration,
                              trailingRequiresClause, semicolon)

    def parse_namespace_object(self) -> ASTNamespace:
        templatePrefix = self._parse_template_declaration_prefix(objectType="namespace")
        name = self._parse_nested_name()
        templatePrefix = self._check_template_consistency(name, templatePrefix,
                                                          fullSpecShorthand=False)
        res = ASTNamespace(name, templatePrefix)
        res.objectType = 'namespace'  # type: ignore
        return res

    def parse_xref_object(self) -> Tuple[Union[ASTNamespace, ASTDeclaration], bool]:
        pos = self.pos
        try:
            templatePrefix = self._parse_template_declaration_prefix(objectType="xref")
            name = self._parse_nested_name()
            # if there are '()' left, just skip them
            self.skip_ws()
            self.skip_string('()')
            self.assert_end()
            templatePrefix = self._check_template_consistency(name, templatePrefix,
                                                              fullSpecShorthand=True)
            res1 = ASTNamespace(name, templatePrefix)
            res1.objectType = 'xref'  # type: ignore
            return res1, True
        except DefinitionError as e1:
            try:
                self.pos = pos
                res2 = self.parse_declaration('function', 'function')
                # if there are '()' left, just skip them
                self.skip_ws()
                self.skip_string('()')
                self.assert_end()
                return res2, False
            except DefinitionError as e2:
                errs = []
                errs.append((e1, "If shorthand ref"))
                errs.append((e2, "If full function ref"))
                msg = "Error in cross-reference."
                raise self._make_multi_error(errs, msg)

    def parse_expression(self) -> Union[ASTExpression, ASTType]:
        pos = self.pos
        try:
            expr = self._parse_expression()
            self.skip_ws()
            self.assert_end()
            return expr
        except DefinitionError as exExpr:
            self.pos = pos
            try:
                typ = self._parse_type(False)
                self.skip_ws()
                self.assert_end()
                return typ
            except DefinitionError as exType:
                header = "Error when parsing (type) expression."
                errs = []
                errs.append((exExpr, "If expression"))
                errs.append((exType, "If type"))
                raise self._make_multi_error(errs, header)


def _make_phony_error_name() -> ASTNestedName:
    nne = ASTNestedNameElement(ASTIdentifier("PhonyNameDueToError"), None)
    return ASTNestedName([nne], [False], rooted=False)


class CPPObject(ObjectDescription):
    """Description of a C++ language object."""

    doc_field_types = [
        GroupedField('parameter', label=_('Parameters'),
                     names=('param', 'parameter', 'arg', 'argument'),
                     can_collapse=True),
        GroupedField('template parameter', label=_('Template Parameters'),
                     names=('tparam', 'template parameter'),
                     can_collapse=True),
        GroupedField('exceptions', label=_('Throws'), rolename='cpp:class',
                     names=('throws', 'throw', 'exception'),
                     can_collapse=True),
        Field('returnvalue', label=_('Returns'), has_arg=False,
              names=('returns', 'return')),
    ]

    option_spec = dict(ObjectDescription.option_spec)
    option_spec['tparam-line-spec'] = directives.flag

    def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
        assert ast.objectType == 'enumerator'
        # find the parent, if it exists && is an enum
        #                     && it's unscoped,
        #                  then add the name to the parent scope
        symbol = ast.symbol
        assert symbol
        assert symbol.identOrOp is not None
        assert symbol.templateParams is None
        assert symbol.templateArgs is None
        parentSymbol = symbol.parent
        assert parentSymbol
        if parentSymbol.parent is None:
            # TODO: we could warn, but it is somewhat equivalent to unscoped
            # enums, without the enum
            return  # no parent
        parentDecl = parentSymbol.declaration
        if parentDecl is None:
            # the parent is not explicitly declared
            # TODO: we could warn, but it could be a style to just assume
            # enumerator parents to be scoped
            return
        if parentDecl.objectType != 'enum':
            # TODO: maybe issue a warning, enumerators in non-enums is weird,
            # but it is somewhat equivalent to unscoped enums, without the enum
            return
        if parentDecl.directiveType != 'enum':
            return

        targetSymbol = parentSymbol.parent
        s = targetSymbol.find_identifier(symbol.identOrOp, matchSelf=False, recurseInAnon=True,
                                         searchInSiblings=False)
        if s is not None:
            # something is already declared with that name
            return
        declClone = symbol.declaration.clone()
        declClone.enumeratorScopedSymbol = symbol
        Symbol(parent=targetSymbol, identOrOp=symbol.identOrOp,
               templateParams=None, templateArgs=None,
               declaration=declClone,
               docname=self.env.docname)

    def add_target_and_index(self, ast: ASTDeclaration, sig: str,
                             signode: TextElement) -> None:
        # general note: name must be lstrip(':')'ed, to remove "::"
        ids = []
        for i in range(1, _max_id + 1):
            try:
                id = ast.get_id(version=i)
                ids.append(id)
            except NoOldIdError:
                assert i < _max_id
        # let's keep the newest first
        ids = list(reversed(ids))
        newestId = ids[0]
        assert newestId  # shouldn't be None
        if not re.compile(r'^[a-zA-Z0-9_]*$').match(newestId):
            logger.warning('Index id generation for C++ object "%s" failed, please '
                           'report as bug (id=%s).', ast, newestId,
                           location=self.get_source_info())

        name = ast.symbol.get_full_nested_name().get_display_string().lstrip(':')
        # Add index entry, but not if it's a declaration inside a concept
        isInConcept = False
        s = ast.symbol.parent
        while s is not None:
            decl = s.declaration
            s = s.parent
            if decl is None:
                continue
            if decl.objectType == 'concept':
                isInConcept = True
                break
        if not isInConcept:
            strippedName = name
            for prefix in self.env.config.cpp_index_common_prefix:
                if name.startswith(prefix):
                    strippedName = strippedName[len(prefix):]
                    break
            indexText = self.get_index_text(strippedName)
            self.indexnode['entries'].append(('single', indexText, newestId, '', None))

        if newestId not in self.state.document.ids:
            # if the name is not unique, the first one will win
            names = self.env.domaindata['cpp']['names']
            if name not in names:
                names[name] = ast.symbol.docname
            # always add the newest id
            assert newestId
            signode['ids'].append(newestId)
            # only add compatibility ids when there are no conflicts
            for id in ids[1:]:
                if not id:  # is None when the element didn't exist in that version
                    continue
                if id not in self.state.document.ids:
                    signode['ids'].append(id)
            self.state.document.note_explicit_target(signode)

    @property
    def object_type(self) -> str:
        raise NotImplementedError()

    @property
    def display_object_type(self) -> str:
        return self.object_type

    def get_index_text(self, name: str) -> str:
        return _('%s (C++ %s)') % (name, self.display_object_type)

    def parse_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_declaration(self.object_type, self.objtype)

    def describe_signature(self, signode: desc_signature,
                           ast: ASTDeclaration, options: Dict) -> None:
        ast.describe_signature(signode, 'lastIsName', self.env, options)

    def run(self) -> List[Node]:
        env = self.state.document.settings.env  # from ObjectDescription.run
        if 'cpp:parent_symbol' not in env.temp_data:
            root = env.domaindata['cpp']['root_symbol']
            env.temp_data['cpp:parent_symbol'] = root
            env.ref_context['cpp:parent_key'] = root.get_lookup_key()

        # The lookup keys assume that no nested scopes exists inside overloaded functions.
        # (see also #5191)
        # Example:
        # .. cpp:function:: void f(int)
        # .. cpp:function:: void f(double)
        #
        #    .. cpp:function:: void g()
        #
        #       :cpp:any:`boom`
        #
        # So we disallow any signatures inside functions.
        parentSymbol = env.temp_data['cpp:parent_symbol']
        parentDecl = parentSymbol.declaration
        if parentDecl is not None and parentDecl.objectType == 'function':
            logger.warning("C++ declarations inside functions are not supported." +
                           " Parent function is " +
                           str(parentSymbol.get_full_nested_name()),
                           location=self.get_source_info())
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            env.temp_data['cpp:last_symbol'] = symbol
            return []
        # When multiple declarations are made in the same directive
        # they need to know about each other to provide symbol lookup for function parameters.
        # We use last_symbol to store the latest added declaration in a directive.
        env.temp_data['cpp:last_symbol'] = None
        return super().run()

    def handle_signature(self, sig: str, signode: desc_signature) -> ASTDeclaration:
        parentSymbol = self.env.temp_data['cpp:parent_symbol']

        parser = DefinitionParser(sig, location=signode, config=self.env.config)
        try:
            ast = self.parse_definition(parser)
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=signode)
            # It is easier to assume some phony name than handling the error in
            # the possibly inner declarations.
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            self.env.temp_data['cpp:last_symbol'] = symbol
            raise ValueError

        try:
            symbol = parentSymbol.add_declaration(ast, docname=self.env.docname)
            # append the new declaration to the sibling list
            assert symbol.siblingAbove is None
            assert symbol.siblingBelow is None
            symbol.siblingAbove = self.env.temp_data['cpp:last_symbol']
            if symbol.siblingAbove is not None:
                assert symbol.siblingAbove.siblingBelow is None
                symbol.siblingAbove.siblingBelow = symbol
            self.env.temp_data['cpp:last_symbol'] = symbol
        except _DuplicateSymbolError as e:
            # Assume we are actually in the old symbol,
            # instead of the newly created duplicate.
            self.env.temp_data['cpp:last_symbol'] = e.symbol
            logger.warning("Duplicate declaration, %s", sig, location=signode)

        if ast.objectType == 'enumerator':
            self._add_enumerator_to_parent(ast)

        # note: handle_signature may be called multiple time per directive,
        # if it has multiple signatures, so don't mess with the original options.
        options = dict(self.options)
        options['tparam-line-spec'] = 'tparam-line-spec' in self.options
        self.describe_signature(signode, ast, options)
        return ast

    def before_content(self) -> None:
        lastSymbol = self.env.temp_data['cpp:last_symbol']  # type: Symbol
        assert lastSymbol
        self.oldParentSymbol = self.env.temp_data['cpp:parent_symbol']
        self.oldParentKey = self.env.ref_context['cpp:parent_key']  # type: LookupKey
        self.env.temp_data['cpp:parent_symbol'] = lastSymbol
        self.env.ref_context['cpp:parent_key'] = lastSymbol.get_lookup_key()

    def after_content(self) -> None:
        self.env.temp_data['cpp:parent_symbol'] = self.oldParentSymbol
        self.env.ref_context['cpp:parent_key'] = self.oldParentKey


class CPPTypeObject(CPPObject):
    object_type = 'type'


class CPPConceptObject(CPPObject):
    object_type = 'concept'


class CPPMemberObject(CPPObject):
    object_type = 'member'


class CPPFunctionObject(CPPObject):
    object_type = 'function'


class CPPClassObject(CPPObject):
    object_type = 'class'

    @property
    def display_object_type(self) -> str:
        # the distinction between class and struct is only cosmetic
        assert self.objtype in ('class', 'struct')
        return self.objtype


class CPPUnionObject(CPPObject):
    object_type = 'union'


class CPPEnumObject(CPPObject):
    object_type = 'enum'


class CPPEnumeratorObject(CPPObject):
    object_type = 'enumerator'


class CPPNamespaceObject(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting stuff in
    namespace foo.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        rootSymbol = self.env.domaindata['cpp']['root_symbol']
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            symbol = rootSymbol
            stack = []  # type: List[Symbol]
        else:
            parser = DefinitionParser(self.arguments[0],
                                      location=self.get_source_info(),
                                      config=self.config)
            try:
                ast = parser.parse_namespace_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=self.get_source_info())
                name = _make_phony_error_name()
                ast = ASTNamespace(name, None)
            symbol = rootSymbol.add_name(ast.nestedName, ast.templatePrefix)
            stack = [symbol]
        self.env.temp_data['cpp:parent_symbol'] = symbol
        self.env.temp_data['cpp:namespace_stack'] = stack
        self.env.ref_context['cpp:parent_key'] = symbol.get_lookup_key()
        return []


class CPPNamespacePushObject(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            return []
        parser = DefinitionParser(self.arguments[0],
                                  location=self.get_source_info(),
                                  config=self.config)
        try:
            ast = parser.parse_namespace_object()
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=self.get_source_info())
            name = _make_phony_error_name()
            ast = ASTNamespace(name, None)
        oldParent = self.env.temp_data.get('cpp:parent_symbol', None)
        if not oldParent:
            oldParent = self.env.domaindata['cpp']['root_symbol']
        symbol = oldParent.add_name(ast.nestedName, ast.templatePrefix)
        stack = self.env.temp_data.get('cpp:namespace_stack', [])
        stack.append(symbol)
        self.env.temp_data['cpp:parent_symbol'] = symbol
        self.env.temp_data['cpp:namespace_stack'] = stack
        self.env.ref_context['cpp:parent_key'] = symbol.get_lookup_key()
        return []


class CPPNamespacePopObject(SphinxDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        stack = self.env.temp_data.get('cpp:namespace_stack', None)
        if not stack or len(stack) == 0:
            logger.warning("C++ namespace pop on empty stack. Defaulting to gobal scope.",
                           location=self.get_source_info())
            stack = []
        else:
            stack.pop()
        if len(stack) > 0:
            symbol = stack[-1]
        else:
            symbol = self.env.domaindata['cpp']['root_symbol']
        self.env.temp_data['cpp:parent_symbol'] = symbol
        self.env.temp_data['cpp:namespace_stack'] = stack
        self.env.ref_context['cpp:parent_key'] = symbol.get_lookup_key()
        return []


class AliasNode(nodes.Element):
    def __init__(self, sig: str, env: "BuildEnvironment" = None,
                 parentKey: LookupKey = None) -> None:
        super().__init__()
        self.sig = sig
        if env is not None:
            if 'cpp:parent_symbol' not in env.temp_data:
                root = env.domaindata['cpp']['root_symbol']
                env.temp_data['cpp:parent_symbol'] = root
            self.parentKey = env.temp_data['cpp:parent_symbol'].get_lookup_key()
        else:
            assert parentKey is not None
            self.parentKey = parentKey

    def copy(self: T) -> T:
        return self.__class__(self.sig, env=None, parentKey=self.parentKey)  # type: ignore


class AliasTransform(SphinxTransform):
    default_priority = ReferencesResolver.default_priority - 1

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.traverse(AliasNode):
            sig = node.sig
            parentKey = node.parentKey
            try:
                parser = DefinitionParser(sig, location=node,
                                          config=self.env.config)
                ast, isShorthand = parser.parse_xref_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=node)
                ast, isShorthand = None, None

            if ast is None:
                # could not be parsed, so stop here
                signode = addnodes.desc_signature(sig, '')
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                node.replace_self(signode)
                continue

            rootSymbol = self.env.domains['cpp'].data['root_symbol']  # type: Symbol
            parentSymbol = rootSymbol.direct_lookup(parentKey)  # type: Symbol
            if not parentSymbol:
                print("Target: ", sig)
                print("ParentKey: ", parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol  # should be there

            symbols = []  # type: List[Symbol]
            if isShorthand:
                assert isinstance(ast, ASTNamespace)
                ns = ast
                name = ns.nestedName
                if ns.templatePrefix:
                    templateDecls = ns.templatePrefix.templates
                else:
                    templateDecls = []
                symbols, failReason = parentSymbol.find_name(
                    nestedName=name,
                    templateDecls=templateDecls,
                    typ='any',
                    templateShorthand=True,
                    matchSelf=True, recurseInAnon=True,
                    searchInSiblings=False)
                if symbols is None:
                    symbols = []
            else:
                assert isinstance(ast, ASTDeclaration)
                decl = ast
                name = decl.name
                s = parentSymbol.find_declaration(decl, 'any',
                                                  templateShorthand=True,
                                                  matchSelf=True, recurseInAnon=True)
                if s is not None:
                    symbols.append(s)

            symbols = [s for s in symbols if s.declaration is not None]

            if len(symbols) == 0:
                signode = addnodes.desc_signature(sig, '')
                node.append(signode)
                signode.clear()
                signode += addnodes.desc_name(sig, sig)

                logger.warning("Could not find C++ declaration for alias '%s'." % ast,
                               location=node)
                node.replace_self(signode)
            else:
                nodes = []
                options = dict()
                options['tparam-line-spec'] = False
                for s in symbols:
                    signode = addnodes.desc_signature(sig, '')
                    nodes.append(signode)
                    s.declaration.describe_signature(signode, 'markName', self.env, options)
                node.replace_self(nodes)


class CPPAliasObject(ObjectDescription):
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        """
        On purpose this doesn't call the ObjectDescription version, but is based on it.
        Each alias signature may expand into multiple real signatures (an overload set).
        The code is therefore based on the ObjectDescription version.
        """
        if ':' in self.name:
            self.domain, self.objtype = self.name.split(':', 1)
        else:
            self.domain, self.objtype = '', self.name

        node = addnodes.desc()
        node.document = self.state.document
        node['domain'] = self.domain
        # 'desctype' is a backwards compatible attribute
        node['objtype'] = node['desctype'] = self.objtype
        node['noindex'] = True

        self.names = []  # type: List[str]
        signatures = self.get_signatures()
        for i, sig in enumerate(signatures):
            node.append(AliasNode(sig, env=self.env))

        contentnode = addnodes.desc_content()
        node.append(contentnode)
        self.before_content()
        self.state.nested_parse(self.content, self.content_offset, contentnode)
        self.env.temp_data['object'] = None
        self.after_content()
        return [node]


class CPPXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element, has_explicit_title: bool,
                     title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)

        if not has_explicit_title:
            # major hax: replace anon names via simple string manipulation.
            # Can this actually fail?
            title = anon_identifier_re.sub("[anonymous]", str(title))

        if refnode['reftype'] == 'any':
            # Assume the removal part of fix_parens for :any: refs.
            # The addition part is done with the reference is resolved.
            if not has_explicit_title and title.endswith('()'):
                title = title[:-2]
            if target.endswith('()'):
                target = target[:-2]
        # TODO: should this really be here?
        if not has_explicit_title:
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[:1] == '~':
                title = title[1:]
                dcolon = title.rfind('::')
                if dcolon != -1:
                    title = title[dcolon + 2:]
        return title, target


class CPPExprRole(SphinxRole):
    def __init__(self, asCode: bool) -> None:
        super().__init__()
        if asCode:
            # render the expression as inline code
            self.class_type = 'cpp-expr'
            self.node_type = nodes.literal  # type: Type[TextElement]
        else:
            # render the expression as inline text
            self.class_type = 'cpp-texpr'
            self.node_type = nodes.inline

    def run(self) -> Tuple[List[Node], List[system_message]]:
        text = self.text.replace('\n', ' ')
        parser = DefinitionParser(text,
                                  location=self.get_source_info(),
                                  config=self.config)
        # attempt to mimic XRefRole classes, except that...
        classes = ['xref', 'cpp', self.class_type]
        try:
            ast = parser.parse_expression()
        except DefinitionError as ex:
            logger.warning('Unparseable C++ expression: %r\n%s', text, ex,
                           location=self.get_source_info())
            # see below
            return [self.node_type(text, text, classes=classes)], []
        parentSymbol = self.env.temp_data.get('cpp:parent_symbol', None)
        if parentSymbol is None:
            parentSymbol = self.env.domaindata['cpp']['root_symbol']
        # ...most if not all of these classes should really apply to the individual references,
        # not the container node
        signode = self.node_type(classes=classes)
        ast.describe_signature(signode, 'markType', self.env, parentSymbol)
        return [signode], []
```
### 9 - sphinx/pycode/ast.py:

Start line: 132, End line: 206

```python
class _UnparseVisitor(ast.NodeVisitor):

    def visit_Attribute(self, node: ast.Attribute) -> str:
        return "%s.%s" % (self.visit(node.value), node.attr)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        return " ".join(self.visit(e) for e in [node.left, node.op, node.right])

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = " %s " % self.visit(node.op)
        return op.join(self.visit(e) for e in node.values)

    def visit_Call(self, node: ast.Call) -> str:
        args = ([self.visit(e) for e in node.args] +
                ["%s=%s" % (k.arg, self.visit(k.value)) for k in node.keywords])
        return "%s(%s)" % (self.visit(node.func), ", ".join(args))

    def visit_Dict(self, node: ast.Dict) -> str:
        keys = (self.visit(k) for k in node.keys)
        values = (self.visit(v) for v in node.values)
        items = (k + ": " + v for k, v in zip(keys, values))
        return "{" + ", ".join(items) + "}"

    def visit_Index(self, node: ast.Index) -> str:
        return self.visit(node.value)

    def visit_Lambda(self, node: ast.Lambda) -> str:
        return "lambda %s: ..." % self.visit(node.args)

    def visit_List(self, node: ast.List) -> str:
        return "[" + ", ".join(self.visit(e) for e in node.elts) + "]"

    def visit_Name(self, node: ast.Name) -> str:
        return node.id

    def visit_Set(self, node: ast.Set) -> str:
        return "{" + ", ".join(self.visit(e) for e in node.elts) + "}"

    def visit_Subscript(self, node: ast.Subscript) -> str:
        return "%s[%s]" % (self.visit(node.value), self.visit(node.slice))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        return "%s %s" % (self.visit(node.op), self.visit(node.operand))

    def visit_Tuple(self, node: ast.Tuple) -> str:
        if node.elts:
            return ", ".join(self.visit(e) for e in node.elts)
        else:
            return "()"

    if sys.version_info >= (3, 6):
        def visit_Constant(self, node: ast.Constant) -> str:
            if node.value is Ellipsis:
                return "..."
            else:
                return repr(node.value)

    if sys.version_info < (3, 8):
        # these ast nodes were deprecated in python 3.8
        def visit_Bytes(self, node: ast.Bytes) -> str:
            return repr(node.s)

        def visit_Ellipsis(self, node: ast.Ellipsis) -> str:
            return "..."

        def visit_NameConstant(self, node: ast.NameConstant) -> str:
            return repr(node.value)

        def visit_Num(self, node: ast.Num) -> str:
            return repr(node.n)

        def visit_Str(self, node: ast.Str) -> str:
            return repr(node.s)

    def generic_visit(self, node):
        raise NotImplementedError('Unable to parse %s object' % type(node).__name__)
```
### 10 - sphinx/util/cfamily.py:

Start line: 11, End line: 80

```python
import re
import warnings
from copy import deepcopy
from typing import (
    Any, Callable, List, Match, Pattern, Tuple, Union
)

from docutils import nodes
from docutils.nodes import TextElement

from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx40Warning
from sphinx.util import logging

logger = logging.getLogger(__name__)

StringifyTransform = Callable[[Any], str]


_whitespace_re = re.compile(r'(?u)\s+')
anon_identifier_re = re.compile(r'(@[a-zA-Z0-9_])[a-zA-Z0-9_]*\b')
identifier_re = re.compile(r'''(?x)
    (   # This 'extends' _anon_identifier_re with the ordinary identifiers,
        # make sure they are in sync.
        (~?\b[a-zA-Z_])  # ordinary identifiers
    |   (@[a-zA-Z0-9_])  # our extension for names of anonymous entities
    )
    [a-zA-Z0-9_]*\b
''')
integer_literal_re = re.compile(r'[1-9][0-9]*')
octal_literal_re = re.compile(r'0[0-7]*')
hex_literal_re = re.compile(r'0[xX][0-9a-fA-F][0-9a-fA-F]*')
binary_literal_re = re.compile(r'0[bB][01][01]*')
integers_literal_suffix_re = re.compile(r'''(?x)
    # unsigned and/or (long) long, in any order, but at least one of them
    (
        ([uU]    ([lL]  |  (ll)  |  (LL))?)
        |
        (([lL]  |  (ll)  |  (LL))    [uU]?)
    )\b
    # the ending word boundary is important for distinguishing
    # between suffixes and UDLs in C++
''')
float_literal_re = re.compile(r'''(?x)
    [+-]?(
    # decimal
      ([0-9]+[eE][+-]?[0-9]+)
    | ([0-9]*\.[0-9]+([eE][+-]?[0-9]+)?)
    | ([0-9]+\.([eE][+-]?[0-9]+)?)
    # hex
    | (0[xX][0-9a-fA-F]+[pP][+-]?[0-9a-fA-F]+)
    | (0[xX][0-9a-fA-F]*\.[0-9a-fA-F]+([pP][+-]?[0-9a-fA-F]+)?)
    | (0[xX][0-9a-fA-F]+\.([pP][+-]?[0-9a-fA-F]+)?)
    )
''')
float_literal_suffix_re = re.compile(r'[fFlL]\b')
# the ending word boundary is important for distinguishing between suffixes and UDLs in C++
char_literal_re = re.compile(r'''(?x)
    ((?:u8)|u|U|L)?
    '(
      (?:[^\\'])
    | (\\(
        (?:['"?\\abfnrtv])
      | (?:[0-7]{1,3})
      | (?:x[0-9a-fA-F]{2})
      | (?:u[0-9a-fA-F]{4})
      | (?:U[0-9a-fA-F]{8})
      ))
    )'
''')
```
### 11 - sphinx/util/cfamily.py:

Start line: 293, End line: 370

```python
class BaseParser:

    def warn(self, msg: str) -> None:
        logger.warning(msg, location=self.location)

    def match(self, regex: Pattern) -> bool:
        match = regex.match(self.definition, self.pos)
        if match is not None:
            self._previous_state = (self.pos, self.last_match)
            self.pos = match.end()
            self.last_match = match
            return True
        return False

    def skip_string(self, string: str) -> bool:
        strlen = len(string)
        if self.definition[self.pos:self.pos + strlen] == string:
            self.pos += strlen
            return True
        return False

    def skip_word(self, word: str) -> bool:
        return self.match(re.compile(r'\b%s\b' % re.escape(word)))

    def skip_ws(self) -> bool:
        return self.match(_whitespace_re)

    def skip_word_and_ws(self, word: str) -> bool:
        if self.skip_word(word):
            self.skip_ws()
            return True
        return False

    def skip_string_and_ws(self, string: str) -> bool:
        if self.skip_string(string):
            self.skip_ws()
            return True
        return False

    @property
    def eof(self) -> bool:
        return self.pos >= self.end

    @property
    def current_char(self) -> str:
        try:
            return self.definition[self.pos]
        except IndexError:
            return 'EOF'

    @property
    def matched_text(self) -> str:
        if self.last_match is not None:
            return self.last_match.group()
        else:
            return None

    def read_rest(self) -> str:
        rv = self.definition[self.pos:]
        self.pos = self.end
        return rv

    def assert_end(self, *, allowSemicolon: bool = False) -> None:
        self.skip_ws()
        if allowSemicolon:
            if not self.eof and self.definition[self.pos:] != ';':
                self.fail('Expected end of definition or ;.')
        else:
            if not self.eof:
                self.fail('Expected end of definition.')

    ################################################################################

    @property
    def id_attributes(self):
        raise NotImplementedError

    @property
    def paren_attributes(self):
        raise NotImplementedError
```
### 16 - sphinx/domains/c.py:

Start line: 2699, End line: 3478

```python
class DefinitionParser(BaseParser):
    # those without signedness and size modifiers
    _simple_fundamental_types =
    # ... other code

    def _parse_declarator(self, named: Union[bool, str], paramMode: str,
                          typed: bool = True) -> ASTDeclarator:
        # 'typed' here means 'parse return type stuff'
        if paramMode not in ('type', 'function'):
            raise Exception(
                "Internal error, unknown paramMode '%s'." % paramMode)
        prevErrors = []
        self.skip_ws()
        if typed and self.skip_string('*'):
            self.skip_ws()
            restrict = False
            volatile = False
            const = False
            attrs = []
            while 1:
                if not restrict:
                    restrict = self.skip_word_and_ws('restrict')
                    if restrict:
                        continue
                if not volatile:
                    volatile = self.skip_word_and_ws('volatile')
                    if volatile:
                        continue
                if not const:
                    const = self.skip_word_and_ws('const')
                    if const:
                        continue
                attr = self._parse_attribute()
                if attr is not None:
                    attrs.append(attr)
                    continue
                break
            next = self._parse_declarator(named, paramMode, typed)
            return ASTDeclaratorPtr(next=next,
                                    restrict=restrict, volatile=volatile, const=const,
                                    attrs=attrs)
        if typed and self.current_char == '(':  # note: peeking, not skipping
            # maybe this is the beginning of params, try that first,
            # otherwise assume it's noptr->declarator > ( ptr-declarator )
            pos = self.pos
            try:
                # assume this is params
                res = self._parse_declarator_name_suffix(named, paramMode,
                                                         typed)
                return res
            except DefinitionError as exParamQual:
                msg = "If declarator-id with parameters"
                if paramMode == 'function':
                    msg += " (e.g., 'void f(int arg)')"
                prevErrors.append((exParamQual, msg))
                self.pos = pos
                try:
                    assert self.current_char == '('
                    self.skip_string('(')
                    # TODO: hmm, if there is a name, it must be in inner, right?
                    # TODO: hmm, if there must be parameters, they must b
                    # inside, right?
                    inner = self._parse_declarator(named, paramMode, typed)
                    if not self.skip_string(')'):
                        self.fail("Expected ')' in \"( ptr-declarator )\"")
                    next = self._parse_declarator(named=False,
                                                  paramMode="type",
                                                  typed=typed)
                    return ASTDeclaratorParen(inner=inner, next=next)
                except DefinitionError as exNoPtrParen:
                    self.pos = pos
                    msg = "If parenthesis in noptr-declarator"
                    if paramMode == 'function':
                        msg += " (e.g., 'void (*f(int arg))(double)')"
                    prevErrors.append((exNoPtrParen, msg))
                    header = "Error in declarator"
                    raise self._make_multi_error(prevErrors, header)
        pos = self.pos
        try:
            return self._parse_declarator_name_suffix(named, paramMode, typed)
        except DefinitionError as e:
            self.pos = pos
            prevErrors.append((e, "If declarator-id"))
            header = "Error in declarator or parameters"
            raise self._make_multi_error(prevErrors, header)

    def _parse_initializer(self, outer: str = None, allowFallback: bool = True
                           ) -> ASTInitializer:
        self.skip_ws()
        if outer == 'member' and False:  # TODO
            bracedInit = self._parse_braced_init_list()
            if bracedInit is not None:
                return ASTInitializer(bracedInit, hasAssign=False)

        if not self.skip_string('='):
            return None

        bracedInit = self._parse_braced_init_list()
        if bracedInit is not None:
            return ASTInitializer(bracedInit)

        if outer == 'member':
            fallbackEnd = []  # type: List[str]
        elif outer is None:  # function parameter
            fallbackEnd = [',', ')']
        else:
            self.fail("Internal error, initializer for outer '%s' not "
                      "implemented." % outer)

        def parser():
            return self._parse_assignment_expression()

        value = self._parse_expression_fallback(fallbackEnd, parser, allow=allowFallback)
        return ASTInitializer(value)

    def _parse_type(self, named: Union[bool, str], outer: str = None) -> ASTType:
        """
        named=False|'maybe'|True: 'maybe' is e.g., for function objects which
        doesn't need to name the arguments
        """
        if outer:  # always named
            if outer not in ('type', 'member', 'function'):
                raise Exception('Internal error, unknown outer "%s".' % outer)
            assert named

        if outer == 'type':
            # We allow type objects to just be a name.
            prevErrors = []
            startPos = self.pos
            # first try without the type
            try:
                declSpecs = self._parse_decl_specs(outer=outer, typed=False)
                decl = self._parse_declarator(named=True, paramMode=outer,
                                              typed=False)
                self.assert_end(allowSemicolon=True)
            except DefinitionError as exUntyped:
                desc = "If just a name"
                prevErrors.append((exUntyped, desc))
                self.pos = startPos
                try:
                    declSpecs = self._parse_decl_specs(outer=outer)
                    decl = self._parse_declarator(named=True, paramMode=outer)
                except DefinitionError as exTyped:
                    self.pos = startPos
                    desc = "If typedef-like declaration"
                    prevErrors.append((exTyped, desc))
                    # Retain the else branch for easier debugging.
                    # TODO: it would be nice to save the previous stacktrace
                    #       and output it here.
                    if True:
                        header = "Type must be either just a name or a "
                        header += "typedef-like declaration."
                        raise self._make_multi_error(prevErrors, header)
                    else:
                        # For testing purposes.
                        # do it again to get the proper traceback (how do you
                        # reliably save a traceback when an exception is
                        # constructed?)
                        self.pos = startPos
                        typed = True
                        declSpecs = self._parse_decl_specs(outer=outer, typed=typed)
                        decl = self._parse_declarator(named=True, paramMode=outer,
                                                      typed=typed)
        elif outer == 'function':
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=True, paramMode=outer)
        else:
            paramMode = 'type'
            if outer == 'member':  # i.e., member
                named = True
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=named, paramMode=paramMode)
        return ASTType(declSpecs, decl)

    def _parse_type_with_init(self, named: Union[bool, str], outer: str) -> ASTTypeWithInit:
        if outer:
            assert outer in ('type', 'member', 'function')
        type = self._parse_type(outer=outer, named=named)
        init = self._parse_initializer(outer=outer)
        return ASTTypeWithInit(type, init)

    def _parse_macro(self) -> ASTMacro:
        self.skip_ws()
        ident = self._parse_nested_name()
        if ident is None:
            self.fail("Expected identifier in macro definition.")
        self.skip_ws()
        if not self.skip_string_and_ws('('):
            return ASTMacro(ident, None)
        if self.skip_string(')'):
            return ASTMacro(ident, [])
        args = []
        while 1:
            self.skip_ws()
            if self.skip_string('...'):
                args.append(ASTMacroParameter(None, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in macro parameters.')
                break
            if not self.match(identifier_re):
                self.fail("Expected identifier in macro parameters.")
            nn = ASTNestedName([ASTIdentifier(self.matched_text)], rooted=False)
            arg = ASTMacroParameter(nn)
            args.append(arg)
            self.skip_ws()
            if self.skip_string_and_ws(','):
                continue
            elif self.skip_string_and_ws(')'):
                break
            else:
                self.fail("Expected identifier, ')', or ',' in macro parameter list.")
        return ASTMacro(ident, args)

    def _parse_struct(self) -> ASTStruct:
        name = self._parse_nested_name()
        return ASTStruct(name)

    def _parse_union(self) -> ASTUnion:
        name = self._parse_nested_name()
        return ASTUnion(name)

    def _parse_enum(self) -> ASTEnum:
        name = self._parse_nested_name()
        return ASTEnum(name)

    def _parse_enumerator(self) -> ASTEnumerator:
        name = self._parse_nested_name()
        self.skip_ws()
        init = None
        if self.skip_string('='):
            self.skip_ws()

            def parser() -> ASTExpression:
                return self._parse_constant_expression()

            initVal = self._parse_expression_fallback([], parser)
            init = ASTInitializer(initVal)
        return ASTEnumerator(name, init)

    def parse_declaration(self, objectType: str, directiveType: str) -> ASTDeclaration:
        if objectType not in ('function', 'member',
                              'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown objectType "%s".' % objectType)
        if directiveType not in ('function', 'member', 'var',
                                 'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown directiveType "%s".' % directiveType)

        declaration = None  # type: Any
        if objectType == 'member':
            declaration = self._parse_type_with_init(named=True, outer='member')
        elif objectType == 'function':
            declaration = self._parse_type(named=True, outer='function')
        elif objectType == 'macro':
            declaration = self._parse_macro()
        elif objectType == 'struct':
            declaration = self._parse_struct()
        elif objectType == 'union':
            declaration = self._parse_union()
        elif objectType == 'enum':
            declaration = self._parse_enum()
        elif objectType == 'enumerator':
            declaration = self._parse_enumerator()
        elif objectType == 'type':
            declaration = self._parse_type(named=True, outer='type')
        else:
            assert False
        if objectType != 'macro':
            self.skip_ws()
            semicolon = self.skip_string(';')
        else:
            semicolon = False
        return ASTDeclaration(objectType, directiveType, declaration, semicolon)

    def parse_namespace_object(self) -> ASTNestedName:
        return self._parse_nested_name()

    def parse_xref_object(self) -> ASTNestedName:
        name = self._parse_nested_name()
        # if there are '()' left, just skip them
        self.skip_ws()
        self.skip_string('()')
        self.assert_end()
        return name

    def parse_expression(self) -> Union[ASTExpression, ASTType]:
        pos = self.pos
        res = None  # type: Union[ASTExpression, ASTType]
        try:
            res = self._parse_expression()
            self.skip_ws()
            self.assert_end()
        except DefinitionError as exExpr:
            self.pos = pos
            try:
                res = self._parse_type(False)
                self.skip_ws()
                self.assert_end()
            except DefinitionError as exType:
                header = "Error when parsing (type) expression."
                errs = []
                errs.append((exExpr, "If expression"))
                errs.append((exType, "If type"))
                raise self._make_multi_error(errs, header)
        return res


def _make_phony_error_name() -> ASTNestedName:
    return ASTNestedName([ASTIdentifier("PhonyNameDueToError")], rooted=False)


class CObject(ObjectDescription):
    """
    Description of a C language object.
    """

    doc_field_types = [
        TypedField('parameter', label=_('Parameters'),
                   names=('param', 'parameter', 'arg', 'argument'),
                   typerolename='type', typenames=('type',)),
        Field('returnvalue', label=_('Returns'), has_arg=False,
              names=('returns', 'return')),
        Field('returntype', label=_('Return type'), has_arg=False,
              names=('rtype',)),
    ]

    def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
        assert ast.objectType == 'enumerator'
        # find the parent, if it exists && is an enum
        #                  then add the name to the parent scope
        symbol = ast.symbol
        assert symbol
        assert symbol.ident is not None
        parentSymbol = symbol.parent
        assert parentSymbol
        if parentSymbol.parent is None:
            # TODO: we could warn, but it is somewhat equivalent to
            # enumeratorss, without the enum
            return  # no parent
        parentDecl = parentSymbol.declaration
        if parentDecl is None:
            # the parent is not explicitly declared
            # TODO: we could warn, but?
            return
        if parentDecl.objectType != 'enum':
            # TODO: maybe issue a warning, enumerators in non-enums is weird,
            # but it is somewhat equivalent to enumeratorss, without the enum
            return
        if parentDecl.directiveType != 'enum':
            return

        targetSymbol = parentSymbol.parent
        s = targetSymbol.find_identifier(symbol.ident, matchSelf=False, recurseInAnon=True,
                                         searchInSiblings=False)
        if s is not None:
            # something is already declared with that name
            return
        declClone = symbol.declaration.clone()
        declClone.enumeratorScopedSymbol = symbol
        Symbol(parent=targetSymbol, ident=symbol.ident,
               declaration=declClone,
               docname=self.env.docname)

    def add_target_and_index(self, ast: ASTDeclaration, sig: str,
                             signode: TextElement) -> None:
        ids = []
        for i in range(1, _max_id + 1):
            try:
                id = ast.get_id(version=i)
                ids.append(id)
            except NoOldIdError:
                assert i < _max_id
        # let's keep the newest first
        ids = list(reversed(ids))
        newestId = ids[0]
        assert newestId  # shouldn't be None

        name = ast.symbol.get_full_nested_name().get_display_string().lstrip('.')
        if newestId not in self.state.document.ids:
            # always add the newest id
            assert newestId
            signode['ids'].append(newestId)
            # only add compatibility ids when there are no conflicts
            for id in ids[1:]:
                if not id:  # is None when the element didn't exist in that version
                    continue
                if id not in self.state.document.ids:
                    signode['ids'].append(id)

            self.state.document.note_explicit_target(signode)

            domain = cast(CDomain, self.env.get_domain('c'))
            domain.note_object(name, self.objtype, newestId)

        indexText = self.get_index_text(name)
        self.indexnode['entries'].append(('single', indexText, newestId, '', None))

    @property
    def object_type(self) -> str:
        raise NotImplementedError()

    @property
    def display_object_type(self) -> str:
        return self.object_type

    def get_index_text(self, name: str) -> str:
        return _('%s (C %s)') % (name, self.display_object_type)

    def parse_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_declaration(self.object_type, self.objtype)

    def describe_signature(self, signode: TextElement, ast: Any, options: Dict) -> None:
        ast.describe_signature(signode, 'lastIsName', self.env, options)

    def run(self) -> List[Node]:
        env = self.state.document.settings.env  # from ObjectDescription.run
        if 'c:parent_symbol' not in env.temp_data:
            root = env.domaindata['c']['root_symbol']
            env.temp_data['c:parent_symbol'] = root
            env.ref_context['c:parent_key'] = root.get_lookup_key()

        # When multiple declarations are made in the same directive
        # they need to know about each other to provide symbol lookup for function parameters.
        # We use last_symbol to store the latest added declaration in a directive.
        env.temp_data['c:last_symbol'] = None
        return super().run()

    def handle_signature(self, sig: str, signode: TextElement) -> ASTDeclaration:
        parentSymbol = self.env.temp_data['c:parent_symbol']  # type: Symbol

        parser = DefinitionParser(sig, location=signode, config=self.env.config)
        try:
            ast = self.parse_definition(parser)
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=signode)
            # It is easier to assume some phony name than handling the error in
            # the possibly inner declarations.
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            self.env.temp_data['c:last_symbol'] = symbol
            raise ValueError

        try:
            symbol = parentSymbol.add_declaration(ast, docname=self.env.docname)
            # append the new declaration to the sibling list
            assert symbol.siblingAbove is None
            assert symbol.siblingBelow is None
            symbol.siblingAbove = self.env.temp_data['c:last_symbol']
            if symbol.siblingAbove is not None:
                assert symbol.siblingAbove.siblingBelow is None
                symbol.siblingAbove.siblingBelow = symbol
            self.env.temp_data['c:last_symbol'] = symbol
        except _DuplicateSymbolError as e:
            # Assume we are actually in the old symbol,
            # instead of the newly created duplicate.
            self.env.temp_data['c:last_symbol'] = e.symbol
            logger.warning("Duplicate declaration, %s", sig, location=signode)

        if ast.objectType == 'enumerator':
            self._add_enumerator_to_parent(ast)

        # note: handle_signature may be called multiple time per directive,
        # if it has multiple signatures, so don't mess with the original options.
        options = dict(self.options)
        self.describe_signature(signode, ast, options)
        return ast

    def before_content(self) -> None:
        lastSymbol = self.env.temp_data['c:last_symbol']  # type: Symbol
        assert lastSymbol
        self.oldParentSymbol = self.env.temp_data['c:parent_symbol']
        self.oldParentKey = self.env.ref_context['c:parent_key']  # type: LookupKey
        self.env.temp_data['c:parent_symbol'] = lastSymbol
        self.env.ref_context['c:parent_key'] = lastSymbol.get_lookup_key()

    def after_content(self) -> None:
        self.env.temp_data['c:parent_symbol'] = self.oldParentSymbol
        self.env.ref_context['c:parent_key'] = self.oldParentKey

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for C objects.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return 'c.' + name


class CMemberObject(CObject):
    object_type = 'member'

    @property
    def display_object_type(self) -> str:
        # the distinction between var and member is only cosmetic
        assert self.objtype in ('member', 'var')
        return self.objtype


class CFunctionObject(CObject):
    object_type = 'function'


class CMacroObject(CObject):
    object_type = 'macro'


class CStructObject(CObject):
    object_type = 'struct'


class CUnionObject(CObject):
    object_type = 'union'


class CEnumObject(CObject):
    object_type = 'enum'


class CEnumeratorObject(CObject):
    object_type = 'enumerator'


class CTypeObject(CObject):
    object_type = 'type'


class CNamespaceObject(SphinxDirective):
    """
    This directive is just to tell Sphinx that we're documenting stuff in
    namespace foo.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        rootSymbol = self.env.domaindata['c']['root_symbol']
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            symbol = rootSymbol
            stack = []  # type: List[Symbol]
        else:
            parser = DefinitionParser(self.arguments[0],
                                      location=self.get_source_info(),
                                      config=self.env.config)
            try:
                name = parser.parse_namespace_object()
                parser.assert_end()
            except DefinitionError as e:
                logger.warning(e, location=self.get_source_info())
                name = _make_phony_error_name()
            symbol = rootSymbol.add_name(name)
            stack = [symbol]
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['c:parent_key'] = symbol.get_lookup_key()
        return []


class CNamespacePushObject(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        if self.arguments[0].strip() in ('NULL', '0', 'nullptr'):
            return []
        parser = DefinitionParser(self.arguments[0],
                                  location=self.get_source_info(),
                                  config=self.env.config)
        try:
            name = parser.parse_namespace_object()
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=self.get_source_info())
            name = _make_phony_error_name()
        oldParent = self.env.temp_data.get('c:parent_symbol', None)
        if not oldParent:
            oldParent = self.env.domaindata['c']['root_symbol']
        symbol = oldParent.add_name(name)
        stack = self.env.temp_data.get('c:namespace_stack', [])
        stack.append(symbol)
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['c:parent_key'] = symbol.get_lookup_key()
        return []


class CNamespacePopObject(SphinxDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}  # type: Dict

    def run(self) -> List[Node]:
        stack = self.env.temp_data.get('c:namespace_stack', None)
        if not stack or len(stack) == 0:
            logger.warning("C namespace pop on empty stack. Defaulting to gobal scope.",
                           location=self.get_source_info())
            stack = []
        else:
            stack.pop()
        if len(stack) > 0:
            symbol = stack[-1]
        else:
            symbol = self.env.domaindata['c']['root_symbol']
        self.env.temp_data['c:parent_symbol'] = symbol
        self.env.temp_data['c:namespace_stack'] = stack
        self.env.ref_context['cp:parent_key'] = symbol.get_lookup_key()
        return []


class CXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)

        if not has_explicit_title:
            # major hax: replace anon names via simple string manipulation.
            # Can this actually fail?
            title = anon_identifier_re.sub("[anonymous]", str(title))

        if not has_explicit_title:
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        return title, target


class CExprRole(SphinxRole):
    def __init__(self, asCode: bool) -> None:
        super().__init__()
        if asCode:
            # render the expression as inline code
            self.class_type = 'c-expr'
            self.node_type = nodes.literal  # type: Type[TextElement]
        else:
            # render the expression as inline text
            self.class_type = 'c-texpr'
            self.node_type = nodes.inline

    def run(self) -> Tuple[List[Node], List[system_message]]:
        text = self.text.replace('\n', ' ')
        parser = DefinitionParser(text, location=self.get_source_info(),
                                  config=self.env.config)
        # attempt to mimic XRefRole classes, except that...
        classes = ['xref', 'c', self.class_type]
        try:
            ast = parser.parse_expression()
        except DefinitionError as ex:
            logger.warning('Unparseable C expression: %r\n%s', text, ex,
                           location=self.get_source_info())
            # see below
            return [self.node_type(text, text, classes=classes)], []
        parentSymbol = self.env.temp_data.get('cpp:parent_symbol', None)
        if parentSymbol is None:
            parentSymbol = self.env.domaindata['c']['root_symbol']
        # ...most if not all of these classes should really apply to the individual references,
        # not the container node
        signode = self.node_type(classes=classes)
        ast.describe_signature(signode, 'markType', self.env, parentSymbol)
        return [signode], []


class CDomain(Domain):
    """C language domain."""
    name = 'c'
    label = 'C'
    object_types = {
        'function': ObjType(_('function'), 'func'),
        'member': ObjType(_('member'), 'member'),
        'macro': ObjType(_('macro'), 'macro'),
        'type': ObjType(_('type'), 'type'),
        'var': ObjType(_('variable'), 'data'),
    }

    directives = {
        'member': CMemberObject,
        'var': CMemberObject,
        'function': CFunctionObject,
        'macro': CMacroObject,
        'struct': CStructObject,
        'union': CUnionObject,
        'enum': CEnumObject,
        'enumerator': CEnumeratorObject,
        'type': CTypeObject,
        # scope control
        'namespace': CNamespaceObject,
        'namespace-push': CNamespacePushObject,
        'namespace-pop': CNamespacePopObject,
    }
    roles = {
        'member': CXRefRole(),
        'data': CXRefRole(),
        'var': CXRefRole(),
        'func': CXRefRole(fix_parens=True),
        'macro': CXRefRole(),
        'struct': CXRefRole(),
        'union': CXRefRole(),
        'enum': CXRefRole(),
        'enumerator': CXRefRole(),
        'type': CXRefRole(),
        'expr': CExprRole(asCode=True),
        'texpr': CExprRole(asCode=False)
    }
    initial_data = {
        'root_symbol': Symbol(None, None, None, None),
        'objects': {},  # fullname -> docname, node_id, objtype
    }  # type: Dict[str, Union[Symbol, Dict[str, Tuple[str, str, str]]]]

    @property
    def objects(self) -> Dict[str, Tuple[str, str, str]]:
        return self.data.setdefault('objects', {})  # fullname -> docname, node_id, objtype

    def note_object(self, name: str, objtype: str, node_id: str, location: Any = None) -> None:
        if name in self.objects:
            docname = self.objects[name][0]
            logger.warning(__('Duplicate C object description of %s, '
                              'other instance in %s, use :noindex: for one of them'),
                           name, docname, location=location)
        self.objects[name] = (self.env.docname, node_id, objtype)

    def clear_doc(self, docname: str) -> None:
        if Symbol.debug_show_tree:
            print("clear_doc:", docname)
            print("\tbefore:")
            print(self.data['root_symbol'].dump(1))
            print("\tbefore end")

        rootSymbol = self.data['root_symbol']
        rootSymbol.clear_doc(docname)

        if Symbol.debug_show_tree:
            print("\tafter:")
            print(self.data['root_symbol'].dump(1))
            print("\tafter end")
            print("clear_doc end:", docname)
        for fullname, (fn, _id, _l) in list(self.objects.items()):
            if fn == docname:
                del self.objects[fullname]

    def process_doc(self, env: BuildEnvironment, docname: str,
                    document: nodes.document) -> None:
        if Symbol.debug_show_tree:
            print("process_doc:", docname)
            print(self.data['root_symbol'].dump(0))
            print("process_doc end:", docname)

    def process_field_xref(self, pnode: pending_xref) -> None:
        pnode.attributes.update(self.env.ref_context)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        if Symbol.debug_show_tree:
            print("merge_domaindata:")
            print("\tself:")
            print(self.data['root_symbol'].dump(1))
            print("\tself end")
            print("\tother:")
            print(otherdata['root_symbol'].dump(1))
            print("\tother end")
            print("merge_domaindata end")

        self.data['root_symbol'].merge_with(otherdata['root_symbol'],
                                            docnames, self.env)
        ourObjects = self.data['objects']
        for fullname, (fn, id_, objtype) in otherdata['objects'].items():
            if fn in docnames:
                if fullname in ourObjects:
                    msg = __("Duplicate declaration, also defined in '%s'.\n"
                             "Name of declaration is '%s'.")
                    msg = msg % (ourObjects[fullname], fullname)
                    logger.warning(msg, location=fn)
                else:
                    ourObjects[fullname] = (fn, id_, objtype)
```
### 17 - sphinx/domains/c.py:

Start line: 3480, End line: 3549

```python
class CDomain(Domain):

    def _resolve_xref_inner(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                            typ: str, target: str, node: pending_xref,
                            contnode: Element) -> Tuple[Element, str]:
        parser = DefinitionParser(target, location=node, config=env.config)
        try:
            name = parser.parse_xref_object()
        except DefinitionError as e:
            logger.warning('Unparseable C cross-reference: %r\n%s', target, e,
                           location=node)
            return None, None
        parentKey = node.get("c:parent_key", None)  # type: LookupKey
        rootSymbol = self.data['root_symbol']
        if parentKey:
            parentSymbol = rootSymbol.direct_lookup(parentKey)  # type: Symbol
            if not parentSymbol:
                print("Target: ", target)
                print("ParentKey: ", parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol  # should be there
        else:
            parentSymbol = rootSymbol
        s = parentSymbol.find_declaration(name, typ,
                                          matchSelf=True, recurseInAnon=True)
        if s is None or s.declaration is None:
            return None, None

        # TODO: check role type vs. object type

        declaration = s.declaration
        displayName = name.get_display_string()
        docname = s.docname
        assert docname

        return make_refnode(builder, fromdocname, docname,
                            declaration.get_newest_id(), contnode, displayName
                            ), declaration.objectType

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                     typ: str, target: str, node: pending_xref,
                     contnode: Element) -> Element:
        return self._resolve_xref_inner(env, fromdocname, builder, typ,
                                        target, node, contnode)[0]

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder,
                         target: str, node: pending_xref, contnode: Element
                         ) -> List[Tuple[str, Element]]:
        with logging.suppress_logging():
            retnode, objtype = self._resolve_xref_inner(env, fromdocname, builder,
                                                        'any', target, node, contnode)
        if retnode:
            return [('c:' + self.role_for_objtype(objtype), retnode)]
        return []

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        for refname, (docname, node_id, objtype) in list(self.objects.items()):
            yield (refname, refname, objtype, docname, node_id, 1)


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_domain(CDomain)
    app.add_config_value("c_id_attributes", [], 'env')
    app.add_config_value("c_paren_attributes", [], 'env')

    return {
        'version': 'builtin',
        'env_version': 2,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
```
### 22 - sphinx/domains/c.py:

Start line: 831, End line: 870

```python
class ASTArray(ASTBase):

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text("["))
        addSpace = False

        def _add(signode: TextElement, text: str) -> bool:
            if addSpace:
                signode += nodes.Text(' ')
            signode += addnodes.desc_annotation(text, text)
            return True

        if self.static:
            addSpace = _add(signode, 'static')
        if self.restrict:
            addSpace = _add(signode, 'restrict')
        if self.volatile:
            addSpace = _add(signode, 'volatile')
        if self.const:
            addSpace = _add(signode, 'const')
        if self.vla:
            signode.append(nodes.Text('*'))
        elif self.size:
            if addSpace:
                signode += nodes.Text(' ')
            self.size.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text("]"))


class ASTDeclarator(ASTBase):
    @property
    def name(self) -> ASTNestedName:
        raise NotImplementedError(repr(self))

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        raise NotImplementedError(repr(self))

    def require_space_after_declSpecs(self) -> bool:
        raise NotImplementedError(repr(self))
```
### 37 - sphinx/domains/cpp.py:

Start line: 1499, End line: 1531

```python
class ASTAssignmentExpr(ASTExpression):
    def __init__(self, exprs: List[Union[ASTExpression, ASTBracedInitList]], ops: List[str]):
        assert len(exprs) > 0
        assert len(exprs) == len(ops) + 1
        self.exprs = exprs
        self.ops = ops

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.exprs[0]))
        for i in range(1, len(self.exprs)):
            res.append(' ')
            res.append(self.ops[i - 1])
            res.append(' ')
            res.append(transform(self.exprs[i]))
        return ''.join(res)

    def get_id(self, version: int) -> str:
        res = []
        for i in range(len(self.ops)):
            res.append(_id_operator_v2[self.ops[i]])
            res.append(self.exprs[i].get_id(version))
        res.append(self.exprs[-1].get_id(version))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.exprs[0].describe_signature(signode, mode, env, symbol)
        for i in range(1, len(self.exprs)):
            signode.append(nodes.Text(' '))
            signode.append(nodes.Text(self.ops[i - 1]))
            signode.append(nodes.Text(' '))
            self.exprs[i].describe_signature(signode, mode, env, symbol)
```
### 38 - sphinx/util/cfamily.py:

Start line: 272, End line: 291

```python
class BaseParser:

    @property
    def language(self) -> str:
        raise NotImplementedError

    def status(self, msg: str) -> None:
        # for debugging
        indicator = '-' * self.pos + '^'
        print("%s\n%s\n%s" % (msg, self.definition, indicator))

    def fail(self, msg: str) -> None:
        errors = []
        indicator = '-' * self.pos + '^'
        exMain = DefinitionError(
            'Invalid %s declaration: %s [error at %d]\n  %s\n  %s' %
            (self.language, msg, self.pos, self.definition, indicator))
        errors.append((exMain, "Main error"))
        for err in self.otherErrors:
            errors.append((err, "Potential other error"))
        self.otherErrors = []
        raise self._make_multi_error(errors, '')
```
### 42 - sphinx/domains/cpp.py:

Start line: 2448, End line: 3293

```python
class ASTDeclaratorRef(ASTDeclarator):
    def __init__(self, next: ASTDeclarator, attrs: List[ASTAttribute]) -> None:
        assert next
        self.next = next
        self.attrs = attrs

    @property
    def name(self) -> ASTNestedName:
        return self.next.name

    @property
    def isPack(self) -> bool:
        return True

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.next.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.next.trailingReturn

    def require_space_after_declSpecs(self) -> bool:
        return self.next.require_space_after_declSpecs()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['&']
        for a in self.attrs:
            res.append(transform(a))
        if len(self.attrs) > 0 and self.next.require_space_after_declSpecs():
            res.append(' ')
        res.append(transform(self.next))
        return ''.join(res)

    def get_modifiers_id(self, version: int) -> str:
        return self.next.get_modifiers_id(version)

    def get_param_id(self, version: int) -> str:  # only the parameters (if any)
        return self.next.get_param_id(version)

    def get_ptr_suffix_id(self, version: int) -> str:
        if version == 1:
            return 'R' + self.next.get_ptr_suffix_id(version)
        else:
            return self.next.get_ptr_suffix_id(version) + 'R'

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        assert version >= 2
        # ReturnType &next, so we are part of the return type of 'next
        return self.next.get_type_id(version, returnTypeId='R' + returnTypeId)

    def is_function_type(self) -> bool:
        return self.next.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text("&")
        for a in self.attrs:
            a.describe_signature(signode)
        if len(self.attrs) > 0 and self.next.require_space_after_declSpecs():
            signode += nodes.Text(' ')
        self.next.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorParamPack(ASTDeclarator):
    def __init__(self, next: ASTDeclarator) -> None:
        assert next
        self.next = next

    @property
    def name(self) -> ASTNestedName:
        return self.next.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.next.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.next.trailingReturn

    def require_space_after_declSpecs(self) -> bool:
        return False

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.next)
        if self.next.name:
            res = ' ' + res
        return '...' + res

    def get_modifiers_id(self, version: int) -> str:
        return self.next.get_modifiers_id(version)

    def get_param_id(self, version: int) -> str:  # only the parameters (if any)
        return self.next.get_param_id(version)

    def get_ptr_suffix_id(self, version: int) -> str:
        if version == 1:
            return 'Dp' + self.next.get_ptr_suffix_id(version)
        else:
            return self.next.get_ptr_suffix_id(version) + 'Dp'

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        assert version >= 2
        # ReturnType... next, so we are part of the return type of 'next
        return self.next.get_type_id(version, returnTypeId='Dp' + returnTypeId)

    def is_function_type(self) -> bool:
        return self.next.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text("...")
        if self.next.name:
            signode += nodes.Text(' ')
        self.next.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorMemPtr(ASTDeclarator):
    def __init__(self, className: ASTNestedName,
                 const: bool, volatile: bool, next: ASTDeclarator) -> None:
        assert className
        assert next
        self.className = className
        self.const = const
        self.volatile = volatile
        self.next = next

    @property
    def name(self) -> ASTNestedName:
        return self.next.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.next.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.next.trailingReturn

    def require_space_after_declSpecs(self) -> bool:
        return True

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.className))
        res.append('::*')
        if self.volatile:
            res.append('volatile')
        if self.const:
            if self.volatile:
                res.append(' ')
            res.append('const')
        if self.next.require_space_after_declSpecs():
            res.append(' ')
        res.append(transform(self.next))
        return ''.join(res)

    def get_modifiers_id(self, version: int) -> str:
        if version == 1:
            raise NoOldIdError()
        else:
            return self.next.get_modifiers_id(version)

    def get_param_id(self, version: int) -> str:  # only the parameters (if any)
        if version == 1:
            raise NoOldIdError()
        else:
            return self.next.get_param_id(version)

    def get_ptr_suffix_id(self, version: int) -> str:
        if version == 1:
            raise NoOldIdError()
        else:
            raise NotImplementedError()
            return self.next.get_ptr_suffix_id(version) + 'Dp'

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        assert version >= 2
        # ReturnType name::* next, so we are part of the return type of next
        nextReturnTypeId = ''
        if self.volatile:
            nextReturnTypeId += 'V'
        if self.const:
            nextReturnTypeId += 'K'
        nextReturnTypeId += 'M'
        nextReturnTypeId += self.className.get_id(version)
        nextReturnTypeId += returnTypeId
        return self.next.get_type_id(version, nextReturnTypeId)

    def is_function_type(self) -> bool:
        return self.next.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.className.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text('::*')

        def _add_anno(signode: TextElement, text: str) -> None:
            signode += addnodes.desc_annotation(text, text)
        if self.volatile:
            _add_anno(signode, 'volatile')
        if self.const:
            if self.volatile:
                signode += nodes.Text(' ')
            _add_anno(signode, 'const')
        if self.next.require_space_after_declSpecs():
            signode += nodes.Text(' ')
        self.next.describe_signature(signode, mode, env, symbol)


class ASTDeclaratorParen(ASTDeclarator):
    def __init__(self, inner: ASTDeclarator, next: ASTDeclarator) -> None:
        assert inner
        assert next
        self.inner = inner
        self.next = next
        # TODO: we assume the name, params, and qualifiers are in inner

    @property
    def name(self) -> ASTNestedName:
        return self.inner.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.inner.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.inner.trailingReturn

    def require_space_after_declSpecs(self) -> bool:
        return True

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['(']
        res.append(transform(self.inner))
        res.append(')')
        res.append(transform(self.next))
        return ''.join(res)

    def get_modifiers_id(self, version: int) -> str:
        return self.inner.get_modifiers_id(version)

    def get_param_id(self, version: int) -> str:  # only the parameters (if any)
        return self.inner.get_param_id(version)

    def get_ptr_suffix_id(self, version: int) -> str:
        if version == 1:
            raise NoOldIdError()  # TODO: was this implemented before?
            return self.next.get_ptr_suffix_id(version) + \
                self.inner.get_ptr_suffix_id(version)
        else:
            return self.inner.get_ptr_suffix_id(version) + \
                self.next.get_ptr_suffix_id(version)

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        assert version >= 2
        # ReturnType (inner)next, so 'inner' returns everything outside
        nextId = self.next.get_type_id(version, returnTypeId)
        return self.inner.get_type_id(version, returnTypeId=nextId)

    def is_function_type(self) -> bool:
        return self.inner.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode += nodes.Text('(')
        self.inner.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text(')')
        self.next.describe_signature(signode, "noneIsName", env, symbol)


# Type and initializer stuff
##############################################################################################

class ASTPackExpansionExpr(ASTExpression):
    def __init__(self, expr: Union[ASTExpression, ASTBracedInitList]):
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.expr) + '...'

    def get_id(self, version: int) -> str:
        id = self.expr.get_id(version)
        return 'sp' + id

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.expr.describe_signature(signode, mode, env, symbol)
        signode += nodes.Text('...')


class ASTParenExprList(ASTBase):
    def __init__(self, exprs: List[Union[ASTExpression, ASTBracedInitList]]) -> None:
        self.exprs = exprs

    def get_id(self, version: int) -> str:
        return "pi%sE" % ''.join(e.get_id(version) for e in self.exprs)

    def _stringify(self, transform: StringifyTransform) -> str:
        exprs = [transform(e) for e in self.exprs]
        return '(%s)' % ', '.join(exprs)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        signode.append(nodes.Text('('))
        first = True
        for e in self.exprs:
            if not first:
                signode.append(nodes.Text(', '))
            else:
                first = False
            e.describe_signature(signode, mode, env, symbol)
        signode.append(nodes.Text(')'))


class ASTInitializer(ASTBase):
    def __init__(self, value: Union[ASTExpression, ASTBracedInitList],
                 hasAssign: bool = True) -> None:
        self.value = value
        self.hasAssign = hasAssign

    def _stringify(self, transform: StringifyTransform) -> str:
        val = transform(self.value)
        if self.hasAssign:
            return ' = ' + val
        else:
            return val

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.hasAssign:
            signode.append(nodes.Text(' = '))
        self.value.describe_signature(signode, 'markType', env, symbol)


class ASTType(ASTBase):
    def __init__(self, declSpecs: ASTDeclSpecs, decl: ASTDeclarator) -> None:
        assert declSpecs
        assert decl
        self.declSpecs = declSpecs
        self.decl = decl

    @property
    def name(self) -> ASTNestedName:
        return self.decl.name

    @property
    def isPack(self) -> bool:
        return self.decl.isPack

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.decl.function_params

    @property
    def trailingReturn(self) -> "ASTType":
        return self.decl.trailingReturn

    def get_id(self, version: int, objectType: str = None,
               symbol: "Symbol" = None) -> str:
        if version == 1:
            res = []
            if objectType:  # needs the name
                if objectType == 'function':  # also modifiers
                    res.append(symbol.get_full_nested_name().get_id(version))
                    res.append(self.decl.get_param_id(version))
                    res.append(self.decl.get_modifiers_id(version))
                    if (self.declSpecs.leftSpecs.constexpr or
                            (self.declSpecs.rightSpecs and
                             self.declSpecs.rightSpecs.constexpr)):
                        res.append('CE')
                elif objectType == 'type':  # just the name
                    res.append(symbol.get_full_nested_name().get_id(version))
                else:
                    print(objectType)
                    assert False
            else:  # only type encoding
                if self.decl.is_function_type():
                    raise NoOldIdError()
                res.append(self.declSpecs.get_id(version))
                res.append(self.decl.get_ptr_suffix_id(version))
                res.append(self.decl.get_param_id(version))
            return ''.join(res)
        # other versions
        res = []
        if objectType:  # needs the name
            if objectType == 'function':  # also modifiers
                modifiers = self.decl.get_modifiers_id(version)
                res.append(symbol.get_full_nested_name().get_id(version, modifiers))
                if version >= 4:
                    # with templates we need to mangle the return type in as well
                    templ = symbol.declaration.templatePrefix
                    if templ is not None:
                        typeId = self.decl.get_ptr_suffix_id(version)
                        if self.trailingReturn:
                            returnTypeId = self.trailingReturn.get_id(version)
                        else:
                            returnTypeId = self.declSpecs.get_id(version)
                        res.append(typeId)
                        res.append(returnTypeId)
                res.append(self.decl.get_param_id(version))
            elif objectType == 'type':  # just the name
                res.append(symbol.get_full_nested_name().get_id(version))
            else:
                print(objectType)
                assert False
        else:  # only type encoding
            # the 'returnType' of a non-function type is simply just the last
            # type, i.e., for 'int*' it is 'int'
            returnTypeId = self.declSpecs.get_id(version)
            typeId = self.decl.get_type_id(version, returnTypeId)
            res.append(typeId)
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        declSpecs = transform(self.declSpecs)
        res.append(declSpecs)
        if self.decl.require_space_after_declSpecs() and len(declSpecs) > 0:
            res.append(' ')
        res.append(transform(self.decl))
        return ''.join(res)

    def get_type_declaration_prefix(self) -> str:
        if self.declSpecs.trailingTypeSpec:
            return 'typedef'
        else:
            return 'type'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.declSpecs.describe_signature(signode, 'markType', env, symbol)
        if (self.decl.require_space_after_declSpecs() and
                len(str(self.declSpecs)) > 0):
            signode += nodes.Text(' ')
        # for parameters that don't really declare new names we get 'markType',
        # this should not be propagated, but be 'noneIsName'.
        if mode == 'markType':
            mode = 'noneIsName'
        self.decl.describe_signature(signode, mode, env, symbol)


class ASTTemplateParamConstrainedTypeWithInit(ASTBase):
    def __init__(self, type: ASTType, init: ASTType) -> None:
        assert type
        self.type = type
        self.init = init

    @property
    def name(self) -> ASTNestedName:
        return self.type.name

    @property
    def isPack(self) -> bool:
        return self.type.isPack

    def get_id(self, version: int, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C++
        assert version >= 2
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id(version, prefixed=False)
        else:
            return self.type.get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.type)
        if self.init:
            res += " = "
            res += transform(self.init)
        return res

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.type.describe_signature(signode, mode, env, symbol)
        if self.init:
            signode += nodes.Text(" = ")
            self.init.describe_signature(signode, mode, env, symbol)


class ASTTypeWithInit(ASTBase):
    def __init__(self, type: ASTType, init: ASTInitializer) -> None:
        self.type = type
        self.init = init

    @property
    def name(self) -> ASTNestedName:
        return self.type.name

    @property
    def isPack(self) -> bool:
        return self.type.isPack

    def get_id(self, version: int, objectType: str = None,
               symbol: "Symbol" = None) -> str:
        if objectType != 'member':
            return self.type.get_id(version, objectType)
        if version == 1:
            return (symbol.get_full_nested_name().get_id(version) + '__' +
                    self.type.get_id(version))
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.type))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.type.describe_signature(signode, mode, env, symbol)
        if self.init:
            self.init.describe_signature(signode, mode, env, symbol)


class ASTTypeUsing(ASTBase):
    def __init__(self, name: ASTNestedName, type: ASTType) -> None:
        self.name = name
        self.type = type

    def get_id(self, version: int, objectType: str = None,
               symbol: "Symbol" = None) -> str:
        if version == 1:
            raise NoOldIdError()
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if self.type:
            res.append(' = ')
            res.append(transform(self.type))
        return ''.join(res)

    def get_type_declaration_prefix(self) -> str:
        return 'using'

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if self.type:
            signode += nodes.Text(' = ')
            self.type.describe_signature(signode, 'markType', env, symbol=symbol)


# Other declarations
##############################################################################################

class ASTConcept(ASTBase):
    def __init__(self, nestedName: ASTNestedName, initializer: ASTInitializer) -> None:
        self.nestedName = nestedName
        self.initializer = initializer

    @property
    def name(self) -> ASTNestedName:
        return self.nestedName

    def get_id(self, version: int, objectType: str = None,
               symbol: "Symbol" = None) -> str:
        if version == 1:
            raise NoOldIdError()
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.nestedName)
        if self.initializer:
            res += transform(self.initializer)
        return res

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.nestedName.describe_signature(signode, mode, env, symbol)
        if self.initializer:
            self.initializer.describe_signature(signode, mode, env, symbol)


class ASTBaseClass(ASTBase):
    def __init__(self, name: ASTNestedName, visibility: str,
                 virtual: bool, pack: bool) -> None:
        self.name = name
        self.visibility = visibility
        self.virtual = virtual
        self.pack = pack

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []

        if self.visibility is not None:
            res.append(self.visibility)
            res.append(' ')
        if self.virtual:
            res.append('virtual ')
        res.append(transform(self.name))
        if self.pack:
            res.append('...')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        if self.visibility is not None:
            signode += addnodes.desc_annotation(self.visibility,
                                                self.visibility)
            signode += nodes.Text(' ')
        if self.virtual:
            signode += addnodes.desc_annotation('virtual', 'virtual')
            signode += nodes.Text(' ')
        self.name.describe_signature(signode, 'markType', env, symbol=symbol)
        if self.pack:
            signode += nodes.Text('...')


class ASTClass(ASTBase):
    def __init__(self, name: ASTNestedName, final: bool, bases: List[ASTBaseClass]) -> None:
        self.name = name
        self.final = final
        self.bases = bases

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if self.final:
            res.append(' final')
        if len(self.bases) > 0:
            res.append(' : ')
            first = True
            for b in self.bases:
                if not first:
                    res.append(', ')
                first = False
                res.append(transform(b))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if self.final:
            signode += nodes.Text(' ')
            signode += addnodes.desc_annotation('final', 'final')
        if len(self.bases) > 0:
            signode += nodes.Text(' : ')
            for b in self.bases:
                b.describe_signature(signode, mode, env, symbol=symbol)
                signode += nodes.Text(', ')
            signode.pop()


class ASTUnion(ASTBase):
    def __init__(self, name: ASTNestedName) -> None:
        self.name = name

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        if version == 1:
            raise NoOldIdError()
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.name)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol=symbol)


class ASTEnum(ASTBase):
    def __init__(self, name: ASTNestedName, scoped: str,
                 underlyingType: ASTType) -> None:
        self.name = name
        self.scoped = scoped
        self.underlyingType = underlyingType

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        if version == 1:
            raise NoOldIdError()
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.scoped:
            res.append(self.scoped)
            res.append(' ')
        res.append(transform(self.name))
        if self.underlyingType:
            res.append(' : ')
            res.append(transform(self.underlyingType))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        # self.scoped has been done by the CPPEnumObject
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if self.underlyingType:
            signode += nodes.Text(' : ')
            self.underlyingType.describe_signature(signode, 'noneIsName',
                                                   env, symbol=symbol)


class ASTEnumerator(ASTBase):
    def __init__(self, name: ASTNestedName, init: ASTInitializer) -> None:
        self.name = name
        self.init = init

    def get_id(self, version: int, objectType: str, symbol: "Symbol") -> str:
        if version == 1:
            raise NoOldIdError()
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol)
        if self.init:
            self.init.describe_signature(signode, 'markType', env, symbol)


################################################################################
# Templates
################################################################################

# Parameters
################################################################################

class ASTTemplateParam(ASTBase):
    def get_identifier(self) -> ASTIdentifier:
        raise NotImplementedError(repr(self))

    def get_id(self, version: int) -> str:
        raise NotImplementedError(repr(self))

    def describe_signature(self, parentNode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        raise NotImplementedError(repr(self))


class ASTTemplateKeyParamPackIdDefault(ASTTemplateParam):
    def __init__(self, key: str, identifier: ASTIdentifier,
                 parameterPack: bool, default: ASTType) -> None:
        assert key
        if parameterPack:
            assert default is None
        self.key = key
        self.identifier = identifier
        self.parameterPack = parameterPack
        self.default = default

    def get_identifier(self) -> ASTIdentifier:
        return self.identifier

    def get_id(self, version: int) -> str:
        assert version >= 2
        # this is not part of the normal name mangling in C++
        res = []
        if self.parameterPack:
            res.append('Dp')
        else:
            res.append('0')  # we need to put something
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = [self.key]
        if self.parameterPack:
            if self.identifier:
                res.append(' ')
            res.append('...')
        if self.identifier:
            if not self.parameterPack:
                res.append(' ')
            res.append(transform(self.identifier))
        if self.default:
            res.append(' = ')
            res.append(transform(self.default))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        signode += nodes.Text(self.key)
        if self.parameterPack:
            if self.identifier:
                signode += nodes.Text(' ')
            signode += nodes.Text('...')
        if self.identifier:
            if not self.parameterPack:
                signode += nodes.Text(' ')
            self.identifier.describe_signature(signode, mode, env, '', '', symbol)
        if self.default:
            signode += nodes.Text(' = ')
            self.default.describe_signature(signode, 'markType', env, symbol)


class ASTTemplateParamType(ASTTemplateParam):
    def __init__(self, data: ASTTemplateKeyParamPackIdDefault) -> None:
        assert data
        self.data = data

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], [False], rooted=False)

    @property
    def isPack(self) -> bool:
        return self.data.parameterPack

    def get_identifier(self) -> ASTIdentifier:
        return self.data.get_identifier()

    def get_id(self, version: int, objectType: str = None, symbol: "Symbol" = None) -> str:
        # this is not part of the normal name mangling in C++
        assert version >= 2
        if symbol:
            # the anchor will be our parent
            return symbol.parent.declaration.get_id(version, prefixed=False)
        else:
            return self.data.get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.data)

    def describe_signature(self, signode: TextElement, mode: str,
                           env: "BuildEnvironment", symbol: "Symbol") -> None:
        self.data.describe_signature(signode, mode, env, symbol)
```
