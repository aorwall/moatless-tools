# pytest-dev__pytest-7122

| **pytest-dev/pytest** | `be68496440508b760ba1f988bcc63d1d09ace206` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 341 |
| **Avg pos** | 36.0 |
| **Min pos** | 1 |
| **Max pos** | 70 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/src/_pytest/mark/expression.py b/src/_pytest/mark/expression.py
new file mode 100644
--- /dev/null
+++ b/src/_pytest/mark/expression.py
@@ -0,0 +1,173 @@
+r"""
+Evaluate match expressions, as used by `-k` and `-m`.
+
+The grammar is:
+
+expression: expr? EOF
+expr:       and_expr ('or' and_expr)*
+and_expr:   not_expr ('and' not_expr)*
+not_expr:   'not' not_expr | '(' expr ')' | ident
+ident:      (\w|:|\+|-|\.|\[|\])+
+
+The semantics are:
+
+- Empty expression evaluates to False.
+- ident evaluates to True of False according to a provided matcher function.
+- or/and/not evaluate according to the usual boolean semantics.
+"""
+import enum
+import re
+from typing import Callable
+from typing import Iterator
+from typing import Optional
+from typing import Sequence
+
+import attr
+
+from _pytest.compat import TYPE_CHECKING
+
+if TYPE_CHECKING:
+    from typing import NoReturn
+
+
+__all__ = [
+    "evaluate",
+    "ParseError",
+]
+
+
+class TokenType(enum.Enum):
+    LPAREN = "left parenthesis"
+    RPAREN = "right parenthesis"
+    OR = "or"
+    AND = "and"
+    NOT = "not"
+    IDENT = "identifier"
+    EOF = "end of input"
+
+
+@attr.s(frozen=True, slots=True)
+class Token:
+    type = attr.ib(type=TokenType)
+    value = attr.ib(type=str)
+    pos = attr.ib(type=int)
+
+
+class ParseError(Exception):
+    """The expression contains invalid syntax.
+
+    :param column: The column in the line where the error occurred (1-based).
+    :param message: A description of the error.
+    """
+
+    def __init__(self, column: int, message: str) -> None:
+        self.column = column
+        self.message = message
+
+    def __str__(self) -> str:
+        return "at column {}: {}".format(self.column, self.message)
+
+
+class Scanner:
+    __slots__ = ("tokens", "current")
+
+    def __init__(self, input: str) -> None:
+        self.tokens = self.lex(input)
+        self.current = next(self.tokens)
+
+    def lex(self, input: str) -> Iterator[Token]:
+        pos = 0
+        while pos < len(input):
+            if input[pos] in (" ", "\t"):
+                pos += 1
+            elif input[pos] == "(":
+                yield Token(TokenType.LPAREN, "(", pos)
+                pos += 1
+            elif input[pos] == ")":
+                yield Token(TokenType.RPAREN, ")", pos)
+                pos += 1
+            else:
+                match = re.match(r"(:?\w|:|\+|-|\.|\[|\])+", input[pos:])
+                if match:
+                    value = match.group(0)
+                    if value == "or":
+                        yield Token(TokenType.OR, value, pos)
+                    elif value == "and":
+                        yield Token(TokenType.AND, value, pos)
+                    elif value == "not":
+                        yield Token(TokenType.NOT, value, pos)
+                    else:
+                        yield Token(TokenType.IDENT, value, pos)
+                    pos += len(value)
+                else:
+                    raise ParseError(
+                        pos + 1, 'unexpected character "{}"'.format(input[pos]),
+                    )
+        yield Token(TokenType.EOF, "", pos)
+
+    def accept(self, type: TokenType, *, reject: bool = False) -> Optional[Token]:
+        if self.current.type is type:
+            token = self.current
+            if token.type is not TokenType.EOF:
+                self.current = next(self.tokens)
+            return token
+        if reject:
+            self.reject((type,))
+        return None
+
+    def reject(self, expected: Sequence[TokenType]) -> "NoReturn":
+        raise ParseError(
+            self.current.pos + 1,
+            "expected {}; got {}".format(
+                " OR ".join(type.value for type in expected), self.current.type.value,
+            ),
+        )
+
+
+def expression(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    if s.accept(TokenType.EOF):
+        return False
+    ret = expr(s, matcher)
+    s.accept(TokenType.EOF, reject=True)
+    return ret
+
+
+def expr(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    ret = and_expr(s, matcher)
+    while s.accept(TokenType.OR):
+        rhs = and_expr(s, matcher)
+        ret = ret or rhs
+    return ret
+
+
+def and_expr(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    ret = not_expr(s, matcher)
+    while s.accept(TokenType.AND):
+        rhs = not_expr(s, matcher)
+        ret = ret and rhs
+    return ret
+
+
+def not_expr(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    if s.accept(TokenType.NOT):
+        return not not_expr(s, matcher)
+    if s.accept(TokenType.LPAREN):
+        ret = expr(s, matcher)
+        s.accept(TokenType.RPAREN, reject=True)
+        return ret
+    ident = s.accept(TokenType.IDENT)
+    if ident:
+        return matcher(ident.value)
+    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))
+
+
+def evaluate(input: str, matcher: Callable[[str], bool]) -> bool:
+    """Evaluate a match expression as used by -k and -m.
+
+    :param input: The input expression - one line.
+    :param matcher: Given an identifier, should return whether it matches or not.
+                    Should be prepared to handle arbitrary strings as input.
+
+    Returns whether the entire expression matches or not.
+    """
+    return expression(Scanner(input), matcher)
diff --git a/src/_pytest/mark/legacy.py b/src/_pytest/mark/legacy.py
--- a/src/_pytest/mark/legacy.py
+++ b/src/_pytest/mark/legacy.py
@@ -2,44 +2,46 @@
 this is a place where we put datastructures used by legacy apis
 we hope to remove
 """
-import keyword
 from typing import Set
 
 import attr
 
 from _pytest.compat import TYPE_CHECKING
 from _pytest.config import UsageError
+from _pytest.mark.expression import evaluate
+from _pytest.mark.expression import ParseError
 
 if TYPE_CHECKING:
     from _pytest.nodes import Item  # noqa: F401 (used in type string)
 
 
 @attr.s
-class MarkMapping:
-    """Provides a local mapping for markers where item access
-    resolves to True if the marker is present. """
+class MarkMatcher:
+    """A matcher for markers which are present."""
 
     own_mark_names = attr.ib()
 
     @classmethod
-    def from_item(cls, item):
+    def from_item(cls, item) -> "MarkMatcher":
         mark_names = {mark.name for mark in item.iter_markers()}
         return cls(mark_names)
 
-    def __getitem__(self, name):
+    def __call__(self, name: str) -> bool:
         return name in self.own_mark_names
 
 
 @attr.s
-class KeywordMapping:
-    """Provides a local mapping for keywords.
-    Given a list of names, map any substring of one of these names to True.
+class KeywordMatcher:
+    """A matcher for keywords.
+
+    Given a list of names, matches any substring of one of these names. The
+    string inclusion check is case-insensitive.
     """
 
     _names = attr.ib(type=Set[str])
 
     @classmethod
-    def from_item(cls, item: "Item") -> "KeywordMapping":
+    def from_item(cls, item: "Item") -> "KeywordMatcher":
         mapped_names = set()
 
         # Add the names of the current item and any parent items
@@ -62,12 +64,7 @@ def from_item(cls, item: "Item") -> "KeywordMapping":
 
         return cls(mapped_names)
 
-    def __getitem__(self, subname: str) -> bool:
-        """Return whether subname is included within stored names.
-
-        The string inclusion check is case-insensitive.
-
-        """
+    def __call__(self, subname: str) -> bool:
         subname = subname.lower()
         names = (name.lower() for name in self._names)
 
@@ -77,18 +74,17 @@ def __getitem__(self, subname: str) -> bool:
         return False
 
 
-python_keywords_allowed_list = ["or", "and", "not"]
-
-
-def matchmark(colitem, markexpr):
+def matchmark(colitem, markexpr: str) -> bool:
     """Tries to match on any marker names, attached to the given colitem."""
     try:
-        return eval(markexpr, {}, MarkMapping.from_item(colitem))
-    except Exception:
-        raise UsageError("Wrong expression passed to '-m': {}".format(markexpr))
+        return evaluate(markexpr, MarkMatcher.from_item(colitem))
+    except ParseError as e:
+        raise UsageError(
+            "Wrong expression passed to '-m': {}: {}".format(markexpr, e)
+        ) from None
 
 
-def matchkeyword(colitem, keywordexpr):
+def matchkeyword(colitem, keywordexpr: str) -> bool:
     """Tries to match given keyword expression to given collector item.
 
     Will match on the name of colitem, including the names of its parents.
@@ -97,20 +93,9 @@ def matchkeyword(colitem, keywordexpr):
     Additionally, matches on names in the 'extra_keyword_matches' set of
     any item, as well as names directly assigned to test functions.
     """
-    mapping = KeywordMapping.from_item(colitem)
-    if " " not in keywordexpr:
-        # special case to allow for simple "-k pass" and "-k 1.3"
-        return mapping[keywordexpr]
-    elif keywordexpr.startswith("not ") and " " not in keywordexpr[4:]:
-        return not mapping[keywordexpr[4:]]
-    for kwd in keywordexpr.split():
-        if keyword.iskeyword(kwd) and kwd not in python_keywords_allowed_list:
-            raise UsageError(
-                "Python keyword '{}' not accepted in expressions passed to '-k'".format(
-                    kwd
-                )
-            )
     try:
-        return eval(keywordexpr, {}, mapping)
-    except Exception:
-        raise UsageError("Wrong expression passed to '-k': {}".format(keywordexpr))
+        return evaluate(keywordexpr, KeywordMatcher.from_item(colitem))
+    except ParseError as e:
+        raise UsageError(
+            "Wrong expression passed to '-k': {}: {}".format(keywordexpr, e)
+        ) from None

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/_pytest/mark/expression.py | 0 | 0 | - | - | -
| src/_pytest/mark/legacy.py | 5 | 42 | - | 1 | -
| src/_pytest/mark/legacy.py | 65 | 70 | 70 | 1 | 43709
| src/_pytest/mark/legacy.py | 80 | 91 | 1 | 1 | 341
| src/_pytest/mark/legacy.py | 100 | 116 | 1 | 1 | 341


## Problem Statement

```
-k mishandles numbers
Using `pytest 5.4.1`.

It seems that pytest cannot handle keyword selection with numbers, like `-k "1 or 2"`.

Considering the following tests:

\`\`\`
def test_1():
    pass

def test_2():
    pass

def test_3():
    pass
\`\`\`

Selecting with `-k 2` works:

\`\`\`
(venv) Victors-MacBook-Pro:keyword_number_bug fikrettiryaki$ pytest --collect-only -k 2
========================================================================================================== test session starts ===========================================================================================================
platform darwin -- Python 3.7.7, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Users/fikrettiryaki/PycharmProjects/keyword_number_bug
collected 3 items / 2 deselected / 1 selected                                                                                                                                                                                            
<Module test_one.py>
  <Function test_2>
\`\`\`

But selecting with `-k "1 or 2"` doesn't, as I get all tests:

\`\`\`
(venv) Victors-MacBook-Pro:keyword_number_bug fikrettiryaki$ pytest --collect-only -k "1 or 2"
========================================================================================================== test session starts ===========================================================================================================
platform darwin -- Python 3.7.7, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Users/fikrettiryaki/PycharmProjects/keyword_number_bug
collected 3 items                                                                                                                                                                                                                        
<Module test_one.py>
  <Function test_1>
  <Function test_2>
  <Function test_3>
\`\`\`

If I make it a string though, using `-k "_1 or _2"`, then it works again:

\`\`\`
(venv) Victors-MacBook-Pro:keyword_number_bug fikrettiryaki$ pytest --collect-only -k "_1 or _2"
========================================================================================================== test session starts ===========================================================================================================
platform darwin -- Python 3.7.7, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /Users/fikrettiryaki/PycharmProjects/keyword_number_bug
collected 3 items / 1 deselected / 2 selected                                                                                                                                                                                            
<Module test_one.py>
  <Function test_1>
  <Function test_2>
\`\`\`

I see there are some notes about selecting based on keywords here but it was not clear if it applied to this case:
http://doc.pytest.org/en/latest/example/markers.html#using-k-expr-to-select-tests-based-on-their-name

So, is this a bug? Thanks!

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 src/_pytest/mark/legacy.py** | 80 | 117| 341 | 341 | 803 | 
| 2 | 2 testing/python/collect.py | 572 | 671| 662 | 1003 | 10342 | 
| 3 | 3 src/_pytest/mark/__init__.py | 44 | 83| 351 | 1354 | 11547 | 
| 4 | 3 src/_pytest/mark/__init__.py | 104 | 128| 159 | 1513 | 11547 | 
| 5 | 4 testing/python/metafunc.py | 1625 | 1642| 158 | 1671 | 26478 | 
| 6 | 5 testing/python/fixtures.py | 85 | 1065| 6310 | 7981 | 53513 | 
| 7 | 5 testing/python/fixtures.py | 1067 | 2059| 6151 | 14132 | 53513 | 
| 8 | 5 testing/python/fixtures.py | 3537 | 4314| 5087 | 19219 | 53513 | 
| 9 | 5 testing/python/metafunc.py | 1410 | 1434| 210 | 19429 | 53513 | 
| 10 | 6 src/pytest/__init__.py | 1 | 98| 658 | 20087 | 54171 | 
| 11 | 7 testing/example_scripts/issue_519.py | 1 | 31| 350 | 20437 | 54637 | 
| 12 | 7 testing/python/collect.py | 279 | 344| 469 | 20906 | 54637 | 
| 13 | 7 testing/python/fixtures.py | 2061 | 2505| 2812 | 23718 | 54637 | 
| 14 | 8 src/_pytest/main.py | 326 | 365| 233 | 23951 | 59924 | 
| 15 | 8 testing/python/metafunc.py | 1709 | 1723| 138 | 24089 | 59924 | 
| 16 | 8 testing/python/metafunc.py | 1693 | 1707| 133 | 24222 | 59924 | 
| 17 | 8 testing/python/metafunc.py | 180 | 214| 298 | 24520 | 59924 | 
| 18 | 9 src/_pytest/python_api.py | 1 | 42| 234 | 24754 | 66493 | 
| 19 | 9 testing/python/metafunc.py | 1267 | 1284| 180 | 24934 | 66493 | 
| 20 | 9 testing/python/collect.py | 346 | 364| 152 | 25086 | 66493 | 
| 21 | 9 testing/python/fixtures.py | 2507 | 3535| 6208 | 31294 | 66493 | 
| 22 | 10 src/_pytest/doctest.py | 665 | 687| 201 | 31495 | 71579 | 
| 23 | 10 src/_pytest/mark/__init__.py | 1 | 21| 150 | 31645 | 71579 | 
| 24 | 10 testing/python/collect.py | 553 | 570| 182 | 31827 | 71579 | 
| 25 | 10 src/_pytest/main.py | 41 | 148| 741 | 32568 | 71579 | 
| 26 | 10 testing/python/collect.py | 474 | 493| 156 | 32724 | 71579 | 
| 27 | 10 testing/python/metafunc.py | 1870 | 1904| 296 | 33020 | 71579 | 
| 28 | 10 testing/python/metafunc.py | 1832 | 1868| 265 | 33285 | 71579 | 
| 29 | 11 testing/python/integration.py | 1 | 35| 244 | 33529 | 74514 | 
| 30 | 11 testing/python/metafunc.py | 1677 | 1691| 130 | 33659 | 74514 | 
| 31 | 11 testing/python/metafunc.py | 378 | 417| 317 | 33976 | 74514 | 
| 32 | 11 testing/python/collect.py | 1121 | 1158| 209 | 34185 | 74514 | 
| 33 | 11 testing/python/metafunc.py | 1601 | 1623| 175 | 34360 | 74514 | 
| 34 | 11 src/_pytest/mark/__init__.py | 24 | 41| 173 | 34533 | 74514 | 
| 35 | 11 testing/python/metafunc.py | 346 | 376| 292 | 34825 | 74514 | 
| 36 | 11 testing/python/metafunc.py | 234 | 278| 338 | 35163 | 74514 | 
| 37 | 11 testing/python/metafunc.py | 1747 | 1768| 166 | 35329 | 74514 | 
| 38 | 11 testing/python/collect.py | 859 | 891| 304 | 35633 | 74514 | 
| 39 | 11 testing/python/metafunc.py | 1725 | 1745| 203 | 35836 | 74514 | 
| 40 | 12 src/_pytest/pytester.py | 1 | 51| 283 | 36119 | 86412 | 
| 41 | 12 src/_pytest/doctest.py | 1 | 52| 343 | 36462 | 86412 | 
| 42 | 13 testing/python/raises.py | 190 | 219| 260 | 36722 | 88278 | 
| 43 | 13 testing/python/metafunc.py | 1788 | 1811| 230 | 36952 | 88278 | 
| 44 | 14 doc/en/example/xfail_demo.py | 1 | 39| 143 | 37095 | 88422 | 
| 45 | 14 testing/python/fixtures.py | 1 | 82| 490 | 37585 | 88422 | 
| 46 | 14 testing/python/metafunc.py | 1661 | 1675| 118 | 37703 | 88422 | 
| 47 | 14 testing/python/metafunc.py | 134 | 178| 451 | 38154 | 88422 | 
| 48 | 14 testing/python/collect.py | 366 | 393| 192 | 38346 | 88422 | 
| 49 | 14 testing/python/collect.py | 495 | 535| 294 | 38640 | 88422 | 
| 50 | 14 testing/python/metafunc.py | 1644 | 1659| 135 | 38775 | 88422 | 
| 51 | 15 src/_pytest/skipping.py | 34 | 71| 363 | 39138 | 89968 | 
| 52 | 15 testing/python/metafunc.py | 1392 | 1408| 110 | 39248 | 89968 | 
| 53 | 15 testing/python/collect.py | 894 | 933| 352 | 39600 | 89968 | 
| 54 | 16 src/_pytest/_code/code.py | 1 | 46| 269 | 39869 | 99471 | 
| 55 | 16 testing/python/metafunc.py | 1770 | 1786| 114 | 39983 | 99471 | 
| 56 | 16 testing/python/collect.py | 537 | 551| 150 | 40133 | 99471 | 
| 57 | 16 testing/python/metafunc.py | 584 | 610| 238 | 40371 | 99471 | 
| 58 | 16 testing/python/metafunc.py | 110 | 132| 182 | 40553 | 99471 | 
| 59 | 17 src/_pytest/config/argparsing.py | 419 | 455| 382 | 40935 | 103872 | 
| 60 | 17 testing/python/metafunc.py | 873 | 888| 218 | 41153 | 103872 | 
| 61 | 17 testing/python/metafunc.py | 612 | 636| 252 | 41405 | 103872 | 
| 62 | 17 testing/python/metafunc.py | 653 | 695| 371 | 41776 | 103872 | 
| 63 | 17 testing/python/metafunc.py | 80 | 108| 272 | 42048 | 103872 | 
| 64 | 17 src/_pytest/main.py | 149 | 175| 175 | 42223 | 103872 | 
| 65 | 18 testing/python/approx.py | 458 | 487| 225 | 42448 | 109761 | 
| 66 | 19 src/_pytest/terminal.py | 1156 | 1179| 222 | 42670 | 119458 | 
| 67 | 19 testing/python/collect.py | 1220 | 1247| 219 | 42889 | 119458 | 
| 68 | 20 src/_pytest/python.py | 72 | 124| 354 | 43243 | 131956 | 
| 69 | 20 testing/python/metafunc.py | 1214 | 1234| 168 | 43411 | 131956 | 
| **-> 70 <-** | **20 src/_pytest/mark/legacy.py** | 33 | 77| 298 | 43709 | 131956 | 
| 71 | 20 src/_pytest/terminal.py | 78 | 170| 663 | 44372 | 131956 | 
| 72 | 20 testing/python/metafunc.py | 435 | 453| 166 | 44538 | 131956 | 
| 73 | 20 testing/python/raises.py | 221 | 253| 254 | 44792 | 131956 | 
| 74 | 20 testing/python/integration.py | 37 | 68| 231 | 45023 | 131956 | 
| 75 | 20 testing/python/metafunc.py | 1023 | 1045| 176 | 45199 | 131956 | 
| 76 | 21 src/_pytest/config/__init__.py | 1 | 92| 564 | 45763 | 141482 | 
| 77 | 21 testing/python/collect.py | 674 | 701| 227 | 45990 | 141482 | 
| 78 | 21 src/_pytest/mark/__init__.py | 131 | 151| 114 | 46104 | 141482 | 
| 79 | 21 testing/python/metafunc.py | 890 | 897| 139 | 46243 | 141482 | 
| 80 | 21 testing/python/metafunc.py | 526 | 557| 235 | 46478 | 141482 | 
| 81 | 21 src/_pytest/doctest.py | 52 | 102| 334 | 46812 | 141482 | 
| 82 | 21 testing/python/metafunc.py | 300 | 328| 224 | 47036 | 141482 | 
| 83 | 21 testing/python/metafunc.py | 1813 | 1830| 133 | 47169 | 141482 | 
| 84 | 21 testing/python/metafunc.py | 928 | 968| 319 | 47488 | 141482 | 
| 85 | 21 testing/python/metafunc.py | 1334 | 1366| 211 | 47699 | 141482 | 
| 86 | 21 testing/python/collect.py | 1161 | 1187| 179 | 47878 | 141482 | 
| 87 | 21 src/_pytest/python.py | 1027 | 1059| 321 | 48199 | 141482 | 
| 88 | 21 testing/python/metafunc.py | 65 | 78| 186 | 48385 | 141482 | 
| 89 | 21 testing/python/collect.py | 1343 | 1376| 239 | 48624 | 141482 | 
| 90 | 21 testing/python/metafunc.py | 1108 | 1121| 149 | 48773 | 141482 | 
| 91 | 21 src/_pytest/skipping.py | 74 | 93| 183 | 48956 | 141482 | 
| 92 | 21 testing/python/metafunc.py | 216 | 232| 197 | 49153 | 141482 | 
| 93 | 21 src/_pytest/doctest.py | 105 | 153| 347 | 49500 | 141482 | 
| 94 | 21 testing/python/metafunc.py | 726 | 756| 239 | 49739 | 141482 | 
| 95 | 22 doc/en/example/assertion/failure_demo.py | 1 | 40| 169 | 49908 | 143141 | 
| 96 | 22 testing/python/integration.py | 415 | 437| 140 | 50048 | 143141 | 
| 97 | 22 testing/python/metafunc.py | 493 | 524| 240 | 50288 | 143141 | 
| 98 | 22 testing/python/metafunc.py | 419 | 433| 147 | 50435 | 143141 | 
| 99 | 22 src/_pytest/skipping.py | 126 | 184| 537 | 50972 | 143141 | 
| 100 | 23 src/_pytest/fixtures.py | 173 | 207| 327 | 51299 | 155125 | 
| 101 | 23 testing/python/metafunc.py | 1566 | 1598| 231 | 51530 | 155125 | 
| 102 | 23 testing/python/metafunc.py | 1306 | 1332| 212 | 51742 | 155125 | 
| 103 | 23 testing/python/metafunc.py | 899 | 926| 222 | 51964 | 155125 | 
| 104 | 23 testing/python/collect.py | 395 | 422| 193 | 52157 | 155125 | 
| 105 | 23 testing/python/collect.py | 1250 | 1267| 119 | 52276 | 155125 | 
| 106 | 23 testing/python/collect.py | 1091 | 1118| 183 | 52459 | 155125 | 
| 107 | 23 testing/python/metafunc.py | 1194 | 1212| 129 | 52588 | 155125 | 
| 108 | 23 src/_pytest/main.py | 297 | 323| 244 | 52832 | 155125 | 
| 109 | 23 testing/python/metafunc.py | 1236 | 1265| 211 | 53043 | 155125 | 
| 110 | 23 testing/python/collect.py | 1 | 33| 225 | 53268 | 155125 | 
| 111 | 23 testing/python/metafunc.py | 1286 | 1304| 149 | 53417 | 155125 | 
| 112 | 23 src/_pytest/doctest.py | 395 | 413| 137 | 53554 | 155125 | 
| 113 | 23 src/_pytest/doctest.py | 524 | 558| 305 | 53859 | 155125 | 
| 114 | 23 src/_pytest/doctest.py | 445 | 477| 269 | 54128 | 155125 | 
| 115 | 23 testing/example_scripts/issue_519.py | 34 | 52| 115 | 54243 | 155125 | 
| 116 | 24 src/_pytest/mark/evaluate.py | 55 | 78| 224 | 54467 | 156002 | 
| 117 | 24 testing/python/metafunc.py | 1075 | 1090| 144 | 54611 | 156002 | 
| 118 | 24 testing/python/metafunc.py | 970 | 989| 132 | 54743 | 156002 | 
| 119 | 24 src/_pytest/skipping.py | 1 | 31| 201 | 54944 | 156002 | 
| 120 | 24 testing/python/collect.py | 1270 | 1287| 130 | 55074 | 156002 | 
| 121 | 24 testing/python/approx.py | 440 | 456| 137 | 55211 | 156002 | 
| 122 | 25 src/_pytest/compat.py | 1 | 80| 405 | 55616 | 158779 | 
| 123 | 25 src/_pytest/doctest.py | 263 | 326| 564 | 56180 | 158779 | 
| 124 | 25 testing/python/collect.py | 1303 | 1340| 320 | 56500 | 158779 | 
| 125 | 25 testing/python/metafunc.py | 1170 | 1192| 162 | 56662 | 158779 | 
| 126 | 26 src/_pytest/runner.py | 200 | 222| 238 | 56900 | 161943 | 
| 127 | 26 testing/python/metafunc.py | 1 | 45| 265 | 57165 | 161943 | 
| 128 | 26 src/_pytest/python_api.py | 565 | 678| 1010 | 58175 | 161943 | 
| 129 | 26 testing/python/collect.py | 424 | 441| 123 | 58298 | 161943 | 
| 130 | 26 src/_pytest/fixtures.py | 677 | 697| 132 | 58430 | 161943 | 
| 131 | 26 src/_pytest/python.py | 1 | 69| 482 | 58912 | 161943 | 
| 132 | 26 testing/python/metafunc.py | 475 | 491| 125 | 59037 | 161943 | 
| 133 | 26 testing/python/collect.py | 703 | 723| 133 | 59170 | 161943 | 
| 134 | 26 testing/python/metafunc.py | 1906 | 1928| 143 | 59313 | 161943 | 
| 135 | 27 src/_pytest/faulthandler.py | 1 | 42| 282 | 59595 | 162779 | 
| 136 | 27 testing/python/metafunc.py | 835 | 850| 120 | 59715 | 162779 | 
| 137 | 27 doc/en/example/assertion/failure_demo.py | 43 | 121| 680 | 60395 | 162779 | 
| 138 | 27 src/_pytest/doctest.py | 587 | 614| 284 | 60679 | 162779 | 
| 139 | 28 testing/conftest.py | 178 | 200| 204 | 60883 | 164133 | 
| 140 | 29 bench/bench_argcomplete.py | 1 | 20| 179 | 61062 | 164312 | 
| 141 | 29 testing/python/metafunc.py | 818 | 833| 132 | 61194 | 164312 | 
| 142 | 29 testing/python/metafunc.py | 455 | 473| 138 | 61332 | 164312 | 
| 143 | 29 testing/python/metafunc.py | 1930 | 1953| 154 | 61486 | 164312 | 
| 144 | 29 testing/python/collect.py | 443 | 472| 195 | 61681 | 164312 | 
| 145 | 29 testing/python/raises.py | 1 | 51| 303 | 61984 | 164312 | 
| 146 | 29 testing/python/integration.py | 325 | 362| 224 | 62208 | 164312 | 
| 147 | 29 src/_pytest/compat.py | 107 | 123| 131 | 62339 | 164312 | 
| 148 | 30 src/_pytest/stepwise.py | 1 | 23| 131 | 62470 | 165026 | 
| 149 | 30 src/_pytest/fixtures.py | 1047 | 1083| 253 | 62723 | 165026 | 
| 150 | 30 src/_pytest/doctest.py | 214 | 261| 406 | 63129 | 165026 | 
| 151 | 30 testing/python/collect.py | 59 | 77| 187 | 63316 | 165026 | 
| 152 | 30 testing/python/metafunc.py | 852 | 871| 156 | 63472 | 165026 | 
| 153 | 30 src/_pytest/config/__init__.py | 140 | 203| 339 | 63811 | 165026 | 
| 154 | 30 testing/python/metafunc.py | 1147 | 1168| 163 | 63974 | 165026 | 
| 155 | 30 testing/python/raises.py | 129 | 156| 194 | 64168 | 165026 | 
| 156 | 30 src/_pytest/python.py | 1165 | 1195| 250 | 64418 | 165026 | 
| 157 | 30 testing/python/approx.py | 1 | 40| 196 | 64614 | 165026 | 
| 158 | 30 testing/python/metafunc.py | 1368 | 1390| 230 | 64844 | 165026 | 
| 159 | 30 testing/python/collect.py | 196 | 249| 318 | 65162 | 165026 | 
| 160 | 30 testing/python/metafunc.py | 1437 | 1462| 165 | 65327 | 165026 | 
| 161 | 30 testing/python/metafunc.py | 758 | 778| 158 | 65485 | 165026 | 
| 162 | 30 testing/python/metafunc.py | 1123 | 1145| 179 | 65664 | 165026 | 
| 163 | 30 src/_pytest/python.py | 1272 | 1301| 269 | 65933 | 165026 | 
| 164 | 30 src/_pytest/mark/evaluate.py | 80 | 133| 345 | 66278 | 165026 | 
| 165 | 30 src/_pytest/python.py | 1250 | 1269| 168 | 66446 | 165026 | 
| 166 | 30 testing/python/metafunc.py | 1504 | 1529| 205 | 66651 | 165026 | 
| 167 | 30 testing/python/collect.py | 801 | 823| 195 | 66846 | 165026 | 
| 168 | 31 src/_pytest/outcomes.py | 53 | 68| 124 | 66970 | 166742 | 
| 169 | 31 testing/python/metafunc.py | 780 | 797| 140 | 67110 | 166742 | 
| 170 | 31 src/_pytest/pytester.py | 54 | 76| 144 | 67254 | 166742 | 
| 171 | 31 testing/python/metafunc.py | 330 | 344| 186 | 67440 | 166742 | 
| 172 | 31 doc/en/example/assertion/failure_demo.py | 206 | 253| 228 | 67668 | 166742 | 
| 173 | 31 testing/python/metafunc.py | 799 | 816| 140 | 67808 | 166742 | 
| 174 | 31 src/_pytest/python.py | 364 | 379| 146 | 67954 | 166742 | 
| 175 | 32 src/_pytest/mark/structures.py | 370 | 408| 215 | 68169 | 169711 | 
| 176 | 32 src/_pytest/python.py | 142 | 159| 212 | 68381 | 169711 | 
| 177 | 32 testing/python/collect.py | 1290 | 1300| 110 | 68491 | 169711 | 
| 178 | 32 testing/conftest.py | 1 | 64| 383 | 68874 | 169711 | 
| 179 | 32 testing/python/integration.py | 439 | 466| 160 | 69034 | 169711 | 
| 180 | 32 src/_pytest/python.py | 127 | 139| 119 | 69153 | 169711 | 
| 181 | 32 src/_pytest/python.py | 333 | 362| 272 | 69425 | 169711 | 
| 182 | 32 src/_pytest/python.py | 819 | 827| 122 | 69547 | 169711 | 
| 183 | 32 testing/python/metafunc.py | 992 | 1021| 227 | 69774 | 169711 | 
| 184 | 32 doc/en/example/assertion/failure_demo.py | 164 | 188| 159 | 69933 | 169711 | 
| 185 | 32 testing/python/metafunc.py | 1464 | 1485| 192 | 70125 | 169711 | 
| 186 | 32 testing/python/collect.py | 935 | 954| 178 | 70303 | 169711 | 
| 187 | 32 src/_pytest/python.py | 1304 | 1336| 239 | 70542 | 169711 | 
| 188 | 32 src/_pytest/skipping.py | 96 | 111| 133 | 70675 | 169711 | 
| 189 | 33 src/_pytest/debugging.py | 22 | 43| 155 | 70830 | 172152 | 
| 190 | 33 src/_pytest/python.py | 175 | 205| 288 | 71118 | 172152 | 
| 191 | 33 src/_pytest/doctest.py | 479 | 503| 218 | 71336 | 172152 | 
| 192 | 33 testing/python/collect.py | 825 | 856| 264 | 71600 | 172152 | 
| 193 | 34 src/_pytest/cacheprovider.py | 367 | 422| 410 | 72010 | 176104 | 
| 194 | 34 src/_pytest/mark/__init__.py | 154 | 169| 133 | 72143 | 176104 | 
| 195 | 34 src/_pytest/python.py | 1211 | 1247| 306 | 72449 | 176104 | 
| 196 | 34 src/_pytest/doctest.py | 364 | 392| 208 | 72657 | 176104 | 
| 197 | 35 src/_pytest/unittest.py | 1 | 29| 205 | 72862 | 178323 | 
| 198 | 35 testing/python/metafunc.py | 697 | 724| 230 | 73092 | 178323 | 
| 199 | 35 src/_pytest/doctest.py | 617 | 635| 149 | 73241 | 178323 | 
| 200 | 35 src/_pytest/python_api.py | 180 | 211| 253 | 73494 | 178323 | 
| 201 | 35 src/_pytest/python.py | 231 | 267| 361 | 73855 | 178323 | 
| 202 | 35 src/_pytest/terminal.py | 1182 | 1201| 160 | 74015 | 178323 | 
| 203 | 36 src/_pytest/capture.py | 30 | 46| 115 | 74130 | 184134 | 
| 204 | 36 src/_pytest/pytester.py | 467 | 498| 254 | 74384 | 184134 | 
| 205 | 36 src/_pytest/python.py | 785 | 817| 235 | 74619 | 184134 | 
| 206 | 37 src/_pytest/nodes.py | 1 | 40| 280 | 74899 | 188667 | 
| 207 | 37 src/_pytest/config/argparsing.py | 108 | 125| 188 | 75087 | 188667 | 
| 208 | 37 testing/python/approx.py | 307 | 323| 206 | 75293 | 188667 | 
| 209 | 37 src/_pytest/compat.py | 189 | 227| 264 | 75557 | 188667 | 
| 210 | 38 doc/en/_themes/flask_theme_support.py | 1 | 88| 1273 | 76830 | 189940 | 
| 211 | 38 testing/python/integration.py | 242 | 264| 183 | 77013 | 189940 | 
| 212 | 38 src/_pytest/mark/structures.py | 109 | 142| 309 | 77322 | 189940 | 
| 213 | 38 src/_pytest/unittest.py | 164 | 204| 303 | 77625 | 189940 | 
| 214 | 39 src/_pytest/pathlib.py | 272 | 291| 210 | 77835 | 192576 | 
| 215 | 39 testing/python/approx.py | 89 | 107| 241 | 78076 | 192576 | 
| 216 | 39 src/_pytest/python.py | 381 | 413| 270 | 78346 | 192576 | 
| 217 | 39 testing/python/approx.py | 325 | 340| 259 | 78605 | 192576 | 
| 218 | 40 doc/en/example/multipython.py | 48 | 73| 173 | 78778 | 193014 | 
| 219 | 40 src/_pytest/pytester.py | 1355 | 1368| 126 | 78904 | 193014 | 
| 220 | 40 src/_pytest/doctest.py | 560 | 585| 248 | 79152 | 193014 | 
| 221 | 40 src/_pytest/_code/code.py | 1130 | 1150| 173 | 79325 | 193014 | 
| 222 | 40 src/_pytest/config/argparsing.py | 402 | 417| 158 | 79483 | 193014 | 
| 223 | 40 testing/python/metafunc.py | 638 | 651| 148 | 79631 | 193014 | 


## Missing Patch Files

 * 1: src/_pytest/mark/expression.py
 * 2: src/_pytest/mark/legacy.py

### Hint

```
IMO this is a bug.

This happens before the `-k` expression is evaluated using `eval`, `1` and `2` are evaluated as numbers, and `1 or 2` is just evaluated to True which means all tests are included. On the other hand `_1` is an identifier which only evaluates to True if matches the test name.

If you are interested on working on it, IMO it would be good to move away from `eval` in favor of parsing ourselves. See here for discussion: https://github.com/pytest-dev/pytest/issues/6822#issuecomment-596075955
@bluetech indeed! Thanks for the pointer, it seems to be really the same problem as the other issue.
```

## Patch

```diff
diff --git a/src/_pytest/mark/expression.py b/src/_pytest/mark/expression.py
new file mode 100644
--- /dev/null
+++ b/src/_pytest/mark/expression.py
@@ -0,0 +1,173 @@
+r"""
+Evaluate match expressions, as used by `-k` and `-m`.
+
+The grammar is:
+
+expression: expr? EOF
+expr:       and_expr ('or' and_expr)*
+and_expr:   not_expr ('and' not_expr)*
+not_expr:   'not' not_expr | '(' expr ')' | ident
+ident:      (\w|:|\+|-|\.|\[|\])+
+
+The semantics are:
+
+- Empty expression evaluates to False.
+- ident evaluates to True of False according to a provided matcher function.
+- or/and/not evaluate according to the usual boolean semantics.
+"""
+import enum
+import re
+from typing import Callable
+from typing import Iterator
+from typing import Optional
+from typing import Sequence
+
+import attr
+
+from _pytest.compat import TYPE_CHECKING
+
+if TYPE_CHECKING:
+    from typing import NoReturn
+
+
+__all__ = [
+    "evaluate",
+    "ParseError",
+]
+
+
+class TokenType(enum.Enum):
+    LPAREN = "left parenthesis"
+    RPAREN = "right parenthesis"
+    OR = "or"
+    AND = "and"
+    NOT = "not"
+    IDENT = "identifier"
+    EOF = "end of input"
+
+
+@attr.s(frozen=True, slots=True)
+class Token:
+    type = attr.ib(type=TokenType)
+    value = attr.ib(type=str)
+    pos = attr.ib(type=int)
+
+
+class ParseError(Exception):
+    """The expression contains invalid syntax.
+
+    :param column: The column in the line where the error occurred (1-based).
+    :param message: A description of the error.
+    """
+
+    def __init__(self, column: int, message: str) -> None:
+        self.column = column
+        self.message = message
+
+    def __str__(self) -> str:
+        return "at column {}: {}".format(self.column, self.message)
+
+
+class Scanner:
+    __slots__ = ("tokens", "current")
+
+    def __init__(self, input: str) -> None:
+        self.tokens = self.lex(input)
+        self.current = next(self.tokens)
+
+    def lex(self, input: str) -> Iterator[Token]:
+        pos = 0
+        while pos < len(input):
+            if input[pos] in (" ", "\t"):
+                pos += 1
+            elif input[pos] == "(":
+                yield Token(TokenType.LPAREN, "(", pos)
+                pos += 1
+            elif input[pos] == ")":
+                yield Token(TokenType.RPAREN, ")", pos)
+                pos += 1
+            else:
+                match = re.match(r"(:?\w|:|\+|-|\.|\[|\])+", input[pos:])
+                if match:
+                    value = match.group(0)
+                    if value == "or":
+                        yield Token(TokenType.OR, value, pos)
+                    elif value == "and":
+                        yield Token(TokenType.AND, value, pos)
+                    elif value == "not":
+                        yield Token(TokenType.NOT, value, pos)
+                    else:
+                        yield Token(TokenType.IDENT, value, pos)
+                    pos += len(value)
+                else:
+                    raise ParseError(
+                        pos + 1, 'unexpected character "{}"'.format(input[pos]),
+                    )
+        yield Token(TokenType.EOF, "", pos)
+
+    def accept(self, type: TokenType, *, reject: bool = False) -> Optional[Token]:
+        if self.current.type is type:
+            token = self.current
+            if token.type is not TokenType.EOF:
+                self.current = next(self.tokens)
+            return token
+        if reject:
+            self.reject((type,))
+        return None
+
+    def reject(self, expected: Sequence[TokenType]) -> "NoReturn":
+        raise ParseError(
+            self.current.pos + 1,
+            "expected {}; got {}".format(
+                " OR ".join(type.value for type in expected), self.current.type.value,
+            ),
+        )
+
+
+def expression(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    if s.accept(TokenType.EOF):
+        return False
+    ret = expr(s, matcher)
+    s.accept(TokenType.EOF, reject=True)
+    return ret
+
+
+def expr(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    ret = and_expr(s, matcher)
+    while s.accept(TokenType.OR):
+        rhs = and_expr(s, matcher)
+        ret = ret or rhs
+    return ret
+
+
+def and_expr(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    ret = not_expr(s, matcher)
+    while s.accept(TokenType.AND):
+        rhs = not_expr(s, matcher)
+        ret = ret and rhs
+    return ret
+
+
+def not_expr(s: Scanner, matcher: Callable[[str], bool]) -> bool:
+    if s.accept(TokenType.NOT):
+        return not not_expr(s, matcher)
+    if s.accept(TokenType.LPAREN):
+        ret = expr(s, matcher)
+        s.accept(TokenType.RPAREN, reject=True)
+        return ret
+    ident = s.accept(TokenType.IDENT)
+    if ident:
+        return matcher(ident.value)
+    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))
+
+
+def evaluate(input: str, matcher: Callable[[str], bool]) -> bool:
+    """Evaluate a match expression as used by -k and -m.
+
+    :param input: The input expression - one line.
+    :param matcher: Given an identifier, should return whether it matches or not.
+                    Should be prepared to handle arbitrary strings as input.
+
+    Returns whether the entire expression matches or not.
+    """
+    return expression(Scanner(input), matcher)
diff --git a/src/_pytest/mark/legacy.py b/src/_pytest/mark/legacy.py
--- a/src/_pytest/mark/legacy.py
+++ b/src/_pytest/mark/legacy.py
@@ -2,44 +2,46 @@
 this is a place where we put datastructures used by legacy apis
 we hope to remove
 """
-import keyword
 from typing import Set
 
 import attr
 
 from _pytest.compat import TYPE_CHECKING
 from _pytest.config import UsageError
+from _pytest.mark.expression import evaluate
+from _pytest.mark.expression import ParseError
 
 if TYPE_CHECKING:
     from _pytest.nodes import Item  # noqa: F401 (used in type string)
 
 
 @attr.s
-class MarkMapping:
-    """Provides a local mapping for markers where item access
-    resolves to True if the marker is present. """
+class MarkMatcher:
+    """A matcher for markers which are present."""
 
     own_mark_names = attr.ib()
 
     @classmethod
-    def from_item(cls, item):
+    def from_item(cls, item) -> "MarkMatcher":
         mark_names = {mark.name for mark in item.iter_markers()}
         return cls(mark_names)
 
-    def __getitem__(self, name):
+    def __call__(self, name: str) -> bool:
         return name in self.own_mark_names
 
 
 @attr.s
-class KeywordMapping:
-    """Provides a local mapping for keywords.
-    Given a list of names, map any substring of one of these names to True.
+class KeywordMatcher:
+    """A matcher for keywords.
+
+    Given a list of names, matches any substring of one of these names. The
+    string inclusion check is case-insensitive.
     """
 
     _names = attr.ib(type=Set[str])
 
     @classmethod
-    def from_item(cls, item: "Item") -> "KeywordMapping":
+    def from_item(cls, item: "Item") -> "KeywordMatcher":
         mapped_names = set()
 
         # Add the names of the current item and any parent items
@@ -62,12 +64,7 @@ def from_item(cls, item: "Item") -> "KeywordMapping":
 
         return cls(mapped_names)
 
-    def __getitem__(self, subname: str) -> bool:
-        """Return whether subname is included within stored names.
-
-        The string inclusion check is case-insensitive.
-
-        """
+    def __call__(self, subname: str) -> bool:
         subname = subname.lower()
         names = (name.lower() for name in self._names)
 
@@ -77,18 +74,17 @@ def __getitem__(self, subname: str) -> bool:
         return False
 
 
-python_keywords_allowed_list = ["or", "and", "not"]
-
-
-def matchmark(colitem, markexpr):
+def matchmark(colitem, markexpr: str) -> bool:
     """Tries to match on any marker names, attached to the given colitem."""
     try:
-        return eval(markexpr, {}, MarkMapping.from_item(colitem))
-    except Exception:
-        raise UsageError("Wrong expression passed to '-m': {}".format(markexpr))
+        return evaluate(markexpr, MarkMatcher.from_item(colitem))
+    except ParseError as e:
+        raise UsageError(
+            "Wrong expression passed to '-m': {}: {}".format(markexpr, e)
+        ) from None
 
 
-def matchkeyword(colitem, keywordexpr):
+def matchkeyword(colitem, keywordexpr: str) -> bool:
     """Tries to match given keyword expression to given collector item.
 
     Will match on the name of colitem, including the names of its parents.
@@ -97,20 +93,9 @@ def matchkeyword(colitem, keywordexpr):
     Additionally, matches on names in the 'extra_keyword_matches' set of
     any item, as well as names directly assigned to test functions.
     """
-    mapping = KeywordMapping.from_item(colitem)
-    if " " not in keywordexpr:
-        # special case to allow for simple "-k pass" and "-k 1.3"
-        return mapping[keywordexpr]
-    elif keywordexpr.startswith("not ") and " " not in keywordexpr[4:]:
-        return not mapping[keywordexpr[4:]]
-    for kwd in keywordexpr.split():
-        if keyword.iskeyword(kwd) and kwd not in python_keywords_allowed_list:
-            raise UsageError(
-                "Python keyword '{}' not accepted in expressions passed to '-k'".format(
-                    kwd
-                )
-            )
     try:
-        return eval(keywordexpr, {}, mapping)
-    except Exception:
-        raise UsageError("Wrong expression passed to '-k': {}".format(keywordexpr))
+        return evaluate(keywordexpr, KeywordMatcher.from_item(colitem))
+    except ParseError as e:
+        raise UsageError(
+            "Wrong expression passed to '-k': {}: {}".format(keywordexpr, e)
+        ) from None

```

## Test Patch

```diff
diff --git a/testing/test_mark.py b/testing/test_mark.py
--- a/testing/test_mark.py
+++ b/testing/test_mark.py
@@ -200,6 +200,8 @@ def test_hello():
     "spec",
     [
         ("xyz", ("test_one",)),
+        ("(((  xyz))  )", ("test_one",)),
+        ("not not xyz", ("test_one",)),
         ("xyz and xyz2", ()),
         ("xyz2", ("test_two",)),
         ("xyz or xyz2", ("test_one", "test_two")),
@@ -258,9 +260,11 @@ def test_nointer():
     "spec",
     [
         ("interface", ("test_interface",)),
-        ("not interface", ("test_nointer", "test_pass")),
+        ("not interface", ("test_nointer", "test_pass", "test_1", "test_2")),
         ("pass", ("test_pass",)),
-        ("not pass", ("test_interface", "test_nointer")),
+        ("not pass", ("test_interface", "test_nointer", "test_1", "test_2")),
+        ("not not not (pass)", ("test_interface", "test_nointer", "test_1", "test_2")),
+        ("1 or 2", ("test_1", "test_2")),
     ],
 )
 def test_keyword_option_custom(spec, testdir):
@@ -272,6 +276,10 @@ def test_nointer():
             pass
         def test_pass():
             pass
+        def test_1():
+            pass
+        def test_2():
+            pass
     """
     )
     opt, passed_result = spec
@@ -293,7 +301,7 @@ def test_keyword_option_considers_mark(testdir):
     "spec",
     [
         ("None", ("test_func[None]",)),
-        ("1.3", ("test_func[1.3]",)),
+        ("[1.3]", ("test_func[1.3]",)),
         ("2-3", ("test_func[2-3]",)),
     ],
 )
@@ -333,10 +341,23 @@ def test_func(arg):
     "spec",
     [
         (
-            "foo or import",
-            "ERROR: Python keyword 'import' not accepted in expressions passed to '-k'",
+            "foo or",
+            "at column 7: expected not OR left parenthesis OR identifier; got end of input",
+        ),
+        (
+            "foo or or",
+            "at column 8: expected not OR left parenthesis OR identifier; got or",
+        ),
+        ("(foo", "at column 5: expected right parenthesis; got end of input",),
+        ("foo bar", "at column 5: expected end of input; got identifier",),
+        (
+            "or or",
+            "at column 1: expected not OR left parenthesis OR identifier; got or",
+        ),
+        (
+            "not or",
+            "at column 5: expected not OR left parenthesis OR identifier; got or",
         ),
-        ("foo or", "ERROR: Wrong expression passed to '-k': foo or"),
     ],
 )
 def test_keyword_option_wrong_arguments(spec, testdir, capsys):
@@ -798,10 +819,12 @@ def test_one():
         passed, skipped, failed = reprec.countoutcomes()
         assert passed + skipped + failed == 0
 
-    def test_no_magic_values(self, testdir):
+    @pytest.mark.parametrize(
+        "keyword", ["__", "+", ".."],
+    )
+    def test_no_magic_values(self, testdir, keyword: str) -> None:
         """Make sure the tests do not match on magic values,
-        no double underscored values, like '__dict__',
-        and no instance values, like '()'.
+        no double underscored values, like '__dict__' and '+'.
         """
         p = testdir.makepyfile(
             """
@@ -809,16 +832,12 @@ def test_one(): assert 1
         """
         )
 
-        def assert_test_is_not_selected(keyword):
-            reprec = testdir.inline_run("-k", keyword, p)
-            passed, skipped, failed = reprec.countoutcomes()
-            dlist = reprec.getcalls("pytest_deselected")
-            assert passed + skipped + failed == 0
-            deselected_tests = dlist[0].items
-            assert len(deselected_tests) == 1
-
-        assert_test_is_not_selected("__")
-        assert_test_is_not_selected("()")
+        reprec = testdir.inline_run("-k", keyword, p)
+        passed, skipped, failed = reprec.countoutcomes()
+        dlist = reprec.getcalls("pytest_deselected")
+        assert passed + skipped + failed == 0
+        deselected_tests = dlist[0].items
+        assert len(deselected_tests) == 1
 
 
 class TestMarkDecorator:
@@ -1023,7 +1042,7 @@ def test_foo():
             pass
         """
     )
-    expected = "ERROR: Wrong expression passed to '-m': {}".format(expr)
+    expected = "ERROR: Wrong expression passed to '-m': {}: *".format(expr)
     result = testdir.runpytest(foo, "-m", expr)
     result.stderr.fnmatch_lines([expected])
     assert result.ret == ExitCode.USAGE_ERROR
diff --git a/testing/test_mark_expression.py b/testing/test_mark_expression.py
new file mode 100644
--- /dev/null
+++ b/testing/test_mark_expression.py
@@ -0,0 +1,162 @@
+import pytest
+from _pytest.mark.expression import evaluate
+from _pytest.mark.expression import ParseError
+
+
+def test_empty_is_false() -> None:
+    assert not evaluate("", lambda ident: False)
+    assert not evaluate("", lambda ident: True)
+    assert not evaluate("   ", lambda ident: False)
+    assert not evaluate("\t", lambda ident: False)
+
+
+@pytest.mark.parametrize(
+    ("expr", "expected"),
+    (
+        ("true", True),
+        ("true", True),
+        ("false", False),
+        ("not true", False),
+        ("not false", True),
+        ("not not true", True),
+        ("not not false", False),
+        ("true and true", True),
+        ("true and false", False),
+        ("false and true", False),
+        ("true and true and true", True),
+        ("true and true and false", False),
+        ("true and true and not true", False),
+        ("false or false", False),
+        ("false or true", True),
+        ("true or true", True),
+        ("true or true or false", True),
+        ("true and true or false", True),
+        ("not true or true", True),
+        ("(not true) or true", True),
+        ("not (true or true)", False),
+        ("true and true or false and false", True),
+        ("true and (true or false) and false", False),
+        ("true and (true or (not (not false))) and false", False),
+    ),
+)
+def test_basic(expr: str, expected: bool) -> None:
+    matcher = {"true": True, "false": False}.__getitem__
+    assert evaluate(expr, matcher) is expected
+
+
+@pytest.mark.parametrize(
+    ("expr", "expected"),
+    (
+        ("               true           ", True),
+        ("               ((((((true))))))           ", True),
+        ("     (         ((\t  (((true)))))  \t   \t)", True),
+        ("(     true     and   (((false))))", False),
+        ("not not not not true", True),
+        ("not not not not not true", False),
+    ),
+)
+def test_syntax_oddeties(expr: str, expected: bool) -> None:
+    matcher = {"true": True, "false": False}.__getitem__
+    assert evaluate(expr, matcher) is expected
+
+
+@pytest.mark.parametrize(
+    ("expr", "column", "message"),
+    (
+        ("(", 2, "expected not OR left parenthesis OR identifier; got end of input"),
+        (" (", 3, "expected not OR left parenthesis OR identifier; got end of input",),
+        (
+            ")",
+            1,
+            "expected not OR left parenthesis OR identifier; got right parenthesis",
+        ),
+        (
+            ") ",
+            1,
+            "expected not OR left parenthesis OR identifier; got right parenthesis",
+        ),
+        ("not", 4, "expected not OR left parenthesis OR identifier; got end of input",),
+        (
+            "not not",
+            8,
+            "expected not OR left parenthesis OR identifier; got end of input",
+        ),
+        (
+            "(not)",
+            5,
+            "expected not OR left parenthesis OR identifier; got right parenthesis",
+        ),
+        ("and", 1, "expected not OR left parenthesis OR identifier; got and"),
+        (
+            "ident and",
+            10,
+            "expected not OR left parenthesis OR identifier; got end of input",
+        ),
+        ("ident and or", 11, "expected not OR left parenthesis OR identifier; got or",),
+        ("ident ident", 7, "expected end of input; got identifier"),
+    ),
+)
+def test_syntax_errors(expr: str, column: int, message: str) -> None:
+    with pytest.raises(ParseError) as excinfo:
+        evaluate(expr, lambda ident: True)
+    assert excinfo.value.column == column
+    assert excinfo.value.message == message
+
+
+@pytest.mark.parametrize(
+    "ident",
+    (
+        ".",
+        "...",
+        ":::",
+        "a:::c",
+        "a+-b",
+        "אבגד",
+        "aaאבגדcc",
+        "a[bcd]",
+        "1234",
+        "1234abcd",
+        "1234and",
+        "notandor",
+        "not_and_or",
+        "not[and]or",
+        "1234+5678",
+        "123.232",
+        "True",
+        "False",
+        "if",
+        "else",
+        "while",
+    ),
+)
+def test_valid_idents(ident: str) -> None:
+    assert evaluate(ident, {ident: True}.__getitem__)
+
+
+@pytest.mark.parametrize(
+    "ident",
+    (
+        "/",
+        "\\",
+        "^",
+        "*",
+        "=",
+        "&",
+        "%",
+        "$",
+        "#",
+        "@",
+        "!",
+        "~",
+        "{",
+        "}",
+        '"',
+        "'",
+        "|",
+        ";",
+        "←",
+    ),
+)
+def test_invalid_idents(ident: str) -> None:
+    with pytest.raises(ParseError):
+        evaluate(ident, lambda ident: True)

```


## Code snippets

### 1 - src/_pytest/mark/legacy.py:

Start line: 80, End line: 117

```python
python_keywords_allowed_list = ["or", "and", "not"]


def matchmark(colitem, markexpr):
    """Tries to match on any marker names, attached to the given colitem."""
    try:
        return eval(markexpr, {}, MarkMapping.from_item(colitem))
    except Exception:
        raise UsageError("Wrong expression passed to '-m': {}".format(markexpr))


def matchkeyword(colitem, keywordexpr):
    """Tries to match given keyword expression to given collector item.

    Will match on the name of colitem, including the names of its parents.
    Only matches names of items which are either a :class:`Class` or a
    :class:`Function`.
    Additionally, matches on names in the 'extra_keyword_matches' set of
    any item, as well as names directly assigned to test functions.
    """
    mapping = KeywordMapping.from_item(colitem)
    if " " not in keywordexpr:
        # special case to allow for simple "-k pass" and "-k 1.3"
        return mapping[keywordexpr]
    elif keywordexpr.startswith("not ") and " " not in keywordexpr[4:]:
        return not mapping[keywordexpr[4:]]
    for kwd in keywordexpr.split():
        if keyword.iskeyword(kwd) and kwd not in python_keywords_allowed_list:
            raise UsageError(
                "Python keyword '{}' not accepted in expressions passed to '-k'".format(
                    kwd
                )
            )
    try:
        return eval(keywordexpr, {}, mapping)
    except Exception:
        raise UsageError("Wrong expression passed to '-k': {}".format(keywordexpr))
```
### 2 - testing/python/collect.py:

Start line: 572, End line: 671

```python
class TestFunction:

    def test_parametrize_skipif(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip_if(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])

    def test_parametrize_skip(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.skip('')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_skip(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 skipped in *"])

    def test_parametrize_skipif_no_skip(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.skipif('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_skipif_no_skip(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 1 failed, 2 passed in *"])

    def test_parametrize_xfail(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_xfail(x):
                assert x < 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 xfailed in *"])

    def test_parametrize_passed(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('True')

            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])
            def test_xfail(x):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed, 1 xpassed in *"])

    def test_parametrize_xfail_passed(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            m = pytest.mark.xfail('False')

            @pytest.mark.parametrize('x', [0, 1, m(2)])
            def test_passed(x):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 3 passed in *"])

    def test_function_original_name(self, testdir):
        items = testdir.getitems(
            """
            import pytest
            @pytest.mark.parametrize('arg', [1,2])
            def test_func(arg):
                pass
        """
        )
        assert [x.originalname for x in items] == ["test_func", "test_func"]
```
### 3 - src/_pytest/mark/__init__.py:

Start line: 44, End line: 83

```python
def pytest_addoption(parser):
    group = parser.getgroup("general")
    group._addoption(
        "-k",
        action="store",
        dest="keyword",
        default="",
        metavar="EXPRESSION",
        help="only run tests which match the given substring expression. "
        "An expression is a python evaluatable expression "
        "where all names are substring-matched against test names "
        "and their parent classes. Example: -k 'test_method or test_"
        "other' matches all test functions and classes whose name "
        "contains 'test_method' or 'test_other', while -k 'not test_method' "
        "matches those that don't contain 'test_method' in their names. "
        "-k 'not test_method and not test_other' will eliminate the matches. "
        "Additionally keywords are matched to classes and functions "
        "containing extra names in their 'extra_keyword_matches' set, "
        "as well as functions which have names assigned directly to them. "
        "The matching is case-insensitive.",
    )

    group._addoption(
        "-m",
        action="store",
        dest="markexpr",
        default="",
        metavar="MARKEXPR",
        help="only run tests matching given mark expression.  "
        "example: -m 'mark1 and not mark2'.",
    )

    group.addoption(
        "--markers",
        action="store_true",
        help="show markers (builtin, plugin and per-project ones).",
    )

    parser.addini("markers", "markers for test functions", "linelist")
    parser.addini(EMPTY_PARAMETERSET_OPTION, "default marker for empty parametersets")
```
### 4 - src/_pytest/mark/__init__.py:

Start line: 104, End line: 128

```python
def deselect_by_keyword(items, config):
    keywordexpr = config.option.keyword.lstrip()
    if not keywordexpr:
        return

    if keywordexpr.startswith("-"):
        keywordexpr = "not " + keywordexpr[1:]
    selectuntil = False
    if keywordexpr[-1:] == ":":
        selectuntil = True
        keywordexpr = keywordexpr[:-1]

    remaining = []
    deselected = []
    for colitem in items:
        if keywordexpr and not matchkeyword(colitem, keywordexpr):
            deselected.append(colitem)
        else:
            if selectuntil:
                keywordexpr = None
            remaining.append(colitem)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining
```
### 5 - testing/python/metafunc.py:

Start line: 1625, End line: 1642

```python
class TestMarkersWithParametrization:

    def test_select_based_on_mark(self, testdir: Testdir) -> None:
        s = """
            import pytest

            @pytest.mark.parametrize(("n", "expected"), [
                (1, 2),
                pytest.param(2, 3, marks=pytest.mark.foo),
                (3, 4),
            ])
            def test_increment(n, expected):
                assert n + 1 == expected
        """
        testdir.makepyfile(s)
        rec = testdir.inline_run("-m", "foo")
        passed, skipped, fail = rec.listoutcomes()
        assert len(passed) == 1
        assert len(skipped) == 0
        assert len(fail) == 0
```
### 6 - testing/python/fixtures.py:

Start line: 85, End line: 1065

```python
@pytest.mark.pytester_example_path("fixtures/fill_fixtures")
class TestFillFixtures:
    def test_fillfuncargs_exposed(self):
        # used by oejskit, kept for compatibility
        assert pytest._fillfuncargs == fixtures.fillfixtures

    def test_funcarg_lookupfails(self, testdir):
        testdir.copy_example()
        result = testdir.runpytest()  # "--collect-only")
        assert result.ret != 0
        result.stdout.fnmatch_lines(
            """
            *def test_func(some)*
            *fixture*some*not found*
            *xyzsomething*
            """
        )

    def test_detect_recursive_dependency_error(self, testdir):
        testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["*recursive dependency involving fixture 'fix1' detected*"]
        )

    def test_funcarg_basic(self, testdir):
        testdir.copy_example()
        item = testdir.getitem(Path("test_funcarg_basic.py"))
        item._request._fillfixtures()
        del item.funcargs["request"]
        assert len(get_public_names(item.funcargs)) == 2
        assert item.funcargs["some"] == "test_func"
        assert item.funcargs["other"] == 42

    def test_funcarg_lookup_modulelevel(self, testdir):
        testdir.copy_example()
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_funcarg_lookup_classlevel(self, testdir):
        p = testdir.copy_example()
        result = testdir.runpytest(p)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_conftest_funcargs_only_available_in_subdir(self, testdir):
        testdir.copy_example()
        result = testdir.runpytest("-v")
        result.assert_outcomes(passed=2)

    def test_extend_fixture_module_class(self, testdir):
        testfile = testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_extend_fixture_conftest_module(self, testdir):
        p = testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(next(p.visit("test_*.py")))
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_extend_fixture_conftest_conftest(self, testdir):
        p = testdir.copy_example()
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(next(p.visit("test_*.py")))
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_extend_fixture_conftest_plugin(self, testdir):
        testdir.makepyfile(
            testplugin="""
            import pytest

            @pytest.fixture
            def foo():
                return 7
        """
        )
        testdir.syspathinsert()
        testdir.makeconftest(
            """
            import pytest

            pytest_plugins = 'testplugin'

            @pytest.fixture
            def foo(foo):
                return foo + 7
        """
        )
        testdir.makepyfile(
            """
            def test_foo(foo):
                assert foo == 14
        """
        )
        result = testdir.runpytest("-s")
        assert result.ret == 0

    def test_extend_fixture_plugin_plugin(self, testdir):
        # Two plugins should extend each order in loading order
        testdir.makepyfile(
            testplugin0="""
            import pytest

            @pytest.fixture
            def foo():
                return 7
        """
        )
        testdir.makepyfile(
            testplugin1="""
            import pytest

            @pytest.fixture
            def foo(foo):
                return foo + 7
        """
        )
        testdir.syspathinsert()
        testdir.makepyfile(
            """
            pytest_plugins = ['testplugin0', 'testplugin1']

            def test_foo(foo):
                assert foo == 14
        """
        )
        result = testdir.runpytest()
        assert result.ret == 0

    def test_override_parametrized_fixture_conftest_module(self, testdir):
        """Test override of the parametrized fixture with non-parametrized one on the test module level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param
        """
        )
        testfile = testdir.makepyfile(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'

            def test_spam(spam):
                assert spam == 'spam'
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_override_parametrized_fixture_conftest_conftest(self, testdir):
        """Test override of the parametrized fixture with non-parametrized one on the conftest level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param
        """
        )
        subdir = testdir.mkpydir("subdir")
        subdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture
                def spam():
                    return 'spam'
                """
            )
        )
        testfile = subdir.join("test_spam.py")
        testfile.write(
            textwrap.dedent(
                """\
                def test_spam(spam):
                    assert spam == "spam"
                """
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_override_non_parametrized_fixture_conftest_module(self, testdir):
        """Test override of the non-parametrized fixture with parametrized one on the test module level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'
        """
        )
        testfile = testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param

            params = {'spam': 1}

            def test_spam(spam):
                assert spam == params['spam']
                params['spam'] += 1
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_override_non_parametrized_fixture_conftest_conftest(self, testdir):
        """Test override of the non-parametrized fixture with parametrized one on the conftest level."""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'
        """
        )
        subdir = testdir.mkpydir("subdir")
        subdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[1, 2, 3])
                def spam(request):
                    return request.param
                """
            )
        )
        testfile = subdir.join("test_spam.py")
        testfile.write(
            textwrap.dedent(
                """\
                params = {'spam': 1}

                def test_spam(spam):
                    assert spam == params['spam']
                    params['spam'] += 1
                """
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_override_autouse_fixture_with_parametrized_fixture_conftest_conftest(
        self, testdir
    ):
        """Test override of the autouse fixture with parametrized one on the conftest level.
        This test covers the issue explained in issue 1601
        """
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(autouse=True)
            def spam():
                return 'spam'
        """
        )
        subdir = testdir.mkpydir("subdir")
        subdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[1, 2, 3])
                def spam(request):
                    return request.param
                """
            )
        )
        testfile = subdir.join("test_spam.py")
        testfile.write(
            textwrap.dedent(
                """\
                params = {'spam': 1}

                def test_spam(spam):
                    assert spam == params['spam']
                    params['spam'] += 1
                """
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = testdir.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_autouse_fixture_plugin(self, testdir):
        # A fixture from a plugin has no baseid set, which screwed up
        # the autouse fixture handling.
        testdir.makepyfile(
            testplugin="""
            import pytest

            @pytest.fixture(autouse=True)
            def foo(request):
                request.function.foo = 7
        """
        )
        testdir.syspathinsert()
        testdir.makepyfile(
            """
            pytest_plugins = 'testplugin'

            def test_foo(request):
                assert request.function.foo == 7
        """
        )
        result = testdir.runpytest()
        assert result.ret == 0

    def test_funcarg_lookup_error(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def a_fixture(): pass

            @pytest.fixture
            def b_fixture(): pass

            @pytest.fixture
            def c_fixture(): pass

            @pytest.fixture
            def d_fixture(): pass
        """
        )
        testdir.makepyfile(
            """
            def test_lookup_error(unknown):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at setup of test_lookup_error*",
                "  def test_lookup_error(unknown):*",
                "E       fixture 'unknown' not found",
                ">       available fixtures:*a_fixture,*b_fixture,*c_fixture,*d_fixture*monkeypatch,*",
                # sorted
                ">       use 'py*test --fixtures *' for help on them.",
                "*1 error*",
            ]
        )
        result.stdout.no_fnmatch_line("*INTERNAL*")

    def test_fixture_excinfo_leak(self, testdir):
        # on python2 sys.excinfo would leak into fixture executions
        testdir.makepyfile(
            """
            import sys
            import traceback
            import pytest

            @pytest.fixture
            def leak():
                if sys.exc_info()[0]:  # python3 bug :)
                    traceback.print_exc()
                #fails
                assert sys.exc_info() == (None, None, None)

            def test_leak(leak):
                if sys.exc_info()[0]:  # python3 bug :)
                    traceback.print_exc()
                assert sys.exc_info() == (None, None, None)
        """
        )
        result = testdir.runpytest()
        assert result.ret == 0


class TestRequestBasic:
    def test_request_attributes(self, testdir):
        item = testdir.getitem(
            """
            import pytest

            @pytest.fixture
            def something(request): pass
            def test_func(something): pass
        """
        )
        req = fixtures.FixtureRequest(item)
        assert req.function == item.obj
        assert req.keywords == item.keywords
        assert hasattr(req.module, "test_func")
        assert req.cls is None
        assert req.function.__name__ == "test_func"
        assert req.config == item.config
        assert repr(req).find(req.function.__name__) != -1

    def test_request_attributes_method(self, testdir):
        (item,) = testdir.getitems(
            """
            import pytest
            class TestB(object):

                @pytest.fixture
                def something(self, request):
                    return 1
                def test_func(self, something):
                    pass
        """
        )
        req = item._request
        assert req.cls.__name__ == "TestB"
        assert req.instance.__class__ == req.cls

    def test_request_contains_funcarg_arg2fixturedefs(self, testdir):
        modcol = testdir.getmodulecol(
            """
            import pytest
            @pytest.fixture
            def something(request):
                pass
            class TestClass(object):
                def test_method(self, something):
                    pass
        """
        )
        (item1,) = testdir.genitems([modcol])
        assert item1.name == "test_method"
        arg2fixturedefs = fixtures.FixtureRequest(item1)._arg2fixturedefs
        assert len(arg2fixturedefs) == 1
        assert arg2fixturedefs["something"][0].argname == "something"

    @pytest.mark.skipif(
        hasattr(sys, "pypy_version_info"),
        reason="this method of test doesn't work on pypy",
    )
    def test_request_garbage(self, testdir):
        try:
            import xdist  # noqa
        except ImportError:
            pass
        else:
            pytest.xfail("this test is flaky when executed with xdist")
        testdir.makepyfile(
            """
            import sys
            import pytest
            from _pytest.fixtures import PseudoFixtureDef
            import gc

            @pytest.fixture(autouse=True)
            def something(request):
                original = gc.get_debug()
                gc.set_debug(gc.DEBUG_SAVEALL)
                gc.collect()

                yield

                try:
                    gc.collect()
                    leaked = [x for _ in gc.garbage if isinstance(_, PseudoFixtureDef)]
                    assert leaked == []
                finally:
                    gc.set_debug(original)

            def test_func():
                pass
        """
        )
        result = testdir.runpytest_subprocess()
        result.stdout.fnmatch_lines(["* 1 passed in *"])

    def test_getfixturevalue_recursive(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def something(request):
                return 1
        """
        )
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture
            def something(request):
                return request.getfixturevalue("something") + 1
            def test_func(something):
                assert something == 2
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_getfixturevalue_teardown(self, testdir):
        """
        Issue #1895

        `test_inner` requests `inner` fixture, which in turn requests `resource`
        using `getfixturevalue`. `test_func` then requests `resource`.

        `resource` is teardown before `inner` because the fixture mechanism won't consider
        `inner` dependent on `resource` when it is used via `getfixturevalue`: `test_func`
        will then cause the `resource`'s finalizer to be called first because of this.
        """
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='session')
            def resource():
                r = ['value']
                yield r
                r.pop()

            @pytest.fixture(scope='session')
            def inner(request):
                resource = request.getfixturevalue('resource')
                assert resource == ['value']
                yield
                assert resource == ['value']

            def test_inner(inner):
                pass

            def test_func(resource):
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed in *"])

    def test_getfixturevalue(self, testdir):
        item = testdir.getitem(
            """
            import pytest
            values = [2]
            @pytest.fixture
            def something(request): return 1
            @pytest.fixture
            def other(request):
                return values.pop()
            def test_func(something): pass
        """
        )
        req = item._request

        with pytest.raises(pytest.FixtureLookupError):
            req.getfixturevalue("notexists")
        val = req.getfixturevalue("something")
        assert val == 1
        val = req.getfixturevalue("something")
        assert val == 1
        val2 = req.getfixturevalue("other")
        assert val2 == 2
        val2 = req.getfixturevalue("other")  # see about caching
        assert val2 == 2
        item._request._fillfixtures()
        assert item.funcargs["something"] == 1
        assert len(get_public_names(item.funcargs)) == 2
        assert "request" in item.funcargs

    def test_request_addfinalizer(self, testdir):
        item = testdir.getitem(
            """
            import pytest
            teardownlist = []
            @pytest.fixture
            def something(request):
                request.addfinalizer(lambda: teardownlist.append(1))
            def test_func(something): pass
        """
        )
        item.session._setupstate.prepare(item)
        item._request._fillfixtures()
        # successively check finalization calls
        teardownlist = item.getparent(pytest.Module).obj.teardownlist
        ss = item.session._setupstate
        assert not teardownlist
        ss.teardown_exact(item, None)
        print(ss.stack)
        assert teardownlist == [1]

    def test_request_addfinalizer_failing_setup(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = [1]
            @pytest.fixture
            def myfix(request):
                request.addfinalizer(values.pop)
                assert 0
            def test_fix(myfix):
                pass
            def test_finalizer_ran():
                assert not values
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(failed=1, passed=1)

    def test_request_addfinalizer_failing_setup_module(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = [1, 2]
            @pytest.fixture(scope="module")
            def myfix(request):
                request.addfinalizer(values.pop)
                request.addfinalizer(values.pop)
                assert 0
            def test_fix(myfix):
                pass
        """
        )
        reprec = testdir.inline_run("-s")
        mod = reprec.getcalls("pytest_runtest_setup")[0].item.module
        assert not mod.values

    def test_request_addfinalizer_partial_setup_failure(self, testdir):
        p = testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture
            def something(request):
                request.addfinalizer(lambda: values.append(None))
            def test_func(something, missingarg):
                pass
            def test_second():
                assert len(values) == 1
        """
        )
        result = testdir.runpytest(p)
        result.stdout.fnmatch_lines(
            ["*1 error*"]  # XXX the whole module collection fails
        )

    def test_request_subrequest_addfinalizer_exceptions(self, testdir):
        """
        Ensure exceptions raised during teardown by a finalizer are suppressed
        until all finalizers are called, re-raising the first exception (#2440)
        """
        testdir.makepyfile(
            """
            import pytest
            values = []
            def _excepts(where):
                raise Exception('Error in %s fixture' % where)
            @pytest.fixture
            def subrequest(request):
                return request
            @pytest.fixture
            def something(subrequest):
                subrequest.addfinalizer(lambda: values.append(1))
                subrequest.addfinalizer(lambda: values.append(2))
                subrequest.addfinalizer(lambda: _excepts('something'))
            @pytest.fixture
            def excepts(subrequest):
                subrequest.addfinalizer(lambda: _excepts('excepts'))
                subrequest.addfinalizer(lambda: values.append(3))
            def test_first(something, excepts):
                pass
            def test_second():
                assert values == [3, 2, 1]
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["*Exception: Error in excepts fixture", "* 2 passed, 1 error in *"]
        )

    def test_request_getmodulepath(self, testdir):
        modcol = testdir.getmodulecol("def test_somefunc(): pass")
        (item,) = testdir.genitems([modcol])
        req = fixtures.FixtureRequest(item)
        assert req.fspath == modcol.fspath

    def test_request_fixturenames(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            from _pytest.pytester import get_public_names
            @pytest.fixture()
            def arg1():
                pass
            @pytest.fixture()
            def farg(arg1):
                pass
            @pytest.fixture(autouse=True)
            def sarg(tmpdir):
                pass
            def test_function(request, farg):
                assert set(get_public_names(request.fixturenames)) == \
                       set(["tmpdir", "sarg", "arg1", "request", "farg",
                            "tmp_path", "tmp_path_factory"])
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_request_fixturenames_dynamic_fixture(self, testdir):
        """Regression test for #3057"""
        testdir.copy_example("fixtures/test_getfixturevalue_dynamic.py")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_funcargnames_compatattr(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            def pytest_generate_tests(metafunc):
                with pytest.warns(pytest.PytestDeprecationWarning):
                    assert metafunc.funcargnames == metafunc.fixturenames
            @pytest.fixture
            def fn(request):
                with pytest.warns(pytest.PytestDeprecationWarning):
                    assert request._pyfuncitem.funcargnames == \
                           request._pyfuncitem.fixturenames
                with pytest.warns(pytest.PytestDeprecationWarning):
                    return request.funcargnames, request.fixturenames

            def test_hello(fn):
                assert fn[0] == fn[1]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_setupdecorator_and_xunit(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope='module', autouse=True)
            def setup_module():
                values.append("module")
            @pytest.fixture(autouse=True)
            def setup_function():
                values.append("function")

            def test_func():
                pass

            class TestClass(object):
                @pytest.fixture(scope="class", autouse=True)
                def setup_class(self):
                    values.append("class")
                @pytest.fixture(autouse=True)
                def setup_method(self):
                    values.append("method")
                def test_method(self):
                    pass
            def test_all():
                assert values == ["module", "function", "class",
                             "function", "method", "function"]
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=3)

    def test_fixtures_sub_subdir_normalize_sep(self, testdir):
        # this tests that normalization of nodeids takes place
        b = testdir.mkdir("tests").mkdir("unit")
        b.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def arg1():
                    pass
                """
            )
        )
        p = b.join("test_module.py")
        p.write("def test_func(arg1): pass")
        result = testdir.runpytest(p, "--fixtures")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            """
            *fixtures defined*conftest*
            *arg1*
        """
        )

    def test_show_fixtures_color_yes(self, testdir):
        testdir.makepyfile("def test_this(): assert 1")
        result = testdir.runpytest("--color=yes", "--fixtures")
        assert "\x1b[32mtmpdir" in result.stdout.str()

    def test_newstyle_with_request(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def arg(request):
                pass
            def test_1(arg):
                pass
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_setupcontext_no_param(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(params=[1,2])
            def arg(request):
                return request.param

            @pytest.fixture(autouse=True)
            def mysetup(request, arg):
                assert not hasattr(request, "param")
            def test_1(arg):
                assert arg in (1,2)
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)


class TestRequestMarking:
    def test_applymarker(self, testdir):
        item1, item2 = testdir.getitems(
            """
            import pytest

            @pytest.fixture
            def something(request):
                pass
            class TestClass(object):
                def test_func1(self, something):
                    pass
                def test_func2(self, something):
                    pass
        """
        )
        req1 = fixtures.FixtureRequest(item1)
        assert "xfail" not in item1.keywords
        req1.applymarker(pytest.mark.xfail)
        assert "xfail" in item1.keywords
        assert "skipif" not in item1.keywords
        req1.applymarker(pytest.mark.skipif)
        assert "skipif" in item1.keywords
        with pytest.raises(ValueError):
            req1.applymarker(42)

    def test_accesskeywords(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def keywords(request):
                return request.keywords
            @pytest.mark.XYZ
            def test_function(keywords):
                assert keywords["XYZ"]
                assert "abc" not in keywords
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_accessmarker_dynamic(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            @pytest.fixture()
            def keywords(request):
                return request.keywords

            @pytest.fixture(scope="class", autouse=True)
            def marking(request):
                request.applymarker(pytest.mark.XYZ("hello"))
        """
        )
        testdir.makepyfile(
            """
            import pytest
            def test_fun1(keywords):
                assert keywords["XYZ"] is not None
                assert "abc" not in keywords
            def test_fun2(keywords):
                assert keywords["XYZ"] is not None
                assert "abc" not in keywords
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)


class TestFixtureUsages:
    def test_noargfixturedec(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture
            def arg1():
                return 1

            def test_func(arg1):
                assert arg1 == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_receives_funcargs(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture()
            def arg1():
                return 1

            @pytest.fixture()
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
            def test_all(arg1, arg2):
                assert arg1 == 1
                assert arg2 == 2
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_receives_funcargs_scope_mismatch(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ScopeMismatch*involved factories*",
                "test_receives_funcargs_scope_mismatch.py:6:  def arg2(arg1)",
                "test_receives_funcargs_scope_mismatch.py:2:  def arg1()",
                "*1 error*",
            ]
        )
```
### 7 - testing/python/fixtures.py:

Start line: 1067, End line: 2059

```python
class TestFixtureUsages:

    def test_receives_funcargs_scope_mismatch_issue660(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg1, arg2):
                assert arg2 == 2
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["*ScopeMismatch*involved factories*", "* def arg2*", "*1 error*"]
        )

    def test_invalid_scope(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="functions")
            def badscope():
                pass

            def test_nothing(badscope):
                pass
        """
        )
        result = testdir.runpytest_inprocess()
        result.stdout.fnmatch_lines(
            "*Fixture 'badscope' from test_invalid_scope.py got an unexpected scope value 'functions'"
        )

    @pytest.mark.parametrize("scope", ["function", "session"])
    def test_parameters_without_eq_semantics(self, scope, testdir):
        testdir.makepyfile(
            """
            class NoEq1:  # fails on `a == b` statement
                def __eq__(self, _):
                    raise RuntimeError

            class NoEq2:  # fails on `if a == b:` statement
                def __eq__(self, _):
                    class NoBool:
                        def __bool__(self):
                            raise RuntimeError
                    return NoBool()

            import pytest
            @pytest.fixture(params=[NoEq1(), NoEq2()], scope={scope!r})
            def no_eq(request):
                return request.param

            def test1(no_eq):
                pass

            def test2(no_eq):
                pass
        """.format(
                scope=scope
            )
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*4 passed*"])

    def test_funcarg_parametrized_and_used_twice(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1,2])
            def arg1(request):
                values.append(1)
                return request.param

            @pytest.fixture()
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg1, arg2):
                assert arg2 == arg1 + 1
                assert len(values) == arg1
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["*2 passed*"])

    def test_factory_uses_unknown_funcarg_as_dependency_error(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture()
            def fail(missing):
                return

            @pytest.fixture()
            def call_fail(fail):
                return

            def test_missing(call_fail):
                pass
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            """
            *pytest.fixture()*
            *def call_fail(fail)*
            *pytest.fixture()*
            *def fail*
            *fixture*'missing'*not found*
        """
        )

    def test_factory_setup_as_classes_fails(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class arg1(object):
                def __init__(self, request):
                    self.x = 1
            arg1 = pytest.fixture()(arg1)

        """
        )
        reprec = testdir.inline_run()
        values = reprec.getfailedcollections()
        assert len(values) == 1

    def test_usefixtures_marker(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []

            @pytest.fixture(scope="class")
            def myfix(request):
                request.cls.hello = "world"
                values.append(1)

            class TestClass(object):
                def test_one(self):
                    assert self.hello == "world"
                    assert len(values) == 1
                def test_two(self):
                    assert self.hello == "world"
                    assert len(values) == 1
            pytest.mark.usefixtures("myfix")(TestClass)
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_usefixtures_ini(self, testdir):
        testdir.makeini(
            """
            [pytest]
            usefixtures = myfix
        """
        )
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope="class")
            def myfix(request):
                request.cls.hello = "world"

        """
        )
        testdir.makepyfile(
            """
            class TestClass(object):
                def test_one(self):
                    assert self.hello == "world"
                def test_two(self):
                    assert self.hello == "world"
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_usefixtures_seen_in_showmarkers(self, testdir):
        result = testdir.runpytest("--markers")
        result.stdout.fnmatch_lines(
            """
            *usefixtures(fixturename1*mark tests*fixtures*
        """
        )

    def test_request_instance_issue203(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            class TestClass(object):
                @pytest.fixture
                def setup1(self, request):
                    assert self == request.instance
                    self.arg1 = 1
                def test_hello(self, setup1):
                    assert self.arg1 == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_fixture_parametrized_with_iterator(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []
            def f():
                yield 1
                yield 2
            dec = pytest.fixture(scope="module", params=f())

            @dec
            def arg(request):
                return request.param
            @dec
            def arg2(request):
                return request.param

            def test_1(arg):
                values.append(arg)
            def test_2(arg2):
                values.append(arg2*10)
        """
        )
        reprec = testdir.inline_run("-v")
        reprec.assertoutcome(passed=4)
        values = reprec.getcalls("pytest_runtest_call")[0].item.module.values
        assert values == [1, 2, 10, 20]

    def test_setup_functions_as_fixtures(self, testdir):
        """Ensure setup_* methods obey fixture scope rules (#517, #3094)."""
        testdir.makepyfile(
            """
            import pytest

            DB_INITIALIZED = None

            @pytest.yield_fixture(scope="session", autouse=True)
            def db():
                global DB_INITIALIZED
                DB_INITIALIZED = True
                yield
                DB_INITIALIZED = False

            def setup_module():
                assert DB_INITIALIZED

            def teardown_module():
                assert DB_INITIALIZED

            class TestClass(object):

                def setup_method(self, method):
                    assert DB_INITIALIZED

                def teardown_method(self, method):
                    assert DB_INITIALIZED

                def test_printer_1(self):
                    pass

                def test_printer_2(self):
                    pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["* 2 passed in *"])


class TestFixtureManagerParseFactories:
    @pytest.fixture
    def testdir(self, request):
        testdir = request.getfixturevalue("testdir")
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def hello(request):
                return "conftest"

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return testdir

    def test_parsefactories_evil_objects_issue214(self, testdir):
        testdir.makepyfile(
            """
            class A(object):
                def __call__(self):
                    pass
                def __getattr__(self, name):
                    raise RuntimeError()
            a = A()
            def test_hello():
                pass
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1, failed=0)

    def test_parsefactories_conftest(self, testdir):
        testdir.makepyfile(
            """
            def test_hello(item, fm):
                for name in ("fm", "hello", "item"):
                    faclist = fm.getfixturedefs(name, item.nodeid)
                    assert len(faclist) == 1
                    fac = faclist[0]
                    assert fac.func.__name__ == name
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_parsefactories_conftest_and_module_and_class(self, testdir):
        testdir.makepyfile(
            """\
            import pytest

            @pytest.fixture
            def hello(request):
                return "module"
            class TestClass(object):
                @pytest.fixture
                def hello(self, request):
                    return "class"
                def test_hello(self, item, fm):
                    faclist = fm.getfixturedefs("hello", item.nodeid)
                    print(faclist)
                    assert len(faclist) == 3

                    assert faclist[0].func(item._request) == "conftest"
                    assert faclist[1].func(item._request) == "module"
                    assert faclist[2].func(item._request) == "class"
            """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_parsefactories_relative_node_ids(self, testdir):
        # example mostly taken from:
        # https://mail.python.org/pipermail/pytest-dev/2014-September/002617.html
        runner = testdir.mkdir("runner")
        package = testdir.mkdir("package")
        package.join("conftest.py").write(
            textwrap.dedent(
                """\
            import pytest
            @pytest.fixture
            def one():
                return 1
            """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 1
                """
            )
        )
        sub = package.mkdir("sub")
        sub.join("__init__.py").ensure()
        sub.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def one():
                    return 2
                """
            )
        )
        sub.join("test_y.py").write(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 2
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)
        with runner.as_cwd():
            reprec = testdir.inline_run("..")
            reprec.assertoutcome(passed=2)

    def test_package_xunit_fixture(self, testdir):
        testdir.makepyfile(
            __init__="""\
            values = []
        """
        )
        package = testdir.mkdir("package")
        package.join("__init__.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package")
                def teardown_module():
                    values[:] = []
                """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package"]
                """
            )
        )
        package = testdir.mkdir("package2")
        package.join("__init__.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package2")
                def teardown_module():
                    values[:] = []
                """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package2"]
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_package_fixture_complex(self, testdir):
        testdir.makepyfile(
            __init__="""\
            values = []
        """
        )
        testdir.syspathinsert(testdir.tmpdir.dirname)
        package = testdir.mkdir("package")
        package.join("__init__.py").write("")
        package.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                from .. import values
                @pytest.fixture(scope="package")
                def one():
                    values.append("package")
                    yield values
                    values.pop()
                @pytest.fixture(scope="package", autouse=True)
                def two():
                    values.append("package-auto")
                    yield values
                    values.pop()
                """
            )
        )
        package.join("test_x.py").write(
            textwrap.dedent(
                """\
                from .. import values
                def test_package_autouse():
                    assert values == ["package-auto"]
                def test_package(one):
                    assert values == ["package-auto", "package"]
                """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_collect_custom_items(self, testdir):
        testdir.copy_example("fixtures/custom_item")
        result = testdir.runpytest("foo")
        result.stdout.fnmatch_lines(["*passed*"])


class TestAutouseDiscovery:
    @pytest.fixture
    def testdir(self, testdir):
        testdir.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def perfunction(request, tmpdir):
                pass

            @pytest.fixture()
            def arg1(tmpdir):
                pass
            @pytest.fixture(autouse=True)
            def perfunction2(arg1):
                pass

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return testdir

    def test_parsefactories_conftest(self, testdir):
        testdir.makepyfile(
            """
            from _pytest.pytester import get_public_names
            def test_check_setup(item, fm):
                autousenames = fm._getautousenames(item.nodeid)
                assert len(get_public_names(autousenames)) == 2
                assert "perfunction2" in autousenames
                assert "perfunction" in autousenames
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_two_classes_separated_autouse(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class TestA(object):
                values = []
                @pytest.fixture(autouse=True)
                def setup1(self):
                    self.values.append(1)
                def test_setup1(self):
                    assert self.values == [1]
            class TestB(object):
                values = []
                @pytest.fixture(autouse=True)
                def setup2(self):
                    self.values.append(1)
                def test_setup2(self):
                    assert self.values == [1]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_setup_at_classlevel(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            class TestClass(object):
                @pytest.fixture(autouse=True)
                def permethod(self, request):
                    request.instance.funcname = request.function.__name__
                def test_method1(self):
                    assert self.funcname == "test_method1"
                def test_method2(self):
                    assert self.funcname == "test_method2"
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)

    @pytest.mark.xfail(reason="'enabled' feature not implemented")
    def test_setup_enabled_functionnode(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            def enabled(parentnode, markers):
                return "needsdb" in markers

            @pytest.fixture(params=[1,2])
            def db(request):
                return request.param

            @pytest.fixture(enabled=enabled, autouse=True)
            def createdb(db):
                pass

            def test_func1(request):
                assert "db" not in request.fixturenames

            @pytest.mark.needsdb
            def test_func2(request):
                assert "db" in request.fixturenames
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)

    def test_callables_nocode(self, testdir):
        """
        an imported mock.call would break setup/factory discovery
        due to it being callable and __code__ not being a code object
        """
        testdir.makepyfile(
            """
           class _call(tuple):
               def __call__(self, *k, **kw):
                   pass
               def __getattr__(self, k):
                   return self

           call = _call()
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(failed=0, passed=0)

    def test_autouse_in_conftests(self, testdir):
        a = testdir.mkdir("a")
        b = testdir.mkdir("a1")
        conftest = testdir.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def hello():
                xxx
        """
        )
        conftest.move(a.join(conftest.basename))
        a.join("test_something.py").write("def test_func(): pass")
        b.join("test_otherthing.py").write("def test_func(): pass")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            """
            *1 passed*1 error*
        """
        )

    def test_autouse_in_module_and_two_classes(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(autouse=True)
            def append1():
                values.append("module")
            def test_x():
                assert values == ["module"]

            class TestA(object):
                @pytest.fixture(autouse=True)
                def append2(self):
                    values.append("A")
                def test_hello(self):
                    assert values == ["module", "module", "A"], values
            class TestA2(object):
                def test_world(self):
                    assert values == ["module", "module", "A", "module"], values
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=3)


class TestAutouseManagement:
    def test_autouse_conftest_mid_directory(self, testdir):
        pkgdir = testdir.mkpydir("xyz123")
        pkgdir.join("conftest.py").write(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture(autouse=True)
                def app():
                    import sys
                    sys._myapp = "hello"
                """
            )
        )
        t = pkgdir.ensure("tests", "test_app.py")
        t.write(
            textwrap.dedent(
                """\
                import sys
                def test_app():
                    assert sys._myapp == "hello"
                """
            )
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_funcarg_and_setup(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="module")
            def arg():
                values.append(1)
                return 0
            @pytest.fixture(scope="module", autouse=True)
            def something(arg):
                values.append(2)

            def test_hello(arg):
                assert len(values) == 2
                assert values == [1,2]
                assert arg == 0

            def test_hello2(arg):
                assert len(values) == 2
                assert values == [1,2]
                assert arg == 0
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_uses_parametrized_resource(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(params=[1,2])
            def arg(request):
                return request.param

            @pytest.fixture(autouse=True)
            def something(arg):
                values.append(arg)

            def test_hello():
                if len(values) == 1:
                    assert values == [1]
                elif len(values) == 2:
                    assert values == [1, 2]
                else:
                    0/0

        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)

    def test_session_parametrized_function(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []

            @pytest.fixture(scope="session", params=[1,2])
            def arg(request):
               return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(request, arg):
                if request.function.__name__ == "test_some":
                    values.append(arg)

            def test_some():
                pass

            def test_result(arg):
                assert len(values) == arg
                assert values[:arg] == [1,2][:arg]
        """
        )
        reprec = testdir.inline_run("-v", "-s")
        reprec.assertoutcome(passed=4)

    def test_class_function_parametrization_finalization(self, testdir):
        p = testdir.makeconftest(
            """
            import pytest
            import pprint

            values = []

            @pytest.fixture(scope="function", params=[1,2])
            def farg(request):
                return request.param

            @pytest.fixture(scope="class", params=list("ab"))
            def carg(request):
                return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(request, farg, carg):
                def fin():
                    values.append("fin_%s%s" % (carg, farg))
                request.addfinalizer(fin)
        """
        )
        testdir.makepyfile(
            """
            import pytest

            class TestClass(object):
                def test_1(self):
                    pass
            class TestClass2(object):
                def test_2(self):
                    pass
        """
        )
        confcut = "--confcutdir={}".format(testdir.tmpdir)
        reprec = testdir.inline_run("-v", "-s", confcut)
        reprec.assertoutcome(passed=8)
        config = reprec.getcalls("pytest_unconfigure")[0].config
        values = config.pluginmanager._getconftestmodules(p)[0].values
        assert values == ["fin_a1", "fin_a2", "fin_b1", "fin_b2"] * 2

    def test_scope_ordering(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(scope="function", autouse=True)
            def fappend2():
                values.append(2)
            @pytest.fixture(scope="class", autouse=True)
            def classappend3():
                values.append(3)
            @pytest.fixture(scope="module", autouse=True)
            def mappend():
                values.append(1)

            class TestHallo(object):
                def test_method(self):
                    assert values == [1,3,2]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    def test_parametrization_setup_teardown_ordering(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            values = []
            def pytest_generate_tests(metafunc):
                if metafunc.cls is None:
                    assert metafunc.function is test_finish
                if metafunc.cls is not None:
                    metafunc.parametrize("item", [1,2], scope="class")
            class TestClass(object):
                @pytest.fixture(scope="class", autouse=True)
                def addteardown(self, item, request):
                    values.append("setup-%d" % item)
                    request.addfinalizer(lambda: values.append("teardown-%d" % item))
                def test_step1(self, item):
                    values.append("step1-%d" % item)
                def test_step2(self, item):
                    values.append("step2-%d" % item)

            def test_finish():
                print(values)
                assert values == ["setup-1", "step1-1", "step2-1", "teardown-1",
                             "setup-2", "step1-2", "step2-2", "teardown-2",]
        """
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=5)

    def test_ordering_autouse_before_explicit(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            values = []
            @pytest.fixture(autouse=True)
            def fix1():
                values.append(1)
            @pytest.fixture()
            def arg1():
                values.append(2)
            def test_hello(arg1):
                assert values == [1,2]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)

    @pytest.mark.parametrize("param1", ["", "params=[1]"], ids=["p00", "p01"])
    @pytest.mark.parametrize("param2", ["", "params=[1]"], ids=["p10", "p11"])
    def test_ordering_dependencies_torndown_first(self, testdir, param1, param2):
        """#226"""
        testdir.makepyfile(
            """
            import pytest
            values = []
            @pytest.fixture(%(param1)s)
            def arg1(request):
                request.addfinalizer(lambda: values.append("fin1"))
                values.append("new1")
            @pytest.fixture(%(param2)s)
            def arg2(request, arg1):
                request.addfinalizer(lambda: values.append("fin2"))
                values.append("new2")

            def test_arg(arg2):
                pass
            def test_check():
                assert values == ["new1", "new2", "fin2", "fin1"]
        """
            % locals()
        )
        reprec = testdir.inline_run("-s")
        reprec.assertoutcome(passed=2)


class TestFixtureMarker:
    def test_parametrize(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            @pytest.fixture(params=["a", "b", "c"])
            def arg(request):
                return request.param
            values = []
            def test_param(arg):
                values.append(arg)
            def test_result():
                assert values == list("abc")
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=4)

    def test_multiple_parametrization_issue_736(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1,2,3])
            def foo(request):
                return request.param

            @pytest.mark.parametrize('foobar', [4,5,6])
            def test_issue(foo, foobar):
                assert foo in [1,2,3]
                assert foobar in [4,5,6]
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=9)

    @pytest.mark.parametrize(
        "param_args",
        ["'fixt, val'", "'fixt,val'", "['fixt', 'val']", "('fixt', 'val')"],
    )
    def test_override_parametrized_fixture_issue_979(self, testdir, param_args):
        """Make sure a parametrized argument can override a parametrized fixture.

        This was a regression introduced in the fix for #736.
        """
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(params=[1, 2])
            def fixt(request):
                return request.param

            @pytest.mark.parametrize(%s, [(3, 'x'), (4, 'x')])
            def test_foo(fixt, val):
                pass
        """
            % param_args
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)
```
### 8 - testing/python/fixtures.py:

Start line: 3537, End line: 4314

```python
class TestContextManagerFixtureFuncs:

    def test_yields_more_than_one(self, testdir, flavor):
        testdir.makepyfile(
            """
            from test_context import fixture
            @fixture(scope="module")
            def arg1():
                yield 1
                yield 2
            def test_1(arg1):
                pass
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *fixture function*
            *test_yields*:2*
        """
        )

    def test_custom_name(self, testdir, flavor):
        testdir.makepyfile(
            """
            from test_context import fixture
            @fixture(name='meow')
            def arg1():
                return 'mew'
            def test_1(meow):
                print(meow)
        """
        )
        result = testdir.runpytest("-s")
        result.stdout.fnmatch_lines(["*mew*"])


class TestParameterizedSubRequest:
    def test_call_from_fixture(self, testdir):
        testdir.makepyfile(
            test_call_from_fixture="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            @pytest.fixture
            def get_named_fixture(request):
                return request.getfixturevalue('fix_with_param')

            def test_foo(request, get_named_fixture):
                pass
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_fixture.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_fixture.py:4",
                "Requested here:",
                "test_call_from_fixture.py:9",
                "*1 error in*",
            ]
        )

    def test_call_from_test(self, testdir):
        testdir.makepyfile(
            test_call_from_test="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_test.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_test.py:4",
                "Requested here:",
                "test_call_from_test.py:8",
                "*1 failed*",
            ]
        )

    def test_external_fixture(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param
            """
        )

        testdir.makepyfile(
            test_external_fixture="""
            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_external_fixture.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "conftest.py:4",
                "Requested here:",
                "test_external_fixture.py:2",
                "*1 failed*",
            ]
        )

    def test_non_relative_path(self, testdir):
        tests_dir = testdir.mkdir("tests")
        fixdir = testdir.mkdir("fixtures")
        fixfile = fixdir.join("fix.py")
        fixfile.write(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[0, 1, 2])
                def fix_with_param(request):
                    return request.param
                """
            )
        )

        testfile = tests_dir.join("test_foos.py")
        testfile.write(
            textwrap.dedent(
                """\
                from fix import fix_with_param

                def test_foo(request):
                    request.getfixturevalue('fix_with_param')
                """
            )
        )

        tests_dir.chdir()
        testdir.syspathinsert(fixdir)
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "{}:4".format(fixfile),
                "Requested here:",
                "test_foos.py:4",
                "*1 failed*",
            ]
        )

        # With non-overlapping rootdir, passing tests_dir.
        rootdir = testdir.mkdir("rootdir")
        rootdir.chdir()
        result = testdir.runpytest("--rootdir", rootdir, tests_dir)
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "{}:4".format(fixfile),
                "Requested here:",
                "{}:4".format(testfile),
                "*1 failed*",
            ]
        )


def test_pytest_fixture_setup_and_post_finalizer_hook(testdir):
    testdir.makeconftest(
        """
        def pytest_fixture_setup(fixturedef, request):
            print('ROOT setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        def pytest_fixture_post_finalizer(fixturedef, request):
            print('ROOT finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
    """
    )
    testdir.makepyfile(
        **{
            "tests/conftest.py": """
            def pytest_fixture_setup(fixturedef, request):
                print('TESTS setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
            def pytest_fixture_post_finalizer(fixturedef, request):
                print('TESTS finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        """,
            "tests/test_hooks.py": """
            import pytest

            @pytest.fixture()
            def my_fixture():
                return 'some'

            def test_func(my_fixture):
                print('TEST test_func')
                assert my_fixture == 'some'
        """,
        }
    )
    result = testdir.runpytest("-s")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*TESTS setup hook called for my_fixture from test_func*",
            "*ROOT setup hook called for my_fixture from test_func*",
            "*TEST test_func*",
            "*TESTS finalizer hook called for my_fixture from test_func*",
            "*ROOT finalizer hook called for my_fixture from test_func*",
        ]
    )


class TestScopeOrdering:
    """Class of tests that ensure fixtures are ordered based on their scopes (#2405)"""

    @pytest.mark.parametrize("variant", ["mark", "autouse"])
    def test_func_closure_module_auto(self, testdir, variant, monkeypatch):
        """Semantically identical to the example posted in #2405 when ``use_mark=True``"""
        monkeypatch.setenv("FIXTURE_ACTIVATION_VARIANT", variant)
        testdir.makepyfile(
            """
            import warnings
            import os
            import pytest
            VAR = 'FIXTURE_ACTIVATION_VARIANT'
            VALID_VARS = ('autouse', 'mark')

            VARIANT = os.environ.get(VAR)
            if VARIANT is None or VARIANT not in VALID_VARS:
                warnings.warn("{!r} is not  in {}, assuming autouse".format(VARIANT, VALID_VARS) )
                variant = 'mark'

            @pytest.fixture(scope='module', autouse=VARIANT == 'autouse')
            def m1(): pass

            if VARIANT=='mark':
                pytestmark = pytest.mark.usefixtures('m1')

            @pytest.fixture(scope='function', autouse=True)
            def f1(): pass

            def test_func(m1):
                pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "m1 f1".split()

    def test_func_closure_with_native_fixtures(self, testdir, monkeypatch):
        """Sanity check that verifies the order returned by the closures and the actual fixture execution order:
        The execution order may differ because of fixture inter-dependencies.
        """
        monkeypatch.setattr(pytest, "FIXTURE_ORDER", [], raising=False)
        testdir.makepyfile(
            """
            import pytest

            FIXTURE_ORDER = pytest.FIXTURE_ORDER

            @pytest.fixture(scope="session")
            def s1():
                FIXTURE_ORDER.append('s1')

            @pytest.fixture(scope="package")
            def p1():
                FIXTURE_ORDER.append('p1')

            @pytest.fixture(scope="module")
            def m1():
                FIXTURE_ORDER.append('m1')

            @pytest.fixture(scope='session')
            def my_tmpdir_factory():
                FIXTURE_ORDER.append('my_tmpdir_factory')

            @pytest.fixture
            def my_tmpdir(my_tmpdir_factory):
                FIXTURE_ORDER.append('my_tmpdir')

            @pytest.fixture
            def f1(my_tmpdir):
                FIXTURE_ORDER.append('f1')

            @pytest.fixture
            def f2():
                FIXTURE_ORDER.append('f2')

            def test_foo(f1, p1, m1, f2, s1): pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        # order of fixtures based on their scope and position in the parameter list
        assert (
            request.fixturenames == "s1 my_tmpdir_factory p1 m1 f1 f2 my_tmpdir".split()
        )
        testdir.runpytest()
        # actual fixture execution differs: dependent fixtures must be created first ("my_tmpdir")
        assert (
            pytest.FIXTURE_ORDER == "s1 my_tmpdir_factory p1 m1 my_tmpdir f1 f2".split()
        )

    def test_func_closure_module(self, testdir):
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module')
            def m1(): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            def test_func(f1, m1):
                pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "m1 f1".split()

    def test_func_closure_scopes_reordered(self, testdir):
        """Test ensures that fixtures are ordered by scope regardless of the order of the parameters, although
        fixtures of same scope keep the declared order
        """
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='module')
            def m1(): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class')
                def c1(cls): pass

                def test_func(self, f2, f1, c1, m1, s1):
                    pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "s1 m1 c1 f2 f1".split()

    def test_func_closure_same_scope_closer_root_first(self, testdir):
        """Auto-use fixtures of same scope are ordered by closer-to-root first"""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m_conf(): pass
        """
        )
        testdir.makepyfile(
            **{
                "sub/conftest.py": """
                import pytest

                @pytest.fixture(scope='package', autouse=True)
                def p_sub(): pass

                @pytest.fixture(scope='module', autouse=True)
                def m_sub(): pass
            """,
                "sub/__init__.py": "",
                "sub/test_func.py": """
                import pytest

                @pytest.fixture(scope='module', autouse=True)
                def m_test(): pass

                @pytest.fixture(scope='function')
                def f1(): pass

                def test_func(m_test, f1):
                    pass
        """,
            }
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "p_sub m_conf m_sub m_test f1".split()

    def test_func_closure_all_scopes_complex(self, testdir):
        """Complex test involving all scopes and mixing autouse with normal fixtures"""
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='package', autouse=True)
            def p1(): pass
        """
        )
        testdir.makepyfile(**{"__init__.py": ""})
        testdir.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m1(): pass

            @pytest.fixture(scope='module')
            def m2(s1): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class', autouse=True)
                def c1(self):
                    pass

                def test_func(self, f2, f1, m2):
                    pass
        """
        )
        items, _ = testdir.inline_genitems()
        request = FixtureRequest(items[0])
        assert request.fixturenames == "s1 p1 m1 m2 c1 f2 f1".split()

    def test_multiple_packages(self, testdir):
        """Complex test involving multiple package fixtures. Make sure teardowns
        are executed in order.
        .
        └── root
            ├── __init__.py
            ├── sub1
            │   ├── __init__.py
            │   ├── conftest.py
            │   └── test_1.py
            └── sub2
                ├── __init__.py
                ├── conftest.py
                └── test_2.py
        """
        root = testdir.mkdir("root")
        root.join("__init__.py").write("values = []")
        sub1 = root.mkdir("sub1")
        sub1.ensure("__init__.py")
        sub1.join("conftest.py").write(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub1")
                yield values
                assert values.pop() == "pre-sub1"
        """
            )
        )
        sub1.join("test_1.py").write(
            textwrap.dedent(
                """\
            from .. import values
            def test_1(fix):
                assert values == ["pre-sub1"]
        """
            )
        )
        sub2 = root.mkdir("sub2")
        sub2.ensure("__init__.py")
        sub2.join("conftest.py").write(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub2")
                yield values
                assert values.pop() == "pre-sub2"
        """
            )
        )
        sub2.join("test_2.py").write(
            textwrap.dedent(
                """\
            from .. import values
            def test_2(fix):
                assert values == ["pre-sub2"]
        """
            )
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=2)

    def test_class_fixture_self_instance(self, testdir):
        """Check that plugin classes which implement fixtures receive the plugin instance
        as self (see #2270).
        """
        testdir.makeconftest(
            """
            import pytest

            def pytest_configure(config):
                config.pluginmanager.register(MyPlugin())

            class MyPlugin():
                def __init__(self):
                    self.arg = 1

                @pytest.fixture(scope='function')
                def myfix(self):
                    assert isinstance(self, MyPlugin)
                    return self.arg
        """
        )

        testdir.makepyfile(
            """
            class TestClass(object):
                def test_1(self, myfix):
                    assert myfix == 1
        """
        )
        reprec = testdir.inline_run()
        reprec.assertoutcome(passed=1)


def test_call_fixture_function_error():
    """Check if an error is raised if a fixture function is called directly (#4545)"""

    @pytest.fixture
    def fix():
        raise NotImplementedError()

    with pytest.raises(pytest.fail.Exception):
        assert fix() == 1


def test_fixture_param_shadowing(testdir):
    """Parametrized arguments would be shadowed if a fixture with the same name also exists (#5036)"""
    testdir.makepyfile(
        """
        import pytest

        @pytest.fixture(params=['a', 'b'])
        def argroot(request):
            return request.param

        @pytest.fixture
        def arg(argroot):
            return argroot

        # This should only be parametrized directly
        @pytest.mark.parametrize("arg", [1])
        def test_direct(arg):
            assert arg == 1

        # This should be parametrized based on the fixtures
        def test_normal_fixture(arg):
            assert isinstance(arg, str)

        # Indirect should still work:

        @pytest.fixture
        def arg2(request):
            return 2*request.param

        @pytest.mark.parametrize("arg2", [1], indirect=True)
        def test_indirect(arg2):
            assert arg2 == 2
    """
    )
    # Only one test should have run
    result = testdir.runpytest("-v")
    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(["*::test_direct[[]1[]]*"])
    result.stdout.fnmatch_lines(["*::test_normal_fixture[[]a[]]*"])
    result.stdout.fnmatch_lines(["*::test_normal_fixture[[]b[]]*"])
    result.stdout.fnmatch_lines(["*::test_indirect[[]1[]]*"])


def test_fixture_named_request(testdir):
    testdir.copy_example("fixtures/test_fixture_named_request.py")
    result = testdir.runpytest()
    result.stdout.fnmatch_lines(
        [
            "*'request' is a reserved word for fixtures, use another name:",
            "  *test_fixture_named_request.py:5",
        ]
    )


def test_fixture_duplicated_arguments():
    """Raise error if there are positional and keyword arguments for the same parameter (#1682)."""
    with pytest.raises(TypeError) as excinfo:

        @pytest.fixture("session", scope="session")
        def arg(arg):
            pass

    assert (
        str(excinfo.value)
        == "The fixture arguments are defined as positional and keyword: scope. "
        "Use only keyword arguments."
    )


def test_fixture_with_positionals():
    """Raise warning, but the positionals should still works (#1682)."""
    from _pytest.deprecated import FIXTURE_POSITIONAL_ARGUMENTS

    with pytest.warns(pytest.PytestDeprecationWarning) as warnings:

        @pytest.fixture("function", [0], True)
        def fixture_with_positionals():
            pass

    assert str(warnings[0].message) == str(FIXTURE_POSITIONAL_ARGUMENTS)

    assert fixture_with_positionals._pytestfixturefunction.scope == "function"
    assert fixture_with_positionals._pytestfixturefunction.params == (0,)
    assert fixture_with_positionals._pytestfixturefunction.autouse


def test_indirect_fixture_does_not_break_scope(testdir):
    """Ensure that fixture scope is respected when using indirect fixtures (#570)"""
    testdir.makepyfile(
        """
        import pytest
        instantiated  = []

        @pytest.fixture(scope="session")
        def fixture_1(request):
            instantiated.append(("fixture_1", request.param))


        @pytest.fixture(scope="session")
        def fixture_2(request):
            instantiated.append(("fixture_2", request.param))


        scenarios = [
            ("A", "a1"),
            ("A", "a2"),
            ("B", "b1"),
            ("B", "b2"),
            ("C", "c1"),
            ("C", "c2"),
        ]

        @pytest.mark.parametrize(
            "fixture_1,fixture_2", scenarios, indirect=["fixture_1", "fixture_2"]
        )
        def test_create_fixtures(fixture_1, fixture_2):
            pass


        def test_check_fixture_instantiations():
            assert instantiated == [
                ('fixture_1', 'A'),
                ('fixture_2', 'a1'),
                ('fixture_2', 'a2'),
                ('fixture_1', 'B'),
                ('fixture_2', 'b1'),
                ('fixture_2', 'b2'),
                ('fixture_1', 'C'),
                ('fixture_2', 'c1'),
                ('fixture_2', 'c2'),
            ]
    """
    )
    result = testdir.runpytest()
    result.assert_outcomes(passed=7)


def test_fixture_parametrization_nparray(testdir):
    pytest.importorskip("numpy")

    testdir.makepyfile(
        """
        from numpy import linspace
        from pytest import fixture

        @fixture(params=linspace(1, 10, 10))
        def value(request):
            return request.param

        def test_bug(value):
            assert value == value
    """
    )
    result = testdir.runpytest()
    result.assert_outcomes(passed=10)


def test_fixture_arg_ordering(testdir):
    """
    This test describes how fixtures in the same scope but without explicit dependencies
    between them are created. While users should make dependencies explicit, often
    they rely on this order, so this test exists to catch regressions in this regard.
    See #6540 and #6492.
    """
    p1 = testdir.makepyfile(
        """
        import pytest

        suffixes = []

        @pytest.fixture
        def fix_1(): suffixes.append("fix_1")
        @pytest.fixture
        def fix_2(): suffixes.append("fix_2")
        @pytest.fixture
        def fix_3(): suffixes.append("fix_3")
        @pytest.fixture
        def fix_4(): suffixes.append("fix_4")
        @pytest.fixture
        def fix_5(): suffixes.append("fix_5")

        @pytest.fixture
        def fix_combined(fix_1, fix_2, fix_3, fix_4, fix_5): pass

        def test_suffix(fix_combined):
            assert suffixes == ["fix_1", "fix_2", "fix_3", "fix_4", "fix_5"]
        """
    )
    result = testdir.runpytest("-vv", str(p1))
    assert result.ret == 0


def test_yield_fixture_with_no_value(testdir):
    testdir.makepyfile(
        """
        import pytest
        @pytest.fixture(name='custom')
        def empty_yield():
            if False:
                yield

        def test_fixt(custom):
            pass
        """
    )
    expected = "E               ValueError: custom did not yield a value"
    result = testdir.runpytest()
    result.assert_outcomes(error=1)
    result.stdout.fnmatch_lines([expected])
    assert result.ret == ExitCode.TESTS_FAILED
```
### 9 - testing/python/metafunc.py:

Start line: 1410, End line: 1434

```python
class TestMetafuncFunctional:

    def test_parametrize_misspelling(self, testdir: Testdir) -> None:
        """#463"""
        testdir.makepyfile(
            """
            import pytest

            @pytest.mark.parametrise("x", range(2))
            def test_foo(x):
                pass
        """
        )
        result = testdir.runpytest("--collectonly")
        result.stdout.fnmatch_lines(
            [
                "collected 0 items / 1 error",
                "",
                "*= ERRORS =*",
                "*_ ERROR collecting test_parametrize_misspelling.py _*",
                "test_parametrize_misspelling.py:3: in <module>",
                '    @pytest.mark.parametrise("x", range(2))',
                "E   Failed: Unknown 'parametrise' mark, did you mean 'parametrize'?",
                "*! Interrupted: 1 error during collection !*",
                "*= 1 error in *",
            ]
        )
```
### 10 - src/pytest/__init__.py:

Start line: 1, End line: 98

```python
# PYTHON_ARGCOMPLETE_OK
"""
pytest: unit and functional testing with Python.
"""
from . import collect
from _pytest import __version__
from _pytest.assertion import register_assert_rewrite
from _pytest.config import cmdline
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import hookspec
from _pytest.config import main
from _pytest.config import UsageError
from _pytest.debugging import pytestPDB as __pytestPDB
from _pytest.fixtures import fillfixtures as _fillfuncargs
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureLookupError
from _pytest.fixtures import yield_fixture
from _pytest.freeze_support import freeze_includes
from _pytest.main import Session
from _pytest.mark import MARK_GEN as mark
from _pytest.mark import param
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Instance
from _pytest.python import Module
from _pytest.python import Package
from _pytest.python_api import approx
from _pytest.python_api import raises
from _pytest.recwarn import deprecated_call
from _pytest.recwarn import warns
from _pytest.warning_types import PytestAssertRewriteWarning
from _pytest.warning_types import PytestCacheWarning
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestExperimentalApiWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
from _pytest.warning_types import PytestUnknownMarkWarning
from _pytest.warning_types import PytestWarning

set_trace = __pytestPDB.set_trace

__all__ = [
    "__version__",
    "_fillfuncargs",
    "approx",
    "Class",
    "cmdline",
    "collect",
    "Collector",
    "deprecated_call",
    "exit",
    "ExitCode",
    "fail",
    "File",
    "fixture",
    "FixtureLookupError",
    "freeze_includes",
    "Function",
    "hookimpl",
    "hookspec",
    "importorskip",
    "Instance",
    "Item",
    "main",
    "mark",
    "Module",
    "Package",
    "param",
    "PytestAssertRewriteWarning",
    "PytestCacheWarning",
    "PytestCollectionWarning",
    "PytestConfigWarning",
    "PytestDeprecationWarning",
    "PytestExperimentalApiWarning",
    "PytestUnhandledCoroutineWarning",
    "PytestUnknownMarkWarning",
    "PytestWarning",
    "raises",
    "register_assert_rewrite",
    "Session",
    "set_trace",
    "skip",
    "UsageError",
    "warns",
    "xfail",
    "yield_fixture",
]
```
### 70 - src/_pytest/mark/legacy.py:

Start line: 33, End line: 77

```python
@attr.s
class KeywordMapping:
    """Provides a local mapping for keywords.
    Given a list of names, map any substring of one of these names to True.
    """

    _names = attr.ib(type=Set[str])

    @classmethod
    def from_item(cls, item: "Item") -> "KeywordMapping":
        mapped_names = set()

        # Add the names of the current item and any parent items
        import pytest

        for item in item.listchain():
            if not isinstance(item, pytest.Instance):
                mapped_names.add(item.name)

        # Add the names added as extra keywords to current or parent items
        mapped_names.update(item.listextrakeywords())

        # Add the names attached to the current function through direct assignment
        function_obj = getattr(item, "function", None)
        if function_obj:
            mapped_names.update(function_obj.__dict__)

        # add the markers to the keywords as we no longer handle them correctly
        mapped_names.update(mark.name for mark in item.iter_markers())

        return cls(mapped_names)

    def __getitem__(self, subname: str) -> bool:
        """Return whether subname is included within stored names.

        The string inclusion check is case-insensitive.

        """
        subname = subname.lower()
        names = (name.lower() for name in self._names)

        for name in names:
            if subname in name:
                return True
        return False
```
