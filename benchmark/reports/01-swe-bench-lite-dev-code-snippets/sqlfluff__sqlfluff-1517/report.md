# sqlfluff__sqlfluff-1517

| **sqlfluff/sqlfluff** | `304a197829f98e7425a46d872ada73176137e5ae` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 2 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/src/sqlfluff/core/parser/helpers.py b/src/sqlfluff/core/parser/helpers.py
--- a/src/sqlfluff/core/parser/helpers.py
+++ b/src/sqlfluff/core/parser/helpers.py
@@ -2,6 +2,7 @@
 
 from typing import Tuple, List, Any, Iterator, TYPE_CHECKING
 
+from sqlfluff.core.errors import SQLParseError
 from sqlfluff.core.string_helpers import curtail_string
 
 if TYPE_CHECKING:
@@ -26,11 +27,11 @@ def check_still_complete(
     """Check that the segments in are the same as the segments out."""
     initial_str = join_segments_raw(segments_in)
     current_str = join_segments_raw(matched_segments + unmatched_segments)
-    if initial_str != current_str:  # pragma: no cover
-        raise RuntimeError(
-            "Dropped elements in sequence matching! {!r} != {!r}".format(
-                initial_str, current_str
-            )
+
+    if initial_str != current_str:
+        raise SQLParseError(
+            f"Could not parse: {current_str}",
+            segment=unmatched_segments[0],
         )
     return True
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/sqlfluff/core/parser/helpers.py | 2 | - | - | - | -
| src/sqlfluff/core/parser/helpers.py | 29 | 31 | - | - | -


## Problem Statement

```
"Dropped elements in sequence matching" when doubled semicolon
## Expected Behaviour
Frankly, I'm not sure whether it (doubled `;`) should be just ignored or rather some specific rule should be triggered.
## Observed Behaviour
\`\`\`console
(.venv) ?master ~/prod/_inne/sqlfluff> echo "select id from tbl;;" | sqlfluff lint -
Traceback (most recent call last):
  File "/home/adam/prod/_inne/sqlfluff/.venv/bin/sqlfluff", line 11, in <module>
    load_entry_point('sqlfluff', 'console_scripts', 'sqlfluff')()
  File "/home/adam/prod/_inne/sqlfluff/.venv/lib/python3.9/site-packages/click/core.py", line 1137, in __call__
    return self.main(*args, **kwargs)
  File "/home/adam/prod/_inne/sqlfluff/.venv/lib/python3.9/site-packages/click/core.py", line 1062, in main
    rv = self.invoke(ctx)
  File "/home/adam/prod/_inne/sqlfluff/.venv/lib/python3.9/site-packages/click/core.py", line 1668, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/home/adam/prod/_inne/sqlfluff/.venv/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/adam/prod/_inne/sqlfluff/.venv/lib/python3.9/site-packages/click/core.py", line 763, in invoke
    return __callback(*args, **kwargs)
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/cli/commands.py", line 347, in lint
    result = lnt.lint_string_wrapped(sys.stdin.read(), fname="stdin")
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/linter/linter.py", line 789, in lint_string_wrapped
    linted_path.add(self.lint_string(string, fname=fname, fix=fix))
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/linter/linter.py", line 668, in lint_string
    parsed = self.parse_string(in_str=in_str, fname=fname, config=config)
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/linter/linter.py", line 607, in parse_string
    return self.parse_rendered(rendered, recurse=recurse)
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/linter/linter.py", line 313, in parse_rendered
    parsed, pvs = cls._parse_tokens(
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/linter/linter.py", line 190, in _parse_tokens
    parsed: Optional[BaseSegment] = parser.parse(
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/parser/parser.py", line 32, in parse
    parsed = root_segment.parse(parse_context=ctx)
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/parser/segments/base.py", line 821, in parse
    check_still_complete(segments, m.matched_segments, m.unmatched_segments)
  File "/home/adam/prod/_inne/sqlfluff/src/sqlfluff/core/parser/helpers.py", line 30, in check_still_complete
    raise RuntimeError(
RuntimeError: Dropped elements in sequence matching! 'select id from tbl;;' != ';'

\`\`\`
## Steps to Reproduce
Run 
\`\`\`console
echo "select id from tbl;;" | sqlfluff lint -
\`\`\`
## Dialect
default (ansi)
## Version
\`\`\`
sqlfluff, version 0.6.6
Python 3.9.5
\`\`\`
## Configuration
None


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 src/sqlfluff/rules/L041.py | 8 | 5| 508 | 508 | 
| 2 | 2 src/sqlfluff/dialects/dialect_ansi.py | 1432 | 1586| 1083 | 1591 | 
| 3 | 3 src/sqlfluff/dialects/dialect_tsql.py | 64 | 113| 437 | 2028 | 
| 4 | 4 src/sqlfluff/rules/L038.py | 41 | 70| 247 | 2275 | 
| 5 | 5 src/sqlfluff/rules/L022.py | 8 | 5| 193 | 2468 | 
| 6 | 6 src/sqlfluff/core/linter/linter.py | 178 | 217| 363 | 2831 | 
| 7 | 7 src/sqlfluff/dialects/dialect_exasol.py | 175 | 262| 604 | 3435 | 
| 8 | 8 src/sqlfluff/rules/L019.py | 116 | 216| 883 | 4318 | 
| 9 | 9 src/sqlfluff/rules/L034.py | 66 | 171| 887 | 5205 | 
| 10 | 10 src/sqlfluff/rules/L044.py | 14 | 11| 445 | 5650 | 
| 11 | 11 src/sqlfluff/rules/L008.py | 35 | 76| 399 | 6049 | 
| 12 | 12 src/sqlfluff/core/errors.py | 147 | 197| 340 | 6389 | 
| 13 | 13 src/sqlfluff/rules/L023.py | 10 | 7| 244 | 6633 | 
| 14 | 14 src/sqlfluff/rules/L016.py | 447 | 523| 750 | 7383 | 
| 15 | 15 src/sqlfluff/dialects/dialect_exasol_fs.py | 0 | 104| 741 | 8124 | 
| 16 | 15 src/sqlfluff/rules/L016.py | 47 | 73| 253 | 8377 | 
| 17 | 15 src/sqlfluff/rules/L022.py | 41 | 190| 1287 | 9664 | 
| 18 | 15 src/sqlfluff/rules/L038.py | 11 | 8| 214 | 9878 | 
| 19 | 15 src/sqlfluff/rules/L016.py | 163 | 283| 848 | 10726 | 
| 20 | 16 src/sqlfluff/rules/L048.py | 9 | 6| 273 | 10999 | 
| 21 | 17 src/sqlfluff/rules/L036.py | 135 | 338| 1444 | 12443 | 
| 22 | 17 src/sqlfluff/rules/L022.py | 38 | 40| 47 | 12490 | 
| 23 | 18 src/sqlfluff/rules/L026.py | 57 | 97| 307 | 12797 | 
| 24 | 19 src/sqlfluff/dialects/dialect_snowflake.py | 151 | 223| 580 | 13377 | 
| 25 | 20 src/sqlfluff/rules/L003.py | 699 | 758| 492 | 13869 | 
| 26 | 21 src/sqlfluff/rules/L004.py | 47 | 88| 431 | 14300 | 
| 27 | 21 src/sqlfluff/dialects/dialect_exasol.py | 2983 | 3044| 341 | 14641 | 
| 28 | 21 src/sqlfluff/rules/L019.py | 13 | 10| 404 | 15045 | 
| 29 | 21 src/sqlfluff/dialects/dialect_ansi.py | 60 | 129| 713 | 15758 | 
| 30 | 21 src/sqlfluff/dialects/dialect_exasol.py | 0 | 80| 595 | 16353 | 
| 31 | 22 src/sqlfluff/rules/L001.py | 5 | 2| 530 | 16883 | 
| 32 | 22 src/sqlfluff/dialects/dialect_exasol.py | 81 | 173| 633 | 17516 | 
| 33 | 22 src/sqlfluff/rules/L023.py | 46 | 91| 327 | 17843 | 
| 34 | 23 src/sqlfluff/rules/L027.py | 31 | 62| 201 | 18044 | 
| 35 | 23 src/sqlfluff/rules/L008.py | 8 | 5| 193 | 18237 | 
| 36 | 24 src/sqlfluff/rules/L010.py | 12 | 9| 262 | 18499 | 
| 37 | 24 src/sqlfluff/rules/L036.py | 125 | 133| 95 | 18594 | 
| 38 | 25 src/sqlfluff/rules/L006.py | 40 | 73| 283 | 18877 | 
| 39 | 25 src/sqlfluff/dialects/dialect_ansi.py | 2523 | 2629| 750 | 19627 | 
| 40 | 25 src/sqlfluff/rules/L003.py | 392 | 473| 774 | 20401 | 
| 41 | 26 src/sqlfluff/dialects/dialect_mysql.py | 225 | 293| 503 | 20904 | 
| 42 | 26 src/sqlfluff/dialects/dialect_exasol.py | 1558 | 1612| 260 | 21164 | 
| 43 | 27 src/sqlfluff/rules/L015.py | 8 | 5| 173 | 21337 | 
| 44 | 27 src/sqlfluff/rules/L036.py | 21 | 44| 195 | 21532 | 
| 45 | 28 src/sqlfluff/rules/L005.py | 36 | 53| 141 | 21673 | 
| 46 | 28 src/sqlfluff/dialects/dialect_snowflake.py | 124 | 149| 245 | 21918 | 
| 47 | 28 src/sqlfluff/rules/L036.py | 59 | 91| 276 | 22194 | 
| 48 | 29 src/sqlfluff/rules/L046.py | 5 | 2| 169 | 22363 | 
| 49 | 30 src/sqlfluff/dialects/dialect_postgres.py | 77 | 138| 546 | 22909 | 
| 50 | 31 src/sqlfluff/rules/L011.py | 39 | 93| 461 | 23370 | 
| 51 | 31 src/sqlfluff/rules/L034.py | 47 | 64| 168 | 23538 | 
| 52 | 31 src/sqlfluff/rules/L003.py | 474 | 476| 490 | 24028 | 
| 53 | 31 src/sqlfluff/rules/L004.py | 11 | 8| 294 | 24322 | 
| 54 | 32 src/sqlfluff/rules/L039.py | 8 | 5| 363 | 24685 | 
| 55 | 32 src/sqlfluff/rules/L046.py | 49 | 82| 320 | 25005 | 
| 56 | 33 src/sqlfluff/rules/L028.py | 46 | 109| 522 | 25527 | 
| 57 | 34 src/sqlfluff/core/rules/base.py | 163 | 195| 275 | 25802 | 
| 58 | 35 src/sqlfluff/cli/commands.py | 694 | 767| 882 | 26684 | 
| 59 | 36 src/sqlfluff/rules/L035.py | 6 | 3| 189 | 26873 | 
| 60 | 36 src/sqlfluff/rules/L034.py | 6 | 3| 283 | 27156 | 
| 61 | 36 src/sqlfluff/dialects/dialect_postgres.py | 1759 | 1756| 264 | 27420 | 
| 62 | 36 src/sqlfluff/cli/commands.py | 769 | 789| 484 | 27904 | 
| 63 | 36 src/sqlfluff/rules/L003.py | 531 | 697| 1407 | 29311 | 
| 64 | 36 src/sqlfluff/dialects/dialect_tsql.py | 0 | 62| 424 | 29735 | 
| 65 | 37 src/sqlfluff/rules/L047.py | 58 | 129| 449 | 30184 | 
| 66 | 37 src/sqlfluff/core/linter/linter.py | 448 | 511| 462 | 30646 | 
| 67 | 38 src/sqlfluff/__init__.py | 0 | 29| 232 | 30878 | 
| 68 | 38 src/sqlfluff/cli/commands.py | 228 | 263| 238 | 31116 | 
| 69 | 39 src/sqlfluff/api/simple.py | 26 | 46| 209 | 31325 | 
| 70 | 39 src/sqlfluff/rules/L011.py | 8 | 5| 201 | 31526 | 
| 71 | 40 src/sqlfluff/rules/L043.py | 53 | 145| 747 | 32273 | 
| 72 | 40 src/sqlfluff/dialects/dialect_ansi.py | 1226 | 1223| 169 | 32442 | 
| 73 | 40 src/sqlfluff/cli/commands.py | 372 | 411| 631 | 33073 | 
| 74 | 40 src/sqlfluff/rules/L003.py | 295 | 390| 799 | 33872 | 
| 75 | 41 src/sqlfluff/rules/L018.py | 8 | 5| 215 | 34087 | 
| 76 | 41 src/sqlfluff/dialects/dialect_exasol.py | 265 | 314| 310 | 34397 | 
| 77 | 42 src/sqlfluff/rules/L014.py | 12 | 9| 189 | 34586 | 
| 78 | 43 src/sqlfluff/rules/L029.py | 8 | 5| 258 | 34844 | 
| 79 | 43 src/sqlfluff/dialects/dialect_tsql.py | 116 | 148| 227 | 35071 | 
| 80 | 44 src/sqlfluff/core/parser/grammar/delimited.py | 68 | 251| 1633 | 36704 | 
| 81 | 45 src/sqlfluff/rules/L002.py | 42 | 66| 195 | 36899 | 
| 82 | 46 src/sqlfluff/rules/L045.py | 9 | 6| 418 | 37317 | 
| 83 | 46 src/sqlfluff/core/linter/linter.py | 629 | 648| 147 | 37464 | 
| 84 | 46 src/sqlfluff/rules/L006.py | 75 | 194| 808 | 38272 | 
| 85 | 46 src/sqlfluff/rules/L035.py | 35 | 69| 354 | 38626 | 
| 86 | 47 src/sqlfluff/cli/formatters.py | 161 | 188| 301 | 38927 | 
| 87 | 47 src/sqlfluff/core/parser/grammar/delimited.py | 67 | 250| 62 | 38989 | 
| 88 | 47 src/sqlfluff/rules/L014.py | 27 | 75| 277 | 39266 | 
| 89 | 47 src/sqlfluff/rules/L002.py | 9 | 6| 208 | 39474 | 
| 90 | 47 src/sqlfluff/core/rules/base.py | 357 | 409| 483 | 39957 | 
| 91 | 47 src/sqlfluff/dialects/dialect_ansi.py | 1899 | 1919| 187 | 40144 | 
| 92 | 47 src/sqlfluff/rules/L003.py | 11 | 8| 410 | 40554 | 
| 93 | 48 src/sqlfluff/rules/L024.py | 7 | 4| 200 | 40754 | 
| 94 | 48 src/sqlfluff/core/linter/linter.py | 763 | 791| 288 | 41042 | 
| 95 | 48 src/sqlfluff/dialects/dialect_ansi.py | 0 | 58| 348 | 41390 | 
| 96 | 48 src/sqlfluff/dialects/dialect_exasol.py | 1713 | 1710| 187 | 41577 | 
| 97 | 48 src/sqlfluff/rules/L005.py | 6 | 3| 179 | 41756 | 
| 98 | 49 src/sqlfluff/rules/L017.py | 6 | 3| 415 | 42171 | 
| 99 | 49 src/sqlfluff/dialects/dialect_postgres.py | 27 | 75| 672 | 42843 | 
| 100 | 50 src/sqlfluff/core/parser/grammar/sequence.py | 48 | 61| 907 | 43750 | 
| 101 | 51 src/sqlfluff/dialects/dialect_bigquery.py | 0 | 76| 632 | 44382 | 
| 102 | 51 src/sqlfluff/rules/L015.py | 31 | 77| 363 | 44745 | 
| 103 | 52 src/sqlfluff/rules/L049.py | 35 | 103| 516 | 45261 | 
| 104 | 52 src/sqlfluff/dialects/dialect_postgres.py | 777 | 953| 1186 | 46447 | 
| 105 | 52 src/sqlfluff/rules/L016.py | 284 | 347| 549 | 46996 | 
| 106 | 53 src/sqlfluff/rules/L020.py | 85 | 117| 239 | 47235 | 
| 107 | 53 src/sqlfluff/dialects/dialect_postgres.py | 1390 | 1506| 612 | 47847 | 
| 108 | 54 src/sqlfluff/rules/L007.py | 48 | 105| 545 | 48392 | 
| 109 | 54 src/sqlfluff/rules/L044.py | 61 | 75| 134 | 48526 | 
| 110 | 55 src/sqlfluff/dialects/dialect_teradata.py | 751 | 768| 136 | 48662 | 
| 111 | 56 src/sqlfluff/rules/L009.py | 8 | 5| 322 | 48984 | 
| 112 | 56 src/sqlfluff/dialects/dialect_ansi.py | 3026 | 3047| 105 | 49089 | 
| 113 | 56 src/sqlfluff/rules/L003.py | 127 | 234| 903 | 49992 | 


## Missing Patch Files

 * 1: src/sqlfluff/core/parser/helpers.py

### Hint

```
Sounds similar to #1458 where we should handle "empty" statement/files better?
Nope, that's the different issue. I doubt that solving one of them would help in other one. I think both issues should stay, just in the case.
But what do you think @tunetheweb - should it just ignore these `;;` or raise something like `Found unparsable section:`? 
Just tested and in BigQuery it's an error.
Interestingly Oracle is fine with it.

I think it should be raised as `Found unparsable section`.
```

## Patch

```diff
diff --git a/src/sqlfluff/core/parser/helpers.py b/src/sqlfluff/core/parser/helpers.py
--- a/src/sqlfluff/core/parser/helpers.py
+++ b/src/sqlfluff/core/parser/helpers.py
@@ -2,6 +2,7 @@
 
 from typing import Tuple, List, Any, Iterator, TYPE_CHECKING
 
+from sqlfluff.core.errors import SQLParseError
 from sqlfluff.core.string_helpers import curtail_string
 
 if TYPE_CHECKING:
@@ -26,11 +27,11 @@ def check_still_complete(
     """Check that the segments in are the same as the segments out."""
     initial_str = join_segments_raw(segments_in)
     current_str = join_segments_raw(matched_segments + unmatched_segments)
-    if initial_str != current_str:  # pragma: no cover
-        raise RuntimeError(
-            "Dropped elements in sequence matching! {!r} != {!r}".format(
-                initial_str, current_str
-            )
+
+    if initial_str != current_str:
+        raise SQLParseError(
+            f"Could not parse: {current_str}",
+            segment=unmatched_segments[0],
         )
     return True
 

```

## Test Patch

```diff
diff --git a/test/dialects/ansi_test.py b/test/dialects/ansi_test.py
--- a/test/dialects/ansi_test.py
+++ b/test/dialects/ansi_test.py
@@ -3,7 +3,7 @@
 import pytest
 import logging
 
-from sqlfluff.core import FluffConfig, Linter
+from sqlfluff.core import FluffConfig, Linter, SQLParseError
 from sqlfluff.core.parser import Lexer
 
 
@@ -214,3 +214,29 @@ def test__dialect__ansi_parse_indented_joins(sql_string, indented_joins, meta_lo
         idx for idx, raw_seg in enumerate(parsed.tree.iter_raw_seg()) if raw_seg.is_meta
     )
     assert res_meta_locs == meta_loc
+
+
+@pytest.mark.parametrize(
+    "raw,expected_message",
+    [
+        (";;", "Line 1, Position 1: Found unparsable section: ';;'"),
+        ("select id from tbl;", ""),
+        ("select id from tbl;;", "Could not parse: ;"),
+        ("select id from tbl;;;;;;", "Could not parse: ;;;;;"),
+        ("select id from tbl;select id2 from tbl2;", ""),
+        (
+            "select id from tbl;;select id2 from tbl2;",
+            "Could not parse: ;select id2 from tbl2;",
+        ),
+    ],
+)
+def test__dialect__ansi_multiple_semicolons(raw: str, expected_message: str) -> None:
+    """Multiple semicolons should be properly handled."""
+    lnt = Linter()
+    parsed = lnt.parse_string(raw)
+
+    assert len(parsed.violations) == (1 if expected_message else 0)
+    if expected_message:
+        violation = parsed.violations[0]
+        assert isinstance(violation, SQLParseError)
+        assert violation.desc() == expected_message

```


## Code snippets

### 1 - src/sqlfluff/rules/L041.py:

Start line: 8, End line: 5

```python
"""Implementation of Rule L040."""

from sqlfluff.core.parser import NewlineSegment, WhitespaceSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L041(BaseRule):
    """SELECT clause modifiers such as DISTINCT must be on the same line as SELECT.

    | **Anti-pattern**

    .. code-block:: sql

        select
            distinct a,
            b
        from x


    | **Best practice**

    .. code-block:: sql

        select distinct
            a,
            b
        from x

    """

    def _eval(self, segment, **kwargs):
        """Select clause modifiers must appear on same line as SELECT."""
        if segment.is_type("select_clause"):
            # Does the select clause have modifiers?
            select_modifier = segment.get_child("select_clause_modifier")
            if not select_modifier:
                return None  # No. We're done.
            select_modifier_idx = segment.segments.index(select_modifier)

            # Does the select clause contain a newline?
            newline = segment.get_child("newline")
            if not newline:
                return None  # No. We're done.
            newline_idx = segment.segments.index(newline)

            # Is there a newline before the select modifier?
            if newline_idx > select_modifier_idx:
                return None  # No, we're done.

            # Yes to all the above. We found an issue.

            # E.g.: " DISTINCT\n"
            replace_newline_with = [
                WhitespaceSegment(),
                select_modifier,
                NewlineSegment(),
            ]
            fixes = [
                # E.g. "\n" -> " DISTINCT\n.
                LintFix("edit", newline, replace_newline_with),
                # E.g. "DISTINCT" -> X
                LintFix("delete", select_modifier),
            ]

            # E.g. " " after "DISTINCT"
            ws_to_delete = segment.select_children(
                start_seg=select_modifier,
                select_if=lambda s: s.is_type("whitespace"),
                loop_while=lambda s: s.is_type("whitespace") or s.is_meta,
            )

            # E.g. " " -> X
            fixes += [LintFix("delete", ws) for ws in ws_to_delete]
            return LintResult(
                anchor=segment,
                fixes=fixes,
            )
```
### 2 - src/sqlfluff/dialects/dialect_ansi.py:

Start line: 1432, End line: 1586

```python
ansi_dialect.add(
    # Expression_A_Grammar https://www.cockroachlabs.com/docs/v20.2/sql-grammar.html#a_expr
    Expression_A_Grammar=Sequence(
        OneOf(
            Ref("Expression_C_Grammar"),
            Sequence(
                OneOf(
                    Ref("PositiveSegment"),
                    Ref("NegativeSegment"),
                    # Ref('TildeSegment'),
                    "NOT",
                    "PRIOR",  # used in CONNECT BY clauses (EXASOL, Snowflake, Postgres...)
                ),
                Ref("Expression_C_Grammar"),
            ),
        ),
        AnyNumberOf(
            OneOf(
                Sequence(
                    OneOf(
                        Sequence(
                            Ref.keyword("NOT", optional=True),
                            Ref("LikeGrammar"),
                        ),
                        Sequence(
                            Ref("BinaryOperatorGrammar"),
                            Ref.keyword("NOT", optional=True),
                        ),
                        # We need to add a lot more here...
                    ),
                    Ref("Expression_C_Grammar"),
                    Sequence(
                        Ref.keyword("ESCAPE"),
                        Ref("Expression_C_Grammar"),
                        optional=True,
                    ),
                ),
                Sequence(
                    Ref.keyword("NOT", optional=True),
                    "IN",
                    Bracketed(
                        OneOf(
                            Delimited(
                                Ref("Expression_A_Grammar"),
                            ),
                            Ref("SelectableGrammar"),
                            ephemeral_name="InExpression",
                        )
                    ),
                ),
                Sequence(
                    Ref.keyword("NOT", optional=True),
                    "IN",
                    Ref("FunctionSegment"),  # E.g. UNNEST()
                ),
                Sequence(
                    "IS",
                    Ref.keyword("NOT", optional=True),
                    Ref("IsClauseGrammar"),
                ),
                Ref("IsNullGrammar"),
                Ref("NotNullGrammar"),
                Sequence(
                    # e.g. NOT EXISTS, but other expressions could be met as
                    # well by inverting the condition with the NOT operator
                    "NOT",
                    Ref("Expression_C_Grammar"),
                ),
                Sequence(
                    Ref.keyword("NOT", optional=True),
                    "BETWEEN",
                    # In a between expression, we're restricted to arithmetic operations
                    # because if we look for all binary operators then we would match AND
                    # as both an operator and also as the delimiter within the BETWEEN
                    # expression.
                    Ref("Expression_C_Grammar"),
                    AnyNumberOf(
                        Sequence(
                            Ref("ArithmeticBinaryOperatorGrammar"),
                            Ref("Expression_C_Grammar"),
                        )
                    ),
                    "AND",
                    Ref("Expression_C_Grammar"),
                    AnyNumberOf(
                        Sequence(
                            Ref("ArithmeticBinaryOperatorGrammar"),
                            Ref("Expression_C_Grammar"),
                        )
                    ),
                ),
            )
        ),
    ),
    # CockroachDB defines Expression_B_Grammar. The SQLFluff implementation of
    # expression parsing pulls that logic into Expression_A_Grammar and so there's
    # currently no need to define Expression_B.
    # https://www.cockroachlabs.com/docs/v20.2/sql-grammar.htm#b_expr
    #
    # Expression_C_Grammar https://www.cockroachlabs.com/docs/v20.2/sql-grammar.htm#c_expr
    Expression_C_Grammar=OneOf(
        Sequence(
            "EXISTS", Bracketed(Ref("SelectStatementSegment"))
        ),  # should be first priority, otherwise EXISTS() would be matched as a function
        Ref("Expression_D_Grammar"),
        Ref("CaseExpressionSegment"),
    ),
    # Expression_D_Grammar https://www.cockroachlabs.com/docs/v20.2/sql-grammar.htm#d_expr
    Expression_D_Grammar=Sequence(
        OneOf(
            Ref("BareFunctionSegment"),
            Ref("FunctionSegment"),
            Bracketed(
                OneOf(
                    # We're using the expression segment here rather than the grammar so
                    # that in the parsed structure we get nested elements.
                    Ref("ExpressionSegment"),
                    Ref("SelectableGrammar"),
                    Delimited(
                        Ref(
                            "ColumnReferenceSegment"
                        ),  # WHERE (a,b,c) IN (select a,b,c FROM...)
                        Ref(
                            "FunctionSegment"
                        ),  # WHERE (a, substr(b,1,3)) IN (select c,d FROM...)
                        Ref("LiteralGrammar"),  # WHERE (a, 2) IN (SELECT b, c FROM ...)
                    ),
                    ephemeral_name="BracketedExpression",
                ),
            ),
            # Allow potential select statement without brackets
            Ref("SelectStatementSegment"),
            Ref("LiteralGrammar"),
            Ref("IntervalExpressionSegment"),
            Ref("ColumnReferenceSegment"),
            Sequence(
                Ref("SimpleArrayTypeGrammar", optional=True), Ref("ArrayLiteralSegment")
            ),
            Sequence(
                Ref("DatatypeSegment"),
                OneOf(
                    Ref("QuotedLiteralSegment"),
                    Ref("NumericLiteralSegment"),
                    Ref("BooleanLiteralGrammar"),
                    Ref("NullLiteralSegment"),
                    Ref("DateTimeLiteralGrammar"),
                ),
            ),
        ),
        Ref("Accessor_Grammar", optional=True),
        AnyNumberOf(Ref("ShorthandCastSegment")),
        allow_gaps=True,
    ),
    Accessor_Grammar=AnyNumberOf(Ref("ArrayAccessorSegment")),
)
```
### 3 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 64, End line: 113

```python
tsql_dialect.replace(
    # Below delimiterstatement might need to be removed in the future as delimiting
    # is optional with semicolon and GO is a end of statement indicator.
    DelimiterSegment=OneOf(
        Sequence(Ref("SemicolonSegment"), Ref("GoStatementSegment")),
        Ref("SemicolonSegment"),
        Ref("GoStatementSegment"),
    ),
    SingleIdentifierGrammar=OneOf(
        Ref("NakedIdentifierSegment"),
        Ref("QuotedIdentifierSegment"),
        Ref("BracketedIdentifierSegment"),
    ),
    LiteralGrammar=OneOf(
        Ref("QuotedLiteralSegment"),
        Ref("QuotedLiteralSegmentWithN"),
        Ref("NumericLiteralSegment"),
        Ref("BooleanLiteralGrammar"),
        Ref("QualifiedNumericLiteralSegment"),
        # NB: Null is included in the literals, because it is a keyword which
        # can otherwise be easily mistaken for an identifier.
        Ref("NullLiteralSegment"),
        Ref("DateTimeLiteralGrammar"),
    ),
    ParameterNameSegment=RegexParser(
        r"[@][A-Za-z0-9_]+", CodeSegment, name="parameter", type="parameter"
    ),
    FunctionNameIdentifierSegment=RegexParser(
        r"[A-Z][A-Z0-9_]*|\[[A-Z][A-Z0-9_]*\]",
        CodeSegment,
        name="function_name_identifier",
        type="function_name_identifier",
    ),
    DatatypeIdentifierSegment=Ref("SingleIdentifierGrammar"),
    PrimaryKeyGrammar=Sequence(
        "PRIMARY", "KEY", OneOf("CLUSTERED", "NONCLUSTERED", optional=True)
    ),
    FromClauseTerminatorGrammar=OneOf(
        "WHERE",
        "LIMIT",
        Sequence("GROUP", "BY"),
        Sequence("ORDER", "BY"),
        "HAVING",
        "PIVOT",
        "UNPIVOT",
        Ref("SetOperatorSegment"),
        Ref("WithNoSchemaBindingClauseSegment"),
    ),
    JoinKeywords=OneOf("JOIN", "APPLY", Sequence("OUTER", "APPLY")),
)
```
### 4 - src/sqlfluff/rules/L038.py:

Start line: 41, End line: 70

```python
@document_configuration
@document_fix_compatible
class Rule_L038(BaseRule):

    def _eval(self, segment, parent_stack, **kwargs):
        """Trailing commas within select clause."""
        if segment.is_type("select_clause"):
            # Iterate content to find last element
            last_content = None
            for seg in segment.segments:
                if seg.is_code:
                    last_content = seg

            # What mode are we in?
            if self.select_clause_trailing_comma == "forbid":
                # Is it a comma?
                if last_content.is_type("comma"):
                    return LintResult(
                        anchor=last_content,
                        fixes=[LintFix("delete", last_content)],
                        description="Trailing comma in select statement forbidden",
                    )
            elif self.select_clause_trailing_comma == "require":
                if not last_content.is_type("comma"):
                    new_comma = SymbolSegment(",", name="comma", type="comma")
                    return LintResult(
                        anchor=last_content,
                        fixes=[
                            LintFix("edit", last_content, [last_content, new_comma])
                        ],
                        description="Trailing comma in select statement required",
                    )
        return None
```
### 5 - src/sqlfluff/rules/L022.py:

Start line: 8, End line: 5

```python
"""Implementation of Rule L022."""

from sqlfluff.core.parser import NewlineSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L022(BaseRule):
    """Blank line expected but not found after CTE closing bracket.

    | **Anti-pattern**
    | There is no blank line after the CTE closing bracket. In queries with many
    | CTEs this hinders readability.

    .. code-block:: sql

        WITH plop AS (
            SELECT * FROM foo
        )
        SELECT a FROM plop

    | **Best practice**
    | Add a blank line.

    .. code-block:: sql

        WITH plop AS (
            SELECT * FROM foo
        )

        SELECT a FROM plop

    """

    config_keywords = ["comma_style"]
```
### 6 - src/sqlfluff/core/linter/linter.py:

Start line: 178, End line: 217

```python
class Linter:

    @staticmethod
    def _parse_tokens(
        tokens: Sequence[BaseSegment],
        config: FluffConfig,
        recurse: bool = True,
        fname: Optional[str] = None,
    ) -> Tuple[Optional[BaseSegment], List[SQLParseError]]:
        parser = Parser(config=config)
        violations = []
        # Parse the file and log any problems
        try:
            parsed: Optional[BaseSegment] = parser.parse(
                tokens, recurse=recurse, fname=fname
            )
        except SQLParseError as err:
            linter_logger.info("PARSING FAILED! : %s", err)
            violations.append(err)
            return None, violations

        if parsed:
            linter_logger.info("\n###\n#\n# {}\n#\n###".format("Parsed Tree:"))
            linter_logger.info("\n" + parsed.stringify())
            # We may succeed parsing, but still have unparsable segments. Extract them here.
            for unparsable in parsed.iter_unparsables():
                # No exception has been raised explicitly, but we still create one here
                # so that we can use the common interface
                violations.append(
                    SQLParseError(
                        "Line {0[0]}, Position {0[1]}: Found unparsable section: {1!r}".format(
                            unparsable.pos_marker.working_loc,
                            unparsable.raw
                            if len(unparsable.raw) < 40
                            else unparsable.raw[:40] + "...",
                        ),
                        segment=unparsable,
                    )
                )
                linter_logger.info("Found unparsable segment...")
                linter_logger.info(unparsable.stringify())
        return parsed, violations
```
### 7 - src/sqlfluff/dialects/dialect_exasol.py:

Start line: 175, End line: 262

```python
exasol_dialect.replace(
    SingleIdentifierGrammar=OneOf(
        Ref("LocalIdentifierSegment"),
        Ref("NakedIdentifierSegment"),
        Ref("QuotedIdentifierSegment"),
        Ref("EscapedIdentifierSegment"),
    ),
    ParameterNameSegment=RegexParser(
        r"\"?[A-Z][A-Z0-9_]*\"?",
        CodeSegment,
        name="parameter",
        type="parameter",
    ),
    LikeGrammar=Ref.keyword("LIKE"),
    IsClauseGrammar=OneOf(
        "NULL",
        Ref("BooleanLiteralGrammar"),
    ),
    SelectClauseSegmentGrammar=Sequence(
        "SELECT",
        Ref("SelectClauseModifierSegment", optional=True),
        Indent,
        Delimited(
            Ref("SelectClauseElementSegment"),
            allow_trailing=True,
            optional=True,  # optional in favor of SELECT INVALID....
        ),
        OneOf(Ref("WithInvalidUniquePKSegment"), Ref("IntoTableSegment"), optional=True)
        # NB: The Dedent for the indent above lives in the
        # SelectStatementSegment so that it sits in the right
        # place corresponding to the whitespace.
    ),
    SelectClauseElementTerminatorGrammar=OneOf(
        Sequence(
            Ref.keyword("WITH", optional=True),
            "INVALID",
            OneOf("UNIQUE", Ref("PrimaryKeyGrammar"), Ref("ForeignKeyGrammar")),
        ),
        Sequence("INTO", "TABLE"),
        "FROM",
        "WHERE",
        "ORDER",
        "LIMIT",
        Ref("CommaSegment"),
        Ref("SetOperatorSegment"),
    ),
    FromClauseTerminatorGrammar=OneOf(
        "WHERE",
        "CONNECT",
        "START",
        "PREFERRING",
        "LIMIT",
        Sequence("GROUP", "BY"),
        Sequence("ORDER", "BY"),
        "HAVING",
        "QUALIFY",
        Ref("SetOperatorSegment"),
    ),
    WhereClauseTerminatorGrammar=OneOf(
        "CONNECT",
        "START",
        "PREFERRING",
        "LIMIT",
        Sequence("GROUP", "BY"),
        Sequence("ORDER", "BY"),
        "HAVING",
        "QUALIFY",
        Ref("SetOperatorSegment"),
    ),
    DateTimeLiteralGrammar=Sequence(
        OneOf("DATE", "TIMESTAMP"), Ref("QuotedLiteralSegment")
    ),
    CharCharacterSetSegment=OneOf(
        Ref.keyword("UTF8"),
        Ref.keyword("ASCII"),
    ),
    PreTableFunctionKeywordsGrammar=Ref.keyword("TABLE"),
    BooleanLiteralGrammar=OneOf(
        Ref("TrueSegment"), Ref("FalseSegment"), Ref("UnknownSegment")
    ),
    PostFunctionGrammar=OneOf(
        Ref("EmitsGrammar"),  # e.g. JSON_EXTRACT()
        Sequence(
            Sequence(OneOf("IGNORE", "RESPECT"), "NULLS", optional=True),
            Ref("OverClauseSegment"),
        ),
    ),
)
```
### 8 - src/sqlfluff/rules/L019.py:

Start line: 116, End line: 216

```python
@document_fix_compatible
@document_configuration
class Rule_L019(BaseRule):

    def _eval(self, segment, raw_stack, memory, **kwargs):
        # ... other code

        if self.comma_style == "trailing":
            # A comma preceded by a new line == a leading comma
            if segment.is_type("comma"):
                last_seg = self._last_code_seg(raw_stack)
                if last_seg.is_type("newline"):
                    # Recorded where the fix should be applied
                    memory["last_leading_comma_seg"] = segment
                    last_comment_seg = self._last_comment_seg(raw_stack)
                    inline_comment = (
                        last_comment_seg.pos_marker.line_no
                        == last_seg.pos_marker.line_no
                        if last_comment_seg
                        else False
                    )
                    # If we have a comment right before the newline, then anchor
                    # the fix at the comment instead
                    memory["anchor_for_new_trailing_comma_seg"] = (
                        last_seg if not inline_comment else last_comment_seg
                    )
                    # Trigger fix routine
                    memory["insert_trailing_comma"] = True
                    memory["whitespace_deletions"] = []
                    return LintResult(memory=memory)
            # Have we found a leading comma violation?
            if memory["insert_trailing_comma"]:
                # Search for trailing whitespace to delete after the leading
                # comma violation
                if segment.is_type("whitespace"):
                    memory["whitespace_deletions"] += [segment]
                    return LintResult(memory=memory)
                else:
                    # We've run out of whitespace to delete, time to fix
                    last_leading_comma_seg = memory["last_leading_comma_seg"]
                    # Scan backwards to find the last code segment, skipping
                    # over lines that are either entirely blank or just a
                    # comment. We want to place the comma immediately after it.
                    last_code_seg = None
                    while last_code_seg is None or last_code_seg.is_type("newline"):
                        last_code_seg = self._last_code_seg(
                            raw_stack[
                                : raw_stack.index(
                                    last_code_seg
                                    if last_code_seg
                                    else memory["last_leading_comma_seg"]
                                )
                            ]
                        )
                    return LintResult(
                        anchor=last_leading_comma_seg,
                        description="Found leading comma. Expected only trailing.",
                        fixes=[
                            LintFix("delete", last_leading_comma_seg),
                            *[
                                LintFix("delete", d)
                                for d in memory["whitespace_deletions"]
                            ],
                            LintFix(
                                "edit",
                                last_code_seg,
                                # Reuse the previous leading comma violation to
                                # create a new trailing comma
                                [last_code_seg, last_leading_comma_seg],
                            ),
                        ],
                    )

        elif self.comma_style == "leading":
            # A new line preceded by a comma == a trailing comma
            if segment.is_type("newline"):
                last_seg = self._last_code_seg(raw_stack)
                # no code precedes the current position: no issue
                if last_seg is None:
                    return None
                if last_seg.is_type("comma"):
                    # Trigger fix routine
                    memory["insert_leading_comma"] = True
                    # Record where the fix should be applied
                    memory["last_trailing_comma_segment"] = last_seg
                    return LintResult(memory=memory)
            # Have we found a trailing comma violation?
            if memory["insert_leading_comma"]:
                # Only insert the comma here if this isn't a comment/whitespace segment
                if segment.is_code:
                    last_comma_seg = memory["last_trailing_comma_segment"]
                    # Create whitespace to insert after the new leading comma
                    new_whitespace_seg = WhitespaceSegment()
                    return LintResult(
                        anchor=last_comma_seg,
                        description="Found trailing comma. Expected only leading.",
                        fixes=[
                            LintFix("delete", anchor=last_comma_seg),
                            LintFix(
                                "create",
                                anchor=segment,
                                edit=[last_comma_seg, new_whitespace_seg],
                            ),
                        ],
                    )
        # Otherwise, no issue
        return None
```
### 9 - src/sqlfluff/rules/L034.py:

Start line: 66, End line: 171

```python
@document_fix_compatible
class Rule_L034(BaseRule):

    def _eval(self, segment, parent_stack, **kwargs):
        # ... other code

        if segment.is_type("select_clause"):
            # Ignore select clauses which belong to:
            # - set expression, which is most commonly a union
            # - insert_statement
            # - create table statement
            #
            # In each of these contexts, the order of columns in a select should
            # be preserved.
            if len(parent_stack) >= 2 and parent_stack[-2].is_type(
                "insert_statement", "set_expression"
            ):
                return None
            if len(parent_stack) >= 3 and parent_stack[-3].is_type(
                "create_table_statement"
            ):
                return None

            select_clause_segment = segment
            select_target_elements = segment.get_children("select_clause_element")
            if not select_target_elements:
                return None

            # Iterate through all the select targets to find any order violations
            for segment in select_target_elements:
                # The band index of the current segment in select_element_order_preference
                self.current_element_band = None

                # Compare the segment to the bands in select_element_order_preference
                for i, band in enumerate(select_element_order_preference):
                    for e in band:
                        # Identify simple select target
                        if segment.get_child(e):
                            self._validate(i, segment)

                        # Identify function
                        elif type(e) == tuple and e[0] == "function":
                            try:
                                if (
                                    segment.get_child("function")
                                    .get_child("function_name")
                                    .raw
                                    == e[1]
                                ):
                                    self._validate(i, segment)
                            except AttributeError:
                                # If the segment doesn't match
                                pass

                        # Identify simple expression
                        elif type(e) == tuple and e[0] == "expression":
                            try:
                                if (
                                    segment.get_child("expression").get_child(e[1])
                                    and segment.get_child("expression").segments[0].type
                                    in (
                                        "column_reference",
                                        "object_reference",
                                        "literal",
                                    )
                                    # len == 2 to ensure the expression is 'simple'
                                    and len(segment.get_child("expression").segments)
                                    == 2
                                ):
                                    self._validate(i, segment)
                            except AttributeError:
                                # If the segment doesn't match
                                pass

                # If the target doesn't exist in select_element_order_preference then it is 'complex' and must go last
                if self.current_element_band is None:
                    self.seen_band_elements[-1].append(segment)

            if self.violation_exists:
                # Create a list of all the edit fixes
                # We have to do this at the end of iterating through all the select_target_elements to get the order correct
                # This means we can't add a lint fix to each individual LintResult as we go
                ordered_select_target_elements = [
                    segment for band in self.seen_band_elements for segment in band
                ]
                # TODO: The "if" in the loop below compares corresponding items
                # to avoid creating "do-nothing" edits. A potentially better
                # approach would leverage difflib.SequenceMatcher.get_opcodes(),
                # which generates a list of edit actions (similar to the
                # command-line "diff" tool in Linux). This is more complex to
                # implement, but minimizing the number of LintFixes makes the
                # final application of patches (in "sqlfluff fix") more robust.
                fixes = [
                    LintFix(
                        "edit",
                        initial_select_target_element,
                        replace_select_target_element,
                    )
                    for initial_select_target_element, replace_select_target_element in zip(
                        select_target_elements, ordered_select_target_elements
                    )
                    if initial_select_target_element
                    is not replace_select_target_element
                ]
                # Anchoring on the select statement segment ensures that
                # select statements which include macro targets are ignored
                # when ignore_templated_areas is set
                lint_result = LintResult(anchor=select_clause_segment, fixes=fixes)
                self.violation_buff = [lint_result]

        return self.violation_buff or None
```
### 10 - src/sqlfluff/rules/L044.py:

Start line: 14, End line: 11

```python
"""Implementation of Rule L044."""
from typing import Dict, List

from sqlfluff.core.rules.analysis.select_crawler import SelectCrawler
from sqlfluff.core.dialects.base import Dialect
from sqlfluff.core.rules.base import BaseRule, LintResult


class RuleFailure(Exception):
    """Exception class for reporting lint failure inside deeply nested code."""

    pass


class Rule_L044(BaseRule):
    """Query produces an unknown number of result columns.

    | **Anti-pattern**
    | Querying all columns using `*` produces a query result where the number
    | or ordering of columns changes if the upstream table's schema changes.
    | This should generally be avoided because it can cause slow performance,
    | cause important schema changes to go undetected, or break production code.
    | For example:
    | * If a query does `SELECT t.*` and is expected to return columns `a`, `b`,
    |   and `c`, the actual columns returned will be wrong/different if columns
    |   are added to or deleted from the input table.
    | * `UNION` and `DIFFERENCE` clauses require the inputs have the same number
    |   of columns (and compatible types).
    | * `JOIN` queries may break due to new column name conflicts, e.g. the
    |   query references a column "c" which initially existed in only one input
    |   table but a column of the same name is added to another table.
    | * `CREATE TABLE (<<column schema>>) AS SELECT *`


    .. code-block:: sql

        WITH cte AS (
            SELECT * FROM foo
        )

        SELECT * FROM cte
        UNION
        SELECT a, b FROM t

    | **Best practice**
    | Somewhere along the "path" to the source data, specify columns explicitly.

    .. code-block:: sql

        WITH cte AS (
            SELECT * FROM foo
        )

        SELECT a, b FROM cte
        UNION
        SELECT a, b FROM t

    """

    _works_on_unparsable = False
```
