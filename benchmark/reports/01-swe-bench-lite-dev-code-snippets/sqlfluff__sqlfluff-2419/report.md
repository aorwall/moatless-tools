# sqlfluff__sqlfluff-2419

| **sqlfluff/sqlfluff** | `f1dba0e1dd764ae72d67c3d5e1471cf14d3db030` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/sqlfluff/rules/L060.py b/src/sqlfluff/rules/L060.py
--- a/src/sqlfluff/rules/L060.py
+++ b/src/sqlfluff/rules/L060.py
@@ -59,4 +59,8 @@ def _eval(self, context: RuleContext) -> Optional[LintResult]:
             ],
         )
 
-        return LintResult(context.segment, [fix])
+        return LintResult(
+            anchor=context.segment,
+            fixes=[fix],
+            description=f"Use 'COALESCE' instead of '{context.segment.raw_upper}'.",
+        )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/sqlfluff/rules/L060.py | 62 | 64 | - | 1 | -


## Problem Statement

```
Rule L060 could give a specific error message
At the moment rule L060 flags something like this:

\`\`\`
L:  21 | P:   9 | L060 | Use 'COALESCE' instead of 'IFNULL' or 'NVL'.
\`\`\`

Since we likely know the wrong word, it might be nice to actually flag that instead of both `IFNULL` and `NVL` - like most of the other rules do.

That is it should flag this:

\`\`\`
L:  21 | P:   9 | L060 | Use 'COALESCE' instead of 'IFNULL'.
\`\`\`
 Or this:

\`\`\`
L:  21 | P:   9 | L060 | Use 'COALESCE' instead of 'NVL'.
\`\`\`

As appropriate.

What do you think @jpy-git ?


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | **1 src/sqlfluff/rules/L060.py** | 39 | 62| 191 | 191 | 
| 2 | **1 src/sqlfluff/rules/L060.py** | 9 | 6| 258 | 449 | 
| 3 | 2 src/sqlfluff/rules/L043.py | 127 | 258| 992 | 1441 | 
| 4 | 3 src/sqlfluff/rules/L035.py | 37 | 73| 388 | 1829 | 
| 5 | 3 src/sqlfluff/rules/L035.py | 8 | 5| 211 | 2040 | 
| 6 | 4 src/sqlfluff/rules/L057.py | 50 | 112| 468 | 2508 | 
| 7 | 5 src/sqlfluff/rules/L040.py | 11 | 8| 270 | 2778 | 
| 8 | 6 src/sqlfluff/rules/L047.py | 60 | 136| 479 | 3257 | 
| 9 | 7 src/sqlfluff/rules/L025.py | 61 | 81| 204 | 3461 | 
| 10 | 8 src/sqlfluff/rules/L010.py | 135 | 183| 508 | 3969 | 
| 11 | 8 src/sqlfluff/rules/L043.py | 15 | 12| 381 | 4350 | 
| 12 | 9 src/sqlfluff/rules/L003.py | 385 | 401| 163 | 4513 | 
| 13 | 10 src/sqlfluff/rules/L016.py | 470 | 553| 770 | 5283 | 
| 14 | 11 src/sqlfluff/rules/L027.py | 31 | 73| 282 | 5565 | 
| 15 | 12 src/sqlfluff/rules/L059.py | 11 | 8| 425 | 5990 | 
| 16 | 13 src/sqlfluff/rules/L029.py | 34 | 65| 215 | 6205 | 
| 17 | 14 src/sqlfluff/rules/L054.py | 9 | 6| 378 | 6583 | 
| 18 | 15 src/sqlfluff/rules/L026.py | 58 | 92| 310 | 6893 | 
| 19 | 15 src/sqlfluff/rules/L003.py | 564 | 566| 227 | 7120 | 
| 20 | 15 src/sqlfluff/rules/L003.py | 796 | 844| 453 | 7573 | 
| 21 | 16 src/sqlfluff/rules/L011.py | 43 | 107| 505 | 8078 | 
| 22 | 17 src/sqlfluff/rules/L006.py | 105 | 206| 681 | 8759 | 
| 23 | 18 src/sqlfluff/rules/L056.py | 51 | 79| 231 | 8990 | 
| 24 | 18 src/sqlfluff/rules/L057.py | 11 | 8| 248 | 9238 | 
| 25 | 19 src/sqlfluff/rules/L034.py | 49 | 71| 196 | 9434 | 
| 26 | 20 src/sqlfluff/rules/L019.py | 132 | 230| 883 | 10317 | 
| 27 | 21 src/sqlfluff/rules/L030.py | 11 | 8| 256 | 10573 | 
| 28 | 22 src/sqlfluff/rules/L049.py | 9 | 6| 205 | 10778 | 
| 29 | 22 src/sqlfluff/rules/L049.py | 35 | 117| 645 | 11423 | 
| 30 | 23 src/sqlfluff/rules/L044.py | 64 | 78| 134 | 11557 | 
| 31 | 23 src/sqlfluff/rules/L034.py | 73 | 182| 917 | 12474 | 
| 32 | 23 src/sqlfluff/rules/L025.py | 116 | 137| 287 | 12761 | 
| 33 | 23 src/sqlfluff/rules/L043.py | 76 | 108| 233 | 12994 | 
| 34 | 23 src/sqlfluff/rules/L010.py | 53 | 134| 691 | 13685 | 
| 35 | 23 src/sqlfluff/rules/L029.py | 8 | 5| 182 | 13867 | 
| 36 | 23 src/sqlfluff/rules/L010.py | 12 | 9| 282 | 14149 | 
| 37 | 23 src/sqlfluff/rules/L003.py | 587 | 625| 386 | 14535 | 
| 38 | 23 src/sqlfluff/rules/L054.py | 81 | 155| 554 | 15089 | 
| 39 | 23 src/sqlfluff/rules/L003.py | 485 | 563| 708 | 15797 | 
| 40 | 24 src/sqlfluff/rules/L039.py | 10 | 7| 797 | 16594 | 
| 41 | 25 src/sqlfluff/rules/L008.py | 74 | 108| 257 | 16851 | 
| 42 | 25 src/sqlfluff/rules/L026.py | 107 | 146| 342 | 17193 | 
| 43 | 26 src/sqlfluff/rules/L028.py | 104 | 122| 168 | 17361 | 
| 44 | 27 src/sqlfluff/rules/L021.py | 7 | 4| 161 | 17522 | 
| 45 | 28 src/sqlfluff/rules/L052.py | 239 | 309| 483 | 18005 | 
| 46 | 28 src/sqlfluff/rules/L016.py | 52 | 88| 317 | 18322 | 
| 47 | 28 src/sqlfluff/rules/L044.py | 80 | 129| 463 | 18785 | 
| 48 | 28 src/sqlfluff/rules/L028.py | 7 | 4| 239 | 19024 | 
| 49 | 29 src/sqlfluff/rules/L041.py | 10 | 7| 774 | 19798 | 
| 50 | 30 src/sqlfluff/rules/L004.py | 46 | 104| 477 | 20275 | 
| 51 | 30 src/sqlfluff/rules/L052.py | 0 | 12| 105 | 20380 | 
| 52 | 30 src/sqlfluff/rules/L056.py | 6 | 3| 262 | 20642 | 
| 53 | 31 src/sqlfluff/rules/L055.py | 6 | 3| 270 | 20912 | 
| 54 | 31 src/sqlfluff/rules/L021.py | 30 | 48| 161 | 21073 | 
| 55 | 31 src/sqlfluff/rules/L026.py | 178 | 204| 272 | 21345 | 
| 56 | 31 src/sqlfluff/rules/L003.py | 0 | 11| 114 | 21459 | 
| 57 | 32 src/sqlfluff/rules/L012.py | 5 | 2| 189 | 21648 | 
| 58 | 33 src/sqlfluff/rules/L014.py | 32 | 80| 274 | 21922 | 
| 59 | 34 src/sqlfluff/rules/L020.py | 90 | 122| 243 | 22165 | 
| 60 | 35 src/sqlfluff/rules/L050.py | 9 | 6| 331 | 22496 | 
| 61 | 36 src/sqlfluff/rules/L001.py | 6 | 3| 546 | 23042 | 
| 62 | 37 src/sqlfluff/rules/L036.py | 329 | 358| 249 | 23291 | 
| 63 | 37 src/sqlfluff/rules/L014.py | 13 | 10| 227 | 23518 | 
| 64 | 37 src/sqlfluff/rules/L027.py | 6 | 3| 189 | 23707 | 
| 65 | 37 src/sqlfluff/rules/L003.py | 626 | 794| 1431 | 25138 | 
| 66 | 37 src/sqlfluff/rules/L050.py | 67 | 103| 365 | 25503 | 
| 67 | 38 src/sqlfluff/rules/L058.py | 8 | 5| 258 | 25761 | 
| 68 | 39 src/sqlfluff/rules/L007.py | 47 | 106| 579 | 26340 | 
| 69 | 39 src/sqlfluff/rules/L025.py | 31 | 59| 120 | 26460 | 
| 70 | 39 src/sqlfluff/rules/L003.py | 846 | 868| 182 | 26642 | 
| 71 | 40 src/sqlfluff/rules/L017.py | 7 | 4| 412 | 27054 | 
| 72 | 41 src/sqlfluff/rules/L051.py | 8 | 5| 314 | 27368 | 
| 73 | 41 src/sqlfluff/rules/L025.py | 83 | 104| 204 | 27572 | 
| 74 | 41 src/sqlfluff/rules/L016.py | 180 | 298| 845 | 28417 | 
| 75 | 41 src/sqlfluff/rules/L044.py | 16 | 13| 475 | 28892 | 
| 76 | 42 src/sqlfluff/rules/L018.py | 8 | 5| 226 | 29118 | 
| 77 | 42 src/sqlfluff/rules/L036.py | 174 | 327| 1249 | 30367 | 
| 78 | 42 src/sqlfluff/rules/L003.py | 53 | 66| 133 | 30500 | 
| 79 | 42 src/sqlfluff/rules/L043.py | 124 | 122| 126 | 30626 | 
| 80 | 42 src/sqlfluff/rules/L058.py | 41 | 87| 482 | 31108 | 
| 81 | 43 src/sqlfluff/rules/L023.py | 10 | 7| 255 | 31363 | 
| 82 | 44 src/sqlfluff/rules/L031.py | 142 | 222| 627 | 31990 | 
| 83 | 45 src/sqlfluff/rules/L024.py | 7 | 4| 207 | 32197 | 
| 84 | 45 src/sqlfluff/rules/L020.py | 11 | 8| 297 | 32494 | 
| 85 | 46 src/sqlfluff/rules/L009.py | 10 | 7| 380 | 32874 | 
| 86 | 46 src/sqlfluff/rules/L003.py | 892 | 909| 164 | 33038 | 
| 87 | 46 src/sqlfluff/rules/L010.py | 193 | 211| 243 | 33281 | 
| 88 | 47 src/sqlfluff/rules/L032.py | 6 | 3| 163 | 33444 | 
| 89 | 48 src/sqlfluff/rules/L022.py | 45 | 211| 1394 | 34838 | 
| 90 | 48 src/sqlfluff/rules/L016.py | 450 | 469| 177 | 35015 | 
| 91 | 48 src/sqlfluff/rules/L020.py | 56 | 88| 233 | 35248 | 
| 92 | 49 src/sqlfluff/rules/L053.py | 44 | 74| 214 | 35462 | 
| 93 | 49 src/sqlfluff/rules/L031.py | 106 | 140| 272 | 35734 | 
| 94 | 49 src/sqlfluff/rules/L036.py | 77 | 109| 302 | 36036 | 
| 95 | 49 src/sqlfluff/rules/L004.py | 9 | 6| 299 | 36335 | 
| 96 | 49 src/sqlfluff/rules/L003.py | 403 | 484| 767 | 37102 | 
| 97 | 49 src/sqlfluff/rules/L053.py | 7 | 4| 204 | 37306 | 
| 98 | 50 src/sqlfluff/rules/L033.py | 33 | 82| 338 | 37644 | 
| 99 | 51 src/sqlfluff/rules/L005.py | 37 | 54| 145 | 37789 | 
| 100 | 52 src/sqlfluff/core/rules/base.py | 191 | 208| 177 | 37966 | 
| 101 | 52 src/sqlfluff/rules/L026.py | 28 | 56| 148 | 38114 | 
| 102 | 53 src/sqlfluff/rules/L013.py | 8 | 5| 174 | 38288 | 
| 103 | 53 src/sqlfluff/rules/L003.py | 870 | 890| 169 | 38457 | 
| 104 | 53 src/sqlfluff/rules/L031.py | 60 | 104| 333 | 38790 | 
| 105 | 53 src/sqlfluff/rules/L036.py | 164 | 172| 102 | 38892 | 
| 106 | 53 src/sqlfluff/rules/L047.py | 11 | 8| 411 | 39303 | 
| 107 | 53 src/sqlfluff/rules/L016.py | 299 | 366| 603 | 39906 | 
| 108 | 54 src/sqlfluff/rules/L015.py | 10 | 7| 202 | 40108 | 
| 109 | 55 src/sqlfluff/rules/L048.py | 10 | 7| 300 | 40408 | 
| 110 | 55 src/sqlfluff/rules/L033.py | 9 | 6| 238 | 40646 | 
| 111 | 56 src/sqlfluff/testing/rules.py | 58 | 88| 327 | 40973 | 
| 112 | 56 src/sqlfluff/rules/L052.py | 123 | 129| 832 | 41805 | 
| 113 | 56 src/sqlfluff/rules/L036.py | 0 | 22| 178 | 41983 | 
| 114 | 56 src/sqlfluff/rules/L003.py | 14 | 51| 216 | 42199 | 
| 115 | 56 src/sqlfluff/core/rules/base.py | 575 | 627| 483 | 42682 | 
| 116 | 56 src/sqlfluff/rules/L031.py | 0 | 17| 132 | 42814 | 
| 117 | 56 src/sqlfluff/testing/rules.py | 110 | 117| 135 | 42949 | 
| 118 | 57 src/sqlfluff/rules/L038.py | 13 | 10| 238 | 43187 | 
| 119 | 57 src/sqlfluff/rules/L052.py | 15 | 48| 148 | 43335 | 
| 120 | 57 src/sqlfluff/rules/L016.py | 19 | 16| 354 | 43689 | 
| 121 | 57 src/sqlfluff/testing/rules.py | 91 | 107| 235 | 43924 | 
| 122 | 57 src/sqlfluff/rules/L026.py | 94 | 105| 125 | 44049 | 
| 123 | 57 src/sqlfluff/rules/L003.py | 68 | 97| 331 | 44380 | 
| 124 | 57 src/sqlfluff/rules/L009.py | 80 | 134| 407 | 44787 | 
| 125 | 58 src/sqlfluff/rules/L037.py | 81 | 120| 298 | 45085 | 
| 126 | 59 src/sqlfluff/rules/L042.py | 8 | 5| 300 | 45385 | 
| 127 | 59 src/sqlfluff/rules/L023.py | 46 | 90| 342 | 45727 | 
| 128 | 59 src/sqlfluff/rules/L034.py | 8 | 5| 189 | 45916 | 
| 129 | 59 src/sqlfluff/rules/L003.py | 99 | 105| 85 | 46001 | 
| 130 | 59 src/sqlfluff/rules/L031.py | 20 | 58| 209 | 46210 | 
| 131 | 59 src/sqlfluff/rules/L044.py | 131 | 146| 144 | 46354 | 
| 132 | 59 src/sqlfluff/rules/L038.py | 43 | 74| 269 | 46623 | 
| 133 | 59 src/sqlfluff/rules/L016.py | 390 | 418| 310 | 46933 | 
| 134 | 59 src/sqlfluff/rules/L019.py | 12 | 9| 520 | 47453 | 
| 135 | 59 src/sqlfluff/rules/L052.py | 99 | 121| 176 | 47629 | 
| 136 | 59 src/sqlfluff/rules/L015.py | 33 | 75| 363 | 47992 | 
| 137 | 59 src/sqlfluff/rules/L019.py | 90 | 130| 391 | 48383 | 
| 138 | 59 src/sqlfluff/rules/L008.py | 12 | 9| 228 | 48611 | 
| 139 | 60 src/sqlfluff/core/linter/linted_file.py | 128 | 143| 158 | 48769 | 
| 140 | 60 src/sqlfluff/core/rules/base.py | 210 | 258| 364 | 49133 | 
| 141 | 61 src/sqlfluff/rules/L002.py | 43 | 76| 232 | 49365 | 
| 142 | 62 src/sqlfluff/rules/L045.py | 7 | 4| 350 | 49715 | 


### Hint

```
@tunetheweb Yeah definitely, should be a pretty quick change 😊
```

## Patch

```diff
diff --git a/src/sqlfluff/rules/L060.py b/src/sqlfluff/rules/L060.py
--- a/src/sqlfluff/rules/L060.py
+++ b/src/sqlfluff/rules/L060.py
@@ -59,4 +59,8 @@ def _eval(self, context: RuleContext) -> Optional[LintResult]:
             ],
         )
 
-        return LintResult(context.segment, [fix])
+        return LintResult(
+            anchor=context.segment,
+            fixes=[fix],
+            description=f"Use 'COALESCE' instead of '{context.segment.raw_upper}'.",
+        )

```

## Test Patch

```diff
diff --git a/test/rules/std_L060_test.py b/test/rules/std_L060_test.py
new file mode 100644
--- /dev/null
+++ b/test/rules/std_L060_test.py
@@ -0,0 +1,12 @@
+"""Tests the python routines within L060."""
+import sqlfluff
+
+
+def test__rules__std_L060_raised() -> None:
+    """L060 is raised for use of ``IFNULL`` or ``NVL``."""
+    sql = "SELECT\n\tIFNULL(NULL, 100),\n\tNVL(NULL,100);"
+    result = sqlfluff.lint(sql, rules=["L060"])
+
+    assert len(result) == 2
+    assert result[0]["description"] == "Use 'COALESCE' instead of 'IFNULL'."
+    assert result[1]["description"] == "Use 'COALESCE' instead of 'NVL'."

```


## Code snippets

### 1 - src/sqlfluff/rules/L060.py:

Start line: 39, End line: 62

```python
@document_fix_compatible
class Rule_L060(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        """Use ``COALESCE`` instead of ``IFNULL`` or ``NVL``."""
        # We only care about function names.
        if context.segment.name != "function_name_identifier":
            return None

        # Only care if the function is ``IFNULL`` or ``NVL``.
        if context.segment.raw_upper not in {"IFNULL", "NVL"}:
            return None

        # Create fix to replace ``IFNULL`` or ``NVL`` with ``COALESCE``.
        fix = LintFix.replace(
            context.segment,
            [
                CodeSegment(
                    raw="COALESCE",
                    name="function_name_identifier",
                    type="function_name_identifier",
                )
            ],
        )

        return LintResult(context.segment, [fix])
```
### 2 - src/sqlfluff/rules/L060.py:

Start line: 9, End line: 6

```python
"""Implementation of Rule L060."""

from typing import Optional

from sqlfluff.core.parser.segments.raw import CodeSegment
from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L060(BaseRule):
    """Use ``COALESCE`` instead of ``IFNULL`` or ``NVL``.

    | **Anti-pattern**
    | ``IFNULL`` or ``NVL`` are used to fill ``NULL`` values.

    .. code-block:: sql

        SELECT ifnull(foo, 0) AS bar,
        FROM baz;

        SELECT nvl(foo, 0) AS bar,
        FROM baz;

    | **Best practice**
    | Use ``COALESCE`` instead.
    | ``COALESCE`` is universally supported,
    | whereas Redshift doesn't support ``IFNULL``
    | and BigQuery doesn't support ``NVL``.
    | Additionally ``COALESCE`` is more flexible
    | and accepts an arbitrary number of arguments.

    .. code-block:: sql

        SELECT coalesce(foo, 0) AS bar,
        FROM baz;

    """
```
### 3 - src/sqlfluff/rules/L043.py:

Start line: 127, End line: 258

```python
@document_fix_compatible
class Rule_L043(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        if (
            context.segment.is_type("case_expression")
            and context.segment.segments[0].name == "case"
        ):
            # Find all 'WHEN' clauses and the optional 'ELSE' clause.
            children = context.functional.segment.children()
            when_clauses = children.select(sp.is_type("when_clause"))
            else_clauses = children.select(sp.is_type("else_clause"))

            # Can't fix if multiple WHEN clauses.
            if len(when_clauses) > 1:
                return None

            # Find condition and then expressions.
            condition_expression = when_clauses.children(sp.is_type("expression"))[0]
            then_expression = when_clauses.children(sp.is_type("expression"))[1]

            # Method 1: Check if THEN/ELSE expressions are both Boolean and can
            # therefore be reduced.
            if else_clauses:
                else_expression = else_clauses.children(sp.is_type("expression"))[0]
                upper_bools = ["TRUE", "FALSE"]
                if (
                    (then_expression.raw_upper in upper_bools)
                    and (else_expression.raw_upper in upper_bools)
                    and (then_expression.raw_upper != else_expression.raw_upper)
                ):
                    coalesce_arg_1 = condition_expression
                    coalesce_arg_2 = KeywordSegment("false")
                    preceding_not = then_expression.raw_upper == "FALSE"

                    fixes = self._coalesce_fix_list(
                        context,
                        coalesce_arg_1,
                        coalesce_arg_2,
                        preceding_not,
                    )

                    return LintResult(
                        anchor=condition_expression,
                        fixes=fixes,
                        description="Unnecessary CASE statement. "
                        "Use COALESCE function instead.",
                    )

            # Method 2: Check if the condition expression is comparing a column
            # reference to NULL and whether that column reference is also in either the
            # THEN/ELSE expression. We can only apply this method when there is only
            # one condition in the condition expression.
            condition_expression_segments_raw = {
                segment.raw_upper for segment in condition_expression.segments
            }
            if {"IS", "NULL"}.issubset(condition_expression_segments_raw) and (
                not condition_expression_segments_raw.intersection({"AND", "OR"})
            ):
                # Check if the comparison is to NULL or NOT NULL.
                is_not_prefix = "NOT" in condition_expression_segments_raw

                # Locate column reference in condition expression.
                column_reference_segment = (
                    Segments(condition_expression)
                    .children(sp.is_type("column_reference"))
                    .get()
                )

                # Return None if none found (this condition does not apply to functions)
                if not column_reference_segment:
                    return None

                if else_clauses:
                    else_expression = else_clauses.children(sp.is_type("expression"))[0]
                    # Check if we can reduce the CASE expression to a single coalesce
                    # function.
                    if (
                        not is_not_prefix
                        and column_reference_segment.raw_upper
                        == else_expression.raw_upper
                    ):
                        coalesce_arg_1 = else_expression
                        coalesce_arg_2 = then_expression
                    elif (
                        is_not_prefix
                        and column_reference_segment.raw_upper
                        == then_expression.raw_upper
                    ):
                        coalesce_arg_1 = then_expression
                        coalesce_arg_2 = else_expression
                    else:
                        return None

                    if coalesce_arg_2.raw_upper == "NULL":
                        # Can just specify the column on it's own
                        # rather than using a COALESCE function.
                        return LintResult(
                            anchor=condition_expression,
                            fixes=self._column_only_fix_list(
                                context,
                                column_reference_segment,
                            ),
                            description="Unnecessary CASE statement. "
                            f"Just use column '{column_reference_segment.raw}'.",
                        )

                    return LintResult(
                        anchor=condition_expression,
                        fixes=self._coalesce_fix_list(
                            context,
                            coalesce_arg_1,
                            coalesce_arg_2,
                        ),
                        description="Unnecessary CASE statement. "
                        "Use COALESCE function instead.",
                    )
                elif (
                    column_reference_segment.raw_segments_upper
                    == then_expression.raw_segments_upper
                ):
                    # Can just specify the column on it's own
                    # rather than using a COALESCE function.
                    # In this case no ELSE statement is equivalent to ELSE NULL.
                    return LintResult(
                        anchor=condition_expression,
                        fixes=self._column_only_fix_list(
                            context,
                            column_reference_segment,
                        ),
                        description="Unnecessary CASE statement. "
                        f"Just use column '{column_reference_segment.raw}'.",
                    )

        return None
```
### 4 - src/sqlfluff/rules/L035.py:

Start line: 37, End line: 73

```python
@document_fix_compatible
class Rule_L035(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        """Find rule violations and provide fixes.

        0. Look for a case expression
        1. Look for "ELSE"
        2. Mark "ELSE" for deletion (populate "fixes")
        3. Backtrack and mark all newlines/whitespaces for deletion
        4. Look for a raw "NULL" segment
        5.a. The raw "NULL" segment is found, we mark it for deletion and return
        5.b. We reach the end of case when without matching "NULL": the rule passes
        """
        if context.segment.is_type("case_expression"):
            children = context.functional.segment.children()
            else_clause = children.first(sp.is_type("else_clause"))

            # Does the "ELSE" have a "NULL"? NOTE: Here, it's safe to look for
            # "NULL", as an expression would *contain* NULL but not be == NULL.
            if else_clause and else_clause.children(
                lambda child: child.raw_upper == "NULL"
            ):
                # Found ELSE with NULL. Delete the whole else clause as well as
                # indents/whitespaces/meta preceding the ELSE. :TRICKY: Note
                # the use of reversed() to make select() effectively search in
                # reverse.
                before_else = children.reversed().select(
                    start_seg=else_clause[0],
                    loop_while=sp.or_(
                        sp.is_name("whitespace", "newline"), sp.is_meta()
                    ),
                )
                return LintResult(
                    anchor=context.segment,
                    fixes=[LintFix.delete(else_clause[0])]
                    + [LintFix.delete(seg) for seg in before_else],
                )
        return None
```
### 5 - src/sqlfluff/rules/L035.py:

Start line: 8, End line: 5

```python
"""Implementation of Rule L035."""
from typing import Optional

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import document_fix_compatible
import sqlfluff.core.rules.functional.segment_predicates as sp


@document_fix_compatible
class Rule_L035(BaseRule):
    """Do not specify ``else null`` in a case when statement (redundant).

    | **Anti-pattern**

    .. code-block:: sql

        select
            case
                when name like '%cat%' then 'meow'
                when name like '%dog%' then 'woof'
                else null
            end
        from x

    | **Best practice**
    |  Omit ``else null``

    .. code-block:: sql

        select
            case
                when name like '%cat%' then 'meow'
                when name like '%dog%' then 'woof'
            end
        from x
    """
```
### 6 - src/sqlfluff/rules/L057.py:

Start line: 50, End line: 112

```python
@document_configuration
class Rule_L057(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        """Do not use special characters in object names."""
        # Config type hints
        self.quoted_identifiers_policy: str
        self.unquoted_identifiers_policy: str
        self.allow_space_in_identifier: bool
        self.additional_allowed_characters: str

        # Exit early if not a single identifier.
        if context.segment.name not in ("naked_identifier", "quoted_identifier"):
            return None

        # Assume unquoted (we'll update if quoted)
        policy = self.unquoted_identifiers_policy

        identifier = context.segment.raw

        # Do some extra processing for quoted identifiers.
        if context.segment.name == "quoted_identifier":

            # Update the default policy to quoted
            policy = self.quoted_identifiers_policy

            # Strip the quotes first
            identifier = context.segment.raw[1:-1]

            # BigQuery table references are quoted in back ticks so allow dots
            #
            # It also allows a star at the end of table_references for wildcards
            # (https://cloud.google.com/bigquery/docs/querying-wildcard-tables)
            #
            # Strip both out before testing the identifier
            if (
                context.dialect.name in ["bigquery"]
                and context.parent_stack
                and context.parent_stack[-1].name == "TableReferenceSegment"
            ):
                if identifier[-1] == "*":
                    identifier = identifier[:-1]
                identifier = identifier.replace(".", "")

            # Strip spaces if allowed (note a separate config as only valid for quoted
            # identifiers)
            if self.allow_space_in_identifier:
                identifier = identifier.replace(" ", "")

        # We always allow underscores so strip them out
        identifier = identifier.replace("_", "")

        # Set the identified minus the allowed characters
        if self.additional_allowed_characters:
            identifier = identifier.translate(
                str.maketrans("", "", self.additional_allowed_characters)
            )

        # Finally test if the remaining identifier is only made up of alphanumerics
        if identifiers_policy_applicable(policy, context.parent_stack) and not (
            identifier.isalnum()
        ):
            return LintResult(anchor=context.segment)

        return None
```
### 7 - src/sqlfluff/rules/L040.py:

Start line: 11, End line: 8

```python
"""Implementation of Rule L040."""

from typing import Tuple, List

from sqlfluff.core.rules.doc_decorators import (
    document_configuration,
    document_fix_compatible,
)
from sqlfluff.rules.L010 import Rule_L010


@document_configuration
@document_fix_compatible
class Rule_L040(Rule_L010):
    """Inconsistent capitalisation of boolean/null literal.

    The functionality for this rule is inherited from :obj:`Rule_L010`.

    | **Anti-pattern**
    | In this example, 'null' and 'false' are in lower-case whereas 'TRUE' is in
    | upper-case.

    .. code-block:: sql

        select
            a,
            null,
            TRUE,
            false
        from foo

    | **Best practice**
    | Ensure all literal null/true/false literals cases are used consistently

    .. code-block:: sql

        select
            a,
            NULL,
            TRUE,
            FALSE
        from foo

        -- Also good

        select
            a,
            null,
            true,
            false
        from foo

    """

    _target_elems: List[Tuple[str, str]] = [
        ("name", "null_literal"),
        ("name", "boolean_literal"),
    ]
    _description_elem = "Boolean/null literals"
```
### 8 - src/sqlfluff/rules/L047.py:

Start line: 60, End line: 136

```python
@document_configuration
@document_fix_compatible
class Rule_L047(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        """Find rule violations and provide fixes."""
        # Config type hints
        self.prefer_count_0: bool
        self.prefer_count_1: bool

        if (
            context.segment.is_type("function")
            and context.segment.get_child("function_name").raw_upper == "COUNT"
        ):
            # Get bracketed content
            f_content = context.functional.segment.children(
                sp.is_type("bracketed")
            ).children(
                sp.and_(
                    sp.not_(sp.is_meta()),
                    sp.not_(
                        sp.is_type(
                            "start_bracket", "end_bracket", "whitespace", "newline"
                        )
                    ),
                )
            )
            if len(f_content) != 1:  # pragma: no cover
                return None

            preferred = "*"
            if self.prefer_count_1:
                preferred = "1"
            elif self.prefer_count_0:
                preferred = "0"

            if f_content[0].is_type("star") and (
                self.prefer_count_1 or self.prefer_count_0
            ):
                return LintResult(
                    anchor=context.segment,
                    fixes=[
                        LintFix.replace(
                            f_content[0],
                            [
                                f_content[0].edit(
                                    f_content[0].raw.replace("*", preferred)
                                )
                            ],
                        ),
                    ],
                )

            if f_content[0].is_type("expression"):
                expression_content = [
                    seg for seg in f_content[0].segments if not seg.is_meta
                ]

                if (
                    len(expression_content) == 1
                    and expression_content[0].is_type("literal")
                    and expression_content[0].raw in ["0", "1"]
                    and expression_content[0].raw != preferred
                ):
                    return LintResult(
                        anchor=context.segment,
                        fixes=[
                            LintFix.replace(
                                expression_content[0],
                                [
                                    expression_content[0].edit(
                                        expression_content[0].raw.replace(
                                            expression_content[0].raw, preferred
                                        )
                                    ),
                                ],
                            ),
                        ],
                    )
        return None
```
### 9 - src/sqlfluff/rules/L025.py:

Start line: 61, End line: 81

```python
@document_fix_compatible
class Rule_L025(BaseRule):

    def _eval(self, context: RuleContext) -> EvalResultType:
        violations: List[LintResult] = []
        if context.segment.is_type("select_statement"):
            # Exit early if the SELECT does not define any aliases.
            select_info = get_select_statement_info(context.segment, context.dialect)
            if not select_info or not select_info.table_aliases:
                return None

            # Analyze the SELECT.
            crawler = SelectCrawler(
                context.segment, context.dialect, query_class=L025Query
            )
            query: L025Query = cast(L025Query, crawler.query_tree)
            self._analyze_table_aliases(query, context.dialect)

            alias: AliasInfo
            for alias in query.aliases:
                if alias.aliased and alias.ref_str not in query.tbl_refs:
                    # Unused alias. Report and fix.
                    violations.append(self._report_unused_alias(alias))
        return violations or None
```
### 10 - src/sqlfluff/rules/L010.py:

Start line: 135, End line: 183

```python
@document_fix_compatible
@document_configuration
class Rule_L010(BaseRule):

    def _eval(self, context: RuleContext) -> LintResult:
        # ... other code
        if concrete_policy in ["upper", "lower", "capitalise"]:
            if concrete_policy == "upper":
                fixed_raw = fixed_raw.upper()
            elif concrete_policy == "lower":
                fixed_raw = fixed_raw.lower()
            elif concrete_policy == "capitalise":
                fixed_raw = fixed_raw.capitalize()
        elif concrete_policy == "pascal":
            # For Pascal we set the first letter in each "word" to uppercase
            # We do not lowercase other letters to allow for PascalCase style
            # words. This does mean we allow all UPPERCASE and also don't
            # correct Pascalcase to PascalCase, but there's only so much we can
            # do. We do correct underscore_words to Underscore_Words.
            fixed_raw = regex.sub(
                "([^a-zA-Z0-9]+|^)([a-zA-Z0-9])([a-zA-Z0-9]*)",
                lambda match: match.group(1) + match.group(2).upper() + match.group(3),
                context.segment.raw,
            )

        if fixed_raw == context.segment.raw:
            # No need to fix
            self.logger.debug(
                f"Capitalisation of segment '{context.segment.raw}' already OK with "
                f"policy '{concrete_policy}', returning with memory {memory}"
            )
            return LintResult(memory=memory)
        else:
            # build description based on the policy in use
            consistency = "consistently " if cap_policy == "consistent" else ""

            if concrete_policy in ["upper", "lower"]:
                policy = f"{concrete_policy} case."
            elif concrete_policy == "capitalise":
                policy = "capitalised."
            elif concrete_policy == "pascal":
                policy = "pascal case."

            # Return the fixed segment
            self.logger.debug(
                f"INCONSISTENT Capitalisation of segment '{context.segment.raw}', "
                f"fixing to '{fixed_raw}' and returning with memory {memory}"
            )

            return LintResult(
                anchor=context.segment,
                fixes=[self._get_fix(context.segment, fixed_raw)],
                memory=memory,
                description=f"{self._description_elem} must be {consistency}{policy}",
            )
```
