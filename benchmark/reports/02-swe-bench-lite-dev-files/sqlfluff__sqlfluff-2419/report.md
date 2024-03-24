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
| 1 | **1 src/sqlfluff/rules/L060.py** | 0 | 63| 436 | 436 | 
| 2 | **1 src/sqlfluff/rules/L060.py** | 0 | 63| 436 | 872 | 
| 3 | 2 src/sqlfluff/rules/L043.py | 0 | 259| 1673 | 2545 | 
| 4 | 3 src/sqlfluff/rules/L035.py | 0 | 74| 587 | 3132 | 
| 5 | 3 src/sqlfluff/rules/L035.py | 0 | 74| 587 | 3719 | 
| 6 | 4 src/sqlfluff/rules/L057.py | 0 | 113| 706 | 4425 | 
| 7 | 5 src/sqlfluff/rules/L040.py | 0 | 60| 270 | 4695 | 
| 8 | 6 src/sqlfluff/rules/L047.py | 0 | 137| 874 | 5569 | 
| 9 | 7 src/sqlfluff/rules/L025.py | 0 | 138| 982 | 6551 | 
| 10 | 8 src/sqlfluff/rules/L010.py | 0 | 212| 1647 | 8198 | 
| 11 | 8 src/sqlfluff/rules/L043.py | 0 | 259| 1673 | 9871 | 
| 12 | 9 src/sqlfluff/rules/L003.py | 0 | 910| 7233 | 17104 | 
| 13 | 10 src/sqlfluff/rules/L016.py | 0 | 554| 4156 | 21260 | 
| 14 | 11 src/sqlfluff/rules/L027.py | 0 | 74| 463 | 21723 | 
| 15 | 12 src/sqlfluff/rules/L059.py | 0 | 80| 426 | 22149 | 
| 16 | 13 src/sqlfluff/rules/L029.py | 0 | 66| 387 | 22536 | 
| 17 | 14 src/sqlfluff/rules/L054.py | 0 | 156| 921 | 23457 | 
| 18 | 15 src/sqlfluff/rules/L026.py | 0 | 205| 1626 | 25083 | 
| 19 | 15 src/sqlfluff/rules/L003.py | 0 | 910| 7233 | 32316 | 
| 20 | 15 src/sqlfluff/rules/L003.py | 0 | 910| 7233 | 39549 | 
| 21 | 16 src/sqlfluff/rules/L011.py | 0 | 108| 710 | 40259 | 
| 22 | 17 src/sqlfluff/rules/L006.py | 0 | 207| 1306 | 41565 | 
| 23 | 18 src/sqlfluff/rules/L056.py | 0 | 80| 487 | 42052 | 
| 24 | 18 src/sqlfluff/rules/L057.py | 0 | 113| 706 | 42758 | 
| 25 | 19 src/sqlfluff/rules/L034.py | 0 | 183| 1371 | 44129 | 
| 26 | 20 src/sqlfluff/rules/L019.py | 0 | 231| 1734 | 45863 | 
| 27 | 21 src/sqlfluff/rules/L030.py | 0 | 49| 256 | 46119 | 
| 28 | 22 src/sqlfluff/rules/L049.py | 0 | 118| 835 | 46954 | 
| 29 | 22 src/sqlfluff/rules/L049.py | 0 | 118| 835 | 47789 | 
| 30 | 23 src/sqlfluff/rules/L044.py | 0 | 147| 1197 | 48986 | 


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
### 3 - src/sqlfluff/rules/L043.py:

```python
"""Implementation of Rule L043."""
from typing import List, Optional

from sqlfluff.core.parser import (
    WhitespaceSegment,
    SymbolSegment,
    KeywordSegment,
)
from sqlfluff.core.parser.segments.base import BaseSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import document_fix_compatible
from sqlfluff.core.rules.functional import Segments, sp


@document_fix_compatible
class Rule_L043(BaseRule):
    """Unnecessary ``CASE`` statement.

    | **Anti-pattern**
    | ``CASE`` statement returns booleans.

    .. code-block:: sql
        :force:

        select
            case
                when fab > 0 then true
                else false
            end as is_fab
        from fancy_table

        -- This rule can also simplify CASE statements
        -- that aim to fill NULL values.

        select
            case
                when fab is null then 0
                else fab
            end as fab_clean
        from fancy_table

        -- This also covers where the case statement
        -- replaces NULL values with NULL values.

        select
            case
                when fab is null then null
                else fab
            end as fab_clean
        from fancy_table

    | **Best practice**
    | Reduce to ``WHEN`` condition within ``COALESCE`` function.

    .. code-block:: sql
        :force:

        select
            coalesce(fab > 0, false) as is_fab
        from fancy_table

        -- To fill NULL values.

        select
            coalesce(fab, 0) as fab_clean
        from fancy_table

        -- NULL filling NULL.

        select fab as fab_clean
        from fancy_table


    """

    @staticmethod
    def _coalesce_fix_list(
        context: RuleContext,
        coalesce_arg_1: BaseSegment,
        coalesce_arg_2: BaseSegment,
        preceding_not: bool = False,
    ) -> List[LintFix]:
        """Generate list of fixes to convert CASE statement to COALESCE function."""
        # Add coalesce and opening parenthesis.
        edits = [
            KeywordSegment("coalesce"),
            SymbolSegment("(", name="start_bracket", type="start_bracket"),
            coalesce_arg_1,
            SymbolSegment(",", name="comma", type="comma"),
            WhitespaceSegment(),
            coalesce_arg_2,
            SymbolSegment(")", name="end_bracket", type="end_bracket"),
        ]

        if preceding_not:
            not_edits: List[BaseSegment] = [
                KeywordSegment("not"),
                WhitespaceSegment(),
            ]
            edits = not_edits + edits

        fixes = [
            LintFix.replace(
                context.segment,
                edits,
            )
        ]
        return fixes

    @staticmethod
    def _column_only_fix_list(
        context: RuleContext,
        column_reference_segment: BaseSegment,
    ) -> List[LintFix]:
        """Generate list of fixes to reduce CASE statement to a single column."""
        fixes = [
            LintFix.replace(
                context.segment,
                [column_reference_segment],
            )
        ]
        return fixes

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        """Unnecessary CASE statement."""
        # Look for CASE expression.
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
### 6 - src/sqlfluff/rules/L057.py:

```python
"""Implementation of Rule L057."""

from typing import Optional

from sqlfluff.core.rules.base import BaseRule, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import (
    document_configuration,
)
from sqlfluff.rules.L014 import identifiers_policy_applicable


@document_configuration
class Rule_L057(BaseRule):
    """Do not use special characters in identifiers.

    | **Anti-pattern**
    | Using special characters within identifiers when creating or aliasing objects.

    .. code-block:: sql

        CREATE TABLE DBO.ColumnNames
        (
            [Internal Space] INT,
            [Greater>Than] INT,
            [Less<Than] INT,
            Number# INT
        )

    | **Best practice**
    | Identifiers should include only alphanumerics and underscores.

    .. code-block:: sql

        CREATE TABLE DBO.ColumnNames
        (
            [Internal_Space] INT,
            [GreaterThan] INT,
            [LessThan] INT,
            NumberVal INT
        )

    """

    config_keywords = [
        "quoted_identifiers_policy",
        "unquoted_identifiers_policy",
        "allow_space_in_identifier",
        "additional_allowed_characters",
    ]

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

```python
"""Implementation of Rule L047."""
from typing import Optional

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import (
    document_configuration,
    document_fix_compatible,
)
import sqlfluff.core.rules.functional.segment_predicates as sp


@document_configuration
@document_fix_compatible
class Rule_L047(BaseRule):
    """Use consistent syntax to express "count number of rows".

    Note:
        If both `prefer_count_1` and `prefer_count_0` are set to true
        then `prefer_count_1` has precedence.

    ``COUNT(*)``, ``COUNT(1)``, and even ``COUNT(0)`` are equivalent syntaxes
    in many SQL engines due to optimizers interpreting these instructions as
    "count number of rows in result".

    The ANSI-92_ spec mentions the ``COUNT(*)`` syntax specifically as
    having a special meaning:

        If COUNT(*) is specified, then
        the result is the cardinality of T.

    So by default, SQLFluff enforces the consistent use of ``COUNT(*)``.

    If the SQL engine you work with, or your team, prefers ``COUNT(1)`` or
    ``COUNT(0)`` over ``COUNT(*)``, you can configure this rule to consistently
    enforce your preference.

    .. _ANSI-92: http://msdn.microsoft.com/en-us/library/ms175997.aspx

    | **Anti-pattern**

    .. code-block:: sql

        select
            count(1)
        from table_a

    | **Best practice**
    | Use ``count(*)`` unless specified otherwise by config ``prefer_count_1``,
    | or ``prefer_count_0`` as preferred.

    .. code-block:: sql

        select
            count(*)
        from table_a

    """

    config_keywords = ["prefer_count_1", "prefer_count_0"]

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

```python
"""Implementation of Rule L025."""

from dataclasses import dataclass, field
from typing import cast, List, Set

from sqlfluff.core.dialects.base import Dialect
from sqlfluff.core.rules.analysis.select import get_select_statement_info
from sqlfluff.core.rules.analysis.select_crawler import (
    Query as SelectCrawlerQuery,
    SelectCrawler,
)
from sqlfluff.core.rules.base import (
    BaseRule,
    LintFix,
    LintResult,
    RuleContext,
    EvalResultType,
)
from sqlfluff.core.rules.doc_decorators import document_fix_compatible
from sqlfluff.core.rules.functional import Segments, sp
from sqlfluff.core.dialects.common import AliasInfo


@dataclass
class L025Query(SelectCrawlerQuery):
    """SelectCrawler Query with custom L025 info."""

    aliases: List[AliasInfo] = field(default_factory=list)
    tbl_refs: Set[str] = field(default_factory=set)


@document_fix_compatible
class Rule_L025(BaseRule):
    """Tables should not be aliased if that alias is not used.

    | **Anti-pattern**

    .. code-block:: sql

        SELECT
            a
        FROM foo AS zoo

    | **Best practice**
    | Use the alias or remove it. An unused alias makes code
    | harder to read without changing any functionality.

    .. code-block:: sql

        SELECT
            zoo.a
        FROM foo AS zoo

        -- Alternatively...

        SELECT
            a
        FROM foo

    """

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

    @classmethod
    def _analyze_table_aliases(cls, query: L025Query, dialect: Dialect):
        # Get table aliases defined in query.
        for selectable in query.selectables:
            select_info = selectable.select_info
            if select_info:
                # Record the aliases.
                query.aliases += select_info.table_aliases

                # Look at each table reference; if it's an alias reference,
                # resolve the alias: could be an alias defined in "query"
                # itself or an "ancestor" query.
                for r in select_info.reference_buffer:
                    for tr in r.extract_possible_references(
                        level=r.ObjectReferenceLevel.TABLE
                    ):
                        # This function walks up the query's parent stack if necessary.
                        cls._resolve_and_mark_reference(query, tr.part)

        # Visit children.
        for child in query.children:
            cls._analyze_table_aliases(cast(L025Query, child), dialect)

    @classmethod
    def _resolve_and_mark_reference(cls, query: L025Query, ref: str):
        # Does this query define the referenced alias?
        if any(ref == a.ref_str for a in query.aliases):
            # Yes. Record the reference.
            query.tbl_refs.add(ref)
        elif query.parent:
            # No. Recursively check the query's parent hierarchy.
            cls._resolve_and_mark_reference(cast(L025Query, query.parent), ref)

    @classmethod
    def _report_unused_alias(cls, alias: AliasInfo) -> LintResult:
        fixes = [LintFix.delete(alias.alias_expression)]  # type: ignore
        # Walk back to remove indents/whitespaces
        to_delete = (
            Segments(*alias.from_expression_element.segments)
            .reversed()
            .select(
                start_seg=alias.alias_expression,
                # Stop once we reach an other, "regular" segment.
                loop_while=sp.or_(sp.is_whitespace(), sp.is_meta()),
            )
        )
        fixes += [LintFix.delete(seg) for seg in to_delete]
        return LintResult(
            anchor=alias.segment,
            description="Alias {!r} is never used in SELECT statement.".format(
                alias.ref_str
            ),
            fixes=fixes,
        )

```
### 10 - src/sqlfluff/rules/L010.py:

```python
"""Implementation of Rule L010."""

import regex
from typing import Tuple, List
from sqlfluff.core.rules.base import BaseRule, LintResult, LintFix, RuleContext
from sqlfluff.core.rules.config_info import get_config_info
from sqlfluff.core.rules.doc_decorators import (
    document_fix_compatible,
    document_configuration,
)


@document_fix_compatible
@document_configuration
class Rule_L010(BaseRule):
    """Inconsistent capitalisation of keywords.

    | **Anti-pattern**
    | In this example, 'select 'is in lower-case whereas 'FROM' is in upper-case.

    .. code-block:: sql

        select
            a
        FROM foo

    | **Best practice**
    | Make all keywords either in upper-case or in lower-case

    .. code-block:: sql

        SELECT
            a
        FROM foo

        -- Also good

        select
            a
        from foo
    """

    # Binary operators behave like keywords too.
    _target_elems: List[Tuple[str, str]] = [
        ("type", "keyword"),
        ("type", "binary_operator"),
        ("type", "date_part"),
        ("type", "data_type_identifier"),
    ]
    config_keywords = ["capitalisation_policy"]
    # Human readable target elem for description
    _description_elem = "Keywords"

    def _eval(self, context: RuleContext) -> LintResult:
        """Inconsistent capitalisation of keywords.

        We use the `memory` feature here to keep track of cases known to be
        INconsistent with what we've seen so far as well as the top choice
        for what the possible case is.

        """
        # Skip if not an element of the specified type/name
        if not self.matches_target_tuples(context.segment, self._target_elems):
            return LintResult(memory=context.memory)

        # Get the capitalisation policy configuration.
        try:
            cap_policy = self.cap_policy
            cap_policy_opts = self.cap_policy_opts
        except AttributeError:
            # First-time only, read the settings from configuration. This is
            # very slow.
            cap_policy, cap_policy_opts = self._init_capitalisation_policy()

        memory = context.memory
        refuted_cases = memory.get("refuted_cases", set())

        # Which cases are definitely inconsistent with the segment?
        if context.segment.raw[0] != context.segment.raw[0].upper():
            refuted_cases.update(["upper", "capitalise", "pascal"])
            if context.segment.raw != context.segment.raw.lower():
                refuted_cases.update(["lower"])
        else:
            refuted_cases.update(["lower"])
            if context.segment.raw != context.segment.raw.upper():
                refuted_cases.update(["upper"])
            if context.segment.raw != context.segment.raw.capitalize():
                refuted_cases.update(["capitalise"])
            if not context.segment.raw.isalnum():
                refuted_cases.update(["pascal"])

        # Update the memory
        memory["refuted_cases"] = refuted_cases

        self.logger.debug(
            f"Refuted cases after segment '{context.segment.raw}': {refuted_cases}"
        )

        # Skip if no inconsistencies, otherwise compute a concrete policy
        # to convert to.
        if cap_policy == "consistent":
            possible_cases = [c for c in cap_policy_opts if c not in refuted_cases]
            self.logger.debug(
                f"Possible cases after segment '{context.segment.raw}': "
                "{possible_cases}"
            )
            if possible_cases:
                # Save the latest possible case and skip
                memory["latest_possible_case"] = possible_cases[0]
                self.logger.debug(
                    f"Consistent capitalization, returning with memory: {memory}"
                )
                return LintResult(memory=memory)
            else:
                concrete_policy = memory.get("latest_possible_case", "upper")
                self.logger.debug(
                    f"Getting concrete policy '{concrete_policy}' from memory"
                )
        else:
            if cap_policy not in refuted_cases:
                # Skip
                self.logger.debug(
                    f"Consistent capitalization {cap_policy}, returning with "
                    f"memory: {memory}"
                )
                return LintResult(memory=memory)
            else:
                concrete_policy = cap_policy
                self.logger.debug(
                    f"Setting concrete policy '{concrete_policy}' from cap_policy"
                )

        # Set the fixed to same as initial in case any of below don't match
        fixed_raw = context.segment.raw
        # We need to change the segment to match the concrete policy
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

    def _get_fix(self, segment, fixed_raw):
        """Given a segment found to have a fix, returns a LintFix for it.

        May be overridden by subclasses, which is useful when the parse tree
        structure varies from this simple base case.
        """
        return LintFix.replace(segment, [segment.edit(fixed_raw)])

    def _init_capitalisation_policy(self):
        """Called first time rule is evaluated to fetch & cache the policy."""
        cap_policy_name = next(
            k for k in self.config_keywords if k.endswith("capitalisation_policy")
        )
        self.cap_policy = getattr(self, cap_policy_name)
        self.cap_policy_opts = [
            opt
            for opt in get_config_info()[cap_policy_name]["validation"]
            if opt != "consistent"
        ]
        self.logger.debug(
            f"Selected '{cap_policy_name}': '{self.cap_policy}' from options "
            f"{self.cap_policy_opts}"
        )
        cap_policy = self.cap_policy
        cap_policy_opts = self.cap_policy_opts
        return cap_policy, cap_policy_opts

```
