# sqlfluff__sqlfluff-1625

| **sqlfluff/sqlfluff** | `14e1a23a3166b9a645a16de96f694c77a5d4abb7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1498 |
| **Any found context length** | 1498 |
| **Avg pos** | 11.0 |
| **Min pos** | 1 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/src/sqlfluff/rules/L031.py b/src/sqlfluff/rules/L031.py
--- a/src/sqlfluff/rules/L031.py
+++ b/src/sqlfluff/rules/L031.py
@@ -211,7 +211,7 @@ def _lint_aliases_in_join(
             violation_buff.append(
                 LintResult(
                     anchor=alias_info.alias_identifier_ref,
-                    description="Avoid using aliases in join condition",
+                    description="Avoid aliases in from clauses and join conditions.",
                     fixes=fixes,
                 )
             )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/sqlfluff/rules/L031.py | 214 | 216 | 8 | 1 | 7421


## Problem Statement

```
TSQL - L031 incorrectly triggers "Avoid using aliases in join condition" when no join present
## Expected Behaviour

Both of these queries should pass, the only difference is the addition of a table alias 'a':

1/ no alias

\`\`\`
SELECT [hello]
FROM
    mytable
\`\`\`

2/ same query with alias

\`\`\`
SELECT a.[hello]
FROM
    mytable AS a
\`\`\`

## Observed Behaviour

1/ passes
2/ fails with: L031: Avoid using aliases in join condition.

But there is no join condition :-)

## Steps to Reproduce

Lint queries above

## Dialect

TSQL

## Version

sqlfluff 0.6.9
Python 3.6.9

## Configuration

N/A

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| **-> 1 <-** | **1 src/sqlfluff/rules/L031.py** | 0 | 220| 1498 | 1498 | 
| **-> 2 <-** | **1 src/sqlfluff/rules/L031.py** | 0 | 220| 1498 | 2996 | 
| 3 | 2 src/sqlfluff/rules/L025.py | 0 | 91| 520 | 3516 | 
| 4 | 3 src/sqlfluff/rules/L020.py | 0 | 118| 676 | 4192 | 
| 5 | 4 src/sqlfluff/rules/L027.py | 0 | 63| 377 | 4569 | 
| 6 | 5 src/sqlfluff/rules/L026.py | 0 | 98| 678 | 5247 | 
| 7 | 5 src/sqlfluff/rules/L020.py | 0 | 118| 676 | 5923 | 
| **-> 8 <-** | **5 src/sqlfluff/rules/L031.py** | 0 | 220| 1498 | 7421 | 
| 9 | 6 src/sqlfluff/rules/L028.py | 0 | 109| 717 | 8138 | 


### Hint

```
Actually, re-reading the docs I think this is the intended behaviour... closing
```

## Patch

```diff
diff --git a/src/sqlfluff/rules/L031.py b/src/sqlfluff/rules/L031.py
--- a/src/sqlfluff/rules/L031.py
+++ b/src/sqlfluff/rules/L031.py
@@ -211,7 +211,7 @@ def _lint_aliases_in_join(
             violation_buff.append(
                 LintResult(
                     anchor=alias_info.alias_identifier_ref,
-                    description="Avoid using aliases in join condition",
+                    description="Avoid aliases in from clauses and join conditions.",
                     fixes=fixes,
                 )
             )

```

## Test Patch

```diff
diff --git a/test/cli/commands_test.py b/test/cli/commands_test.py
--- a/test/cli/commands_test.py
+++ b/test/cli/commands_test.py
@@ -49,7 +49,7 @@ def invoke_assert_code(
 expected_output = """== [test/fixtures/linter/indentation_error_simple.sql] FAIL
 L:   2 | P:   4 | L003 | Indentation not hanging or a multiple of 4 spaces
 L:   5 | P:  10 | L010 | Keywords must be consistently upper case.
-L:   5 | P:  13 | L031 | Avoid using aliases in join condition
+L:   5 | P:  13 | L031 | Avoid aliases in from clauses and join conditions.
 """
 
 

```


## Code snippets

### 1 - src/sqlfluff/rules/L031.py:

```python
"""Implementation of Rule L031."""

from collections import Counter, defaultdict
from typing import Generator, NamedTuple

from sqlfluff.core.parser import BaseSegment
from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L031(BaseRule):
    """Avoid table aliases in from clauses and join conditions.

    | **Anti-pattern**
    | In this example, alias 'o' is used for the orders table, and 'c' is used for 'customers' table.

    .. code-block:: sql

        SELECT
            COUNT(o.customer_id) as order_amount,
            c.name
        FROM orders as o
        JOIN customers as c on o.id = c.user_id


    | **Best practice**
    |  Avoid aliases.

    .. code-block:: sql

        SELECT
            COUNT(orders.customer_id) as order_amount,
            customers.name
        FROM orders
        JOIN customers on orders.id = customers.user_id

        -- Self-join will not raise issue

        SELECT
            table.a,
            table_alias.b,
        FROM
            table
            LEFT JOIN table AS table_alias ON table.foreign_key = table_alias.foreign_key

    """

    def _eval(self, segment, **kwargs):
        """Identify aliases in from clause and join conditions.

        Find base table, table expressions in join, and other expressions in select clause
        and decide if it's needed to report them.
        """
        if segment.is_type("select_statement"):
            # A buffer for all table expressions in join conditions
            from_expression_elements = []
            column_reference_segments = []

            from_clause_segment = segment.get_child("from_clause")

            if not from_clause_segment:
                return None

            from_expression = from_clause_segment.get_child("from_expression")
            from_expression_element = None
            if from_expression:
                from_expression_element = from_expression.get_child(
                    "from_expression_element"
                )

            if not from_expression_element:
                return None
            from_expression_element = from_expression_element.get_child(
                "table_expression"
            )

            # Find base table
            base_table = None
            if from_expression_element:
                base_table = from_expression_element.get_child("object_reference")

            from_clause_index = segment.segments.index(from_clause_segment)
            from_clause_and_after = segment.segments[from_clause_index:]

            for clause in from_clause_and_after:
                for from_expression_element in clause.recursive_crawl(
                    "from_expression_element"
                ):
                    from_expression_elements.append(from_expression_element)
                for column_reference in clause.recursive_crawl("column_reference"):
                    column_reference_segments.append(column_reference)

            return (
                self._lint_aliases_in_join(
                    base_table,
                    from_expression_elements,
                    column_reference_segments,
                    segment,
                )
                or None
            )
        return None

    class TableAliasInfo(NamedTuple):
        """Structure yielded by_filter_table_expressions()."""

        table_ref: BaseSegment
        whitespace_ref: BaseSegment
        alias_exp_ref: BaseSegment
        alias_identifier_ref: BaseSegment

    @classmethod
    def _filter_table_expressions(
        cls, base_table, from_expression_elements
    ) -> Generator[TableAliasInfo, None, None]:
        for from_expression in from_expression_elements:
            table_expression = from_expression.get_child("table_expression")
            if not table_expression:
                continue
            table_ref = table_expression.get_child("object_reference")

            # If the from_expression_element has no object_references - skip it
            # An example case is a lateral flatten, where we have a function segment
            # instead of a table_reference segment.
            if not table_ref:
                continue

            # If this is self-join - skip it
            if (
                base_table
                and base_table.raw == table_ref.raw
                and base_table != table_ref
            ):
                continue

            whitespace_ref = from_expression.get_child("whitespace")

            # If there's no alias expression - skip it
            alias_exp_ref = from_expression.get_child("alias_expression")
            if alias_exp_ref is None:
                continue

            alias_identifier_ref = alias_exp_ref.get_child("identifier")
            yield cls.TableAliasInfo(
                table_ref, whitespace_ref, alias_exp_ref, alias_identifier_ref
            )

    def _lint_aliases_in_join(
        self, base_table, from_expression_elements, column_reference_segments, segment
    ):
        """Lint and fix all aliases in joins - except for self-joins."""
        # A buffer to keep any violations.
        violation_buff = []

        to_check = list(
            self._filter_table_expressions(base_table, from_expression_elements)
        )

        # How many times does each table appear in the FROM clause?
        table_counts = Counter(ai.table_ref.raw for ai in to_check)

        # What is the set of aliases used for each table? (We are mainly
        # interested in the NUMBER of different aliases used.)
        table_aliases = defaultdict(set)
        for ai in to_check:
            table_aliases[ai.table_ref.raw].add(ai.alias_identifier_ref.raw)

        # For each aliased table, check whether to keep or remove it.
        for alias_info in to_check:
            # If the same table appears more than once in the FROM clause with
            # different alias names, do not consider removing its aliases.
            # The aliases may have been introduced simply to make each
            # occurrence of the table independent within the query.
            if (
                table_counts[alias_info.table_ref.raw] > 1
                and len(table_aliases[alias_info.table_ref.raw]) > 1
            ):
                continue

            select_clause = segment.get_child("select_clause")

            ids_refs = []

            # Find all references to alias in select clause
            alias_name = alias_info.alias_identifier_ref.raw
            for alias_with_column in select_clause.recursive_crawl("object_reference"):
                used_alias_ref = alias_with_column.get_child("identifier")
                if used_alias_ref and used_alias_ref.raw == alias_name:
                    ids_refs.append(used_alias_ref)

            # Find all references to alias in column references
            for exp_ref in column_reference_segments:
                used_alias_ref = exp_ref.get_child("identifier")
                # exp_ref.get_child('dot') ensures that the column reference includes a table reference
                if used_alias_ref.raw == alias_name and exp_ref.get_child("dot"):
                    ids_refs.append(used_alias_ref)

            # Fixes for deleting ` as sth` and for editing references to aliased tables
            fixes = [
                *[
                    LintFix("delete", d)
                    for d in [alias_info.alias_exp_ref, alias_info.whitespace_ref]
                ],
                *[
                    LintFix("edit", alias, alias.edit(alias_info.table_ref.raw))
                    for alias in [alias_info.alias_identifier_ref, *ids_refs]
                ],
            ]

            violation_buff.append(
                LintResult(
                    anchor=alias_info.alias_identifier_ref,
                    description="Avoid using aliases in join condition",
                    fixes=fixes,
                )
            )

        return violation_buff or None

```
### 2 - src/sqlfluff/rules/L031.py:

```python
"""Implementation of Rule L031."""

from collections import Counter, defaultdict
from typing import Generator, NamedTuple

from sqlfluff.core.parser import BaseSegment
from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L031(BaseRule):
    """Avoid table aliases in from clauses and join conditions.

    | **Anti-pattern**
    | In this example, alias 'o' is used for the orders table, and 'c' is used for 'customers' table.

    .. code-block:: sql

        SELECT
            COUNT(o.customer_id) as order_amount,
            c.name
        FROM orders as o
        JOIN customers as c on o.id = c.user_id


    | **Best practice**
    |  Avoid aliases.

    .. code-block:: sql

        SELECT
            COUNT(orders.customer_id) as order_amount,
            customers.name
        FROM orders
        JOIN customers on orders.id = customers.user_id

        -- Self-join will not raise issue

        SELECT
            table.a,
            table_alias.b,
        FROM
            table
            LEFT JOIN table AS table_alias ON table.foreign_key = table_alias.foreign_key

    """

    def _eval(self, segment, **kwargs):
        """Identify aliases in from clause and join conditions.

        Find base table, table expressions in join, and other expressions in select clause
        and decide if it's needed to report them.
        """
        if segment.is_type("select_statement"):
            # A buffer for all table expressions in join conditions
            from_expression_elements = []
            column_reference_segments = []

            from_clause_segment = segment.get_child("from_clause")

            if not from_clause_segment:
                return None

            from_expression = from_clause_segment.get_child("from_expression")
            from_expression_element = None
            if from_expression:
                from_expression_element = from_expression.get_child(
                    "from_expression_element"
                )

            if not from_expression_element:
                return None
            from_expression_element = from_expression_element.get_child(
                "table_expression"
            )

            # Find base table
            base_table = None
            if from_expression_element:
                base_table = from_expression_element.get_child("object_reference")

            from_clause_index = segment.segments.index(from_clause_segment)
            from_clause_and_after = segment.segments[from_clause_index:]

            for clause in from_clause_and_after:
                for from_expression_element in clause.recursive_crawl(
                    "from_expression_element"
                ):
                    from_expression_elements.append(from_expression_element)
                for column_reference in clause.recursive_crawl("column_reference"):
                    column_reference_segments.append(column_reference)

            return (
                self._lint_aliases_in_join(
                    base_table,
                    from_expression_elements,
                    column_reference_segments,
                    segment,
                )
                or None
            )
        return None

    class TableAliasInfo(NamedTuple):
        """Structure yielded by_filter_table_expressions()."""

        table_ref: BaseSegment
        whitespace_ref: BaseSegment
        alias_exp_ref: BaseSegment
        alias_identifier_ref: BaseSegment

    @classmethod
    def _filter_table_expressions(
        cls, base_table, from_expression_elements
    ) -> Generator[TableAliasInfo, None, None]:
        for from_expression in from_expression_elements:
            table_expression = from_expression.get_child("table_expression")
            if not table_expression:
                continue
            table_ref = table_expression.get_child("object_reference")

            # If the from_expression_element has no object_references - skip it
            # An example case is a lateral flatten, where we have a function segment
            # instead of a table_reference segment.
            if not table_ref:
                continue

            # If this is self-join - skip it
            if (
                base_table
                and base_table.raw == table_ref.raw
                and base_table != table_ref
            ):
                continue

            whitespace_ref = from_expression.get_child("whitespace")

            # If there's no alias expression - skip it
            alias_exp_ref = from_expression.get_child("alias_expression")
            if alias_exp_ref is None:
                continue

            alias_identifier_ref = alias_exp_ref.get_child("identifier")
            yield cls.TableAliasInfo(
                table_ref, whitespace_ref, alias_exp_ref, alias_identifier_ref
            )

    def _lint_aliases_in_join(
        self, base_table, from_expression_elements, column_reference_segments, segment
    ):
        """Lint and fix all aliases in joins - except for self-joins."""
        # A buffer to keep any violations.
        violation_buff = []

        to_check = list(
            self._filter_table_expressions(base_table, from_expression_elements)
        )

        # How many times does each table appear in the FROM clause?
        table_counts = Counter(ai.table_ref.raw for ai in to_check)

        # What is the set of aliases used for each table? (We are mainly
        # interested in the NUMBER of different aliases used.)
        table_aliases = defaultdict(set)
        for ai in to_check:
            table_aliases[ai.table_ref.raw].add(ai.alias_identifier_ref.raw)

        # For each aliased table, check whether to keep or remove it.
        for alias_info in to_check:
            # If the same table appears more than once in the FROM clause with
            # different alias names, do not consider removing its aliases.
            # The aliases may have been introduced simply to make each
            # occurrence of the table independent within the query.
            if (
                table_counts[alias_info.table_ref.raw] > 1
                and len(table_aliases[alias_info.table_ref.raw]) > 1
            ):
                continue

            select_clause = segment.get_child("select_clause")

            ids_refs = []

            # Find all references to alias in select clause
            alias_name = alias_info.alias_identifier_ref.raw
            for alias_with_column in select_clause.recursive_crawl("object_reference"):
                used_alias_ref = alias_with_column.get_child("identifier")
                if used_alias_ref and used_alias_ref.raw == alias_name:
                    ids_refs.append(used_alias_ref)

            # Find all references to alias in column references
            for exp_ref in column_reference_segments:
                used_alias_ref = exp_ref.get_child("identifier")
                # exp_ref.get_child('dot') ensures that the column reference includes a table reference
                if used_alias_ref.raw == alias_name and exp_ref.get_child("dot"):
                    ids_refs.append(used_alias_ref)

            # Fixes for deleting ` as sth` and for editing references to aliased tables
            fixes = [
                *[
                    LintFix("delete", d)
                    for d in [alias_info.alias_exp_ref, alias_info.whitespace_ref]
                ],
                *[
                    LintFix("edit", alias, alias.edit(alias_info.table_ref.raw))
                    for alias in [alias_info.alias_identifier_ref, *ids_refs]
                ],
            ]

            violation_buff.append(
                LintResult(
                    anchor=alias_info.alias_identifier_ref,
                    description="Avoid using aliases in join condition",
                    fixes=fixes,
                )
            )

        return violation_buff or None

```
### 3 - src/sqlfluff/rules/L025.py:

```python
"""Implementation of Rule L025."""

from sqlfluff.core.rules.base import LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import document_fix_compatible
from sqlfluff.rules.L020 import Rule_L020
from sqlfluff.core.dialects.common import AliasInfo


@document_fix_compatible
class Rule_L025(Rule_L020):
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

    def _lint_references_and_aliases(
        self,
        table_aliases,
        standalone_aliases,
        references,
        col_aliases,
        using_cols,
        parent_select,
    ):
        """Check all aliased references against tables referenced in the query."""
        # A buffer to keep any violations.
        violation_buff = []
        # Check all the references that we have, keep track of which aliases we refer to.
        tbl_refs = set()
        for r in references:
            tbl_refs.update(
                tr.part
                for tr in r.extract_possible_references(
                    level=r.ObjectReferenceLevel.TABLE
                )
            )

        alias: AliasInfo
        for alias in table_aliases:
            if alias.aliased and alias.ref_str not in tbl_refs:
                fixes = [LintFix("delete", alias.alias_expression)]
                found_alias_segment = False
                # Walk back to remove indents/whitespaces
                for segment in reversed(alias.from_expression_element.segments):
                    if not found_alias_segment:
                        if segment is alias.alias_expression:
                            found_alias_segment = True
                    else:
                        if (
                            segment.name == "whitespace"
                            or segment.name == "newline"
                            or segment.is_meta
                        ):
                            fixes.append(LintFix("delete", segment))
                        else:
                            # Stop once we reach an other, "regular" segment.
                            break
                violation_buff.append(
                    LintResult(
                        anchor=alias.segment,
                        description="Alias {!r} is never used in SELECT statement.".format(
                            alias.ref_str
                        ),
                        fixes=fixes,
                    )
                )
        return violation_buff or None

```
### 4 - src/sqlfluff/rules/L020.py:

```python
"""Implementation of Rule L020."""

import itertools

from sqlfluff.core.rules.base import BaseRule, LintResult
from sqlfluff.core.rules.analysis.select import get_select_statement_info


class Rule_L020(BaseRule):
    """Table aliases should be unique within each clause.

    | **Anti-pattern**
    | In this example, the alias 't' is reused for two different ables:

    .. code-block:: sql

        SELECT
            t.a,
            t.b
        FROM foo AS t, bar AS t

        -- this can also happen when using schemas where the implicit alias is the table name:

        SELECT
            a,
            b
        FROM
            2020.foo,
            2021.foo

    | **Best practice**
    | Make all tables have a unique alias

    .. code-block:: sql

        SELECT
            f.a,
            b.b
        FROM foo AS f, bar AS b

        -- Also use explicit alias's when referencing two tables with same name from two different schemas

        SELECT
            f1.a,
            f2.b
        FROM
            2020.foo AS f1,
            2021.foo AS f2

    """

    def _lint_references_and_aliases(
        self,
        table_aliases,
        standalone_aliases,
        references,
        col_aliases,
        using_cols,
        parent_select,
    ):
        """Check whether any aliases are duplicates.

        NB: Subclasses of this error should override this function.

        """
        # Are any of the aliases the same?
        duplicate = set()
        for a1, a2 in itertools.combinations(table_aliases, 2):
            # Compare the strings
            if a1.ref_str == a2.ref_str and a1.ref_str:
                duplicate.add(a2)
        if duplicate:
            return [
                LintResult(
                    # Reference the element, not the string.
                    anchor=aliases.segment,
                    description=(
                        "Duplicate table alias {!r}. Table " "aliases should be unique."
                    ).format(aliases.ref_str),
                )
                for aliases in duplicate
            ]
        else:
            return None

    def _eval(self, segment, parent_stack, dialect, **kwargs):
        """Get References and Aliases and allow linting.

        This rule covers a lot of potential cases of odd usages of
        references, see the code for each of the potential cases.

        Subclasses of this rule should override the
        `_lint_references_and_aliases` method.
        """
        if segment.is_type("select_statement"):
            select_info = get_select_statement_info(segment, dialect)
            if not select_info:
                return None

            # Work out if we have a parent select function
            parent_select = None
            for seg in reversed(parent_stack):
                if seg.is_type("select_statement"):
                    parent_select = seg
                    break

            # Pass them all to the function that does all the work.
            # NB: Subclasses of this rules should override the function below
            return self._lint_references_and_aliases(
                select_info.table_aliases,
                select_info.standalone_aliases,
                select_info.reference_buffer,
                select_info.col_aliases,
                select_info.using_cols,
                parent_select,
            )
        return None

```
### 5 - src/sqlfluff/rules/L027.py:

```python
"""Implementation of Rule L027."""

from sqlfluff.core.rules.base import LintResult
from sqlfluff.rules.L025 import Rule_L025


class Rule_L027(Rule_L025):
    """References should be qualified if select has more than one referenced table/view.

    NB: Except if they're present in a USING clause.

    | **Anti-pattern**
    | In this example, the reference 'vee' has not been declared
    | and the variables 'a' and 'b' are potentially ambiguous.

    .. code-block:: sql

        SELECT a, b
        FROM foo
        LEFT JOIN vee ON vee.a = foo.a

    | **Best practice**
    |  Add the references.

    .. code-block:: sql

        SELECT foo.a, vee.b
        FROM foo
        LEFT JOIN vee ON vee.a = foo.a
    """

    def _lint_references_and_aliases(
        self,
        table_aliases,
        standalone_aliases,
        references,
        col_aliases,
        using_cols,
        parent_select,
    ):
        # Do we have more than one? If so, all references should be qualified.
        if len(table_aliases) <= 1:
            return None
        # A buffer to keep any violations.
        violation_buff = []
        # Check all the references that we have.
        for r in references:
            this_ref_type = r.qualification()
            if (
                this_ref_type == "unqualified"
                and r.raw not in col_aliases
                and r.raw not in using_cols
            ):
                violation_buff.append(
                    LintResult(
                        anchor=r,
                        description=f"Unqualified reference {r.raw!r} found in "
                        "select with more than one referenced table/view.",
                    )
                )

        return violation_buff or None

```
### 6 - src/sqlfluff/rules/L026.py:

```python
"""Implementation of Rule L026."""

from sqlfluff.core.rules.analysis.select import get_aliases_from_select
from sqlfluff.core.rules.base import LintResult
from sqlfluff.core.rules.doc_decorators import document_configuration
from sqlfluff.rules.L025 import Rule_L020


@document_configuration
class Rule_L026(Rule_L020):
    """References cannot reference objects not present in FROM clause.

    NB: This rule is disabled by default for BigQuery due to its use of
    structs which trigger false positives. It can be enabled with the
    `force_enable = True` flag.

    | **Anti-pattern**
    | In this example, the reference 'vee' has not been declared.

    .. code-block:: sql

        SELECT
            vee.a
        FROM foo

    | **Best practice**
    |  Remove the reference.

    .. code-block:: sql

        SELECT
            a
        FROM foo

    """

    config_keywords = ["force_enable"]

    @staticmethod
    def _is_bad_tbl_ref(table_aliases, parent_select, tbl_ref):
        """Given a table reference, try to find what it's referring to."""
        # Is it referring to one of the table aliases?
        if tbl_ref[0] in [a.ref_str for a in table_aliases]:
            # Yes. Therefore okay.
            return False

        # Not a table alias. It it referring to a correlated subquery?
        if parent_select:
            parent_aliases, _ = get_aliases_from_select(parent_select)
            if parent_aliases and tbl_ref[0] in [a[0] for a in parent_aliases]:
                # Yes. Therefore okay.
                return False

        # It's not referring to an alias or a correlated subquery. Looks like a
        # bad reference (i.e. referring to something unknown.)
        return True

    def _lint_references_and_aliases(
        self,
        table_aliases,
        standalone_aliases,
        references,
        col_aliases,
        using_cols,
        parent_select,
    ):
        # A buffer to keep any violations.
        violation_buff = []

        # Check all the references that we have, do they reference present aliases?
        for r in references:
            tbl_refs = r.extract_possible_references(level=r.ObjectReferenceLevel.TABLE)
            if tbl_refs and all(
                self._is_bad_tbl_ref(table_aliases, parent_select, tbl_ref)
                for tbl_ref in tbl_refs
            ):
                violation_buff.append(
                    LintResult(
                        # Return the first segment rather than the string
                        anchor=tbl_refs[0].segments[0],
                        description=f"Reference {r.raw!r} refers to table/view "
                        "not found in the FROM clause or found in parent "
                        "subquery.",
                    )
                )
        return violation_buff or None

    def _eval(self, segment, parent_stack, dialect, **kwargs):
        """Override Rule L020 for dialects that use structs.

        Some dialects use structs (e.g. column.field) which look like
        table references and so incorrectly trigger this rule.
        """
        if dialect.name in ["bigquery"] and not self.force_enable:
            return LintResult()

        return super()._eval(segment, parent_stack, dialect, **kwargs)

```
### 7 - src/sqlfluff/rules/L020.py:

```python
"""Implementation of Rule L020."""

import itertools

from sqlfluff.core.rules.base import BaseRule, LintResult
from sqlfluff.core.rules.analysis.select import get_select_statement_info


class Rule_L020(BaseRule):
    """Table aliases should be unique within each clause.

    | **Anti-pattern**
    | In this example, the alias 't' is reused for two different ables:

    .. code-block:: sql

        SELECT
            t.a,
            t.b
        FROM foo AS t, bar AS t

        -- this can also happen when using schemas where the implicit alias is the table name:

        SELECT
            a,
            b
        FROM
            2020.foo,
            2021.foo

    | **Best practice**
    | Make all tables have a unique alias

    .. code-block:: sql

        SELECT
            f.a,
            b.b
        FROM foo AS f, bar AS b

        -- Also use explicit alias's when referencing two tables with same name from two different schemas

        SELECT
            f1.a,
            f2.b
        FROM
            2020.foo AS f1,
            2021.foo AS f2

    """

    def _lint_references_and_aliases(
        self,
        table_aliases,
        standalone_aliases,
        references,
        col_aliases,
        using_cols,
        parent_select,
    ):
        """Check whether any aliases are duplicates.

        NB: Subclasses of this error should override this function.

        """
        # Are any of the aliases the same?
        duplicate = set()
        for a1, a2 in itertools.combinations(table_aliases, 2):
            # Compare the strings
            if a1.ref_str == a2.ref_str and a1.ref_str:
                duplicate.add(a2)
        if duplicate:
            return [
                LintResult(
                    # Reference the element, not the string.
                    anchor=aliases.segment,
                    description=(
                        "Duplicate table alias {!r}. Table " "aliases should be unique."
                    ).format(aliases.ref_str),
                )
                for aliases in duplicate
            ]
        else:
            return None

    def _eval(self, segment, parent_stack, dialect, **kwargs):
        """Get References and Aliases and allow linting.

        This rule covers a lot of potential cases of odd usages of
        references, see the code for each of the potential cases.

        Subclasses of this rule should override the
        `_lint_references_and_aliases` method.
        """
        if segment.is_type("select_statement"):
            select_info = get_select_statement_info(segment, dialect)
            if not select_info:
                return None

            # Work out if we have a parent select function
            parent_select = None
            for seg in reversed(parent_stack):
                if seg.is_type("select_statement"):
                    parent_select = seg
                    break

            # Pass them all to the function that does all the work.
            # NB: Subclasses of this rules should override the function below
            return self._lint_references_and_aliases(
                select_info.table_aliases,
                select_info.standalone_aliases,
                select_info.reference_buffer,
                select_info.col_aliases,
                select_info.using_cols,
                parent_select,
            )
        return None

```
### 8 - src/sqlfluff/rules/L031.py:

```python
"""Implementation of Rule L031."""

from collections import Counter, defaultdict
from typing import Generator, NamedTuple

from sqlfluff.core.parser import BaseSegment
from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L031(BaseRule):
    """Avoid table aliases in from clauses and join conditions.

    | **Anti-pattern**
    | In this example, alias 'o' is used for the orders table, and 'c' is used for 'customers' table.

    .. code-block:: sql

        SELECT
            COUNT(o.customer_id) as order_amount,
            c.name
        FROM orders as o
        JOIN customers as c on o.id = c.user_id


    | **Best practice**
    |  Avoid aliases.

    .. code-block:: sql

        SELECT
            COUNT(orders.customer_id) as order_amount,
            customers.name
        FROM orders
        JOIN customers on orders.id = customers.user_id

        -- Self-join will not raise issue

        SELECT
            table.a,
            table_alias.b,
        FROM
            table
            LEFT JOIN table AS table_alias ON table.foreign_key = table_alias.foreign_key

    """

    def _eval(self, segment, **kwargs):
        """Identify aliases in from clause and join conditions.

        Find base table, table expressions in join, and other expressions in select clause
        and decide if it's needed to report them.
        """
        if segment.is_type("select_statement"):
            # A buffer for all table expressions in join conditions
            from_expression_elements = []
            column_reference_segments = []

            from_clause_segment = segment.get_child("from_clause")

            if not from_clause_segment:
                return None

            from_expression = from_clause_segment.get_child("from_expression")
            from_expression_element = None
            if from_expression:
                from_expression_element = from_expression.get_child(
                    "from_expression_element"
                )

            if not from_expression_element:
                return None
            from_expression_element = from_expression_element.get_child(
                "table_expression"
            )

            # Find base table
            base_table = None
            if from_expression_element:
                base_table = from_expression_element.get_child("object_reference")

            from_clause_index = segment.segments.index(from_clause_segment)
            from_clause_and_after = segment.segments[from_clause_index:]

            for clause in from_clause_and_after:
                for from_expression_element in clause.recursive_crawl(
                    "from_expression_element"
                ):
                    from_expression_elements.append(from_expression_element)
                for column_reference in clause.recursive_crawl("column_reference"):
                    column_reference_segments.append(column_reference)

            return (
                self._lint_aliases_in_join(
                    base_table,
                    from_expression_elements,
                    column_reference_segments,
                    segment,
                )
                or None
            )
        return None

    class TableAliasInfo(NamedTuple):
        """Structure yielded by_filter_table_expressions()."""

        table_ref: BaseSegment
        whitespace_ref: BaseSegment
        alias_exp_ref: BaseSegment
        alias_identifier_ref: BaseSegment

    @classmethod
    def _filter_table_expressions(
        cls, base_table, from_expression_elements
    ) -> Generator[TableAliasInfo, None, None]:
        for from_expression in from_expression_elements:
            table_expression = from_expression.get_child("table_expression")
            if not table_expression:
                continue
            table_ref = table_expression.get_child("object_reference")

            # If the from_expression_element has no object_references - skip it
            # An example case is a lateral flatten, where we have a function segment
            # instead of a table_reference segment.
            if not table_ref:
                continue

            # If this is self-join - skip it
            if (
                base_table
                and base_table.raw == table_ref.raw
                and base_table != table_ref
            ):
                continue

            whitespace_ref = from_expression.get_child("whitespace")

            # If there's no alias expression - skip it
            alias_exp_ref = from_expression.get_child("alias_expression")
            if alias_exp_ref is None:
                continue

            alias_identifier_ref = alias_exp_ref.get_child("identifier")
            yield cls.TableAliasInfo(
                table_ref, whitespace_ref, alias_exp_ref, alias_identifier_ref
            )

    def _lint_aliases_in_join(
        self, base_table, from_expression_elements, column_reference_segments, segment
    ):
        """Lint and fix all aliases in joins - except for self-joins."""
        # A buffer to keep any violations.
        violation_buff = []

        to_check = list(
            self._filter_table_expressions(base_table, from_expression_elements)
        )

        # How many times does each table appear in the FROM clause?
        table_counts = Counter(ai.table_ref.raw for ai in to_check)

        # What is the set of aliases used for each table? (We are mainly
        # interested in the NUMBER of different aliases used.)
        table_aliases = defaultdict(set)
        for ai in to_check:
            table_aliases[ai.table_ref.raw].add(ai.alias_identifier_ref.raw)

        # For each aliased table, check whether to keep or remove it.
        for alias_info in to_check:
            # If the same table appears more than once in the FROM clause with
            # different alias names, do not consider removing its aliases.
            # The aliases may have been introduced simply to make each
            # occurrence of the table independent within the query.
            if (
                table_counts[alias_info.table_ref.raw] > 1
                and len(table_aliases[alias_info.table_ref.raw]) > 1
            ):
                continue

            select_clause = segment.get_child("select_clause")

            ids_refs = []

            # Find all references to alias in select clause
            alias_name = alias_info.alias_identifier_ref.raw
            for alias_with_column in select_clause.recursive_crawl("object_reference"):
                used_alias_ref = alias_with_column.get_child("identifier")
                if used_alias_ref and used_alias_ref.raw == alias_name:
                    ids_refs.append(used_alias_ref)

            # Find all references to alias in column references
            for exp_ref in column_reference_segments:
                used_alias_ref = exp_ref.get_child("identifier")
                # exp_ref.get_child('dot') ensures that the column reference includes a table reference
                if used_alias_ref.raw == alias_name and exp_ref.get_child("dot"):
                    ids_refs.append(used_alias_ref)

            # Fixes for deleting ` as sth` and for editing references to aliased tables
            fixes = [
                *[
                    LintFix("delete", d)
                    for d in [alias_info.alias_exp_ref, alias_info.whitespace_ref]
                ],
                *[
                    LintFix("edit", alias, alias.edit(alias_info.table_ref.raw))
                    for alias in [alias_info.alias_identifier_ref, *ids_refs]
                ],
            ]

            violation_buff.append(
                LintResult(
                    anchor=alias_info.alias_identifier_ref,
                    description="Avoid using aliases in join condition",
                    fixes=fixes,
                )
            )

        return violation_buff or None

```
### 9 - src/sqlfluff/rules/L028.py:

```python
"""Implementation of Rule L028."""

from sqlfluff.core.rules.base import LintResult
from sqlfluff.core.rules.doc_decorators import document_configuration
from sqlfluff.rules.L025 import Rule_L025


@document_configuration
class Rule_L028(Rule_L025):
    """References should be consistent in statements with a single table.

    NB: This rule is disabled by default for BigQuery due to its use of
    structs which trigger false positives. It can be enabled with the
    `force_enable = True` flag.

    | **Anti-pattern**
    | In this example, only the field `b` is referenced.

    .. code-block:: sql

        SELECT
            a,
            foo.b
        FROM foo

    | **Best practice**
    |  Remove all the reference or reference all the fields.

    .. code-block:: sql

        SELECT
            a,
            b
        FROM foo

        -- Also good

        SELECT
            foo.a,
            foo.b
        FROM foo

    """

    config_keywords = ["single_table_references", "force_enable"]

    def _lint_references_and_aliases(
        self,
        table_aliases,
        standalone_aliases,
        references,
        col_aliases,
        using_cols,
        parent_select,
    ):
        """Iterate through references and check consistency."""
        # How many aliases are there? If more than one then abort.
        if len(table_aliases) > 1:
            return None
        # A buffer to keep any violations.
        violation_buff = []
        # Check all the references that we have.
        seen_ref_types = set()
        for ref in references:
            # We skip any unqualified wildcard references (i.e. *). They shouldn't count.
            if not ref.is_qualified() and ref.is_type("wildcard_identifier"):
                continue
            # Oddball case: Column aliases provided via function calls in by
            # FROM or JOIN. References to these don't need to be qualified.
            # Note there could be a table with a column by the same name as
            # this alias, so avoid bogus warnings by just skipping them
            # entirely rather than trying to enforce anything.
            if ref.raw in standalone_aliases:
                continue
            this_ref_type = ref.qualification()
            if self.single_table_references == "consistent":
                if seen_ref_types and this_ref_type not in seen_ref_types:
                    violation_buff.append(
                        LintResult(
                            anchor=ref,
                            description=f"{this_ref_type.capitalize()} reference "
                            f"{ref.raw!r} found in single table select which is "
                            "inconsistent with previous references.",
                        )
                    )
            elif self.single_table_references != this_ref_type:
                violation_buff.append(
                    LintResult(
                        anchor=ref,
                        description="{} reference {!r} found in single table select.".format(
                            this_ref_type.capitalize(), ref.raw
                        ),
                    )
                )
            seen_ref_types.add(this_ref_type)

        return violation_buff or None

    def _eval(self, segment, parent_stack, dialect, **kwargs):
        """Override Rule L025 for dialects that use structs.

        Some dialects use structs (e.g. column.field) which look like
        table references and so incorrectly trigger this rule.
        """
        if dialect.name in ["bigquery"] and not self.force_enable:
            return LintResult()

        return super()._eval(segment, parent_stack, dialect, **kwargs)

```
