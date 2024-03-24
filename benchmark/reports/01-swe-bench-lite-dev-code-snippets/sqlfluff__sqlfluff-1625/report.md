# sqlfluff__sqlfluff-1625

| **sqlfluff/sqlfluff** | `14e1a23a3166b9a645a16de96f694c77a5d4abb7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 948 |
| **Any found context length** | 948 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
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
| src/sqlfluff/rules/L031.py | 214 | 216 | 2 | 1 | 948


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
| 1 | **1 src/sqlfluff/rules/L031.py** | 48 | 102| 348 | 348 | 
| **-> 2 <-** | **1 src/sqlfluff/rules/L031.py** | 148 | 219| 600 | 948 | 
| 3 | 2 src/sqlfluff/rules/L025.py | 38 | 90| 349 | 1297 | 
| 4 | 3 src/sqlfluff/rules/L020.py | 51 | 83| 199 | 1496 | 
| 5 | 4 src/sqlfluff/rules/L027.py | 31 | 62| 199 | 1695 | 
| 6 | 5 src/sqlfluff/rules/L026.py | 57 | 97| 305 | 2000 | 
| 7 | 5 src/sqlfluff/rules/L020.py | 85 | 117| 238 | 2238 | 
| 8 | **5 src/sqlfluff/rules/L031.py** | 112 | 110| 317 | 2555 | 
| 9 | 6 src/sqlfluff/rules/L028.py | 46 | 108| 503 | 3058 | 
| 10 | 7 src/sqlfluff/dialects/dialect_ansi.py | 1323 | 1320| 264 | 3322 | 
| 11 | 8 src/sqlfluff/dialects/dialect_tsql.py | 92 | 152| 494 | 3816 | 
| 12 | 8 src/sqlfluff/dialects/dialect_ansi.py | 1240 | 1301| 438 | 4254 | 
| 13 | 9 src/sqlfluff/dialects/dialect_snowflake.py | 565 | 611| 276 | 4530 | 
| 14 | 10 src/sqlfluff/core/rules/analysis/select.py | 99 | 123| 194 | 4724 | 
| 15 | 10 src/sqlfluff/dialects/dialect_ansi.py | 785 | 813| 172 | 4896 | 
| 16 | 11 src/sqlfluff/dialects/dialect_tsql_keywords.py | 5 | 230| 1167 | 6063 | 
| 17 | 11 src/sqlfluff/dialects/dialect_ansi.py | 1425 | 1579| 1083 | 7146 | 
| 18 | 11 src/sqlfluff/dialects/dialect_tsql.py | 0 | 90| 572 | 7718 | 
| 19 | 11 src/sqlfluff/dialects/dialect_ansi.py | 2694 | 2738| 235 | 7953 | 
| 20 | 12 src/sqlfluff/testing/rules.py | 95 | 111| 235 | 8188 | 
| 21 | 12 src/sqlfluff/dialects/dialect_ansi.py | 1351 | 1378| 226 | 8414 | 
| 22 | 12 src/sqlfluff/testing/rules.py | 58 | 92| 371 | 8785 | 
| 23 | 12 src/sqlfluff/dialects/dialect_ansi.py | 1892 | 1912| 187 | 8972 | 
| 24 | 12 src/sqlfluff/dialects/dialect_tsql_keywords.py | 233 | 280| 258 | 9230 | 
| 25 | 12 src/sqlfluff/core/rules/analysis/select.py | 126 | 138| 135 | 9365 | 
| 26 | 12 src/sqlfluff/dialects/dialect_ansi.py | 1711 | 1726| 110 | 9475 | 
| 27 | 12 src/sqlfluff/core/rules/analysis/select.py | 0 | 17| 133 | 9608 | 
| 28 | 12 src/sqlfluff/dialects/dialect_ansi.py | 695 | 782| 450 | 10058 | 
| 29 | 12 src/sqlfluff/dialects/dialect_ansi.py | 2152 | 2149| 282 | 10340 | 
| 30 | 13 src/sqlfluff/dialects/dialect_exasol.py | 214 | 301| 604 | 10944 | 
| 31 | 13 src/sqlfluff/dialects/dialect_exasol.py | 3022 | 3083| 340 | 11284 | 
| 32 | 13 src/sqlfluff/dialects/dialect_exasol.py | 356 | 374| 135 | 11419 | 
| 33 | 13 src/sqlfluff/dialects/dialect_ansi.py | 1054 | 1066| 419 | 11838 | 
| 34 | 13 src/sqlfluff/dialects/dialect_exasol.py | 90 | 107| 176 | 12014 | 
| 35 | 14 src/sqlfluff/rules/L002.py | 11 | 8| 229 | 12243 | 
| 36 | 14 src/sqlfluff/dialects/dialect_tsql.py | 174 | 190| 123 | 12366 | 
| 37 | 14 src/sqlfluff/dialects/dialect_snowflake.py | 243 | 305| 505 | 12871 | 


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

Start line: 48, End line: 102

```python
@document_fix_compatible
class Rule_L031(BaseRule):

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
```
### 2 - src/sqlfluff/rules/L031.py:

Start line: 148, End line: 219

```python
@document_fix_compatible
class Rule_L031(BaseRule):

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

Start line: 38, End line: 90

```python
@document_fix_compatible
class Rule_L025(Rule_L020):

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

Start line: 51, End line: 83

```python
class Rule_L020(BaseRule):

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
```
### 5 - src/sqlfluff/rules/L027.py:

Start line: 31, End line: 62

```python
class Rule_L027(Rule_L025):

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

Start line: 57, End line: 97

```python
@document_configuration
class Rule_L026(Rule_L020):

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

Start line: 85, End line: 117

```python
class Rule_L020(BaseRule):

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

Start line: 112, End line: 110

```python
@document_fix_compatible
class Rule_L031(BaseRule):

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
```
### 9 - src/sqlfluff/rules/L028.py:

Start line: 46, End line: 108

```python
@document_configuration
class Rule_L028(Rule_L025):

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
### 10 - src/sqlfluff/dialects/dialect_ansi.py:

Start line: 1323, End line: 1320

```python
@ansi_dialect.segment()
class JoinOnConditionSegment(BaseSegment):
    """The `ON` condition within a `JOIN` clause."""

    type = "join_on_condition"
    match_grammar = Sequence(
        "ON",
        Indent,
        OptionallyBracketed(Ref("ExpressionSegment")),
        Dedent,
    )


ansi_dialect.add(
    # This is a hook point to allow subclassing for other dialects
    JoinLikeClauseGrammar=Nothing(),
)


@ansi_dialect.segment()
class FromClauseSegment(BaseSegment):
    """A `FROM` clause like in `SELECT`.

    NOTE: this is a delimited set of table expressions, with a variable
    number of optional join clauses with those table expressions. The
    delmited aspect is the higher of the two such that the following is
    valid (albeit unusual):

    ```
    SELECT *
    FROM a JOIN b, c JOIN d
    ```
    """

    type = "from_clause"
    match_grammar = StartsWith(
        "FROM",
        terminator=Ref("FromClauseTerminatorGrammar"),
        enforce_whitespace_preceding_terminator=True,
    )
    parse_grammar = Sequence(
        "FROM",
        Delimited(
            Ref("FromExpressionSegment"),
        ),
    )
```
