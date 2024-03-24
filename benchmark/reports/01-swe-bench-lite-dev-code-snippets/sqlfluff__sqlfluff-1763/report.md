# sqlfluff__sqlfluff-1763

| **sqlfluff/sqlfluff** | `a10057635e5b2559293a676486f0b730981f037a` |
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
diff --git a/src/sqlfluff/core/linter/linted_file.py b/src/sqlfluff/core/linter/linted_file.py
--- a/src/sqlfluff/core/linter/linted_file.py
+++ b/src/sqlfluff/core/linter/linted_file.py
@@ -7,6 +7,8 @@
 
 import os
 import logging
+import shutil
+import tempfile
 from typing import (
     Any,
     Iterable,
@@ -493,7 +495,24 @@ def persist_tree(self, suffix: str = "") -> bool:
             if suffix:
                 root, ext = os.path.splitext(fname)
                 fname = root + suffix + ext
-            # Actually write the file.
-            with open(fname, "w", encoding=self.encoding) as f:
-                f.write(write_buff)
+            self._safe_create_replace_file(fname, write_buff, self.encoding)
         return success
+
+    @staticmethod
+    def _safe_create_replace_file(fname, write_buff, encoding):
+        # Write to a temporary file first, so in case of encoding or other
+        # issues, we don't delete or corrupt the user's existing file.
+        dirname, basename = os.path.split(fname)
+        with tempfile.NamedTemporaryFile(
+            mode="w",
+            encoding=encoding,
+            prefix=basename,
+            dir=dirname,
+            suffix=os.path.splitext(fname)[1],
+            delete=False,
+        ) as tmp:
+            tmp.file.write(write_buff)
+            tmp.flush()
+            os.fsync(tmp.fileno())
+        # Once the temp file is safely written, replace the existing file.
+        shutil.move(tmp.name, fname)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/sqlfluff/core/linter/linted_file.py | 7 | - | - | - | -
| src/sqlfluff/core/linter/linted_file.py | 496 | 498 | - | - | -


## Problem Statement

```
dbt postgres fix command errors with UnicodeEncodeError and also wipes the .sql file
_If this is a parsing or linting issue, please include a minimal SQL example which reproduces the issue, along with the `sqlfluff parse` output, `sqlfluff lint` output and `sqlfluff fix` output when relevant._

## Expected Behaviour
Violation failure notice at a minimum, without wiping the file. Would like a way to ignore the known error at a minimum as --noqa is not getting past this. Actually would expect --noqa to totally ignore this.

## Observed Behaviour
Reported error: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 120: character maps to <undefined>`

## Steps to Reproduce
SQL file:
\`\`\`sql
SELECT
    reacted_table_name_right.descendant_id AS category_id,
    string_agg(redacted_table_name_left.name, ' → ' ORDER BY reacted_table_name_right.generations DESC) AS breadcrumbs -- noqa
FROM {{ ref2('redacted_schema_name', 'redacted_table_name_left') }} AS redacted_table_name_left
INNER JOIN {{ ref2('redacted_schema_name', 'reacted_table_name_right') }} AS reacted_table_name_right
    ON redacted_table_name_left.id = order_issue_category_hierarchies.ancestor_id
GROUP BY reacted_table_name_right.descendant_id
\`\`\`
Running `sqlfluff fix --ignore templating,parsing,lexing -vvvv` and accepting proposed fixes for linting violations.

## Dialect
`postgres`, with `dbt` templater

## Version
`python 3.7.12`
`sqlfluff 0.7.0`
`sqlfluff-templater-dbt 0.7.0`

## Configuration
I've tried a few, here's one:
\`\`\`
[sqlfluff]
verbose = 2
dialect = postgres
templater = dbt
exclude_rules = None
output_line_length = 80
runaway_limit = 10
ignore_templated_areas = True
processes = 3
# Comma separated list of file extensions to lint.

# NB: This config will only apply in the root folder.
sql_file_exts = .sql

[sqlfluff:indentation]
indented_joins = False
indented_using_on = True
template_blocks_indent = True

[sqlfluff:templater]
unwrap_wrapped_queries = True

[sqlfluff:templater:jinja]
apply_dbt_builtins = True

[sqlfluff:templater:jinja:macros]
# Macros provided as builtins for dbt projects
dbt_ref = {% macro ref(model_ref) %}{{model_ref}}{% endmacro %}
dbt_source = {% macro source(source_name, table) %}{{source_name}}_{{table}}{% endmacro %}
dbt_config = {% macro config() %}{% for k in kwargs %}{% endfor %}{% endmacro %}
dbt_var = {% macro var(variable, default='') %}item{% endmacro %}
dbt_is_incremental = {% macro is_incremental() %}True{% endmacro %}

# Common config across rules
[sqlfluff:rules]
tab_space_size = 4
indent_unit = space
single_table_references = consistent
unquoted_identifiers_policy = all

# L001 - Remove trailing whitespace (fix)
# L002 - Single section of whitespace should not contain both tabs and spaces (fix)
# L003 - Keep consistent indentation (fix)
# L004 - We use 4 spaces for indentation just for completeness (fix)
# L005 - Remove space before commas (fix)
# L006 - Operators (+, -, *, /) will be wrapped by a single space each side (fix)

# L007 - Operators should not be at the end of a line
[sqlfluff:rules:L007]  # Keywords
operator_new_lines = after

# L008 - Always use a single whitespace after a comma (fix)
# L009 - Files will always end with a trailing newline

# L010 - All keywords will use full upper case (fix)
[sqlfluff:rules:L010]  # Keywords
capitalisation_policy = upper

# L011 - Always explicitly alias tables (fix)
[sqlfluff:rules:L011]  # Aliasing
aliasing = explicit

# L012 - Do not have to explicitly alias all columns
[sqlfluff:rules:L012]  # Aliasing
aliasing = explicit

# L013 - Always explicitly alias a column with an expression in it (fix)
[sqlfluff:rules:L013]  # Aliasing
allow_scalar = False

# L014 - Always user full lower case for 'quoted identifiers' -> column refs. without an alias (fix)
[sqlfluff:rules:L014]  # Unquoted identifiers
extended_capitalisation_policy = lower

# L015 - Always remove parenthesis when using DISTINCT to be clear that DISTINCT applies to all columns (fix)

# L016 - Lines should be 120 characters of less. Comment lines should not be ignored (fix)
[sqlfluff:rules:L016]
ignore_comment_lines = False
max_line_length = 120

# L017 - There should not be whitespace between function name and brackets (fix)
# L018 - Always align closing bracket of WITH to the WITH keyword (fix)

# L019 - Always use trailing commas / commas at the end of the line (fix)
[sqlfluff:rules:L019]
comma_style = trailing

# L020 - Table aliases will always be unique per statement
# L021 - Remove any use of ambiguous DISTINCT and GROUP BY combinations. Lean on removing the GROUP BY.
# L022 - Add blank lines after common table expressions (CTE) / WITH.
# L023 - Always add a single whitespace after AS in a WITH clause (fix)

[sqlfluff:rules:L026]
force_enable = False

# L027 - Always add references if more than one referenced table or view is used

[sqlfluff:rules:L028]
force_enable = False

[sqlfluff:rules:L029]  # Keyword identifiers
unquoted_identifiers_policy = aliases

[sqlfluff:rules:L030]  # Function names
capitalisation_policy = upper

# L032 - We prefer use of join keys rather than USING
# L034 - We prefer ordering of columns in select statements as (fix):
# 1. wildcards
# 2. single identifiers
# 3. calculations and aggregates

# L035 - Omit 'else NULL'; it is redundant (fix)
# L036 - Move select targets / identifiers onto new lines each (fix)
# L037 - When using ORDER BY, make the direction explicit (fix)

# L038 - Never use trailing commas at the end of the SELECT clause
[sqlfluff:rules:L038]
select_clause_trailing_comma = forbid

# L039 - Remove unnecessary whitespace (fix)

[sqlfluff:rules:L040]  # Null & Boolean Literals
capitalisation_policy = upper

# L042 - Join clauses should not contain subqueries. Use common tables expressions (CTE) instead.
[sqlfluff:rules:L042]
# By default, allow subqueries in from clauses, but not join clauses.
forbid_subquery_in = join

# L043 - Reduce CASE WHEN conditions to COALESCE (fix)
# L044 - Prefer a known number of columns along the path to the source data
# L045 - Remove unused common tables expressions (CTE) / WITH statements (fix)
# L046 - Jinja tags should have a single whitespace on both sides

# L047 - Use COUNT(*) instead of COUNT(0) or COUNT(1) alternatives (fix)
[sqlfluff:rules:L047]  # Consistent syntax to count all rows
prefer_count_1 = False
prefer_count_0 = False

# L048 - Quoted literals should be surrounded by a single whitespace (fix)
# L049 - Always use IS or IS NOT for comparisons with NULL (fix)
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 setup.py | 31 | 130| 810 | 810 | 
| 2 | 2 src/sqlfluff/dialects/dialect_tsql.py | 1827 | 1883| 288 | 1098 | 
| 3 | 3 src/sqlfluff/rules/L039.py | 9 | 6| 409 | 1507 | 
| 4 | 3 src/sqlfluff/dialects/dialect_tsql.py | 1683 | 1729| 274 | 1781 | 
| 5 | 3 src/sqlfluff/dialects/dialect_tsql.py | 1033 | 1121| 470 | 2251 | 
| 6 | 3 src/sqlfluff/dialects/dialect_tsql.py | 2026 | 2093| 390 | 2641 | 
| 7 | 3 src/sqlfluff/dialects/dialect_tsql.py | 1360 | 1441| 379 | 3020 | 
| 8 | 3 src/sqlfluff/dialects/dialect_tsql.py | 1906 | 2004| 577 | 3597 | 
| 9 | 3 src/sqlfluff/dialects/dialect_tsql.py | 839 | 854| 117 | 3714 | 
| 10 | 3 src/sqlfluff/dialects/dialect_tsql.py | 2007 | 2023| 116 | 3830 | 
| 11 | 3 src/sqlfluff/dialects/dialect_tsql.py | 977 | 974| 233 | 4063 | 
| 12 | 3 src/sqlfluff/dialects/dialect_tsql.py | 1338 | 1335| 215 | 4278 | 
| 13 | 3 src/sqlfluff/dialects/dialect_tsql.py | 799 | 836| 283 | 4561 | 
| 14 | 4 plugins/sqlfluff-templater-dbt/setup.py | 0 | 68| 625 | 5186 | 
| 15 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1145 | 1181| 240 | 5426 | 
| 16 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1013 | 1010| 217 | 5643 | 
| 17 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1773 | 1804| 195 | 5838 | 
| 18 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1732 | 1770| 219 | 6057 | 
| 19 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1444 | 1465| 196 | 6253 | 
| 20 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1807 | 1824| 149 | 6402 | 
| 21 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1184 | 1279| 505 | 6907 | 
| 22 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1886 | 1903| 108 | 7015 | 
| 23 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1561 | 1603| 247 | 7262 | 
| 24 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1648 | 1680| 204 | 7466 | 
| 25 | 4 src/sqlfluff/dialects/dialect_tsql.py | 857 | 892| 283 | 7749 | 
| 26 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1629 | 1645| 136 | 7885 | 
| 27 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1282 | 1321| 275 | 8160 | 
| 28 | 4 src/sqlfluff/dialects/dialect_tsql.py | 895 | 960| 424 | 8584 | 
| 29 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1489 | 1531| 205 | 8789 | 
| 30 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1606 | 1626| 159 | 8948 | 
| 31 | 4 src/sqlfluff/dialects/dialect_tsql.py | 2096 | 2117| 135 | 9083 | 
| 32 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1468 | 1486| 186 | 9269 | 
| 33 | 4 src/sqlfluff/dialects/dialect_tsql.py | 2120 | 2158| 241 | 9510 | 
| 34 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1534 | 1558| 224 | 9734 | 
| 35 | 4 src/sqlfluff/dialects/dialect_tsql.py | 1124 | 1142| 186 | 9920 | 


## Missing Patch Files

 * 1: src/sqlfluff/core/linter/linted_file.py

### Hint

```
I get a dbt-related error -- can you provide your project file as well? Also, what operating system are you running this on? I tested a simplified (non-dbt) version of your file on my Mac, and it worked okay.

\`\`\`
dbt.exceptions.DbtProjectError: Runtime Error
  no dbt_project.yml found at expected path /Users/bhart/dev/sqlfluff/dbt_project.yml
\`\`\`
Never mind the questions above -- I managed to reproduce the error in a sample dbt project. Taking a look now...
@Tumble17: Have you tried setting the `encoding` parameter in `.sqlfluff`? Do you know what encoding you're using? The default is `autodetect`, and SQLFluff "thinks" the file uses "Windows-1252" encoding, which I assume is incorrect -- that's why SQLFluff is unable to write out the updated file.
I added this line to the first section of your `.sqlfluff`, and now it seems to work. I'll look into changing the behavior of `sqlfluff fix` so it doesn't erase the file when it fails.

\`\`\`
encoding = utf-8
\`\`\`
```

## Patch

```diff
diff --git a/src/sqlfluff/core/linter/linted_file.py b/src/sqlfluff/core/linter/linted_file.py
--- a/src/sqlfluff/core/linter/linted_file.py
+++ b/src/sqlfluff/core/linter/linted_file.py
@@ -7,6 +7,8 @@
 
 import os
 import logging
+import shutil
+import tempfile
 from typing import (
     Any,
     Iterable,
@@ -493,7 +495,24 @@ def persist_tree(self, suffix: str = "") -> bool:
             if suffix:
                 root, ext = os.path.splitext(fname)
                 fname = root + suffix + ext
-            # Actually write the file.
-            with open(fname, "w", encoding=self.encoding) as f:
-                f.write(write_buff)
+            self._safe_create_replace_file(fname, write_buff, self.encoding)
         return success
+
+    @staticmethod
+    def _safe_create_replace_file(fname, write_buff, encoding):
+        # Write to a temporary file first, so in case of encoding or other
+        # issues, we don't delete or corrupt the user's existing file.
+        dirname, basename = os.path.split(fname)
+        with tempfile.NamedTemporaryFile(
+            mode="w",
+            encoding=encoding,
+            prefix=basename,
+            dir=dirname,
+            suffix=os.path.splitext(fname)[1],
+            delete=False,
+        ) as tmp:
+            tmp.file.write(write_buff)
+            tmp.flush()
+            os.fsync(tmp.fileno())
+        # Once the temp file is safely written, replace the existing file.
+        shutil.move(tmp.name, fname)

```

## Test Patch

```diff
diff --git a/test/core/linter_test.py b/test/core/linter_test.py
--- a/test/core/linter_test.py
+++ b/test/core/linter_test.py
@@ -641,3 +641,56 @@ def test__attempt_to_change_templater_warning(caplog):
         assert "Attempt to set templater to " in caplog.text
     finally:
         logger.propagate = original_propagate_value
+
+
+@pytest.mark.parametrize(
+    "case",
+    [
+        dict(
+            name="utf8_create",
+            fname="test.sql",
+            encoding="utf-8",
+            existing=None,
+            update="def",
+            expected="def",
+        ),
+        dict(
+            name="utf8_update",
+            fname="test.sql",
+            encoding="utf-8",
+            existing="abc",
+            update="def",
+            expected="def",
+        ),
+        dict(
+            name="utf8_special_char",
+            fname="test.sql",
+            encoding="utf-8",
+            existing="abc",
+            update="→",  # Special utf-8 character
+            expected="→",
+        ),
+        dict(
+            name="incorrect_encoding",
+            fname="test.sql",
+            encoding="Windows-1252",
+            existing="abc",
+            update="→",  # Not valid in Windows-1252
+            expected="abc",  # File should be unchanged
+        ),
+    ],
+    ids=lambda case: case["name"],
+)
+def test_safe_create_replace_file(case, tmp_path):
+    """Test creating or updating .sql files, various content and encoding."""
+    p = tmp_path / case["fname"]
+    if case["existing"]:
+        p.write_text(case["existing"])
+    try:
+        linter.LintedFile._safe_create_replace_file(
+            str(p), case["update"], case["encoding"]
+        )
+    except:  # noqa: E722
+        pass
+    actual = p.read_text(encoding=case["encoding"])
+    assert case["expected"] == actual

```


## Code snippets

### 1 - setup.py:

Start line: 31, End line: 130

```python
setup(
    name="sqlfluff",
    version=version,
    license="MIT License",
    description="The SQL Linter for Humans",
    long_description=read("README.md"),
    # Make sure pypi is expecting markdown!
    long_description_content_type="text/markdown",
    author="Alan Cruickshank",
    author_email="alan@designingoverload.com",
    url="https://github.com/sqlfluff/sqlfluff",
    python_requires=">=3.6",
    keywords=[
        "sqlfluff",
        "sql",
        "linter",
        "formatter",
        "bigquery",
        "exasol",
        "hive",
        "mysql",
        "postgres",
        "redshift",
        "snowflake",
        "spark3",
        "sqlite",
        "teradata",
        "tsql",
        "dbt",
    ],
    project_urls={
        "Homepage": "https://www.sqlfluff.com",
        "Documentation": "https://docs.sqlfluff.com",
        "Changes": "https://github.com/sqlfluff/sqlfluff/blob/main/CHANGELOG.md",
        "Source": "https://github.com/sqlfluff/sqlfluff",
        "Issue Tracker": "https://github.com/sqlfluff/sqlfluff/issues",
        "Twitter": "https://twitter.com/SQLFluff",
        "Chat": "https://github.com/sqlfluff/sqlfluff#sqlfluff-on-slack",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        # 'Development Status :: 5 - Production/Stable',
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
        "Topic :: Software Development :: Quality Assurance",
    ],
    install_requires=[
        # Core
        "click>=7.1",
        "colorama>=0.3",
        "configparser",
        "oyaml",
        "Jinja2",
        # Used for diffcover plugin
        "diff-cover>=2.5.0",
        # Used for .sqlfluffignore
        "pathspec",
        # Used for finding os-specific application config dirs
        "appdirs",
        # Cached property for performance gains
        "cached-property",
        # dataclasses backport for python 3.6
        "dataclasses; python_version < '3.7'",
        # better type hints for older python versions
        "typing_extensions",
        # We provide a testing library for plugins in sqlfluff.testing
        "pytest",
        # For parsing pyproject.toml
        "toml",
        # For returning exceptions from multiprocessing.Pool.map()
        "tblib",
    ],
    entry_points={
        "console_scripts": [
            "sqlfluff = sqlfluff.cli.commands:cli",
        ],
        "diff_cover": ["sqlfluff = sqlfluff.diff_quality_plugin"],
        "sqlfluff": ["sqlfluff = sqlfluff.core.plugin.lib"],
    },
)
```
### 2 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 1827, End line: 1883

```python
@tsql_dialect.segment(replace=True)
class SetClauseListSegment(BaseSegment):
    """set clause list.

    Overriding ANSI to remove Delimited
    """

    type = "set_clause_list"
    match_grammar = Sequence(
        "SET",
        Indent,
        Ref("SetClauseSegment"),
        AnyNumberOf(
            Ref("CommaSegment"),
            Ref("SetClauseSegment"),
        ),
        Dedent,
    )


@tsql_dialect.segment(replace=True)
class SetClauseSegment(BaseSegment):
    """Set clause.

    Overriding ANSI to allow for ExpressionSegment on the right
    """

    type = "set_clause"

    match_grammar = Sequence(
        Ref("ColumnReferenceSegment"),
        Ref("EqualsSegment"),
        Ref("ExpressionSegment"),
    )


@tsql_dialect.segment(replace=True)
class DatePartFunctionNameSegment(BaseSegment):
    """DATEADD function name segment.

    Override to support DATEDIFF as well
    """

    type = "function_name"
    match_grammar = OneOf("DATEADD", "DATEDIFF", "DATEDIFF_BIG", "DATENAME")


@tsql_dialect.segment()
class PrintStatementSegment(BaseSegment):
    """PRINT statement segment."""

    type = "print_statement"
    match_grammar = Sequence(
        "PRINT",
        Ref("ExpressionSegment"),
        Ref("DelimiterSegment", optional=True),
    )
```
### 3 - src/sqlfluff/rules/L039.py:

Start line: 9, End line: 6

```python
"""Implementation of Rule L039."""
from typing import List, Optional

from sqlfluff.core.parser import WhitespaceSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L039(BaseRule):
    """Unnecessary whitespace found.

    | **Anti-pattern**

    .. code-block:: sql

        SELECT
            a,        b
        FROM foo

    | **Best practice**
    | Unless an indent or preceding a comment, whitespace should
    | be a single space.

    .. code-block:: sql

        SELECT
            a, b
        FROM foo
    """

    def _eval(self, context: RuleContext) -> Optional[List[LintResult]]:
        """Unnecessary whitespace."""
        # For the given segment, lint whitespace directly within it.
        prev_newline = True
        prev_whitespace = None
        violations = []
        for seg in context.segment.segments:
            if seg.is_type("newline"):
                prev_newline = True
                prev_whitespace = None
            elif seg.is_type("whitespace"):
                # This is to avoid indents
                if not prev_newline:
                    prev_whitespace = seg
                # We won't set prev_newline to False, just for whitespace
                # in case there's multiple indents, inserted by other rule
                # fixes (see #1713)
            elif seg.is_type("comment"):
                prev_newline = False
                prev_whitespace = None
            else:
                if prev_whitespace:
                    if prev_whitespace.raw != " ":
                        violations.append(
                            LintResult(
                                anchor=prev_whitespace,
                                fixes=[
                                    LintFix(
                                        "edit",
                                        prev_whitespace,
                                        WhitespaceSegment(),
                                    )
                                ],
                            )
                        )
                prev_newline = False
                prev_whitespace = None
        return violations or None
```
### 4 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 1683, End line: 1729

```python
@tsql_dialect.segment(replace=True)
class GroupByClauseSegment(BaseSegment):
    """A `GROUP BY` clause like in `SELECT`.

    Overriding ANSI to remove Delimited logic which assumes statements have been delimited
    """

    type = "groupby_clause"
    match_grammar = Sequence(
        "GROUP",
        "BY",
        Indent,
        OneOf(
            Ref("ColumnReferenceSegment"),
            # Can `GROUP BY 1`
            Ref("NumericLiteralSegment"),
            # Can `GROUP BY coalesce(col, 1)`
            Ref("ExpressionSegment"),
        ),
        AnyNumberOf(
            Ref("CommaSegment"),
            OneOf(
                Ref("ColumnReferenceSegment"),
                # Can `GROUP BY 1`
                Ref("NumericLiteralSegment"),
                # Can `GROUP BY coalesce(col, 1)`
                Ref("ExpressionSegment"),
            ),
        ),
        Dedent,
    )


@tsql_dialect.segment(replace=True)
class HavingClauseSegment(BaseSegment):
    """A `HAVING` clause like in `SELECT`.

    Overriding ANSI to remove StartsWith with greedy terminator
    """

    type = "having_clause"
    match_grammar = Sequence(
        "HAVING",
        Indent,
        OptionallyBracketed(Ref("ExpressionSegment")),
        Dedent,
    )
```
### 5 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 1033, End line: 1121

```python
@tsql_dialect.segment(replace=True)
class IntervalExpressionSegment(BaseSegment):
    """An interval expression segment.

    Not present in T-SQL.
    """

    type = "interval_expression"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class CreateExtensionStatementSegment(BaseSegment):
    """A `CREATE EXTENSION` statement.

    Not present in T-SQL.
    """

    type = "create_extension_statement"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class CreateModelStatementSegment(BaseSegment):
    """A BigQuery `CREATE MODEL` statement.

    Not present in T-SQL.
    """

    type = "create_model_statement"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class DropModelStatementSegment(BaseSegment):
    """A `DROP MODEL` statement.

    Not present in T-SQL.
    """

    type = "drop_MODELstatement"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class OverlapsClauseSegment(BaseSegment):
    """An `OVERLAPS` clause like in `SELECT.

    Not present in T-SQL.
    """

    type = "overlaps_clause"
    match_grammar = Nothing()


@tsql_dialect.segment()
class ConvertFunctionNameSegment(BaseSegment):
    """CONVERT function name segment.

    Need to be able to specify this as type function_name
    so that linting rules identify it properly
    """

    type = "function_name"
    match_grammar = Sequence("CONVERT")


@tsql_dialect.segment()
class CastFunctionNameSegment(BaseSegment):
    """CAST function name segment.

    Need to be able to specify this as type function_name
    so that linting rules identify it properly
    """

    type = "function_name"
    match_grammar = Sequence("CAST")


@tsql_dialect.segment()
class RankFunctionNameSegment(BaseSegment):
    """Rank function name segment.

    Need to be able to specify this as type function_name
    so that linting rules identify it properly
    """

    type = "function_name"
    match_grammar = OneOf("DENSE_RANK", "NTILE", "RANK", "ROW_NUMBER")
```
### 6 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 2026, End line: 2093

```python
@tsql_dialect.segment()
class TableHintSegment(BaseSegment):
    """Table Hint segment.

    https://docs.microsoft.com/en-us/sql/t-sql/queries/hints-transact-sql-table?view=sql-server-ver15
    """

    type = "query_hint_segment"
    match_grammar = OneOf(
        "NOEXPAND",
        Sequence(
            "INDEX",
            Bracketed(
                OneOf(Ref("IndexReferenceSegment"), Ref("NumericLiteralSegment")),
                AnyNumberOf(
                    Ref("CommaSegment"),
                    OneOf(
                        Ref("IndexReferenceSegment"),
                        Ref("NumericLiteralSegment"),
                    ),
                ),
            ),
        ),
        Sequence(
            "INDEX",
            Ref("EqualsSegment"),
            Bracketed(
                OneOf(Ref("IndexReferenceSegment"), Ref("NumericLiteralSegment")),
            ),
        ),
        "KEEPIDENTITY",
        "KEEPDEFAULTS",
        Sequence(
            "FORCESEEK",
            Bracketed(
                Ref("IndexReferenceSegment"),
                Bracketed(
                    Ref("SingleIdentifierGrammar"),
                    AnyNumberOf(Ref("CommaSegment"), Ref("SingleIdentifierGrammar")),
                ),
                optional=True,
            ),
        ),
        "FORCESCAN",
        "HOLDLOCK",
        "IGNORE_CONSTRAINTS",
        "IGNORE_TRIGGERS",
        "NOLOCK",
        "NOWAIT",
        "PAGLOCK",
        "READCOMMITTED",
        "READCOMMITTEDLOCK",
        "READPAST",
        "READUNCOMMITTED",
        "REPEATABLEREAD",
        "ROWLOCK",
        "SERIALIZABLE",
        "SNAPSHOT",
        Sequence(
            "SPATIAL_WINDOW_MAX_CELLS",
            Ref("EqualsSegment"),
            Ref("NumericLiteralSegment"),
        ),
        "TABLOCK",
        "TABLOCKX",
        "UPDLOCK",
        "XLOCK",
    )
```
### 7 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 1360, End line: 1441

```python
@tsql_dialect.segment()
class TableDistributionIndexClause(BaseSegment):
    """`CREATE TABLE` distribution / index clause.

    This is specific to Azure Synapse Analytics.
    """

    type = "table_distribution_index_clause"

    match_grammar = Sequence(
        "WITH",
        Bracketed(
            Delimited(
                Ref("TableDistributionClause"),
                Ref("TableIndexClause"),
                Ref("TableLocationClause"),
            ),
        ),
    )


@tsql_dialect.segment()
class TableDistributionClause(BaseSegment):
    """`CREATE TABLE` distribution clause.

    This is specific to Azure Synapse Analytics.
    """

    type = "table_distribution_clause"

    match_grammar = Sequence(
        "DISTRIBUTION",
        Ref("EqualsSegment"),
        OneOf(
            "REPLICATE",
            "ROUND_ROBIN",
            Sequence(
                "HASH",
                Bracketed(Ref("ColumnReferenceSegment")),
            ),
        ),
    )


@tsql_dialect.segment()
class TableIndexClause(BaseSegment):
    """`CREATE TABLE` table index clause.

    This is specific to Azure Synapse Analytics.
    """

    type = "table_index_clause"

    match_grammar = Sequence(
        OneOf(
            "HEAP",
            Sequence(
                "CLUSTERED",
                "COLUMNSTORE",
                "INDEX",
            ),
        ),
    )


@tsql_dialect.segment()
class TableLocationClause(BaseSegment):
    """`CREATE TABLE` location clause.

    This is specific to Azure Synapse Analytics (deprecated) or to an external table.
    """

    type = "table_location_clause"

    match_grammar = Sequence(
        "LOCATION",
        Ref("EqualsSegment"),
        OneOf(
            "USER_DB",  # Azure Synapse Analytics specific
            Ref("QuotedLiteralSegment"),  # External Table
        ),
    )
```
### 8 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 1906, End line: 2004

```python
@tsql_dialect.segment()
class QueryHintSegment(BaseSegment):
    """Query Hint segment.

    https://docs.microsoft.com/en-us/sql/t-sql/queries/hints-transact-sql-query?view=sql-server-ver15
    """

    type = "query_hint_segment"
    match_grammar = OneOf(
        Sequence(  # Azure Synapse Analytics specific
            "LABEL",
            Ref("EqualsSegment"),
            Ref("QuotedLiteralSegment"),
        ),
        Sequence(
            OneOf("HASH", "ORDER"),
            "GROUP",
        ),
        Sequence(OneOf("MERGE", "HASH", "CONCAT"), "UNION"),
        Sequence(OneOf("LOOP", "MERGE", "HASH"), "JOIN"),
        Sequence("EXPAND", "VIEWS"),
        Sequence(
            OneOf(
                "FAST",
                "MAXDOP",
                "MAXRECURSION",
                "QUERYTRACEON",
                Sequence(
                    OneOf(
                        "MAX_GRANT_PERCENT",
                        "MIN_GRANT_PERCENT",
                    ),
                    Ref("EqualsSegment"),
                ),
            ),
            Ref("NumericLiteralSegment"),
        ),
        Sequence("FORCE", "ORDER"),
        Sequence(
            OneOf("FORCE", "DISABLE"),
            OneOf("EXTERNALPUSHDOWN", "SCALEOUTEXECUTION"),
        ),
        Sequence(
            OneOf(
                "KEEP",
                "KEEPFIXED",
                "ROBUST",
            ),
            "PLAN",
        ),
        "IGNORE_NONCLUSTERED_COLUMNSTORE_INDEX",
        "NO_PERFORMANCE_SPOOL",
        Sequence(
            "OPTIMIZE",
            "FOR",
            OneOf(
                "UNKNOWN",
                Bracketed(
                    Ref("ParameterNameSegment"),
                    OneOf(
                        "UNKNOWN", Sequence(Ref("EqualsSegment"), Ref("LiteralGrammar"))
                    ),
                    AnyNumberOf(
                        Ref("CommaSegment"),
                        Ref("ParameterNameSegment"),
                        OneOf(
                            "UNKNOWN",
                            Sequence(Ref("EqualsSegment"), Ref("LiteralGrammar")),
                        ),
                    ),
                ),
            ),
        ),
        Sequence("PARAMETERIZATION", OneOf("SIMPLE", "FORCED")),
        "RECOMPILE",
        Sequence(
            "USE",
            "HINT",
            Bracketed(
                Ref("QuotedLiteralSegment"),
                AnyNumberOf(Ref("CommaSegment"), Ref("QuotedLiteralSegment")),
            ),
        ),
        Sequence(
            "USE",
            "PLAN",
            OneOf(Ref("QuotedLiteralSegment"), Ref("QuotedLiteralSegmentWithN")),
        ),
        Sequence(
            "TABLE",
            "HINT",
            Ref("ObjectReferenceSegment"),
            Ref("TableHintSegment"),
            AnyNumberOf(
                Ref("CommaSegment"),
                Ref("TableHintSegment"),
            ),
        ),
    )
```
### 9 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 839, End line: 854

```python
@tsql_dialect.segment(replace=True)
class FunctionParameterListGrammar(BaseSegment):
    """The parameters for a function ie. `(@city_name NVARCHAR(30), @postal_code NVARCHAR(15))`.

    Overriding ANSI (1) to optionally bracket and (2) remove Delimited
    """

    type = "function_parameter_list"
    # Function parameter list
    match_grammar = OptionallyBracketed(
        Ref("FunctionParameterGrammar"),
        AnyNumberOf(
            Ref("CommaSegment"),
            Ref("FunctionParameterGrammar"),
        ),
    )
```
### 10 - src/sqlfluff/dialects/dialect_tsql.py:

Start line: 2007, End line: 2023

```python
@tsql_dialect.segment(replace=True)
class PostTableExpressionGrammar(BaseSegment):
    """Table Hint clause.  Overloading the PostTableExpressionGrammar to implement.

    https://docs.microsoft.com/en-us/sql/t-sql/queries/hints-transact-sql-table?view=sql-server-ver15
    """

    match_grammar = Sequence(
        Sequence("WITH", optional=True),
        Bracketed(
            Ref("TableHintSegment"),
            AnyNumberOf(
                Ref("CommaSegment"),
                Ref("TableHintSegment"),
            ),
        ),
    )
```
