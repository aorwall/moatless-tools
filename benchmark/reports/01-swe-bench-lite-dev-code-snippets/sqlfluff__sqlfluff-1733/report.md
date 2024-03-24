# sqlfluff__sqlfluff-1733

| **sqlfluff/sqlfluff** | `a1579a16b1d8913d9d7c7d12add374a290bcc78c` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 1 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/src/sqlfluff/rules/L039.py b/src/sqlfluff/rules/L039.py
--- a/src/sqlfluff/rules/L039.py
+++ b/src/sqlfluff/rules/L039.py
@@ -44,7 +44,9 @@ def _eval(self, context: RuleContext) -> Optional[List[LintResult]]:
                 # This is to avoid indents
                 if not prev_newline:
                     prev_whitespace = seg
-                prev_newline = False
+                # We won't set prev_newline to False, just for whitespace
+                # in case there's multiple indents, inserted by other rule
+                # fixes (see #1713)
             elif seg.is_type("comment"):
                 prev_newline = False
                 prev_whitespace = None

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| src/sqlfluff/rules/L039.py | 47 | 49 | - | - | -


## Problem Statement

```
Extra space when first field moved to new line in a WITH statement
Note, the query below uses a `WITH` statement. If I just try to fix the SQL within the CTE, this works fine.

Given the following SQL:

\`\`\`sql
WITH example AS (
    SELECT my_id,
        other_thing,
        one_more
    FROM
        my_table
)

SELECT *
FROM example
\`\`\`

## Expected Behaviour

after running `sqlfluff fix` I'd expect (`my_id` gets moved down and indented properly):

\`\`\`sql
WITH example AS (
    SELECT
        my_id,
        other_thing,
        one_more
    FROM
        my_table
)

SELECT *
FROM example
\`\`\`

## Observed Behaviour

after running `sqlfluff fix` we get (notice that `my_id` is indented one extra space)

\`\`\`sql
WITH example AS (
    SELECT
         my_id,
        other_thing,
        one_more
    FROM
        my_table
)

SELECT *
FROM example
\`\`\`

## Steps to Reproduce

Noted above. Create a file with the initial SQL and fun `sqfluff fix` on it.

## Dialect

Running with default config.

## Version
Include the output of `sqlfluff --version` along with your Python version

sqlfluff, version 0.7.0
Python 3.7.5

## Configuration

Default config.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- |
| 1 | 1 src/sqlfluff/dialects/dialect_ansi.py | 1921 | 1949| 182 | 182 | 
| 2 | 2 src/sqlfluff/rules/L018.py | 41 | 151| 840 | 1022 | 
| 3 | 2 src/sqlfluff/rules/L018.py | 8 | 5| 218 | 1240 | 
| 4 | 3 src/sqlfluff/rules/L022.py | 39 | 44| 68 | 1308 | 
| 5 | 3 src/sqlfluff/rules/L022.py | 9 | 6| 203 | 1511 | 
| 6 | 3 src/sqlfluff/rules/L022.py | 45 | 194| 1297 | 2808 | 
| 7 | 4 src/sqlfluff/rules/L003.py | 323 | 410| 796 | 3604 | 
| 8 | 5 src/sqlfluff/rules/L023.py | 10 | 7| 250 | 3854 | 
| 9 | 6 src/sqlfluff/rules/L004.py | 45 | 101| 485 | 4339 | 
| 10 | 6 src/sqlfluff/dialects/dialect_ansi.py | 1952 | 1970| 131 | 4470 | 
| 11 | 7 src/sqlfluff/rules/L016.py | 451 | 530| 764 | 5234 | 
| 12 | 8 src/sqlfluff/dialects/dialect_exasol.py | 211 | 298| 608 | 5842 | 
| 13 | 9 src/sqlfluff/dialects/dialect_tsql.py | 108 | 228| 981 | 6823 | 
| 14 | 9 src/sqlfluff/dialects/dialect_exasol.py | 87 | 104| 176 | 6999 | 
| 15 | 9 src/sqlfluff/rules/L003.py | 52 | 74| 218 | 7217 | 
| 16 | 10 src/sqlfluff/rules/L036.py | 134 | 337| 1444 | 8661 | 
| 17 | 10 src/sqlfluff/rules/L003.py | 253 | 286| 256 | 8917 | 
| 18 | 10 src/sqlfluff/dialects/dialect_exasol.py | 353 | 371| 135 | 9052 | 
| 19 | 10 src/sqlfluff/dialects/dialect_exasol.py | 374 | 396| 174 | 9226 | 
| 20 | 10 src/sqlfluff/dialects/dialect_ansi.py | 2158 | 2155| 282 | 9508 | 
| 21 | 11 src/sqlfluff/rules/L019.py | 121 | 221| 895 | 10403 | 
| 22 | 11 src/sqlfluff/rules/L004.py | 9 | 6| 297 | 10700 | 
| 23 | 12 src/sqlfluff/core/rules/base.py | 165 | 197| 275 | 10975 | 
| 24 | 12 src/sqlfluff/dialects/dialect_tsql.py | 1144 | 1180| 240 | 11215 | 
| 25 | 12 src/sqlfluff/dialects/dialect_ansi.py | 1425 | 1583| 1094 | 12309 | 
| 26 | 12 src/sqlfluff/rules/L003.py | 13 | 10| 316 | 12625 | 
| 27 | 12 src/sqlfluff/rules/L003.py | 511 | 513| 499 | 13124 | 
| 28 | 13 src/sqlfluff/rules/L045.py | 9 | 6| 435 | 13559 | 
| 29 | 13 src/sqlfluff/rules/L003.py | 568 | 734| 1416 | 14975 | 
| 30 | 13 src/sqlfluff/dialects/dialect_exasol.py | 301 | 350| 310 | 15285 | 
| 31 | 14 src/sqlfluff/dialects/dialect_snowflake.py | 73 | 172| 750 | 16035 | 
| 32 | 14 src/sqlfluff/dialects/dialect_tsql.py | 1421 | 1442| 196 | 16231 | 
| 33 | 14 src/sqlfluff/rules/L016.py | 431 | 450| 178 | 16409 | 
| 34 | 14 src/sqlfluff/dialects/dialect_ansi.py | 1898 | 1918| 187 | 16596 | 
| 35 | 15 src/sqlfluff/dialects/dialect_mysql.py | 709 | 770| 351 | 16947 | 
| 36 | 16 src/sqlfluff/rules/L002.py | 43 | 75| 231 | 17178 | 
| 37 | 16 src/sqlfluff/dialects/dialect_tsql.py | 0 | 106| 748 | 17926 | 
| 38 | 16 src/sqlfluff/dialects/dialect_snowflake.py | 2042 | 2059| 135 | 18061 | 
| 39 | 17 src/sqlfluff/rules/L011.py | 43 | 104| 507 | 18568 | 
| 40 | 17 src/sqlfluff/dialects/dialect_exasol.py | 1188 | 1185| 212 | 18780 | 
| 41 | 17 src/sqlfluff/dialects/dialect_exasol.py | 1245 | 1282| 277 | 19057 | 
| 42 | 17 src/sqlfluff/dialects/dialect_exasol.py | 1610 | 1664| 260 | 19317 | 
| 43 | 18 src/sqlfluff/rules/L044.py | 14 | 11| 450 | 19767 | 
| 44 | 18 src/sqlfluff/dialects/dialect_tsql.py | 350 | 370| 172 | 19939 | 
| 45 | 18 src/sqlfluff/dialects/dialect_ansi.py | 2970 | 3000| 173 | 20112 | 
| 46 | 18 src/sqlfluff/rules/L003.py | 97 | 138| 338 | 20450 | 
| 47 | 18 src/sqlfluff/dialects/dialect_exasol.py | 105 | 209| 729 | 21179 | 
| 48 | 18 src/sqlfluff/dialects/dialect_snowflake.py | 1222 | 1361| 732 | 21911 | 
| 49 | 18 src/sqlfluff/rules/L036.py | 21 | 56| 187 | 22098 | 
| 50 | 18 src/sqlfluff/dialects/dialect_mysql.py | 1229 | 1247| 114 | 22212 | 
| 51 | 19 src/sqlfluff/dialects/dialect_postgres.py | 486 | 551| 448 | 22660 | 
| 52 | 19 src/sqlfluff/dialects/dialect_exasol.py | 989 | 1050| 340 | 23000 | 
| 53 | 20 src/sqlfluff/dialects/dialect_snowflake_keywords.py | 97 | 394| 932 | 23932 | 
| 54 | 20 src/sqlfluff/dialects/dialect_exasol.py | 1667 | 1692| 148 | 24080 | 
| 55 | 20 src/sqlfluff/dialects/dialect_tsql.py | 1883 | 1981| 577 | 24657 | 
| 56 | 20 src/sqlfluff/dialects/dialect_tsql.py | 976 | 973| 233 | 24890 | 
| 57 | 20 src/sqlfluff/dialects/dialect_exasol.py | 1573 | 1607| 176 | 25066 | 
| 58 | 20 src/sqlfluff/dialects/dialect_ansi.py | 2303 | 2345| 265 | 25331 | 
| 59 | 20 src/sqlfluff/dialects/dialect_snowflake.py | 1571 | 1704| 659 | 25990 | 
| 60 | 20 src/sqlfluff/dialects/dialect_exasol.py | 425 | 452| 167 | 26157 | 
| 61 | 20 src/sqlfluff/dialects/dialect_postgres.py | 1009 | 1185| 1186 | 27343 | 
| 62 | 20 src/sqlfluff/dialects/dialect_exasol.py | 3714 | 3737| 147 | 27490 | 
| 63 | 20 src/sqlfluff/dialects/dialect_ansi.py | 3026 | 3047| 105 | 27595 | 
| 64 | 20 src/sqlfluff/dialects/dialect_postgres.py | 916 | 997| 459 | 28054 | 
| 65 | 20 src/sqlfluff/rules/L003.py | 736 | 799| 547 | 28601 | 
| 66 | 20 src/sqlfluff/dialects/dialect_exasol.py | 1718 | 1746| 180 | 28781 | 
| 67 | 21 src/sqlfluff/dialects/dialect_teradata.py | 732 | 749| 128 | 28909 | 
| 68 | 22 src/sqlfluff/rules/L042.py | 52 | 93| 366 | 29275 | 
| 69 | 22 src/sqlfluff/dialects/dialect_mysql.py | 208 | 230| 141 | 29416 | 
| 70 | 22 src/sqlfluff/dialects/dialect_tsql.py | 1784 | 1801| 149 | 29565 | 
| 71 | 22 src/sqlfluff/dialects/dialect_tsql.py | 252 | 273| 175 | 29740 | 
| 72 | 22 src/sqlfluff/dialects/dialect_tsql.py | 384 | 381| 207 | 29947 | 
| 73 | 22 src/sqlfluff/dialects/dialect_snowflake.py | 697 | 720| 226 | 30173 | 
| 74 | 22 src/sqlfluff/dialects/dialect_mysql.py | 480 | 547| 376 | 30549 | 
| 75 | 22 src/sqlfluff/dialects/dialect_tsql.py | 1660 | 1706| 274 | 30823 | 
| 76 | 23 src/sqlfluff/rules/L008.py | 36 | 78| 411 | 31234 | 
| 77 | 23 src/sqlfluff/dialects/dialect_ansi.py | 3095 | 3092| 280 | 31514 | 
| 78 | 23 src/sqlfluff/rules/L023.py | 46 | 91| 348 | 31862 | 
| 79 | 23 src/sqlfluff/dialects/dialect_exasol.py | 1332 | 1347| 113 | 31975 | 
| 80 | 23 src/sqlfluff/dialects/dialect_mysql.py | 233 | 301| 503 | 32478 | 
| 81 | 23 src/sqlfluff/dialects/dialect_teradata.py | 714 | 711| 192 | 32670 | 
| 82 | 23 src/sqlfluff/dialects/dialect_exasol.py | 3035 | 3096| 340 | 33010 | 
| 83 | 23 src/sqlfluff/dialects/dialect_snowflake.py | 1847 | 1939| 570 | 33580 | 
| 84 | 24 src/sqlfluff/rules/L009.py | 9 | 6| 330 | 33910 | 
| 85 | 24 src/sqlfluff/rules/L003.py | 429 | 510| 783 | 34693 | 
| 86 | 25 src/sqlfluff/rules/L026.py | 57 | 100| 313 | 35006 | 
| 87 | 26 src/sqlfluff/dialects/dialect_tsql_keywords.py | 233 | 324| 555 | 35561 | 
| 88 | 26 src/sqlfluff/dialects/dialect_teradata.py | 752 | 769| 136 | 35697 | 
| 89 | 26 src/sqlfluff/dialects/dialect_tsql.py | 1323 | 1418| 459 | 36156 | 
| 90 | 26 src/sqlfluff/dialects/dialect_exasol.py | 1378 | 1432| 317 | 36473 | 
| 91 | 26 src/sqlfluff/dialects/dialect_postgres.py | 761 | 768| 762 | 37235 | 
| 92 | 26 src/sqlfluff/dialects/dialect_tsql.py | 323 | 347| 228 | 37463 | 
| 93 | 26 src/sqlfluff/dialects/dialect_tsql.py | 2003 | 2070| 390 | 37853 | 
| 94 | 26 src/sqlfluff/dialects/dialect_mysql.py | 370 | 395| 198 | 38051 | 
| 95 | 26 src/sqlfluff/dialects/dialect_mysql.py | 649 | 687| 206 | 38257 | 
| 96 | 26 src/sqlfluff/dialects/dialect_exasol.py | 408 | 405| 188 | 38445 | 
| 97 | 27 src/sqlfluff/rules/L005.py | 37 | 54| 147 | 38592 | 
| 98 | 27 src/sqlfluff/dialects/dialect_snowflake.py | 723 | 789| 338 | 38930 | 
| 99 | 27 src/sqlfluff/dialects/dialect_postgres.py | 663 | 681| 106 | 39036 | 
| 100 | 27 src/sqlfluff/dialects/dialect_tsql.py | 1123 | 1141| 186 | 39222 | 
| 101 | 27 src/sqlfluff/dialects/dialect_mysql.py | 304 | 367| 375 | 39597 | 
| 102 | 27 src/sqlfluff/dialects/dialect_postgres.py | 1747 | 1835| 479 | 40076 | 
| 103 | 27 src/sqlfluff/rules/L003.py | 411 | 427| 161 | 40237 | 
| 104 | 27 src/sqlfluff/rules/L002.py | 10 | 7| 216 | 40453 | 
| 105 | 27 src/sqlfluff/dialects/dialect_snowflake.py | 174 | 247| 601 | 41054 | 
| 106 | 27 src/sqlfluff/dialects/dialect_mysql.py | 841 | 860| 140 | 41194 | 
| 107 | 27 src/sqlfluff/dialects/dialect_tsql.py | 1750 | 1781| 195 | 41389 | 
| 108 | 27 src/sqlfluff/dialects/dialect_postgres.py | 706 | 729| 178 | 41567 | 
| 109 | 27 src/sqlfluff/dialects/dialect_postgres.py | 1395 | 1415| 165 | 41732 | 
| 110 | 27 src/sqlfluff/dialects/dialect_ansi.py | 2008 | 2021| 106 | 41838 | 
| 111 | 27 src/sqlfluff/dialects/dialect_snowflake.py | 874 | 897| 232 | 42070 | 
| 112 | 27 src/sqlfluff/dialects/dialect_exasol.py | 1972 | 2009| 174 | 42244 | 
| 113 | 27 src/sqlfluff/dialects/dialect_snowflake.py | 491 | 505| 104 | 42348 | 
| 114 | 28 src/sqlfluff/rules/L038.py | 42 | 74| 277 | 42625 | 
| 115 | 29 src/sqlfluff/rules/L031.py | 148 | 223| 614 | 43239 | 
| 116 | 30 src/sqlfluff/rules/L006.py | 46 | 79| 283 | 43522 | 
| 117 | 30 src/sqlfluff/dialects/dialect_tsql.py | 759 | 796| 210 | 43732 | 
| 118 | 30 src/sqlfluff/dialects/dialect_exasol.py | 1786 | 1829| 255 | 43987 | 
| 119 | 30 src/sqlfluff/rules/L019.py | 79 | 119| 391 | 44378 | 
| 120 | 30 src/sqlfluff/dialects/dialect_postgres.py | 82 | 144| 557 | 44935 | 
| 121 | 31 src/sqlfluff/rules/L041.py | 9 | 6| 531 | 45466 | 
| 122 | 31 src/sqlfluff/dialects/dialect_mysql.py | 435 | 432| 199 | 45665 | 
| 123 | 31 src/sqlfluff/dialects/dialect_exasol.py | 942 | 986| 262 | 45927 | 
| 124 | 31 src/sqlfluff/rules/L011.py | 12 | 9| 217 | 46144 | 
| 125 | 32 src/sqlfluff/dialects/dialect_spark3.py | 379 | 477| 587 | 46731 | 
| 126 | 32 src/sqlfluff/dialects/dialect_ansi.py | 3120 | 3159| 239 | 46970 | 
| 127 | 33 src/sqlfluff/rules/L010.py | 190 | 208| 244 | 47214 | 
| 128 | 33 src/sqlfluff/dialects/dialect_tsql.py | 1625 | 1657| 204 | 47418 | 
| 129 | 33 src/sqlfluff/dialects/dialect_mysql.py | 583 | 604| 140 | 47558 | 
| 130 | 33 src/sqlfluff/dialects/dialect_ansi.py | 2799 | 2796| 305 | 47863 | 
| 131 | 34 src/sqlfluff/rules/L048.py | 10 | 7| 295 | 48158 | 
| 132 | 34 src/sqlfluff/dialects/dialect_ansi.py | 1869 | 1895| 232 | 48390 | 
| 133 | 34 src/sqlfluff/dialects/dialect_teradata.py | 270 | 310| 362 | 48752 | 
| 134 | 35 src/sqlfluff/dialects/dialect_hive.py | 40 | 130| 699 | 49451 | 


## Missing Patch Files

 * 1: src/sqlfluff/rules/L039.py

### Hint

```
Does running `sqlfluff fix` again correct the SQL?
@tunetheweb yes, yes it does. Is that something that the user is supposed to do (run it multiple times) or is this indeed a bug?
Ideally not, but there are some circumstances where it’s understandable that would happen. This however seems an easy enough example where it should not happen.
This appears to be a combination of rules L036, L003, and L039 not playing nicely together.

The original error is rule L036 and it produces this:

\`\`\`sql
WITH example AS (
    SELECT
my_id,
        other_thing,
        one_more
    FROM
        my_table
)

SELECT *
FROM example
\`\`\`

That is, it moves the `my_id` down to the newline but does not even try to fix the indentation.

Then we have another run through and L003 spots the lack of indentation and fixes it by adding the first set of whitespace:

\`\`\`sql
WITH example AS (
    SELECT
    my_id,
        other_thing,
        one_more
    FROM
        my_table
)

SELECT *
FROM example
\`\`\`

Then we have another run through and L003 spots that there still isn't enough indentation and fixes it by adding the second set of whitespace:

\`\`\`sql
WITH example AS (
    SELECT
        my_id,
        other_thing,
        one_more
    FROM
        my_table
)

SELECT *
FROM example
\`\`\`

At this point we're all good.

However then L039 has a look. It never expects two sets of whitespace following a new line and is specifically coded to only assume one set of spaces (which it normally would be if the other rules hadn't interfered as it would be parsed as one big space), so it think's the second set is too much indentation, so it replaces it with a single space.

Then another run and L003 and the whitespace back in so we end up with two indents, and a single space.

Luckily the fix is easier than that explanation. PR coming up...


```

## Patch

```diff
diff --git a/src/sqlfluff/rules/L039.py b/src/sqlfluff/rules/L039.py
--- a/src/sqlfluff/rules/L039.py
+++ b/src/sqlfluff/rules/L039.py
@@ -44,7 +44,9 @@ def _eval(self, context: RuleContext) -> Optional[List[LintResult]]:
                 # This is to avoid indents
                 if not prev_newline:
                     prev_whitespace = seg
-                prev_newline = False
+                # We won't set prev_newline to False, just for whitespace
+                # in case there's multiple indents, inserted by other rule
+                # fixes (see #1713)
             elif seg.is_type("comment"):
                 prev_newline = False
                 prev_whitespace = None

```

## Test Patch

```diff
diff --git a/test/rules/std_L003_L036_L039_combo_test.py b/test/rules/std_L003_L036_L039_combo_test.py
new file mode 100644
--- /dev/null
+++ b/test/rules/std_L003_L036_L039_combo_test.py
@@ -0,0 +1,36 @@
+"""Tests issue #1373 doesn't reoccur.
+
+The combination of L003 (incorrect indentation), L036 (select targets),
+and L039 (unnecessary white space) can result in incorrect indentation.
+"""
+
+import sqlfluff
+
+
+def test__rules__std_L003_L036_L039():
+    """Verify that double indents don't flag L039."""
+    sql = """
+    WITH example AS (
+        SELECT my_id,
+            other_thing,
+            one_more
+        FROM
+            my_table
+    )
+
+    SELECT *
+    FROM example\n"""
+    fixed_sql = """
+    WITH example AS (
+        SELECT
+            my_id,
+            other_thing,
+            one_more
+        FROM
+            my_table
+    )
+
+    SELECT *
+    FROM example\n"""
+    result = sqlfluff.fix(sql)
+    assert result == fixed_sql
diff --git a/test/rules/std_L016_L36_combo.py b/test/rules/std_L016_L36_combo_test.py
similarity index 100%
rename from test/rules/std_L016_L36_combo.py
rename to test/rules/std_L016_L36_combo_test.py

```


## Code snippets

### 1 - src/sqlfluff/dialects/dialect_ansi.py:

Start line: 1921, End line: 1949

```python
@ansi_dialect.segment()
class CTEDefinitionSegment(BaseSegment):
    """A CTE Definition from a WITH statement.

    `tab (col1,col2) AS (SELECT a,b FROM x)`
    """

    type = "common_table_expression"
    match_grammar = Sequence(
        Ref("SingleIdentifierGrammar"),
        Bracketed(
            Ref("SingleIdentifierListSegment"),
            optional=True,
        ),
        "AS",
        Bracketed(
            # Ephemeral here to subdivide the query.
            Ref("SelectableGrammar", ephemeral_name="SelectableGrammar")
        ),
    )

    def get_identifier(self) -> BaseSegment:
        """Gets the identifier of this CTE.

        Note: it blindly get the first identifier it finds
        which given the structure of a CTE definition is
        usually the right one.
        """
        return self.get_child("identifier")
```
### 2 - src/sqlfluff/rules/L018.py:

Start line: 41, End line: 151

```python
@document_fix_compatible
class Rule_L018(BaseRule):

    def _eval(self, context: RuleContext) -> LintResult:
        """WITH clause closing bracket should be aligned with WITH keyword.

        Look for a with clause and evaluate the position of closing brackets.
        """
        # We only trigger on start_bracket (open parenthesis)
        if context.segment.is_type("with_compound_statement"):
            raw_stack_buff = list(context.raw_stack)
            # Look for the with keyword
            for seg in context.segment.segments:
                if seg.name.lower() == "with":
                    seg_line_no = seg.pos_marker.line_no
                    break
            else:  # pragma: no cover
                # This *could* happen if the with statement is unparsable,
                # in which case then the user will have to fix that first.
                if any(s.is_type("unparsable") for s in context.segment.segments):
                    return LintResult()
                # If it's parsable but we still didn't find a with, then
                # we should raise that.
                raise RuntimeError("Didn't find WITH keyword!")

            def indent_size_up_to(segs):
                seg_buff = []
                # Get any segments running up to the WITH
                for elem in reversed(segs):
                    if elem.is_type("newline"):
                        break
                    elif elem.is_meta:
                        continue
                    else:
                        seg_buff.append(elem)
                # reverse the indent if we have one
                if seg_buff:
                    seg_buff = list(reversed(seg_buff))
                indent_str = "".join(seg.raw for seg in seg_buff).replace(
                    "\t", " " * self.tab_space_size
                )
                indent_size = len(indent_str)
                return indent_size, indent_str

            balance = 0
            with_indent, with_indent_str = indent_size_up_to(raw_stack_buff)
            for seg in context.segment.iter_segments(
                expanding=["common_table_expression", "bracketed"], pass_through=True
            ):
                if seg.name == "start_bracket":
                    balance += 1
                elif seg.name == "end_bracket":
                    balance -= 1
                    if balance == 0:
                        closing_bracket_indent, _ = indent_size_up_to(raw_stack_buff)
                        indent_diff = closing_bracket_indent - with_indent
                        # Is indent of closing bracket not the same as
                        # indent of WITH keyword.
                        if seg.pos_marker.line_no == seg_line_no:
                            # Skip if it's the one-line version. That's ok
                            pass
                        elif indent_diff < 0:
                            return LintResult(
                                anchor=seg,
                                fixes=[
                                    LintFix(
                                        "create",
                                        seg,
                                        WhitespaceSegment(" " * (-indent_diff)),
                                    )
                                ],
                            )
                        elif indent_diff > 0:
                            # Is it all whitespace before the bracket on this line?
                            prev_segs_on_line = [
                                elem
                                for elem in context.segment.iter_segments(
                                    expanding=["common_table_expression", "bracketed"],
                                    pass_through=True,
                                )
                                if elem.pos_marker.line_no == seg.pos_marker.line_no
                                and elem.pos_marker.line_pos < seg.pos_marker.line_pos
                            ]
                            if all(
                                elem.is_type("whitespace") for elem in prev_segs_on_line
                            ):
                                # We can move it back, it's all whitespace
                                fixes = [
                                    LintFix(
                                        "create",
                                        seg,
                                        [WhitespaceSegment(with_indent_str)],
                                    )
                                ] + [
                                    LintFix("delete", elem)
                                    for elem in prev_segs_on_line
                                ]
                            else:
                                # We have to move it to a newline
                                fixes = [
                                    LintFix(
                                        "create",
                                        seg,
                                        [
                                            NewlineSegment(),
                                            WhitespaceSegment(with_indent_str),
                                        ],
                                    )
                                ]
                            return LintResult(anchor=seg, fixes=fixes)
                else:
                    raw_stack_buff.append(seg)
        return LintResult()
```
### 3 - src/sqlfluff/rules/L018.py:

Start line: 8, End line: 5

```python
"""Implementation of Rule L018."""

from sqlfluff.core.parser import NewlineSegment, WhitespaceSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L018(BaseRule):
    """WITH clause closing bracket should be aligned with WITH keyword.

    | **Anti-pattern**
    | The • character represents a space.
    | In this example, the closing bracket is not aligned with WITH keyword.

    .. code-block:: sql
       :force:

        WITH zoo AS (
            SELECT a FROM foo
        ••••)

        SELECT * FROM zoo

    | **Best practice**
    | Remove the spaces to align the WITH keyword with the closing bracket.

    .. code-block:: sql

        WITH zoo AS (
            SELECT a FROM foo
        )

        SELECT * FROM zoo

    """

    _works_on_unparsable = False
    config_keywords = ["tab_space_size"]
```
### 4 - src/sqlfluff/rules/L022.py:

Start line: 39, End line: 44

```python
@document_fix_compatible
class Rule_L022(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[List[LintResult]]:
        """Blank line expected but not found after CTE definition."""
        # Config type hints
        self.comma_style: str

        error_buffer = []
        # ... other code
```
### 5 - src/sqlfluff/rules/L022.py:

Start line: 9, End line: 6

```python
"""Implementation of Rule L022."""

from typing import Optional, List
from sqlfluff.core.parser import NewlineSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
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
### 6 - src/sqlfluff/rules/L022.py:

Start line: 45, End line: 194

```python
@document_fix_compatible
class Rule_L022(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[List[LintResult]]:
        # ... other code
        if context.segment.is_type("with_compound_statement"):
            # First we need to find all the commas, the end brackets, the
            # things that come after that and the blank lines in between.

            # Find all the closing brackets. They are our anchor points.
            bracket_indices = []
            expanded_segments = list(
                context.segment.iter_segments(expanding=["common_table_expression"])
            )
            for idx, seg in enumerate(expanded_segments):
                if seg.is_type("bracketed"):
                    bracket_indices.append(idx)

            # Work through each point and deal with it individually
            for bracket_idx in bracket_indices:
                forward_slice = expanded_segments[bracket_idx:]
                seg_idx = 1
                line_idx = 0
                comma_seg_idx = 0
                blank_lines = 0
                comma_line_idx = None
                line_blank = False
                comma_style = None
                line_starts = {}
                comment_lines = []

                self.logger.info(
                    "## CTE closing bracket found at %s, idx: %s. Forward slice: %.20r",
                    forward_slice[0].pos_marker,
                    bracket_idx,
                    "".join(elem.raw for elem in forward_slice),
                )

                # Work forward to map out the following segments.
                while (
                    forward_slice[seg_idx].is_type("comma")
                    or not forward_slice[seg_idx].is_code
                ):
                    if forward_slice[seg_idx].is_type("newline"):
                        if line_blank:
                            # It's a blank line!
                            blank_lines += 1
                        line_blank = True
                        line_idx += 1
                        line_starts[line_idx] = seg_idx + 1
                    elif forward_slice[seg_idx].is_type("comment"):
                        # Lines with comments aren't blank
                        line_blank = False
                        comment_lines.append(line_idx)
                    elif forward_slice[seg_idx].is_type("comma"):
                        # Keep track of where the comma is.
                        # We'll evaluate it later.
                        comma_line_idx = line_idx
                        comma_seg_idx = seg_idx
                    seg_idx += 1

                # Infer the comma style (NB this could be different for each case!)
                if comma_line_idx is None:
                    comma_style = "final"
                elif line_idx == 0:
                    comma_style = "oneline"
                elif comma_line_idx == 0:
                    comma_style = "trailing"
                elif comma_line_idx == line_idx:
                    comma_style = "leading"
                else:
                    comma_style = "floating"

                # Readout of findings
                self.logger.info(
                    "blank_lines: %s, comma_line_idx: %s. final_line_idx: %s, final_seg_idx: %s",
                    blank_lines,
                    comma_line_idx,
                    line_idx,
                    seg_idx,
                )
                self.logger.info(
                    "comma_style: %r, line_starts: %r, comment_lines: %r",
                    comma_style,
                    line_starts,
                    comment_lines,
                )

                if blank_lines < 1:
                    # We've got an issue
                    self.logger.info("!! Found CTE without enough blank lines.")

                    # Based on the current location of the comma we insert newlines
                    # to correct the issue.
                    fix_type = "create"  # In most cases we just insert newlines.
                    if comma_style == "oneline":
                        # Here we respect the target comma style to insert at the relevant point.
                        if self.comma_style == "trailing":
                            # Add a blank line after the comma
                            fix_point = forward_slice[comma_seg_idx + 1]
                            # Optionally here, if the segment we've landed on is
                            # whitespace then we REPLACE it rather than inserting.
                            if forward_slice[comma_seg_idx + 1].is_type("whitespace"):
                                fix_type = "edit"
                        elif self.comma_style == "leading":
                            # Add a blank line before the comma
                            fix_point = forward_slice[comma_seg_idx]
                        # In both cases it's a double newline.
                        num_newlines = 2
                    else:
                        # In the following cases we only care which one we're in
                        # when comments don't get in the way. If they *do*, then
                        # we just work around them.
                        if not comment_lines or line_idx - 1 not in comment_lines:
                            self.logger.info("Comment routines not applicable")
                            if comma_style in ("trailing", "final", "floating"):
                                # Detected an existing trailing comma or it's a final CTE,
                                # OR the comma isn't leading or trailing.
                                # If the preceding segment is whitespace, replace it
                                if forward_slice[seg_idx - 1].is_type("whitespace"):
                                    fix_point = forward_slice[seg_idx - 1]
                                    fix_type = "edit"
                                else:
                                    # Otherwise add a single newline before the end content.
                                    fix_point = forward_slice[seg_idx]
                            elif comma_style == "leading":
                                # Detected an existing leading comma.
                                fix_point = forward_slice[comma_seg_idx]
                        else:
                            self.logger.info("Handling preceding comments")
                            offset = 1
                            while line_idx - offset in comment_lines:
                                offset += 1
                            fix_point = forward_slice[
                                line_starts[line_idx - (offset - 1)]
                            ]
                        # Note: There is an edge case where this isn't enough, if
                        # comments are in strange places, but we'll catch them on
                        # the next iteration.
                        num_newlines = 1

                    fixes = [
                        LintFix(
                            fix_type,
                            fix_point,
                            [NewlineSegment()] * num_newlines,
                        )
                    ]
                    # Create a result, anchored on the start of the next content.
                    error_buffer.append(
                        LintResult(anchor=forward_slice[seg_idx], fixes=fixes)
                    )
        # Return the buffer if we have one.
        return error_buffer or None
```
### 7 - src/sqlfluff/rules/L003.py:

Start line: 323, End line: 410

```python
@document_fix_compatible
@document_configuration
class Rule_L003(BaseRule):

    def _eval(self, context: RuleContext) -> Optional[LintResult]:
        """Indentation not consistent with previous lines.

        To set the default tab size, set the `tab_space_size` value
        in the appropriate configuration.

        We compare each line (first non-whitespace element of the
        line), with the indentation of previous lines. The presence
        (or lack) of indent or dedent meta-characters indicate whether
        the indent is appropriate.

        - Any line is assessed by the indent level at the first non
          whitespace element.
        - Any increase in indentation may be _up to_ the number of
          indent characters.
        - Any line must be in line with the previous line which had
          the same indent balance at its start.
        - Apart from "whole" indents, a "hanging" indent is possible
          if the line starts in line with either the indent of the
          previous line or if it starts at the same indent as the *last*
          indent meta segment in the previous line.

        """
        # Config type hints
        self.tab_space_size: int
        self.indent_unit: str

        raw_stack = context.raw_stack

        # We ignore certain types (e.g. non-SQL scripts in functions)
        # so check if on ignore list
        if context.segment.type in self._ignore_types:
            return LintResult()
        for parent in context.parent_stack:
            if parent.type in self._ignore_types:
                return LintResult()

        # Memory keeps track of what we've seen
        if not context.memory:
            memory: dict = {
                # in_indent keeps track of whether we're in an indent right now
                "in_indent": True,
                # problem_lines keeps track of lines with problems so that we
                # don't compare to them.
                "problem_lines": [],
                # hanging_lines keeps track of hanging lines so that we don't
                # compare to them when assessing indent.
                "hanging_lines": [],
                # comment_lines keeps track of lines which are all comment.
                "comment_lines": [],
                # segments we've seen the last child of
                "finished": set(),
                # First non-whitespace node on a line.
                "trigger": None,
            }
        else:
            memory = context.memory

        if context.segment.is_type("newline"):
            memory["in_indent"] = True
        elif memory["in_indent"]:
            if context.segment.is_type("whitespace"):
                # it's whitespace, carry on
                pass
            elif context.segment.segments or (context.segment.is_meta and context.segment.indent_val != 0):  # type: ignore
                # it's not a raw segment or placeholder. Carry on.
                pass
            else:
                memory["in_indent"] = False
                # we're found a non-whitespace element. This is our trigger,
                # which we'll handle after this if-statement
                memory["trigger"] = context.segment
        else:
            # Not in indent and not a newline, don't trigger here.
            pass

        # Is this the last segment? If so, need to "flush" any leftovers.
        is_last = self._is_last_segment(
            context.segment, memory, context.parent_stack, context.siblings_post
        )

        if not context.segment.is_type("newline") and not is_last:
            # We only process complete lines or on the very last segment
            # (since there may not be a newline on the very last line)..
            return LintResult(memory=memory)

        if raw_stack and raw_stack[-1] is not context.segment:
            raw_stack = raw_stack + (context.segment,)
        # ... other code
```
### 8 - src/sqlfluff/rules/L023.py:

Start line: 10, End line: 7

```python
"""Implementation of Rule L023."""

from typing import Optional, List

from sqlfluff.core.parser import BaseSegment, WhitespaceSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult, RuleContext
from sqlfluff.core.rules.doc_decorators import document_fix_compatible


@document_fix_compatible
class Rule_L023(BaseRule):
    """Single whitespace expected after AS in WITH clause.

    | **Anti-pattern**

    .. code-block:: sql

        WITH plop AS(
            SELECT * FROM foo
        )

        SELECT a FROM plop


    | **Best practice**
    | The • character represents a space.
    | Add a space after AS, to avoid confusing
    | it for a function.

    .. code-block:: sql
       :force:

        WITH plop AS•(
            SELECT * FROM foo
        )

        SELECT a FROM plop
    """

    expected_mother_segment_type = "with_compound_statement"
    pre_segment_identifier = ("name", "as")
    post_segment_identifier = ("type", "bracketed")
    allow_newline = False
    expand_children: Optional[List[str]] = ["common_table_expression"]
```
### 9 - src/sqlfluff/rules/L004.py:

Start line: 45, End line: 101

```python
@document_fix_compatible
@document_configuration
class Rule_L004(BaseRule):
    def _eval(self, context: RuleContext) -> LintResult:
        """Incorrect indentation found in file."""
        # Config type hints
        self.tab_space_size: int
        self.indent_unit: str

        tab = "\t"
        space = " "
        correct_indent = (
            space * self.tab_space_size if self.indent_unit == "space" else tab
        )
        wrong_indent = (
            tab if self.indent_unit == "space" else space * self.tab_space_size
        )
        if (
            context.segment.is_type("whitespace")
            and wrong_indent in context.segment.raw
        ):
            fixes = []
            description = "Incorrect indentation type found in file."
            edit_indent = context.segment.raw.replace(wrong_indent, correct_indent)
            # Ensure that the number of space indents is a multiple of tab_space_size
            # before attempting to convert spaces to tabs to avoid mixed indents
            # unless we are converted tabs to spaces (indent_unit = space)
            if (
                (
                    self.indent_unit == "space"
                    or context.segment.raw.count(space) % self.tab_space_size == 0
                )
                # Only attempt a fix at the start of a newline for now
                and (
                    len(context.raw_stack) == 0
                    or context.raw_stack[-1].is_type("newline")
                )
            ):
                fixes = [
                    LintFix(
                        "edit",
                        context.segment,
                        WhitespaceSegment(raw=edit_indent),
                    )
                ]
            elif not (
                len(context.raw_stack) == 0 or context.raw_stack[-1].is_type("newline")
            ):
                # give a helpful message if the wrong indent has been found and is not at the start of a newline
                description += (
                    " The indent occurs after other text, so a manual fix is needed."
                )
            else:
                # If we get here, the indent_unit is tabs, and the number of spaces is not a multiple of tab_space_size
                description += " The number of spaces is not a multiple of tab_space_size, so a manual fix is needed."
            return LintResult(
                anchor=context.segment, fixes=fixes, description=description
            )
        return LintResult()
```
### 10 - src/sqlfluff/dialects/dialect_ansi.py:

Start line: 1952, End line: 1970

```python
@ansi_dialect.segment()
class WithCompoundStatementSegment(BaseSegment):
    """A `SELECT` statement preceded by a selection of `WITH` clauses.

    `WITH tab (col1,col2) AS (SELECT a,b FROM x)`
    """

    type = "with_compound_statement"
    # match grammar
    match_grammar = StartsWith("WITH")
    parse_grammar = Sequence(
        "WITH",
        Ref.keyword("RECURSIVE", optional=True),
        Delimited(
            Ref("CTEDefinitionSegment"),
            terminator=Ref.keyword("SELECT"),
        ),
        Ref("NonWithSelectableGrammar"),
    )
```
