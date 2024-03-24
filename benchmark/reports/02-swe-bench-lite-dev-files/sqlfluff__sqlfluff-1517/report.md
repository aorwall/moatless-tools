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
| 1 | 1 src/sqlfluff/rules/L041.py | 0 | 81| 508 | 508 | 
| 2 | 2 src/sqlfluff/dialects/dialect_ansi.py | 0 | 3160| 21074 | 21582 | 
| 3 | 3 src/sqlfluff/dialects/dialect_tsql.py | 0 | 984| 6054 | 27636 | 
| 4 | 4 src/sqlfluff/rules/L038.py | 0 | 71| 445 | 28081 | 
| 5 | 5 src/sqlfluff/rules/L022.py | 0 | 191| 1480 | 29561 | 
| 6 | 6 src/sqlfluff/core/linter/linter.py | 0 | 874| 6996 | 36557 | 


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

```python
"""The core ANSI dialect.

This is the core SQL grammar. We'll probably extend this or make it pluggable
for other dialects. Here we encode the structure of the language.

There shouldn't be any underlying "machinery" here, that should all
be defined elsewhere.

A lot of the inspiration for this sql grammar is taken from the cockroach
labs full sql grammar. In particular their way for dividing up the expression
grammar. Check out their docs, they're awesome.
https://www.cockroachlabs.com/docs/stable/sql-grammar.html#select_stmt
"""

from enum import Enum
from typing import Generator, List, Tuple, NamedTuple, Optional, Union

from sqlfluff.core.parser import (
    Matchable,
    BaseSegment,
    BaseFileSegment,
    KeywordSegment,
    SymbolSegment,
    Sequence,
    GreedyUntil,
    StartsWith,
    OneOf,
    Delimited,
    Bracketed,
    AnyNumberOf,
    Ref,
    SegmentGenerator,
    Anything,
    Indent,
    Dedent,
    Nothing,
    OptionallyBracketed,
    StringLexer,
    RegexLexer,
    CodeSegment,
    CommentSegment,
    WhitespaceSegment,
    NewlineSegment,
    StringParser,
    NamedParser,
    RegexParser,
    Conditional,
)

from sqlfluff.core.dialects.base import Dialect
from sqlfluff.core.dialects.common import AliasInfo
from sqlfluff.core.parser.segments.base import BracketedSegment

from sqlfluff.dialects.ansi_keywords import (
    ansi_reserved_keywords,
    ansi_unreserved_keywords,
)

ansi_dialect = Dialect("ansi", root_segment_name="FileSegment")

ansi_dialect.set_lexer_matchers(
    [
        RegexLexer("whitespace", r"[\t ]+", WhitespaceSegment),
        RegexLexer(
            "inline_comment",
            r"(--|#)[^\n]*",
            CommentSegment,
            segment_kwargs={"trim_start": ("--", "#")},
        ),
        RegexLexer(
            "block_comment",
            r"\/\*([^\*]|\*(?!\/))*\*\/",
            CommentSegment,
            subdivider=RegexLexer(
                "newline",
                r"\r\n|\n",
                NewlineSegment,
            ),
            trim_post_subdivide=RegexLexer(
                "whitespace",
                r"[\t ]+",
                WhitespaceSegment,
            ),
        ),
        RegexLexer("single_quote", r"'([^'\\]|\\.)*'", CodeSegment),
        RegexLexer("double_quote", r'"([^"\\]|\\.)*"', CodeSegment),
        RegexLexer("back_quote", r"`[^`]*`", CodeSegment),
        # See https://www.geeksforgeeks.org/postgresql-dollar-quoted-string-constants/
        RegexLexer("dollar_quote", r"\$(\w*)\$[^\1]*?\$\1\$", CodeSegment),
        RegexLexer(
            "numeric_literal", r"(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?", CodeSegment
        ),
        RegexLexer("not_equal", r"!=|<>", CodeSegment),
        RegexLexer("like_operator", r"!?~~?\*?", CodeSegment),
        StringLexer("greater_than_or_equal", ">=", CodeSegment),
        StringLexer("less_than_or_equal", "<=", CodeSegment),
        RegexLexer("newline", r"\r\n|\n", NewlineSegment),
        StringLexer("casting_operator", "::", CodeSegment),
        StringLexer("concat_operator", "||", CodeSegment),
        StringLexer("equals", "=", CodeSegment),
        StringLexer("greater_than", ">", CodeSegment),
        StringLexer("less_than", "<", CodeSegment),
        StringLexer("dot", ".", CodeSegment),
        StringLexer("comma", ",", CodeSegment, segment_kwargs={"type": "comma"}),
        StringLexer("plus", "+", CodeSegment),
        StringLexer("minus", "-", CodeSegment),
        StringLexer("divide", "/", CodeSegment),
        StringLexer("percent", "%", CodeSegment),
        StringLexer("ampersand", "&", CodeSegment),
        StringLexer("vertical_bar", "|", CodeSegment),
        StringLexer("caret", "^", CodeSegment),
        StringLexer("star", "*", CodeSegment),
        StringLexer("bracket_open", "(", CodeSegment),
        StringLexer("bracket_close", ")", CodeSegment),
        StringLexer("sq_bracket_open", "[", CodeSegment),
        StringLexer("sq_bracket_close", "]", CodeSegment),
        StringLexer("crly_bracket_open", "{", CodeSegment),
        StringLexer("crly_bracket_close", "}", CodeSegment),
        StringLexer("colon", ":", CodeSegment),
        StringLexer("semicolon", ";", CodeSegment),
        RegexLexer("code", r"[0-9a-zA-Z_]+", CodeSegment),
    ]
)

# Set the bare functions
ansi_dialect.sets("bare_functions").update(
    ["current_timestamp", "current_time", "current_date"]
)

# Set the datetime units
ansi_dialect.sets("datetime_units").update(
    [
        "DAY",
        "DAYOFYEAR",
        "HOUR",
        "MILLISECOND",
        "MINUTE",
        "MONTH",
        "QUARTER",
        "SECOND",
        "WEEK",
        "WEEKDAY",
        "YEAR",
    ]
)

# Set Keywords
ansi_dialect.sets("unreserved_keywords").update(
    [n.strip().upper() for n in ansi_unreserved_keywords.split("\n")]
)

ansi_dialect.sets("reserved_keywords").update(
    [n.strip().upper() for n in ansi_reserved_keywords.split("\n")]
)

# Bracket pairs (a set of tuples).
# (name, startref, endref, persists)
# NOTE: The `persists` value controls whether this type
# of bracket is persisted during matching to speed up other
# parts of the matching process. Round brackets are the most
# common and match the largest areas and so are sufficient.
ansi_dialect.sets("bracket_pairs").update(
    [
        ("round", "StartBracketSegment", "EndBracketSegment", True),
        ("square", "StartSquareBracketSegment", "EndSquareBracketSegment", False),
        ("curly", "StartCurlyBracketSegment", "EndCurlyBracketSegment", False),
    ]
)

# Set the value table functions. These are functions that, if they appear as
# an item in "FROM', are treated as returning a COLUMN, not a TABLE. Apparently,
# among dialects supported by SQLFluff, only BigQuery has this concept, but this
# set is defined in the ANSI dialect because:
# - It impacts core linter rules (see L020 and several other rules that subclass
#   from it) and how they interpret the contents of table_expressions
# - At least one other database (DB2) has the same value table function,
#   UNNEST(), as BigQuery. DB2 is not currently supported by SQLFluff.
ansi_dialect.sets("value_table_functions").update([])

ansi_dialect.add(
    # Real segments
    DelimiterSegment=Ref("SemicolonSegment"),
    SemicolonSegment=StringParser(
        ";", SymbolSegment, name="semicolon", type="statement_terminator"
    ),
    ColonSegment=StringParser(":", SymbolSegment, name="colon", type="colon"),
    SliceSegment=StringParser(":", SymbolSegment, name="slice", type="slice"),
    StartBracketSegment=StringParser(
        "(", SymbolSegment, name="start_bracket", type="start_bracket"
    ),
    EndBracketSegment=StringParser(
        ")", SymbolSegment, name="end_bracket", type="end_bracket"
    ),
    StartSquareBracketSegment=StringParser(
        "[", SymbolSegment, name="start_square_bracket", type="start_square_bracket"
    ),
    EndSquareBracketSegment=StringParser(
        "]", SymbolSegment, name="end_square_bracket", type="end_square_bracket"
    ),
    StartCurlyBracketSegment=StringParser(
        "{", SymbolSegment, name="start_curly_bracket", type="start_curly_bracket"
    ),
    EndCurlyBracketSegment=StringParser(
        "}", SymbolSegment, name="end_curly_bracket", type="end_curly_bracket"
    ),
    CommaSegment=StringParser(",", SymbolSegment, name="comma", type="comma"),
    DotSegment=StringParser(".", SymbolSegment, name="dot", type="dot"),
    StarSegment=StringParser("*", SymbolSegment, name="star", type="star"),
    TildeSegment=StringParser("~", SymbolSegment, name="tilde", type="tilde"),
    CastOperatorSegment=StringParser(
        "::", SymbolSegment, name="casting_operator", type="casting_operator"
    ),
    PlusSegment=StringParser("+", SymbolSegment, name="plus", type="binary_operator"),
    MinusSegment=StringParser("-", SymbolSegment, name="minus", type="binary_operator"),
    PositiveSegment=StringParser(
        "+", SymbolSegment, name="positive", type="sign_indicator"
    ),
    NegativeSegment=StringParser(
        "-", SymbolSegment, name="negative", type="sign_indicator"
    ),
    DivideSegment=StringParser(
        "/", SymbolSegment, name="divide", type="binary_operator"
    ),
    MultiplySegment=StringParser(
        "*", SymbolSegment, name="multiply", type="binary_operator"
    ),
    ModuloSegment=StringParser(
        "%", SymbolSegment, name="modulo", type="binary_operator"
    ),
    ConcatSegment=StringParser(
        "||", SymbolSegment, name="concatenate", type="binary_operator"
    ),
    BitwiseAndSegment=StringParser(
        "&", SymbolSegment, name="binary_and", type="binary_operator"
    ),
    BitwiseOrSegment=StringParser(
        "|", SymbolSegment, name="binary_or", type="binary_operator"
    ),
    BitwiseXorSegment=StringParser(
        "^", SymbolSegment, name="binary_xor", type="binary_operator"
    ),
    EqualsSegment=StringParser(
        "=", SymbolSegment, name="equals", type="comparison_operator"
    ),
    LikeOperatorSegment=NamedParser(
        "like_operator", SymbolSegment, name="like_operator", type="comparison_operator"
    ),
    GreaterThanSegment=StringParser(
        ">", SymbolSegment, name="greater_than", type="comparison_operator"
    ),
    LessThanSegment=StringParser(
        "<", SymbolSegment, name="less_than", type="comparison_operator"
    ),
    GreaterThanOrEqualToSegment=StringParser(
        ">=", SymbolSegment, name="greater_than_equal_to", type="comparison_operator"
    ),
    LessThanOrEqualToSegment=StringParser(
        "<=", SymbolSegment, name="less_than_equal_to", type="comparison_operator"
    ),
    NotEqualToSegment_a=StringParser(
        "!=", SymbolSegment, name="not_equal_to", type="comparison_operator"
    ),
    NotEqualToSegment_b=StringParser(
        "<>", SymbolSegment, name="not_equal_to", type="comparison_operator"
    ),
    # The following functions can be called without parentheses per ANSI specification
    BareFunctionSegment=SegmentGenerator(
        lambda dialect: RegexParser(
            r"^(" + r"|".join(dialect.sets("bare_functions")) + r")$",
            CodeSegment,
            name="bare_function",
            type="bare_function",
        )
    ),
    # The strange regex here it to make sure we don't accidentally match numeric literals. We
    # also use a regex to explicitly exclude disallowed keywords.
    NakedIdentifierSegment=SegmentGenerator(
        # Generate the anti template from the set of reserved keywords
        lambda dialect: RegexParser(
            r"[A-Z0-9_]*[A-Z][A-Z0-9_]*",
            CodeSegment,
            name="naked_identifier",
            type="identifier",
            anti_template=r"^(" + r"|".join(dialect.sets("reserved_keywords")) + r")$",
        )
    ),
    VersionIdentifierSegment=RegexParser(
        r"[A-Z0-9_.]*", CodeSegment, name="version", type="identifier"
    ),
    ParameterNameSegment=RegexParser(
        r"[A-Z][A-Z0-9_]*", CodeSegment, name="parameter", type="parameter"
    ),
    FunctionNameIdentifierSegment=RegexParser(
        r"[A-Z][A-Z0-9_]*",
        CodeSegment,
        name="function_name_identifier",
        type="function_name_identifier",
    ),
    # Maybe data types should be more restrictive?
    DatatypeIdentifierSegment=SegmentGenerator(
        # Generate the anti template from the set of reserved keywords
        lambda dialect: RegexParser(
            r"[A-Z][A-Z0-9_]*",
            CodeSegment,
            name="data_type_identifier",
            type="data_type_identifier",
            anti_template=r"^(NOT)$",  # TODO - this is a stopgap until we implement explicit data types
        ),
    ),
    # Ansi Intervals
    DatetimeUnitSegment=SegmentGenerator(
        lambda dialect: RegexParser(
            r"^(" + r"|".join(dialect.sets("datetime_units")) + r")$",
            CodeSegment,
            name="date_part",
            type="date_part",
        )
    ),
    QuotedIdentifierSegment=NamedParser(
        "double_quote", CodeSegment, name="quoted_identifier", type="identifier"
    ),
    QuotedLiteralSegment=NamedParser(
        "single_quote", CodeSegment, name="quoted_literal", type="literal"
    ),
    NumericLiteralSegment=NamedParser(
        "numeric_literal", CodeSegment, name="numeric_literal", type="literal"
    ),
    # NullSegment is defined seperately to the keyword so we can give it a different type
    NullLiteralSegment=StringParser(
        "null", KeywordSegment, name="null_literal", type="literal"
    ),
    TrueSegment=StringParser(
        "true", KeywordSegment, name="boolean_literal", type="literal"
    ),
    FalseSegment=StringParser(
        "false", KeywordSegment, name="boolean_literal", type="literal"
    ),
    # We use a GRAMMAR here not a Segment. Otherwise we get an unnecessary layer
    SingleIdentifierGrammar=OneOf(
        Ref("NakedIdentifierSegment"), Ref("QuotedIdentifierSegment")
    ),
    BooleanLiteralGrammar=OneOf(Ref("TrueSegment"), Ref("FalseSegment")),
    # We specifically define a group of arithmetic operators to make it easier to override this
    # if some dialects have different available operators
    ArithmeticBinaryOperatorGrammar=OneOf(
        Ref("PlusSegment"),
        Ref("MinusSegment"),
        Ref("DivideSegment"),
        Ref("MultiplySegment"),
        Ref("ModuloSegment"),
        Ref("BitwiseAndSegment"),
        Ref("BitwiseOrSegment"),
        Ref("BitwiseXorSegment"),
        Ref("BitwiseLShiftSegment"),
        Ref("BitwiseRShiftSegment"),
    ),
    StringBinaryOperatorGrammar=OneOf(Ref("ConcatSegment")),
    BooleanBinaryOperatorGrammar=OneOf(
        Ref("AndKeywordSegment"), Ref("OrKeywordSegment")
    ),
    ComparisonOperatorGrammar=OneOf(
        Ref("EqualsSegment"),
        Ref("GreaterThanSegment"),
        Ref("LessThanSegment"),
        Ref("GreaterThanOrEqualToSegment"),
        Ref("LessThanOrEqualToSegment"),
        Ref("NotEqualToSegment_a"),
        Ref("NotEqualToSegment_b"),
        Ref("LikeOperatorSegment"),
    ),
    # hookpoint for other dialects
    # e.g. EXASOL str to date cast with DATE '2021-01-01'
    DateTimeLiteralGrammar=Sequence(
        OneOf("DATE", "TIME", "TIMESTAMP", "INTERVAL"), Ref("QuotedLiteralSegment")
    ),
    LiteralGrammar=OneOf(
        Ref("QuotedLiteralSegment"),
        Ref("NumericLiteralSegment"),
        Ref("BooleanLiteralGrammar"),
        Ref("QualifiedNumericLiteralSegment"),
        # NB: Null is included in the literals, because it is a keyword which
        # can otherwise be easily mistaken for an identifier.
        Ref("NullLiteralSegment"),
        Ref("DateTimeLiteralGrammar"),
    ),
    AndKeywordSegment=StringParser("and", KeywordSegment, type="binary_operator"),
    OrKeywordSegment=StringParser("or", KeywordSegment, type="binary_operator"),
    # This is a placeholder for other dialects.
    PreTableFunctionKeywordsGrammar=Nothing(),
    BinaryOperatorGrammar=OneOf(
        Ref("ArithmeticBinaryOperatorGrammar"),
        Ref("StringBinaryOperatorGrammar"),
        Ref("BooleanBinaryOperatorGrammar"),
        Ref("ComparisonOperatorGrammar"),
    ),
    # This pattern is used in a lot of places.
    # Defined here to avoid repetition.
    BracketedColumnReferenceListGrammar=Bracketed(
        Delimited(
            Ref("ColumnReferenceSegment"),
            ephemeral_name="ColumnReferenceList",
        )
    ),
    OrReplaceGrammar=Sequence("OR", "REPLACE"),
    TemporaryTransientGrammar=OneOf("TRANSIENT", Ref("TemporaryGrammar")),
    TemporaryGrammar=OneOf("TEMP", "TEMPORARY"),
    IfExistsGrammar=Sequence("IF", "EXISTS"),
    IfNotExistsGrammar=Sequence("IF", "NOT", "EXISTS"),
    LikeGrammar=OneOf("LIKE", "RLIKE", "ILIKE"),
    IsClauseGrammar=OneOf(
        "NULL",
        "NAN",
        Ref("BooleanLiteralGrammar"),
    ),
    SelectClauseSegmentGrammar=Sequence(
        "SELECT",
        Ref("SelectClauseModifierSegment", optional=True),
        Indent,
        Delimited(
            Ref("SelectClauseElementSegment"),
            allow_trailing=True,
        ),
        # NB: The Dedent for the indent above lives in the
        # SelectStatementSegment so that it sits in the right
        # place corresponding to the whitespace.
    ),
    SelectClauseElementTerminatorGrammar=OneOf(
        "FROM",
        "WHERE",
        "ORDER",
        "LIMIT",
        Ref("CommaSegment"),
        Ref("SetOperatorSegment"),
    ),
    # Define these as grammars to allow child dialects to enable them (since they are non-standard
    # keywords)
    IsNullGrammar=Nothing(),
    NotNullGrammar=Nothing(),
    FromClauseTerminatorGrammar=OneOf(
        "WHERE",
        "LIMIT",
        Sequence("GROUP", "BY"),
        Sequence("ORDER", "BY"),
        "HAVING",
        "QUALIFY",
        "WINDOW",
        Ref("SetOperatorSegment"),
        Ref("WithNoSchemaBindingClauseSegment"),
    ),
    WhereClauseTerminatorGrammar=OneOf(
        "LIMIT",
        Sequence("GROUP", "BY"),
        Sequence("ORDER", "BY"),
        "HAVING",
        "QUALIFY",
        "WINDOW",
        "OVERLAPS",
    ),
    PrimaryKeyGrammar=Sequence("PRIMARY", "KEY"),
    ForeignKeyGrammar=Sequence("FOREIGN", "KEY"),
    # Odd syntax, but prevents eager parameters being confused for data types
    FunctionParameterGrammar=OneOf(
        Sequence(
            Ref("ParameterNameSegment", optional=True),
            OneOf(Sequence("ANY", "TYPE"), Ref("DatatypeSegment")),
        ),
        OneOf(Sequence("ANY", "TYPE"), Ref("DatatypeSegment")),
    ),
    # This is a placeholder for other dialects.
    SimpleArrayTypeGrammar=Nothing(),
    BaseExpressionElementGrammar=OneOf(
        Ref("LiteralGrammar"),
        Ref("BareFunctionSegment"),
        Ref("FunctionSegment"),
        Ref("IntervalExpressionSegment"),
        Ref("ColumnReferenceSegment"),
        Ref("ExpressionSegment"),
    ),
    FilterClauseGrammar=Sequence(
        "FILTER", Bracketed(Sequence("WHERE", Ref("ExpressionSegment")))
    ),
    FrameClauseUnitGrammar=OneOf("ROWS", "RANGE"),
    # It's as a sequence to allow to parametrize that in Postgres dialect with LATERAL
    JoinKeywords=Sequence("JOIN"),
)


@ansi_dialect.segment()
class FileSegment(BaseFileSegment):
    """A segment representing a whole file or script.

    This is also the default "root" segment of the dialect,
    and so is usually instantiated directly. It therefore
    has no match_grammar.
    """

    # NB: We don't need a match_grammar here because we're
    # going straight into instantiating it directly usually.
    parse_grammar = Delimited(
        Ref("StatementSegment"),
        delimiter=Ref("DelimiterSegment"),
        allow_gaps=True,
        allow_trailing=True,
    )

    def get_table_references(self):
        """Use parsed tree to extract table references."""
        references = set()
        for stmt in self.get_children("statement"):
            references |= stmt.get_table_references()
        return references


@ansi_dialect.segment()
class IntervalExpressionSegment(BaseSegment):
    """An interval expression segment."""

    type = "interval_expression"
    match_grammar = Sequence(
        "INTERVAL",
        OneOf(
            # The Numeric Version
            Sequence(
                Ref("NumericLiteralSegment"),
                OneOf(Ref("QuotedLiteralSegment"), Ref("DatetimeUnitSegment")),
            ),
            # The String version
            Ref("QuotedLiteralSegment"),
        ),
    )


@ansi_dialect.segment()
class ArrayLiteralSegment(BaseSegment):
    """An array literal segment."""

    type = "array_literal"
    match_grammar = Bracketed(
        Delimited(Ref("ExpressionSegment"), optional=True),
        bracket_type="square",
    )


@ansi_dialect.segment()
class DatatypeSegment(BaseSegment):
    """A data type segment.

    Supports timestamp with(out) time zone. Doesn't currently support intervals.
    """

    type = "data_type"
    match_grammar = OneOf(
        Sequence(
            OneOf("time", "timestamp"),
            Bracketed(Ref("NumericLiteralSegment"), optional=True),
            Sequence(OneOf("WITH", "WITHOUT"), "TIME", "ZONE", optional=True),
        ),
        Sequence(
            "DOUBLE",
            "PRECISION",
        ),
        Sequence(
            OneOf(
                Sequence(
                    OneOf("CHARACTER", "BINARY"),
                    OneOf("VARYING", Sequence("LARGE", "OBJECT")),
                ),
                Sequence(
                    # Some dialects allow optional qualification of data types with schemas
                    Sequence(
                        Ref("SingleIdentifierGrammar"),
                        Ref("DotSegment"),
                        allow_gaps=False,
                        optional=True,
                    ),
                    Ref("DatatypeIdentifierSegment"),
                    allow_gaps=False,
                ),
            ),
            Bracketed(
                OneOf(
                    Delimited(Ref("ExpressionSegment")),
                    # The brackets might be empty for some cases...
                    optional=True,
                ),
                # There may be no brackets for some data types
                optional=True,
            ),
            Ref("CharCharacterSetSegment", optional=True),
        ),
    )


# hookpoint
ansi_dialect.add(CharCharacterSetSegment=Nothing())


@ansi_dialect.segment()
class ObjectReferenceSegment(BaseSegment):
    """A reference to an object."""

    type = "object_reference"
    # match grammar (don't allow whitespace)
    match_grammar: Matchable = Delimited(
        Ref("SingleIdentifierGrammar"),
        delimiter=OneOf(
            Ref("DotSegment"), Sequence(Ref("DotSegment"), Ref("DotSegment"))
        ),
        terminator=OneOf(
            "ON",
            "AS",
            "USING",
            Ref("CommaSegment"),
            Ref("CastOperatorSegment"),
            Ref("StartSquareBracketSegment"),
            Ref("StartBracketSegment"),
            Ref("BinaryOperatorGrammar"),
            Ref("ColonSegment"),
            Ref("DelimiterSegment"),
            BracketedSegment,
        ),
        allow_gaps=False,
    )

    class ObjectReferencePart(NamedTuple):
        """Details about a table alias."""

        part: str  # Name of the part
        # Segment(s) comprising the part. Usuaully just one segment, but could
        # be multiple in dialects (e.g. BigQuery) that support unusual
        # characters in names (e.g. "-")
        segments: List[BaseSegment]

    @classmethod
    def _iter_reference_parts(cls, elem) -> Generator[ObjectReferencePart, None, None]:
        """Extract the elements of a reference and yield."""
        # trim on quotes and split out any dots.
        for part in elem.raw_trimmed().split("."):
            yield cls.ObjectReferencePart(part, [elem])

    def iter_raw_references(self) -> Generator[ObjectReferencePart, None, None]:
        """Generate a list of reference strings and elements.

        Each reference is an ObjectReferencePart. If some are split, then a
        segment may appear twice, but the substring will only appear once.
        """
        # Extract the references from those identifiers (because some may be quoted)
        for elem in self.recursive_crawl("identifier"):
            yield from self._iter_reference_parts(elem)

    def is_qualified(self):
        """Return if there is more than one element to the reference."""
        return len(list(self.iter_raw_references())) > 1

    def qualification(self):
        """Return the qualification type of this reference."""
        return "qualified" if self.is_qualified() else "unqualified"

    class ObjectReferenceLevel(Enum):
        """Labels for the "levels" of a reference.

        Note: Since SQLFluff does not have access to database catalog
        information, interpreting references will often be ambiguous.
        Typical example: The first part *may* refer to a schema, but that is
        almost always optional if referring to an object in some default or
        currently "active" schema. For this reason, use of this enum is optional
        and intended mainly to clarify the intent of the code -- no guarantees!
        Additionally, the terminology may vary by dialect, e.g. in BigQuery,
        "project" would be a more accurate term than "schema".
        """

        OBJECT = 1
        TABLE = 2
        SCHEMA = 3

    def extract_possible_references(
        self, level: Union[ObjectReferenceLevel, int]
    ) -> List[ObjectReferencePart]:
        """Extract possible references of a given level.

        "level" may be (but is not required to be) a value from the
        ObjectReferenceLevel enum defined above.

        NOTE: The base implementation here returns at most one part, but
        dialects such as BigQuery that support nesting (e.g. STRUCT) may return
        multiple reference parts.
        """
        level = self._level_to_int(level)
        refs = list(self.iter_raw_references())
        if len(refs) >= level:
            return [refs[-level]]
        return []

    @staticmethod
    def _level_to_int(level: Union[ObjectReferenceLevel, int]) -> int:
        # If it's an ObjectReferenceLevel, get the value. Otherwise, assume it's
        # an int.
        level = getattr(level, "value", level)
        assert isinstance(level, int)
        return level


@ansi_dialect.segment()
class TableReferenceSegment(ObjectReferenceSegment):
    """A reference to an table, CTE, subquery or alias."""

    type = "table_reference"


@ansi_dialect.segment()
class SchemaReferenceSegment(ObjectReferenceSegment):
    """A reference to a schema."""

    type = "schema_reference"


@ansi_dialect.segment()
class DatabaseReferenceSegment(ObjectReferenceSegment):
    """A reference to a database."""

    type = "database_reference"


@ansi_dialect.segment()
class IndexReferenceSegment(ObjectReferenceSegment):
    """A reference to an index."""

    type = "index_reference"


@ansi_dialect.segment()
class ExtensionReferenceSegment(ObjectReferenceSegment):
    """A reference to an extension."""

    type = "extension_reference"


@ansi_dialect.segment()
class ColumnReferenceSegment(ObjectReferenceSegment):
    """A reference to column, field or alias."""

    type = "column_reference"


@ansi_dialect.segment()
class SequenceReferenceSegment(ObjectReferenceSegment):
    """A reference to a sequence."""

    type = "sequence_reference"


@ansi_dialect.segment()
class SingleIdentifierListSegment(BaseSegment):
    """A comma delimited list of identifiers."""

    type = "identifier_list"
    match_grammar = Delimited(Ref("SingleIdentifierGrammar"))


@ansi_dialect.segment()
class ArrayAccessorSegment(BaseSegment):
    """An array accessor e.g. [3:4]."""

    type = "array_accessor"
    match_grammar = Bracketed(
        Delimited(
            OneOf(Ref("NumericLiteralSegment"), Ref("ExpressionSegment")),
            delimiter=Ref("SliceSegment"),
            ephemeral_name="ArrayAccessorContent",
        ),
        bracket_type="square",
    )


@ansi_dialect.segment()
class AliasedObjectReferenceSegment(BaseSegment):
    """A reference to an object with an `AS` clause."""

    type = "object_reference"
    match_grammar = Sequence(
        Ref("ObjectReferenceSegment"), Ref("AliasExpressionSegment")
    )


ansi_dialect.add(
    # This is a hook point to allow subclassing for other dialects
    AliasedTableReferenceGrammar=Sequence(
        Ref("TableReferenceSegment"), Ref("AliasExpressionSegment")
    )
)


@ansi_dialect.segment()
class AliasExpressionSegment(BaseSegment):
    """A reference to an object with an `AS` clause.

    The optional AS keyword allows both implicit and explicit aliasing.
    """

    type = "alias_expression"
    match_grammar = Sequence(
        Ref.keyword("AS", optional=True),
        OneOf(
            Sequence(
                Ref("SingleIdentifierGrammar"),
                # Column alias in VALUES clause
                Bracketed(Ref("SingleIdentifierListSegment"), optional=True),
            ),
            Ref("QuotedLiteralSegment"),
        ),
    )


@ansi_dialect.segment()
class ShorthandCastSegment(BaseSegment):
    """A casting operation using '::'."""

    type = "cast_expression"
    match_grammar = Sequence(
        Ref("CastOperatorSegment"), Ref("DatatypeSegment"), allow_gaps=False
    )


@ansi_dialect.segment()
class QualifiedNumericLiteralSegment(BaseSegment):
    """A numeric literal with a + or - sign preceding.

    The qualified numeric literal is a compound of a raw
    literal and a plus/minus sign. We do it this way rather
    than at the lexing step because the lexer doesn't deal
    well with ambiguity.
    """

    type = "numeric_literal"
    match_grammar = Sequence(
        OneOf(Ref("PlusSegment"), Ref("MinusSegment")),
        Ref("NumericLiteralSegment"),
        allow_gaps=False,
    )


ansi_dialect.add(
    # FunctionContentsExpressionGrammar intended as a hook to override
    # in other dialects.
    FunctionContentsExpressionGrammar=Ref("ExpressionSegment"),
    FunctionContentsGrammar=AnyNumberOf(
        Ref("ExpressionSegment"),
        # A Cast-like function
        Sequence(Ref("ExpressionSegment"), "AS", Ref("DatatypeSegment")),
        # An extract-like or substring-like function
        Sequence(
            OneOf(Ref("DatetimeUnitSegment"), Ref("ExpressionSegment")),
            "FROM",
            Ref("ExpressionSegment"),
        ),
        Sequence(
            # Allow an optional distinct keyword here.
            Ref.keyword("DISTINCT", optional=True),
            OneOf(
                # Most functions will be using the delimited route
                # but for COUNT(*) or similar we allow the star segment
                # here.
                Ref("StarSegment"),
                Delimited(Ref("FunctionContentsExpressionGrammar")),
            ),
        ),
        Ref(
            "OrderByClauseSegment"
        ),  # used by string_agg (postgres), group_concat (exasol), listagg (snowflake)...
        Sequence(Ref.keyword("SEPARATOR"), Ref("LiteralGrammar")),
        # like a function call: POSITION ( 'QL' IN 'SQL')
        Sequence(
            OneOf(Ref("QuotedLiteralSegment"), Ref("SingleIdentifierGrammar")),
            "IN",
            OneOf(Ref("QuotedLiteralSegment"), Ref("SingleIdentifierGrammar")),
        ),
        Sequence(OneOf("IGNORE", "RESPECT"), "NULLS"),
    ),
    PostFunctionGrammar=OneOf(
        # Optional OVER suffix for window functions.
        # This is supported in bigquery & postgres (and its derivatives)
        # and so is included here for now.
        Ref("OverClauseSegment"),
        # Filter clause supported by both Postgres and SQLite
        Ref("FilterClauseGrammar"),
    ),
)


@ansi_dialect.segment()
class OverClauseSegment(BaseSegment):
    """An OVER clause for window functions."""

    type = "over_clause"
    match_grammar = Sequence(
        "OVER",
        OneOf(
            Ref("SingleIdentifierGrammar"),  # Window name
            Bracketed(
                Ref("WindowSpecificationSegment", optional=True),
            ),
        ),
    )


@ansi_dialect.segment()
class WindowSpecificationSegment(BaseSegment):
    """Window specification within OVER(...)."""

    type = "window_specification"
    match_grammar = Sequence(
        Ref("SingleIdentifierGrammar", optional=True),  # "Base" window name
        Ref("PartitionClauseSegment", optional=True),
        Ref("OrderByClauseSegment", optional=True),
        Ref("FrameClauseSegment", optional=True),
        optional=True,
        ephemeral_name="OverClauseContent",
    )


@ansi_dialect.segment()
class FunctionNameSegment(BaseSegment):
    """Function name, including any prefix bits, e.g. project or schema."""

    type = "function_name"
    match_grammar = Sequence(
        # Project name, schema identifier, etc.
        AnyNumberOf(
            Sequence(
                Ref("SingleIdentifierGrammar"),
                Ref("DotSegment"),
            ),
        ),
        # Base function name
        OneOf(
            Ref("FunctionNameIdentifierSegment"),
            Ref("QuotedIdentifierSegment"),
        ),
        allow_gaps=False,
    )


@ansi_dialect.segment()
class DatePartClause(BaseSegment):
    """DatePart clause for use within DATEADD() or related functions."""

    type = "date_part"

    match_grammar = OneOf(
        "DAY",
        "DAYOFYEAR",
        "HOUR",
        "MINUTE",
        "MONTH",
        "QUARTER",
        "SECOND",
        "WEEK",
        "WEEKDAY",
        "YEAR",
    )


@ansi_dialect.segment()
class FunctionSegment(BaseSegment):
    """A scalar or aggregate function.

    Maybe in the future we should distinguish between
    aggregate functions and other functions. For now
    we treat them the same because they look the same
    for our purposes.
    """

    type = "function"
    match_grammar = OneOf(
        Sequence(
            Sequence(
                Ref("DateAddFunctionNameSegment"),
                Bracketed(
                    Delimited(
                        Ref("DatePartClause"),
                        Ref(
                            "FunctionContentsGrammar",
                            # The brackets might be empty for some functions...
                            optional=True,
                            ephemeral_name="FunctionContentsGrammar",
                        ),
                    )
                ),
            )
        ),
        Sequence(
            Sequence(
                AnyNumberOf(
                    Ref("FunctionNameSegment"),
                    max_times=1,
                    min_times=1,
                    exclude=Ref("DateAddFunctionNameSegment"),
                ),
                Bracketed(
                    Ref(
                        "FunctionContentsGrammar",
                        # The brackets might be empty for some functions...
                        optional=True,
                        ephemeral_name="FunctionContentsGrammar",
                    )
                ),
            ),
            Ref("PostFunctionGrammar", optional=True),
        ),
    )


@ansi_dialect.segment()
class PartitionClauseSegment(BaseSegment):
    """A `PARTITION BY` for window functions."""

    type = "partitionby_clause"
    match_grammar = StartsWith(
        "PARTITION",
        terminator=OneOf("ORDER", Ref("FrameClauseUnitGrammar")),
        enforce_whitespace_preceding_terminator=True,
    )
    parse_grammar = Sequence(
        "PARTITION",
        "BY",
        Indent,
        # Brackets are optional in a partition by statement
        OptionallyBracketed(Delimited(Ref("ExpressionSegment"))),
        Dedent,
    )


@ansi_dialect.segment()
class FrameClauseSegment(BaseSegment):
    """A frame clause for window functions.

    As specified in https://docs.oracle.com/cd/E17952_01/mysql-8.0-en/window-functions-frames.html
    """

    type = "frame_clause"

    _frame_extent = OneOf(
        Sequence("CURRENT", "ROW"),
        Sequence(
            OneOf(Ref("NumericLiteralSegment"), "UNBOUNDED"),
            OneOf("PRECEDING", "FOLLOWING"),
        ),
    )

    match_grammar = Sequence(
        Ref("FrameClauseUnitGrammar"),
        OneOf(_frame_extent, Sequence("BETWEEN", _frame_extent, "AND", _frame_extent)),
    )


ansi_dialect.add(
    # This is a hook point to allow subclassing for other dialects
    PostTableExpressionGrammar=Nothing()
)


@ansi_dialect.segment()
class FromExpressionElementSegment(BaseSegment):
    """A table expression."""

    type = "from_expression_element"
    match_grammar = Sequence(
        Ref("PreTableFunctionKeywordsGrammar", optional=True),
        OptionallyBracketed(Ref("TableExpressionSegment")),
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/arrays#flattening_arrays
        Sequence("WITH", "OFFSET", optional=True),
        Ref("AliasExpressionSegment", optional=True),
        Ref("PostTableExpressionGrammar", optional=True),
    )

    def get_eventual_alias(self) -> Optional[AliasInfo]:
        """Return the eventual table name referred to by this table expression.

        Returns:
            :obj:`tuple` of (:obj:`str`, :obj:`BaseSegment`, :obj:`bool`) containing
                a string representation of the alias, a reference to the
                segment containing it, and whether it's an alias.

        """
        alias_expression = self.get_child("alias_expression")
        tbl_expression = self.get_child("table_expression")
        if not tbl_expression:  # pragma: no cover
            tbl_expression = self.get_child("bracketed").get_child("table_expression")
        ref = tbl_expression.get_child("object_reference")
        if alias_expression:
            # If it has an alias, return that
            segment = alias_expression.get_child("identifier")
            return AliasInfo(segment.raw, segment, True, self, alias_expression, ref)

        # If not return the object name (or None if there isn't one)
        # ref = self.get_child("object_reference")
        if ref:
            # Return the last element of the reference.
            penultimate_ref: ObjectReferenceSegment.ObjectReferencePart = list(
                ref.iter_raw_references()
            )[-1]
            return AliasInfo(
                penultimate_ref.part,
                penultimate_ref.segments[0],
                False,
                self,
                None,
                ref,
            )
        # No references or alias, return None
        return None


@ansi_dialect.segment()
class FromExpressionSegment(BaseSegment):
    """A from expression segment."""

    type = "from_expression"
    match_grammar = Sequence(
        Indent,
        OneOf(
            # check first for MLTableExpression, because of possible FunctionSegment in MainTableExpression
            Ref("MLTableExpressionSegment"),
            Ref("FromExpressionElementSegment"),
        ),
        Conditional(Dedent, indented_joins=False),
        AnyNumberOf(
            Ref("JoinClauseSegment"), Ref("JoinLikeClauseGrammar"), optional=True
        ),
        Conditional(Dedent, indented_joins=True),
    )


@ansi_dialect.segment()
class TableExpressionSegment(BaseSegment):
    """The main table expression e.g. within a FROM clause."""

    type = "table_expression"
    match_grammar = OneOf(
        Ref("BareFunctionSegment"),
        Ref("FunctionSegment"),
        Ref("TableReferenceSegment"),
        # Nested Selects
        Bracketed(Ref("SelectableGrammar")),
        # Values clause?
    )


@ansi_dialect.segment()
class WildcardIdentifierSegment(ObjectReferenceSegment):
    """Any identifier of the form a.b.*.

    This inherits iter_raw_references from the
    ObjectReferenceSegment.
    """

    type = "wildcard_identifier"
    match_grammar = Sequence(
        # *, blah.*, blah.blah.*, etc.
        AnyNumberOf(
            Sequence(Ref("SingleIdentifierGrammar"), Ref("DotSegment"), allow_gaps=True)
        ),
        Ref("StarSegment"),
        allow_gaps=False,
    )

    def iter_raw_references(self):
        """Generate a list of reference strings and elements.

        Each element is a tuple of (str, segment). If some are
        split, then a segment may appear twice, but the substring
        will only appear once.
        """
        # Extract the references from those identifiers (because some may be quoted)
        for elem in self.recursive_crawl("identifier", "star"):
            yield from self._iter_reference_parts(elem)


@ansi_dialect.segment()
class WildcardExpressionSegment(BaseSegment):
    """A star (*) expression for a SELECT clause.

    This is separate from the identifier to allow for
    some dialects which extend this logic to allow
    REPLACE, EXCEPT or similar clauses e.g. BigQuery.
    """

    type = "wildcard_expression"
    match_grammar = Sequence(
        # *, blah.*, blah.blah.*, etc.
        Ref("WildcardIdentifierSegment")
    )


@ansi_dialect.segment()
class SelectClauseElementSegment(BaseSegment):
    """An element in the targets of a select statement."""

    type = "select_clause_element"
    # Important to split elements before parsing, otherwise debugging is really hard.
    match_grammar = GreedyUntil(
        Ref("SelectClauseElementTerminatorGrammar"),
        enforce_whitespace_preceding_terminator=True,
    )

    parse_grammar = OneOf(
        # *, blah.*, blah.blah.*, etc.
        Ref("WildcardExpressionSegment"),
        Sequence(
            Ref("BaseExpressionElementGrammar"),
            Ref("AliasExpressionSegment", optional=True),
        ),
    )


@ansi_dialect.segment()
class SelectClauseModifierSegment(BaseSegment):
    """Things that come after SELECT but before the columns."""

    type = "select_clause_modifier"
    match_grammar = OneOf(
        "DISTINCT",
        "ALL",
    )


@ansi_dialect.segment()
class SelectClauseSegment(BaseSegment):
    """A group of elements in a select target statement."""

    type = "select_clause"
    match_grammar = StartsWith(
        Sequence("SELECT", Ref("WildcardExpressionSegment", optional=True)),
        terminator=OneOf(
            "FROM",
            "WHERE",
            "ORDER",
            "LIMIT",
            "OVERLAPS",
            Ref("SetOperatorSegment"),
        ),
        enforce_whitespace_preceding_terminator=True,
    )

    parse_grammar = Ref("SelectClauseSegmentGrammar")


@ansi_dialect.segment()
class JoinClauseSegment(BaseSegment):
    """Any number of join clauses, including the `JOIN` keyword."""

    type = "join_clause"
    match_grammar = Sequence(
        # NB These qualifiers are optional
        # TODO: Allow nested joins like:
        # ....FROM S1.T1 t1 LEFT JOIN ( S2.T2 t2 JOIN S3.T3 t3 ON t2.col1=t3.col1) ON tab1.col1 = tab2.col1
        OneOf(
            "CROSS",
            "INNER",
            Sequence(
                OneOf(
                    "FULL",
                    "LEFT",
                    "RIGHT",
                ),
                Ref.keyword("OUTER", optional=True),
            ),
            optional=True,
        ),
        Ref("JoinKeywords"),
        Indent,
        Sequence(
            Ref("FromExpressionElementSegment"),
            Conditional(Dedent, indented_using_on=False),
            # NB: this is optional
            OneOf(
                # ON clause
                Ref("JoinOnConditionSegment"),
                # USING clause
                Sequence(
                    "USING",
                    Indent,
                    Bracketed(
                        # NB: We don't use BracketedColumnReferenceListGrammar
                        # here because we're just using SingleIdentifierGrammar,
                        # rather than ObjectReferenceSegment or ColumnReferenceSegment.
                        # This is a) so that we don't lint it as a reference and
                        # b) because the column will probably be returned anyway
                        # during parsing.
                        Delimited(
                            Ref("SingleIdentifierGrammar"),
                            ephemeral_name="UsingClauseContents",
                        )
                    ),
                    Dedent,
                ),
                # Unqualified joins *are* allowed. They just might not
                # be a good idea.
                optional=True,
            ),
            Conditional(Indent, indented_using_on=False),
        ),
        Dedent,
    )

    def get_eventual_alias(self) -> AliasInfo:
        """Return the eventual table name referred to by this join clause."""
        from_expression_element = self.get_child("from_expression_element")
        return from_expression_element.get_eventual_alias()


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

    def get_eventual_aliases(self) -> List[Tuple[BaseSegment, AliasInfo]]:
        """List the eventual aliases of this from clause.

        Comes as a list of tuples (table expr, tuple (string, segment, bool)).
        """
        buff = []
        direct_table_children = []
        join_clauses = []

        for from_expression in self.get_children("from_expression"):
            direct_table_children += from_expression.get_children(
                "from_expression_element"
            )
            join_clauses += from_expression.get_children("join_clause")

        # Iterate through the potential sources of aliases
        for clause in (*direct_table_children, *join_clauses):
            ref: AliasInfo = clause.get_eventual_alias()
            # Only append if non null. A None reference, may
            # indicate a generator expression or similar.
            table_expr = (
                clause
                if clause in direct_table_children
                else clause.get_child("from_expression_element")
            )
            if ref:
                buff.append((table_expr, ref))
        return buff


@ansi_dialect.segment()
class CaseExpressionSegment(BaseSegment):
    """A `CASE WHEN` clause."""

    type = "case_expression"
    match_grammar = OneOf(
        Sequence(
            "CASE",
            Indent,
            AnyNumberOf(
                Sequence(
                    "WHEN",
                    Indent,
                    Ref("ExpressionSegment"),
                    "THEN",
                    Ref("ExpressionSegment"),
                    Dedent,
                )
            ),
            Sequence("ELSE", Indent, Ref("ExpressionSegment"), Dedent, optional=True),
            Dedent,
            "END",
        ),
        Sequence(
            "CASE",
            OneOf(Ref("ExpressionSegment")),
            Indent,
            AnyNumberOf(
                Sequence(
                    "WHEN",
                    Indent,
                    Ref("ExpressionSegment"),
                    "THEN",
                    Ref("ExpressionSegment"),
                    Dedent,
                )
            ),
            Sequence("ELSE", Indent, Ref("ExpressionSegment"), Dedent, optional=True),
            Dedent,
            "END",
        ),
    )


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


@ansi_dialect.segment()
class BitwiseLShiftSegment(BaseSegment):
    """Bitwise left-shift operator."""

    type = "binary_operator"
    match_grammar = Sequence(
        Ref("LessThanSegment"), Ref("LessThanSegment"), allow_gaps=False
    )


@ansi_dialect.segment()
class BitwiseRShiftSegment(BaseSegment):
    """Bitwise right-shift operator."""

    type = "binary_operator"
    match_grammar = Sequence(
        Ref("GreaterThanSegment"), Ref("GreaterThanSegment"), allow_gaps=False
    )


@ansi_dialect.segment()
class ExpressionSegment(BaseSegment):
    """A expression, either arithmetic or boolean.

    NB: This is potentially VERY recursive and

    mostly uses the grammars above. This version
    also doesn't bound itself first, and so is potentially
    VERY SLOW. I don't really like this solution.

    We rely on elements of the expression to bound
    themselves rather than bounding at the expression
    level. Trying to bound the ExpressionSegment itself
    has been too unstable and not resilient enough to
    other bugs.
    """

    type = "expression"
    match_grammar = Ref("Expression_A_Grammar")


@ansi_dialect.segment()
class WhereClauseSegment(BaseSegment):
    """A `WHERE` clause like in `SELECT` or `INSERT`."""

    type = "where_clause"
    match_grammar = StartsWith(
        "WHERE",
        terminator=Ref("WhereClauseTerminatorGrammar"),
        enforce_whitespace_preceding_terminator=True,
    )
    parse_grammar = Sequence(
        "WHERE",
        Indent,
        OptionallyBracketed(Ref("ExpressionSegment")),
        Dedent,
    )


@ansi_dialect.segment()
class OrderByClauseSegment(BaseSegment):
    """A `ORDER BY` clause like in `SELECT`."""

    type = "orderby_clause"
    match_grammar = StartsWith(
        "ORDER",
        terminator=OneOf(
            "LIMIT",
            "HAVING",
            "QUALIFY",
            # For window functions
            "WINDOW",
            Ref("FrameClauseUnitGrammar"),
            "SEPARATOR",
        ),
    )
    parse_grammar = Sequence(
        "ORDER",
        "BY",
        Indent,
        Delimited(
            Sequence(
                OneOf(
                    Ref("ColumnReferenceSegment"),
                    # Can `ORDER BY 1`
                    Ref("NumericLiteralSegment"),
                    # Can order by an expression
                    Ref("ExpressionSegment"),
                ),
                OneOf("ASC", "DESC", optional=True),
                # NB: This isn't really ANSI, and isn't supported in Mysql, but
                # is supported in enough other dialects for it to make sense here
                # for now.
                Sequence("NULLS", OneOf("FIRST", "LAST"), optional=True),
            ),
            terminator=OneOf(Ref.keyword("LIMIT"), Ref("FrameClauseUnitGrammar")),
        ),
        Dedent,
    )


@ansi_dialect.segment()
class GroupByClauseSegment(BaseSegment):
    """A `GROUP BY` clause like in `SELECT`."""

    type = "groupby_clause"
    match_grammar = StartsWith(
        Sequence("GROUP", "BY"),
        terminator=OneOf("ORDER", "LIMIT", "HAVING", "QUALIFY", "WINDOW"),
        enforce_whitespace_preceding_terminator=True,
    )
    parse_grammar = Sequence(
        "GROUP",
        "BY",
        Indent,
        Delimited(
            OneOf(
                Ref("ColumnReferenceSegment"),
                # Can `GROUP BY 1`
                Ref("NumericLiteralSegment"),
                # Can `GROUP BY coalesce(col, 1)`
                Ref("ExpressionSegment"),
            ),
            terminator=OneOf("ORDER", "LIMIT", "HAVING", "QUALIFY", "WINDOW"),
        ),
        Dedent,
    )


@ansi_dialect.segment()
class HavingClauseSegment(BaseSegment):
    """A `HAVING` clause like in `SELECT`."""

    type = "having_clause"
    match_grammar = StartsWith(
        "HAVING",
        terminator=OneOf("ORDER", "LIMIT", "QUALIFY", "WINDOW"),
        enforce_whitespace_preceding_terminator=True,
    )
    parse_grammar = Sequence(
        "HAVING",
        Indent,
        OptionallyBracketed(Ref("ExpressionSegment")),
        Dedent,
    )


@ansi_dialect.segment()
class LimitClauseSegment(BaseSegment):
    """A `LIMIT` clause like in `SELECT`."""

    type = "limit_clause"
    match_grammar = Sequence(
        "LIMIT",
        OneOf(
            Ref("NumericLiteralSegment"),
            Sequence(
                Ref("NumericLiteralSegment"), "OFFSET", Ref("NumericLiteralSegment")
            ),
            Sequence(
                Ref("NumericLiteralSegment"),
                Ref("CommaSegment"),
                Ref("NumericLiteralSegment"),
            ),
        ),
    )


@ansi_dialect.segment()
class OverlapsClauseSegment(BaseSegment):
    """An `OVERLAPS` clause like in `SELECT."""

    type = "overlaps_clause"
    match_grammar = StartsWith(
        "OVERLAPS",
    )
    parse_grammar = Sequence(
        "OVERLAPS",
        OneOf(
            Sequence(
                Bracketed(
                    Ref("DateTimeLiteralGrammar"),
                    Ref("CommaSegment"),
                    Ref("DateTimeLiteralGrammar"),
                )
            ),
            Ref("ColumnReferenceSegment"),
        ),
    )


@ansi_dialect.segment()
class NamedWindowSegment(BaseSegment):
    """A WINDOW clause."""

    type = "named_window"
    match_grammar = Sequence(
        "WINDOW",
        Delimited(
            Ref("NamedWindowExpressionSegment"),
        ),
    )


@ansi_dialect.segment()
class NamedWindowExpressionSegment(BaseSegment):
    """Named window expression."""

    type = "named_window_expression"
    match_grammar = Sequence(
        Ref("SingleIdentifierGrammar"),  # Window name
        "AS",
        Bracketed(
            Ref("WindowSpecificationSegment"),
        ),
    )


@ansi_dialect.segment()
class ValuesClauseSegment(BaseSegment):
    """A `VALUES` clause like in `INSERT`."""

    type = "values_clause"
    match_grammar = Sequence(
        OneOf("VALUE", "VALUES"),
        Delimited(
            Bracketed(
                Delimited(
                    Ref("LiteralGrammar"),
                    Ref("IntervalExpressionSegment"),
                    Ref("FunctionSegment"),
                    "DEFAULT",  # not in `FROM` clause, rule?
                    ephemeral_name="ValuesClauseElements",
                )
            ),
        ),
        Ref("AliasExpressionSegment", optional=True),
    )


@ansi_dialect.segment()
class UnorderedSelectStatementSegment(BaseSegment):
    """A `SELECT` statement without any ORDER clauses or later.

    This is designed for use in the context of set operations,
    for other use cases, we should use the main
    SelectStatementSegment.
    """

    type = "select_statement"
    # match grammar. This one makes sense in the context of knowing that it's
    # definitely a statement, we just don't know what type yet.
    match_grammar = StartsWith(
        # NB: In bigquery, the select clause may include an EXCEPT, which
        # will also match the set operator, but by starting with the whole
        # select clause rather than just the SELECT keyword, we mitigate that
        # here.
        Ref("SelectClauseSegment"),
        terminator=OneOf(
            Ref("SetOperatorSegment"),
            Ref("WithNoSchemaBindingClauseSegment"),
            Ref("OrderByClauseSegment"),
            Ref("LimitClauseSegment"),
            Ref("NamedWindowSegment"),
        ),
        enforce_whitespace_preceding_terminator=True,
    )

    parse_grammar = Sequence(
        Ref("SelectClauseSegment"),
        # Dedent for the indent in the select clause.
        # It's here so that it can come AFTER any whitespace.
        Dedent,
        Ref("FromClauseSegment", optional=True),
        Ref("WhereClauseSegment", optional=True),
        Ref("GroupByClauseSegment", optional=True),
        Ref("HavingClauseSegment", optional=True),
        Ref("OverlapsClauseSegment", optional=True),
    )


@ansi_dialect.segment()
class SelectStatementSegment(BaseSegment):
    """A `SELECT` statement."""

    type = "select_statement"
    # match grammar. This one makes sense in the context of knowing that it's
    # definitely a statement, we just don't know what type yet.
    match_grammar = StartsWith(
        # NB: In bigquery, the select clause may include an EXCEPT, which
        # will also match the set operator, but by starting with the whole
        # select clause rather than just the SELECT keyword, we mitigate that
        # here.
        Ref("SelectClauseSegment"),
        terminator=OneOf(
            Ref("SetOperatorSegment"), Ref("WithNoSchemaBindingClauseSegment")
        ),
        enforce_whitespace_preceding_terminator=True,
    )

    # Inherit most of the parse grammar from the original.
    parse_grammar = UnorderedSelectStatementSegment.parse_grammar.copy(
        insert=[
            Ref("OrderByClauseSegment", optional=True),
            Ref("LimitClauseSegment", optional=True),
            Ref("NamedWindowSegment", optional=True),
        ]
    )


ansi_dialect.add(
    # Things that behave like select statements
    SelectableGrammar=OneOf(
        Ref("WithCompoundStatementSegment"), Ref("NonWithSelectableGrammar")
    ),
    # Things that behave like select statements, which can form part of with expressions.
    NonWithSelectableGrammar=OneOf(
        Ref("SetExpressionSegment"),
        OptionallyBracketed(Ref("SelectStatementSegment")),
        Ref("NonSetSelectableGrammar"),
    ),
    # Things that behave like select statements, which can form part of set expressions.
    NonSetSelectableGrammar=OneOf(
        Ref("ValuesClauseSegment"),
        Ref("UnorderedSelectStatementSegment"),
        # If it's bracketed, we can have the full select statment here,
        # otherwise we can't because any order by clauses should belong
        # to the set expression.
        Bracketed(Ref("SelectStatementSegment")),
    ),
)


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


@ansi_dialect.segment()
class SetOperatorSegment(BaseSegment):
    """A set operator such as Union, Minus, Except or Intersect."""

    type = "set_operator"
    match_grammar = OneOf(
        Sequence("UNION", OneOf("DISTINCT", "ALL", optional=True)),
        "INTERSECT",
        "EXCEPT",
        "MINUS",
        exclude=Sequence("EXCEPT", Bracketed(Anything())),
    )


@ansi_dialect.segment()
class SetExpressionSegment(BaseSegment):
    """A set expression with either Union, Minus, Except or Intersect."""

    type = "set_expression"
    # match grammar
    match_grammar = Sequence(
        Ref("NonSetSelectableGrammar"),
        AnyNumberOf(
            Sequence(
                Ref("SetOperatorSegment"),
                Ref("NonSetSelectableGrammar"),
            ),
            min_times=1,
        ),
        Ref("OrderByClauseSegment", optional=True),
        Ref("LimitClauseSegment", optional=True),
        Ref("NamedWindowSegment", optional=True),
    )


@ansi_dialect.segment()
class InsertStatementSegment(BaseSegment):
    """A `INSERT` statement."""

    type = "insert_statement"
    match_grammar = StartsWith("INSERT")
    parse_grammar = Sequence(
        "INSERT",
        Ref.keyword("OVERWRITE", optional=True),  # Maybe this is just snowflake?
        Ref.keyword("INTO", optional=True),
        Ref("TableReferenceSegment"),
        Ref("BracketedColumnReferenceListGrammar", optional=True),
        Ref("SelectableGrammar"),
    )


@ansi_dialect.segment()
class TransactionStatementSegment(BaseSegment):
    """A `COMMIT`, `ROLLBACK` or `TRANSACTION` statement."""

    type = "transaction_statement"
    match_grammar = Sequence(
        # COMMIT [ WORK ] [ AND [ NO ] CHAIN ]
        # ROLLBACK [ WORK ] [ AND [ NO ] CHAIN ]
        # BEGIN | END TRANSACTION | WORK
        # NOTE: "TO SAVEPOINT" is not yet supported
        # https://docs.snowflake.com/en/sql-reference/sql/begin.html
        # https://www.postgresql.org/docs/current/sql-end.html
        OneOf("START", "BEGIN", "COMMIT", "ROLLBACK", "END"),
        OneOf("TRANSACTION", "WORK", optional=True),
        Sequence("NAME", Ref("SingleIdentifierGrammar"), optional=True),
        Sequence("AND", Ref.keyword("NO", optional=True), "CHAIN", optional=True),
    )


@ansi_dialect.segment()
class ColumnConstraintSegment(BaseSegment):
    """A column option; each CREATE TABLE column can have 0 or more."""

    type = "column_constraint_segment"
    # Column constraint from
    # https://www.postgresql.org/docs/12/sql-createtable.html
    match_grammar = Sequence(
        Sequence(
            "CONSTRAINT",
            Ref("ObjectReferenceSegment"),  # Constraint name
            optional=True,
        ),
        OneOf(
            Sequence(Ref.keyword("NOT", optional=True), "NULL"),  # NOT NULL or NULL
            Sequence("CHECK", Bracketed(Ref("ExpressionSegment"))),
            Sequence(  # DEFAULT <value>
                "DEFAULT",
                OneOf(
                    Ref("LiteralGrammar"),
                    Ref("FunctionSegment"),
                    # ?? Ref('IntervalExpressionSegment')
                ),
            ),
            Ref("PrimaryKeyGrammar"),
            "UNIQUE",  # UNIQUE
            "AUTO_INCREMENT",  # AUTO_INCREMENT (MySQL)
            "UNSIGNED",  # UNSIGNED (MySQL)
            Sequence(  # REFERENCES reftable [ ( refcolumn) ]
                "REFERENCES",
                Ref("ColumnReferenceSegment"),
                # Foreign columns making up FOREIGN KEY constraint
                Ref("BracketedColumnReferenceListGrammar", optional=True),
            ),
            Ref("CommentClauseSegment"),
        ),
    )


@ansi_dialect.segment()
class ColumnDefinitionSegment(BaseSegment):
    """A column definition, e.g. for CREATE TABLE or ALTER TABLE."""

    type = "column_definition"
    match_grammar = Sequence(
        Ref("SingleIdentifierGrammar"),  # Column name
        Ref("DatatypeSegment"),  # Column type
        Bracketed(Anything(), optional=True),  # For types like VARCHAR(100)
        AnyNumberOf(
            Ref("ColumnConstraintSegment", optional=True),
        ),
    )


@ansi_dialect.segment()
class IndexColumnDefinitionSegment(BaseSegment):
    """A column definition for CREATE INDEX."""

    type = "index_column_definition"
    match_grammar = Sequence(
        Ref("SingleIdentifierGrammar"),  # Column name
        OneOf("ASC", "DESC", optional=True),
    )


@ansi_dialect.segment()
class TableConstraintSegment(BaseSegment):
    """A table constraint, e.g. for CREATE TABLE."""

    type = "table_constraint_segment"
    # Later add support for CHECK constraint, others?
    # e.g. CONSTRAINT constraint_1 PRIMARY KEY(column_1)
    match_grammar = Sequence(
        Sequence(  # [ CONSTRAINT <Constraint name> ]
            "CONSTRAINT", Ref("ObjectReferenceSegment"), optional=True
        ),
        OneOf(
            Sequence(  # UNIQUE ( column_name [, ... ] )
                "UNIQUE",
                Ref("BracketedColumnReferenceListGrammar"),
                # Later add support for index_parameters?
            ),
            Sequence(  # PRIMARY KEY ( column_name [, ... ] ) index_parameters
                Ref("PrimaryKeyGrammar"),
                # Columns making up PRIMARY KEY constraint
                Ref("BracketedColumnReferenceListGrammar"),
                # Later add support for index_parameters?
            ),
            Sequence(  # FOREIGN KEY ( column_name [, ... ] )
                # REFERENCES reftable [ ( refcolumn [, ... ] ) ]
                Ref("ForeignKeyGrammar"),
                # Local columns making up FOREIGN KEY constraint
                Ref("BracketedColumnReferenceListGrammar"),
                "REFERENCES",
                Ref("ColumnReferenceSegment"),
                # Foreign columns making up FOREIGN KEY constraint
                Ref("BracketedColumnReferenceListGrammar"),
                # Later add support for [MATCH FULL/PARTIAL/SIMPLE] ?
                # Later add support for [ ON DELETE/UPDATE action ] ?
            ),
        ),
    )


@ansi_dialect.segment()
class TableEndClauseSegment(BaseSegment):
    """Allow for additional table endings.

    (like WITHOUT ROWID for SQLite)
    """

    type = "table_end_clause_segment"
    match_grammar = Nothing()


@ansi_dialect.segment()
class CreateTableStatementSegment(BaseSegment):
    """A `CREATE TABLE` statement."""

    type = "create_table_statement"
    # https://crate.io/docs/sql-99/en/latest/chapters/18.html
    # https://www.postgresql.org/docs/12/sql-createtable.html
    match_grammar = Sequence(
        "CREATE",
        Ref("OrReplaceGrammar", optional=True),
        Ref("TemporaryTransientGrammar", optional=True),
        "TABLE",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("TableReferenceSegment"),
        OneOf(
            # Columns and comment syntax:
            Sequence(
                Bracketed(
                    Delimited(
                        OneOf(
                            Ref("TableConstraintSegment"),
                            Ref("ColumnDefinitionSegment"),
                        ),
                    )
                ),
                Ref("CommentClauseSegment", optional=True),
            ),
            # Create AS syntax:
            Sequence(
                "AS",
                OptionallyBracketed(Ref("SelectableGrammar")),
            ),
            # Create like syntax
            Sequence("LIKE", Ref("TableReferenceSegment")),
        ),
        Ref("TableEndClauseSegment", optional=True),
    )


@ansi_dialect.segment()
class CommentClauseSegment(BaseSegment):
    """A comment clause.

    e.g. COMMENT 'view/table/column description'
    """

    type = "comment_clause"
    match_grammar = Sequence("COMMENT", Ref("QuotedLiteralSegment"))


@ansi_dialect.segment()
class CreateSchemaStatementSegment(BaseSegment):
    """A `CREATE SCHEMA` statement."""

    type = "create_schema_statement"
    match_grammar = Sequence(
        "CREATE",
        "SCHEMA",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("SchemaReferenceSegment"),
    )


@ansi_dialect.segment()
class SetSchemaStatementSegment(BaseSegment):
    """A `SET SCHEMA` statement."""

    type = "set_schema_statement"
    match_grammar = Sequence(
        "SET",
        "SCHEMA",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("SchemaReferenceSegment"),
    )


@ansi_dialect.segment()
class DropSchemaStatementSegment(BaseSegment):
    """A `DROP SCHEMA` statement."""

    type = "drop_schema_statement"
    match_grammar = Sequence(
        "DROP",
        "SCHEMA",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("SchemaReferenceSegment"),
    )


@ansi_dialect.segment()
class CreateDatabaseStatementSegment(BaseSegment):
    """A `CREATE DATABASE` statement."""

    type = "create_database_statement"
    match_grammar = Sequence(
        "CREATE",
        "DATABASE",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("DatabaseReferenceSegment"),
    )


@ansi_dialect.segment()
class CreateExtensionStatementSegment(BaseSegment):
    """A `CREATE EXTENSION` statement.

    https://www.postgresql.org/docs/9.1/sql-createextension.html
    """

    type = "create_extension_statement"
    match_grammar = Sequence(
        "CREATE",
        "EXTENSION",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("ExtensionReferenceSegment"),
        Ref.keyword("WITH", optional=True),
        Sequence("SCHEMA", Ref("SchemaReferenceSegment"), optional=True),
        Sequence("VERSION", Ref("VersionIdentifierSegment"), optional=True),
        Sequence("FROM", Ref("VersionIdentifierSegment"), optional=True),
    )


@ansi_dialect.segment()
class CreateIndexStatementSegment(BaseSegment):
    """A `CREATE INDEX` statement."""

    type = "create_index_statement"
    match_grammar = Sequence(
        "CREATE",
        Ref("OrReplaceGrammar", optional=True),
        "INDEX",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("IndexReferenceSegment"),
        "ON",
        Ref("TableReferenceSegment"),
        Sequence(
            Bracketed(
                Delimited(
                    Ref("IndexColumnDefinitionSegment"),
                ),
            )
        ),
    )


@ansi_dialect.segment()
class AlterTableStatementSegment(BaseSegment):
    """An `ALTER TABLE` statement."""

    type = "alter_table_statement"
    # Based loosely on:
    # https://dev.mysql.com/doc/refman/8.0/en/alter-table.html
    # TODO: Flesh this out with more detail.
    match_grammar = Sequence(
        "ALTER",
        "TABLE",
        Ref("TableReferenceSegment"),
        Delimited(
            OneOf(
                # Table options
                Sequence(
                    Ref("ParameterNameSegment"),
                    Ref("EqualsSegment", optional=True),
                    OneOf(Ref("LiteralGrammar"), Ref("NakedIdentifierSegment")),
                ),
                # Add things
                Sequence(
                    OneOf("ADD", "MODIFY"),
                    Ref.keyword("COLUMN", optional=True),
                    Ref("ColumnDefinitionSegment"),
                    OneOf(
                        Sequence(
                            OneOf("FIRST", "AFTER"), Ref("ColumnReferenceSegment")
                        ),
                        # Bracketed Version of the same
                        Ref("BracketedColumnReferenceListGrammar"),
                        optional=True,
                    ),
                ),
                # Rename
                Sequence(
                    "RENAME",
                    OneOf("AS", "TO", optional=True),
                    Ref("TableReferenceSegment"),
                ),
            ),
        ),
    )


@ansi_dialect.segment()
class CreateViewStatementSegment(BaseSegment):
    """A `CREATE VIEW` statement."""

    type = "create_view_statement"
    # https://crate.io/docs/sql-99/en/latest/chapters/18.html#create-view-statement
    # https://dev.mysql.com/doc/refman/8.0/en/create-view.html
    # https://www.postgresql.org/docs/12/sql-createview.html
    match_grammar = Sequence(
        "CREATE",
        Ref("OrReplaceGrammar", optional=True),
        "VIEW",
        Ref("TableReferenceSegment"),
        # Optional list of column names
        Ref("BracketedColumnReferenceListGrammar", optional=True),
        "AS",
        Ref("SelectableGrammar"),
        Ref("WithNoSchemaBindingClauseSegment", optional=True),
    )


@ansi_dialect.segment()
class DropStatementSegment(BaseSegment):
    """A `DROP` statement."""

    type = "drop_statement"
    # DROP {TABLE | VIEW} <Table name> [IF EXISTS} {RESTRICT | CASCADE}
    match_grammar = Sequence(
        "DROP",
        OneOf(
            "TABLE",
            "VIEW",
            "USER",
        ),
        Ref("IfExistsGrammar", optional=True),
        Ref("TableReferenceSegment"),
        OneOf("RESTRICT", Ref.keyword("CASCADE", optional=True), optional=True),
    )


@ansi_dialect.segment()
class TruncateStatementSegment(BaseSegment):
    """`TRUNCATE TABLE` statement."""

    type = "truncate_table"

    match_grammar = Sequence(
        "TRUNCATE",
        Ref.keyword("TABLE", optional=True),
        Ref("TableReferenceSegment"),
    )


@ansi_dialect.segment()
class DropIndexStatementSegment(BaseSegment):
    """A `DROP INDEX` statement."""

    type = "drop_statement"
    # DROP INDEX <Index name> [CONCURRENTLY] [IF EXISTS] {RESTRICT | CASCADE}
    match_grammar = Sequence(
        "DROP",
        "INDEX",
        Ref.keyword("CONCURRENTLY", optional=True),
        Ref("IfExistsGrammar", optional=True),
        Ref("IndexReferenceSegment"),
        OneOf("RESTRICT", Ref.keyword("CASCADE", optional=True), optional=True),
    )


@ansi_dialect.segment()
class AccessStatementSegment(BaseSegment):
    """A `GRANT` or `REVOKE` statement.

    In order to help reduce code duplication we decided to implement other dialect specific grants (like Snowflake)
    here too which will help with maintainability. We also note that this causes the grammar to be less "correct",
    but the benefits outweigh the con in our opinion.


    Grant specific information:
     * https://www.postgresql.org/docs/9.0/sql-grant.html
     * https://docs.snowflake.com/en/sql-reference/sql/grant-privilege.html

    Revoke specific information:
     * https://www.postgresql.org/docs/9.0/sql-revoke.html
     * https://docs.snowflake.com/en/sql-reference/sql/revoke-role.html
     * https://docs.snowflake.com/en/sql-reference/sql/revoke-privilege.html
     * https://docs.snowflake.com/en/sql-reference/sql/revoke-privilege-share.html
    """

    type = "access_statement"

    # Privileges that can be set on the account (specific to snowflake)
    _global_permissions = OneOf(
        Sequence(
            "CREATE",
            OneOf(
                "ROLE",
                "USER",
                "WAREHOUSE",
                "DATABASE",
                "INTEGRATION",
            ),
        ),
        Sequence("APPLY", "MASKING", "POLICY"),
        Sequence("EXECUTE", "TASK"),
        Sequence("MANAGE", "GRANTS"),
        Sequence("MONITOR", OneOf("EXECUTION", "USAGE")),
    )

    _schema_object_names = [
        "TABLE",
        "VIEW",
        "STAGE",
        "FUNCTION",
        "PROCEDURE",
        "ROUTINE",
        "SEQUENCE",
        "STREAM",
        "TASK",
    ]

    _schema_object_types = OneOf(
        *_schema_object_names,
        Sequence("MATERIALIZED", "VIEW"),
        Sequence("EXTERNAL", "TABLE"),
        Sequence("FILE", "FORMAT"),
    )

    # We reuse the object names above and simply append an `S` to the end of them to get plurals
    _schema_object_types_plural = OneOf(
        *[f"{object_name}S" for object_name in _schema_object_names]
    )

    _permissions = Sequence(
        OneOf(
            Sequence(
                "CREATE",
                OneOf(
                    "SCHEMA",
                    Sequence("MASKING", "POLICY"),
                    "PIPE",
                    _schema_object_types,
                ),
            ),
            Sequence("IMPORTED", "PRIVILEGES"),
            "APPLY",
            "CONNECT",
            "CREATE",
            "DELETE",
            "EXECUTE",
            "INSERT",
            "MODIFY",
            "MONITOR",
            "OPERATE",
            "OWNERSHIP",
            "READ",
            "REFERENCE_USAGE",
            "REFERENCES",
            "SELECT",
            "TEMP",
            "TEMPORARY",
            "TRIGGER",
            "TRUNCATE",
            "UPDATE",
            "USAGE",
            "USE_ANY_ROLE",
            "WRITE",
            Sequence("ALL", Ref.keyword("PRIVILEGES", optional=True)),
        ),
        Ref("BracketedColumnReferenceListGrammar", optional=True),
    )

    # All of the object types that we can grant permissions on.
    # This list will contain ansi sql objects as well as dialect specific ones.
    _objects = OneOf(
        "ACCOUNT",
        Sequence(
            OneOf(
                Sequence("RESOURCE", "MONITOR"),
                "WAREHOUSE",
                "DATABASE",
                "DOMAIN",
                "INTEGRATION",
                "LANGUAGE",
                "SCHEMA",
                "ROLE",
                "TABLESPACE",
                "TYPE",
                Sequence(
                    "FOREIGN",
                    OneOf("SERVER", Sequence("DATA", "WRAPPER")),
                ),
                Sequence("ALL", "SCHEMAS", "IN", "DATABASE"),
                Sequence("FUTURE", "SCHEMAS", "IN", "DATABASE"),
                _schema_object_types,
                Sequence("ALL", _schema_object_types_plural, "IN", "SCHEMA"),
                Sequence(
                    "FUTURE",
                    _schema_object_types_plural,
                    "IN",
                    OneOf("DATABASE", "SCHEMA"),
                ),
                optional=True,
            ),
            Ref("ObjectReferenceSegment"),
            Ref("FunctionParameterListGrammar", optional=True),
        ),
        Sequence("LARGE", "OBJECT", Ref("NumericLiteralSegment")),
    )

    match_grammar = OneOf(
        # Based on https://www.postgresql.org/docs/13/sql-grant.html
        # and https://docs.snowflake.com/en/sql-reference/sql/grant-privilege.html
        Sequence(
            "GRANT",
            OneOf(
                Sequence(
                    Delimited(
                        OneOf(_global_permissions, _permissions),
                        delimiter=Ref("CommaSegment"),
                        terminator="ON",
                    ),
                    "ON",
                    _objects,
                ),
                Sequence("ROLE", Ref("ObjectReferenceSegment")),
                Sequence("OWNERSHIP", "ON", "USER", Ref("ObjectReferenceSegment")),
                # In the case where a role is granted non-explicitly,
                # e.g. GRANT ROLE_NAME TO OTHER_ROLE_NAME
                # See https://www.postgresql.org/docs/current/sql-grant.html
                Ref("ObjectReferenceSegment"),
            ),
            "TO",
            OneOf("GROUP", "USER", "ROLE", "SHARE", optional=True),
            Delimited(
                OneOf(Ref("ObjectReferenceSegment"), Ref("FunctionSegment"), "PUBLIC"),
                delimiter=Ref("CommaSegment"),
            ),
            OneOf(
                Sequence("WITH", "GRANT", "OPTION"),
                Sequence("WITH", "ADMIN", "OPTION"),
                Sequence("COPY", "CURRENT", "GRANTS"),
                optional=True,
            ),
            Sequence(
                "GRANTED",
                "BY",
                OneOf(
                    "CURRENT_USER",
                    "SESSION_USER",
                    Ref("ObjectReferenceSegment"),
                ),
                optional=True,
            ),
        ),
        # Based on https://www.postgresql.org/docs/12/sql-revoke.html
        Sequence(
            "REVOKE",
            Sequence("GRANT", "OPTION", "FOR", optional=True),
            OneOf(
                Sequence(
                    Delimited(
                        OneOf(_global_permissions, _permissions),
                        delimiter=Ref("CommaSegment"),
                        terminator="ON",
                    ),
                    "ON",
                    _objects,
                ),
                Sequence("ROLE", Ref("ObjectReferenceSegment")),
                Sequence("OWNERSHIP", "ON", "USER", Ref("ObjectReferenceSegment")),
            ),
            "FROM",
            OneOf("GROUP", "USER", "ROLE", "SHARE", optional=True),
            Delimited(
                Ref("ObjectReferenceSegment"),
                delimiter=Ref("CommaSegment"),
            ),
            OneOf("RESTRICT", Ref.keyword("CASCADE", optional=True), optional=True),
        ),
    )


@ansi_dialect.segment()
class DeleteStatementSegment(BaseSegment):
    """A `DELETE` statement.

    DELETE FROM <table name> [ WHERE <search condition> ]
    """

    type = "delete_statement"
    # match grammar. This one makes sense in the context of knowing that it's
    # definitely a statement, we just don't know what type yet.
    match_grammar = StartsWith("DELETE")
    parse_grammar = Sequence(
        "DELETE",
        Ref("FromClauseSegment"),
        Ref("WhereClauseSegment", optional=True),
    )


@ansi_dialect.segment()
class UpdateStatementSegment(BaseSegment):
    """A `Update` statement.

    UPDATE <table name> SET <set clause list> [ WHERE <search condition> ]
    """

    type = "update_statement"
    match_grammar = StartsWith("UPDATE")
    parse_grammar = Sequence(
        "UPDATE",
        OneOf(Ref("TableReferenceSegment"), Ref("AliasedTableReferenceGrammar")),
        Ref("SetClauseListSegment"),
        Ref("FromClauseSegment", optional=True),
        Ref("WhereClauseSegment", optional=True),
    )


@ansi_dialect.segment()
class SetClauseListSegment(BaseSegment):
    """SQL 1992 set clause list.

    <set clause list> ::=
              <set clause> [ { <comma> <set clause> }... ]

         <set clause> ::=
              <object column> <equals operator> <update source>

         <update source> ::=
                <value expression>
              | <null specification>
              | DEFAULT

         <object column> ::= <column name>
    """

    type = "set_clause_list"
    match_grammar = Sequence(
        "SET",
        Indent,
        OneOf(
            Ref("SetClauseSegment"),
            # set clause
            AnyNumberOf(
                Delimited(Ref("SetClauseSegment")),
            ),
        ),
        Dedent,
    )


@ansi_dialect.segment()
class SetClauseSegment(BaseSegment):
    """SQL 1992 set clause.

    <set clause> ::=
              <object column> <equals operator> <update source>

         <update source> ::=
                <value expression>
              | <null specification>
              | DEFAULT

         <object column> ::= <column name>
    """

    type = "set_clause"

    match_grammar = Sequence(
        Ref("ColumnReferenceSegment"),
        Ref("EqualsSegment"),
        OneOf(
            Ref("LiteralGrammar"),
            Ref("BareFunctionSegment"),
            Ref("FunctionSegment"),
            Ref("ColumnReferenceSegment"),
            "DEFAULT",
        ),
    )


@ansi_dialect.segment()
class FunctionDefinitionGrammar(BaseSegment):
    """This is the body of a `CREATE FUNCTION AS` statement."""

    match_grammar = Sequence(
        "AS",
        Ref("QuotedLiteralSegment"),
        Sequence(
            "LANGUAGE",
            # Not really a parameter, but best fit for now.
            Ref("ParameterNameSegment"),
            optional=True,
        ),
    )


@ansi_dialect.segment()
class CreateFunctionStatementSegment(BaseSegment):
    """A `CREATE FUNCTION` statement.

    This version in the ANSI dialect should be a "common subset" of the
    structure of the code for those dialects.
    postgres: https://www.postgresql.org/docs/9.1/sql-createfunction.html
    snowflake: https://docs.snowflake.com/en/sql-reference/sql/create-function.html
    bigquery: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
    """

    type = "create_function_statement"

    match_grammar = Sequence(
        "CREATE",
        Sequence("OR", "REPLACE", optional=True),
        Ref("TemporaryGrammar", optional=True),
        "FUNCTION",
        Anything(),
    )

    parse_grammar = Sequence(
        "CREATE",
        Sequence("OR", "REPLACE", optional=True),
        Ref("TemporaryGrammar", optional=True),
        "FUNCTION",
        Sequence("IF", "NOT", "EXISTS", optional=True),
        Ref("FunctionNameSegment"),
        Ref("FunctionParameterListGrammar"),
        Sequence(  # Optional function return type
            "RETURNS",
            Ref("DatatypeSegment"),
            optional=True,
        ),
        Ref("FunctionDefinitionGrammar"),
    )


@ansi_dialect.segment()
class FunctionParameterListGrammar(BaseSegment):
    """The parameters for a function ie. `(string, number)`."""

    # Function parameter list
    match_grammar = Bracketed(
        Delimited(
            Ref("FunctionParameterGrammar"),
            delimiter=Ref("CommaSegment"),
            optional=True,
        ),
    )


@ansi_dialect.segment()
class CreateModelStatementSegment(BaseSegment):
    """A BigQuery `CREATE MODEL` statement."""

    type = "create_model_statement"
    # https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create
    match_grammar = Sequence(
        "CREATE",
        Ref("OrReplaceGrammar", optional=True),
        "MODEL",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("ObjectReferenceSegment"),
        Sequence(
            "OPTIONS",
            Bracketed(
                Delimited(
                    Sequence(
                        Ref("ParameterNameSegment"),
                        Ref("EqualsSegment"),
                        OneOf(
                            # This covers many but not all the extensive list of
                            # possible 'CREATE MODEL' options.
                            Ref("LiteralGrammar"),  # Single value
                            Bracketed(
                                # E.g. input_label_cols: list of column names
                                Delimited(Ref("QuotedLiteralSegment")),
                                bracket_type="square",
                                optional=True,
                            ),
                        ),
                    ),
                )
            ),
            optional=True,
        ),
        "AS",
        Ref("SelectableGrammar"),
    )


@ansi_dialect.segment()
class CreateTypeStatementSegment(BaseSegment):
    """A `CREATE TYPE` statement.

    This is based around the Postgres syntax.
    https://www.postgresql.org/docs/current/sql-createtype.html

    Note: This is relatively permissive currently
    and does not lint the syntax strictly, to allow
    for some deviation between dialects.
    """

    type = "create_type_statement"
    match_grammar = Sequence(
        "CREATE",
        "TYPE",
        Ref("ObjectReferenceSegment"),
        Sequence("AS", OneOf("ENUM", "RANGE", optional=True), optional=True),
        Bracketed(Delimited(Anything()), optional=True),
    )


@ansi_dialect.segment()
class CreateRoleStatementSegment(BaseSegment):
    """A `CREATE ROLE` statement.

    A very simple create role syntax which can be extended
    by other dialects.
    """

    type = "create_role_statement"
    match_grammar = Sequence(
        "CREATE",
        "ROLE",
        Ref("ObjectReferenceSegment"),
    )


@ansi_dialect.segment()
class DropModelStatementSegment(BaseSegment):
    """A `DROP MODEL` statement."""

    type = "drop_MODELstatement"
    # DROP MODEL <Model name> [IF EXISTS}
    # https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-drop-model
    match_grammar = Sequence(
        "DROP",
        "MODEL",
        Ref("IfExistsGrammar", optional=True),
        Ref("ObjectReferenceSegment"),
    )


@ansi_dialect.segment()
class MLTableExpressionSegment(BaseSegment):
    """An ML table expression."""

    type = "ml_table_expression"
    # E.g. ML.WEIGHTS(MODEL `project.dataset.model`)
    match_grammar = Sequence(
        "ML",
        Ref("DotSegment"),
        Ref("SingleIdentifierGrammar"),
        Bracketed(
            Sequence("MODEL", Ref("ObjectReferenceSegment")),
            OneOf(
                Sequence(
                    Ref("CommaSegment"),
                    Bracketed(
                        Ref("SelectableGrammar"),
                    ),
                ),
                optional=True,
            ),
        ),
    )


@ansi_dialect.segment()
class StatementSegment(BaseSegment):
    """A generic segment, to any of its child subsegments."""

    type = "statement"
    match_grammar = GreedyUntil(Ref("DelimiterSegment"))

    parse_grammar = OneOf(
        Ref("SelectableGrammar"),
        Ref("InsertStatementSegment"),
        Ref("TransactionStatementSegment"),
        Ref("DropStatementSegment"),
        Ref("TruncateStatementSegment"),
        Ref("AccessStatementSegment"),
        Ref("CreateTableStatementSegment"),
        Ref("CreateTypeStatementSegment"),
        Ref("CreateRoleStatementSegment"),
        Ref("AlterTableStatementSegment"),
        Ref("CreateSchemaStatementSegment"),
        Ref("SetSchemaStatementSegment"),
        Ref("DropSchemaStatementSegment"),
        Ref("CreateDatabaseStatementSegment"),
        Ref("CreateExtensionStatementSegment"),
        Ref("CreateIndexStatementSegment"),
        Ref("DropIndexStatementSegment"),
        Ref("CreateViewStatementSegment"),
        Ref("DeleteStatementSegment"),
        Ref("UpdateStatementSegment"),
        Ref("CreateFunctionStatementSegment"),
        Ref("CreateModelStatementSegment"),
        Ref("DropModelStatementSegment"),
        Ref("DescribeStatementSegment"),
        Ref("UseStatementSegment"),
        Ref("ExplainStatementSegment"),
        Ref("CreateSequenceStatementSegment"),
        Ref("AlterSequenceStatementSegment"),
        Ref("DropSequenceStatementSegment"),
    )

    def get_table_references(self):
        """Use parsed tree to extract table references."""
        table_refs = {
            tbl_ref.raw for tbl_ref in self.recursive_crawl("table_reference")
        }
        cte_refs = {
            cte_def.get_identifier().raw
            for cte_def in self.recursive_crawl("common_table_expression")
        }
        # External references are any table references which aren't
        # also cte aliases.
        return table_refs - cte_refs


@ansi_dialect.segment()
class WithNoSchemaBindingClauseSegment(BaseSegment):
    """WITH NO SCHEMA BINDING clause for Redshift's Late Binding Views.

    https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_VIEW.html
    """

    type = "with_no_schema_binding_clause"
    match_grammar = Sequence(
        "WITH",
        "NO",
        "SCHEMA",
        "BINDING",
    )


@ansi_dialect.segment()
class DescribeStatementSegment(BaseSegment):
    """A `Describe` statement.

    DESCRIBE <object type> <object name>
    """

    type = "describe_statement"
    match_grammar = StartsWith("DESCRIBE")

    parse_grammar = Sequence(
        "DESCRIBE",
        Ref("NakedIdentifierSegment"),
        Ref("ObjectReferenceSegment"),
    )


@ansi_dialect.segment()
class UseStatementSegment(BaseSegment):
    """A `USE` statement.

    USE [ ROLE ] <name>

    USE [ WAREHOUSE ] <name>

    USE [ DATABASE ] <name>

    USE [ SCHEMA ] [<db_name>.]<name>
    """

    type = "use_statement"
    match_grammar = StartsWith("USE")

    parse_grammar = Sequence(
        "USE",
        OneOf("ROLE", "WAREHOUSE", "DATABASE", "SCHEMA", optional=True),
        Ref("ObjectReferenceSegment"),
    )


@ansi_dialect.segment()
class ExplainStatementSegment(BaseSegment):
    """An `Explain` statement.

    EXPLAIN explainable_stmt
    """

    type = "explain_statement"

    explainable_stmt = OneOf(
        Ref("SelectableGrammar"),
        Ref("InsertStatementSegment"),
        Ref("UpdateStatementSegment"),
        Ref("DeleteStatementSegment"),
    )

    match_grammar = StartsWith("EXPLAIN")

    parse_grammar = Sequence(
        "EXPLAIN",
        explainable_stmt,
    )


@ansi_dialect.segment()
class CreateSequenceOptionsSegment(BaseSegment):
    """Options for Create Sequence statement.

    As specified in https://docs.oracle.com/cd/B19306_01/server.102/b14200/statements_6015.htm
    """

    type = "create_sequence_options_segment"

    match_grammar = OneOf(
        Sequence("INCREMENT", "BY", Ref("NumericLiteralSegment")),
        Sequence(
            "START", Ref.keyword("WITH", optional=True), Ref("NumericLiteralSegment")
        ),
        OneOf(
            Sequence("MINVALUE", Ref("NumericLiteralSegment")),
            Sequence("NO", "MINVALUE"),
        ),
        OneOf(
            Sequence("MAXVALUE", Ref("NumericLiteralSegment")),
            Sequence("NO", "MAXVALUE"),
        ),
        OneOf(Sequence("CACHE", Ref("NumericLiteralSegment")), "NOCACHE"),
        OneOf("CYCLE", "NOCYCLE"),
        OneOf("ORDER", "NOORDER"),
    )


@ansi_dialect.segment()
class CreateSequenceStatementSegment(BaseSegment):
    """Create Sequence statement.

    As specified in https://docs.oracle.com/cd/B19306_01/server.102/b14200/statements_6015.htm
    """

    type = "create_sequence_statement"

    match_grammar = Sequence(
        "CREATE",
        "SEQUENCE",
        Ref("SequenceReferenceSegment"),
        AnyNumberOf(Ref("CreateSequenceOptionsSegment"), optional=True),
    )


@ansi_dialect.segment()
class AlterSequenceOptionsSegment(BaseSegment):
    """Options for Alter Sequence statement.

    As specified in https://docs.oracle.com/cd/B19306_01/server.102/b14200/statements_2011.htm
    """

    type = "alter_sequence_options_segment"

    match_grammar = OneOf(
        Sequence("INCREMENT", "BY", Ref("NumericLiteralSegment")),
        OneOf(
            Sequence("MINVALUE", Ref("NumericLiteralSegment")),
            Sequence("NO", "MINVALUE"),
        ),
        OneOf(
            Sequence("MAXVALUE", Ref("NumericLiteralSegment")),
            Sequence("NO", "MAXVALUE"),
        ),
        OneOf(Sequence("CACHE", Ref("NumericLiteralSegment")), "NOCACHE"),
        OneOf("CYCLE", "NOCYCLE"),
        OneOf("ORDER", "NOORDER"),
    )


@ansi_dialect.segment()
class AlterSequenceStatementSegment(BaseSegment):
    """Alter Sequence Statement.

    As specified in https://docs.oracle.com/cd/B19306_01/server.102/b14200/statements_2011.htm
    """

    type = "alter_sequence_statement"

    match_grammar = Sequence(
        "ALTER",
        "SEQUENCE",
        Ref("SequenceReferenceSegment"),
        AnyNumberOf(Ref("AlterSequenceOptionsSegment")),
    )


@ansi_dialect.segment()
class DropSequenceStatementSegment(BaseSegment):
    """Drop Sequence Statement.

    As specified in https://docs.oracle.com/cd/E11882_01/server.112/e41084/statements_9001.htm
    """

    type = "drop_sequence_statement"

    match_grammar = Sequence("DROP", "SEQUENCE", Ref("SequenceReferenceSegment"))


@ansi_dialect.segment()
class DateAddFunctionNameSegment(BaseSegment):
    """DATEADD function name segment.

    Need to be able to specify this as type function_name
    so that linting rules identify it properly
    """

    type = "function_name"
    match_grammar = Sequence("DATEADD")

```
### 3 - src/sqlfluff/dialects/dialect_tsql.py:

```python
"""The MSSQL T-SQL dialect.

https://docs.microsoft.com/en-us/sql/t-sql/language-elements/language-elements-transact-sql
"""

from sqlfluff.core.parser import (
    BaseSegment,
    Sequence,
    OneOf,
    Bracketed,
    Ref,
    Anything,
    Nothing,
    RegexLexer,
    CodeSegment,
    RegexParser,
    Delimited,
    Matchable,
    NamedParser,
    StartsWith,
    OptionallyBracketed,
    Dedent,
    AnyNumberOf,
)

from sqlfluff.core.dialects import load_raw_dialect
from sqlfluff.dialects.tsql_keywords import RESERVED_KEYWORDS, UNRESERVED_KEYWORDS

ansi_dialect = load_raw_dialect("ansi")
tsql_dialect = ansi_dialect.copy_as("tsql")

# Should really clear down the old keywords but some are needed by certain segments
# tsql_dialect.sets("reserved_keywords").clear()
# tsql_dialect.sets("unreserved_keywords").clear()
tsql_dialect.sets("reserved_keywords").update(RESERVED_KEYWORDS)
tsql_dialect.sets("unreserved_keywords").update(UNRESERVED_KEYWORDS)

tsql_dialect.insert_lexer_matchers(
    [
        RegexLexer(
            "atsign",
            r"[@][a-zA-Z0-9_]+",
            CodeSegment,
        ),
        RegexLexer(
            "square_quote",
            r"\[([a-zA-Z0-9][^\[\]]*)*\]",
            CodeSegment,
        ),
        # T-SQL unicode strings
        RegexLexer("single_quote_with_n", r"N'([^'\\]|\\.)*'", CodeSegment),
    ],
    before="back_quote",
)

tsql_dialect.add(
    BracketedIdentifierSegment=NamedParser(
        "square_quote", CodeSegment, name="quoted_identifier", type="identifier"
    ),
    QuotedLiteralSegmentWithN=NamedParser(
        "single_quote_with_n", CodeSegment, name="quoted_literal", type="literal"
    ),
)

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


@tsql_dialect.segment(replace=True)
class StatementSegment(ansi_dialect.get_segment("StatementSegment")):  # type: ignore
    """Overriding StatementSegment to allow for additional segment parsing."""

    parse_grammar = ansi_dialect.get_segment("StatementSegment").parse_grammar.copy(
        insert=[
            Ref("CreateProcedureStatementSegment"),
            Ref("IfExpressionStatement"),
            Ref("DeclareStatementSegment"),
            Ref("SetStatementSegment"),
            Ref("AlterTableSwitchStatementSegment"),
            Ref(
                "CreateTableAsSelectStatementSegment"
            ),  # Azure Synapse Analytics specific
        ],
    )


@tsql_dialect.segment(replace=True)
class SelectClauseModifierSegment(BaseSegment):
    """Things that come after SELECT but before the columns."""

    type = "select_clause_modifier"
    match_grammar = OneOf(
        "DISTINCT",
        "ALL",
        Sequence(
            "TOP",
            OptionallyBracketed(Ref("ExpressionSegment")),
            Sequence("PERCENT", optional=True),
            Sequence("WITH", "TIES", optional=True),
        ),
    )


@tsql_dialect.segment(replace=True)
class UnorderedSelectStatementSegment(BaseSegment):
    """A `SELECT` statement without any ORDER clauses or later.

    We need to change ANSI slightly to remove LimitClauseSegment
    and NamedWindowSegment which don't exist in T-SQL.
    """

    type = "select_statement"
    # match grammar. This one makes sense in the context of knowing that it's
    # definitely a statement, we just don't know what type yet.
    match_grammar = StartsWith(
        # NB: In bigquery, the select clause may include an EXCEPT, which
        # will also match the set operator, but by starting with the whole
        # select clause rather than just the SELECT keyword, we mitigate that
        # here.
        Ref("SelectClauseSegment"),
        terminator=OneOf(
            Ref("SetOperatorSegment"),
            Ref("WithNoSchemaBindingClauseSegment"),
            Ref("OrderByClauseSegment"),
        ),
        enforce_whitespace_preceding_terminator=True,
    )

    parse_grammar = Sequence(
        Ref("SelectClauseSegment"),
        # Dedent for the indent in the select clause.
        # It's here so that it can come AFTER any whitespace.
        Dedent,
        Ref("FromClauseSegment", optional=True),
        Ref("PivotUnpivotStatementSegment", optional=True),
        Ref("WhereClauseSegment", optional=True),
        Ref("GroupByClauseSegment", optional=True),
        Ref("HavingClauseSegment", optional=True),
    )


@tsql_dialect.segment(replace=True)
class SelectStatementSegment(BaseSegment):
    """A `SELECT` statement.

    We need to change ANSI slightly to remove LimitClauseSegment
    and NamedWindowSegment which don't exist in T-SQL.
    """

    type = "select_statement"
    match_grammar = ansi_dialect.get_segment(
        "SelectStatementSegment"
    ).match_grammar.copy()

    # Remove the Limit and Window statements from ANSI
    parse_grammar = UnorderedSelectStatementSegment.parse_grammar.copy(
        insert=[
            Ref("OrderByClauseSegment", optional=True),
        ]
    )


@tsql_dialect.segment(replace=True)
class CreateIndexStatementSegment(BaseSegment):
    """A `CREATE INDEX` statement.

    https://docs.microsoft.com/en-us/sql/t-sql/statements/create-index-transact-sql?view=sql-server-ver15
    """

    type = "create_index_statement"
    match_grammar = Sequence(
        "CREATE",
        Ref("OrReplaceGrammar", optional=True),
        Sequence("UNIQUE", optional=True),
        OneOf("CLUSTERED", "NONCLUSTERED", optional=True),
        "INDEX",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("IndexReferenceSegment"),
        "ON",
        Ref("TableReferenceSegment"),
        Sequence(
            Bracketed(
                Delimited(
                    Ref("IndexColumnDefinitionSegment"),
                ),
            )
        ),
        Sequence(
            "INCLUDE",
            Sequence(
                Bracketed(
                    Delimited(
                        Ref("IndexColumnDefinitionSegment"),
                    ),
                )
            ),
            optional=True,
        ),
    )


@tsql_dialect.segment()
class PivotUnpivotStatementSegment(BaseSegment):
    """Declaration of a variable.

    https://docs.microsoft.com/en-us/sql/t-sql/queries/from-using-pivot-and-unpivot?view=sql-server-ver15
    """

    type = "pivotunpivot_segment"
    match_grammar = StartsWith(
        OneOf("PIVOT", "UNPIVOT"),
        terminator=Ref("FromClauseTerminatorGrammar"),
        enforce_whitespace_preceding_terminator=True,
    )
    parse_grammar = Sequence(
        OneOf(
            Sequence(
                "PIVOT",
                OptionallyBracketed(
                    Sequence(
                        OptionallyBracketed(Ref("FunctionSegment")),
                        "FOR",
                        Ref("ColumnReferenceSegment"),
                        "IN",
                        Bracketed(Delimited(Ref("ColumnReferenceSegment"))),
                    )
                ),
            ),
            Sequence(
                "UNPIVOT",
                OptionallyBracketed(
                    Sequence(
                        OptionallyBracketed(Ref("ColumnReferenceSegment")),
                        "FOR",
                        Ref("ColumnReferenceSegment"),
                        "IN",
                        Bracketed(Delimited(Ref("ColumnReferenceSegment"))),
                    )
                ),
            ),
        ),
        "AS",
        Ref("TableReferenceSegment"),
    )


@tsql_dialect.segment()
class DeclareStatementSegment(BaseSegment):
    """Declaration of a variable.

    https://docs.microsoft.com/en-us/sql/t-sql/language-elements/declare-local-variable-transact-sql?view=sql-server-ver15
    """

    type = "declare_segment"
    match_grammar = StartsWith("DECLARE")
    parse_grammar = Sequence(
        "DECLARE",
        Delimited(Ref("ParameterNameSegment")),
        Ref("DatatypeSegment"),
        Sequence(
            Ref("EqualsSegment"),
            OneOf(
                Ref("LiteralGrammar"),
                Bracketed(Ref("SelectStatementSegment")),
                Ref("BareFunctionSegment"),
                Ref("FunctionSegment"),
            ),
            optional=True,
        ),
    )


@tsql_dialect.segment(replace=True)
class ObjectReferenceSegment(BaseSegment):
    """A reference to an object.

    Update ObjectReferenceSegment to only allow dot separated SingleIdentifierGrammar
    So Square Bracketed identifiers can be matched.
    """

    type = "object_reference"
    # match grammar (don't allow whitespace)
    match_grammar: Matchable = Delimited(
        Ref("SingleIdentifierGrammar"),
        delimiter=OneOf(
            Ref("DotSegment"), Sequence(Ref("DotSegment"), Ref("DotSegment"))
        ),
        allow_gaps=False,
    )


@tsql_dialect.segment()
class GoStatementSegment(BaseSegment):
    """GO signals the end of a batch of Transact-SQL statements to the SQL Server utilities.

    GO statements are not part of the TSQL language. They are used to signal batch statements
    so that clients know in how batches of statements can be executed.
    """

    type = "go_statement"
    match_grammar = Sequence("GO")


@tsql_dialect.segment(replace=True)
class DatatypeSegment(BaseSegment):
    """A data type segment.

    Updated for Transact-SQL to allow bracketed data types with bracketed schemas.
    """

    type = "data_type"
    match_grammar = Sequence(
        # Some dialects allow optional qualification of data types with schemas
        Sequence(
            Ref("SingleIdentifierGrammar"),
            Ref("DotSegment"),
            allow_gaps=False,
            optional=True,
        ),
        OneOf(
            Ref("DatatypeIdentifierSegment"),
            Bracketed(Ref("DatatypeIdentifierSegment"), bracket_type="square"),
        ),
        Bracketed(
            OneOf(
                Delimited(Ref("ExpressionSegment")),
                # The brackets might be empty for some cases...
                optional=True,
            ),
            # There may be no brackets for some data types
            optional=True,
        ),
        Ref("CharCharacterSetSegment", optional=True),
    )


@tsql_dialect.segment()
class NextValueSequenceSegment(BaseSegment):
    """Segment to get next value from a sequence."""

    type = "sequence_next_value"
    match_grammar = Sequence(
        "NEXT",
        "VALUE",
        "FOR",
        Ref("ObjectReferenceSegment"),
    )


@tsql_dialect.segment()
class IfExpressionStatement(BaseSegment):
    """IF-ELSE-END IF statement.

    https://docs.microsoft.com/en-us/sql/t-sql/language-elements/if-else-transact-sql?view=sql-server-ver15
    """

    type = "if_then_statement"

    match_grammar = Sequence(
        OneOf(
            Sequence(Ref("IfNotExistsGrammar"), Ref("SelectStatementSegment")),
            Sequence(Ref("IfExistsGrammar"), Ref("SelectStatementSegment")),
            "IF",
            Ref("ExpressionSegment"),
        ),
        Ref("StatementSegment"),
        Sequence("ELSE", Ref("StatementSegment"), optional=True),
    )


@tsql_dialect.segment(replace=True)
class ColumnConstraintSegment(BaseSegment):
    """A column option; each CREATE TABLE column can have 0 or more."""

    type = "column_constraint_segment"
    # Column constraint from
    # https://www.postgresql.org/docs/12/sql-createtable.html
    match_grammar = Sequence(
        Sequence(
            "CONSTRAINT",
            Ref("ObjectReferenceSegment"),  # Constraint name
            optional=True,
        ),
        OneOf(
            Sequence(Ref.keyword("NOT", optional=True), "NULL"),  # NOT NULL or NULL
            Sequence(  # DEFAULT <value>
                "DEFAULT",
                OneOf(
                    Ref("LiteralGrammar"),
                    Ref("FunctionSegment"),
                    # ?? Ref('IntervalExpressionSegment')
                    OptionallyBracketed(Ref("NextValueSequenceSegment")),
                ),
            ),
            Ref("PrimaryKeyGrammar"),
            "UNIQUE",  # UNIQUE
            "AUTO_INCREMENT",  # AUTO_INCREMENT (MySQL)
            "UNSIGNED",  # UNSIGNED (MySQL)
            Sequence(  # REFERENCES reftable [ ( refcolumn) ]
                "REFERENCES",
                Ref("ColumnReferenceSegment"),
                # Foreign columns making up FOREIGN KEY constraint
                Ref("BracketedColumnReferenceListGrammar", optional=True),
            ),
            Ref("CommentClauseSegment"),
        ),
    )


@tsql_dialect.segment(replace=True)
class CreateFunctionStatementSegment(BaseSegment):
    """A `CREATE FUNCTION` statement.

    This version in the TSQL dialect should be a "common subset" of the
    structure of the code for those dialects.

    Updated to include AS after declaration of RETURNS. Might be integrated in ANSI though.

    postgres: https://www.postgresql.org/docs/9.1/sql-createfunction.html
    snowflake: https://docs.snowflake.com/en/sql-reference/sql/create-function.html
    bigquery: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
    tsql/mssql : https://docs.microsoft.com/en-us/sql/t-sql/statements/create-function-transact-sql?view=sql-server-ver15
    """

    type = "create_function_statement"

    match_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        "FUNCTION",
        Anything(),
    )
    parse_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        "FUNCTION",
        Ref("ObjectReferenceSegment"),
        Ref("FunctionParameterListGrammar"),
        Sequence(  # Optional function return type
            "RETURNS",
            Ref("DatatypeSegment"),
            optional=True,
        ),
        Ref("FunctionDefinitionGrammar"),
    )


@tsql_dialect.segment()
class SetStatementSegment(BaseSegment):
    """A Set statement.

    Setting an already declared variable or global variable.
    https://docs.microsoft.com/en-us/sql/t-sql/statements/set-statements-transact-sql?view=sql-server-ver15
    """

    type = "set_segment"
    match_grammar = StartsWith("SET")
    parse_grammar = Sequence(
        "SET",
        OneOf(
            Ref("ParameterNameSegment"),
            "DATEFIRST",
            "DATEFORMAT",
            "DEADLOCK_PRIORITY",
            "LOCK_TIMEOUT",
            "CONCAT_NULL_YIELDS_NULL",
            "CURSOR_CLOSE_ON_COMMIT",
            "FIPS_FLAGGER",
            "IDENTITY_INSERT",
            "LANGUAGE",
            "OFFSETS",
            "QUOTED_IDENTIFIER",
            "ARITHABORT",
            "ARITHIGNORE",
            "FMTONLY",
            "NOCOUNT",
            "NOEXEC",
            "NUMERIC_ROUNDABORT",
            "PARSEONLY",
            "QUERY_GOVERNOR_COST_LIMIT",
            "RESULT CACHING (Preview)",
            "ROWCOUNT",
            "TEXTSIZE",
            "ANSI_DEFAULTS",
            "ANSI_NULL_DFLT_OFF",
            "ANSI_NULL_DFLT_ON",
            "ANSI_NULLS",
            "ANSI_PADDING",
            "ANSI_WARNINGS",
            "FORCEPLAN",
            "SHOWPLAN_ALL",
            "SHOWPLAN_TEXT",
            "SHOWPLAN_XML",
            "STATISTICS IO",
            "STATISTICS XML",
            "STATISTICS PROFILE",
            "STATISTICS TIME",
            "IMPLICIT_TRANSACTIONS",
            "REMOTE_PROC_TRANSACTIONS",
            "TRANSACTION ISOLATION LEVEL",
            "XACT_ABORT",
        ),
        OneOf(
            "ON",
            "OFF",
            Sequence(
                Ref("EqualsSegment"),
                OneOf(
                    Delimited(
                        OneOf(
                            Ref("LiteralGrammar"),
                            Bracketed(Ref("SelectStatementSegment")),
                            Ref("FunctionSegment"),
                            Bracketed(
                                Delimited(
                                    OneOf(
                                        Ref("LiteralGrammar"),
                                        Bracketed(Ref("SelectStatementSegment")),
                                        Ref("BareFunctionSegment"),
                                        Ref("FunctionSegment"),
                                    )
                                )
                            ),
                        )
                    )
                ),
            ),
        ),
    )


@tsql_dialect.segment(replace=True)
class FunctionDefinitionGrammar(BaseSegment):
    """This is the body of a `CREATE FUNCTION AS` statement.

    Adjusted from ansi as Transact SQL does not seem to have the QuotedLiteralSegmentand Language.
    Futhermore the body can contain almost anything like a function with table output.
    """

    type = "function_statement"
    name = "function_statement"

    match_grammar = Sequence("AS", Sequence(Anything()))


@tsql_dialect.segment()
class CreateProcedureStatementSegment(BaseSegment):
    """A `CREATE OR ALTER PROCEDURE` statement.

    https://docs.microsoft.com/en-us/sql/t-sql/statements/create-procedure-transact-sql?view=sql-server-ver15
    """

    type = "create_procedure_statement"

    match_grammar = StartsWith(
        Sequence(
            "CREATE", Sequence("OR", "ALTER", optional=True), OneOf("PROCEDURE", "PROC")
        )
    )
    parse_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        OneOf("PROCEDURE", "PROC"),
        Ref("ObjectReferenceSegment"),
        Ref("FunctionParameterListGrammar", optional=True),
        "AS",
        # Pending to add BEGIN END optional
        Ref("ProcedureDefinitionGrammar"),
    )


@tsql_dialect.segment()
class ProcedureDefinitionGrammar(BaseSegment):
    """This is the body of a `CREATE OR ALTER PROCEDURE AS` statement."""

    type = "procedure_statement"
    name = "procedure_statement"

    match_grammar = Ref("StatementSegment")


@tsql_dialect.segment(replace=True)
class CreateViewStatementSegment(BaseSegment):
    """A `CREATE VIEW` statement.

    Adjusted to allow CREATE OR ALTER instead of CREATE OR REPLACE.
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/create-view-transact-sql?view=sql-server-ver15#examples
    """

    type = "create_view_statement"
    match_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        "VIEW",
        Ref("ObjectReferenceSegment"),
        "AS",
        Ref("SelectableGrammar"),
    )


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


@tsql_dialect.segment(replace=True)
class FunctionSegment(BaseSegment):
    """A scalar or aggregate function.

    Maybe in the future we should distinguish between
    aggregate functions and other functions. For now
    we treat them the same because they look the same
    for our purposes.
    """

    type = "function"
    match_grammar = OneOf(
        Sequence(
            Sequence(
                Ref("DateAddFunctionNameSegment"),
                Bracketed(
                    Delimited(
                        Ref("DatePartClause"),
                        Ref(
                            "FunctionContentsGrammar",
                            # The brackets might be empty for some functions...
                            optional=True,
                            ephemeral_name="FunctionContentsGrammar",
                        ),
                    )
                ),
            )
        ),
        Sequence(
            Sequence(
                Ref("ConvertFunctionNameSegment"),
                Bracketed(
                    Delimited(
                        Ref("DatatypeSegment"),
                        Ref(
                            "FunctionContentsGrammar",
                            # The brackets might be empty for some functions...
                            optional=True,
                            ephemeral_name="FunctionContentsGrammar",
                        ),
                    )
                ),
            )
        ),
        Sequence(
            Sequence(
                AnyNumberOf(
                    Ref("FunctionNameSegment"),
                    max_times=1,
                    min_times=1,
                    exclude=OneOf(
                        Ref("ConvertFunctionNameSegment"),
                        Ref("DateAddFunctionNameSegment"),
                    ),
                ),
                Bracketed(
                    Ref(
                        "FunctionContentsGrammar",
                        # The brackets might be empty for some functions...
                        optional=True,
                        ephemeral_name="FunctionContentsGrammar",
                    )
                ),
            ),
            Ref("PostFunctionGrammar", optional=True),
        ),
    )


@tsql_dialect.segment(replace=True)
class CreateTableStatementSegment(BaseSegment):
    """A `CREATE TABLE` statement."""

    type = "create_table_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql?view=sql-server-ver15
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/create-table-azure-sql-data-warehouse?view=aps-pdw-2016-au7
    match_grammar = Sequence(
        "CREATE",
        "TABLE",
        Ref("TableReferenceSegment"),
        OneOf(
            # Columns and comment syntax:
            Sequence(
                Bracketed(
                    Delimited(
                        OneOf(
                            Ref("TableConstraintSegment"),
                            Ref("ColumnDefinitionSegment"),
                        ),
                    )
                ),
                Ref("CommentClauseSegment", optional=True),
            ),
            # Create AS syntax:
            Sequence(
                "AS",
                OptionallyBracketed(Ref("SelectableGrammar")),
            ),
            # Create like syntax
            Sequence("LIKE", Ref("TableReferenceSegment")),
        ),
        Ref(
            "TableDistributionIndexClause", optional=True
        ),  # Azure Synapse Analytics specific
    )

    parse_grammar = match_grammar


@tsql_dialect.segment()
class TableDistributionIndexClause(BaseSegment):
    """`CREATE TABLE` distribution / index clause.

    This is specific to Azure Synapse Analytics.
    """

    type = "table_distribution_index_clause"

    match_grammar = Sequence(
        "WITH",
        Bracketed(
            OneOf(
                Sequence(
                    Ref("TableDistributionClause"),
                    Ref("CommaSegment"),
                    Ref("TableIndexClause"),
                ),
                Sequence(
                    Ref("TableIndexClause"),
                    Ref("CommaSegment"),
                    Ref("TableDistributionClause"),
                ),
                Ref("TableDistributionClause"),
                Ref("TableIndexClause"),
            )
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
class AlterTableSwitchStatementSegment(BaseSegment):
    """An `ALTER TABLE SWITCH` statement."""

    type = "alter_table_switch_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/alter-table-transact-sql?view=sql-server-ver15
    # T-SQL's ALTER TABLE SWITCH grammar is different enough to core ALTER TABLE grammar to merit its own definition
    match_grammar = Sequence(
        "ALTER",
        "TABLE",
        Ref("ObjectReferenceSegment"),
        "SWITCH",
        Sequence("PARTITION", Ref("NumericLiteralSegment"), optional=True),
        "TO",
        Ref("ObjectReferenceSegment"),
        Sequence(  # Azure Synapse Analytics specific
            "WITH",
            Bracketed("TRUNCATE_TARGET", Ref("EqualsSegment"), OneOf("ON", "OFF")),
            optional=True,
        ),
    )


@tsql_dialect.segment()
class CreateTableAsSelectStatementSegment(BaseSegment):
    """A `CREATE TABLE AS SELECT` statement.

    This is specific to Azure Synapse Analytics.
    """

    type = "create_table_as_select_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/create-table-as-select-azure-sql-data-warehouse?toc=/azure/synapse-analytics/sql-data-warehouse/toc.json&bc=/azure/synapse-analytics/sql-data-warehouse/breadcrumb/toc.json&view=azure-sqldw-latest&preserve-view=true
    match_grammar = Sequence(
        "CREATE",
        "TABLE",
        Ref("TableReferenceSegment"),
        Ref("TableDistributionIndexClause"),
        "AS",
        Ref("SelectableGrammar"),
    )


@tsql_dialect.segment(replace=True)
class DatePartClause(BaseSegment):
    """DatePart clause for use within DATEADD() or related functions."""

    type = "date_part"

    match_grammar = OneOf(
        "D",
        "DAY",
        "DAYOFYEAR",
        "DD",
        "DW",
        "DY",
        "HH",
        "HOUR",
        "M",
        "MCS",
        "MI",
        "MICROSECOND",
        "MILLISECOND",
        "MINUTE",
        "MM",
        "MONTH",
        "MS",
        "N",
        "NANOSECOND",
        "NS",
        "Q",
        "QQ",
        "QUARTER",
        "S",
        "SECOND",
        "SS",
        "W",
        "WEEK",
        "WEEKDAY",
        "WK",
        "WW",
        "YEAR",
        "Y",
        "YY",
        "YYYY",
    )

```
### 4 - src/sqlfluff/rules/L038.py:

```python
"""Implementation of Rule L038."""

from sqlfluff.core.parser import SymbolSegment

from sqlfluff.core.rules.base import BaseRule, LintFix, LintResult
from sqlfluff.core.rules.doc_decorators import (
    document_fix_compatible,
    document_configuration,
)


@document_configuration
@document_fix_compatible
class Rule_L038(BaseRule):
    """Trailing commas within select clause.

    For some database backends this is allowed. For some users
    this may be something they wish to enforce (in line with
    python best practice). Many database backends regard this
    as a syntax error, and as such the sqlfluff default is to
    forbid trailing commas in the select clause.

    | **Anti-pattern**

    .. code-block:: sql

        SELECT
            a, b,
        FROM foo

    | **Best practice**

    .. code-block:: sql

        SELECT
            a, b
        FROM foo
    """

    config_keywords = ["select_clause_trailing_comma"]

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

    def _eval(self, segment, **kwargs):
        """Blank line expected but not found after CTE definition."""
        error_buffer = []
        if segment.is_type("with_compound_statement"):
            # First we need to find all the commas, the end brackets, the
            # things that come after that and the blank lines in between.

            # Find all the closing brackets. They are our anchor points.
            bracket_indices = []
            expanded_segments = list(
                segment.iter_segments(expanding=["common_table_expression"])
            )
            for idx, seg in enumerate(expanded_segments):
                if seg.is_type("bracketed"):
                    bracket_indices.append(idx)

            # Work through each point and deal with it individually
            for bracket_idx in bracket_indices:
                forward_slice = expanded_segments[bracket_idx:]
                seg_idx = 1
                line_idx = 0
                comma_seg_idx = None
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
### 6 - src/sqlfluff/core/linter/linter.py:

```python
"""Defines the linter class."""

import os
import time
import logging
from typing import (
    Any,
    Generator,
    List,
    Sequence,
    Optional,
    Tuple,
    Union,
    cast,
    Iterable,
)

import pathspec

from sqlfluff.core.errors import (
    SQLBaseError,
    SQLLexError,
    SQLLintError,
    SQLParseError,
    SQLTemplaterSkipFile,
)
from sqlfluff.core.parser import Lexer, Parser
from sqlfluff.core.file_helpers import get_encoding
from sqlfluff.core.templaters import TemplatedFile
from sqlfluff.core.rules import get_ruleset
from sqlfluff.core.config import FluffConfig, ConfigLoader

# Classes needed only for type checking
from sqlfluff.core.linter.runner import get_runner
from sqlfluff.core.parser.segments.base import BaseSegment
from sqlfluff.core.parser.segments.meta import MetaSegment
from sqlfluff.core.parser.segments.raw import RawSegment
from sqlfluff.core.rules.base import BaseRule

from sqlfluff.core.linter.common import (
    RuleTuple,
    ParsedString,
    NoQaDirective,
    RenderedFile,
)
from sqlfluff.core.linter.linted_file import LintedFile
from sqlfluff.core.linter.linted_dir import LintedDir
from sqlfluff.core.linter.linting_result import LintingResult


WalkableType = Iterable[Tuple[str, Optional[List[str]], List[str]]]

# Instantiate the linter logger
linter_logger: logging.Logger = logging.getLogger("sqlfluff.linter")


class Linter:
    """The interface class to interact with the linter."""

    # Default to allowing process parallelism
    allow_process_parallelism = True

    def __init__(
        self,
        config: Optional[FluffConfig] = None,
        formatter: Any = None,
        dialect: Optional[str] = None,
        rules: Optional[Union[str, List[str]]] = None,
        user_rules: Optional[Union[str, List[str]]] = None,
    ) -> None:
        # Store the config object
        self.config = FluffConfig.from_kwargs(
            config=config, dialect=dialect, rules=rules
        )
        # Get the dialect and templater
        self.dialect = self.config.get("dialect_obj")
        self.templater = self.config.get("templater_obj")
        # Store the formatter for output
        self.formatter = formatter
        # Store references to user rule classes
        self.user_rules = user_rules or []

    def get_ruleset(self, config: Optional[FluffConfig] = None) -> List[BaseRule]:
        """Get hold of a set of rules."""
        rs = get_ruleset()
        # Register any user rules
        for rule in self.user_rules:
            rs.register(rule)
        cfg = config or self.config
        return rs.get_rulelist(config=cfg)

    def rule_tuples(self) -> List[RuleTuple]:
        """A simple pass through to access the rule tuples of the rule set."""
        rs = self.get_ruleset()
        return [RuleTuple(rule.code, rule.description) for rule in rs]

    # #### Static methods
    # These are the building blocks of the linting process.

    @staticmethod
    def _load_raw_file_and_config(fname, root_config):
        """Load a raw file and the associated config."""
        file_config = root_config.make_child_from_path(fname)
        encoding = get_encoding(fname=fname, config=file_config)
        with open(fname, encoding=encoding, errors="backslashreplace") as target_file:
            raw_file = target_file.read()
        # Scan the raw file for config commands.
        file_config.process_raw_file_for_config(raw_file)
        # Return the raw file and config
        return raw_file, file_config, encoding

    @staticmethod
    def _lex_templated_file(
        templated_file: TemplatedFile, config: FluffConfig
    ) -> Tuple[Optional[Sequence[BaseSegment]], List[SQLLexError], FluffConfig]:
        """Lex a templated file.

        NOTE: This potentially mutates the config, so make sure to
        use the returned one.
        """
        violations = []
        linter_logger.info("LEXING RAW (%s)", templated_file.fname)
        # Get the lexer
        lexer = Lexer(config=config)
        # Lex the file and log any problems
        try:
            tokens, lex_vs = lexer.lex(templated_file)
            # We might just get the violations as a list
            violations += lex_vs
            linter_logger.info(
                "Lexed tokens: %s", [seg.raw for seg in tokens] if tokens else None
            )
        except SQLLexError as err:
            linter_logger.info("LEXING FAILED! (%s): %s", templated_file.fname, err)
            violations.append(err)
            return None, violations, config

        if not tokens:  # pragma: no cover TODO?
            return None, violations, config

        # Check that we've got sensible indentation from the lexer.
        # We might need to suppress if it's a complicated file.
        templating_blocks_indent = config.get("template_blocks_indent", "indentation")
        if isinstance(templating_blocks_indent, str):
            force_block_indent = templating_blocks_indent.lower().strip() == "force"
        else:
            force_block_indent = False
        templating_blocks_indent = bool(templating_blocks_indent)
        # If we're forcing it through we don't check.
        if templating_blocks_indent and not force_block_indent:
            indent_balance = sum(
                getattr(elem, "indent_val", 0)
                for elem in cast(Tuple[BaseSegment, ...], tokens)
            )
            if indent_balance != 0:
                linter_logger.debug(
                    "Indent balance test failed for %r. Template indents will not be linted for this file.",
                    templated_file.fname,
                )
                # Don't enable the templating blocks.
                templating_blocks_indent = False
                # Disable the linting of L003 on templated tokens.
                config.set_value(["rules", "L003", "lint_templated_tokens"], False)

        # The file will have been lexed without config, so check all indents
        # are enabled.
        new_tokens = []
        for token in cast(Tuple[BaseSegment, ...], tokens):
            if token.is_meta:
                token = cast(MetaSegment, token)
                if token.indent_val != 0:
                    # Don't allow it if we're not linting templating block indents.
                    if not templating_blocks_indent:
                        continue
            new_tokens.append(token)
        # Return new buffer
        return new_tokens, violations, config

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

    @staticmethod
    def parse_noqa(comment: str, line_no: int):
        """Extract ignore mask entries from a comment string."""
        # Also trim any whitespace afterward
        if comment.startswith("noqa"):
            # This is an ignore identifier
            comment_remainder = comment[4:]
            if comment_remainder:
                if not comment_remainder.startswith(":"):
                    return SQLParseError(
                        "Malformed 'noqa' section. Expected 'noqa: <rule>[,...]",
                        line_no=line_no,
                    )
                comment_remainder = comment_remainder[1:].strip()
                if comment_remainder:
                    action: Optional[str]
                    if "=" in comment_remainder:
                        action, rule_part = comment_remainder.split("=", 1)
                        if action not in {"disable", "enable"}:  # pragma: no cover
                            return SQLParseError(
                                "Malformed 'noqa' section. "
                                "Expected 'noqa: enable=<rule>[,...] | all' "
                                "or 'noqa: disable=<rule>[,...] | all",
                                line_no=line_no,
                            )
                    else:
                        action = None
                        rule_part = comment_remainder
                        if rule_part in {"disable", "enable"}:
                            return SQLParseError(
                                "Malformed 'noqa' section. "
                                "Expected 'noqa: enable=<rule>[,...] | all' "
                                "or 'noqa: disable=<rule>[,...] | all",
                                line_no=line_no,
                            )
                    rules: Optional[Tuple[str, ...]]
                    if rule_part != "all":
                        rules = tuple(r.strip() for r in rule_part.split(","))
                    else:
                        rules = None
                    return NoQaDirective(line_no, rules, action)
            return NoQaDirective(line_no, None, None)
        return None

    @staticmethod
    def remove_templated_errors(
        linting_errors: List[SQLBaseError],
    ) -> List[SQLBaseError]:
        """Filter a list of lint errors, removing those which only occur in templated slices."""
        # Filter out any linting errors in templated sections if relevant.
        result: List[SQLBaseError] = []
        for e in linting_errors:
            if isinstance(e, SQLLintError):
                if (
                    # Is it in a literal section?
                    e.segment.pos_marker.is_literal()
                    # Is it a rule that is designed to work on templated sections?
                    or e.rule.targets_templated
                ):
                    result.append(e)
            else:
                # If it's another type, just keep it. (E.g. SQLParseError from
                # malformed "noqa" comment).
                result.append(e)
        return result

    @staticmethod
    def _warn_unfixable(code: str):
        linter_logger.warning(
            f"One fix for {code} not applied, it would re-cause the same error."
        )

    # ### Class Methods
    # These compose the base static methods into useful recipes.

    @classmethod
    def parse_rendered(cls, rendered: RenderedFile, recurse: bool = True):
        """Parse a rendered file."""
        t0 = time.monotonic()
        violations = cast(List[SQLBaseError], rendered.templater_violations)
        tokens: Optional[Sequence[BaseSegment]]
        if rendered.templated_file:
            tokens, lvs, config = cls._lex_templated_file(
                rendered.templated_file, rendered.config
            )
            violations += lvs
        else:
            tokens = None

        t1 = time.monotonic()
        linter_logger.info("PARSING (%s)", rendered.fname)

        if tokens:
            parsed, pvs = cls._parse_tokens(
                tokens, rendered.config, recurse=recurse, fname=rendered.fname
            )
            violations += pvs
        else:
            parsed = None

        time_dict = {
            **rendered.time_dict,
            "lexing": t1 - t0,
            "parsing": time.monotonic() - t1,
        }
        return ParsedString(
            parsed,
            violations,
            time_dict,
            rendered.templated_file,
            rendered.config,
            rendered.fname,
        )

    @classmethod
    def extract_ignore_from_comment(cls, comment: RawSegment):
        """Extract ignore mask entries from a comment segment."""
        # Also trim any whitespace afterward
        comment_content = comment.raw_trimmed().strip()
        comment_line, _ = comment.pos_marker.source_position()
        result = cls.parse_noqa(comment_content, comment_line)
        if isinstance(result, SQLParseError):
            result.segment = comment
        return result

    @classmethod
    def extract_ignore_mask(
        cls, tree: BaseSegment
    ) -> Tuple[List[NoQaDirective], List[SQLBaseError]]:
        """Look for inline ignore comments and return NoQaDirectives."""
        ignore_buff: List[NoQaDirective] = []
        violations: List[SQLBaseError] = []
        for comment in tree.recursive_crawl("comment"):
            if comment.name == "inline_comment":
                ignore_entry = cls.extract_ignore_from_comment(comment)
                if isinstance(ignore_entry, SQLParseError):
                    violations.append(ignore_entry)
                elif ignore_entry:
                    ignore_buff.append(ignore_entry)
        if ignore_buff:
            linter_logger.info("Parsed noqa directives from file: %r", ignore_buff)
        return ignore_buff, violations

    @classmethod
    def lint_fix_parsed(
        cls,
        tree: BaseSegment,
        config: FluffConfig,
        rule_set: List[BaseRule],
        fix: bool = False,
        fname: Optional[str] = None,
        templated_file: Optional[TemplatedFile] = None,
        formatter: Any = None,
    ) -> Tuple[BaseSegment, List[SQLBaseError], List[NoQaDirective]]:
        """Lint and optionally fix a tree object."""
        # Keep track of the linting errors
        all_linting_errors = []
        # A placeholder for the fixes we had on the previous loop
        last_fixes = None
        # Keep a set of previous versions to catch infinite loops.
        previous_versions = {tree.raw}

        # If we are fixing then we want to loop up to the runaway_limit, otherwise just once for linting.
        loop_limit = config.get("runaway_limit") if fix else 1

        # Dispatch the output for the lint header
        if formatter:
            formatter.dispatch_lint_header(fname)

        # Look for comment segments which might indicate lines to ignore.
        ignore_buff, ivs = cls.extract_ignore_mask(tree)
        all_linting_errors += ivs

        for loop in range(loop_limit):
            changed = False
            for crawler in rule_set:
                # fixes should be a dict {} with keys edit, delete, create
                # delete is just a list of segments to delete
                # edit and create are list of tuples. The first element is the
                # "anchor", the segment to look for either to edit or to insert BEFORE.
                # The second is the element to insert or create.
                linting_errors, _, fixes, _ = crawler.crawl(
                    tree,
                    ignore_mask=ignore_buff,
                    dialect=config.get("dialect_obj"),
                    fname=fname,
                    templated_file=templated_file,
                )
                all_linting_errors += linting_errors

                if fix and fixes:
                    linter_logger.info(f"Applying Fixes [{crawler.code}]: {fixes}")
                    # Do some sanity checks on the fixes before applying.
                    if fixes == last_fixes:  # pragma: no cover
                        cls._warn_unfixable(crawler.code)
                    else:
                        last_fixes = fixes
                        new_tree, _ = tree.apply_fixes(fixes)
                        # Check for infinite loops
                        if new_tree.raw not in previous_versions:
                            # We've not seen this version of the file so far. Continue.
                            tree = new_tree
                            previous_versions.add(tree.raw)
                            changed = True
                            continue
                        else:
                            # Applying these fixes took us back to a state which we've
                            # seen before. Abort.
                            cls._warn_unfixable(crawler.code)

            if loop == 0:
                # Keep track of initial errors for reporting.
                initial_linting_errors = all_linting_errors.copy()

            if fix and not changed:
                # We did not change the file. Either the file is clean (no fixes), or
                # any fixes which are present will take us back to a previous state.
                linter_logger.info(
                    f"Fix loop complete. Stability achieved after {loop}/{loop_limit} loops."
                )
                break
        if fix and loop + 1 == loop_limit:
            linter_logger.warning(f"Loop limit on fixes reached [{loop_limit}].")

        if config.get("ignore_templated_areas", default=True):
            initial_linting_errors = cls.remove_templated_errors(initial_linting_errors)

        return tree, initial_linting_errors, ignore_buff

    @classmethod
    def lint_parsed(
        cls,
        parsed: ParsedString,
        rule_set: List[BaseRule],
        fix: bool = False,
        formatter: Any = None,
        encoding: str = "utf8",
    ):
        """Lint a ParsedString and return a LintedFile."""
        violations = parsed.violations
        time_dict = parsed.time_dict
        tree: Optional[BaseSegment]
        if parsed.tree:
            t0 = time.monotonic()
            linter_logger.info("LINTING (%s)", parsed.fname)
            tree, initial_linting_errors, ignore_buff = cls.lint_fix_parsed(
                parsed.tree,
                config=parsed.config,
                rule_set=rule_set,
                fix=fix,
                fname=parsed.fname,
                templated_file=parsed.templated_file,
                formatter=formatter,
            )
            # Update the timing dict
            time_dict["linting"] = time.monotonic() - t0

            # We're only going to return the *initial* errors, rather
            # than any generated during the fixing cycle.
            violations += initial_linting_errors
        else:
            # If no parsed tree, set to None
            tree = None
            ignore_buff = []

        # We process the ignore config here if appropriate
        for violation in violations:
            violation.ignore_if_in(parsed.config.get("ignore"))

        linted_file = LintedFile(
            parsed.fname,
            violations,
            time_dict,
            tree,
            ignore_mask=ignore_buff,
            templated_file=parsed.templated_file,
            encoding=encoding,
        )

        # This is the main command line output from linting.
        if formatter:
            formatter.dispatch_file_violations(
                parsed.fname, linted_file, only_fixable=fix
            )

        # Safety flag for unset dialects
        if parsed.config.get("dialect") == "ansi" and linted_file.get_violations(
            fixable=True if fix else None, types=SQLParseError
        ):
            if formatter:  # pragma: no cover TODO?
                formatter.dispatch_dialect_warning()

        return linted_file

    @classmethod
    def lint_rendered(
        cls,
        rendered: RenderedFile,
        rule_set: List[BaseRule],
        fix: bool = False,
        formatter: Any = None,
    ) -> LintedFile:
        """Take a RenderedFile and return a LintedFile."""
        parsed = cls.parse_rendered(rendered)
        return cls.lint_parsed(
            parsed,
            rule_set=rule_set,
            fix=fix,
            formatter=formatter,
            encoding=rendered.encoding,
        )

    # ### Instance Methods
    # These are tied to a specific instance and so are not necessarily
    # safe to use in parallel operations.

    def render_string(
        self, in_str: str, fname: str, config: FluffConfig, encoding: str
    ) -> RenderedFile:
        """Template the file."""
        linter_logger.info("TEMPLATING RAW [%s] (%s)", self.templater.name, fname)

        # Start the templating timer
        t0 = time.monotonic()

        if not config.get("templater_obj") == self.templater:
            linter_logger.warning(
                (
                    f"Attempt to set templater to {config.get('templater_obj').name} failed. Using {self.templater.name} "
                    "templater. Templater cannot be set in a .sqlfluff file in a subdirectory of the current working "
                    "directory. It can be set in a .sqlfluff in the current working directory. See Nesting section of the "
                    "docs for more details."
                )
            )
        try:
            templated_file, templater_violations = self.templater.process(
                in_str=in_str, fname=fname, config=config, formatter=self.formatter
            )
        except SQLTemplaterSkipFile as s:  # pragma: no cover
            linter_logger.warning(str(s))
            templated_file = None
            templater_violations = []

        if not templated_file:
            linter_logger.info("TEMPLATING FAILED: %s", templater_violations)

        # Record time
        time_dict = {"templating": time.monotonic() - t0}

        return RenderedFile(
            templated_file, templater_violations, config, time_dict, fname, encoding
        )

    def render_file(self, fname: str, root_config: FluffConfig) -> RenderedFile:
        """Load and render a file with relevant config."""
        # Load the raw file.
        raw_file, config, encoding = self._load_raw_file_and_config(fname, root_config)
        # Render the file
        return self.render_string(raw_file, fname, config, encoding)

    def parse_string(
        self,
        in_str: str,
        fname: str = "<string>",
        recurse: bool = True,
        config: Optional[FluffConfig] = None,
        encoding: str = "utf-8",
    ) -> ParsedString:
        """Parse a string."""
        violations: List[SQLBaseError] = []

        # Dispatch the output for the template header (including the config diff)
        if self.formatter:
            self.formatter.dispatch_template_header(fname, self.config, config)

        # Just use the local config from here:
        config = config or self.config

        # Scan the raw file for config commands.
        config.process_raw_file_for_config(in_str)
        rendered = self.render_string(in_str, fname, config, encoding)
        violations += rendered.templater_violations

        # Dispatch the output for the parse header
        if self.formatter:
            self.formatter.dispatch_parse_header(fname)

        return self.parse_rendered(rendered, recurse=recurse)

    def fix(
        self,
        tree: BaseSegment,
        config: Optional[FluffConfig] = None,
        fname: Optional[str] = None,
        templated_file: Optional[TemplatedFile] = None,
    ) -> Tuple[BaseSegment, List[SQLBaseError]]:
        """Return the fixed tree and violations from lintfix when we're fixing."""
        config = config or self.config
        rule_set = self.get_ruleset(config=config)
        fixed_tree, violations, _ = self.lint_fix_parsed(
            tree,
            config,
            rule_set,
            fix=True,
            fname=fname,
            templated_file=templated_file,
            formatter=self.formatter,
        )
        return fixed_tree, violations

    def lint(
        self,
        tree: BaseSegment,
        config: Optional[FluffConfig] = None,
        fname: Optional[str] = None,
        templated_file: Optional[TemplatedFile] = None,
    ) -> List[SQLBaseError]:
        """Return just the violations from lintfix when we're only linting."""
        config = config or self.config
        rule_set = self.get_ruleset(config=config)
        _, violations, _ = self.lint_fix_parsed(
            tree,
            config,
            rule_set,
            fix=False,
            fname=fname,
            templated_file=templated_file,
            formatter=self.formatter,
        )
        return violations

    def lint_string(
        self,
        in_str: str = "",
        fname: str = "<string input>",
        fix: bool = False,
        config: Optional[FluffConfig] = None,
        encoding: str = "utf8",
    ) -> LintedFile:
        """Lint a string.

        Returns:
            :obj:`LintedFile`: an object representing that linted file.

        """
        # Sort out config, defaulting to the built in config if no override
        config = config or self.config
        # Parse the string.
        parsed = self.parse_string(in_str=in_str, fname=fname, config=config)
        # Get rules as appropriate
        rule_set = self.get_ruleset(config=config)
        # Lint the file and return the LintedFile
        return self.lint_parsed(
            parsed, rule_set, fix=fix, formatter=self.formatter, encoding=encoding
        )

    def paths_from_path(
        self,
        path: str,
        ignore_file_name: str = ".sqlfluffignore",
        ignore_non_existent_files: bool = False,
        ignore_files: bool = True,
        working_path: str = os.getcwd(),
    ) -> List[str]:
        """Return a set of sql file paths from a potentially more ambiguous path string.

        Here we also deal with the .sqlfluffignore file if present.

        When a path to a file to be linted is explicitly passed
        we look for ignore files in all directories that are parents of the file,
        up to the current directory.

        If the current directory is not a parent of the file we only
        look for an ignore file in the direct parent of the file.

        """
        if not os.path.exists(path):
            if ignore_non_existent_files:
                return []
            else:
                raise OSError("Specified path does not exist")

        # Files referred to exactly are also ignored if
        # matched, but we warn the users when that happens
        is_exact_file = os.path.isfile(path)

        if is_exact_file:
            # When the exact file to lint is passed, we
            # fill path_walk with an input that follows
            # the structure of `os.walk`:
            #   (root, directories, files)
            dirpath = os.path.dirname(path)
            files = [os.path.basename(path)]
            ignore_file_paths = ConfigLoader.find_ignore_config_files(
                path=path, working_path=working_path, ignore_file_name=ignore_file_name
            )
            # Add paths that could contain "ignore files"
            # to the path_walk list
            path_walk_ignore_file = [
                (
                    os.path.dirname(ignore_file_path),
                    None,
                    # Only one possible file, since we only
                    # have one "ignore file name"
                    [os.path.basename(ignore_file_path)],
                )
                for ignore_file_path in ignore_file_paths
            ]
            path_walk: WalkableType = [(dirpath, None, files)] + path_walk_ignore_file
        else:
            path_walk = os.walk(path)

        # If it's a directory then expand the path!
        buffer = []
        ignore_set = set()
        for dirpath, _, filenames in path_walk:
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                # Handle potential .sqlfluffignore files
                if ignore_files and fname == ignore_file_name:
                    with open(fpath) as fh:
                        spec = pathspec.PathSpec.from_lines("gitwildmatch", fh)
                    matches = spec.match_tree(dirpath)
                    for m in matches:
                        ignore_path = os.path.join(dirpath, m)
                        ignore_set.add(os.path.abspath(ignore_path))
                    # We don't need to process the ignore file any futher
                    continue

                # We won't purge files *here* because there's an edge case
                # that the ignore file is processed after the sql file.

                # Scan for remaining files
                for ext in self.config.get("sql_file_exts", default=".sql").split(","):
                    # is it a sql file?
                    if fname.endswith(ext):
                        buffer.append(fpath)

        if not ignore_files:
            return sorted(buffer)

        # Check the buffer for ignore items and normalise the rest.
        filtered_buffer = []

        for fpath in buffer:
            if os.path.abspath(fpath) not in ignore_set:
                filtered_buffer.append(os.path.normpath(fpath))
            elif is_exact_file:
                linter_logger.warning(
                    "Exact file path %s was given but "
                    "it was ignored by a %s pattern, "
                    "re-run with `--disregard-sqlfluffignores` to "
                    "skip %s"
                    % (
                        path,
                        ignore_file_name,
                        ignore_file_name,
                    )
                )

        # Return
        return sorted(filtered_buffer)

    def lint_string_wrapped(
        self, string: str, fname: str = "<string input>", fix: bool = False
    ) -> LintingResult:
        """Lint strings directly."""
        result = LintingResult()
        linted_path = LintedDir(fname)
        linted_path.add(self.lint_string(string, fname=fname, fix=fix))
        result.add(linted_path)
        result.stop_timer()
        return result

    def lint_path(
        self,
        path: str,
        fix: bool = False,
        ignore_non_existent_files: bool = False,
        ignore_files: bool = True,
        processes: int = 1,
    ) -> LintedDir:
        """Lint a path."""
        linted_path = LintedDir(path)
        if self.formatter:
            self.formatter.dispatch_path(path)
        fnames = list(
            self.paths_from_path(
                path,
                ignore_non_existent_files=ignore_non_existent_files,
                ignore_files=ignore_files,
            )
        )
        runner = get_runner(
            self,
            self.config,
            processes=processes,
            allow_process_parallelism=self.allow_process_parallelism,
        )
        for linted_file in runner.run(fnames, fix):
            linted_path.add(linted_file)
            # If any fatal errors, then stop iteration.
            if any(v.fatal for v in linted_file.violations):  # pragma: no cover
                linter_logger.error("Fatal linting error. Halting further linting.")
                break
        return linted_path

    def lint_paths(
        self,
        paths: Tuple[str, ...],
        fix: bool = False,
        ignore_non_existent_files: bool = False,
        ignore_files: bool = True,
        processes: int = 1,
    ) -> LintingResult:
        """Lint an iterable of paths."""
        # If no paths specified - assume local
        if len(paths) == 0:  # pragma: no cover
            paths = (os.getcwd(),)
        # Set up the result to hold what we get back
        result = LintingResult()
        for path in paths:
            # Iterate through files recursively in the specified directory (if it's a directory)
            # or read the file directly if it's not
            result.add(
                self.lint_path(
                    path,
                    fix=fix,
                    ignore_non_existent_files=ignore_non_existent_files,
                    ignore_files=ignore_files,
                    processes=processes,
                )
            )
        result.stop_timer()
        return result

    def parse_path(
        self, path: str, recurse: bool = True
    ) -> Generator[ParsedString, None, None]:
        """Parse a path of sql files.

        NB: This a generator which will yield the result of each file
        within the path iteratively.
        """
        for fname in self.paths_from_path(path):
            if self.formatter:
                self.formatter.dispatch_path(path)
            # Load the file with the config and yield the result.
            raw_file, config, encoding = self._load_raw_file_and_config(
                fname, self.config
            )
            yield self.parse_string(
                raw_file, fname=fname, recurse=recurse, config=config, encoding=encoding
            )

```
