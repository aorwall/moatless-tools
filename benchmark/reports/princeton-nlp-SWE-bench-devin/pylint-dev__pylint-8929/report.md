# pylint-dev__pylint-8929

| **pylint-dev/pylint** | `f40e9ffd766bb434a0181dd9db3886115d2dfb2f` |
| ---- | ---- |
| **No of patches** | 5 |
| **All found context length** | 5357 |
| **Any found context length** | 190 |
| **Avg pos** | 5.2 |
| **Min pos** | 1 |
| **Max pos** | 12 |
| **Top file pos** | 1 |
| **Missing snippets** | 10 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/pylint/interfaces.py b/pylint/interfaces.py
--- a/pylint/interfaces.py
+++ b/pylint/interfaces.py
@@ -35,3 +35,4 @@ class Confidence(NamedTuple):
 
 CONFIDENCE_LEVELS = [HIGH, CONTROL_FLOW, INFERENCE, INFERENCE_FAILURE, UNDEFINED]
 CONFIDENCE_LEVEL_NAMES = [i.name for i in CONFIDENCE_LEVELS]
+CONFIDENCE_MAP = {i.name: i for i in CONFIDENCE_LEVELS}
diff --git a/pylint/lint/base_options.py b/pylint/lint/base_options.py
--- a/pylint/lint/base_options.py
+++ b/pylint/lint/base_options.py
@@ -102,9 +102,10 @@ def _make_linter_options(linter: PyLinter) -> Options:
                 "metavar": "<format>",
                 "short": "f",
                 "group": "Reports",
-                "help": "Set the output format. Available formats are text,"
-                " parseable, colorized, json and msvs (visual studio)."
-                " You can also give a reporter class, e.g. mypackage.mymodule."
+                "help": "Set the output format. Available formats are: text, "
+                "parseable, colorized, json2 (improved json format), json "
+                "(old json format) and msvs (visual studio). "
+                "You can also give a reporter class, e.g. mypackage.mymodule."
                 "MyReporterClass.",
                 "kwargs": {"linter": linter},
             },
diff --git a/pylint/reporters/__init__.py b/pylint/reporters/__init__.py
--- a/pylint/reporters/__init__.py
+++ b/pylint/reporters/__init__.py
@@ -11,7 +11,7 @@
 from pylint import utils
 from pylint.reporters.base_reporter import BaseReporter
 from pylint.reporters.collecting_reporter import CollectingReporter
-from pylint.reporters.json_reporter import JSONReporter
+from pylint.reporters.json_reporter import JSON2Reporter, JSONReporter
 from pylint.reporters.multi_reporter import MultiReporter
 from pylint.reporters.reports_handler_mix_in import ReportsHandlerMixIn
 
@@ -28,6 +28,7 @@ def initialize(linter: PyLinter) -> None:
     "BaseReporter",
     "ReportsHandlerMixIn",
     "JSONReporter",
+    "JSON2Reporter",
     "CollectingReporter",
     "MultiReporter",
 ]
diff --git a/pylint/reporters/json_reporter.py b/pylint/reporters/json_reporter.py
--- a/pylint/reporters/json_reporter.py
+++ b/pylint/reporters/json_reporter.py
@@ -9,7 +9,7 @@
 import json
 from typing import TYPE_CHECKING, Optional, TypedDict
 
-from pylint.interfaces import UNDEFINED
+from pylint.interfaces import CONFIDENCE_MAP, UNDEFINED
 from pylint.message import Message
 from pylint.reporters.base_reporter import BaseReporter
 from pylint.typing import MessageLocationTuple
@@ -37,8 +37,12 @@
 )
 
 
-class BaseJSONReporter(BaseReporter):
-    """Report messages and layouts in JSON."""
+class JSONReporter(BaseReporter):
+    """Report messages and layouts in JSON.
+
+    Consider using JSON2Reporter instead, as it is superior and this reporter
+    is no longer maintained.
+    """
 
     name = "json"
     extension = "json"
@@ -54,25 +58,6 @@ def display_reports(self, layout: Section) -> None:
     def _display(self, layout: Section) -> None:
         """Do nothing."""
 
-    @staticmethod
-    def serialize(message: Message) -> OldJsonExport:
-        raise NotImplementedError
-
-    @staticmethod
-    def deserialize(message_as_json: OldJsonExport) -> Message:
-        raise NotImplementedError
-
-
-class JSONReporter(BaseJSONReporter):
-
-    """
-    TODO: 3.0: Remove this JSONReporter in favor of the new one handling abs-path
-    and confidence.
-
-    TODO: 3.0: Add a new JSONReporter handling abs-path, confidence and scores.
-    (Ultimately all other breaking change related to json for 3.0).
-    """
-
     @staticmethod
     def serialize(message: Message) -> OldJsonExport:
         return {
@@ -96,7 +81,6 @@ def deserialize(message_as_json: OldJsonExport) -> Message:
             symbol=message_as_json["symbol"],
             msg=message_as_json["message"],
             location=MessageLocationTuple(
-                # TODO: 3.0: Add abs-path and confidence in a new JSONReporter
                 abspath=message_as_json["path"],
                 path=message_as_json["path"],
                 module=message_as_json["module"],
@@ -106,10 +90,112 @@ def deserialize(message_as_json: OldJsonExport) -> Message:
                 end_line=message_as_json["endLine"],
                 end_column=message_as_json["endColumn"],
             ),
-            # TODO: 3.0: Make confidence available in a new JSONReporter
             confidence=UNDEFINED,
         )
 
 
+class JSONMessage(TypedDict):
+    type: str
+    message: str
+    messageId: str
+    symbol: str
+    confidence: str
+    module: str
+    path: str
+    absolutePath: str
+    line: int
+    endLine: int | None
+    column: int
+    endColumn: int | None
+    obj: str
+
+
+class JSON2Reporter(BaseReporter):
+    name = "json2"
+    extension = "json2"
+
+    def display_reports(self, layout: Section) -> None:
+        """Don't do anything in this reporter."""
+
+    def _display(self, layout: Section) -> None:
+        """Do nothing."""
+
+    def display_messages(self, layout: Section | None) -> None:
+        """Launch layouts display."""
+        output = {
+            "messages": [self.serialize(message) for message in self.messages],
+            "statistics": self.serialize_stats(),
+        }
+        print(json.dumps(output, indent=4), file=self.out)
+
+    @staticmethod
+    def serialize(message: Message) -> JSONMessage:
+        return JSONMessage(
+            type=message.category,
+            symbol=message.symbol,
+            message=message.msg or "",
+            messageId=message.msg_id,
+            confidence=message.confidence.name,
+            module=message.module,
+            obj=message.obj,
+            line=message.line,
+            column=message.column,
+            endLine=message.end_line,
+            endColumn=message.end_column,
+            path=message.path,
+            absolutePath=message.abspath,
+        )
+
+    @staticmethod
+    def deserialize(message_as_json: JSONMessage) -> Message:
+        return Message(
+            msg_id=message_as_json["messageId"],
+            symbol=message_as_json["symbol"],
+            msg=message_as_json["message"],
+            location=MessageLocationTuple(
+                abspath=message_as_json["absolutePath"],
+                path=message_as_json["path"],
+                module=message_as_json["module"],
+                obj=message_as_json["obj"],
+                line=message_as_json["line"],
+                column=message_as_json["column"],
+                end_line=message_as_json["endLine"],
+                end_column=message_as_json["endColumn"],
+            ),
+            confidence=CONFIDENCE_MAP[message_as_json["confidence"]],
+        )
+
+    def serialize_stats(self) -> dict[str, str | int | dict[str, int]]:
+        """Serialize the linter stats into something JSON dumpable."""
+        stats = self.linter.stats
+
+        counts_dict = {
+            "fatal": stats.fatal,
+            "error": stats.error,
+            "warning": stats.warning,
+            "refactor": stats.refactor,
+            "convention": stats.convention,
+            "info": stats.info,
+        }
+
+        # Calculate score based on the evaluation option
+        evaluation = self.linter.config.evaluation
+        try:
+            note: int = eval(  # pylint: disable=eval-used
+                evaluation, {}, {**counts_dict, "statement": stats.statement or 1}
+            )
+        except Exception as ex:  # pylint: disable=broad-except
+            score: str | int = f"An exception occurred while rating: {ex}"
+        else:
+            score = round(note, 2)
+
+        return {
+            "messageTypeCount": counts_dict,
+            "modulesLinted": len(stats.by_module),
+            "score": score,
+        }
+
+
 def register(linter: PyLinter) -> None:
     linter.register_reporter(JSONReporter)
+    linter.register_reporter(JSON2Reporter)
diff --git a/pylint/testutils/_primer/primer_run_command.py b/pylint/testutils/_primer/primer_run_command.py
--- a/pylint/testutils/_primer/primer_run_command.py
+++ b/pylint/testutils/_primer/primer_run_command.py
@@ -13,8 +13,7 @@
 
 from pylint.lint import Run
 from pylint.message import Message
-from pylint.reporters import JSONReporter
-from pylint.reporters.json_reporter import OldJsonExport
+from pylint.reporters.json_reporter import JSONReporter, OldJsonExport
 from pylint.testutils._primer.package_to_lint import PackageToLint
 from pylint.testutils._primer.primer_command import (
     PackageData,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/interfaces.py | 38 | 38 | - | - | -
| pylint/lint/base_options.py | 105 | 107 | - | 63 | -
| pylint/reporters/__init__.py | 14 | 14 | - | - | -
| pylint/reporters/__init__.py | 31 | 31 | - | - | -
| pylint/reporters/json_reporter.py | 12 | 12 | 1 | 1 | 190
| pylint/reporters/json_reporter.py | 40 | 41 | 7 | 1 | 2753
| pylint/reporters/json_reporter.py | 57 | 75 | - | 1 | -
| pylint/reporters/json_reporter.py | 99 | 99 | 3 | 1 | 569
| pylint/reporters/json_reporter.py | 109 | 109 | 3 | 1 | 569
| pylint/testutils/_primer/primer_run_command.py | 16 | 17 | 12 | 8 | 5357


## Problem Statement

```
Exporting to JSON does not honor score option
<!--
  Hi there! Thank you for discovering and submitting an issue.

  Before you submit this, make sure that the issue doesn't already exist
  or if it is not closed.

  Is your issue fixed on the preview release?: pip install pylint astroid --pre -U

-->

### Steps to reproduce
1. Run pylint on some random Python file or module:
\`\`\`
pylint  ~/Desktop/pylint_test.py
\`\`\`
As you can see this outputs some warnings/scoring:
\`\`\`
************* Module pylint_test
/home/administrator/Desktop/pylint_test.py:1:0: C0111: Missing module docstring (missing-docstring)
/home/administrator/Desktop/pylint_test.py:1:0: W0611: Unused import requests (unused-import)

------------------------------------------------------------------
Your code has been rated at 0.00/10 (previous run: 0.00/10, +0.00)
\`\`\`
2. Now run the same command but with `-f json` to export it to JSON:
\`\`\`
pylint ~/Desktop/pylint_test.py  -f json
\`\`\`
The output doesn't contain the scores now anymore:
\`\`\`
[
    {
        "type": "convention",
        "module": "pylint_test",
        "obj": "",
        "line": 1,
        "column": 0,
        "path": "/home/administrator/Desktop/pylint_test.py",
        "symbol": "missing-docstring",
        "message": "Missing module docstring",
        "message-id": "C0111"
    },
    {
        "type": "warning",
        "module": "pylint_test",
        "obj": "",
        "line": 1,
        "column": 0,
        "path": "/home/administrator/Desktop/pylint_test.py",
        "symbol": "unused-import",
        "message": "Unused import requests",
        "message-id": "W0611"
    }
]
\`\`\`

3. Now execute it with `-f json` again but also supply the `--score=y` option:
\`\`\`
[
    {
        "type": "convention",
        "module": "pylint_test",
        "obj": "",
        "line": 1,
        "column": 0,
        "path": "/home/administrator/Desktop/pylint_test.py",
        "symbol": "missing-docstring",
        "message": "Missing module docstring",
        "message-id": "C0111"
    },
    {
        "type": "warning",
        "module": "pylint_test",
        "obj": "",
        "line": 1,
        "column": 0,
        "path": "/home/administrator/Desktop/pylint_test.py",
        "symbol": "unused-import",
        "message": "Unused import requests",
        "message-id": "W0611"
    }
]
\`\`\`

### Current behavior
The score is not outputted when exporting to JSON, not even when `--score=y` is activated.

### Expected behavior
The score is added to the JSON, at least when `--score=y` is activated.

### pylint --version output
\`\`\`
pylint 2.3.0
astroid 2.2.0
Python 3.7.5 (default, Nov 20 2019, 09:21:52) 
[GCC 9.2.1 20191008]
\`\`\`



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 pylint/reporters/json_reporter.py** | 7 | 37| 190 | 190 | 793 | 
| 2 | **1 pylint/reporters/json_reporter.py** | 66 | 90| 179 | 369 | 793 | 
| **-> 3 <-** | **1 pylint/reporters/json_reporter.py** | 92 | 116| 200 | 569 | 793 | 
| 4 | 2 pylint/lint/pylinter.py | 1111 | 1146| 358 | 927 | 11038 | 
| 5 | 2 pylint/lint/pylinter.py | 101 | 252| 1165 | 2092 | 11038 | 
| 6 | 3 pylint/lint/utils.py | 5 | 104| 498 | 2590 | 11905 | 
| **-> 7 <-** | **3 pylint/reporters/json_reporter.py** | 40 | 63| 163 | 2753 | 11905 | 
| 8 | 4 pylint/checkers/raw_metrics.py | 5 | 40| 337 | 3090 | 12840 | 
| 9 | 5 pylint/checkers/logging.py | 7 | 100| 741 | 3831 | 16172 | 
| 10 | 6 pylint/checkers/imports.py | 7 | 81| 558 | 4389 | 25164 | 
| 11 | 7 pylint/constants.py | 5 | 120| 841 | 5230 | 27367 | 
| **-> 12 <-** | **8 pylint/testutils/_primer/primer_run_command.py** | 5 | 26| 127 | 5357 | 28372 | 
| 13 | 9 pylint/checkers/format.py | 14 | 114| 719 | 6076 | 34293 | 
| 14 | 9 pylint/lint/pylinter.py | 5 | 100| 627 | 6703 | 34293 | 
| 15 | 10 pylint/checkers/base/basic_checker.py | 60 | 976| 463 | 7166 | 42762 | 
| 16 | 11 pylint/message/_deleted_message_ids.py | 5 | 126| 1409 | 8575 | 44662 | 
| 17 | 12 pylint/__init__.py | 5 | 66| 375 | 8950 | 45423 | 
| 18 | 13 pylint/testutils/reporter_for_tests.py | 5 | 80| 466 | 9416 | 45953 | 
| 19 | 14 pylint/lint/report_functions.py | 5 | 23| 135 | 9551 | 46651 | 
| 20 | 15 pylint/utils/linterstats.py | 139 | 156| 133 | 9684 | 49509 | 
| 21 | 15 pylint/lint/report_functions.py | 26 | 42| 134 | 9818 | 49509 | 
| 22 | 15 pylint/utils/linterstats.py | 321 | 389| 675 | 10493 | 49509 | 
| 23 | 16 doc/data/messages/l/logging-unsupported-format/good.py | 1 | 4| 0 | 10493 | 49526 | 
| 24 | 17 pylint/checkers/stdlib.py | 7 | 46| 356 | 10849 | 56826 | 
| 25 | 18 doc/exts/pylint_options.py | 82 | 166| 709 | 11558 | 58681 | 
| 26 | 19 pylint/checkers/unsupported_version.py | 9 | 62| 420 | 11978 | 59351 | 
| 27 | 20 pylint/config/_pylint_config/generate_command.py | 8 | 50| 310 | 12288 | 59739 | 
| 28 | 21 doc/data/messages/l/logging-too-few-args/good.py | 1 | 8| 0 | 12288 | 59771 | 
| 29 | 22 pylint/reporters/text.py | 109 | 144| 342 | 12630 | 61849 | 
| 30 | 23 pylint/lint/caching.py | 30 | 54| 194 | 12824 | 62438 | 
| 31 | 24 pylint/reporters/multi_reporter.py | 5 | 112| 791 | 13615 | 63292 | 
| 32 | 25 doc/data/messages/u/unknown-option-value/good.py | 1 | 2| 0 | 13615 | 63301 | 
| 33 | 26 pylint/lint/__init__.py | 17 | 49| 203 | 13818 | 63632 | 
| 34 | 27 doc/conf.py | 5 | 58| 436 | 14254 | 66328 | 
| 35 | 28 doc/data/messages/u/unrecognized-option/good.py | 1 | 2| 0 | 14254 | 66340 | 
| 36 | 28 pylint/lint/report_functions.py | 45 | 86| 366 | 14620 | 66340 | 
| 37 | 29 pylint/pyreverse/main.py | 7 | 47| 287 | 14907 | 68394 | 
| 38 | 29 pylint/lint/caching.py | 57 | 72| 159 | 15066 | 68394 | 
| 39 | 30 pylint/utils/__init__.py | 9 | 54| 259 | 15325 | 68736 | 
| 40 | 31 pylint/checkers/typecheck.py | 7 | 102| 503 | 15828 | 87300 | 
| 41 | 32 doc/data/messages/t/too-few-format-args/good.py | 1 | 2| 0 | 15828 | 87321 | 
| 42 | 33 doc/data/messages/u/useless-option-value/good.py | 1 | 2| 0 | 15828 | 87348 | 
| 43 | 33 pylint/checkers/stdlib.py | 48 | 100| 633 | 16461 | 87348 | 
| 44 | 34 doc/data/messages/m/missing-format-argument-key/good.py | 1 | 2| 0 | 16461 | 87369 | 
| 45 | 34 pylint/checkers/stdlib.py | 103 | 267| 1248 | 17709 | 87369 | 
| 46 | 34 pylint/reporters/text.py | 169 | 204| 263 | 17972 | 87369 | 
| 47 | 35 pylint/checkers/__init__.py | 43 | 140| 675 | 18647 | 88328 | 
| 48 | 36 pylint/checkers/refactoring/recommendation_checker.py | 5 | 81| 698 | 19345 | 92031 | 
| 49 | 36 doc/conf.py | 59 | 123| 768 | 20113 | 92031 | 
| 50 | 37 pylint/checkers/design_analysis.py | 7 | 91| 703 | 20816 | 96984 | 
| 51 | 38 pylint/checkers/similar.py | 30 | 94| 415 | 21231 | 104773 | 
| 52 | 39 pylint/reporters/reports_handler_mix_in.py | 5 | 20| 114 | 21345 | 105461 | 
| 53 | 40 doc/data/messages/i/invalid-bytes-returned/good.py | 1 | 6| 0 | 21345 | 105491 | 
| 54 | 41 pylint/config/help_formatter.py | 32 | 65| 315 | 21660 | 106047 | 
| 55 | 42 pylint/reporters/collecting_reporter.py | 5 | 29| 109 | 21769 | 106220 | 
| 56 | 43 pylint/testutils/_primer/primer_compare_command.py | 4 | 44| 274 | 22043 | 107677 | 
| 57 | 44 pylint/checkers/base/docstring_checker.py | 7 | 42| 207 | 22250 | 109300 | 
| 58 | 45 pylint/pyreverse/utils.py | 7 | 96| 547 | 22797 | 111192 | 
| 59 | 46 doc/data/messages/m/missing-yield-doc/good.py | 1 | 16| 0 | 22797 | 111265 | 
| 60 | 47 pylint/lint/run.py | 5 | 37| 221 | 23018 | 113240 | 
| 61 | 48 doc/data/messages/l/logging-unsupported-format/bad.py | 1 | 4| 0 | 23018 | 113265 | 
| 62 | 49 pylint/extensions/code_style.py | 5 | 108| 815 | 23833 | 116119 | 
| 63 | 50 pylint/typing.py | 7 | 138| 699 | 24532 | 116889 | 
| 64 | 51 pylint/__pkginfo__.py | 10 | 44| 222 | 24754 | 117194 | 
| 65 | 52 doc/data/messages/l/logging-too-few-args/bad.py | 1 | 8| 0 | 24754 | 117233 | 
| 66 | 53 doc/data/messages/i/invalid-format-returned/good.py | 1 | 6| 0 | 24754 | 117264 | 
| 67 | 54 pylint/checkers/newstyle.py | 7 | 32| 144 | 24898 | 118200 | 
| 68 | 55 pylint/checkers/base_checker.py | 5 | 31| 155 | 25053 | 120281 | 
| 69 | 56 doc/data/messages/m/missing-yield-type-doc/good.py | 1 | 16| 0 | 25053 | 120354 | 
| 70 | 57 doc/data/messages/s/suppressed-message/good.py | 1 | 2| 0 | 25053 | 120372 | 
| 71 | 58 pylint/testutils/output_line.py | 5 | 30| 166 | 25219 | 121315 | 
| 72 | 59 doc/data/messages/r/redundant-yields-doc/good.py | 1 | 11| 0 | 25219 | 121352 | 
| 73 | 60 doc/data/messages/m/missing-format-string-key/good.py | 1 | 8| 0 | 25219 | 121394 | 
| 74 | 61 doc/data/messages/t/too-few-format-args/bad.py | 1 | 2| 0 | 25219 | 121422 | 
| 75 | 62 doc/data/messages/u/unknown-option-value/bad.py | 1 | 2| 0 | 25219 | 121439 | 
| 76 | **63 pylint/lint/base_options.py** | 419 | 596| 1145 | 26364 | 125584 | 
| 77 | 63 doc/conf.py | 124 | 249| 848 | 27212 | 125584 | 
| 78 | 63 pylint/checkers/format.py | 577 | 597| 203 | 27415 | 125584 | 
| 79 | 64 doc/data/messages/m/missing-kwoa/good.py | 1 | 7| 0 | 27415 | 125616 | 
| 80 | 65 doc/data/messages/u/useless-option-value/bad.py | 1 | 4| 0 | 27415 | 125660 | 
| 81 | 66 pylint/checkers/strings.py | 67 | 196| 1222 | 28637 | 134687 | 
| 82 | 66 pylint/lint/run.py | 118 | 243| 994 | 29631 | 134687 | 
| 83 | 66 pylint/checkers/stdlib.py | 270 | 332| 318 | 29949 | 134687 | 
| 84 | 66 pylint/reporters/text.py | 207 | 238| 310 | 30259 | 134687 | 
| 85 | 67 doc/data/messages/t/too-many-return-statements/bad.py | 1 | 17| 125 | 30384 | 134812 | 
| 86 | 68 doc/data/messages/r/raw-checker-failed/good.py | 1 | 2| 0 | 30384 | 134824 | 
| 87 | 69 doc/data/messages/u/useless-suppression/good.py | 1 | 6| 0 | 30384 | 134840 | 
| 88 | 69 pylint/lint/pylinter.py | 1004 | 1024| 176 | 30560 | 134840 | 
| 89 | 70 doc/data/messages/d/dict-iter-missing-items/bad.py | 1 | 4| 0 | 30560 | 134904 | 
| 90 | 71 doc/data/messages/i/invalid-bytes-returned/bad.py | 1 | 6| 0 | 30560 | 134941 | 
| 91 | 72 pylint/config/utils.py | 134 | 211| 728 | 31288 | 137009 | 
| 92 | 73 doc/data/messages/u/unused-argument/good.py | 1 | 3| 0 | 31288 | 137029 | 
| 93 | 73 pylint/checkers/imports.py | 226 | 316| 764 | 32052 | 137029 | 
| 94 | 74 pylint/checkers/misc.py | 7 | 41| 199 | 32251 | 138132 | 
| 95 | 74 pylint/lint/caching.py | 5 | 27| 172 | 32423 | 138132 | 
| 96 | 75 pylint/checkers/base/basic_error_checker.py | 100 | 206| 938 | 33361 | 142880 | 
| 97 | 75 pylint/testutils/output_line.py | 101 | 122| 164 | 33525 | 142880 | 
| 98 | 76 doc/data/messages/m/missing-format-attribute/good.py | 1 | 2| 0 | 33525 | 142889 | 
| 99 | 77 doc/data/messages/l/logging-format-truncated/good.py | 1 | 4| 0 | 33525 | 142906 | 
| 100 | 78 pylint/checkers/variables.py | 137 | 170| 295 | 33820 | 169437 | 
| 101 | 79 doc/data/messages/m/missing-format-argument-key/bad.py | 1 | 2| 0 | 33820 | 169462 | 
| 102 | 80 doc/data/messages/u/unexpected-keyword-arg/good.py | 1 | 6| 0 | 33820 | 169495 | 
| 103 | 81 doc/data/messages/l/logging-too-many-args/good.py | 1 | 8| 0 | 33820 | 169527 | 
| 104 | 82 pylint/checkers/async.py | 7 | 52| 361 | 34181 | 170321 | 
| 105 | 83 doc/data/messages/i/import-outside-toplevel/good.py | 1 | 6| 0 | 34181 | 170335 | 
| 106 | 84 pylint/config/arguments_manager.py | 7 | 45| 208 | 34389 | 173210 | 
| 107 | 85 doc/data/messages/u/unrecognized-inline-option/good.py | 1 | 2| 0 | 34389 | 173221 | 
| 108 | 86 doc/data/messages/u/unused-format-string-key/good.py | 1 | 5| 0 | 34389 | 173253 | 
| 109 | 86 pylint/checkers/stdlib.py | 365 | 484| 1272 | 35661 | 173253 | 
| 110 | 87 doc/data/messages/r/redundant-yields-doc/bad.py | 1 | 10| 0 | 35661 | 173295 | 
| 111 | 88 doc/data/messages/i/inconsistent-return-statements/good.py | 1 | 5| 0 | 35661 | 173320 | 
| 112 | 88 pylint/pyreverse/main.py | 49 | 258| 1309 | 36970 | 173320 | 
| 113 | 88 pylint/checkers/variables.py | 373 | 517| 1323 | 38293 | 173320 | 
| 114 | 89 pylint/testutils/__init__.py | 7 | 36| 228 | 38521 | 173620 | 
| 115 | 90 doc/data/messages/m/missing-yield-doc/bad.py | 1 | 10| 0 | 38521 | 173684 | 
| 116 | 91 doc/data/messages/y/yield-outside-function/good.py | 1 | 4| 0 | 38521 | 173701 | 
| 117 | 91 pylint/checkers/strings.py | 7 | 65| 379 | 38900 | 173701 | 
| 118 | 92 doc/data/messages/u/unused-import/good.py | 1 | 4| 0 | 38900 | 173712 | 
| 119 | 93 doc/data/messages/t/too-many-format-args/good.py | 1 | 2| 0 | 38900 | 173733 | 
| 120 | 93 pylint/lint/pylinter.py | 288 | 364| 700 | 39600 | 173733 | 
| 121 | 94 pylint/testutils/constants.py | 5 | 30| 280 | 39880 | 174076 | 
| 122 | 94 pylint/utils/linterstats.py | 231 | 318| 748 | 40628 | 174076 | 
| 123 | 95 doc/data/messages/n/no-value-for-parameter/good.py | 1 | 6| 0 | 40628 | 174095 | 
| 124 | 95 pylint/lint/pylinter.py | 255 | 286| 284 | 40912 | 174095 | 
| 125 | 96 doc/data/messages/s/suppressed-message/bad.py | 1 | 10| 0 | 40912 | 174194 | 
| 126 | 96 pylint/checkers/imports.py | 1001 | 1030| 293 | 41205 | 174194 | 
| 127 | 97 pylint/extensions/typing.py | 81 | 159| 765 | 41970 | 178679 | 
| 128 | 98 doc/data/messages/n/non-iterator-returned/good.py | 1 | 26| 205 | 42175 | 178884 | 
| 129 | 99 doc/data/messages/m/missing-yield-type-doc/bad.py | 1 | 13| 0 | 42175 | 178957 | 
| 130 | 100 doc/data/messages/u/unused-format-string-argument/good.py | 1 | 4| 0 | 42175 | 178999 | 
| 131 | 100 doc/conf.py | 251 | 308| 377 | 42552 | 178999 | 
| 132 | 101 pylint/reporters/ureports/__init__.py | 1 | 8| 0 | 42552 | 179084 | 
| 133 | 102 doc/data/messages/u/unexpected-special-method-signature/good.py | 1 | 7| 0 | 42552 | 179114 | 
| 134 | 103 doc/data/messages/y/yield-inside-async-function/good.py | 1 | 8| 0 | 42552 | 179146 | 
| 135 | 104 doc/data/messages/n/non-iterator-returned/bad.py | 1 | 19| 149 | 42701 | 179295 | 
| 136 | 105 doc/data/messages/i/invalid-repr-returned/good.py | 1 | 6| 0 | 42701 | 179324 | 
| 137 | 105 pylint/checkers/typecheck.py | 831 | 983| 1076 | 43777 | 179324 | 
| 138 | 106 pylint/pyreverse/dot_printer.py | 7 | 51| 330 | 44107 | 180821 | 
| 139 | 107 doc/data/messages/u/useless-return/good.py | 1 | 6| 0 | 44107 | 180834 | 
| 140 | 107 pylint/checkers/logging.py | 127 | 155| 182 | 44289 | 180834 | 
| 141 | 108 doc/data/messages/i/invalid-format-returned/bad.py | 1 | 6| 0 | 44289 | 180873 | 
| 142 | 109 doc/data/messages/i/invalid-getnewargs-returned/good.py | 1 | 6| 0 | 44289 | 180910 | 
| 143 | 110 doc/data/messages/r/raising-format-tuple/good.py | 1 | 2| 0 | 44289 | 180928 | 
| 144 | 111 doc/data/messages/l/logging-not-lazy/good.py | 1 | 8| 0 | 44289 | 180956 | 
| 145 | 112 doc/data/messages/i/invalid-getnewargs-ex-returned/good.py | 1 | 6| 0 | 44289 | 180999 | 
| 146 | 112 pylint/checkers/raw_metrics.py | 43 | 77| 248 | 44537 | 180999 | 
| 147 | 113 doc/data/messages/s/super-with-arguments/good.py | 1 | 8| 0 | 44537 | 181022 | 
| 148 | 114 doc/data/messages/s/super-without-brackets/good.py | 1 | 12| 0 | 44537 | 181070 | 
| 149 | 114 pylint/checkers/base/basic_checker.py | 103 | 269| 1561 | 46098 | 181070 | 
| 150 | 115 doc/data/messages/y/yield-outside-function/bad.py | 1 | 3| 0 | 46098 | 181088 | 
| 151 | 116 doc/data/messages/l/logging-format-interpolation/good.py | 1 | 5| 0 | 46098 | 181106 | 
| 152 | 117 script/bump_changelog.py | 39 | 66| 279 | 46377 | 182089 | 
| 153 | 118 doc/data/messages/m/missing-format-string-key/bad.py | 1 | 8| 0 | 46377 | 182133 | 
| 154 | 119 pylint/checkers/non_ascii_names.py | 13 | 27| 133 | 46510 | 183741 | 
| 155 | 120 doc/data/messages/e/eval-used/good.py | 1 | 4| 0 | 46510 | 183758 | 
| 156 | 121 doc/data/messages/i/inconsistent-return-statements/bad.py | 1 | 4| 0 | 46510 | 183788 | 
| 157 | 122 doc/data/messages/u/useless-suppression/bad.py | 1 | 7| 0 | 46510 | 183823 | 
| 158 | 122 pylint/checkers/format.py | 149 | 257| 646 | 47156 | 183823 | 
| 159 | 123 doc/exts/pylint_features.py | 1 | 50| 278 | 47434 | 184182 | 
| 160 | 124 doc/data/messages/i/invalid-str-returned/good.py | 1 | 6| 0 | 47434 | 184210 | 
| 161 | 125 script/check_newsfragments.py | 9 | 47| 226 | 47660 | 185081 | 
| 162 | 125 pylint/checkers/unsupported_version.py | 64 | 85| 171 | 47831 | 185081 | 
| 163 | 126 doc/data/messages/u/unexpected-keyword-arg/bad.py | 1 | 6| 0 | 47831 | 185127 | 
| 164 | 127 pylint/config/_pylint_config/utils.py | 7 | 32| 178 | 48009 | 186002 | 
| 165 | 127 pylint/lint/pylinter.py | 532 | 569| 308 | 48317 | 186002 | 
| 166 | 128 doc/data/messages/u/unused-wildcard-import/good.py | 1 | 6| 0 | 48317 | 186014 | 
| 167 | 129 doc/data/messages/i/import-error/good.py | 1 | 2| 0 | 48317 | 186019 | 
| 168 | 130 doc/data/messages/i/invalid-format-index/good.py | 1 | 3| 0 | 48317 | 186052 | 


## Missing Patch Files

 * 1: pylint/interfaces.py
 * 2: pylint/lint/base_options.py
 * 3: pylint/reporters/__init__.py
 * 4: pylint/reporters/json_reporter.py
 * 5: pylint/testutils/_primer/primer_run_command.py

### Hint

```
Thank you for the report, I can reproduce this bug. 
I have a fix, but I think this has the potential to break countless continuous integration and annoy a lot of persons, so I'm going to wait for a review by someone else before merging.
The fix is not going to be merged before a major version see https://github.com/PyCQA/pylint/pull/3514#issuecomment-619834791
Ahh that's a pity that it won't come in a minor release :( Is there an estimate on when 3.0 more or less lands?
Yeah, sorry about that. I don't think there is a release date for 3.0.0 yet, @PCManticore might want to correct me though.
Shouldn't you have a branch for your next major release so things like this won't bitrot?
I created a 3.0.0.alpha branch, where it's fixed. Will close if we release alpha version ``3.0.0a0``.
Released in 3.0.0a0.
🥳 thanks a lot @Pierre-Sassoulas!
Reopening because the change was reverted in the 3.0 alpha branch. We can also simply add a new reporter for json directly in 2.x branch and deprecate the other json reporter.
```

## Patch

```diff
diff --git a/pylint/interfaces.py b/pylint/interfaces.py
--- a/pylint/interfaces.py
+++ b/pylint/interfaces.py
@@ -35,3 +35,4 @@ class Confidence(NamedTuple):
 
 CONFIDENCE_LEVELS = [HIGH, CONTROL_FLOW, INFERENCE, INFERENCE_FAILURE, UNDEFINED]
 CONFIDENCE_LEVEL_NAMES = [i.name for i in CONFIDENCE_LEVELS]
+CONFIDENCE_MAP = {i.name: i for i in CONFIDENCE_LEVELS}
diff --git a/pylint/lint/base_options.py b/pylint/lint/base_options.py
--- a/pylint/lint/base_options.py
+++ b/pylint/lint/base_options.py
@@ -102,9 +102,10 @@ def _make_linter_options(linter: PyLinter) -> Options:
                 "metavar": "<format>",
                 "short": "f",
                 "group": "Reports",
-                "help": "Set the output format. Available formats are text,"
-                " parseable, colorized, json and msvs (visual studio)."
-                " You can also give a reporter class, e.g. mypackage.mymodule."
+                "help": "Set the output format. Available formats are: text, "
+                "parseable, colorized, json2 (improved json format), json "
+                "(old json format) and msvs (visual studio). "
+                "You can also give a reporter class, e.g. mypackage.mymodule."
                 "MyReporterClass.",
                 "kwargs": {"linter": linter},
             },
diff --git a/pylint/reporters/__init__.py b/pylint/reporters/__init__.py
--- a/pylint/reporters/__init__.py
+++ b/pylint/reporters/__init__.py
@@ -11,7 +11,7 @@
 from pylint import utils
 from pylint.reporters.base_reporter import BaseReporter
 from pylint.reporters.collecting_reporter import CollectingReporter
-from pylint.reporters.json_reporter import JSONReporter
+from pylint.reporters.json_reporter import JSON2Reporter, JSONReporter
 from pylint.reporters.multi_reporter import MultiReporter
 from pylint.reporters.reports_handler_mix_in import ReportsHandlerMixIn
 
@@ -28,6 +28,7 @@ def initialize(linter: PyLinter) -> None:
     "BaseReporter",
     "ReportsHandlerMixIn",
     "JSONReporter",
+    "JSON2Reporter",
     "CollectingReporter",
     "MultiReporter",
 ]
diff --git a/pylint/reporters/json_reporter.py b/pylint/reporters/json_reporter.py
--- a/pylint/reporters/json_reporter.py
+++ b/pylint/reporters/json_reporter.py
@@ -9,7 +9,7 @@
 import json
 from typing import TYPE_CHECKING, Optional, TypedDict
 
-from pylint.interfaces import UNDEFINED
+from pylint.interfaces import CONFIDENCE_MAP, UNDEFINED
 from pylint.message import Message
 from pylint.reporters.base_reporter import BaseReporter
 from pylint.typing import MessageLocationTuple
@@ -37,8 +37,12 @@
 )
 
 
-class BaseJSONReporter(BaseReporter):
-    """Report messages and layouts in JSON."""
+class JSONReporter(BaseReporter):
+    """Report messages and layouts in JSON.
+
+    Consider using JSON2Reporter instead, as it is superior and this reporter
+    is no longer maintained.
+    """
 
     name = "json"
     extension = "json"
@@ -54,25 +58,6 @@ def display_reports(self, layout: Section) -> None:
     def _display(self, layout: Section) -> None:
         """Do nothing."""
 
-    @staticmethod
-    def serialize(message: Message) -> OldJsonExport:
-        raise NotImplementedError
-
-    @staticmethod
-    def deserialize(message_as_json: OldJsonExport) -> Message:
-        raise NotImplementedError
-
-
-class JSONReporter(BaseJSONReporter):
-
-    """
-    TODO: 3.0: Remove this JSONReporter in favor of the new one handling abs-path
-    and confidence.
-
-    TODO: 3.0: Add a new JSONReporter handling abs-path, confidence and scores.
-    (Ultimately all other breaking change related to json for 3.0).
-    """
-
     @staticmethod
     def serialize(message: Message) -> OldJsonExport:
         return {
@@ -96,7 +81,6 @@ def deserialize(message_as_json: OldJsonExport) -> Message:
             symbol=message_as_json["symbol"],
             msg=message_as_json["message"],
             location=MessageLocationTuple(
-                # TODO: 3.0: Add abs-path and confidence in a new JSONReporter
                 abspath=message_as_json["path"],
                 path=message_as_json["path"],
                 module=message_as_json["module"],
@@ -106,10 +90,112 @@ def deserialize(message_as_json: OldJsonExport) -> Message:
                 end_line=message_as_json["endLine"],
                 end_column=message_as_json["endColumn"],
             ),
-            # TODO: 3.0: Make confidence available in a new JSONReporter
             confidence=UNDEFINED,
         )
 
 
+class JSONMessage(TypedDict):
+    type: str
+    message: str
+    messageId: str
+    symbol: str
+    confidence: str
+    module: str
+    path: str
+    absolutePath: str
+    line: int
+    endLine: int | None
+    column: int
+    endColumn: int | None
+    obj: str
+
+
+class JSON2Reporter(BaseReporter):
+    name = "json2"
+    extension = "json2"
+
+    def display_reports(self, layout: Section) -> None:
+        """Don't do anything in this reporter."""
+
+    def _display(self, layout: Section) -> None:
+        """Do nothing."""
+
+    def display_messages(self, layout: Section | None) -> None:
+        """Launch layouts display."""
+        output = {
+            "messages": [self.serialize(message) for message in self.messages],
+            "statistics": self.serialize_stats(),
+        }
+        print(json.dumps(output, indent=4), file=self.out)
+
+    @staticmethod
+    def serialize(message: Message) -> JSONMessage:
+        return JSONMessage(
+            type=message.category,
+            symbol=message.symbol,
+            message=message.msg or "",
+            messageId=message.msg_id,
+            confidence=message.confidence.name,
+            module=message.module,
+            obj=message.obj,
+            line=message.line,
+            column=message.column,
+            endLine=message.end_line,
+            endColumn=message.end_column,
+            path=message.path,
+            absolutePath=message.abspath,
+        )
+
+    @staticmethod
+    def deserialize(message_as_json: JSONMessage) -> Message:
+        return Message(
+            msg_id=message_as_json["messageId"],
+            symbol=message_as_json["symbol"],
+            msg=message_as_json["message"],
+            location=MessageLocationTuple(
+                abspath=message_as_json["absolutePath"],
+                path=message_as_json["path"],
+                module=message_as_json["module"],
+                obj=message_as_json["obj"],
+                line=message_as_json["line"],
+                column=message_as_json["column"],
+                end_line=message_as_json["endLine"],
+                end_column=message_as_json["endColumn"],
+            ),
+            confidence=CONFIDENCE_MAP[message_as_json["confidence"]],
+        )
+
+    def serialize_stats(self) -> dict[str, str | int | dict[str, int]]:
+        """Serialize the linter stats into something JSON dumpable."""
+        stats = self.linter.stats
+
+        counts_dict = {
+            "fatal": stats.fatal,
+            "error": stats.error,
+            "warning": stats.warning,
+            "refactor": stats.refactor,
+            "convention": stats.convention,
+            "info": stats.info,
+        }
+
+        # Calculate score based on the evaluation option
+        evaluation = self.linter.config.evaluation
+        try:
+            note: int = eval(  # pylint: disable=eval-used
+                evaluation, {}, {**counts_dict, "statement": stats.statement or 1}
+            )
+        except Exception as ex:  # pylint: disable=broad-except
+            score: str | int = f"An exception occurred while rating: {ex}"
+        else:
+            score = round(note, 2)
+
+        return {
+            "messageTypeCount": counts_dict,
+            "modulesLinted": len(stats.by_module),
+            "score": score,
+        }
+
+
 def register(linter: PyLinter) -> None:
     linter.register_reporter(JSONReporter)
+    linter.register_reporter(JSON2Reporter)
diff --git a/pylint/testutils/_primer/primer_run_command.py b/pylint/testutils/_primer/primer_run_command.py
--- a/pylint/testutils/_primer/primer_run_command.py
+++ b/pylint/testutils/_primer/primer_run_command.py
@@ -13,8 +13,7 @@
 
 from pylint.lint import Run
 from pylint.message import Message
-from pylint.reporters import JSONReporter
-from pylint.reporters.json_reporter import OldJsonExport
+from pylint.reporters.json_reporter import JSONReporter, OldJsonExport
 from pylint.testutils._primer.package_to_lint import PackageToLint
 from pylint.testutils._primer.primer_command import (
     PackageData,

```

## Test Patch

```diff
diff --git a/tests/reporters/unittest_json_reporter.py b/tests/reporters/unittest_json_reporter.py
--- a/tests/reporters/unittest_json_reporter.py
+++ b/tests/reporters/unittest_json_reporter.py
@@ -8,15 +8,16 @@
 
 import json
 from io import StringIO
+from pathlib import Path
 from typing import Any
 
 import pytest
 
 from pylint import checkers
-from pylint.interfaces import UNDEFINED
+from pylint.interfaces import HIGH, UNDEFINED
 from pylint.lint import PyLinter
 from pylint.message import Message
-from pylint.reporters import JSONReporter
+from pylint.reporters.json_reporter import JSON2Reporter, JSONReporter
 from pylint.reporters.ureports.nodes import EvaluationSection
 from pylint.typing import MessageLocationTuple
 
@@ -132,6 +133,133 @@ def get_linter_result(score: bool, message: dict[str, Any]) -> list[dict[str, An
     ],
 )
 def test_serialize_deserialize(message: Message) -> None:
-    # TODO: 3.0: Add confidence handling, add path and abs path handling or a new JSONReporter
     json_message = JSONReporter.serialize(message)
     assert message == JSONReporter.deserialize(json_message)
+
+
+def test_simple_json2_output() -> None:
+    """Test JSON2 reporter."""
+    message = {
+        "msg": "line-too-long",
+        "line": 1,
+        "args": (1, 2),
+        "end_line": 1,
+        "end_column": 4,
+    }
+    expected = {
+        "messages": [
+            {
+                "type": "convention",
+                "symbol": "line-too-long",
+                "message": "Line too long (1/2)",
+                "messageId": "C0301",
+                "confidence": "HIGH",
+                "module": "0123",
+                "obj": "",
+                "line": 1,
+                "column": 0,
+                "endLine": 1,
+                "endColumn": 4,
+                "path": "0123",
+                "absolutePath": "0123",
+            }
+        ],
+        "statistics": {
+            "messageTypeCount": {
+                "fatal": 0,
+                "error": 0,
+                "warning": 0,
+                "refactor": 0,
+                "convention": 1,
+                "info": 0,
+            },
+            "modulesLinted": 1,
+            "score": 5.0,
+        },
+    }
+    report = get_linter_result_for_v2(message=message)
+    assert len(report) == 2
+    assert json.dumps(report) == json.dumps(expected)
+
+
+def get_linter_result_for_v2(message: dict[str, Any]) -> list[dict[str, Any]]:
+    output = StringIO()
+    reporter = JSON2Reporter(output)
+    linter = PyLinter(reporter=reporter)
+    checkers.initialize(linter)
+    linter.config.persistent = 0
+    linter.open()
+    linter.set_current_module("0123")
+    linter.add_message(
+        message["msg"],
+        line=message["line"],
+        args=message["args"],
+        end_lineno=message["end_line"],
+        end_col_offset=message["end_column"],
+        confidence=HIGH,
+    )
+    linter.stats.statement = 2
+    reporter.display_messages(None)
+    report_result = json.loads(output.getvalue())
+    return report_result  # type: ignore[no-any-return]
+
+
+@pytest.mark.parametrize(
+    "message",
+    [
+        pytest.param(
+            Message(
+                msg_id="C0111",
+                symbol="missing-docstring",
+                location=MessageLocationTuple(
+                    # The abspath is nonsensical, but should be serialized correctly
+                    abspath=str(Path(__file__).parent),
+                    path=__file__,
+                    module="unittest_json_reporter",
+                    obj="obj",
+                    line=1,
+                    column=3,
+                    end_line=3,
+                    end_column=5,
+                ),
+                msg="This is the actual message",
+                confidence=HIGH,
+            ),
+            id="everything-defined",
+        ),
+        pytest.param(
+            Message(
+                msg_id="C0111",
+                symbol="missing-docstring",
+                location=MessageLocationTuple(
+                    # The abspath is nonsensical, but should be serialized correctly
+                    abspath=str(Path(__file__).parent),
+                    path=__file__,
+                    module="unittest_json_reporter",
+                    obj="obj",
+                    line=1,
+                    column=3,
+                    end_line=None,
+                    end_column=None,
+                ),
+                msg="This is the actual message",
+                confidence=None,
+            ),
+            id="not-everything-defined",
+        ),
+    ],
+)
+def test_serialize_deserialize_for_v2(message: Message) -> None:
+    json_message = JSON2Reporter.serialize(message)
+    assert message == JSON2Reporter.deserialize(json_message)
+
+
+def test_json2_result_with_broken_score() -> None:
+    """Test that the JSON2 reporter can handle broken score evaluations."""
+    output = StringIO()
+    reporter = JSON2Reporter(output)
+    linter = PyLinter(reporter=reporter)
+    linter.config.evaluation = "1/0"
+    reporter.display_messages(None)
+    report_result = json.loads(output.getvalue())
+    assert "division by zero" in report_result["statistics"]["score"]
diff --git a/tests/reporters/unittest_reporting.py b/tests/reporters/unittest_reporting.py
--- a/tests/reporters/unittest_reporting.py
+++ b/tests/reporters/unittest_reporting.py
@@ -176,10 +176,10 @@ def test_multi_format_output(tmp_path: Path) -> None:
 
     source_file = tmp_path / "somemodule.py"
     source_file.write_text('NOT_EMPTY = "This module is not empty"\n')
-    escaped_source_file = dumps(str(source_file))
+    dumps(str(source_file))
 
     nop_format = NopReporter.__module__ + "." + NopReporter.__name__
-    formats = ",".join(["json:" + str(json), "text", nop_format])
+    formats = ",".join(["json2:" + str(json), "text", nop_format])
 
     with redirect_stdout(text):
         linter = PyLinter()
@@ -208,37 +208,7 @@ def test_multi_format_output(tmp_path: Path) -> None:
         del linter.reporter
 
     with open(json, encoding="utf-8") as f:
-        assert (
-            f.read() == "[\n"
-            "    {\n"
-            '        "type": "convention",\n'
-            '        "module": "somemodule",\n'
-            '        "obj": "",\n'
-            '        "line": 1,\n'
-            '        "column": 0,\n'
-            '        "endLine": null,\n'
-            '        "endColumn": null,\n'
-            f'        "path": {escaped_source_file},\n'
-            '        "symbol": "missing-module-docstring",\n'
-            '        "message": "Missing module docstring",\n'
-            '        "message-id": "C0114"\n'
-            "    },\n"
-            "    {\n"
-            '        "type": "convention",\n'
-            '        "module": "somemodule",\n'
-            '        "obj": "",\n'
-            '        "line": 1,\n'
-            '        "column": 0,\n'
-            '        "endLine": null,\n'
-            '        "endColumn": null,\n'
-            f'        "path": {escaped_source_file},\n'
-            '        "symbol": "line-too-long",\n'
-            '        "message": "Line too long (1/2)",\n'
-            '        "message-id": "C0301"\n'
-            "    }\n"
-            "]\n"
-            "direct output\n"
-        )
+        assert '"messageId": "C0114"' in f.read()
 
     assert (
         text.getvalue() == "A NopReporter was initialized.\n"
diff --git a/tests/test_self.py b/tests/test_self.py
--- a/tests/test_self.py
+++ b/tests/test_self.py
@@ -31,7 +31,8 @@
 from pylint.constants import MAIN_CHECKER_NAME, MSG_TYPES_STATUS
 from pylint.lint.pylinter import PyLinter
 from pylint.message import Message
-from pylint.reporters import BaseReporter, JSONReporter
+from pylint.reporters import BaseReporter
+from pylint.reporters.json_reporter import JSON2Reporter
 from pylint.reporters.text import ColorizedTextReporter, TextReporter
 from pylint.testutils._run import _add_rcfile_default_pylintrc
 from pylint.testutils._run import _Run as Run
@@ -187,7 +188,7 @@ def test_all(self) -> None:
         reporters = [
             TextReporter(StringIO()),
             ColorizedTextReporter(StringIO()),
-            JSONReporter(StringIO()),
+            JSON2Reporter(StringIO()),
         ]
         self._runtest(
             [join(HERE, "functional", "a", "arguments.py")],
@@ -347,8 +348,8 @@ def test_reject_empty_indent_strings(self) -> None:
     def test_json_report_when_file_has_syntax_error(self) -> None:
         out = StringIO()
         module = join(HERE, "regrtest_data", "syntax_error.py")
-        self._runtest([module], code=2, reporter=JSONReporter(out))
-        output = json.loads(out.getvalue())
+        self._runtest([module], code=2, reporter=JSON2Reporter(out))
+        output = json.loads(out.getvalue())["messages"]
         assert isinstance(output, list)
         assert len(output) == 1
         assert isinstance(output[0], dict)
@@ -372,8 +373,8 @@ def test_json_report_when_file_has_syntax_error(self) -> None:
     def test_json_report_when_file_is_missing(self) -> None:
         out = StringIO()
         module = join(HERE, "regrtest_data", "totally_missing.py")
-        self._runtest([module], code=1, reporter=JSONReporter(out))
-        output = json.loads(out.getvalue())
+        self._runtest([module], code=1, reporter=JSON2Reporter(out))
+        output = json.loads(out.getvalue())["messages"]
         assert isinstance(output, list)
         assert len(output) == 1
         assert isinstance(output[0], dict)
@@ -394,8 +395,8 @@ def test_json_report_when_file_is_missing(self) -> None:
     def test_json_report_does_not_escape_quotes(self) -> None:
         out = StringIO()
         module = join(HERE, "regrtest_data", "unused_variable.py")
-        self._runtest([module], code=4, reporter=JSONReporter(out))
-        output = json.loads(out.getvalue())
+        self._runtest([module], code=4, reporter=JSON2Reporter(out))
+        output = json.loads(out.getvalue())["messages"]
         assert isinstance(output, list)
         assert len(output) == 1
         assert isinstance(output[0], dict)
@@ -404,7 +405,7 @@ def test_json_report_does_not_escape_quotes(self) -> None:
             "module": "unused_variable",
             "column": 4,
             "message": "Unused variable 'variable'",
-            "message-id": "W0612",
+            "messageId": "W0612",
             "line": 4,
             "type": "warning",
         }
@@ -1066,6 +1067,7 @@ def test_fail_on_info_only_exit_code(self, args: list[str], expected: int) -> No
                 ),
             ),
             ("json", '"message": "Unused variable \'variable\'",'),
+            ("json2", '"message": "Unused variable \'variable\'",'),
         ],
     )
     def test_output_file_can_be_combined_with_output_format_option(

```


## Code snippets

### 1 - pylint/reporters/json_reporter.py:

Start line: 7, End line: 37

```python
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional, TypedDict

from pylint.interfaces import UNDEFINED
from pylint.message import Message
from pylint.reporters.base_reporter import BaseReporter
from pylint.typing import MessageLocationTuple

if TYPE_CHECKING:
    from pylint.lint.pylinter import PyLinter
    from pylint.reporters.ureports.nodes import Section

# Since message-id is an invalid name we need to use the alternative syntax
OldJsonExport = TypedDict(
    "OldJsonExport",
    {
        "type": str,
        "module": str,
        "obj": str,
        "line": int,
        "column": int,
        "endLine": Optional[int],
        "endColumn": Optional[int],
        "path": str,
        "symbol": str,
        "message": str,
        "message-id": str,
    },
)
```
### 2 - pylint/reporters/json_reporter.py:

Start line: 66, End line: 90

```python
class JSONReporter(BaseJSONReporter):

    """
    TODO: 3.0: Remove this JSONReporter in favor of the new one handling abs-path
    and confidence.

    TODO: 3.0: Add a new JSONReporter handling abs-path, confidence and scores.
    (Ultimately all other breaking change related to json for 3.0).
    """

    @staticmethod
    def serialize(message: Message) -> OldJsonExport:
        return {
            "type": message.category,
            "module": message.module,
            "obj": message.obj,
            "line": message.line,
            "column": message.column,
            "endLine": message.end_line,
            "endColumn": message.end_column,
            "path": message.path,
            "symbol": message.symbol,
            "message": message.msg or "",
            "message-id": message.msg_id,
        }
```
### 3 - pylint/reporters/json_reporter.py:

Start line: 92, End line: 116

```python
class JSONReporter(BaseJSONReporter):

    @staticmethod
    def deserialize(message_as_json: OldJsonExport) -> Message:
        return Message(
            msg_id=message_as_json["message-id"],
            symbol=message_as_json["symbol"],
            msg=message_as_json["message"],
            location=MessageLocationTuple(
                # TODO: 3.0: Add abs-path and confidence in a new JSONReporter
                abspath=message_as_json["path"],
                path=message_as_json["path"],
                module=message_as_json["module"],
                obj=message_as_json["obj"],
                line=message_as_json["line"],
                column=message_as_json["column"],
                end_line=message_as_json["endLine"],
                end_column=message_as_json["endColumn"],
            ),
            # TODO: 3.0: Make confidence available in a new JSONReporter
            confidence=UNDEFINED,
        )


def register(linter: PyLinter) -> None:
    linter.register_reporter(JSONReporter)
```
### 4 - pylint/lint/pylinter.py:

Start line: 1111, End line: 1146

```python
class PyLinter(
    _ArgumentsManager,
    _MessageStateHandler,
    reporters.ReportsHandlerMixIn,
    checkers.BaseChecker,
):

    def _report_evaluation(self) -> int | None:
        """Make the global evaluation report."""
        # check with at least a statement (usually 0 when there is a
        # syntax error preventing pylint from further processing)
        note = None
        previous_stats = load_results(self.file_state.base_name)
        if self.stats.statement == 0:
            return note

        # get a global note for the code
        evaluation = self.config.evaluation
        try:
            stats_dict = {
                "fatal": self.stats.fatal,
                "error": self.stats.error,
                "warning": self.stats.warning,
                "refactor": self.stats.refactor,
                "convention": self.stats.convention,
                "statement": self.stats.statement,
                "info": self.stats.info,
            }
            note = eval(evaluation, {}, stats_dict)  # pylint: disable=eval-used
        except Exception as ex:  # pylint: disable=broad-except
            msg = f"An exception occurred while rating: {ex}"
        else:
            self.stats.global_note = note
            msg = f"Your code has been rated at {note:.2f}/10"
            if previous_stats:
                pnote = previous_stats.global_note
                if pnote is not None:
                    msg += f" (previous run: {pnote:.2f}/10, {note - pnote:+.2f})"

        if self.config.score:
            sect = report_nodes.EvaluationSection(msg)
            self.reporter.display_reports(sect)
        return note
```
### 5 - pylint/lint/pylinter.py:

Start line: 101, End line: 252

```python
MSGS: dict[str, MessageDefinitionTuple] = {
    "F0001": (
        "%s",
        "fatal",
        "Used when an error occurred preventing the analysis of a \
              module (unable to find it for instance).",
        {"scope": WarningScope.LINE},
    ),
    "F0002": (
        "%s: %s",
        "astroid-error",
        "Used when an unexpected error occurred while building the "
        "Astroid  representation. This is usually accompanied by a "
        "traceback. Please report such errors !",
        {"scope": WarningScope.LINE},
    ),
    "F0010": (
        "error while code parsing: %s",
        "parse-error",
        "Used when an exception occurred while building the Astroid "
        "representation which could be handled by astroid.",
        {"scope": WarningScope.LINE},
    ),
    "F0011": (
        "error while parsing the configuration: %s",
        "config-parse-error",
        "Used when an exception occurred while parsing a pylint configuration file.",
        {"scope": WarningScope.LINE},
    ),
    "I0001": (
        "Unable to run raw checkers on built-in module %s",
        "raw-checker-failed",
        "Used to inform that a built-in module has not been checked "
        "using the raw checkers.",
        {
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "I0010": (
        "Unable to consider inline option %r",
        "bad-inline-option",
        "Used when an inline option is either badly formatted or can't "
        "be used inside modules.",
        {
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "I0011": (
        "Locally disabling %s (%s)",
        "locally-disabled",
        "Used when an inline option disables a message or a messages category.",
        {
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "I0013": (
        "Ignoring entire file",
        "file-ignored",
        "Used to inform that the file will not be checked",
        {
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "I0020": (
        "Suppressed %s (from line %d)",
        "suppressed-message",
        "A message was triggered on a line, but suppressed explicitly "
        "by a disable= comment in the file. This message is not "
        "generated for messages that are ignored due to configuration "
        "settings.",
        {
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "I0021": (
        "Useless suppression of %s",
        "useless-suppression",
        "Reported when a message is explicitly disabled for a line or "
        "a block of code, but never triggered.",
        {
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "I0022": (
        'Pragma "%s" is deprecated, use "%s" instead',
        "deprecated-pragma",
        "Some inline pylint options have been renamed or reworked, "
        "only the most recent form should be used. "
        "NOTE:skip-all is only available with pylint >= 0.26",
        {
            "old_names": [("I0014", "deprecated-disable-all")],
            "scope": WarningScope.LINE,
            "default_enabled": False,
        },
    ),
    "E0001": (
        "%s",
        "syntax-error",
        "Used when a syntax error is raised for a module.",
        {"scope": WarningScope.LINE},
    ),
    "E0011": (
        "Unrecognized file option %r",
        "unrecognized-inline-option",
        "Used when an unknown inline option is encountered.",
        {"scope": WarningScope.LINE},
    ),
    "W0012": (
        "Unknown option value for '%s', expected a valid pylint message and got '%s'",
        "unknown-option-value",
        "Used when an unknown value is encountered for an option.",
        {
            "scope": WarningScope.LINE,
            "old_names": [("E0012", "bad-option-value")],
        },
    ),
    "R0022": (
        "Useless option value for '%s', %s",
        "useless-option-value",
        "Used when a value for an option that is now deleted from pylint"
        " is encountered.",
        {
            "scope": WarningScope.LINE,
            "old_names": [("E0012", "bad-option-value")],
        },
    ),
    "E0013": (
        "Plugin '%s' is impossible to load, is it installed ? ('%s')",
        "bad-plugin-value",
        "Used when a bad value is used in 'load-plugins'.",
        {"scope": WarningScope.LINE},
    ),
    "E0014": (
        "Out-of-place setting encountered in top level configuration-section '%s' : '%s'",
        "bad-configuration-section",
        "Used when we detect a setting in the top level of a toml configuration that"
        " shouldn't be there.",
        {"scope": WarningScope.LINE},
    ),
    "E0015": (
        "Unrecognized option found: %s",
        "unrecognized-option",
        "Used when we detect an option that we do not recognize.",
        {"scope": WarningScope.LINE},
    ),
}
```
### 6 - pylint/lint/utils.py:

Start line: 5, End line: 104

```python
from __future__ import annotations

import contextlib
import platform
import sys
import traceback
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

from pylint.constants import PYLINT_HOME, full_version


def prepare_crash_report(ex: Exception, filepath: str, crash_file_path: str) -> Path:
    issue_template_path = (
        Path(PYLINT_HOME) / datetime.now().strftime(str(crash_file_path))
    ).resolve()
    with open(filepath, encoding="utf8") as f:
        file_content = f.read()
    template = ""
    if not issue_template_path.exists():
        template = """\
First, please verify that the bug is not already filled:
https://github.com/pylint-dev/pylint/issues/

Then create a new issue:
https://github.com/pylint-dev/pylint/issues/new?labels=Crash 💥%2CNeeds triage 📥


"""
    template += f"""
    e title:
    h ``{ex}`` (if possible, be more specific about what made pylint crash)

    Bug description

     parsing the following ``a.py``:


    sharing the code is not an option, please state so,
     providing only the stacktrace would still be helpful.


    ython
    e_content}


    Command used

    hell
    nt a.py


    Pylint output

    ails open>
    <summary>
        pylint crashed with a ``{ex.__class__.__name__}`` and with the following stacktrace:
    </summary>

    ython

    template += traceback.format_exc()
    template += f"""



    tails>

    Expected behavior

    rash.

    Pylint version

    hell
    l_version}


    OS / Environment

    .platform} ({platform.system()})

    Additional dependencies


    se remove this part if you're not using any of
     dependencies in the example.


    try:
        with open(issue_template_path, "a", encoding="utf8") as f:
            f.write(template)
    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"Can't write the issue template for the crash in {issue_template_path} "
            f"because of: '{exc}'\nHere's the content anyway:\n{template}.",
            file=sys.stderr,
        )
    return issue_template_path
```
### 7 - pylint/reporters/json_reporter.py:

Start line: 40, End line: 63

```python
class BaseJSONReporter(BaseReporter):
    """Report messages and layouts in JSON."""

    name = "json"
    extension = "json"

    def display_messages(self, layout: Section | None) -> None:
        """Launch layouts display."""
        json_dumpable = [self.serialize(message) for message in self.messages]
        print(json.dumps(json_dumpable, indent=4), file=self.out)

    def display_reports(self, layout: Section) -> None:
        """Don't do anything in this reporter."""

    def _display(self, layout: Section) -> None:
        """Do nothing."""

    @staticmethod
    def serialize(message: Message) -> OldJsonExport:
        raise NotImplementedError

    @staticmethod
    def deserialize(message_as_json: OldJsonExport) -> Message:
        raise NotImplementedError
```
### 8 - pylint/checkers/raw_metrics.py:

Start line: 5, End line: 40

```python
from __future__ import annotations

import tokenize
from typing import TYPE_CHECKING, Any, Literal, cast

from pylint.checkers import BaseTokenChecker
from pylint.reporters.ureports.nodes import Paragraph, Section, Table, Text
from pylint.utils import LinterStats, diff_string

if TYPE_CHECKING:
    from pylint.lint import PyLinter


def report_raw_stats(
    sect: Section,
    stats: LinterStats,
    old_stats: LinterStats | None,
) -> None:
    """Calculate percentage of code / doc / comment / empty."""
    total_lines = stats.code_type_count["total"]
    sect.insert(0, Paragraph([Text(f"{total_lines} lines have been analyzed\n")]))
    lines = ["type", "number", "%", "previous", "difference"]
    for node_type in ("code", "docstring", "comment", "empty"):
        node_type = cast(Literal["code", "docstring", "comment", "empty"], node_type)
        total = stats.code_type_count[node_type]
        percent = float(total * 100) / total_lines if total_lines else None
        old = old_stats.code_type_count[node_type] if old_stats else None
        diff_str = diff_string(old, total) if old else None
        lines += [
            node_type,
            str(total),
            f"{percent:.2f}" if percent is not None else "NC",
            str(old) if old else "NC",
            diff_str if diff_str else "NC",
        ]
    sect.append(Table(children=lines, cols=5, rheaders=1))
```
### 9 - pylint/checkers/logging.py:

Start line: 7, End line: 100

```python
from __future__ import annotations

import string
from typing import TYPE_CHECKING, Literal

import astroid
from astroid import bases, nodes
from astroid.typing import InferenceResult

from pylint import checkers
from pylint.checkers import utils
from pylint.checkers.utils import infer_all
from pylint.typing import MessageDefinitionTuple

if TYPE_CHECKING:
    from pylint.lint import PyLinter

MSGS: dict[
    str, MessageDefinitionTuple
] = {  # pylint: disable=consider-using-namedtuple-or-dataclass
    "W1201": (
        "Use %s formatting in logging functions",
        "logging-not-lazy",
        "Used when a logging statement has a call form of "
        '"logging.<logging method>(format_string % (format_args...))". '
        "Use another type of string formatting instead. "
        "You can use % formatting but leave interpolation to "
        "the logging function by passing the parameters as arguments. "
        "If logging-fstring-interpolation is disabled then "
        "you can use fstring formatting. "
        "If logging-format-interpolation is disabled then "
        "you can use str.format.",
    ),
    "W1202": (
        "Use %s formatting in logging functions",
        "logging-format-interpolation",
        "Used when a logging statement has a call form of "
        '"logging.<logging method>(format_string.format(format_args...))". '
        "Use another type of string formatting instead. "
        "You can use % formatting but leave interpolation to "
        "the logging function by passing the parameters as arguments. "
        "If logging-fstring-interpolation is disabled then "
        "you can use fstring formatting. "
        "If logging-not-lazy is disabled then "
        "you can use % formatting as normal.",
    ),
    "W1203": (
        "Use %s formatting in logging functions",
        "logging-fstring-interpolation",
        "Used when a logging statement has a call form of "
        '"logging.<logging method>(f"...")".'
        "Use another type of string formatting instead. "
        "You can use % formatting but leave interpolation to "
        "the logging function by passing the parameters as arguments. "
        "If logging-format-interpolation is disabled then "
        "you can use str.format. "
        "If logging-not-lazy is disabled then "
        "you can use % formatting as normal.",
    ),
    "E1200": (
        "Unsupported logging format character %r (%#02x) at index %d",
        "logging-unsupported-format",
        "Used when an unsupported format character is used in a logging "
        "statement format string.",
    ),
    "E1201": (
        "Logging format string ends in middle of conversion specifier",
        "logging-format-truncated",
        "Used when a logging statement format string terminates before "
        "the end of a conversion specifier.",
    ),
    "E1205": (
        "Too many arguments for logging format string",
        "logging-too-many-args",
        "Used when a logging format string is given too many arguments.",
    ),
    "E1206": (
        "Not enough arguments for logging format string",
        "logging-too-few-args",
        "Used when a logging format string is given too few arguments.",
    ),
}


CHECKED_CONVENIENCE_FUNCTIONS = {
    "critical",
    "debug",
    "error",
    "exception",
    "fatal",
    "info",
    "warn",
    "warning",
}
```
### 10 - pylint/checkers/imports.py:

Start line: 7, End line: 81

```python
from __future__ import annotations

import collections
import copy
import os
import sys
from collections import defaultdict
from collections.abc import ItemsView, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Union

import astroid
from astroid import nodes
from astroid.nodes._base_nodes import ImportNode

from pylint.checkers import BaseChecker, DeprecatedMixin
from pylint.checkers.utils import (
    get_import_name,
    in_type_checking_block,
    is_from_fallback_block,
    is_module_ignored,
    is_sys_guard,
    node_ignores_exception,
)
from pylint.exceptions import EmptyReportError
from pylint.graph import DotBackend, get_cycles
from pylint.interfaces import HIGH
from pylint.reporters.ureports.nodes import Paragraph, Section, VerbatimText
from pylint.typing import MessageDefinitionTuple
from pylint.utils import IsortDriver
from pylint.utils.linterstats import LinterStats

if TYPE_CHECKING:
    from pylint.lint import PyLinter


# The dictionary with Any should actually be a _ImportTree again
# but mypy doesn't support recursive types yet
_ImportTree = Dict[str, Union[List[Dict[str, Any]], List[str]]]

DEPRECATED_MODULES = {
    (0, 0, 0): {"tkinter.tix", "fpectl"},
    (3, 2, 0): {"optparse"},
    (3, 3, 0): {"xml.etree.cElementTree"},
    (3, 4, 0): {"imp"},
    (3, 5, 0): {"formatter"},
    (3, 6, 0): {"asynchat", "asyncore", "smtpd"},
    (3, 7, 0): {"macpath"},
    (3, 9, 0): {"lib2to3", "parser", "symbol", "binhex"},
    (3, 10, 0): {"distutils", "typing.io", "typing.re"},
    (3, 11, 0): {
        "aifc",
        "audioop",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "imghdr",
        "msilib",
        "mailcap",
        "nis",
        "nntplib",
        "ossaudiodev",
        "pipes",
        "sndhdr",
        "spwd",
        "sunau",
        "sre_compile",
        "sre_constants",
        "sre_parse",
        "telnetlib",
        "uu",
        "xdrlib",
    },
}
```
### 12 - pylint/testutils/_primer/primer_run_command.py:

Start line: 5, End line: 26

```python
from __future__ import annotations

import json
import sys
import warnings
from io import StringIO

from git.repo import Repo

from pylint.lint import Run
from pylint.message import Message
from pylint.reporters import JSONReporter
from pylint.reporters.json_reporter import OldJsonExport
from pylint.testutils._primer.package_to_lint import PackageToLint
from pylint.testutils._primer.primer_command import (
    PackageData,
    PackageMessages,
    PrimerCommand,
)

GITHUB_CRASH_TEMPLATE_LOCATION = "/home/runner/.cache"
CRASH_TEMPLATE_INTRO = "There is a pre-filled template"
```
### 76 - pylint/lint/base_options.py:

Start line: 419, End line: 596

```python
def _make_run_options(self: Run) -> Options:
    """Return the options used in a Run class."""
    return (
        (
            "rcfile",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "group": "Commands",
                "help": "Specify a configuration file to load.",
                "hide_from_config_file": True,
            },
        ),
        (
            "output",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "group": "Commands",
                "help": "Specify an output file.",
                "hide_from_config_file": True,
            },
        ),
        (
            "init-hook",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "help": "Python code to execute, usually for sys.path "
                "manipulation such as pygtk.require().",
            },
        ),
        (
            "help-msg",
            {
                "action": _MessageHelpAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Display a help message for the given message id and "
                "exit. The value may be a comma separated list of message ids.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-msgs",
            {
                "action": _ListMessagesAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Display a list of all pylint's messages divided by whether "
                "they are emittable with the given interpreter.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-msgs-enabled",
            {
                "action": _ListMessagesEnabledAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Display a list of what messages are enabled, "
                "disabled and non-emittable with the given configuration.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-groups",
            {
                "action": _ListCheckGroupsAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "List pylint's message groups.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-conf-levels",
            {
                "action": _ListConfidenceLevelsAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate pylint's confidence levels.",
                "hide_from_config_file": True,
            },
        ),
        (
            "list-extensions",
            {
                "action": _ListExtensionsAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "List available extensions.",
                "hide_from_config_file": True,
            },
        ),
        (
            "full-documentation",
            {
                "action": _FullDocumentationAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate pylint's full documentation.",
                "hide_from_config_file": True,
            },
        ),
        (
            "generate-rcfile",
            {
                "action": _GenerateRCFileAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate a sample configuration file according to "
                "the current configuration. You can put other options "
                "before this one to get them in the generated "
                "configuration.",
                "hide_from_config_file": True,
            },
        ),
        (
            "generate-toml-config",
            {
                "action": _GenerateConfigFileAction,
                "kwargs": {"Run": self},
                "group": "Commands",
                "help": "Generate a sample configuration file according to "
                "the current configuration. You can put other options "
                "before this one to get them in the generated "
                "configuration. The config is in the .toml format.",
                "hide_from_config_file": True,
            },
        ),
        (
            "errors-only",
            {
                "action": _ErrorsOnlyModeAction,
                "kwargs": {"Run": self},
                "short": "E",
                "help": "In error mode, messages with a category besides "
                "ERROR or FATAL are suppressed, and no reports are done by default. "
                "Error mode is compatible with disabling specific errors. ",
                "hide_from_config_file": True,
            },
        ),
        (
            "verbose",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "short": "v",
                "help": "In verbose mode, extra non-checker-related info "
                "will be displayed.",
                "hide_from_config_file": True,
                "metavar": "",
            },
        ),
        (
            "enable-all-extensions",
            {
                "action": _DoNothingAction,
                "kwargs": {},
                "help": "Load and enable all available extensions. "
                "Use --list-extensions to see a list all available extensions.",
                "hide_from_config_file": True,
                "metavar": "",
            },
        ),
        (
            "long-help",
            {
                "action": _LongHelpAction,
                "kwargs": {"Run": self},
                "help": "Show more verbose help.",
                "group": "Commands",
                "hide_from_config_file": True,
            },
        ),
    )
```
