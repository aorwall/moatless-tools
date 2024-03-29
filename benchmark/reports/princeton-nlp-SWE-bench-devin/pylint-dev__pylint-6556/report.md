# pylint-dev__pylint-6556

| **pylint-dev/pylint** | `fa183c7d15b5f3c7dd8dee86fc74caae42c3926c` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 3 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -937,8 +937,6 @@ def _check_astroid_module(
             self.process_tokens(tokens)
             if self._ignore_file:
                 return False
-            # walk ast to collect line numbers
-            self.file_state.collect_block_lines(self.msgs_store, node)
             # run raw and tokens checkers
             for raw_checker in rawcheckers:
                 raw_checker.process_module(node)
diff --git a/pylint/utils/file_state.py b/pylint/utils/file_state.py
--- a/pylint/utils/file_state.py
+++ b/pylint/utils/file_state.py
@@ -74,25 +74,32 @@ def collect_block_lines(
         self, msgs_store: MessageDefinitionStore, module_node: nodes.Module
     ) -> None:
         """Walk the AST to collect block level options line numbers."""
+        warnings.warn(
+            "'collect_block_lines' has been deprecated and will be removed in pylint 3.0.",
+            DeprecationWarning,
+        )
         for msg, lines in self._module_msgs_state.items():
             self._raw_module_msgs_state[msg] = lines.copy()
         orig_state = self._module_msgs_state.copy()
         self._module_msgs_state = {}
         self._suppression_mapping = {}
         self._effective_max_line_number = module_node.tolineno
-        self._collect_block_lines(msgs_store, module_node, orig_state)
+        for msgid, lines in orig_state.items():
+            for msgdef in msgs_store.get_message_definitions(msgid):
+                self._set_state_on_block_lines(msgs_store, module_node, msgdef, lines)
 
-    def _collect_block_lines(
+    def _set_state_on_block_lines(
         self,
         msgs_store: MessageDefinitionStore,
         node: nodes.NodeNG,
-        msg_state: MessageStateDict,
+        msg: MessageDefinition,
+        msg_state: dict[int, bool],
     ) -> None:
         """Recursively walk (depth first) AST to collect block level options
-        line numbers.
+        line numbers and set the state correctly.
         """
         for child in node.get_children():
-            self._collect_block_lines(msgs_store, child, msg_state)
+            self._set_state_on_block_lines(msgs_store, child, msg, msg_state)
         # first child line number used to distinguish between disable
         # which are the first child of scoped node with those defined later.
         # For instance in the code below:
@@ -115,9 +122,7 @@ def _collect_block_lines(
             firstchildlineno = node.body[0].fromlineno
         else:
             firstchildlineno = node.tolineno
-        for msgid, lines in msg_state.items():
-            for msg in msgs_store.get_message_definitions(msgid):
-                self._set_message_state_in_block(msg, lines, node, firstchildlineno)
+        self._set_message_state_in_block(msg, msg_state, node, firstchildlineno)
 
     def _set_message_state_in_block(
         self,
@@ -139,18 +144,61 @@ def _set_message_state_in_block(
                 if lineno > firstchildlineno:
                     state = True
                 first_, last_ = node.block_range(lineno)
+                # pylint: disable=useless-suppression
+                # For block nodes first_ is their definition line. For example, we
+                # set the state of line zero for a module to allow disabling
+                # invalid-name for the module. For example:
+                # 1. # pylint: disable=invalid-name
+                # 2. ...
+                # OR
+                # 1. """Module docstring"""
+                # 2. # pylint: disable=invalid-name
+                # 3. ...
+                #
+                # But if we already visited line 0 we don't need to set its state again
+                # 1. # pylint: disable=invalid-name
+                # 2. # pylint: enable=invalid-name
+                # 3. ...
+                # The state should come from line 1, not from line 2
+                # Therefore, if the 'fromlineno' is already in the states we just start
+                # with the lineno we were originally visiting.
+                # pylint: enable=useless-suppression
+                if (
+                    first_ == node.fromlineno
+                    and first_ >= firstchildlineno
+                    and node.fromlineno in self._module_msgs_state.get(msg.msgid, ())
+                ):
+                    first_ = lineno
+
             else:
                 first_ = lineno
                 last_ = last
             for line in range(first_, last_ + 1):
-                # do not override existing entries
-                if line in self._module_msgs_state.get(msg.msgid, ()):
+                # Do not override existing entries. This is especially important
+                # when parsing the states for a scoped node where some line-disables
+                # have already been parsed.
+                if (
+                    (
+                        isinstance(node, nodes.Module)
+                        and node.fromlineno <= line < lineno
+                    )
+                    or (
+                        not isinstance(node, nodes.Module)
+                        and node.fromlineno < line < lineno
+                    )
+                ) and line in self._module_msgs_state.get(msg.msgid, ()):
                     continue
                 if line in lines:  # state change in the same block
                     state = lines[line]
                     original_lineno = line
+
+                # Update suppression mapping
                 if not state:
                     self._suppression_mapping[(msg.msgid, line)] = original_lineno
+                else:
+                    self._suppression_mapping.pop((msg.msgid, line), None)
+
+                # Update message state for respective line
                 try:
                     self._module_msgs_state[msg.msgid][line] = state
                 except KeyError:
@@ -160,10 +208,20 @@ def _set_message_state_in_block(
     def set_msg_status(self, msg: MessageDefinition, line: int, status: bool) -> None:
         """Set status (enabled/disable) for a given message at a given line."""
         assert line > 0
+        assert self._module
+        # TODO: 3.0: Remove unnecessary assertion
+        assert self._msgs_store
+
+        # Expand the status to cover all relevant block lines
+        self._set_state_on_block_lines(
+            self._msgs_store, self._module, msg, {line: status}
+        )
+
+        # Store the raw value
         try:
-            self._module_msgs_state[msg.msgid][line] = status
+            self._raw_module_msgs_state[msg.msgid][line] = status
         except KeyError:
-            self._module_msgs_state[msg.msgid] = {line: status}
+            self._raw_module_msgs_state[msg.msgid] = {line: status}
 
     def handle_ignored_message(
         self, state_scope: Literal[0, 1, 2] | None, msgid: str, line: int | None

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/lint/pylinter.py | 940 | 941 | - | 3 | -
| pylint/utils/file_state.py | 77 | 89 | - | 95 | -
| pylint/utils/file_state.py | 118 | 120 | - | 95 | -
| pylint/utils/file_state.py | 142 | 143 | - | 95 | -
| pylint/utils/file_state.py | 163 | 165 | - | 95 | -


## Problem Statement

```
Can't disable bad-option-value
### Steps to reproduce
1. Write code on a computer with a somewhat new pylint (2.4.3 in my example). Get a warning like `useless-object-inheritance` that I want to ignore, as I'm writing code compatible with python2 and python3.
2. Disable said warning with `# pylint: disable=useless-object-inheritance`.
3. Get a "Bad option value" when other people run their pylint version (example: 2.3.1; and by people, sometimes I mean docker instances ran from Jenkins that I would rather not rebuild or that depend on other people and I can't modify)
4. Try to disable said error with a global `# pylint: disable=bad-option-value`

### Current behavior
`# pylint: disable=bad-option-value` is ignored
`# pylint: disable=E0012` is ignored

### Expected behavior
To be able to write code that works on several versions of pylint and not having to make sure every computer in the company and every docker container has the same pylint version.



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 doc/data/messages/b/bad-option-value/bad.py | 1 | 3| 0 | 0 | 21 | 
| 2 | 2 doc/data/messages/b/bad-option-value/good.py | 1 | 3| 0 | 0 | 36 | 
| 3 | **3 pylint/lint/pylinter.py** | 97 | 215| 982 | 982 | 9528 | 
| 4 | 4 pylint/config/option_parser.py | 5 | 54| 371 | 1353 | 9965 | 
| 5 | 5 pylint/config/option.py | 204 | 219| 177 | 1530 | 11709 | 
| 6 | 6 pylint/constants.py | 92 | 199| 1388 | 2918 | 14363 | 
| 7 | 7 pylint/extensions/bad_builtin.py | 25 | 50| 166 | 3084 | 14874 | 
| 8 | 7 pylint/config/option.py | 149 | 182| 313 | 3397 | 14874 | 
| 9 | 8 pylint/config/arguments_provider.py | 7 | 28| 145 | 3542 | 16517 | 
| 10 | 8 pylint/config/arguments_provider.py | 115 | 133| 212 | 3754 | 16517 | 
| 11 | 8 pylint/config/option.py | 184 | 202| 179 | 3933 | 16517 | 
| 12 | 9 doc/data/messages/d/dict-iter-missing-items/bad.py | 1 | 4| 0 | 3933 | 16581 | 
| 13 | 10 pylint/checkers/stdlib.py | 332 | 433| 1075 | 5008 | 22730 | 
| 14 | 11 pylint/config/arguments_manager.py | 647 | 655| 110 | 5118 | 28716 | 
| 15 | 12 doc/data/messages/u/undefined-variable/bad.py | 1 | 2| 0 | 5118 | 28728 | 
| 16 | 12 pylint/config/option.py | 5 | 55| 305 | 5423 | 28728 | 
| 17 | 13 pylint/config/options_provider_mixin.py | 5 | 53| 346 | 5769 | 29696 | 
| 18 | 14 doc/data/messages/d/duplicate-value/bad.py | 1 | 2| 0 | 5769 | 29718 | 
| 19 | 15 doc/data/messages/c/consider-using-with/bad.py | 1 | 4| 0 | 5769 | 29748 | 
| 20 | 15 pylint/config/option.py | 58 | 68| 114 | 5883 | 29748 | 
| 21 | 16 pylint/config/callback_actions.py | 368 | 385| 149 | 6032 | 32367 | 
| 22 | 16 pylint/extensions/bad_builtin.py | 7 | 22| 107 | 6139 | 32367 | 
| 23 | 17 doc/data/messages/u/undefined-all-variable/bad.py | 1 | 5| 0 | 6139 | 32393 | 
| 24 | 18 pylint/config/__init__.py | 5 | 37| 273 | 6412 | 32966 | 
| 25 | 18 pylint/constants.py | 200 | 213| 122 | 6534 | 32966 | 
| 26 | 19 doc/data/messages/u/unreachable/bad.py | 1 | 4| 0 | 6534 | 32989 | 
| 27 | 19 pylint/constants.py | 216 | 269| 469 | 7003 | 32989 | 
| 28 | 19 pylint/config/arguments_provider.py | 180 | 195| 137 | 7140 | 32989 | 
| 29 | 20 doc/data/messages/b/bad-builtin/bad.py | 1 | 3| 0 | 7140 | 33020 | 
| 30 | 21 doc/data/messages/u/undefined-variable/good.py | 1 | 3| 0 | 7140 | 33031 | 
| 31 | 21 pylint/checkers/stdlib.py | 43 | 89| 556 | 7696 | 33031 | 
| 32 | 22 doc/data/messages/o/overridden-final-method/bad.py | 1 | 13| 0 | 7696 | 33079 | 
| 33 | 23 doc/data/messages/u/unnecessary-lambda-assignment/bad.py | 1 | 2| 0 | 7696 | 33104 | 
| 34 | 24 pylint/checkers/format.py | 597 | 617| 197 | 7893 | 39073 | 
| 35 | 24 pylint/constants.py | 5 | 90| 608 | 8501 | 39073 | 
| 36 | 25 pylint/lint/message_state_handler.py | 170 | 200| 220 | 8721 | 42370 | 
| 37 | 26 doc/data/messages/d/duplicate-value/good.py | 1 | 2| 0 | 8721 | 42383 | 
| 38 | 27 doc/data/messages/u/undefined-all-variable/good.py | 1 | 5| 0 | 8721 | 42402 | 
| 39 | 28 doc/data/messages/u/unnecessary-dunder-call/bad.py | 1 | 7| 0 | 8721 | 42465 | 
| 40 | 28 pylint/extensions/bad_builtin.py | 52 | 67| 173 | 8894 | 42465 | 
| 41 | 29 doc/data/messages/g/global-at-module-level/bad.py | 1 | 3| 0 | 8894 | 42480 | 
| 42 | 30 doc/data/messages/b/bad-builtin/good.py | 1 | 3| 0 | 8894 | 42501 | 
| 43 | 31 pylint/utils/utils.py | 265 | 282| 171 | 9065 | 45596 | 
| 44 | 32 doc/data/messages/c/consider-using-sys-exit/bad.py | 1 | 5| 0 | 9065 | 45638 | 
| 45 | 33 doc/data/messages/u/unnecessary-direct-lambda-call/bad.py | 1 | 2| 0 | 9065 | 45667 | 
| 46 | 34 pylint/checkers/unsupported_version.py | 64 | 85| 171 | 9236 | 46339 | 
| 47 | 35 pylint/checkers/base/basic_error_checker.py | 99 | 207| 952 | 10188 | 50855 | 
| 48 | 35 pylint/config/option.py | 71 | 146| 601 | 10789 | 50855 | 
| 49 | 35 pylint/checkers/stdlib.py | 719 | 750| 243 | 11032 | 50855 | 
| 50 | 36 pylint/checkers/variables.py | 360 | 499| 1243 | 12275 | 73162 | 
| 51 | 37 doc/data/messages/a/arguments-out-of-order/bad.py | 1 | 14| 0 | 12275 | 73246 | 
| 52 | 38 doc/data/messages/u/useless-return/bad.py | 1 | 7| 0 | 12275 | 73270 | 
| 53 | 39 pylint/lint/run.py | 5 | 27| 148 | 12423 | 74784 | 
| 54 | 39 pylint/checkers/unsupported_version.py | 9 | 62| 420 | 12843 | 74784 | 
| 55 | 40 doc/data/messages/g/global-statement/bad.py | 1 | 12| 0 | 12843 | 74817 | 
| 56 | 41 pylint/config/option_manager_mixin.py | 7 | 51| 287 | 13130 | 77593 | 
| 57 | 42 doc/data/messages/m/missing-format-argument-key/bad.py | 1 | 2| 0 | 13130 | 77618 | 
| 58 | 42 pylint/config/arguments_manager.py | 357 | 375| 181 | 13311 | 77618 | 
| 59 | 43 doc/data/messages/a/arguments-renamed/bad.py | 1 | 14| 0 | 13311 | 77720 | 
| 60 | 44 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 13311 | 77743 | 
| 61 | 45 doc/data/messages/u/unidiomatic-typecheck/bad.py | 1 | 4| 0 | 13311 | 77779 | 
| 62 | 46 pylint/checkers/exceptions.py | 58 | 163| 1015 | 14326 | 82284 | 
| 63 | 47 pylint/lint/base_options.py | 7 | 392| 188 | 14514 | 86218 | 
| 64 | 48 doc/data/messages/c/consider-using-with/good.py | 1 | 3| 0 | 14514 | 86241 | 
| 65 | 48 pylint/config/arguments_manager.py | 7 | 56| 280 | 14794 | 86241 | 
| 66 | 49 doc/data/messages/b/binary-op-exception/bad.py | 1 | 5| 0 | 14794 | 86268 | 
| 67 | 50 doc/data/messages/u/unreachable/good.py | 1 | 4| 0 | 14794 | 86285 | 
| 68 | 51 doc/data/messages/u/use-dict-literal/bad.py | 1 | 2| 0 | 14794 | 86299 | 
| 69 | 52 doc/data/messages/d/duplicate-argument-name/bad.py | 1 | 3| 0 | 14794 | 86321 | 
| 70 | 53 doc/data/messages/n/no-else-continue/bad.py | 1 | 7| 0 | 14794 | 86364 | 
| 71 | 54 doc/data/messages/n/no-else-raise/bad.py | 1 | 6| 0 | 14794 | 86422 | 
| 72 | 55 doc/data/messages/u/unnecessary-lambda-assignment/good.py | 1 | 3| 0 | 14794 | 86439 | 
| 73 | 56 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 14794 | 86458 | 
| 74 | 57 doc/data/messages/c/comparison-of-constants/bad.py | 1 | 3| 0 | 14794 | 86481 | 
| 75 | 58 pylint/pyreverse/main.py | 7 | 33| 160 | 14954 | 87950 | 
| 76 | 59 doc/data/messages/u/use-list-literal/bad.py | 1 | 2| 0 | 14954 | 87963 | 
| 77 | 59 pylint/config/arguments_manager.py | 320 | 355| 312 | 15266 | 87963 | 
| 78 | 60 doc/data/messages/u/unnecessary-direct-lambda-call/good.py | 1 | 2| 0 | 15266 | 87976 | 
| 79 | 61 doc/data/messages/s/self-assigning-variable/bad.py | 1 | 3| 0 | 15266 | 87994 | 
| 80 | 62 doc/data/messages/c/confusing-consecutive-elif/bad.py | 1 | 7| 0 | 15266 | 88052 | 
| 81 | 63 doc/data/messages/u/unnecessary-dunder-call/good.py | 1 | 7| 0 | 15266 | 88085 | 
| 82 | 64 doc/data/messages/n/no-else-break/bad.py | 1 | 7| 0 | 15266 | 88126 | 
| 83 | 65 doc/data/messages/o/overridden-final-method/good.py | 1 | 13| 0 | 15266 | 88166 | 
| 84 | 65 pylint/checkers/format.py | 14 | 114| 718 | 15984 | 88166 | 
| 85 | 66 doc/data/messages/u/unnecessary-list-index-lookup/bad.py | 1 | 5| 0 | 15984 | 88202 | 
| 86 | 67 doc/data/messages/s/super-with-arguments/bad.py | 1 | 8| 0 | 15984 | 88237 | 
| 87 | 68 pylint/extensions/typing.py | 78 | 161| 762 | 16746 | 92126 | 
| 88 | 69 doc/data/messages/b/broad-except/bad.py | 1 | 5| 0 | 16746 | 92149 | 
| 89 | 70 doc/data/messages/b/bad-staticmethod-argument/good.py | 1 | 5| 0 | 16746 | 92166 | 
| 90 | 71 doc/data/messages/t/typevar-name-mismatch/bad.py | 1 | 4| 0 | 16746 | 92188 | 
| 91 | 72 doc/data/messages/u/useless-import-alias/bad.py | 1 | 2| 0 | 16746 | 92201 | 
| 92 | 73 doc/data/messages/r/return-in-init/bad.py | 1 | 5| 0 | 16746 | 92228 | 
| 93 | 74 doc/data/messages/m/missing-yield-doc/bad.py | 1 | 10| 0 | 16746 | 92292 | 
| 94 | 75 doc/data/messages/t/too-many-arguments/bad.py | 1 | 17| 0 | 16746 | 92363 | 
| 95 | 75 pylint/checkers/stdlib.py | 246 | 299| 260 | 17006 | 92363 | 
| 96 | 76 doc/data/messages/t/too-many-format-args/bad.py | 1 | 2| 0 | 17006 | 92396 | 
| 97 | 77 doc/data/messages/b/bad-super-call/bad.py | 1 | 8| 0 | 17006 | 92431 | 
| 98 | 78 doc/data/messages/t/typevar-double-variance/bad.py | 1 | 4| 0 | 17006 | 92462 | 
| 99 | 79 examples/deprecation_checker.py | 1 | 48| 462 | 17468 | 93362 | 
| 100 | 79 pylint/checkers/stdlib.py | 92 | 243| 1148 | 18616 | 93362 | 
| 101 | 80 doc/data/messages/a/assignment-from-none/bad.py | 1 | 6| 0 | 18616 | 93380 | 
| 102 | 81 doc/data/messages/m/missing-yield-type-doc/bad.py | 1 | 13| 0 | 18616 | 93453 | 
| 103 | 82 pylint/checkers/design_analysis.py | 7 | 97| 739 | 19355 | 98324 | 
| 104 | 83 doc/data/messages/b/bare-except/bad.py | 1 | 5| 0 | 19355 | 98345 | 
| 105 | 84 doc/data/messages/m/method-cache-max-size-none/bad.py | 1 | 10| 0 | 19355 | 98417 | 
| 106 | 85 doc/data/messages/b/bad-open-mode/bad.py | 1 | 4| 0 | 19355 | 98448 | 
| 107 | 86 pylint/checkers/refactoring/implicit_booleaness_checker.py | 5 | 75| 495 | 19850 | 100234 | 
| 108 | 87 doc/data/messages/i/invalid-enum-extension/bad.py | 1 | 11| 0 | 19850 | 100275 | 
| 109 | 88 doc/data/messages/a/abstract-method/bad.py | 1 | 21| 0 | 19850 | 100339 | 
| 110 | 89 doc/data/messages/m/missing-raises-doc/bad.py | 1 | 9| 0 | 19850 | 100416 | 
| 111 | 90 pylint/checkers/base/basic_checker.py | 104 | 251| 1362 | 21212 | 107614 | 
| 112 | 91 doc/data/messages/m/missing-format-argument-key/good.py | 1 | 2| 0 | 21212 | 107635 | 
| 113 | 91 pylint/checkers/stdlib.py | 435 | 481| 510 | 21722 | 107635 | 
| 114 | 91 pylint/checkers/design_analysis.py | 98 | 176| 510 | 22232 | 107635 | 
| 115 | 92 pylint/config/utils.py | 124 | 146| 263 | 22495 | 109587 | 
| 116 | 93 doc/data/messages/b/bad-super-call/good.py | 1 | 8| 0 | 22495 | 109610 | 
| 117 | 94 doc/data/messages/t/too-few-format-args/bad.py | 1 | 2| 0 | 22495 | 109638 | 
| 118 | **95 pylint/utils/file_state.py** | 5 | 31| 125 | 22620 | 111394 | 
| 119 | 96 doc/data/messages/d/duplicate-argument-name/good.py | 1 | 3| 0 | 22620 | 111408 | 
| 120 | 97 doc/data/messages/b/bad-staticmethod-argument/bad.py | 1 | 5| 0 | 22620 | 111432 | 
| 121 | 98 doc/data/messages/c/comparison-with-callable/bad.py | 1 | 7| 0 | 22620 | 111505 | 
| 122 | 99 pylint/config/exceptions.py | 5 | 24| 117 | 22737 | 111687 | 
| 123 | 100 doc/data/messages/u/ungrouped-imports/bad.py | 1 | 6| 0 | 22737 | 111717 | 
| 124 | 101 pylint/checkers/logging.py | 7 | 99| 727 | 23464 | 114717 | 
| 125 | **101 pylint/lint/pylinter.py** | 481 | 518| 308 | 23772 | 114717 | 
| 126 | 102 doc/data/messages/b/binary-op-exception/good.py | 1 | 5| 0 | 23772 | 114737 | 
| 127 | 103 doc/data/messages/c/catching-non-exception/bad.py | 1 | 9| 0 | 23772 | 114769 | 
| 128 | 104 doc/data/messages/m/missing-type-doc/bad.py | 1 | 7| 0 | 23772 | 114817 | 
| 129 | 105 pylint/checkers/typecheck.py | 800 | 952| 1076 | 24848 | 132002 | 
| 130 | 106 doc/data/messages/c/chained-comparison/bad.py | 1 | 6| 0 | 24848 | 132037 | 
| 131 | 107 doc/data/messages/u/unused-wildcard-import/good.py | 1 | 5| 0 | 24848 | 132048 | 
| 132 | 108 pylint/checkers/imports.py | 199 | 285| 745 | 25593 | 139891 | 
| 133 | 109 doc/data/messages/u/useless-return/good.py | 1 | 6| 0 | 25593 | 139904 | 
| 134 | 110 doc/data/messages/b/bad-except-order/good.py | 1 | 7| 0 | 25593 | 139923 | 
| 135 | **110 pylint/lint/pylinter.py** | 450 | 479| 298 | 25891 | 139923 | 
| 136 | 111 doc/data/messages/b/bad-except-order/bad.py | 1 | 9| 0 | 25891 | 139970 | 
| 137 | 112 doc/data/messages/a/arguments-out-of-order/good.py | 1 | 12| 0 | 25891 | 140042 | 
| 138 | 113 doc/data/messages/u/useless-import-alias/good.py | 1 | 2| 0 | 25891 | 140047 | 
| 139 | 114 pylint/config/argument.py | 124 | 138| 139 | 26030 | 143130 | 
| 140 | 115 doc/data/messages/u/unnecessary-ellipsis/bad.py | 1 | 4| 0 | 26030 | 143150 | 
| 141 | 116 doc/data/messages/m/misplaced-future/bad.py | 1 | 4| 0 | 26030 | 143168 | 
| 142 | 116 pylint/config/callback_actions.py | 9 | 55| 255 | 26285 | 143168 | 
| 143 | 117 doc/data/messages/n/no-else-return/bad.py | 1 | 8| 0 | 26285 | 143220 | 
| 144 | 118 pylint/epylint.py | 126 | 143| 114 | 26399 | 144921 | 
| 145 | 118 pylint/config/arguments_manager.py | 728 | 754| 221 | 26620 | 144921 | 
| 146 | 119 doc/data/messages/m/missing-param-doc/bad.py | 1 | 6| 0 | 26620 | 144961 | 
| 147 | 120 doc/data/messages/d/duplicate-bases/bad.py | 1 | 7| 0 | 26620 | 144984 | 
| 148 | 120 pylint/checkers/stdlib.py | 7 | 41| 300 | 26920 | 144984 | 
| 149 | 121 doc/data/messages/f/function-redefined/bad.py | 1 | 7| 0 | 26920 | 145005 | 
| 150 | 122 doc/data/messages/c/confusing-consecutive-elif/good.py | 1 | 23| 131 | 27051 | 145136 | 
| 151 | 123 doc/data/messages/c/comparison-with-itself/bad.py | 1 | 4| 0 | 27051 | 145165 | 
| 152 | 124 doc/data/messages/a/arguments-renamed/good.py | 1 | 14| 0 | 27051 | 145261 | 
| 153 | 125 doc/data/messages/g/global-at-module-level/good.py | 1 | 2| 0 | 27051 | 145266 | 
| 154 | 126 doc/data/messages/l/literal-comparison/bad.py | 1 | 3| 0 | 27051 | 145287 | 
| 155 | 126 pylint/checkers/exceptions.py | 7 | 29| 123 | 27174 | 145287 | 
| 156 | 127 doc/data/messages/d/duplicate-key/bad.py | 1 | 2| 0 | 27174 | 145317 | 
| 157 | 128 pylint/config/deprecation_actions.py | 94 | 108| 120 | 27294 | 145954 | 
| 158 | 129 doc/data/messages/t/too-many-format-args/good.py | 1 | 2| 0 | 27294 | 145975 | 
| 159 | 130 doc/data/messages/b/bad-exception-context/good.py | 1 | 8| 0 | 27294 | 146031 | 
| 160 | 131 pylint/checkers/misc.py | 7 | 41| 213 | 27507 | 147335 | 
| 161 | 131 pylint/config/deprecation_actions.py | 9 | 59| 273 | 27780 | 147335 | 
| 162 | 131 pylint/pyreverse/main.py | 35 | 203| 892 | 28672 | 147335 | 
| 163 | 132 pylint/checkers/strings.py | 61 | 190| 1222 | 29894 | 155571 | 
| 164 | 133 doc/data/messages/c/consider-using-sys-exit/good.py | 1 | 7| 0 | 29894 | 155607 | 
| 165 | 134 doc/data/messages/b/bad-classmethod-argument/good.py | 1 | 6| 0 | 29894 | 155625 | 
| 166 | 135 doc/data/messages/b/bad-format-character/bad.py | 1 | 2| 0 | 29894 | 155645 | 
| 167 | 136 doc/data/messages/u/unidiomatic-typecheck/good.py | 1 | 4| 0 | 29894 | 155671 | 
| 168 | 137 doc/data/messages/s/super-without-brackets/bad.py | 1 | 12| 0 | 29894 | 155727 | 
| 169 | 138 doc/data/messages/b/bad-open-mode/good.py | 1 | 4| 0 | 29894 | 155750 | 
| 170 | 139 doc/data/messages/a/attribute-defined-outside-init/bad.py | 1 | 4| 0 | 29894 | 155773 | 
| 171 | 140 doc/data/messages/b/bad-str-strip-call/good.py | 1 | 5| 0 | 29894 | 155803 | 
| 172 | 141 doc/data/messages/i/invalid-enum-extension/good.py | 1 | 13| 0 | 29894 | 155850 | 
| 173 | 142 doc/data/messages/t/typevar-name-mismatch/good.py | 1 | 4| 0 | 29894 | 155863 | 
| 174 | 143 pylint/__pkginfo__.py | 10 | 44| 224 | 30118 | 156172 | 
| 175 | 143 pylint/utils/utils.py | 336 | 354| 214 | 30332 | 156172 | 
| 176 | 143 pylint/checkers/stdlib.py | 680 | 717| 273 | 30605 | 156172 | 
| 177 | 144 doc/data/messages/a/assignment-from-no-return/bad.py | 1 | 6| 0 | 30605 | 156201 | 
| 178 | 145 pylint/__init__.py | 5 | 57| 335 | 30940 | 156930 | 
| 179 | 146 pylint/checkers/__init__.py | 44 | 64| 117 | 31057 | 157928 | 
| 180 | 147 doc/data/messages/n/no-else-raise/good.py | 1 | 5| 0 | 31057 | 157974 | 
| 181 | 148 doc/data/messages/b/bad-classmethod-argument/bad.py | 1 | 6| 0 | 31057 | 158001 | 
| 182 | 149 pylint/testutils/functional_test_file.py | 5 | 24| 93 | 31150 | 158159 | 
| 183 | 150 doc/data/messages/b/broad-except/good.py | 1 | 5| 0 | 31150 | 158176 | 
| 184 | 151 doc/data/messages/w/wildcard-import/bad.py | 1 | 2| 0 | 31150 | 158187 | 
| 185 | 152 doc/data/messages/b/bad-str-strip-call/bad.py | 1 | 5| 0 | 31150 | 158233 | 
| 186 | 153 doc/data/messages/u/unspecified-encoding/bad.py | 1 | 4| 0 | 31150 | 158261 | 
| 187 | 154 doc/data/messages/t/try-except-raise/bad.py | 1 | 5| 0 | 31150 | 158289 | 
| 188 | 155 doc/data/messages/s/singleton-comparison/bad.py | 1 | 4| 0 | 31150 | 158316 | 
| 189 | 156 doc/data/messages/m/missing-return-type-doc/bad.py | 1 | 8| 0 | 31150 | 158378 | 
| 190 | 157 doc/data/messages/u/unnecessary-list-index-lookup/good.py | 1 | 5| 0 | 31150 | 158402 | 
| 191 | 157 pylint/config/arguments_manager.py | 559 | 573| 127 | 31277 | 158402 | 
| 192 | 158 doc/data/messages/b/bad-exception-context/bad.py | 1 | 8| 0 | 31277 | 158463 | 
| 193 | 158 pylint/config/utils.py | 7 | 28| 118 | 31395 | 158463 | 
| 194 | 159 doc/data/messages/m/missing-return-doc/bad.py | 1 | 7| 0 | 31395 | 158513 | 
| 195 | 160 doc/data/messages/u/use-dict-literal/good.py | 1 | 2| 0 | 31395 | 158517 | 
| 196 | 160 pylint/lint/message_state_handler.py | 5 | 35| 151 | 31546 | 158517 | 
| 197 | 161 doc/data/messages/t/too-few-format-args/good.py | 1 | 2| 0 | 31546 | 158538 | 
| 198 | 162 doc/data/messages/p/potential-index-error/bad.py | 1 | 2| 0 | 31546 | 158557 | 
| 199 | 163 doc/data/messages/t/too-many-arguments/good.py | 1 | 19| 0 | 31546 | 158633 | 
| 200 | **163 pylint/lint/pylinter.py** | 251 | 341| 815 | 32361 | 158633 | 
| 201 | 164 doc/data/messages/u/use-maxsplit-arg/bad.py | 1 | 3| 0 | 32361 | 158657 | 
| 202 | 165 doc/data/messages/b/bare-except/good.py | 1 | 5| 0 | 32361 | 158674 | 
| 203 | 165 pylint/epylint.py | 146 | 198| 491 | 32852 | 158674 | 
| 204 | 166 doc/data/messages/g/global-statement/good.py | 1 | 11| 0 | 32852 | 158698 | 
| 205 | 166 pylint/lint/base_options.py | 395 | 572| 1142 | 33994 | 158698 | 
| 206 | 167 doc/data/messages/a/assigning-non-slot/bad.py | 1 | 11| 0 | 33994 | 158752 | 
| 207 | 168 doc/data/messages/i/import-outside-toplevel/good.py | 1 | 6| 0 | 33994 | 158766 | 
| 208 | 169 doc/data/messages/w/wrong-import-order/bad.py | 1 | 5| 0 | 33994 | 158792 | 
| 209 | 170 doc/data/messages/n/no-else-continue/good.py | 1 | 6| 0 | 33994 | 158823 | 
| 210 | 171 doc/data/messages/a/assert-on-tuple/bad.py | 1 | 2| 0 | 33994 | 158837 | 
| 211 | 172 pylint/checkers/base_checker.py | 5 | 32| 166 | 34160 | 161109 | 
| 212 | 173 doc/data/messages/s/self-assigning-variable/good.py | 1 | 2| 0 | 34160 | 161115 | 
| 213 | 174 doc/data/messages/a/access-member-before-definition/bad.py | 1 | 6| 0 | 34160 | 161149 | 


### Hint

```
Thanks for the report, this is definitely something we should be able to fix.
Hi. It seems to work when it's on the same line but not globally (which could be useful but I didn't found anything on the documentation). So I have to do:
`# pylint: disable=bad-option-value,useless-object-inheritance`
If I later want to use another option, I have to repeat it again:
`# pylint: disable=bad-option-value,consider-using-sys-exit`
My (unwarranted) two cents: I don't quite see the point of allowing to run different versions of pylint on the same code base.

Different versions of pylint are likely to yield different warnings, because checks are added (and sometimes removed) between versions, and bugs (such as false positives) are fixed. So if your teammate does not run the same version as you do, they may have a "perfect pylint score", while your version gets warnings. Surely you'd rather have a reproducible output, and the best way to achieve that is to warrant a specific version of pylint across the team and your CI environment.
@dbaty Running different versions of pylint helps when porting code from Python2 to Python3
So, what's the status ?
Dropping support for an option, and also having a bad-option-value be an error makes pylint not backwards compatible. It looks like you expect everyone to be able to atomically update their pylint version *and* all of their code annotations in all of their environments at the same time.

I maintain packages supporting python 2.7, 3.4-3.10, and since pylint no longer supports all of those, I'm forced to use older versions of pylint in some environments, which now are not compatible with the latest. I don't know if this is a new behavior or has always been this way, but today was the first time I've had any problem with pylint , and am going to have to disable it in CI now.

@jacobtylerwalls I have a fix, but it requires changing the signature of `PyLinter.process_tokens`. Do we consider that public API?

The issue is that we need to call `file_state.collect_block_lines` to expand a block-level disable over its respective block. This can only be done with access to the `ast` node. Changing `PyLinter.disable` to have access to `node` seems like a very breaking change, although it would be much better to do the expansion immediately upon disabling instead of somewhere later.
My approach does it every time we find a `disable` pragma. We can explore whether we can do something like `collect_block_lines_one_message` instead of iterating over the complete `_module_msgs_state` every time. Still, whatever we do we require access to `node` inside of `process_tokens`.
How likely is it that plugins are calling `PyLinter.process_tokens`?

\`\`\`diff
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
index 7c40f4bf2..4b00ba5ac 100644
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -527,7 +527,9 @@ class PyLinter(
     # block level option handling #############################################
     # see func_block_disable_msg.py test case for expected behaviour
 
-    def process_tokens(self, tokens: list[tokenize.TokenInfo]) -> None:
+    def process_tokens(
+        self, tokens: list[tokenize.TokenInfo], node: nodes.Module
+    ) -> None:
         """Process tokens from the current module to search for module/block level
         options.
         """
@@ -595,6 +597,7 @@ class PyLinter(
                             l_start -= 1
                         try:
                             meth(msgid, "module", l_start)
+                            self.file_state.collect_block_lines(self.msgs_store, node)
                         except exceptions.UnknownMessageError:
                             msg = f"{pragma_repr.action}. Don't recognize message {msgid}."
                             self.add_message(
@@ -1043,7 +1046,7 @@ class PyLinter(
             # assert astroid.file.endswith('.py')
             # invoke ITokenChecker interface on self to fetch module/block
             # level options
-            self.process_tokens(tokens)
+            self.process_tokens(tokens, node)
             if self._ignore_file:
                 return False
             # walk ast to collect line numbers
diff --git a/tests/functional/b/bad_option_value_disable.py b/tests/functional/b/bad_option_value_disable.py
new file mode 100644
index 000000000..cde604411
--- /dev/null
+++ b/tests/functional/b/bad_option_value_disable.py
@@ -0,0 +1,6 @@
+"""Tests for the disabling of bad-option-value."""
+# pylint: disable=invalid-name
+
+# pylint: disable=bad-option-value
+
+var = 1  # pylint: disable=a-removed-option
\`\`\`
The PyLinter is pylint's internal god class, I think if we don't dare to modify it it's a problem. On the other hand, it might be possible to make the node ``Optional`` with a default value of ``None`` and raise a deprecation warning if the node is ``None`` without too much hassle ?

\`\`\`diff
- self.file_state.collect_block_lines(self.msgs_store, node)
+ if node is not None:
+     self.file_state.collect_block_lines(self.msgs_store, node)
+ else:
+     warnings.warn(" process_tokens... deprecation... 3.0.. bad option value won't be disablable...")
Yeah, just making it optional and immediately deprecating it sounds good. Let's add a test for a disabling a message by ID also.
I found some issue while working on this so it will take a little longer.

However, would you guys be okay with one refactor before this to create a `_MessageStateHandler`. I would move all methods like `disable`, `enable`, `set_message_state` etc into this class. Similar to how we moved some of the `PyLinter` stuff to `_ArgumentsManager`. I know we had something similar with `MessageHandlerMixIn` previously, but the issue was that that class was always mixed with `PyLinter` but `mypy` didn't know this.
There are two main benefits:
1) Clean up of `PyLinter` and its file
2) We no longer need to inherit from `BaseTokenChecker`. This is better as `pylint` will rightfully complain about `process_tokens` signature not having `node` as argument. If we keep inheriting we will need to keep giving a default value, which is something we want to avoid in the future.

For now I would only:
1) Create `_MessageStateHandler` and let `PyLinter` inherit from it.
2) Make `PyLinter` a `BaseChecker` and move `process_tokens` to `_MessageStateHandler`.
3) Fix the issue in this issue.

Edit: A preparing PR has been opened with https://github.com/PyCQA/pylint/pull/6537.
Before I look closer to come up with an opinion, I just remember we have #5156, so we should keep it in mind or close it.
I think #5156 is compatible with the proposition Daniel made, it can be a first step as it's better to inherit from ``BaseChecker`` than from ``BaseTokenChecker``.
@DanielNoord sounds good to me!
Well, the refactor turned out to be rather pointless as I ran into multiple subtle regressions with my first approach.

I'm now looking into a different solution: modifying `FileState` to always have access to the relevant `Module` and then expanding pragma's whenever we encounter them. However, this is also proving tricky... I can see why this issue has remained open for so long.
Thank you for looking into it!
Maybe a setting is enough?
```

## Patch

```diff
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -937,8 +937,6 @@ def _check_astroid_module(
             self.process_tokens(tokens)
             if self._ignore_file:
                 return False
-            # walk ast to collect line numbers
-            self.file_state.collect_block_lines(self.msgs_store, node)
             # run raw and tokens checkers
             for raw_checker in rawcheckers:
                 raw_checker.process_module(node)
diff --git a/pylint/utils/file_state.py b/pylint/utils/file_state.py
--- a/pylint/utils/file_state.py
+++ b/pylint/utils/file_state.py
@@ -74,25 +74,32 @@ def collect_block_lines(
         self, msgs_store: MessageDefinitionStore, module_node: nodes.Module
     ) -> None:
         """Walk the AST to collect block level options line numbers."""
+        warnings.warn(
+            "'collect_block_lines' has been deprecated and will be removed in pylint 3.0.",
+            DeprecationWarning,
+        )
         for msg, lines in self._module_msgs_state.items():
             self._raw_module_msgs_state[msg] = lines.copy()
         orig_state = self._module_msgs_state.copy()
         self._module_msgs_state = {}
         self._suppression_mapping = {}
         self._effective_max_line_number = module_node.tolineno
-        self._collect_block_lines(msgs_store, module_node, orig_state)
+        for msgid, lines in orig_state.items():
+            for msgdef in msgs_store.get_message_definitions(msgid):
+                self._set_state_on_block_lines(msgs_store, module_node, msgdef, lines)
 
-    def _collect_block_lines(
+    def _set_state_on_block_lines(
         self,
         msgs_store: MessageDefinitionStore,
         node: nodes.NodeNG,
-        msg_state: MessageStateDict,
+        msg: MessageDefinition,
+        msg_state: dict[int, bool],
     ) -> None:
         """Recursively walk (depth first) AST to collect block level options
-        line numbers.
+        line numbers and set the state correctly.
         """
         for child in node.get_children():
-            self._collect_block_lines(msgs_store, child, msg_state)
+            self._set_state_on_block_lines(msgs_store, child, msg, msg_state)
         # first child line number used to distinguish between disable
         # which are the first child of scoped node with those defined later.
         # For instance in the code below:
@@ -115,9 +122,7 @@ def _collect_block_lines(
             firstchildlineno = node.body[0].fromlineno
         else:
             firstchildlineno = node.tolineno
-        for msgid, lines in msg_state.items():
-            for msg in msgs_store.get_message_definitions(msgid):
-                self._set_message_state_in_block(msg, lines, node, firstchildlineno)
+        self._set_message_state_in_block(msg, msg_state, node, firstchildlineno)
 
     def _set_message_state_in_block(
         self,
@@ -139,18 +144,61 @@ def _set_message_state_in_block(
                 if lineno > firstchildlineno:
                     state = True
                 first_, last_ = node.block_range(lineno)
+                # pylint: disable=useless-suppression
+                # For block nodes first_ is their definition line. For example, we
+                # set the state of line zero for a module to allow disabling
+                # invalid-name for the module. For example:
+                # 1. # pylint: disable=invalid-name
+                # 2. ...
+                # OR
+                # 1. """Module docstring"""
+                # 2. # pylint: disable=invalid-name
+                # 3. ...
+                #
+                # But if we already visited line 0 we don't need to set its state again
+                # 1. # pylint: disable=invalid-name
+                # 2. # pylint: enable=invalid-name
+                # 3. ...
+                # The state should come from line 1, not from line 2
+                # Therefore, if the 'fromlineno' is already in the states we just start
+                # with the lineno we were originally visiting.
+                # pylint: enable=useless-suppression
+                if (
+                    first_ == node.fromlineno
+                    and first_ >= firstchildlineno
+                    and node.fromlineno in self._module_msgs_state.get(msg.msgid, ())
+                ):
+                    first_ = lineno
+
             else:
                 first_ = lineno
                 last_ = last
             for line in range(first_, last_ + 1):
-                # do not override existing entries
-                if line in self._module_msgs_state.get(msg.msgid, ()):
+                # Do not override existing entries. This is especially important
+                # when parsing the states for a scoped node where some line-disables
+                # have already been parsed.
+                if (
+                    (
+                        isinstance(node, nodes.Module)
+                        and node.fromlineno <= line < lineno
+                    )
+                    or (
+                        not isinstance(node, nodes.Module)
+                        and node.fromlineno < line < lineno
+                    )
+                ) and line in self._module_msgs_state.get(msg.msgid, ()):
                     continue
                 if line in lines:  # state change in the same block
                     state = lines[line]
                     original_lineno = line
+
+                # Update suppression mapping
                 if not state:
                     self._suppression_mapping[(msg.msgid, line)] = original_lineno
+                else:
+                    self._suppression_mapping.pop((msg.msgid, line), None)
+
+                # Update message state for respective line
                 try:
                     self._module_msgs_state[msg.msgid][line] = state
                 except KeyError:
@@ -160,10 +208,20 @@ def _set_message_state_in_block(
     def set_msg_status(self, msg: MessageDefinition, line: int, status: bool) -> None:
         """Set status (enabled/disable) for a given message at a given line."""
         assert line > 0
+        assert self._module
+        # TODO: 3.0: Remove unnecessary assertion
+        assert self._msgs_store
+
+        # Expand the status to cover all relevant block lines
+        self._set_state_on_block_lines(
+            self._msgs_store, self._module, msg, {line: status}
+        )
+
+        # Store the raw value
         try:
-            self._module_msgs_state[msg.msgid][line] = status
+            self._raw_module_msgs_state[msg.msgid][line] = status
         except KeyError:
-            self._module_msgs_state[msg.msgid] = {line: status}
+            self._raw_module_msgs_state[msg.msgid] = {line: status}
 
     def handle_ignored_message(
         self, state_scope: Literal[0, 1, 2] | None, msgid: str, line: int | None

```

## Test Patch

```diff
diff --git a/tests/functional/b/bad_option_value_disable.py b/tests/functional/b/bad_option_value_disable.py
new file mode 100644
--- /dev/null
+++ b/tests/functional/b/bad_option_value_disable.py
@@ -0,0 +1,14 @@
+"""Tests for the disabling of bad-option-value."""
+# pylint: disable=invalid-name
+
+# pylint: disable=bad-option-value
+
+var = 1  # pylint: disable=a-removed-option
+
+# pylint: enable=bad-option-value
+
+var = 1  # pylint: disable=a-removed-option # [bad-option-value]
+
+# bad-option-value needs to be disabled before the bad option
+var = 1  # pylint: disable=a-removed-option, bad-option-value # [bad-option-value]
+var = 1  # pylint: disable=bad-option-value, a-removed-option
diff --git a/tests/functional/b/bad_option_value_disable.txt b/tests/functional/b/bad_option_value_disable.txt
new file mode 100644
--- /dev/null
+++ b/tests/functional/b/bad_option_value_disable.txt
@@ -0,0 +1,2 @@
+bad-option-value:10:0:None:None::Bad option value for disable. Don't recognize message a-removed-option.:UNDEFINED
+bad-option-value:13:0:None:None::Bad option value for disable. Don't recognize message a-removed-option.:UNDEFINED
diff --git a/tests/lint/unittest_lint.py b/tests/lint/unittest_lint.py
--- a/tests/lint/unittest_lint.py
+++ b/tests/lint/unittest_lint.py
@@ -190,8 +190,14 @@ def reporter():
 @pytest.fixture
 def initialized_linter(linter: PyLinter) -> PyLinter:
     linter.open()
-    linter.set_current_module("toto", "mydir/toto")
-    linter.file_state = FileState("toto", linter.msgs_store)
+    linter.set_current_module("long_test_file", "long_test_file")
+    linter.file_state = FileState(
+        "long_test_file",
+        linter.msgs_store,
+        linter.get_ast(
+            str(join(REGRTEST_DATA_DIR, "long_test_file.py")), "long_test_file"
+        ),
+    )
     return linter
 
 
@@ -271,9 +277,9 @@ def test_enable_message_block(initialized_linter: PyLinter) -> None:
     filepath = join(REGRTEST_DATA_DIR, "func_block_disable_msg.py")
     linter.set_current_module("func_block_disable_msg")
     astroid = linter.get_ast(filepath, "func_block_disable_msg")
+    linter.file_state = FileState("func_block_disable_msg", linter.msgs_store, astroid)
     linter.process_tokens(tokenize_module(astroid))
     fs = linter.file_state
-    fs.collect_block_lines(linter.msgs_store, astroid)
     # global (module level)
     assert linter.is_message_enabled("W0613")
     assert linter.is_message_enabled("E1101")
@@ -310,7 +316,6 @@ def test_enable_message_block(initialized_linter: PyLinter) -> None:
     assert linter.is_message_enabled("E1101", 75)
     assert linter.is_message_enabled("E1101", 77)
 
-    fs = linter.file_state
     assert fs._suppression_mapping["W0613", 18] == 17
     assert fs._suppression_mapping["E1101", 33] == 30
     assert ("E1101", 46) not in fs._suppression_mapping
diff --git a/tests/regrtest_data/long_test_file.py b/tests/regrtest_data/long_test_file.py
new file mode 100644
--- /dev/null
+++ b/tests/regrtest_data/long_test_file.py
@@ -0,0 +1,100 @@
+"""
+This file is used for bad-option-value's test. We needed a module that isn’t restricted by line numbers
+as the tests use various line numbers to test the behaviour.
+
+Using an empty module creates issue as you can’t disable something on line 10 if it doesn’t exist. Thus, we created an extra long file so we never run into an issue with that.
+"""
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+
+print(1)
diff --git a/tests/test_deprecation.py b/tests/test_deprecation.py
--- a/tests/test_deprecation.py
+++ b/tests/test_deprecation.py
@@ -9,6 +9,7 @@
 from typing import Any
 
 import pytest
+from astroid import nodes
 
 from pylint.checkers import BaseChecker
 from pylint.checkers.mapreduce_checker import MapReduceMixin
@@ -100,3 +101,10 @@ def test_filestate() -> None:
     with pytest.warns(DeprecationWarning):
         FileState(msg_store=MessageDefinitionStore())
     FileState("foo", MessageDefinitionStore())
+
+
+def test_collectblocklines() -> None:
+    """Test FileState.collect_block_lines."""
+    state = FileState("foo", MessageDefinitionStore())
+    with pytest.warns(DeprecationWarning):
+        state.collect_block_lines(MessageDefinitionStore(), nodes.Module("foo"))

```


## Code snippets

### 1 - doc/data/messages/b/bad-option-value/bad.py:

Start line: 1, End line: 3

```python

```
### 2 - doc/data/messages/b/bad-option-value/good.py:

Start line: 1, End line: 3

```python

```
### 3 - pylint/lint/pylinter.py:

Start line: 97, End line: 215

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
        {"scope": WarningScope.LINE},
    ),
    "I0010": (
        "Unable to consider inline option %r",
        "bad-inline-option",
        "Used when an inline option is either badly formatted or can't "
        "be used inside modules.",
        {"scope": WarningScope.LINE},
    ),
    "I0011": (
        "Locally disabling %s (%s)",
        "locally-disabled",
        "Used when an inline option disables a message or a messages category.",
        {"scope": WarningScope.LINE},
    ),
    "I0013": (
        "Ignoring entire file",
        "file-ignored",
        "Used to inform that the file will not be checked",
        {"scope": WarningScope.LINE},
    ),
    "I0020": (
        "Suppressed %s (from line %d)",
        "suppressed-message",
        "A message was triggered on a line, but suppressed explicitly "
        "by a disable= comment in the file. This message is not "
        "generated for messages that are ignored due to configuration "
        "settings.",
        {"scope": WarningScope.LINE},
    ),
    "I0021": (
        "Useless suppression of %s",
        "useless-suppression",
        "Reported when a message is explicitly disabled for a line or "
        "a block of code, but never triggered.",
        {"scope": WarningScope.LINE},
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
    "E0012": (
        "Bad option value for %s",
        "bad-option-value",
        "Used when a bad value for an inline option is encountered.",
        {"scope": WarningScope.LINE},
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
        "Used when we detect a setting in the top level of a toml configuration that shouldn't be there.",
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
### 4 - pylint/config/option_parser.py:

Start line: 5, End line: 54

```python
import optparse  # pylint: disable=deprecated-module
import warnings

from pylint.config.option import Option


def _level_options(group, outputlevel):
    return [
        option
        for option in group.option_list
        if (getattr(option, "level", 0) or 0) <= outputlevel
        and option.help is not optparse.SUPPRESS_HELP
    ]


class OptionParser(optparse.OptionParser):
    def __init__(self, option_class, *args, **kwargs):
        # TODO: 3.0: Remove deprecated class
        warnings.warn(
            "OptionParser has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        super().__init__(option_class=Option, *args, **kwargs)

    def format_option_help(self, formatter=None):
        if formatter is None:
            formatter = self.formatter
        outputlevel = getattr(formatter, "output_level", 0)
        formatter.store_option_strings(self)
        result = [formatter.format_heading("Options")]
        formatter.indent()
        if self.option_list:
            result.append(optparse.OptionContainer.format_option_help(self, formatter))
            result.append("\n")
        for group in self.option_groups:
            if group.level <= outputlevel and (
                group.description or _level_options(group, outputlevel)
            ):
                result.append(group.format_help(formatter))
                result.append("\n")
        formatter.dedent()
        # Drop the last "\n", or the header if no options or option groups:
        return "".join(result[:-1])

    def _match_long_opt(self, opt):  # pragma: no cover # Unused
        """Disable abbreviations."""
        if opt not in self._long_opt:
            raise optparse.BadOptionError(opt)
        return opt
```
### 5 - pylint/config/option.py:

Start line: 204, End line: 219

```python
class Option(optparse.Option):

    def process(self, opt, value, values, parser):  # pragma: no cover # Argparse
        if self.callback and self.callback.__module__ == "pylint.lint.run":
            return 1
        # First, convert the value(s) to the right type.  Howl if any
        # value(s) are bogus.
        value = self.convert_value(opt, value)
        if self.type == "named":
            existent = getattr(values, self.dest)
            if existent:
                existent.update(value)
                value = existent
        # And then take whatever action is expected of us.
        # This is a separate method to make life easier for
        # subclasses to add new actions.
        return self.take_action(self.action, self.dest, opt, value, values, parser)
```
### 6 - pylint/constants.py:

Start line: 92, End line: 199

```python
DELETED_MESSAGES = [
    # Everything until the next comment is from the
    # PY3K+ checker, see https://github.com/PyCQA/pylint/pull/4942
    DeletedMessage("W1601", "apply-builtin"),
    DeletedMessage("E1601", "print-statement"),
    DeletedMessage("E1602", "parameter-unpacking"),
    DeletedMessage(
        "E1603", "unpacking-in-except", [("W0712", "old-unpacking-in-except")]
    ),
    DeletedMessage("E1604", "old-raise-syntax", [("W0121", "old-old-raise-syntax")]),
    DeletedMessage("E1605", "backtick", [("W0333", "old-backtick")]),
    DeletedMessage("E1609", "import-star-module-level"),
    DeletedMessage("W1601", "apply-builtin"),
    DeletedMessage("W1602", "basestring-builtin"),
    DeletedMessage("W1603", "buffer-builtin"),
    DeletedMessage("W1604", "cmp-builtin"),
    DeletedMessage("W1605", "coerce-builtin"),
    DeletedMessage("W1606", "execfile-builtin"),
    DeletedMessage("W1607", "file-builtin"),
    DeletedMessage("W1608", "long-builtin"),
    DeletedMessage("W1609", "raw_input-builtin"),
    DeletedMessage("W1610", "reduce-builtin"),
    DeletedMessage("W1611", "standarderror-builtin"),
    DeletedMessage("W1612", "unicode-builtin"),
    DeletedMessage("W1613", "xrange-builtin"),
    DeletedMessage("W1614", "coerce-method"),
    DeletedMessage("W1615", "delslice-method"),
    DeletedMessage("W1616", "getslice-method"),
    DeletedMessage("W1617", "setslice-method"),
    DeletedMessage("W1618", "no-absolute-import"),
    DeletedMessage("W1619", "old-division"),
    DeletedMessage("W1620", "dict-iter-method"),
    DeletedMessage("W1621", "dict-view-method"),
    DeletedMessage("W1622", "next-method-called"),
    DeletedMessage("W1623", "metaclass-assignment"),
    DeletedMessage(
        "W1624", "indexing-exception", [("W0713", "old-indexing-exception")]
    ),
    DeletedMessage("W1625", "raising-string", [("W0701", "old-raising-string")]),
    DeletedMessage("W1626", "reload-builtin"),
    DeletedMessage("W1627", "oct-method"),
    DeletedMessage("W1628", "hex-method"),
    DeletedMessage("W1629", "nonzero-method"),
    DeletedMessage("W1630", "cmp-method"),
    DeletedMessage("W1632", "input-builtin"),
    DeletedMessage("W1633", "round-builtin"),
    DeletedMessage("W1634", "intern-builtin"),
    DeletedMessage("W1635", "unichr-builtin"),
    DeletedMessage(
        "W1636", "map-builtin-not-iterating", [("W1631", "implicit-map-evaluation")]
    ),
    DeletedMessage("W1637", "zip-builtin-not-iterating"),
    DeletedMessage("W1638", "range-builtin-not-iterating"),
    DeletedMessage("W1639", "filter-builtin-not-iterating"),
    DeletedMessage("W1640", "using-cmp-argument"),
    DeletedMessage("W1642", "div-method"),
    DeletedMessage("W1643", "idiv-method"),
    DeletedMessage("W1644", "rdiv-method"),
    DeletedMessage("W1645", "exception-message-attribute"),
    DeletedMessage("W1646", "invalid-str-codec"),
    DeletedMessage("W1647", "sys-max-int"),
    DeletedMessage("W1648", "bad-python3-import"),
    DeletedMessage("W1649", "deprecated-string-function"),
    DeletedMessage("W1650", "deprecated-str-translate-call"),
    DeletedMessage("W1651", "deprecated-itertools-function"),
    DeletedMessage("W1652", "deprecated-types-field"),
    DeletedMessage("W1653", "next-method-defined"),
    DeletedMessage("W1654", "dict-items-not-iterating"),
    DeletedMessage("W1655", "dict-keys-not-iterating"),
    DeletedMessage("W1656", "dict-values-not-iterating"),
    DeletedMessage("W1657", "deprecated-operator-function"),
    DeletedMessage("W1658", "deprecated-urllib-function"),
    DeletedMessage("W1659", "xreadlines-attribute"),
    DeletedMessage("W1660", "deprecated-sys-function"),
    DeletedMessage("W1661", "exception-escape"),
    DeletedMessage("W1662", "comprehension-escape"),
    # https://github.com/PyCQA/pylint/pull/3578
    DeletedMessage("W0312", "mixed-indentation"),
    # https://github.com/PyCQA/pylint/pull/3577
    DeletedMessage(
        "C0326",
        "bad-whitespace",
        [
            ("C0323", "no-space-after-operator"),
            ("C0324", "no-space-after-comma"),
            ("C0322", "no-space-before-operator"),
        ],
    ),
    # https://github.com/PyCQA/pylint/pull/3571
    DeletedMessage("C0330", "bad-continuation"),
    # No PR
    DeletedMessage("R0921", "abstract-class-not-used"),
    # https://github.com/PyCQA/pylint/pull/3577
    DeletedMessage("C0326", "bad-whitespace"),
    # Pylint 1.4.3
    DeletedMessage("W0142", "star-args"),
    # https://github.com/PyCQA/pylint/issues/2409
    DeletedMessage("W0232", "no-init"),
    # https://github.com/PyCQA/pylint/pull/6421
    DeletedMessage("W0111", "assign-to-new-keyword"),
]


# ignore some messages when emitting useless-suppression:
# - cyclic-import: can show false positives due to incomplete context
# - deprecated-{module, argument, class, method, decorator}:
#   can cause false positives for multi-interpreter projects
#   when linting with an interpreter on a lower python version
```
### 7 - pylint/extensions/bad_builtin.py:

Start line: 25, End line: 50

```python
class BadBuiltinChecker(BaseChecker):

    name = "deprecated_builtins"
    msgs = {
        "W0141": (
            "Used builtin function %s",
            "bad-builtin",
            "Used when a disallowed builtin function is used (see the "
            "bad-function option). Usual disallowed functions are the ones "
            "like map, or filter , where Python offers now some cleaner "
            "alternative like list comprehension.",
        )
    }

    options = (
        (
            "bad-functions",
            {
                "default": BAD_FUNCTIONS,
                "type": "csv",
                "metavar": "<builtin function names>",
                "help": "List of builtins function names that should not be "
                "used, separated by a comma",
            },
        ),
    )
```
### 8 - pylint/config/option.py:

Start line: 149, End line: 182

```python
# pylint: disable=no-member
class Option(optparse.Option):
    TYPES = optparse.Option.TYPES + (
        "regexp",
        "regexp_csv",
        "regexp_paths_csv",
        "csv",
        "yn",
        "confidence",
        "multiple_choice",
        "non_empty_string",
        "py_version",
    )
    ATTRS = optparse.Option.ATTRS + ["hide", "level"]
    TYPE_CHECKER = copy.copy(optparse.Option.TYPE_CHECKER)
    TYPE_CHECKER["regexp"] = _regexp_validator
    TYPE_CHECKER["regexp_csv"] = _regexp_csv_validator
    TYPE_CHECKER["regexp_paths_csv"] = _regexp_paths_csv_validator
    TYPE_CHECKER["csv"] = _csv_validator
    TYPE_CHECKER["yn"] = _yn_validator
    TYPE_CHECKER["confidence"] = _multiple_choices_validating_option
    TYPE_CHECKER["multiple_choice"] = _multiple_choices_validating_option
    TYPE_CHECKER["non_empty_string"] = _non_empty_string_validator
    TYPE_CHECKER["py_version"] = _py_version_validator

    def __init__(self, *opts, **attrs):
        # TODO: 3.0: Remove deprecated class
        warnings.warn(
            "Option has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        super().__init__(*opts, **attrs)
        if hasattr(self, "hide") and self.hide:
            self.help = optparse.SUPPRESS_HELP
```
### 9 - pylint/config/arguments_provider.py:

Start line: 7, End line: 28

```python
from __future__ import annotations

import argparse
import optparse  # pylint: disable=deprecated-module
import warnings
from collections.abc import Iterator
from typing import Any

from pylint.config.arguments_manager import _ArgumentsManager
from pylint.typing import OptionDict, Options


class UnsupportedAction(Exception):
    """Raised by set_option when it doesn't know what to do for an action."""

    def __init__(self, *args: object) -> None:
        # TODO: 3.0: Remove deprecated exception
        warnings.warn(
            "UnsupportedAction has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        super().__init__(*args)
```
### 10 - pylint/config/arguments_provider.py:

Start line: 115, End line: 133

```python
class _ArgumentsProvider:

    def option_value(self, opt: str) -> Any:  # pragma: no cover
        """DEPRECATED: Get the current value for the given option."""
        warnings.warn(
            "option_value has been deprecated. It will be removed "
            "in a future release.",
            DeprecationWarning,
        )
        return getattr(self._arguments_manager.config, opt.replace("-", "_"), None)

    # pylint: disable-next=unused-argument
    def set_option(self, optname, value, action=None, optdict=None):  # pragma: no cover
        """DEPRECATED: Method called to set an option (registered in the options list)."""
        # TODO: 3.0: Remove deprecated method.
        warnings.warn(
            "set_option has been deprecated. You can use _arguments_manager.set_option "
            "or linter.set_option to set options on the global configuration object.",
            DeprecationWarning,
        )
        self._arguments_manager.set_option(optname, value)
```
### 118 - pylint/utils/file_state.py:

Start line: 5, End line: 31

```python
from __future__ import annotations

import collections
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Dict

from astroid import nodes

from pylint.constants import (
    INCOMPATIBLE_WITH_USELESS_SUPPRESSION,
    MSG_STATE_SCOPE_MODULE,
    WarningScope,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from pylint.message import MessageDefinition, MessageDefinitionStore


MessageStateDict = Dict[str, Dict[int, bool]]
```
### 125 - pylint/lint/pylinter.py:

Start line: 481, End line: 518

```python
class PyLinter(
    _ArgumentsManager,
    _MessageStateHandler,
    reporters.ReportsHandlerMixIn,
    checkers.BaseChecker,
):

    def any_fail_on_issues(self) -> bool:
        return any(x in self.fail_on_symbols for x in self.stats.by_msg.keys())

    def disable_reporters(self) -> None:
        """Disable all reporters."""
        for _reporters in self._reports.values():
            for report_id, _, _ in _reporters:
                self.disable_report(report_id)

    def _parse_error_mode(self) -> None:
        """Parse the current state of the error mode.

        Error mode: enable only errors; no reports, no persistent.
        """
        if not self._error_mode:
            return

        self.disable_noerror_messages()
        self.disable("miscellaneous")
        self.set_option("reports", False)
        self.set_option("persistent", False)
        self.set_option("score", False)

    # code checking methods ###################################################

    def get_checkers(self) -> list[BaseChecker]:
        """Return all available checkers as an ordered list."""
        return sorted(c for _checkers in self._checkers.values() for c in _checkers)

    def get_checker_names(self) -> list[str]:
        """Get all the checker names that this linter knows about."""
        return sorted(
            {
                checker.name
                for checker in self.get_checkers()
                if checker.name != MAIN_CHECKER_NAME
            }
        )
```
### 135 - pylint/lint/pylinter.py:

Start line: 450, End line: 479

```python
class PyLinter(
    _ArgumentsManager,
    _MessageStateHandler,
    reporters.ReportsHandlerMixIn,
    checkers.BaseChecker,
):

    def enable_fail_on_messages(self) -> None:
        """Enable 'fail on' msgs.

        Convert values in config.fail_on (which might be msg category, msg id,
        or symbol) to specific msgs, then enable and flag them for later.
        """
        fail_on_vals = self.config.fail_on
        if not fail_on_vals:
            return

        fail_on_cats = set()
        fail_on_msgs = set()
        for val in fail_on_vals:
            # If value is a category, add category, else add message
            if val in MSG_TYPES:
                fail_on_cats.add(val)
            else:
                fail_on_msgs.add(val)

        # For every message in every checker, if cat or msg flagged, enable check
        for all_checkers in self._checkers.values():
            for checker in all_checkers:
                for msg in checker.messages:
                    if msg.msgid in fail_on_msgs or msg.symbol in fail_on_msgs:
                        # message id/symbol matched, enable and flag it
                        self.enable(msg.msgid)
                        self.fail_on_symbols.append(msg.symbol)
                    elif msg.msgid[0] in fail_on_cats:
                        # message starts with a category value, flag (but do not enable) it
                        self.fail_on_symbols.append(msg.symbol)
```
### 200 - pylint/lint/pylinter.py:

Start line: 251, End line: 341

```python
class PyLinter(
    _ArgumentsManager,
    _MessageStateHandler,
    reporters.ReportsHandlerMixIn,
    checkers.BaseChecker,
):

    def __init__(
        self,
        options: Options = (),
        reporter: reporters.BaseReporter | reporters.MultiReporter | None = None,
        option_groups: tuple[tuple[str, str], ...] = (),
        # TODO: Deprecate passing the pylintrc parameter
        pylintrc: str | None = None,  # pylint: disable=unused-argument
    ) -> None:
        _ArgumentsManager.__init__(self, prog="pylint")
        _MessageStateHandler.__init__(self, self)

        # Some stuff has to be done before initialization of other ancestors...
        # messages store / checkers / reporter / astroid manager

        # Attributes for reporters
        self.reporter: reporters.BaseReporter | reporters.MultiReporter
        if reporter:
            self.set_reporter(reporter)
        else:
            self.set_reporter(TextReporter())
        self._reporters: dict[str, type[reporters.BaseReporter]] = {}
        """Dictionary of possible but non-initialized reporters."""

        # Attributes for checkers and plugins
        self._checkers: defaultdict[
            str, list[checkers.BaseChecker]
        ] = collections.defaultdict(list)
        """Dictionary of registered and initialized checkers."""
        self._dynamic_plugins: set[str] = set()
        """Set of loaded plugin names."""

        # Attributes related to registering messages and their handling
        self.msgs_store = MessageDefinitionStore()
        self.msg_status = 0
        self._by_id_managed_msgs: list[ManagedMessage] = []

        # Attributes related to visiting files
        self.file_state = FileState("", self.msgs_store, is_base_filestate=True)
        self.current_name: str | None = None
        self.current_file: str | None = None
        self._ignore_file = False

        # Attributes related to stats
        self.stats = LinterStats()

        # Attributes related to (command-line) options and their parsing
        self.options: Options = options + _make_linter_options(self)
        for opt_group in option_groups:
            self.option_groups_descs[opt_group[0]] = opt_group[1]
        self._option_groups: tuple[tuple[str, str], ...] = option_groups + (
            ("Messages control", "Options controlling analysis messages"),
            ("Reports", "Options related to output formatting and reporting"),
        )
        self.fail_on_symbols: list[str] = []
        """List of message symbols on which pylint should fail, set by --fail-on."""
        self._error_mode = False

        reporters.ReportsHandlerMixIn.__init__(self)
        checkers.BaseChecker.__init__(self, self)
        # provided reports
        self.reports = (
            ("RP0001", "Messages by category", report_total_messages_stats),
            (
                "RP0002",
                "% errors / warnings by module",
                report_messages_by_module_stats,
            ),
            ("RP0003", "Messages", report_messages_stats),
        )
        self.register_checker(self)

    @property
    def option_groups(self) -> tuple[tuple[str, str], ...]:
        # TODO: 3.0: Remove deprecated attribute
        warnings.warn(
            "The option_groups attribute has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        return self._option_groups

    @option_groups.setter
    def option_groups(self, value: tuple[tuple[str, str], ...]) -> None:
        warnings.warn(
            "The option_groups attribute has been deprecated and will be removed in pylint 3.0",
            DeprecationWarning,
        )
        self._option_groups = value

    def load_default_plugins(self) -> None:
        checkers.initialize(self)
        reporters.initialize(self)
```
