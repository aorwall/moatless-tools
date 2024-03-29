# pylint-dev__pylint-7993

| **pylint-dev/pylint** | `e90702074e68e20dc8e5df5013ee3ecf22139c3e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 18735 |
| **Any found context length** | 18735 |
| **Avg pos** | 83.0 |
| **Min pos** | 83 |
| **Max pos** | 83 |
| **Top file pos** | 9 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/pylint/reporters/text.py b/pylint/reporters/text.py
--- a/pylint/reporters/text.py
+++ b/pylint/reporters/text.py
@@ -175,7 +175,7 @@ def on_set_current_module(self, module: str, filepath: str | None) -> None:
         self._template = template
 
         # Check to see if all parameters in the template are attributes of the Message
-        arguments = re.findall(r"\{(.+?)(:.*)?\}", template)
+        arguments = re.findall(r"\{(\w+?)(:.*)?\}", template)
         for argument in arguments:
             if argument[0] not in MESSAGE_FIELDS:
                 warnings.warn(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| pylint/reporters/text.py | 178 | 178 | 83 | 9 | 18735


## Problem Statement

```
Using custom braces in message template does not work
### Bug description

Have any list of errors:

On pylint 1.7 w/ python3.6 - I am able to use this as my message template
\`\`\`
$ pylint test.py --msg-template='{{ "Category": "{category}" }}'
No config file found, using default configuration
************* Module [redacted].test
{ "Category": "convention" }
{ "Category": "error" }
{ "Category": "error" }
{ "Category": "convention" }
{ "Category": "convention" }
{ "Category": "convention" }
{ "Category": "error" }
\`\`\`

However, on Python3.9 with Pylint 2.12.2, I get the following:
\`\`\`
$ pylint test.py --msg-template='{{ "Category": "{category}" }}'
[redacted]/site-packages/pylint/reporters/text.py:206: UserWarning: Don't recognize the argument '{ "Category"' in the --msg-template. Are you sure it is supported on the current version of pylint?
  warnings.warn(
************* Module [redacted].test
" }
" }
" }
" }
" }
" }
\`\`\`

Is this intentional or a bug?

### Configuration

_No response_

### Command used

\`\`\`shell
pylint test.py --msg-template='{{ "Category": "{category}" }}'
\`\`\`


### Pylint output

\`\`\`shell
[redacted]/site-packages/pylint/reporters/text.py:206: UserWarning: Don't recognize the argument '{ "Category"' in the --msg-template. Are you sure it is supported on the current version of pylint?
  warnings.warn(
************* Module [redacted].test
" }
" }
" }
" }
" }
" }
\`\`\`


### Expected behavior

Expect the dictionary to print out with `"Category"` as the key.

### Pylint version

\`\`\`shell
Affected Version:
pylint 2.12.2
astroid 2.9.2
Python 3.9.9+ (heads/3.9-dirty:a2295a4, Dec 21 2021, 22:32:52) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-44)]


Previously working version:
No config file found, using default configuration
pylint 1.7.4, 
astroid 1.6.6
Python 3.6.8 (default, Nov 16 2020, 16:55:22) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-44)]
\`\`\`


### OS / Environment

_No response_

### Additional dependencies

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 pylint/lint/pylinter.py | 104 | 235| 1089 | 1089 | 10902 | 
| 2 | 2 pylint/constants.py | 5 | 102| 755 | 1844 | 12247 | 
| 3 | 3 pylint/message/_deleted_message_ids.py | 5 | 126| 1415 | 3259 | 14155 | 
| 4 | 4 pylint/checkers/logging.py | 7 | 94| 727 | 3986 | 17387 | 
| 5 | 5 pylint/checkers/format.py | 14 | 117| 739 | 4725 | 23501 | 
| 6 | 6 pylint/message/message.py | 5 | 93| 535 | 5260 | 24101 | 
| 7 | 7 doc/data/messages/c/consider-using-f-string/bad.py | 1 | 11| 209 | 5469 | 24310 | 
| 8 | 8 pylint/checkers/imports.py | 215 | 305| 745 | 6214 | 32524 | 
| 9 | **9 pylint/reporters/text.py** | 11 | 69| 375 | 6589 | 35029 | 
| 10 | 10 pylint/lint/message_state_handler.py | 5 | 37| 162 | 6751 | 38500 | 
| 11 | 11 pylint/checkers/strings.py | 68 | 197| 1222 | 7973 | 46922 | 
| 12 | 12 pylint/testutils/output_line.py | 5 | 33| 186 | 8159 | 48192 | 
| 13 | 13 pylint/checkers/exceptions.py | 64 | 168| 1028 | 9187 | 52938 | 
| 14 | 14 pylint/message/message_definition.py | 5 | 79| 553 | 9740 | 54031 | 
| 15 | 15 doc/data/messages/u/unexpected-keyword-arg/bad.py | 1 | 6| 0 | 9740 | 54077 | 
| 16 | 16 doc/data/messages/u/unrecognized-option/good.py | 1 | 2| 0 | 9740 | 54089 | 
| 17 | 17 doc/data/messages/u/unexpected-keyword-arg/good.py | 1 | 6| 0 | 9740 | 54122 | 
| 18 | 18 pylint/lint/utils.py | 5 | 63| 412 | 10152 | 54949 | 
| 19 | 19 doc/data/messages/l/logging-unsupported-format/good.py | 1 | 2| 0 | 10152 | 54961 | 
| 20 | 20 pylint/config/help_formatter.py | 32 | 66| 331 | 10483 | 55542 | 
| 21 | 21 doc/data/messages/u/unrecognized-inline-option/good.py | 1 | 2| 0 | 10483 | 55554 | 
| 22 | 22 pylint/checkers/newstyle.py | 7 | 32| 144 | 10627 | 56495 | 
| 23 | 22 pylint/checkers/imports.py | 7 | 74| 483 | 11110 | 56495 | 
| 24 | 23 doc/data/messages/s/super-without-brackets/bad.py | 1 | 12| 0 | 11110 | 56551 | 
| 25 | 24 doc/data/messages/u/use-list-literal/bad.py | 1 | 2| 0 | 11110 | 56564 | 
| 26 | 25 pylint/checkers/variables.py | 370 | 509| 1243 | 12353 | 80012 | 
| 27 | 26 doc/data/messages/u/use-dict-literal/bad.py | 1 | 2| 0 | 12353 | 80026 | 
| 28 | 27 doc/data/messages/u/use-symbolic-message-instead/good.py | 1 | 2| 0 | 12353 | 80038 | 
| 29 | 28 doc/data/messages/s/super-without-brackets/good.py | 1 | 12| 0 | 12353 | 80086 | 
| 30 | 29 pylint/message/__init__.py | 7 | 18| 65 | 12418 | 80225 | 
| 31 | 30 doc/data/messages/i/invalid-all-format/bad.py | 1 | 4| 0 | 12418 | 80244 | 
| 32 | 31 pylint/checkers/base/basic_error_checker.py | 99 | 206| 941 | 13359 | 84995 | 
| 33 | 32 pylint/interfaces.py | 7 | 49| 291 | 13650 | 85898 | 
| 34 | 33 doc/data/messages/u/use-dict-literal/good.py | 1 | 2| 0 | 13650 | 85902 | 
| 35 | 34 doc/data/messages/c/consider-using-with/bad.py | 1 | 6| 0 | 13650 | 85957 | 
| 36 | 35 pylint/config/_pylint_config/utils.py | 7 | 35| 182 | 13832 | 86791 | 
| 37 | 36 doc/data/messages/c/consider-using-with/good.py | 1 | 6| 0 | 13832 | 86837 | 
| 38 | 37 doc/data/messages/i/invalid-format-returned/good.py | 1 | 2| 0 | 13832 | 86849 | 
| 39 | 38 doc/data/messages/c/consider-using-dict-items/bad.py | 1 | 11| 0 | 13832 | 86923 | 
| 40 | 39 doc/data/messages/m/missing-format-argument-key/bad.py | 1 | 2| 0 | 13832 | 86948 | 
| 41 | 40 doc/data/messages/i/invalid-all-format/good.py | 1 | 4| 0 | 13832 | 86961 | 
| 42 | 41 doc/data/messages/c/confusing-consecutive-elif/bad.py | 1 | 7| 0 | 13832 | 87019 | 
| 43 | 42 doc/data/messages/u/use-list-literal/good.py | 1 | 2| 0 | 13832 | 87023 | 
| 44 | 43 doc/data/messages/u/unused-format-string-argument/bad.py | 1 | 2| 0 | 13832 | 87052 | 
| 45 | 44 doc/data/messages/f/format-needs-mapping/bad.py | 1 | 2| 0 | 13832 | 87077 | 
| 46 | 45 pylint/checkers/design_analysis.py | 7 | 91| 704 | 14536 | 91948 | 
| 47 | 46 script/get_unused_message_id_category.py | 1 | 38| 297 | 14833 | 92310 | 
| 48 | 47 doc/data/messages/m/missing-format-argument-key/good.py | 1 | 2| 0 | 14833 | 92331 | 
| 49 | 48 doc/data/messages/c/consider-using-dict-items/good.py | 1 | 11| 0 | 14833 | 92393 | 
| 50 | 49 doc/data/messages/c/consider-using-any-or-all/good.py | 2 | 9| 0 | 14833 | 92458 | 
| 51 | 50 doc/conf.py | 5 | 55| 403 | 15236 | 95137 | 
| 52 | 51 pylint/config/__init__.py | 5 | 37| 273 | 15509 | 95710 | 
| 53 | 52 doc/data/messages/b/bad-exception-cause/good.py | 1 | 8| 0 | 15509 | 95766 | 
| 54 | 53 pylint/config/arguments_manager.py | 7 | 56| 284 | 15793 | 102051 | 
| 55 | 54 pylint/reporters/json_reporter.py | 7 | 43| 218 | 16011 | 102874 | 
| 56 | 55 doc/data/messages/c/config-parse-error/good.py | 1 | 2| 0 | 16011 | 102886 | 
| 57 | 56 doc/data/messages/u/unexpected-line-ending-format/good.py | 1 | 2| 0 | 16011 | 102898 | 
| 58 | 57 doc/data/messages/a/astroid-error/good.py | 1 | 2| 0 | 16011 | 102910 | 
| 59 | 58 pylint/testutils/constants.py | 5 | 30| 280 | 16291 | 103255 | 
| 60 | 59 doc/data/messages/u/unexpected-special-method-signature/good.py | 1 | 7| 0 | 16291 | 103285 | 
| 61 | 60 doc/data/messages/u/unused-argument/bad.py | 1 | 3| 0 | 16291 | 103312 | 
| 62 | 60 pylint/lint/pylinter.py | 5 | 103| 634 | 16925 | 103312 | 
| 63 | 61 doc/data/messages/c/confusing-with-statement/good.py | 1 | 2| 0 | 16925 | 103324 | 
| 64 | 62 doc/data/messages/u/unknown-option-value/bad.py | 1 | 2| 0 | 16925 | 103341 | 
| 65 | 63 doc/data/messages/c/consider-using-dict-comprehension/bad.py | 1 | 4| 0 | 16925 | 103385 | 
| 66 | 64 pylint/checkers/stdlib.py | 98 | 251| 1168 | 18093 | 109787 | 
| 67 | 65 doc/data/messages/u/unused-wildcard-import/bad.py | 1 | 5| 0 | 18093 | 109806 | 
| 68 | 66 doc/data/messages/e/expression-not-assigned/good.py | 1 | 2| 0 | 18093 | 109819 | 
| 69 | 67 doc/data/messages/r/raising-format-tuple/bad.py | 1 | 2| 0 | 18093 | 109844 | 
| 70 | 68 doc/data/messages/i/invalid-format-index/bad.py | 1 | 3| 0 | 18093 | 109884 | 
| 71 | 69 doc/data/messages/u/unused-format-string-argument/good.py | 1 | 4| 0 | 18093 | 109926 | 
| 72 | 70 doc/data/messages/u/unexpected-special-method-signature/bad.py | 1 | 7| 0 | 18093 | 109972 | 
| 73 | 71 pylint/testutils/_primer/primer_run_command.py | 5 | 26| 127 | 18220 | 110891 | 
| 74 | 72 doc/data/messages/i/invalid-getnewargs-returned/good.py | 1 | 2| 0 | 18220 | 110903 | 
| 75 | 73 doc/data/messages/g/global-at-module-level/bad.py | 1 | 3| 0 | 18220 | 110918 | 
| 76 | 74 doc/data/messages/c/consider-using-in/good.py | 1 | 3| 0 | 18220 | 110938 | 
| 77 | 75 doc/data/messages/m/missing-parentheses-for-call-in-test/good.py | 1 | 2| 0 | 18220 | 110950 | 
| 78 | 76 doc/data/messages/m/missing-format-attribute/bad.py | 1 | 2| 0 | 18220 | 110967 | 
| 79 | 77 doc/data/messages/c/consider-using-in/bad.py | 1 | 3| 0 | 18220 | 111000 | 
| 80 | 78 doc/data/messages/u/unsupported-assignment-operation/good.py | 1 | 2| 0 | 18220 | 111012 | 
| 81 | 79 doc/data/messages/i/inconsistent-quotes/good.py | 1 | 2| 0 | 18220 | 111024 | 
| 82 | 80 pylint/config/_pylint_config/help_message.py | 8 | 29| 180 | 18400 | 111461 | 
| **-> 83 <-** | **80 pylint/reporters/text.py** | 152 | 186| 335 | 18735 | 111461 | 
| 84 | 81 doc/data/messages/u/undefined-variable/good.py | 1 | 3| 0 | 18735 | 111472 | 
| 85 | 82 doc/data/messages/n/nonlocal-and-global/bad.py | 1 | 12| 0 | 18735 | 111522 | 
| 86 | 83 doc/data/messages/c/consider-using-any-or-all/bad.py | 1 | 15| 0 | 18735 | 111623 | 
| 87 | 84 doc/data/messages/b/bad-configuration-section/good.py | 1 | 2| 0 | 18735 | 111635 | 
| 88 | 85 doc/data/messages/i/invalid-format-index/good.py | 1 | 3| 0 | 18735 | 111668 | 
| 89 | 86 doc/data/messages/e/expression-not-assigned/bad.py | 1 | 2| 0 | 18735 | 111684 | 
| 90 | 87 doc/data/messages/c/consider-using-dict-comprehension/good.py | 1 | 4| 0 | 18735 | 111715 | 
| 91 | 88 doc/data/messages/i/import-outside-toplevel/bad.py | 1 | 4| 0 | 18735 | 111738 | 
| 92 | 89 pylint/checkers/unsupported_version.py | 64 | 85| 171 | 18906 | 112410 | 
| 93 | 90 doc/data/messages/u/unused-import/bad.py | 1 | 5| 0 | 18906 | 112431 | 
| 94 | 90 pylint/checkers/stdlib.py | 47 | 95| 587 | 19493 | 112431 | 
| 95 | 91 doc/data/messages/i/invalid-envvar-default/good.py | 1 | 4| 0 | 19493 | 112445 | 
| 96 | 92 doc/data/messages/f/format-needs-mapping/good.py | 1 | 2| 0 | 19493 | 112467 | 
| 97 | 93 doc/data/messages/u/unused-format-string-key/good.py | 1 | 2| 0 | 19493 | 112479 | 
| 98 | 94 pylint/config/_pylint_config/setup.py | 8 | 26| 108 | 19601 | 112833 | 
| 99 | 95 pylint/utils/file_state.py | 5 | 31| 125 | 19726 | 115205 | 
| 100 | 96 doc/data/messages/u/unused-argument/good.py | 1 | 3| 0 | 19726 | 115225 | 
| 101 | 97 doc/data/messages/t/too-few-format-args/good.py | 1 | 2| 0 | 19726 | 115246 | 
| 102 | 98 doc/data/messages/i/import-error/good.py | 1 | 2| 0 | 19726 | 115258 | 
| 103 | 99 doc/data/messages/u/undefined-variable/bad.py | 1 | 2| 0 | 19726 | 115270 | 
| 104 | 100 doc/data/messages/u/used-prior-global-declaration/bad.py | 1 | 8| 0 | 19726 | 115314 | 
| 105 | 101 pylint/config/callback_actions.py | 153 | 166| 121 | 19847 | 118308 | 
| 106 | 102 doc/data/messages/u/used-prior-global-declaration/good.py | 1 | 7| 0 | 19847 | 118337 | 
| 107 | 103 doc/data/messages/m/missing-format-string-key/good.py | 1 | 2| 0 | 19847 | 118349 | 
| 108 | **103 pylint/reporters/text.py** | 114 | 149| 330 | 20177 | 118349 | 
| 109 | 104 doc/data/messages/u/unknown-option-value/good.py | 1 | 2| 0 | 20177 | 118358 | 
| 110 | 105 doc/data/messages/m/missing-format-attribute/good.py | 1 | 2| 0 | 20177 | 118367 | 
| 111 | 106 doc/data/messages/s/suppressed-message/good.py | 1 | 2| 0 | 20177 | 118379 | 
| 112 | 107 doc/data/messages/r/raising-bad-type/good.py | 1 | 11| 0 | 20177 | 118460 | 
| 113 | 108 doc/data/messages/i/invalid-str-returned/good.py | 1 | 2| 0 | 20177 | 118472 | 
| 114 | 109 doc/data/messages/n/no-self-use/bad.py | 1 | 11| 0 | 20177 | 118533 | 
| 115 | 110 doc/data/messages/t/too-few-format-args/bad.py | 1 | 2| 0 | 20177 | 118561 | 
| 116 | 111 doc/data/messages/f/format-combined-specification/bad.py | 1 | 2| 0 | 20177 | 118583 | 
| 117 | 112 doc/data/messages/f/format-string-without-interpolation/bad.py | 1 | 2| 0 | 20177 | 118601 | 
| 118 | 113 pylint/extensions/typing.py | 81 | 156| 733 | 20910 | 122472 | 
| 119 | 113 pylint/checkers/stdlib.py | 340 | 444| 1106 | 22016 | 122472 | 
| 120 | 114 doc/data/messages/i/invalid-star-assignment-target/bad.py | 1 | 2| 0 | 22016 | 122493 | 
| 121 | 115 doc/data/messages/i/invalid-star-assignment-target/good.py | 1 | 2| 0 | 22016 | 122504 | 
| 122 | 116 pylint/checkers/base_checker.py | 5 | 32| 166 | 22182 | 124856 | 
| 123 | 116 pylint/checkers/stdlib.py | 254 | 307| 277 | 22459 | 124856 | 
| 124 | 116 pylint/checkers/stdlib.py | 7 | 45| 344 | 22803 | 124856 | 
| 125 | 117 doc/data/messages/n/nonlocal-and-global/good.py | 1 | 11| 0 | 22803 | 124893 | 
| 126 | 118 doc/data/messages/r/raising-format-tuple/good.py | 1 | 2| 0 | 22803 | 124911 | 
| 127 | 119 doc/data/messages/r/redundant-keyword-arg/bad.py | 1 | 6| 0 | 22803 | 124941 | 
| 128 | 120 doc/data/messages/u/unidiomatic-typecheck/bad.py | 1 | 4| 0 | 22803 | 124977 | 
| 129 | 121 doc/data/messages/g/global-at-module-level/good.py | 1 | 2| 0 | 22803 | 124982 | 
| 130 | 122 doc/data/messages/u/undefined-all-variable/bad.py | 1 | 5| 0 | 22803 | 125008 | 
| 131 | 123 doc/data/messages/b/bad-exception-cause/bad.py | 1 | 8| 0 | 22803 | 125070 | 
| 132 | 124 doc/data/messages/f/format-string-without-interpolation/good.py | 1 | 2| 0 | 22803 | 125079 | 
| 133 | 125 pylint/typing.py | 7 | 135| 699 | 23502 | 125850 | 
| 134 | 126 doc/data/messages/c/consider-using-get/bad.py | 1 | 7| 0 | 23502 | 125900 | 
| 135 | 127 doc/data/messages/u/unnecessary-list-index-lookup/bad.py | 1 | 5| 0 | 23502 | 125936 | 
| 136 | 127 pylint/lint/pylinter.py | 271 | 358| 813 | 24315 | 125936 | 
| 137 | 128 doc/data/messages/r/redefined-argument-from-local/good.py | 1 | 4| 0 | 24315 | 125983 | 
| 138 | 129 doc/data/messages/u/unspecified-encoding/bad.py | 1 | 4| 0 | 24315 | 126011 | 
| 139 | 130 doc/data/messages/n/no-self-use/good.py | 1 | 16| 0 | 24315 | 126090 | 
| 140 | 131 doc/data/messages/i/invalid-envvar-default/bad.py | 1 | 4| 0 | 24315 | 126112 | 
| 141 | 132 doc/data/messages/u/undefined-all-variable/good.py | 1 | 5| 0 | 24315 | 126131 | 
| 142 | 133 doc/data/messages/u/unused-wildcard-import/good.py | 1 | 5| 0 | 24315 | 126142 | 
| 143 | 134 doc/data/messages/g/global-variable-not-assigned/good.py | 1 | 7| 0 | 24315 | 126171 | 
| 144 | 135 doc/data/messages/u/unneeded-not/good.py | 1 | 3| 0 | 24315 | 126177 | 
| 145 | 135 doc/conf.py | 285 | 304| 185 | 24500 | 126177 | 
| 146 | 136 doc/data/messages/u/unused-import/good.py | 1 | 4| 0 | 24500 | 126188 | 
| 147 | 137 doc/data/messages/c/consider-using-get/good.py | 1 | 4| 0 | 24500 | 126217 | 
| 148 | 138 doc/data/messages/i/import-outside-toplevel/good.py | 1 | 6| 0 | 24500 | 126231 | 
| 149 | 139 doc/data/messages/t/too-many-format-args/good.py | 1 | 2| 0 | 24500 | 126252 | 
| 150 | 139 pylint/lint/utils.py | 66 | 85| 144 | 24644 | 126252 | 
| 151 | 140 doc/data/messages/n/not-a-mapping/good.py | 1 | 2| 0 | 24644 | 126264 | 
| 152 | 141 doc/data/messages/u/unspecified-encoding/good.py | 1 | 4| 0 | 24644 | 126290 | 
| 153 | 142 doc/data/messages/g/global-variable-not-assigned/bad.py | 1 | 7| 0 | 24644 | 126323 | 
| 154 | 143 doc/data/messages/u/unidiomatic-typecheck/good.py | 1 | 4| 0 | 24644 | 126349 | 
| 155 | 144 doc/data/messages/t/too-many-format-args/bad.py | 1 | 2| 0 | 24644 | 126382 | 
| 156 | 145 doc/data/messages/m/misplaced-format-function/bad.py | 1 | 2| 0 | 24644 | 126400 | 
| 157 | 146 pylint/exceptions.py | 8 | 54| 299 | 24943 | 126775 | 
| 158 | 147 doc/data/messages/m/missing-kwoa/good.py | 1 | 2| 0 | 24943 | 126787 | 
| 159 | 148 doc/data/messages/m/mixed-format-string/bad.py | 1 | 2| 0 | 24943 | 126813 | 
| 160 | 149 doc/data/messages/i/invalid-getnewargs-ex-returned/good.py | 1 | 2| 0 | 24943 | 126825 | 
| 161 | 150 doc/data/messages/c/consider-using-alias/good.py | 1 | 2| 0 | 24943 | 126837 | 
| 162 | 151 doc/exts/pylint_messages.py | 290 | 316| 268 | 25211 | 130270 | 
| 163 | 152 doc/data/messages/u/use-a-generator/bad.py | 1 | 5| 0 | 25211 | 130329 | 
| 164 | 153 doc/data/messages/c/consider-using-tuple/bad.py | 1 | 3| 0 | 25211 | 130353 | 
| 165 | 154 doc/data/messages/w/wildcard-import/bad.py | 1 | 2| 0 | 25211 | 130364 | 
| 166 | 155 doc/data/messages/r/raising-bad-type/bad.py | 1 | 11| 0 | 25211 | 130443 | 
| 167 | 156 doc/data/messages/a/anomalous-unicode-escape-in-string/good.py | 1 | 2| 0 | 25211 | 130456 | 
| 168 | 157 doc/data/messages/n/nonlocal-without-binding/good.py | 1 | 2| 0 | 25211 | 130468 | 
| 169 | 158 doc/data/messages/u/using-constant-test/bad.py | 1 | 5| 0 | 25211 | 130510 | 
| 170 | 159 doc/data/messages/c/consider-using-min-builtin/bad.py | 1 | 8| 0 | 25211 | 130557 | 
| 171 | 159 pylint/lint/message_state_handler.py | 216 | 234| 167 | 25378 | 130557 | 
| 172 | 160 doc/data/messages/l/locally-disabled/good.py | 1 | 2| 0 | 25378 | 130569 | 
| 173 | 161 doc/data/messages/i/invalid-repr-returned/good.py | 1 | 2| 0 | 25378 | 130581 | 
| 174 | 162 doc/data/messages/m/missing-type-doc/bad.py | 1 | 7| 0 | 25378 | 130629 | 
| 175 | 162 pylint/checkers/base_checker.py | 193 | 228| 375 | 25753 | 130629 | 
| 176 | 163 doc/data/messages/r/redefined-outer-name/good.py | 1 | 2| 0 | 25753 | 130641 | 
| 177 | 164 doc/data/messages/r/redefined-variable-type/good.py | 1 | 3| 0 | 25753 | 130651 | 
| 178 | 164 pylint/constants.py | 105 | 144| 394 | 26147 | 130651 | 
| 179 | 165 doc/data/messages/g/global-statement/bad.py | 1 | 12| 0 | 26147 | 130684 | 
| 180 | 166 doc/data/messages/c/consider-using-set-comprehension/bad.py | 1 | 4| 0 | 26147 | 130739 | 
| 181 | 167 doc/data/messages/l/logging-too-few-args/good.py | 1 | 8| 0 | 26147 | 130771 | 
| 182 | 168 doc/data/messages/r/raising-non-exception/bad.py | 1 | 2| 0 | 26147 | 130781 | 
| 183 | 168 pylint/config/callback_actions.py | 169 | 224| 365 | 26512 | 130781 | 
| 184 | 169 doc/data/messages/r/redefined-argument-from-local/bad.py | 1 | 4| 0 | 26512 | 130833 | 
| 185 | 170 pylint/__init__.py | 5 | 77| 456 | 26968 | 131679 | 
| 186 | 171 doc/data/messages/n/no-self-argument/bad.py | 1 | 4| 0 | 26968 | 131705 | 
| 187 | 171 pylint/checkers/design_analysis.py | 92 | 170| 510 | 27478 | 131705 | 
| 188 | 172 doc/data/messages/w/wrong-import-order/bad.py | 1 | 5| 0 | 27478 | 131731 | 
| 189 | 173 doc/data/messages/b/bad-string-format-type/bad.py | 1 | 2| 0 | 27478 | 131747 | 
| 190 | 174 doc/data/messages/e/eval-used/bad.py | 1 | 2| 0 | 27478 | 131763 | 
| 191 | 175 doc/data/messages/b/bad-string-format-type/good.py | 1 | 2| 0 | 27478 | 131771 | 
| 192 | 176 doc/data/messages/r/redefined-variable-type/bad.py | 1 | 3| 0 | 27478 | 131789 | 
| 193 | 177 doc/data/messages/u/useless-import-alias/bad.py | 1 | 2| 0 | 27478 | 131802 | 
| 194 | 178 doc/data/messages/m/missing-type-doc/good.py | 1 | 7| 0 | 27478 | 131845 | 
| 195 | 179 doc/data/messages/n/non-ascii-module-import/good.py | 1 | 2| 0 | 27478 | 131857 | 
| 196 | 180 doc/data/messages/c/consider-using-tuple/good.py | 1 | 3| 0 | 27478 | 131873 | 
| 197 | 181 doc/data/messages/u/unnecessary-list-index-lookup/good.py | 1 | 5| 0 | 27478 | 131897 | 
| 198 | 182 doc/data/messages/a/assignment-from-none/bad.py | 1 | 6| 0 | 27478 | 131915 | 
| 199 | 183 doc/data/messages/n/no-self-argument/good.py | 1 | 4| 0 | 27478 | 131933 | 
| 200 | 184 doc/data/messages/d/deprecated-argument/good.py | 1 | 2| 0 | 27478 | 131945 | 
| 201 | 185 doc/data/messages/w/wrong-import-order/good.py | 1 | 7| 0 | 27478 | 131959 | 
| 202 | 186 doc/data/messages/l/logging-format-interpolation/bad.py | 1 | 5| 0 | 27478 | 131984 | 
| 203 | 187 doc/data/messages/d/differing-type-doc/good.py | 1 | 2| 0 | 27478 | 131996 | 
| 204 | 188 doc/data/messages/u/useless-suppression/good.py | 1 | 2| 0 | 27478 | 132008 | 
| 205 | 189 doc/data/messages/c/consider-using-assignment-expr/good.py | 1 | 2| 0 | 27478 | 132020 | 
| 206 | 190 doc/data/messages/u/useless-import-alias/good.py | 1 | 2| 0 | 27478 | 132025 | 
| 207 | 191 doc/data/messages/i/invalid-length-returned/bad.py | 1 | 7| 0 | 27478 | 132076 | 
| 208 | 192 doc/data/messages/e/eval-used/good.py | 1 | 4| 0 | 27478 | 132093 | 
| 209 | 193 doc/data/messages/a/anomalous-unicode-escape-in-string/bad.py | 1 | 2| 0 | 27478 | 132118 | 
| 210 | 194 doc/data/messages/w/while-used/good.py | 1 | 11| 0 | 27478 | 132169 | 
| 211 | 195 doc/data/messages/d/disallowed-name/good.py | 1 | 2| 0 | 27478 | 132181 | 
| 212 | 196 doc/data/messages/m/misplaced-bare-raise/bad.py | 1 | 4| 0 | 27478 | 132205 | 
| 213 | 197 doc/data/messages/u/using-f-string-in-unsupported-version/good.py | 1 | 2| 0 | 27478 | 132217 | 
| 214 | 198 doc/data/messages/w/while-used/bad.py | 1 | 13| 0 | 27478 | 132282 | 
| 215 | 199 doc/data/messages/d/dangerous-default-value/bad.py | 1 | 4| 0 | 27478 | 132315 | 
| 216 | 200 doc/data/messages/b/bad-format-string-key/good.py | 1 | 2| 0 | 27478 | 132327 | 
| 217 | 201 doc/data/messages/p/pointless-statement/bad.py | 1 | 2| 0 | 27478 | 132344 | 
| 218 | 202 doc/data/messages/c/consider-using-from-import/good.py | 1 | 2| 0 | 27478 | 132349 | 
| 219 | 203 doc/data/messages/i/invalid-length-returned/good.py | 1 | 7| 0 | 27478 | 132391 | 
| 220 | 204 doc/data/messages/l/logging-too-few-args/bad.py | 1 | 8| 0 | 27478 | 132430 | 
| 221 | 205 doc/data/messages/u/useless-option-value/bad.py | 1 | 4| 0 | 27478 | 132475 | 
| 222 | 206 doc/data/messages/p/preferred-module/good.py | 1 | 2| 0 | 27478 | 132487 | 
| 223 | 207 doc/data/messages/n/no-value-for-parameter/good.py | 1 | 5| 0 | 27478 | 132506 | 
| 224 | 208 doc/data/messages/p/potential-index-error/bad.py | 1 | 2| 0 | 27478 | 132525 | 
| 225 | 209 doc/data/messages/r/redundant-keyword-arg/good.py | 1 | 8| 0 | 27478 | 132547 | 
| 226 | 210 doc/data/messages/a/assert-on-string-literal/bad.py | 1 | 4| 0 | 27478 | 132580 | 


### Hint

```
Subsequently, there is also this behavior with the quotes
\`\`\`
$ pylint test.py --msg-template='"Category": "{category}"'
************* Module test
Category": "convention
Category": "error
Category": "error
Category": "convention
Category": "convention
Category": "error

$ pylint test.py --msg-template='""Category": "{category}""'
************* Module test
"Category": "convention"
"Category": "error"
"Category": "error"
"Category": "convention"
"Category": "convention"
"Category": "error"
\`\`\`
Commit that changed the behavior was probably this one: https://github.com/PyCQA/pylint/commit/7c3533ca48e69394391945de1563ef7f639cd27d#diff-76025f0bc82e83cb406321006fbca12c61a10821834a3164620fc17c978f9b7e

And I tested on 2.11.1 that it is working as intended on that version.
Thanks for digging into this !
```

## Patch

```diff
diff --git a/pylint/reporters/text.py b/pylint/reporters/text.py
--- a/pylint/reporters/text.py
+++ b/pylint/reporters/text.py
@@ -175,7 +175,7 @@ def on_set_current_module(self, module: str, filepath: str | None) -> None:
         self._template = template
 
         # Check to see if all parameters in the template are attributes of the Message
-        arguments = re.findall(r"\{(.+?)(:.*)?\}", template)
+        arguments = re.findall(r"\{(\w+?)(:.*)?\}", template)
         for argument in arguments:
             if argument[0] not in MESSAGE_FIELDS:
                 warnings.warn(

```

## Test Patch

```diff
diff --git a/tests/reporters/unittest_reporting.py b/tests/reporters/unittest_reporting.py
--- a/tests/reporters/unittest_reporting.py
+++ b/tests/reporters/unittest_reporting.py
@@ -14,6 +14,7 @@
 from typing import TYPE_CHECKING
 
 import pytest
+from _pytest.recwarn import WarningsRecorder
 
 from pylint import checkers
 from pylint.interfaces import HIGH
@@ -88,16 +89,12 @@ def test_template_option_non_existing(linter) -> None:
     """
     output = StringIO()
     linter.reporter.out = output
-    linter.config.msg_template = (
-        "{path}:{line}:{a_new_option}:({a_second_new_option:03d})"
-    )
+    linter.config.msg_template = "{path}:{line}:{categ}:({a_second_new_option:03d})"
     linter.open()
     with pytest.warns(UserWarning) as records:
         linter.set_current_module("my_mod")
         assert len(records) == 2
-        assert (
-            "Don't recognize the argument 'a_new_option'" in records[0].message.args[0]
-        )
+        assert "Don't recognize the argument 'categ'" in records[0].message.args[0]
     assert (
         "Don't recognize the argument 'a_second_new_option'"
         in records[1].message.args[0]
@@ -113,7 +110,24 @@ def test_template_option_non_existing(linter) -> None:
     assert out_lines[2] == "my_mod:2::()"
 
 
-def test_deprecation_set_output(recwarn):
+def test_template_option_with_header(linter: PyLinter) -> None:
+    output = StringIO()
+    linter.reporter.out = output
+    linter.config.msg_template = '{{ "Category": "{category}" }}'
+    linter.open()
+    linter.set_current_module("my_mod")
+
+    linter.add_message("C0301", line=1, args=(1, 2))
+    linter.add_message(
+        "line-too-long", line=2, end_lineno=2, end_col_offset=4, args=(3, 4)
+    )
+
+    out_lines = output.getvalue().split("\n")
+    assert out_lines[1] == '{ "Category": "convention" }'
+    assert out_lines[2] == '{ "Category": "convention" }'
+
+
+def test_deprecation_set_output(recwarn: WarningsRecorder) -> None:
     """TODO remove in 3.0."""
     reporter = BaseReporter()
     # noinspection PyDeprecation

```


## Code snippets

### 1 - pylint/lint/pylinter.py:

Start line: 104, End line: 235

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
### 2 - pylint/constants.py:

Start line: 5, End line: 102

```python
from __future__ import annotations

import os
import pathlib
import platform
import sys
from datetime import datetime

import astroid
import platformdirs

from pylint.__pkginfo__ import __version__
from pylint.typing import MessageTypesFullName

PY38_PLUS = sys.version_info[:2] >= (3, 8)
PY39_PLUS = sys.version_info[:2] >= (3, 9)

IS_PYPY = platform.python_implementation() == "PyPy"

PY_EXTS = (".py", ".pyc", ".pyo", ".pyw", ".so", ".dll")

MSG_STATE_CONFIDENCE = 2
_MSG_ORDER = "EWRCIF"
MSG_STATE_SCOPE_CONFIG = 0
MSG_STATE_SCOPE_MODULE = 1

# The line/node distinction does not apply to fatal errors and reports.
_SCOPE_EXEMPT = "FR"

MSG_TYPES: dict[str, MessageTypesFullName] = {
    "I": "info",
    "C": "convention",
    "R": "refactor",
    "W": "warning",
    "E": "error",
    "F": "fatal",
}
MSG_TYPES_LONG: dict[str, str] = {v: k for k, v in MSG_TYPES.items()}

MSG_TYPES_STATUS = {"I": 0, "C": 16, "R": 8, "W": 4, "E": 2, "F": 1}

# You probably don't want to change the MAIN_CHECKER_NAME
# This would affect rcfile generation and retro-compatibility
# on all project using [MAIN] in their rcfile.
MAIN_CHECKER_NAME = "main"

USER_HOME = os.path.expanduser("~")
# TODO: 3.0: Remove in 3.0 with all the surrounding code
OLD_DEFAULT_PYLINT_HOME = ".pylint.d"
DEFAULT_PYLINT_HOME = platformdirs.user_cache_dir("pylint")

DEFAULT_IGNORE_LIST = ("CVS",)


class WarningScope:
    LINE = "line-based-msg"
    NODE = "node-based-msg"


full_version = f"""pylint {__version__}
astroid {astroid.__version__}
Python {sys.version}"""

HUMAN_READABLE_TYPES = {
    "file": "file",
    "module": "module",
    "const": "constant",
    "class": "class",
    "function": "function",
    "method": "method",
    "attr": "attribute",
    "argument": "argument",
    "variable": "variable",
    "class_attribute": "class attribute",
    "class_const": "class constant",
    "inlinevar": "inline iteration",
    "typevar": "type variable",
}

# ignore some messages when emitting useless-suppression:
# - cyclic-import: can show false positives due to incomplete context
# - deprecated-{module, argument, class, method, decorator}:
#   can cause false positives for multi-interpreter projects
#   when linting with an interpreter on a lower python version
INCOMPATIBLE_WITH_USELESS_SUPPRESSION = frozenset(
    [
        "R0401",  # cyclic-import
        "W0402",  # deprecated-module
        "W1505",  # deprecated-method
        "W1511",  # deprecated-argument
        "W1512",  # deprecated-class
        "W1513",  # deprecated-decorator
        "R0801",  # duplicate-code
    ]
)


TYPING_TYPE_CHECKS_GUARDS = frozenset({"typing.TYPE_CHECKING", "TYPE_CHECKING"})
```
### 3 - pylint/message/_deleted_message_ids.py:

Start line: 5, End line: 126

```python
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple


class DeletedMessage(NamedTuple):
    msgid: str
    symbol: str
    old_names: list[tuple[str, str]] = []


DELETED_MSGID_PREFIXES: list[int] = []

DELETED_MESSAGES_IDS = {
    # Everything until the next comment is from the PY3K+ checker
    "https://github.com/PyCQA/pylint/pull/4942": [
        DeletedMessage("W1601", "apply-builtin"),
        DeletedMessage("E1601", "print-statement"),
        DeletedMessage("E1602", "parameter-unpacking"),
        DeletedMessage(
            "E1603", "unpacking-in-except", [("W0712", "old-unpacking-in-except")]
        ),
        DeletedMessage(
            "E1604", "old-raise-syntax", [("W0121", "old-old-raise-syntax")]
        ),
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
    ],
    "https://github.com/PyCQA/pylint/pull/3578": [
        DeletedMessage("W0312", "mixed-indentation"),
    ],
    "https://github.com/PyCQA/pylint/pull/3577": [
        DeletedMessage(
            "C0326",
            "bad-whitespace",
            [
                ("C0323", "no-space-after-operator"),
                ("C0324", "no-space-after-comma"),
                ("C0322", "no-space-before-operator"),
            ],
        ),
    ],
    "https://github.com/PyCQA/pylint/pull/3571": [
        DeletedMessage("C0330", "bad-continuation")
    ],
    "https://pylint.pycqa.org/en/latest/whatsnew/1/1.4.html#what-s-new-in-pylint-1-4-3": [
        DeletedMessage("R0921", "abstract-class-not-used"),
        DeletedMessage("R0922", "abstract-class-little-used"),
        DeletedMessage("W0142", "star-args"),
    ],
    "https://github.com/PyCQA/pylint/issues/2409": [
        DeletedMessage("W0232", "no-init"),
    ],
    "https://github.com/PyCQA/pylint/pull/6421": [
        DeletedMessage("W0111", "assign-to-new-keyword"),
    ],
}
```
### 4 - pylint/checkers/logging.py:

Start line: 7, End line: 94

```python
from __future__ import annotations

import string
import sys
from typing import TYPE_CHECKING

import astroid
from astroid import bases, nodes
from astroid.typing import InferenceResult

from pylint import checkers
from pylint.checkers import utils
from pylint.checkers.utils import infer_all
from pylint.typing import MessageDefinitionTuple

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

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
```
### 5 - pylint/checkers/format.py:

Start line: 14, End line: 117

```python
from __future__ import annotations

import sys
import tokenize
from functools import reduce
from re import Match
from typing import TYPE_CHECKING

from astroid import nodes

from pylint.checkers import BaseRawFileChecker, BaseTokenChecker
from pylint.checkers.utils import only_required_for_messages
from pylint.constants import WarningScope
from pylint.interfaces import HIGH
from pylint.typing import MessageDefinitionTuple
from pylint.utils.pragma_parser import OPTION_PO, PragmaParserError, parse_pragma

if TYPE_CHECKING:
    from pylint.lint import PyLinter

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_KEYWORD_TOKENS = {
    "assert",
    "del",
    "elif",
    "except",
    "for",
    "if",
    "in",
    "not",
    "raise",
    "return",
    "while",
    "yield",
    "with",
}
_JUNK_TOKENS = {tokenize.COMMENT, tokenize.NL}


MSGS: dict[str, MessageDefinitionTuple] = {
    "C0301": (
        "Line too long (%s/%s)",
        "line-too-long",
        "Used when a line is longer than a given number of characters.",
    ),
    "C0302": (
        "Too many lines in module (%s/%s)",  # was W0302
        "too-many-lines",
        "Used when a module has too many lines, reducing its readability.",
    ),
    "C0303": (
        "Trailing whitespace",
        "trailing-whitespace",
        "Used when there is whitespace between the end of a line and the newline.",
    ),
    "C0304": (
        "Final newline missing",
        "missing-final-newline",
        "Used when the last line in a file is missing a newline.",
    ),
    "C0305": (
        "Trailing newlines",
        "trailing-newlines",
        "Used when there are trailing blank lines in a file.",
    ),
    "W0311": (
        "Bad indentation. Found %s %s, expected %s",
        "bad-indentation",
        "Used when an unexpected number of indentation's tabulations or "
        "spaces has been found.",
    ),
    "W0301": (
        "Unnecessary semicolon",  # was W0106
        "unnecessary-semicolon",
        'Used when a statement is ended by a semi-colon (";"), which '
        "isn't necessary (that's python, not C ;).",
    ),
    "C0321": (
        "More than one statement on a single line",
        "multiple-statements",
        "Used when more than on statement are found on the same line.",
        {"scope": WarningScope.NODE},
    ),
    "C0325": (
        "Unnecessary parens after %r keyword",
        "superfluous-parens",
        "Used when a single item in parentheses follows an if, for, or "
        "other keyword.",
    ),
    "C0327": (
        "Mixed line endings LF and CRLF",
        "mixed-line-endings",
        "Used when there are mixed (LF and CRLF) newline signs in a file.",
    ),
    "C0328": (
        "Unexpected line ending format. There is '%s' while it should be '%s'.",
        "unexpected-line-ending-format",
        "Used when there is different newline than expected.",
    ),
}
```
### 6 - pylint/message/message.py:

Start line: 5, End line: 93

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from warnings import warn

from pylint.constants import MSG_TYPES
from pylint.interfaces import UNDEFINED, Confidence
from pylint.typing import MessageLocationTuple


@dataclass(unsafe_hash=True)
class Message:  # pylint: disable=too-many-instance-attributes
    """This class represent a message to be issued by the reporters."""

    msg_id: str
    symbol: str
    msg: str
    C: str
    category: str
    confidence: Confidence
    abspath: str
    path: str
    module: str
    obj: str
    line: int
    column: int
    end_line: int | None
    end_column: int | None

    def __init__(
        self,
        msg_id: str,
        symbol: str,
        location: tuple[str, str, str, str, int, int] | MessageLocationTuple,
        msg: str,
        confidence: Confidence | None,
    ) -> None:
        if not isinstance(location, MessageLocationTuple):
            warn(
                "In pylint 3.0, Messages will only accept a MessageLocationTuple as location parameter",
                DeprecationWarning,
            )
            location = MessageLocationTuple(
                location[0],
                location[1],
                location[2],
                location[3],
                location[4],
                location[5],
                None,
                None,
            )

        self.msg_id = msg_id
        self.symbol = symbol
        self.msg = msg
        self.C = msg_id[0]
        self.category = MSG_TYPES[msg_id[0]]
        self.confidence = confidence or UNDEFINED
        self.abspath = location.abspath
        self.path = location.path
        self.module = location.module
        self.obj = location.obj
        self.line = location.line
        self.column = location.column
        self.end_line = location.end_line
        self.end_column = location.end_column

    def format(self, template: str) -> str:
        """Format the message according to the given template.

        The template format is the one of the format method :
        cf. https://docs.python.org/2/library/string.html#formatstrings
        """
        return template.format(**asdict(self))

    @property
    def location(self) -> MessageLocationTuple:
        return MessageLocationTuple(
            self.abspath,
            self.path,
            self.module,
            self.obj,
            self.line,
            self.column,
            self.end_line,
            self.end_column,
        )
```
### 7 - doc/data/messages/c/consider-using-f-string/bad.py:

Start line: 1, End line: 11

```python
from string import Template

menu = ('eggs', 'spam', 42.4)

old_order = "%s and %s: %.2f ¤" % menu # [consider-using-f-string]
beginner_order = menu[0] + " and " + menu[1] + ": " + str(menu[2]) + " ¤"
joined_order = " and ".join(menu[:2])
format_order = "{} and {}: {:0.2f} ¤".format(menu[0], menu[1], menu[2]) # [consider-using-f-string]
named_format_order = "{eggs} and {spam}: {price:0.2f} ¤".format(eggs=menu[0], spam=menu[1], price=menu[2]) # [consider-using-f-string]
template_order = Template('$eggs and $spam: $price ¤').substitute(eggs=menu[0], spam=menu[1], price=menu[2])
```
### 8 - pylint/checkers/imports.py:

Start line: 215, End line: 305

```python
MSGS: dict[str, MessageDefinitionTuple] = {
    "E0401": (
        "Unable to import %s",
        "import-error",
        "Used when pylint has been unable to import a module.",
        {"old_names": [("F0401", "old-import-error")]},
    ),
    "E0402": (
        "Attempted relative import beyond top-level package",
        "relative-beyond-top-level",
        "Used when a relative import tries to access too many levels "
        "in the current package.",
    ),
    "R0401": (
        "Cyclic import (%s)",
        "cyclic-import",
        "Used when a cyclic import between two or more modules is detected.",
    ),
    "R0402": (
        "Use 'from %s import %s' instead",
        "consider-using-from-import",
        "Emitted when a submodule of a package is imported and "
        "aliased with the same name, "
        "e.g., instead of ``import concurrent.futures as futures`` use "
        "``from concurrent import futures``.",
    ),
    "W0401": (
        "Wildcard import %s",
        "wildcard-import",
        "Used when `from module import *` is detected.",
    ),
    "W0404": (
        "Reimport %r (imported line %s)",
        "reimported",
        "Used when a module is reimported multiple times.",
    ),
    "W0406": (
        "Module import itself",
        "import-self",
        "Used when a module is importing itself.",
    ),
    "W0407": (
        "Prefer importing %r instead of %r",
        "preferred-module",
        "Used when a module imported has a preferred replacement module.",
    ),
    "W0410": (
        "__future__ import is not the first non docstring statement",
        "misplaced-future",
        "Python 2.5 and greater require __future__ import to be the "
        "first non docstring statement in the module.",
    ),
    "C0410": (
        "Multiple imports on one line (%s)",
        "multiple-imports",
        "Used when import statement importing multiple modules is detected.",
    ),
    "C0411": (
        "%s should be placed before %s",
        "wrong-import-order",
        "Used when PEP8 import order is not respected (standard imports "
        "first, then third-party libraries, then local imports).",
    ),
    "C0412": (
        "Imports from package %s are not grouped",
        "ungrouped-imports",
        "Used when imports are not grouped by packages.",
    ),
    "C0413": (
        'Import "%s" should be placed at the top of the module',
        "wrong-import-position",
        "Used when code and imports are mixed.",
    ),
    "C0414": (
        "Import alias does not rename original package",
        "useless-import-alias",
        "Used when an import alias is same as original package, "
        "e.g., using import numpy as numpy instead of import numpy as np.",
    ),
    "C0415": (
        "Import outside toplevel (%s)",
        "import-outside-toplevel",
        "Used when an import statement is used anywhere other than the module "
        "toplevel. Move this import to the top of the file.",
    ),
}


DEFAULT_STANDARD_LIBRARY = ()
DEFAULT_KNOWN_THIRD_PARTY = ("enchant",)
DEFAULT_PREFERRED_MODULES = ()
```
### 9 - pylint/reporters/text.py:

Start line: 11, End line: 69

```python
from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, TextIO, cast, overload

from pylint.message import Message
from pylint.reporters import BaseReporter
from pylint.reporters.ureports.text_writer import TextWriter
from pylint.utils import _splitstrip

if TYPE_CHECKING:
    from pylint.lint import PyLinter
    from pylint.reporters.ureports.nodes import Section


class MessageStyle(NamedTuple):
    """Styling of a message."""

    color: str | None
    """The color name (see `ANSI_COLORS` for available values)
    or the color number when 256 colors are available.
    """
    style: tuple[str, ...] = ()
    """Tuple of style strings (see `ANSI_COLORS` for available values)."""


ColorMappingDict = Dict[str, MessageStyle]

TITLE_UNDERLINES = ["", "=", "-", "."]

ANSI_PREFIX = "\033["
ANSI_END = "m"
ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "reset": "0",
    "bold": "1",
    "italic": "3",
    "underline": "4",
    "blink": "5",
    "inverse": "7",
    "strike": "9",
}
ANSI_COLORS = {
    "reset": "0",
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
}

MESSAGE_FIELDS = {i.name for i in fields(Message)}
```
### 10 - pylint/lint/message_state_handler.py:

Start line: 5, End line: 37

```python
from __future__ import annotations

import sys
import tokenize
from collections import defaultdict
from typing import TYPE_CHECKING

from pylint import exceptions, interfaces
from pylint.constants import (
    MSG_STATE_CONFIDENCE,
    MSG_STATE_SCOPE_CONFIG,
    MSG_STATE_SCOPE_MODULE,
    MSG_TYPES,
    MSG_TYPES_LONG,
)
from pylint.interfaces import HIGH
from pylint.message import MessageDefinition
from pylint.typing import ManagedMessage
from pylint.utils.pragma_parser import (
    OPTION_PO,
    InvalidPragmaError,
    UnRecognizedOptionError,
    parse_pragma,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


if TYPE_CHECKING:
    from pylint.lint.pylinter import PyLinter
```
### 83 - pylint/reporters/text.py:

Start line: 152, End line: 186

```python
class TextReporter(BaseReporter):
    """Reports messages and layouts in plain text."""

    name = "text"
    extension = "txt"
    line_format = "{path}:{line}:{column}: {msg_id}: {msg} ({symbol})"

    def __init__(self, output: TextIO | None = None) -> None:
        super().__init__(output)
        self._modules: set[str] = set()
        self._template = self.line_format
        self._fixed_template = self.line_format
        """The output format template with any unrecognized arguments removed."""

    def on_set_current_module(self, module: str, filepath: str | None) -> None:
        """Set the format template to be used and check for unrecognized arguments."""
        template = str(self.linter.config.msg_template or self._template)

        # Return early if the template is the same as the previous one
        if template == self._template:
            return

        # Set template to the currently selected template
        self._template = template

        # Check to see if all parameters in the template are attributes of the Message
        arguments = re.findall(r"\{(.+?)(:.*)?\}", template)
        for argument in arguments:
            if argument[0] not in MESSAGE_FIELDS:
                warnings.warn(
                    f"Don't recognize the argument '{argument[0]}' in the --msg-template. "
                    "Are you sure it is supported on the current version of pylint?"
                )
                template = re.sub(r"\{" + argument[0] + r"(:.*?)?\}", "", template)
        self._fixed_template = template
```
### 108 - pylint/reporters/text.py:

Start line: 114, End line: 149

```python
def colorize_ansi(
    msg: str,
    msg_style: MessageStyle | str | None = None,
    style: str = "",
    **kwargs: str | None,
) -> str:
    r"""colorize message by wrapping it with ANSI escape codes

    :param msg: the message string to colorize

    :param msg_style: the message style
        or color (for backwards compatibility): the color of the message style

    :param style: the message's style elements, this will be deprecated

    :param \**kwargs: used to accept `color` parameter while it is being deprecated

    :return: the ANSI escaped string
    """
    # TODO: 3.0: Remove deprecated typing and only accept MessageStyle as parameter
    if not isinstance(msg_style, MessageStyle):
        warnings.warn(
            "In pylint 3.0, the colorize_ansi function of Text reporters will only accept a MessageStyle parameter",
            DeprecationWarning,
        )
        color = kwargs.get("color")
        style_attrs = tuple(_splitstrip(style))
        msg_style = MessageStyle(color or msg_style, style_attrs)
    # If both color and style are not defined, then leave the text as is
    if msg_style.color is None and len(msg_style.style) == 0:
        return msg
    escape_code = _get_ansi_code(msg_style)
    # If invalid (or unknown) color, don't wrap msg with ANSI codes
    if escape_code:
        return f"{escape_code}{msg}{ANSI_RESET}"
    return msg
```
