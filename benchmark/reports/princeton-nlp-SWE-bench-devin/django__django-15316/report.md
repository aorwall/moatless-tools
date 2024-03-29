# django__django-15316

| **django/django** | `fdfa97fb166ef5065aa2b229f19cb4ce303084e5` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 16282 |
| **Any found context length** | 111 |
| **Avg pos** | 30.5 |
| **Min pos** | 1 |
| **Max pos** | 42 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admindocs/utils.py b/django/contrib/admindocs/utils.py
--- a/django/contrib/admindocs/utils.py
+++ b/django/contrib/admindocs/utils.py
@@ -137,9 +137,10 @@ def default_reference_role(name, rawtext, text, lineno, inliner, options=None, c
     for name, urlbase in ROLES.items():
         create_reference_role(name, urlbase)
 
-# Match the beginning of a named or unnamed group.
+# Match the beginning of a named, unnamed, or non-capturing groups.
 named_group_matcher = _lazy_re_compile(r'\(\?P(<\w+>)')
 unnamed_group_matcher = _lazy_re_compile(r'\(')
+non_capturing_group_matcher = _lazy_re_compile(r'\(\?\:')
 
 
 def replace_metacharacters(pattern):
@@ -210,3 +211,18 @@ def replace_unnamed_groups(pattern):
         final_pattern += pattern[:start] + '<var>'
         prev_end = end
     return final_pattern + pattern[prev_end:]
+
+
+def remove_non_capturing_groups(pattern):
+    r"""
+    Find non-capturing groups in the given `pattern` and remove them, e.g.
+    1. (?P<a>\w+)/b/(?:\w+)c(?:\w+) => (?P<a>\\w+)/b/c
+    2. ^(?:\w+(?:\w+))a => ^a
+    3. ^a(?:\w+)/b(?:\w+) => ^a/b
+    """
+    group_start_end_indices = _find_groups(pattern, non_capturing_group_matcher)
+    final_pattern, prev_end = '', None
+    for start, end, _ in group_start_end_indices:
+        final_pattern += pattern[prev_end:start]
+        prev_end = end
+    return final_pattern + pattern[prev_end:]
diff --git a/django/contrib/admindocs/views.py b/django/contrib/admindocs/views.py
--- a/django/contrib/admindocs/views.py
+++ b/django/contrib/admindocs/views.py
@@ -8,7 +8,8 @@
 from django.contrib.admin.views.decorators import staff_member_required
 from django.contrib.admindocs import utils
 from django.contrib.admindocs.utils import (
-    replace_metacharacters, replace_named_groups, replace_unnamed_groups,
+    remove_non_capturing_groups, replace_metacharacters, replace_named_groups,
+    replace_unnamed_groups,
 )
 from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
 from django.db import models
@@ -410,6 +411,7 @@ def simplify_regex(pattern):
     example, turn "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"
     into "/<sport_slug>/athletes/<athlete_slug>/".
     """
+    pattern = remove_non_capturing_groups(pattern)
     pattern = replace_named_groups(pattern)
     pattern = replace_unnamed_groups(pattern)
     pattern = replace_metacharacters(pattern)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admindocs/utils.py | 140 | 140 | 42 | 10 | 16282
| django/contrib/admindocs/utils.py | 213 | 213 | 18 | 10 | 5174
| django/contrib/admindocs/views.py | 11 | 11 | - | 1 | -
| django/contrib/admindocs/views.py | 413 | 413 | 1 | 1 | 111


## Problem Statement

```
simplify_regex() doesn't handle non-capturing groups
Description
	 
		(last modified by Mariusz Felisiak)
	 
While using Django REST Framework's Schema generator, I found out they're using simplify_regex(); however, current version has a few shortcomings, namely non-capturing groups are broken.
simplify_regex() doesn't handle non-capturing groups
Description
	 
		(last modified by Mariusz Felisiak)
	 
While using Django REST Framework's Schema generator, I found out they're using simplify_regex(); however, current version has a few shortcomings, namely non-capturing groups are broken.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admindocs/views.py** | 407 | 419| 111 | 111 | 3329 | 
| 2 | 2 django/db/migrations/serializer.py | 235 | 246| 133 | 244 | 6008 | 
| 3 | 3 django/utils/regex_helper.py | 78 | 190| 962 | 1206 | 8649 | 
| 4 | 3 django/utils/regex_helper.py | 1 | 38| 250 | 1456 | 8649 | 
| 5 | 3 django/utils/regex_helper.py | 41 | 76| 375 | 1831 | 8649 | 
| 6 | 3 django/utils/regex_helper.py | 286 | 352| 484 | 2315 | 8649 | 
| 7 | 4 django/db/models/lookups.py | 543 | 560| 134 | 2449 | 13913 | 
| 8 | 5 django/contrib/gis/geometry.py | 1 | 18| 215 | 2664 | 14128 | 
| 9 | 6 django/urls/resolvers.py | 185 | 219| 265 | 2929 | 19963 | 
| 10 | 7 django/core/validators.py | 17 | 59| 336 | 3265 | 24506 | 
| 11 | 7 django/urls/resolvers.py | 159 | 183| 203 | 3468 | 24506 | 
| 12 | 8 django/forms/fields.py | 513 | 542| 207 | 3675 | 33927 | 
| 13 | 9 django/utils/jslex.py | 30 | 47| 126 | 3801 | 35712 | 
| 14 | **10 django/contrib/admindocs/utils.py** | 154 | 178| 272 | 4073 | 37449 | 
| 15 | 11 django/utils/http.py | 1 | 37| 481 | 4554 | 40708 | 
| 16 | 11 django/utils/regex_helper.py | 193 | 232| 274 | 4828 | 40708 | 
| 17 | 12 django/db/migrations/utils.py | 1 | 22| 121 | 4949 | 41551 | 
| **-> 18 <-** | **12 django/contrib/admindocs/utils.py** | 198 | 213| 225 | 5174 | 41551 | 
| 19 | 13 django/utils/xmlutils.py | 1 | 34| 247 | 5421 | 41798 | 
| 20 | 14 django/db/backends/mysql/features.py | 64 | 127| 640 | 6061 | 43983 | 
| 21 | 15 django/template/base.py | 602 | 637| 361 | 6422 | 52178 | 
| 22 | 15 django/db/migrations/serializer.py | 78 | 105| 233 | 6655 | 52178 | 
| 23 | **15 django/contrib/admindocs/utils.py** | 181 | 195| 207 | 6862 | 52178 | 
| 24 | 15 django/utils/jslex.py | 146 | 182| 321 | 7183 | 52178 | 
| 25 | 16 django/db/backends/sqlite3/base.py | 49 | 123| 775 | 7958 | 55224 | 
| 26 | 16 django/core/validators.py | 62 | 101| 605 | 8563 | 55224 | 
| 27 | 17 django/template/smartif.py | 150 | 209| 423 | 8986 | 56750 | 
| 28 | 17 django/utils/jslex.py | 76 | 144| 763 | 9749 | 56750 | 
| 29 | 17 django/core/validators.py | 231 | 246| 135 | 9884 | 56750 | 
| 30 | 17 django/db/migrations/serializer.py | 295 | 328| 288 | 10172 | 56750 | 
| 31 | 17 django/urls/resolvers.py | 222 | 265| 379 | 10551 | 56750 | 
| 32 | 18 django/views/i18n.py | 79 | 182| 702 | 11253 | 59211 | 
| 33 | 19 django/conf/locale/nl/formats.py | 32 | 67| 1371 | 12624 | 61094 | 
| 34 | 19 django/urls/resolvers.py | 103 | 123| 182 | 12806 | 61094 | 
| 35 | 19 django/db/migrations/serializer.py | 249 | 270| 166 | 12972 | 61094 | 
| 36 | 20 django/conf/locale/sr/formats.py | 5 | 40| 726 | 13698 | 61865 | 
| 37 | 21 django/conf/locale/sl/formats.py | 5 | 43| 708 | 14406 | 62618 | 
| 38 | 22 django/conf/locale/sr_Latn/formats.py | 5 | 40| 726 | 15132 | 63389 | 
| 39 | 22 django/core/validators.py | 150 | 185| 390 | 15522 | 63389 | 
| 40 | 23 django/db/backends/sqlite3/schema.py | 429 | 453| 162 | 15684 | 67635 | 
| 41 | 24 django/core/serializers/xml_serializer.py | 349 | 383| 327 | 16011 | 71147 | 
| **-> 42 <-** | **24 django/contrib/admindocs/utils.py** | 118 | 151| 271 | 16282 | 71147 | 
| 43 | 25 django/contrib/postgres/serializers.py | 1 | 11| 0 | 16282 | 71248 | 
| 44 | 26 django/template/defaultfilters.py | 358 | 442| 499 | 16781 | 77689 | 
| 45 | 27 django/utils/dateparse.py | 1 | 66| 771 | 17552 | 79257 | 
| 46 | 27 django/urls/resolvers.py | 268 | 290| 173 | 17725 | 79257 | 
| 47 | 27 django/core/validators.py | 103 | 147| 470 | 18195 | 79257 | 
| 48 | 27 django/db/migrations/serializer.py | 1 | 75| 436 | 18631 | 79257 | 
| 49 | 28 django/db/backends/mysql/base.py | 98 | 167| 717 | 19348 | 82707 | 
| 50 | 29 django/conf/locale/id/formats.py | 5 | 47| 670 | 20018 | 83422 | 
| 51 | 29 django/template/base.py | 1 | 93| 758 | 20776 | 83422 | 
| 52 | 30 django/conf/locale/mk/formats.py | 5 | 39| 594 | 21370 | 84061 | 
| 53 | 31 django/views/debug.py | 199 | 211| 143 | 21513 | 88764 | 
| 54 | 32 django/conf/locale/lv/formats.py | 5 | 45| 700 | 22213 | 89509 | 
| 55 | 33 django/utils/html.py | 139 | 165| 149 | 22362 | 92756 | 
| 56 | 34 django/conf/locale/uz/formats.py | 5 | 31| 418 | 22780 | 93219 | 
| 57 | 35 docs/_ext/djangodocs.py | 109 | 176| 572 | 23352 | 96428 | 
| 58 | 36 django/conf/locale/hr/formats.py | 5 | 43| 742 | 24094 | 97215 | 
| 59 | 37 django/conf/locale/ru/formats.py | 5 | 31| 367 | 24461 | 97627 | 
| 60 | 37 django/db/backends/sqlite3/base.py | 321 | 342| 183 | 24644 | 97627 | 
| 61 | 37 django/db/migrations/serializer.py | 184 | 195| 117 | 24761 | 97627 | 
| 62 | 38 django/utils/dateformat.py | 31 | 44| 121 | 24882 | 100221 | 
| 63 | 39 django/db/models/__init__.py | 1 | 53| 619 | 25501 | 100840 | 
| 64 | 40 django/conf/locale/sv/formats.py | 5 | 36| 481 | 25982 | 101366 | 
| 65 | 41 django/conf/locale/cs/formats.py | 5 | 41| 600 | 26582 | 102011 | 
| 66 | 42 django/conf/locale/ka/formats.py | 5 | 43| 773 | 27355 | 102829 | 
| 67 | 43 django/db/backends/mysql/schema.py | 52 | 94| 388 | 27743 | 104403 | 
| 68 | 44 django/contrib/postgres/constraints.py | 122 | 155| 250 | 27993 | 106013 | 
| 69 | 44 django/db/models/lookups.py | 485 | 522| 195 | 28188 | 106013 | 
| 70 | 45 django/conf/locale/it/formats.py | 5 | 41| 764 | 28952 | 106822 | 
| 71 | 46 django/conf/locale/ml/formats.py | 5 | 38| 610 | 29562 | 107477 | 
| 72 | 47 django/conf/locale/pt/formats.py | 5 | 36| 577 | 30139 | 108099 | 
| 73 | 48 django/db/backends/base/features.py | 218 | 322| 888 | 31027 | 111077 | 
| 74 | 49 django/conf/locale/pt_BR/formats.py | 5 | 32| 459 | 31486 | 111581 | 
| 75 | 50 django/db/backends/sqlite3/features.py | 54 | 87| 342 | 31828 | 112654 | 
| 76 | 50 django/core/validators.py | 1 | 14| 111 | 31939 | 112654 | 
| 77 | 51 django/conf/locale/zh_Hant/formats.py | 5 | 43| 635 | 32574 | 113334 | 
| 78 | 51 django/template/defaultfilters.py | 457 | 492| 233 | 32807 | 113334 | 
| 79 | 52 django/forms/models.py | 686 | 763| 750 | 33557 | 125252 | 
| 80 | 53 django/conf/locale/cy/formats.py | 5 | 33| 529 | 34086 | 125826 | 
| 81 | 54 django/conf/locale/zh_Hans/formats.py | 5 | 43| 635 | 34721 | 126506 | 
| 82 | 55 django/db/backends/mysql/operations.py | 357 | 366| 143 | 34864 | 130346 | 
| 83 | 56 django/db/models/sql/compiler.py | 161 | 209| 523 | 35387 | 145212 | 
| 84 | 57 django/conf/locale/lt/formats.py | 5 | 44| 676 | 36063 | 145933 | 
| 85 | 57 django/db/backends/sqlite3/schema.py | 40 | 66| 243 | 36306 | 145933 | 
| 86 | 57 django/urls/resolvers.py | 33 | 78| 426 | 36732 | 145933 | 
| 87 | 58 django/conf/locale/el/formats.py | 5 | 33| 455 | 37187 | 146433 | 
| 88 | 59 django/db/backends/base/schema.py | 1310 | 1332| 173 | 37360 | 159315 | 
| 89 | 60 django/conf/locale/uk/formats.py | 5 | 36| 425 | 37785 | 159785 | 
| 90 | 60 django/db/models/sql/compiler.py | 74 | 159| 890 | 38675 | 159785 | 
| 91 | 60 django/db/backends/mysql/schema.py | 1 | 39| 428 | 39103 | 159785 | 
| 92 | 61 django/conf/locale/nn/formats.py | 5 | 37| 593 | 39696 | 160423 | 
| 93 | 61 django/db/migrations/serializer.py | 273 | 292| 162 | 39858 | 160423 | 
| 94 | 61 django/db/backends/base/schema.py | 1289 | 1308| 163 | 40021 | 160423 | 
| 95 | 62 django/conf/locale/fr/formats.py | 5 | 32| 448 | 40469 | 160916 | 
| 96 | 63 django/utils/translation/template.py | 1 | 34| 381 | 40850 | 162870 | 
| 97 | 64 django/conf/locale/nb/formats.py | 5 | 37| 593 | 41443 | 163508 | 
| 98 | 64 django/db/backends/sqlite3/schema.py | 228 | 310| 731 | 42174 | 163508 | 
| 99 | 65 django/conf/locale/sk/formats.py | 5 | 29| 330 | 42504 | 163883 | 
| 100 | 66 django/conf/locale/az/formats.py | 5 | 31| 364 | 42868 | 164292 | 
| 101 | 66 django/contrib/postgres/constraints.py | 20 | 102| 691 | 43559 | 164292 | 
| 102 | 66 django/utils/html.py | 356 | 386| 227 | 43786 | 164292 | 
| 103 | 66 django/db/backends/base/schema.py | 1213 | 1244| 233 | 44019 | 164292 | 
| 104 | 67 django/urls/__init__.py | 1 | 24| 239 | 44258 | 164531 | 
| 105 | 67 django/db/migrations/serializer.py | 165 | 181| 151 | 44409 | 164531 | 
| 106 | 67 django/views/debug.py | 160 | 172| 148 | 44557 | 164531 | 
| 107 | 68 django/contrib/postgres/expressions.py | 1 | 15| 0 | 44557 | 164619 | 
| 108 | 69 django/db/backends/postgresql/base.py | 65 | 131| 689 | 45246 | 167543 | 
| 109 | 70 django/db/models/expressions.py | 847 | 881| 292 | 45538 | 178958 | 
| 110 | 70 django/template/defaultfilters.py | 445 | 454| 111 | 45649 | 178958 | 
| 111 | 71 django/db/models/options.py | 1 | 35| 300 | 45949 | 186320 | 
| 112 | 72 django/contrib/gis/db/backends/base/features.py | 1 | 112| 821 | 46770 | 187142 | 


### Hint

```
I left comments for improvement on the PR. Please uncheck "Patch needs improvement" after updating.
This PR is no longer related to this issue.
I have opened a ​pull request that allows simplify_regex() to handle non-capturing groups and additional tests for them in test_simplify_regex().
I've submitted a patch here ​https://github.com/django/django/pull/15256.
You need to uncheck "Patch needs improvement" so the ticket appears in the review queue. Please don't use mass @ mentions on the PR to request a review.
Replying to Tim Graham: You need to uncheck "Patch needs improvement" so the ticket appears in the review queue. Please don't use mass @ mentions on the PR to request a review. Sorry, Felisiak never came so I have to ask someone else lol
Replying to Mariusz Felisiak: Okay I updated that PR to only include feature, I'll create new PR for code refactor :thumbsup:.
In 827bc070: Refs #28135 -- Refactored out _find_groups()/_get_group_start_end() hooks in admindocs.
I left comments for improvement on the PR. Please uncheck "Patch needs improvement" after updating.
This PR is no longer related to this issue.
I have opened a ​pull request that allows simplify_regex() to handle non-capturing groups and additional tests for them in test_simplify_regex().
I've submitted a patch here ​https://github.com/django/django/pull/15256.
You need to uncheck "Patch needs improvement" so the ticket appears in the review queue. Please don't use mass @ mentions on the PR to request a review.
Replying to Tim Graham: You need to uncheck "Patch needs improvement" so the ticket appears in the review queue. Please don't use mass @ mentions on the PR to request a review. Sorry, Felisiak never came so I have to ask someone else lol
Replying to Mariusz Felisiak: Okay I updated that PR to only include feature, I'll create new PR for code refactor :thumbsup:.
In 827bc070: Refs #28135 -- Refactored out _find_groups()/_get_group_start_end() hooks in admindocs.
```

## Patch

```diff
diff --git a/django/contrib/admindocs/utils.py b/django/contrib/admindocs/utils.py
--- a/django/contrib/admindocs/utils.py
+++ b/django/contrib/admindocs/utils.py
@@ -137,9 +137,10 @@ def default_reference_role(name, rawtext, text, lineno, inliner, options=None, c
     for name, urlbase in ROLES.items():
         create_reference_role(name, urlbase)
 
-# Match the beginning of a named or unnamed group.
+# Match the beginning of a named, unnamed, or non-capturing groups.
 named_group_matcher = _lazy_re_compile(r'\(\?P(<\w+>)')
 unnamed_group_matcher = _lazy_re_compile(r'\(')
+non_capturing_group_matcher = _lazy_re_compile(r'\(\?\:')
 
 
 def replace_metacharacters(pattern):
@@ -210,3 +211,18 @@ def replace_unnamed_groups(pattern):
         final_pattern += pattern[:start] + '<var>'
         prev_end = end
     return final_pattern + pattern[prev_end:]
+
+
+def remove_non_capturing_groups(pattern):
+    r"""
+    Find non-capturing groups in the given `pattern` and remove them, e.g.
+    1. (?P<a>\w+)/b/(?:\w+)c(?:\w+) => (?P<a>\\w+)/b/c
+    2. ^(?:\w+(?:\w+))a => ^a
+    3. ^a(?:\w+)/b(?:\w+) => ^a/b
+    """
+    group_start_end_indices = _find_groups(pattern, non_capturing_group_matcher)
+    final_pattern, prev_end = '', None
+    for start, end, _ in group_start_end_indices:
+        final_pattern += pattern[prev_end:start]
+        prev_end = end
+    return final_pattern + pattern[prev_end:]
diff --git a/django/contrib/admindocs/views.py b/django/contrib/admindocs/views.py
--- a/django/contrib/admindocs/views.py
+++ b/django/contrib/admindocs/views.py
@@ -8,7 +8,8 @@
 from django.contrib.admin.views.decorators import staff_member_required
 from django.contrib.admindocs import utils
 from django.contrib.admindocs.utils import (
-    replace_metacharacters, replace_named_groups, replace_unnamed_groups,
+    remove_non_capturing_groups, replace_metacharacters, replace_named_groups,
+    replace_unnamed_groups,
 )
 from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
 from django.db import models
@@ -410,6 +411,7 @@ def simplify_regex(pattern):
     example, turn "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"
     into "/<sport_slug>/athletes/<athlete_slug>/".
     """
+    pattern = remove_non_capturing_groups(pattern)
     pattern = replace_named_groups(pattern)
     pattern = replace_unnamed_groups(pattern)
     pattern = replace_metacharacters(pattern)

```

## Test Patch

```diff
diff --git a/tests/admin_docs/test_views.py b/tests/admin_docs/test_views.py
--- a/tests/admin_docs/test_views.py
+++ b/tests/admin_docs/test_views.py
@@ -397,6 +397,13 @@ def test_simplify_regex(self):
             (r'^(?P<a>(x|y))/b/(?P<c>\w+)', '/<a>/b/<c>'),
             (r'^(?P<a>(x|y))/b/(?P<c>\w+)ab', '/<a>/b/<c>ab'),
             (r'^(?P<a>(x|y)(\(|\)))/b/(?P<c>\w+)ab', '/<a>/b/<c>ab'),
+            # Non-capturing groups.
+            (r'^a(?:\w+)b', '/ab'),
+            (r'^a(?:(x|y))', '/a'),
+            (r'^(?:\w+(?:\w+))a', '/a'),
+            (r'^a(?:\w+)/b(?:\w+)', '/a/b'),
+            (r'(?P<a>\w+)/b/(?:\w+)c(?:\w+)', '/<a>/b/c'),
+            (r'(?P<a>\w+)/b/(\w+)/(?:\w+)c(?:\w+)', '/<a>/b/<var>/c'),
             # Single and repeated metacharacters.
             (r'^a', '/a'),
             (r'^^a', '/a'),

```


## Code snippets

### 1 - django/contrib/admindocs/views.py:

Start line: 407, End line: 419

```python
def simplify_regex(pattern):
    r"""
    Clean up urlpattern regexes into something more readable by humans. For
    example, turn "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"
    into "/<sport_slug>/athletes/<athlete_slug>/".
    """
    pattern = replace_named_groups(pattern)
    pattern = replace_unnamed_groups(pattern)
    pattern = replace_metacharacters(pattern)
    if not pattern.startswith('/'):
        pattern = '/' + pattern
    return pattern
```
### 2 - django/db/migrations/serializer.py:

Start line: 235, End line: 246

```python
class RegexSerializer(BaseSerializer):
    def serialize(self):
        regex_pattern, pattern_imports = serializer_factory(self.value.pattern).serialize()
        # Turn off default implicit flags (e.g. re.U) because regexes with the
        # same implicit and explicit flags aren't equal.
        flags = self.value.flags ^ re.compile('').flags
        regex_flags, flag_imports = serializer_factory(flags).serialize()
        imports = {'import re', *pattern_imports, *flag_imports}
        args = [regex_pattern]
        if flags:
            args.append(regex_flags)
        return "re.compile(%s)" % ', '.join(args), imports
```
### 3 - django/utils/regex_helper.py:

Start line: 78, End line: 190

```python
def normalize(pattern):
    # ... other code

    try:
        while True:
            if escaped:
                result.append(ch)
            elif ch == '.':
                # Replace "any character" with an arbitrary representative.
                result.append(".")
            elif ch == '|':
                # FIXME: One day we'll should do this, but not in 1.0.
                raise NotImplementedError('Awaiting Implementation')
            elif ch == "^":
                pass
            elif ch == '$':
                break
            elif ch == ')':
                # This can only be the end of a non-capturing group, since all
                # other unescaped parentheses are handled by the grouping
                # section later (and the full group is handled there).
                #
                # We regroup everything inside the capturing group so that it
                # can be quantified, if necessary.
                start = non_capturing_groups.pop()
                inner = NonCapture(result[start:])
                result = result[:start] + [inner]
            elif ch == '[':
                # Replace ranges with the first character in the range.
                ch, escaped = next(pattern_iter)
                result.append(ch)
                ch, escaped = next(pattern_iter)
                while escaped or ch != ']':
                    ch, escaped = next(pattern_iter)
            elif ch == '(':
                # Some kind of group.
                ch, escaped = next(pattern_iter)
                if ch != '?' or escaped:
                    # A positional group
                    name = "_%d" % num_args
                    num_args += 1
                    result.append(Group((("%%(%s)s" % name), name)))
                    walk_to_end(ch, pattern_iter)
                else:
                    ch, escaped = next(pattern_iter)
                    if ch in '!=<':
                        # All of these are ignorable. Walk to the end of the
                        # group.
                        walk_to_end(ch, pattern_iter)
                    elif ch == ':':
                        # Non-capturing group
                        non_capturing_groups.append(len(result))
                    elif ch != 'P':
                        # Anything else, other than a named group, is something
                        # we cannot reverse.
                        raise ValueError("Non-reversible reg-exp portion: '(?%s'" % ch)
                    else:
                        ch, escaped = next(pattern_iter)
                        if ch not in ('<', '='):
                            raise ValueError("Non-reversible reg-exp portion: '(?P%s'" % ch)
                        # We are in a named capturing group. Extra the name and
                        # then skip to the end.
                        if ch == '<':
                            terminal_char = '>'
                        # We are in a named backreference.
                        else:
                            terminal_char = ')'
                        name = []
                        ch, escaped = next(pattern_iter)
                        while ch != terminal_char:
                            name.append(ch)
                            ch, escaped = next(pattern_iter)
                        param = ''.join(name)
                        # Named backreferences have already consumed the
                        # parenthesis.
                        if terminal_char != ')':
                            result.append(Group((("%%(%s)s" % param), param)))
                            walk_to_end(ch, pattern_iter)
                        else:
                            result.append(Group((("%%(%s)s" % param), None)))
            elif ch in "*?+{":
                # Quantifiers affect the previous item in the result list.
                count, ch = get_quantifier(ch, pattern_iter)
                if ch:
                    # We had to look ahead, but it wasn't need to compute the
                    # quantifier, so use this character next time around the
                    # main loop.
                    consume_next = False

                if count == 0:
                    if contains(result[-1], Group):
                        # If we are quantifying a capturing group (or
                        # something containing such a group) and the minimum is
                        # zero, we must also handle the case of one occurrence
                        # being present. All the quantifiers (except {0,0},
                        # which we conveniently ignore) that have a 0 minimum
                        # also allow a single occurrence.
                        result[-1] = Choice([None, result[-1]])
                    else:
                        result.pop()
                elif count > 1:
                    result.extend([result[-1]] * (count - 1))
            else:
                # Anything else is a literal.
                result.append(ch)

            if consume_next:
                ch, escaped = next(pattern_iter)
            consume_next = True
    except StopIteration:
        pass
    except NotImplementedError:
        # A case of using the disjunctive form. No results for you!
        return [('', [])]

    return list(zip(*flatten_result(result)))
```
### 4 - django/utils/regex_helper.py:

Start line: 1, End line: 38

```python
"""
Functions for reversing a regular expression (used in reverse URL resolving).
Used internally by Django and not intended for external use.

This is not, and is not intended to be, a complete reg-exp decompiler. It
should be good enough for a large class of URLS, however.
"""
import re

from django.utils.functional import SimpleLazyObject

# Mapping of an escape character to a representative of that class. So, e.g.,
# "\w" is replaced by "x" in a reverse URL. A value of None means to ignore
# this sequence. Any missing key is mapped to itself.
ESCAPE_MAPPINGS = {
    "A": None,
    "b": None,
    "B": None,
    "d": "0",
    "D": "x",
    "s": " ",
    "S": "x",
    "w": "x",
    "W": "!",
    "Z": None,
}


class Choice(list):
    """Represent multiple possibilities at this point in a pattern string."""


class Group(list):
    """Represent a capturing group in the pattern string."""


class NonCapture(list):
    """Represent a non-capturing group in the pattern string."""
```
### 5 - django/utils/regex_helper.py:

Start line: 41, End line: 76

```python
def normalize(pattern):
    r"""
    Given a reg-exp pattern, normalize it to an iterable of forms that
    suffice for reverse matching. This does the following:

    (1) For any repeating sections, keeps the minimum number of occurrences
        permitted (this means zero for optional groups).
    (2) If an optional group includes parameters, include one occurrence of
        that group (along with the zero occurrence case from step (1)).
    (3) Select the first (essentially an arbitrary) element from any character
        class. Select an arbitrary character for any unordered class (e.g. '.'
        or '\w') in the pattern.
    (4) Ignore look-ahead and look-behind assertions.
    (5) Raise an error on any disjunctive ('|') constructs.

    Django's URLs for forward resolving are either all positional arguments or
    all keyword arguments. That is assumed here, as well. Although reverse
    resolving can be done using positional args when keyword args are
    specified, the two cannot be mixed in the same reverse() call.
    """
    # Do a linear scan to work out the special features of this pattern. The
    # idea is that we scan once here and collect all the information we need to
    # make future decisions.
    result = []
    non_capturing_groups = []
    consume_next = True
    pattern_iter = next_char(iter(pattern))
    num_args = 0

    # A "while" loop is used here because later on we need to be able to peek
    # at the next character and possibly go around without consuming another
    # one at the top of the loop.
    try:
        ch, escaped = next(pattern_iter)
    except StopIteration:
        return [('', [])]
    # ... other code
```
### 6 - django/utils/regex_helper.py:

Start line: 286, End line: 352

```python
def flatten_result(source):
    """
    Turn the given source sequence into a list of reg-exp possibilities and
    their arguments. Return a list of strings and a list of argument lists.
    Each of the two lists will be of the same length.
    """
    if source is None:
        return [''], [[]]
    if isinstance(source, Group):
        if source[1] is None:
            params = []
        else:
            params = [source[1]]
        return [source[0]], [params]
    result = ['']
    result_args = [[]]
    pos = last = 0
    for pos, elt in enumerate(source):
        if isinstance(elt, str):
            continue
        piece = ''.join(source[last:pos])
        if isinstance(elt, Group):
            piece += elt[0]
            param = elt[1]
        else:
            param = None
        last = pos + 1
        for i in range(len(result)):
            result[i] += piece
            if param:
                result_args[i].append(param)
        if isinstance(elt, (Choice, NonCapture)):
            if isinstance(elt, NonCapture):
                elt = [elt]
            inner_result, inner_args = [], []
            for item in elt:
                res, args = flatten_result(item)
                inner_result.extend(res)
                inner_args.extend(args)
            new_result = []
            new_args = []
            for item, args in zip(result, result_args):
                for i_item, i_args in zip(inner_result, inner_args):
                    new_result.append(item + i_item)
                    new_args.append(args[:] + i_args)
            result = new_result
            result_args = new_args
    if pos >= last:
        piece = ''.join(source[last:])
        for i in range(len(result)):
            result[i] += piece
    return result, result_args


def _lazy_re_compile(regex, flags=0):
    """Lazily compile a regex with flags."""
    def _compile():
        # Compile the regex if it was not passed pre-compiled.
        if isinstance(regex, (str, bytes)):
            return re.compile(regex, flags)
        else:
            assert not flags, (
                'flags must be empty if regex is passed pre-compiled'
            )
            return regex
    return SimpleLazyObject(_compile)
```
### 7 - django/db/models/lookups.py:

Start line: 543, End line: 560

```python
@Field.register_lookup
class Regex(BuiltinLookup):
    lookup_name = 'regex'
    prepare_rhs = False

    def as_sql(self, compiler, connection):
        if self.lookup_name in connection.operators:
            return super().as_sql(compiler, connection)
        else:
            lhs, lhs_params = self.process_lhs(compiler, connection)
            rhs, rhs_params = self.process_rhs(compiler, connection)
            sql_template = connection.ops.regex_lookup(self.lookup_name)
            return sql_template % (lhs, rhs), lhs_params + rhs_params


@Field.register_lookup
class IRegex(Regex):
    lookup_name = 'iregex'
```
### 8 - django/contrib/gis/geometry.py:

Start line: 1, End line: 18

```python
import re

from django.utils.regex_helper import _lazy_re_compile

# Regular expression for recognizing HEXEWKB and WKT.  A prophylactic measure
# to prevent potentially malicious input from reaching the underlying C
# library. Not a substitute for good web security programming practices.
hex_regex = _lazy_re_compile(r'^[0-9A-F]+$', re.I)
wkt_regex = _lazy_re_compile(
    r'^(SRID=(?P<srid>\-?[0-9]+);)?'
    r'(?P<wkt>'
    r'(?P<type>POINT|LINESTRING|LINEARRING|POLYGON|MULTIPOINT|'
    r'MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)'
    r'[ACEGIMLONPSRUTYZ0-9,\.\-\+\(\) ]+)$',
    re.I
)
json_regex = _lazy_re_compile(r'^(\s+)?\{.*}(\s+)?$', re.DOTALL)
```
### 9 - django/urls/resolvers.py:

Start line: 185, End line: 219

```python
class RegexPattern(CheckURLMixin):

    def check(self):
        warnings = []
        warnings.extend(self._check_pattern_startswith_slash())
        if not self._is_endpoint:
            warnings.extend(self._check_include_trailing_dollar())
        return warnings

    def _check_include_trailing_dollar(self):
        regex_pattern = self.regex.pattern
        if regex_pattern.endswith('$') and not regex_pattern.endswith(r'\$'):
            return [Warning(
                "Your URL pattern {} uses include with a route ending with a '$'. "
                "Remove the dollar from the route to avoid problems including "
                "URLs.".format(self.describe()),
                id='urls.W001',
            )]
        else:
            return []

    def _compile(self, regex):
        """Compile and return the given regular expression."""
        try:
            return re.compile(regex)
        except re.error as e:
            raise ImproperlyConfigured(
                '"%s" is not a valid regular expression: %s' % (regex, e)
            ) from e

    def __str__(self):
        return str(self._regex)


_PATH_PARAMETER_COMPONENT_RE = _lazy_re_compile(
    r'<(?:(?P<converter>[^>:]+):)?(?P<parameter>[^>]+)>'
)
```
### 10 - django/core/validators.py:

Start line: 17, End line: 59

```python
@deconstructible
class RegexValidator:
    regex = ''
    message = _('Enter a valid value.')
    code = 'invalid'
    inverse_match = False
    flags = 0

    def __init__(self, regex=None, message=None, code=None, inverse_match=None, flags=None):
        if regex is not None:
            self.regex = regex
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        if inverse_match is not None:
            self.inverse_match = inverse_match
        if flags is not None:
            self.flags = flags
        if self.flags and not isinstance(self.regex, str):
            raise TypeError("If the flags are set, regex must be a regular expression string.")

        self.regex = _lazy_re_compile(self.regex, self.flags)

    def __call__(self, value):
        """
        Validate that the input contains (or does *not* contain, if
        inverse_match is True) a match for the regular expression.
        """
        regex_matches = self.regex.search(str(value))
        invalid_input = regex_matches if self.inverse_match else not regex_matches
        if invalid_input:
            raise ValidationError(self.message, code=self.code, params={'value': value})

    def __eq__(self, other):
        return (
            isinstance(other, RegexValidator) and
            self.regex.pattern == other.regex.pattern and
            self.regex.flags == other.regex.flags and
            (self.message == other.message) and
            (self.code == other.code) and
            (self.inverse_match == other.inverse_match)
        )
```
### 14 - django/contrib/admindocs/utils.py:

Start line: 154, End line: 178

```python
def _get_group_start_end(start, end, pattern):
    # Handle nested parentheses, e.g. '^(?P<a>(x|y))/b' or '^b/((x|y)\w+)$'.
    unmatched_open_brackets, prev_char = 1, None
    for idx, val in enumerate(pattern[end:]):
        # Check for unescaped `(` and `)`. They mark the start and end of a
        # nested group.
        if val == '(' and prev_char != '\\':
            unmatched_open_brackets += 1
        elif val == ')' and prev_char != '\\':
            unmatched_open_brackets -= 1
        prev_char = val
        # If brackets are balanced, the end of the string for the current named
        # capture group pattern has been reached.
        if unmatched_open_brackets == 0:
            return start, end + idx + 1


def _find_groups(pattern, group_matcher):
    prev_end = None
    for match in group_matcher.finditer(pattern):
        if indices := _get_group_start_end(match.start(0), match.end(0), pattern):
            start, end = indices
            if prev_end and start > prev_end or not prev_end:
                yield start, end, match
            prev_end = end
```
### 18 - django/contrib/admindocs/utils.py:

Start line: 198, End line: 213

```python
def replace_unnamed_groups(pattern):
    r"""
    Find unnamed groups in `pattern` and replace them with '<var>'. E.g.,
    1. ^(?P<a>\w+)/b/(\w+)$ ==> ^(?P<a>\w+)/b/<var>$
    2. ^(?P<a>\w+)/b/((x|y)\w+)$ ==> ^(?P<a>\w+)/b/<var>$
    3. ^(?P<a>\w+)/b/(\w+) ==> ^(?P<a>\w+)/b/<var>
    4. ^(?P<a>\w+)/b/((x|y)\w+) ==> ^(?P<a>\w+)/b/<var>
    """
    final_pattern, prev_end = '', None
    for start, end, _ in _find_groups(pattern, unnamed_group_matcher):
        if prev_end:
            final_pattern += pattern[prev_end:start]
        final_pattern += pattern[:start] + '<var>'
        prev_end = end
    return final_pattern + pattern[prev_end:]
```
### 23 - django/contrib/admindocs/utils.py:

Start line: 181, End line: 195

```python
def replace_named_groups(pattern):
    r"""
    Find named groups in `pattern` and replace them with the group name. E.g.,
    1. ^(?P<a>\w+)/b/(\w+)$ ==> ^<a>/b/(\w+)$
    2. ^(?P<a>\w+)/b/(?P<c>\w+)/$ ==> ^<a>/b/<c>/$
    3. ^(?P<a>\w+)/b/(\w+) ==> ^<a>/b/(\w+)
    4. ^(?P<a>\w+)/b/(?P<c>\w+) ==> ^<a>/b/<c>
    """
    group_pattern_and_name = [
        (pattern[start:end], match[1])
        for start, end, match in _find_groups(pattern, named_group_matcher)
    ]
    for group_pattern, group_name in group_pattern_and_name:
        pattern = pattern.replace(group_pattern, group_name)
    return pattern
```
### 42 - django/contrib/admindocs/utils.py:

Start line: 118, End line: 151

```python
def default_reference_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    if options is None:
        options = {}
    context = inliner.document.settings.default_reference_context
    node = docutils.nodes.reference(
        rawtext,
        text,
        refuri=(ROLES[context] % (
            inliner.document.settings.link_base,
            text.lower(),
        )),
        **options
    )
    return [node], []


if docutils_is_available:
    docutils.parsers.rst.roles.register_canonical_role('cmsreference', default_reference_role)

    for name, urlbase in ROLES.items():
        create_reference_role(name, urlbase)

# Match the beginning of a named or unnamed group.
named_group_matcher = _lazy_re_compile(r'\(\?P(<\w+>)')
unnamed_group_matcher = _lazy_re_compile(r'\(')


def replace_metacharacters(pattern):
    """Remove unescaped metacharacters from the pattern."""
    return re.sub(
        r'((?:^|(?<!\\))(?:\\\\)*)(\\?)([?*+^$]|\\[bBAZ])',
        lambda m: m[1] + m[3] if m[2] else m[1],
        pattern,
    )
```
