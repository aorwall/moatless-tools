# django__django-14880

| **django/django** | `402ae37873974afa5093e6d6149175a118979cd9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 945 |
| **Any found context length** | 674 |
| **Avg pos** | 3.0 |
| **Min pos** | 1 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -239,7 +239,9 @@ def _check_clashes(self):
             if not rel_is_hidden and clash_field.name == rel_name:
                 errors.append(
                     checks.Error(
-                        "Reverse accessor for '%s' clashes with field name '%s'." % (field_name, clash_name),
+                        f"Reverse accessor '{rel_opts.object_name}.{rel_name}' "
+                        f"for '{field_name}' clashes with field name "
+                        f"'{clash_name}'.",
                         hint=("Rename field '%s', or add/change a related_name "
                               "argument to the definition for field '%s'.") % (clash_name, field_name),
                         obj=self,
@@ -271,7 +273,9 @@ def _check_clashes(self):
             if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                 errors.append(
                     checks.Error(
-                        "Reverse accessor for '%s' clashes with reverse accessor for '%s'." % (field_name, clash_name),
+                        f"Reverse accessor '{rel_opts.object_name}.{rel_name}' "
+                        f"for '{field_name}' clashes with reverse accessor for "
+                        f"'{clash_name}'.",
                         hint=("Add or change a related_name argument "
                               "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                         obj=self,

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related.py | 242 | 242 | 1 | 1 | 674
| django/db/models/fields/related.py | 274 | 274 | 2 | 1 | 945


## Problem Statement

```
Improve error messages for reverse accessor clashes.
Description
	
refer: ​https://github.com/django/django/pull/14880
RelatedField._check_clashes() provides feedback when it finds a clash, but fails to mentioned what the clashing name was. This cost me some significant time to track because of inadequate feedback and would have become immediately clear had the feedback listed the clashing name. 
A proposed patch appears above, but alas this impacts some unit tests as well. Happy to add fixes to those to the patch, but have been requested to file and issue here.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/fields/related.py** | 198 | 264| 674 | 674 | 13970 | 
| **-> 2 <-** | **1 django/db/models/fields/related.py** | 265 | 294| 271 | 945 | 13970 | 
| 3 | 2 django/db/models/base.py | 1546 | 1578| 231 | 1176 | 31300 | 
| 4 | **2 django/db/models/fields/related.py** | 139 | 166| 201 | 1377 | 31300 | 
| 5 | 2 django/db/models/base.py | 1440 | 1495| 491 | 1868 | 31300 | 
| 6 | 2 django/db/models/base.py | 1522 | 1544| 171 | 2039 | 31300 | 
| 7 | 2 django/db/models/base.py | 1497 | 1520| 176 | 2215 | 31300 | 
| 8 | 2 django/db/models/base.py | 1269 | 1300| 267 | 2482 | 31300 | 
| 9 | **2 django/db/models/fields/related.py** | 183 | 196| 140 | 2622 | 31300 | 
| 10 | 3 django/core/checks/messages.py | 54 | 77| 161 | 2783 | 31877 | 
| 11 | 3 django/core/checks/messages.py | 27 | 51| 259 | 3042 | 31877 | 
| 12 | 4 django/core/checks/model_checks.py | 129 | 153| 268 | 3310 | 33662 | 
| 13 | 5 django/db/models/fields/__init__.py | 367 | 393| 199 | 3509 | 51830 | 
| 14 | 5 django/core/checks/model_checks.py | 178 | 211| 332 | 3841 | 51830 | 
| 15 | **5 django/db/models/fields/related.py** | 120 | 137| 155 | 3996 | 51830 | 
| 16 | 6 django/db/models/fields/reverse_related.py | 180 | 205| 269 | 4265 | 54150 | 
| 17 | 7 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 4421 | 64539 | 
| 18 | 7 django/core/checks/model_checks.py | 155 | 176| 263 | 4684 | 64539 | 
| 19 | **7 django/db/models/fields/related.py** | 1262 | 1379| 963 | 5647 | 64539 | 
| 20 | 7 django/db/models/base.py | 1723 | 1771| 348 | 5995 | 64539 | 
| 21 | 7 django/db/models/base.py | 1161 | 1176| 138 | 6133 | 64539 | 
| 22 | 7 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 6253 | 64539 | 
| 23 | 8 django/contrib/admin/checks.py | 904 | 952| 416 | 6669 | 73721 | 
| 24 | 9 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 6806 | 73858 | 
| 25 | 9 django/core/checks/messages.py | 1 | 25| 160 | 6966 | 73858 | 
| 26 | 9 django/db/models/fields/__init__.py | 338 | 365| 203 | 7169 | 73858 | 
| 27 | 9 django/db/models/fields/related_descriptors.py | 1 | 79| 683 | 7852 | 73858 | 
| 28 | **9 django/db/models/fields/related.py** | 1670 | 1686| 286 | 8138 | 73858 | 
| 29 | 9 django/db/models/fields/__init__.py | 1117 | 1149| 237 | 8375 | 73858 | 
| 30 | 9 django/db/models/base.py | 1178 | 1206| 213 | 8588 | 73858 | 
| 31 | 9 django/contrib/admin/checks.py | 608 | 619| 128 | 8716 | 73858 | 
| 32 | **9 django/db/models/fields/related.py** | 1455 | 1496| 418 | 9134 | 73858 | 
| 33 | 10 django/contrib/auth/checks.py | 105 | 211| 776 | 9910 | 75356 | 
| 34 | 11 django/forms/models.py | 765 | 786| 194 | 10104 | 87274 | 
| 35 | **11 django/db/models/fields/related.py** | 1381 | 1453| 616 | 10720 | 87274 | 
| 36 | **11 django/db/models/fields/related.py** | 1230 | 1260| 172 | 10892 | 87274 | 
| 37 | 11 django/contrib/admin/checks.py | 571 | 606| 304 | 11196 | 87274 | 
| 38 | **11 django/db/models/fields/related.py** | 1627 | 1668| 497 | 11693 | 87274 | 
| 39 | 11 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 11877 | 87274 | 
| 40 | 12 django/db/models/query_utils.py | 134 | 198| 492 | 12369 | 89762 | 
| 41 | **12 django/db/models/fields/related.py** | 168 | 181| 144 | 12513 | 89762 | 
| 42 | 13 django/db/models/constraints.py | 38 | 90| 416 | 12929 | 91823 | 
| 43 | 14 django/contrib/postgres/utils.py | 1 | 30| 218 | 13147 | 92041 | 
| 44 | 15 django/core/checks/__init__.py | 1 | 28| 307 | 13454 | 92348 | 
| 45 | 15 django/db/models/base.py | 1132 | 1159| 286 | 13740 | 92348 | 
| 46 | **15 django/db/models/fields/related.py** | 505 | 525| 138 | 13878 | 92348 | 
| 47 | 16 django/utils/deprecation.py | 1 | 30| 195 | 14073 | 93373 | 
| 48 | 17 django/db/models/options.py | 557 | 585| 231 | 14304 | 100740 | 
| 49 | **17 django/db/models/fields/related.py** | 1 | 34| 246 | 14550 | 100740 | 
| 50 | 17 django/db/models/base.py | 1087 | 1130| 404 | 14954 | 100740 | 
| 51 | 17 django/db/models/fields/related_descriptors.py | 672 | 730| 548 | 15502 | 100740 | 
| 52 | 18 django/core/checks/compatibility/django_4_0.py | 1 | 19| 138 | 15640 | 100879 | 
| 53 | 19 django/urls/resolvers.py | 653 | 726| 681 | 16321 | 106671 | 
| 54 | 19 django/contrib/admin/checks.py | 232 | 253| 208 | 16529 | 106671 | 
| 55 | 19 django/contrib/admin/checks.py | 1117 | 1146| 176 | 16705 | 106671 | 
| 56 | 20 django/contrib/messages/views.py | 1 | 19| 95 | 16800 | 106767 | 
| 57 | 20 django/core/checks/model_checks.py | 89 | 110| 168 | 16968 | 106767 | 
| 58 | **20 django/db/models/fields/related.py** | 768 | 786| 222 | 17190 | 106767 | 
| 59 | 20 django/db/models/base.py | 1 | 50| 328 | 17518 | 106767 | 
| 60 | 20 django/contrib/admin/checks.py | 794 | 815| 190 | 17708 | 106767 | 
| 61 | 20 django/db/models/query_utils.py | 251 | 276| 293 | 18001 | 106767 | 
| 62 | 20 django/db/models/fields/reverse_related.py | 160 | 178| 167 | 18168 | 106767 | 
| 63 | 21 django/db/models/deletion.py | 1 | 75| 561 | 18729 | 110597 | 
| 64 | 22 django/contrib/messages/__init__.py | 1 | 3| 23 | 18752 | 110621 | 
| 65 | 22 django/db/models/base.py | 1875 | 1948| 572 | 19324 | 110621 | 
| 66 | 22 django/db/models/base.py | 1966 | 2124| 1178 | 20502 | 110621 | 
| 67 | **22 django/db/models/fields/related.py** | 296 | 330| 293 | 20795 | 110621 | 
| 68 | 23 django/db/backends/base/schema.py | 1 | 29| 209 | 21004 | 123503 | 
| 69 | 24 django/core/checks/caches.py | 22 | 56| 291 | 21295 | 124024 | 
| 70 | 24 django/db/models/options.py | 1 | 35| 300 | 21595 | 124024 | 
| 71 | **24 django/db/models/fields/related.py** | 885 | 911| 240 | 21835 | 124024 | 
| 72 | 25 django/db/migrations/exceptions.py | 1 | 55| 249 | 22084 | 124274 | 
| 73 | 26 django/contrib/admin/utils.py | 444 | 472| 225 | 22309 | 128432 | 
| 74 | 26 django/utils/deprecation.py | 33 | 73| 336 | 22645 | 128432 | 
| 75 | 26 django/db/models/fields/related_descriptors.py | 1015 | 1042| 334 | 22979 | 128432 | 
| 76 | 26 django/db/models/base.py | 1333 | 1358| 184 | 23163 | 128432 | 
| 77 | 26 django/contrib/admin/checks.py | 220 | 230| 127 | 23290 | 128432 | 
| 78 | 27 django/core/checks/security/csrf.py | 45 | 68| 157 | 23447 | 128894 | 
| 79 | 27 django/contrib/admin/checks.py | 1043 | 1070| 194 | 23641 | 128894 | 
| 80 | 28 django/forms/forms.py | 338 | 383| 445 | 24086 | 132917 | 
| 81 | 28 django/db/models/fields/related_descriptors.py | 732 | 758| 222 | 24308 | 132917 | 
| 82 | 28 django/contrib/admin/checks.py | 350 | 376| 221 | 24529 | 132917 | 
| 83 | 29 django/core/checks/database.py | 1 | 15| 68 | 24597 | 132986 | 
| 84 | 29 django/db/models/fields/related_descriptors.py | 609 | 641| 323 | 24920 | 132986 | 
| 85 | 29 django/db/models/fields/__init__.py | 208 | 242| 234 | 25154 | 132986 | 
| 86 | 29 django/contrib/admin/utils.py | 289 | 307| 175 | 25329 | 132986 | 
| 87 | 30 django/contrib/contenttypes/fields.py | 433 | 454| 248 | 25577 | 138444 | 
| 88 | 30 django/contrib/admin/checks.py | 621 | 643| 155 | 25732 | 138444 | 
| 89 | 31 django/db/backends/base/creation.py | 301 | 322| 258 | 25990 | 141232 | 
| 90 | 32 django/db/migrations/autodetector.py | 921 | 1001| 871 | 26861 | 152862 | 
| 91 | 32 django/db/models/fields/__init__.py | 308 | 336| 205 | 27066 | 152862 | 
| 92 | **32 django/db/models/fields/related.py** | 527 | 592| 492 | 27558 | 152862 | 
| 93 | 32 django/db/models/fields/__init__.py | 1150 | 1182| 267 | 27825 | 152862 | 
| 94 | 32 django/db/models/fields/__init__.py | 525 | 542| 182 | 28007 | 152862 | 
| 95 | 33 django/core/management/commands/inspectdb.py | 175 | 229| 478 | 28485 | 155495 | 
| 96 | 34 django/forms/utils.py | 170 | 177| 133 | 28618 | 156988 | 
| 97 | **34 django/db/models/fields/related.py** | 648 | 668| 168 | 28786 | 156988 | 
| 98 | 34 django/urls/resolvers.py | 641 | 651| 120 | 28906 | 156988 | 
| 99 | 34 django/db/models/fields/__init__.py | 244 | 306| 448 | 29354 | 156988 | 
| 100 | 34 django/db/models/fields/related_descriptors.py | 643 | 671| 247 | 29601 | 156988 | 
| 101 | 34 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 29783 | 156988 | 
| 102 | 35 django/db/migrations/operations/models.py | 598 | 615| 163 | 29946 | 163438 | 
| 103 | 35 django/db/migrations/operations/models.py | 1 | 38| 235 | 30181 | 163438 | 
| 104 | 36 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 30298 | 163555 | 
| 105 | 37 django/contrib/admindocs/utils.py | 1 | 28| 184 | 30482 | 165579 | 
| 106 | 37 django/contrib/admin/checks.py | 146 | 163| 155 | 30637 | 165579 | 
| 107 | 38 django/db/models/fields/related_lookups.py | 107 | 122| 215 | 30852 | 167048 | 
| 108 | 38 django/contrib/auth/checks.py | 1 | 102| 720 | 31572 | 167048 | 
| 109 | 38 django/core/checks/model_checks.py | 1 | 86| 665 | 32237 | 167048 | 
| 110 | 39 django/db/backends/mysql/validation.py | 1 | 31| 239 | 32476 | 167568 | 
| 111 | 39 django/db/migrations/operations/models.py | 370 | 390| 213 | 32689 | 167568 | 
| 112 | 40 django/views/debug.py | 150 | 162| 148 | 32837 | 172321 | 
| 113 | 40 django/db/models/base.py | 1360 | 1389| 205 | 33042 | 172321 | 
| 114 | **40 django/db/models/fields/related.py** | 956 | 969| 126 | 33168 | 172321 | 
| 115 | 40 django/contrib/admin/checks.py | 646 | 665| 183 | 33351 | 172321 | 
| 116 | 40 django/core/checks/security/csrf.py | 1 | 42| 304 | 33655 | 172321 | 
| 117 | 40 django/db/models/fields/related_descriptors.py | 945 | 967| 218 | 33873 | 172321 | 
| 118 | 40 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 34278 | 172321 | 
| 119 | 40 django/views/debug.py | 466 | 510| 429 | 34707 | 172321 | 
| 120 | 41 django/core/checks/templates.py | 1 | 36| 259 | 34966 | 172581 | 
| 121 | 41 django/db/models/fields/related_descriptors.py | 1075 | 1086| 138 | 35104 | 172581 | 
| 122 | 41 django/contrib/admin/checks.py | 546 | 569| 230 | 35334 | 172581 | 
| 123 | 42 django/middleware/common.py | 149 | 175| 254 | 35588 | 174110 | 
| 124 | 42 django/db/models/fields/related_descriptors.py | 969 | 986| 190 | 35778 | 174110 | 
| 125 | 42 django/db/models/fields/reverse_related.py | 20 | 139| 749 | 36527 | 174110 | 
| 126 | 42 django/db/models/fields/related_descriptors.py | 494 | 548| 338 | 36865 | 174110 | 
| 127 | 43 django/db/models/sql/datastructures.py | 103 | 114| 133 | 36998 | 175560 | 
| 128 | 43 django/db/models/fields/__init__.py | 1203 | 1234| 230 | 37228 | 175560 | 
| 129 | **43 django/db/models/fields/related.py** | 688 | 712| 218 | 37446 | 175560 | 
| 130 | 44 django/db/models/functions/text.py | 214 | 230| 149 | 37595 | 177896 | 
| 131 | 44 django/db/models/fields/related_descriptors.py | 551 | 607| 487 | 38082 | 177896 | 
| 132 | 45 django/db/models/fields/mixins.py | 31 | 57| 173 | 38255 | 178239 | 
| 133 | **45 django/db/models/fields/related.py** | 1498 | 1532| 356 | 38611 | 178239 | 
| 134 | 46 django/contrib/admin/options.py | 1 | 96| 756 | 39367 | 196931 | 


## Patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -239,7 +239,9 @@ def _check_clashes(self):
             if not rel_is_hidden and clash_field.name == rel_name:
                 errors.append(
                     checks.Error(
-                        "Reverse accessor for '%s' clashes with field name '%s'." % (field_name, clash_name),
+                        f"Reverse accessor '{rel_opts.object_name}.{rel_name}' "
+                        f"for '{field_name}' clashes with field name "
+                        f"'{clash_name}'.",
                         hint=("Rename field '%s', or add/change a related_name "
                               "argument to the definition for field '%s'.") % (clash_name, field_name),
                         obj=self,
@@ -271,7 +273,9 @@ def _check_clashes(self):
             if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                 errors.append(
                     checks.Error(
-                        "Reverse accessor for '%s' clashes with reverse accessor for '%s'." % (field_name, clash_name),
+                        f"Reverse accessor '{rel_opts.object_name}.{rel_name}' "
+                        f"for '{field_name}' clashes with reverse accessor for "
+                        f"'{clash_name}'.",
                         hint=("Add or change a related_name argument "
                               "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                         obj=self,

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -862,8 +862,8 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.rel' "
-                "clashes with field name "
+                "Reverse accessor 'Target.model_set' for "
+                "'invalid_models_tests.Model.rel' clashes with field name "
                 "'invalid_models_tests.Target.model_set'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Target.model_set', or "
@@ -885,9 +885,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.foreign' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.m2m'.",
+                "Reverse accessor 'Target.model_set' for "
+                "'invalid_models_tests.Model.foreign' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.m2m'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.foreign' or "
@@ -897,9 +897,9 @@ class Model(models.Model):
                 id='fields.E304',
             ),
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.m2m' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.foreign'.",
+                "Reverse accessor 'Target.model_set' for "
+                "'invalid_models_tests.Model.m2m' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.foreign'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.m2m' or "
@@ -927,9 +927,9 @@ class Child(Parent):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.children' "
-                "clashes with field name "
-                "'invalid_models_tests.Child.m2m_clash'.",
+                "Reverse accessor 'Child.m2m_clash' for "
+                "'invalid_models_tests.Model.children' clashes with field "
+                "name 'invalid_models_tests.Child.m2m_clash'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Child.m2m_clash', or "
                     "add/change a related_name argument to the definition for "
@@ -1085,8 +1085,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.rel' "
-                "clashes with field name 'invalid_models_tests.Target.clash'.",
+                "Reverse accessor 'Target.clash' for "
+                "'invalid_models_tests.Model.rel' clashes with field name "
+                "'invalid_models_tests.Target.clash'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Target.clash', or "
                     "add/change a related_name argument to the definition for "
@@ -1218,9 +1219,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.first_m2m' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.second_m2m'.",
+                "Reverse accessor 'Model.model_set' for "
+                "'invalid_models_tests.Model.first_m2m' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.second_m2m'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.first_m2m' or "
@@ -1230,9 +1231,9 @@ class Model(models.Model):
                 id='fields.E304',
             ),
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.second_m2m' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.first_m2m'.",
+                "Reverse accessor 'Model.model_set' for "
+                "'invalid_models_tests.Model.second_m2m' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.first_m2m'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.second_m2m' or "
@@ -1249,9 +1250,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.model_set' "
-                "clashes with field name "
-                "'invalid_models_tests.Model.model_set'.",
+                "Reverse accessor 'Model.model_set' for "
+                "'invalid_models_tests.Model.model_set' clashes with field "
+                "name 'invalid_models_tests.Model.model_set'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Model.model_set', or "
                     "add/change a related_name argument to the definition for "
@@ -1287,8 +1288,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.m2m' "
-                "clashes with field name 'invalid_models_tests.Model.clash'.",
+                "Reverse accessor 'Model.clash' for "
+                "'invalid_models_tests.Model.m2m' clashes with field name "
+                "'invalid_models_tests.Model.clash'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Model.clash', or "
                     "add/change a related_name argument to the definition for "
@@ -1327,9 +1329,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.model_set' "
-                "clashes with field name "
-                "'invalid_models_tests.Model.model_set'.",
+                "Reverse accessor 'Model.model_set' for "
+                "'invalid_models_tests.Model.model_set' clashes with field "
+                "name 'invalid_models_tests.Model.model_set'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Model.model_set', or "
                     "add/change a related_name argument to the definition for "
@@ -1365,8 +1367,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.foreign' "
-                "clashes with field name 'invalid_models_tests.Model.clash'.",
+                "Reverse accessor 'Model.clash' for "
+                "'invalid_models_tests.Model.foreign' clashes with field name "
+                "'invalid_models_tests.Model.clash'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Model.clash', or "
                     "add/change a related_name argument to the definition for "
@@ -1413,8 +1416,9 @@ class Model(models.Model):
 
         self.assertEqual(Model.check(), [
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.foreign_1' "
-                "clashes with field name 'invalid_models_tests.Target.id'.",
+                "Reverse accessor 'Target.id' for "
+                "'invalid_models_tests.Model.foreign_1' clashes with field "
+                "name 'invalid_models_tests.Target.id'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Target.id', or "
                     "add/change a related_name argument to the definition for "
@@ -1435,9 +1439,9 @@ class Model(models.Model):
                 id='fields.E303',
             ),
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.foreign_1' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.m2m_1'.",
+                "Reverse accessor 'Target.id' for "
+                "'invalid_models_tests.Model.foreign_1' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.m2m_1'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.foreign_1' or "
@@ -1460,9 +1464,9 @@ class Model(models.Model):
             ),
 
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.foreign_2' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.m2m_2'.",
+                "Reverse accessor 'Target.src_safe' for "
+                "'invalid_models_tests.Model.foreign_2' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.m2m_2'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.foreign_2' or "
@@ -1485,8 +1489,9 @@ class Model(models.Model):
             ),
 
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.m2m_1' "
-                "clashes with field name 'invalid_models_tests.Target.id'.",
+                "Reverse accessor 'Target.id' for "
+                "'invalid_models_tests.Model.m2m_1' clashes with field name "
+                "'invalid_models_tests.Target.id'.",
                 hint=(
                     "Rename field 'invalid_models_tests.Target.id', or "
                     "add/change a related_name argument to the definition for "
@@ -1507,9 +1512,9 @@ class Model(models.Model):
                 id='fields.E303',
             ),
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.m2m_1' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.foreign_1'.",
+                "Reverse accessor 'Target.id' for "
+                "'invalid_models_tests.Model.m2m_1' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.foreign_1'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.m2m_1' or "
@@ -1531,9 +1536,9 @@ class Model(models.Model):
                 id='fields.E305',
             ),
             Error(
-                "Reverse accessor for 'invalid_models_tests.Model.m2m_2' "
-                "clashes with reverse accessor for "
-                "'invalid_models_tests.Model.foreign_2'.",
+                "Reverse accessor 'Target.src_safe' for "
+                "'invalid_models_tests.Model.m2m_2' clashes with reverse "
+                "accessor for 'invalid_models_tests.Model.foreign_2'.",
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Model.m2m_2' or "
@@ -1564,16 +1569,16 @@ class Child(Parent):
             other_parent = models.OneToOneField(Parent, models.CASCADE)
 
         errors = [
-            ('fields.E304', 'accessor', 'parent_ptr', 'other_parent'),
-            ('fields.E305', 'query name', 'parent_ptr', 'other_parent'),
-            ('fields.E304', 'accessor', 'other_parent', 'parent_ptr'),
-            ('fields.E305', 'query name', 'other_parent', 'parent_ptr'),
+            ('fields.E304', 'accessor', " 'Parent.child'", 'parent_ptr', 'other_parent'),
+            ('fields.E305', 'query name', '', 'parent_ptr', 'other_parent'),
+            ('fields.E304', 'accessor', " 'Parent.child'", 'other_parent', 'parent_ptr'),
+            ('fields.E305', 'query name', '', 'other_parent', 'parent_ptr'),
         ]
         self.assertEqual(Child.check(), [
             Error(
-                "Reverse %s for 'invalid_models_tests.Child.%s' clashes with "
+                "Reverse %s%s for 'invalid_models_tests.Child.%s' clashes with "
                 "reverse %s for 'invalid_models_tests.Child.%s'."
-                % (attr, field_name, attr, clash_name),
+                % (attr, rel_name, field_name, attr, clash_name),
                 hint=(
                     "Add or change a related_name argument to the definition "
                     "for 'invalid_models_tests.Child.%s' or "
@@ -1582,7 +1587,7 @@ class Child(Parent):
                 obj=Child._meta.get_field(field_name),
                 id=error_id,
             )
-            for error_id, attr, field_name, clash_name in errors
+            for error_id, attr, rel_name, field_name, clash_name in errors
         ])
 
 
diff --git a/tests/model_inheritance/test_abstract_inheritance.py b/tests/model_inheritance/test_abstract_inheritance.py
--- a/tests/model_inheritance/test_abstract_inheritance.py
+++ b/tests/model_inheritance/test_abstract_inheritance.py
@@ -292,8 +292,9 @@ class Foo(models.Model):
             Foo._meta.get_field('foo').check(),
             [
                 Error(
-                    "Reverse accessor for 'model_inheritance.Foo.foo' clashes "
-                    "with field name 'model_inheritance.Descendant.foo'.",
+                    "Reverse accessor 'Descendant.foo' for "
+                    "'model_inheritance.Foo.foo' clashes with field name "
+                    "'model_inheritance.Descendant.foo'.",
                     hint=(
                         "Rename field 'model_inheritance.Descendant.foo', or "
                         "add/change a related_name argument to the definition "

```


## Code snippets

### 1 - django/db/models/fields/related.py:

Start line: 198, End line: 264

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        """Check accessor and reverse query name clashes."""
        from django.db.models.base import ModelBase

        errors = []
        opts = self.model._meta

        # `f.remote_field.model` may be a string instead of a model. Skip if model name is
        # not resolved.
        if not isinstance(self.remote_field.model, ModelBase):
            return []

        # Consider that we are checking field `Model.foreign` and the models
        # are:
        #
        #     class Target(models.Model):
        #         model = models.IntegerField()
        #         model_set = models.IntegerField()
        #
        #     class Model(models.Model):
        #         foreign = models.ForeignKey(Target)
        #         m2m = models.ManyToManyField(Target)

        # rel_opts.object_name == "Target"
        rel_opts = self.remote_field.model._meta
        # If the field doesn't install a backward relation on the target model
        # (so `is_hidden` returns True), then there are no clashes to check
        # and we can skip these fields.
        rel_is_hidden = self.remote_field.is_hidden()
        rel_name = self.remote_field.get_accessor_name()  # i. e. "model_set"
        rel_query_name = self.related_query_name()  # i. e. "model"
        # i.e. "app_label.Model.field".
        field_name = '%s.%s' % (opts.label, self.name)

        # Check clashes between accessor or reverse query name of `field`
        # and any other field name -- i.e. accessor for Model.foreign is
        # model_set and it clashes with Target.model_set.
        potential_clashes = rel_opts.fields + rel_opts.many_to_many
        for clash_field in potential_clashes:
            # i.e. "app_label.Target.model_set".
            clash_name = '%s.%s' % (rel_opts.label, clash_field.name)
            if not rel_is_hidden and clash_field.name == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E302',
                    )
                )

            if clash_field.name == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with field name '%s'." % (field_name, clash_name),
                        hint=("Rename field '%s', or add/change a related_name "
                              "argument to the definition for field '%s'.") % (clash_name, field_name),
                        obj=self,
                        id='fields.E303',
                    )
                )

        # Check clashes between accessors/reverse query names of `field` and
        # any other field accessor -- i. e. Model.foreign accessor clashes with
        # Model.m2m accessor.
        potential_clashes = (r for r in rel_opts.related_objects if r.field is not self)
        # ... other code
```
### 2 - django/db/models/fields/related.py:

Start line: 265, End line: 294

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            # i.e. "app_label.Model.m2m".
            clash_name = '%s.%s' % (
                clash_field.related_model._meta.label,
                clash_field.field.name,
            )
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(
                    checks.Error(
                        "Reverse accessor for '%s' clashes with reverse accessor for '%s'." % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E304',
                    )
                )

            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with reverse query name for '%s'."
                        % (field_name, clash_name),
                        hint=("Add or change a related_name argument "
                              "to the definition for '%s' or '%s'.") % (field_name, clash_name),
                        obj=self,
                        id='fields.E305',
                    )
                )

        return errors
```
### 3 - django/db/models/base.py:

Start line: 1546, End line: 1578

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_property_name_related_field_accessor_clashes(cls):
        errors = []
        property_names = cls._meta._property_names
        related_field_accessors = (
            f.get_attname() for f in cls._meta._get_fields(reverse=False)
            if f.is_relation and f.related_model is not None
        )
        for accessor in related_field_accessors:
            if accessor in property_names:
                errors.append(
                    checks.Error(
                        "The property '%s' clashes with a related field "
                        "accessor." % accessor,
                        obj=cls,
                        id='models.E025',
                    )
                )
        return errors

    @classmethod
    def _check_single_primary_key(cls):
        errors = []
        if sum(1 for f in cls._meta.local_fields if f.primary_key) > 1:
            errors.append(
                checks.Error(
                    "The model cannot have more than one field with "
                    "'primary_key=True'.",
                    obj=cls,
                    id='models.E026',
                )
            )
        return errors
```
### 4 - django/db/models/fields/related.py:

Start line: 139, End line: 166

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_query_name_is_valid(self):
        if self.remote_field.is_hidden():
            return []
        rel_query_name = self.related_query_name()
        errors = []
        if rel_query_name.endswith('_'):
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not end with an underscore."
                    % rel_query_name,
                    hint=("Add or change a related_name or related_query_name "
                          "argument for this field."),
                    obj=self,
                    id='fields.E308',
                )
            )
        if LOOKUP_SEP in rel_query_name:
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not contain '%s'."
                    % (rel_query_name, LOOKUP_SEP),
                    hint=("Add or change a related_name or related_query_name "
                          "argument for this field."),
                    obj=self,
                    id='fields.E309',
                )
            )
        return errors
```
### 5 - django/db/models/base.py:

Start line: 1440, End line: 1495

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_field_name_clashes(cls):
        """Forbid field shadowing in multi-table inheritance."""
        errors = []
        used_fields = {}  # name or attname -> field

        # Check that multi-inheritance doesn't cause field name shadowing.
        for parent in cls._meta.get_parent_list():
            for f in parent._meta.local_fields:
                clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
                if clash:
                    errors.append(
                        checks.Error(
                            "The field '%s' from parent model "
                            "'%s' clashes with the field '%s' "
                            "from parent model '%s'." % (
                                clash.name, clash.model._meta,
                                f.name, f.model._meta
                            ),
                            obj=cls,
                            id='models.E005',
                        )
                    )
                used_fields[f.name] = f
                used_fields[f.attname] = f

        # Check that fields defined in the model don't clash with fields from
        # parents, including auto-generated fields like multi-table inheritance
        # child accessors.
        for parent in cls._meta.get_parent_list():
            for f in parent._meta.get_fields():
                if f not in used_fields:
                    used_fields[f.name] = f

        for f in cls._meta.local_fields:
            clash = used_fields.get(f.name) or used_fields.get(f.attname) or None
            # Note that we may detect clash between user-defined non-unique
            # field "id" and automatically added unique field "id", both
            # defined at the same model. This special case is considered in
            # _check_id_field and here we ignore it.
            id_conflict = f.name == "id" and clash and clash.name == "id" and clash.model == cls
            if clash and not id_conflict:
                errors.append(
                    checks.Error(
                        "The field '%s' clashes with the field '%s' "
                        "from model '%s'." % (
                            f.name, clash.name, clash.model._meta
                        ),
                        obj=f,
                        id='models.E006',
                    )
                )
            used_fields[f.name] = f
            used_fields[f.attname] = f

        return errors
```
### 6 - django/db/models/base.py:

Start line: 1522, End line: 1544

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_model_name_db_lookup_clashes(cls):
        errors = []
        model_name = cls.__name__
        if model_name.startswith('_') or model_name.endswith('_'):
            errors.append(
                checks.Error(
                    "The model name '%s' cannot start or end with an underscore "
                    "as it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E023'
                )
            )
        elif LOOKUP_SEP in model_name:
            errors.append(
                checks.Error(
                    "The model name '%s' cannot contain double underscores as "
                    "it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E024'
                )
            )
        return errors
```
### 7 - django/db/models/base.py:

Start line: 1497, End line: 1520

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_column_name_clashes(cls):
        # Store a list of column names which have already been used by other fields.
        used_column_names = []
        errors = []

        for f in cls._meta.local_fields:
            _, column_name = f.get_attname_column()

            # Ensure the column name is not already in use.
            if column_name and column_name in used_column_names:
                errors.append(
                    checks.Error(
                        "Field '%s' has column name '%s' that is used by "
                        "another field." % (f.name, column_name),
                        hint="Specify a 'db_column' for the field.",
                        obj=cls,
                        id='models.E007'
                    )
                )
            else:
                used_column_names.append(column_name)

        return errors
```
### 8 - django/db/models/base.py:

Start line: 1269, End line: 1300

```python
class Model(metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [*cls._check_swappable(), *cls._check_model(), *cls._check_managers(**kwargs)]
        if not cls._meta.swapped:
            databases = kwargs.get('databases') or []
            errors += [
                *cls._check_fields(**kwargs),
                *cls._check_m2m_through_same_relationship(),
                *cls._check_long_column_names(databases),
            ]
            clash_errors = (
                *cls._check_id_field(),
                *cls._check_field_name_clashes(),
                *cls._check_model_name_db_lookup_clashes(),
                *cls._check_property_name_related_field_accessor_clashes(),
                *cls._check_single_primary_key(),
            )
            errors.extend(clash_errors)
            # If there are field name clashes, hide consequent column name
            # clashes.
            if not clash_errors:
                errors.extend(cls._check_column_name_clashes())
            errors += [
                *cls._check_index_together(),
                *cls._check_unique_together(),
                *cls._check_indexes(databases),
                *cls._check_ordering(),
                *cls._check_constraints(databases),
                *cls._check_default_pk(),
            ]

        return errors
```
### 9 - django/db/models/fields/related.py:

Start line: 183, End line: 196

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (self.remote_field.model not in self.opts.apps.get_models() and
                not isinstance(self.remote_field.model, str) and
                self.remote_field.model._meta.swapped):
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % self.remote_field.model._meta.label,
                    hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable,
                    obj=self,
                    id='fields.E301',
                )
            ]
        return []
```
### 10 - django/core/checks/messages.py:

Start line: 54, End line: 77

```python
class Debug(CheckMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(DEBUG, *args, **kwargs)


class Info(CheckMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(INFO, *args, **kwargs)


class Warning(CheckMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(WARNING, *args, **kwargs)


class Error(CheckMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(ERROR, *args, **kwargs)


class Critical(CheckMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(CRITICAL, *args, **kwargs)
```
### 15 - django/db/models/fields/related.py:

Start line: 120, End line: 137

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_name_is_valid(self):
        import keyword
        related_name = self.remote_field.related_name
        if related_name is None:
            return []
        is_valid_id = not keyword.iskeyword(related_name) and related_name.isidentifier()
        if not (is_valid_id or related_name.endswith('+')):
            return [
                checks.Error(
                    "The name '%s' is invalid related_name for field %s.%s" %
                    (self.remote_field.related_name, self.model._meta.object_name,
                     self.name),
                    hint="Related name must be a valid Python identifier or end with a '+'",
                    obj=self,
                    id='fields.E306',
                )
            ]
        return []
```
### 19 - django/db/models/fields/related.py:

Start line: 1262, End line: 1379

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, '_meta'):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label, self.remote_field.through.__name__)
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(include_auto_created=True):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id='fields.E331',
                )
            )

        else:
            assert from_model is not None, (
                "ManyToManyField with intermediate "
                "tables cannot be checked if you don't pass the model "
                "where the field is attached to."
            )
            # Set some useful local variables
            to_model = resolve_relation(from_model, self.remote_field.model)
            from_model_name = from_model._meta.object_name
            if isinstance(to_model, str):
                to_model_name = to_model
            else:
                to_model_name = to_model._meta.object_name
            relationship_model_name = self.remote_field.through._meta.object_name
            self_referential = from_model == to_model
            # Count foreign keys in intermediate model
            if self_referential:
                seen_self = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument." % (self, from_model_name),
                            hint="Use through_fields to specify which two foreign keys Django should use.",
                            obj=self.remote_field.through,
                            id='fields.E333',
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, 'model', None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            ("The model is used as an intermediate model by "
                             "'%s', but it has more than one foreign key "
                             "from '%s', which is ambiguous. You must specify "
                             "which foreign key Django should use via the "
                             "through_fields keyword argument.") % (self, from_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E334',
                        )
                    )

                if seen_to > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than one foreign key "
                            "to '%s', which is ambiguous. You must specify "
                            "which foreign key Django should use via the "
                            "through_fields keyword argument." % (self, to_model_name),
                            hint=(
                                'If you want to create a recursive relationship, '
                                'use ManyToManyField("%s", through="%s").'
                            ) % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id='fields.E335',
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'." % (
                                self, from_model_name, to_model_name
                            ),
                            obj=self.remote_field.through,
                            id='fields.E336',
                        )
                    )
        # ... other code
```
### 28 - django/db/models/fields/related.py:

Start line: 1670, End line: 1686

```python
class ManyToManyField(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal M2Ms (i.e., those with a related name ending with '+')
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls, related.get_accessor_name(), ManyToManyDescriptor(self.remote_field, reverse=True))

        # Set up the accessors for the column names on the m2m table.
        self.m2m_column_name = partial(self._get_m2m_attr, related, 'column')
        self.m2m_reverse_name = partial(self._get_m2m_reverse_attr, related, 'column')

        self.m2m_field_name = partial(self._get_m2m_attr, related, 'name')
        self.m2m_reverse_field_name = partial(self._get_m2m_reverse_attr, related, 'name')

        get_m2m_rel = partial(self._get_m2m_attr, related, 'remote_field')
        self.m2m_target_field_name = lambda: get_m2m_rel().field_name
        get_m2m_reverse_rel = partial(self._get_m2m_reverse_attr, related, 'remote_field')
        self.m2m_reverse_target_field_name = lambda: get_m2m_reverse_rel().field_name
```
### 32 - django/db/models/fields/related.py:

Start line: 1455, End line: 1496

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if isinstance(self.remote_field.through, str) or not self.remote_field.through._meta.managed:
            return []
        registered_tables = {
            model._meta.db_table: model
            for model in self.opts.apps.get_models(include_auto_created=True)
            if model != self.remote_field.through and model._meta.managed
        }
        m2m_db_table = self.m2m_db_table()
        model = registered_tables.get(m2m_db_table)
        # The second condition allows multiple m2m relations on a model if
        # some point to a through model that proxies another through model.
        if model and model._meta.concrete_model != self.remote_field.through._meta.concrete_model:
            if model._meta.auto_created:
                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name
                opts = model._meta.auto_created._meta
                clashing_obj = '%s.%s' % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = checks.Warning, 'fields.W344'
                error_hint = (
                    'You have configured settings.DATABASE_ROUTERS. Verify '
                    'that the table of %r is correctly routed to a separate '
                    'database.' % clashing_obj
                )
            else:
                error_class, error_id = checks.Error, 'fields.E340'
                error_hint = None
            return [
                error_class(
                    "The field's intermediary table '%s' clashes with the "
                    "table name of '%s'." % (m2m_db_table, clashing_obj),
                    obj=self,
                    hint=error_hint,
                    id=error_id,
                )
            ]
        return []
```
### 35 - django/db/models/fields/related.py:

Start line: 1381, End line: 1453

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):

        # Validate `through_fields`.
        if self.remote_field.through_fields is not None:
            # Validate that we're given an iterable of at least two items
            # and that none of them is "falsy".
            if not (len(self.remote_field.through_fields) >= 2 and
                    self.remote_field.through_fields[0] and self.remote_field.through_fields[1]):
                errors.append(
                    checks.Error(
                        "Field specifies 'through_fields' but does not provide "
                        "the names of the two link fields that should be used "
                        "for the relation through model '%s'." % qualified_model_name,
                        hint="Make sure you specify 'through_fields' as through_fields=('field1', 'field2')",
                        obj=self,
                        id='fields.E337',
                    )
                )

            # Validate the given through fields -- they should be actual
            # fields on the through model, and also be foreign keys to the
            # expected models.
            else:
                assert from_model is not None, (
                    "ManyToManyField with intermediate "
                    "tables cannot be checked if you don't pass the model "
                    "where the field is attached to."
                )

                source, through, target = from_model, self.remote_field.through, self.remote_field.model
                source_field_name, target_field_name = self.remote_field.through_fields[:2]

                for field_name, related_model in ((source_field_name, source),
                                                  (target_field_name, target)):

                    possible_field_names = []
                    for f in through._meta.fields:
                        if hasattr(f, 'remote_field') and getattr(f.remote_field, 'model', None) == related_model:
                            possible_field_names.append(f.name)
                    if possible_field_names:
                        hint = "Did you mean one of the following foreign keys to '%s': %s?" % (
                            related_model._meta.object_name,
                            ', '.join(possible_field_names),
                        )
                    else:
                        hint = None

                    try:
                        field = through._meta.get_field(field_name)
                    except exceptions.FieldDoesNotExist:
                        errors.append(
                            checks.Error(
                                "The intermediary model '%s' has no field '%s'."
                                % (qualified_model_name, field_name),
                                hint=hint,
                                obj=self,
                                id='fields.E338',
                            )
                        )
                    else:
                        if not (hasattr(field, 'remote_field') and
                                getattr(field.remote_field, 'model', None) == related_model):
                            errors.append(
                                checks.Error(
                                    "'%s.%s' is not a foreign key to '%s'." % (
                                        through._meta.object_name, field_name,
                                        related_model._meta.object_name,
                                    ),
                                    hint=hint,
                                    obj=self,
                                    id='fields.E339',
                                )
                            )

        return errors
```
### 36 - django/db/models/fields/related.py:

Start line: 1230, End line: 1260

```python
class ManyToManyField(RelatedField):

    def _check_ignored_options(self, **kwargs):
        warnings = []

        if self.has_null_arg:
            warnings.append(
                checks.Warning(
                    'null has no effect on ManyToManyField.',
                    obj=self,
                    id='fields.W340',
                )
            )

        if self._validators:
            warnings.append(
                checks.Warning(
                    'ManyToManyField does not support validators.',
                    obj=self,
                    id='fields.W341',
                )
            )
        if self.remote_field.symmetrical and self._related_name:
            warnings.append(
                checks.Warning(
                    'related_name has no effect on ManyToManyField '
                    'with a symmetrical relationship, e.g. to "self".',
                    obj=self,
                    id='fields.W345',
                )
            )

        return warnings
```
### 38 - django/db/models/fields/related.py:

Start line: 1627, End line: 1668

```python
class ManyToManyField(RelatedField):

    def contribute_to_class(self, cls, name, **kwargs):
        # To support multiple relations to self, it's useful to have a non-None
        # related name on symmetrical relations for internal reasons. The
        # concept doesn't make a lot of sense externally ("you want me to
        # specify *what* on my non-reversible relation?!"), so we set it up
        # automatically. The funky name reduces the chance of an accidental
        # clash.
        if self.remote_field.symmetrical and (
            self.remote_field.model == RECURSIVE_RELATIONSHIP_CONSTANT or
            self.remote_field.model == cls._meta.object_name
        ):
            self.remote_field.related_name = "%s_rel_+" % name
        elif self.remote_field.is_hidden():
            # If the backwards relation is disabled, replace the original
            # related_name with one generated from the m2m field name. Django
            # still uses backwards relations internally and we need to avoid
            # clashes between multiple m2m fields with related_name == '+'.
            self.remote_field.related_name = '_%s_%s_%s_+' % (
                cls._meta.app_label,
                cls.__name__.lower(),
                name,
            )

        super().contribute_to_class(cls, name, **kwargs)

        # The intermediate m2m model is not auto created if:
        #  1) There is a manually specified intermediate, or
        #  2) The class owning the m2m field is abstract.
        #  3) The class owning the m2m field has been swapped out.
        if not cls._meta.abstract:
            if self.remote_field.through:
                def resolve_through_model(_, model, field):
                    field.remote_field.through = model
                lazy_related_operation(resolve_through_model, cls, self.remote_field.through, field=self)
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(self, cls)

        # Add the descriptor for the m2m relation.
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))

        # Set up the accessor for the m2m table name for the relation.
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)
```
### 41 - django/db/models/fields/related.py:

Start line: 168, End line: 181

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_relation_model_exists(self):
        rel_is_missing = self.remote_field.model not in self.opts.apps.get_models()
        rel_is_string = isinstance(self.remote_field.model, str)
        model_name = self.remote_field.model if rel_is_string else self.remote_field.model._meta.object_name
        if rel_is_missing and (rel_is_string or not self.remote_field.model._meta.swapped):
            return [
                checks.Error(
                    "Field defines a relation with model '%s', which is either "
                    "not installed, or is abstract." % model_name,
                    obj=self,
                    id='fields.E300',
                )
            ]
        return []
```
### 46 - django/db/models/fields/related.py:

Start line: 505, End line: 525

```python
class ForeignObject(RelatedField):

    def _check_to_fields_exist(self):
        # Skip nonexistent models.
        if isinstance(self.remote_field.model, str):
            return []

        errors = []
        for to_field in self.to_fields:
            if to_field:
                try:
                    self.remote_field.model._meta.get_field(to_field)
                except exceptions.FieldDoesNotExist:
                    errors.append(
                        checks.Error(
                            "The to_field '%s' doesn't exist on the related "
                            "model '%s'."
                            % (to_field, self.remote_field.model._meta.label),
                            obj=self,
                            id='fields.E312',
                        )
                    )
        return errors
```
### 49 - django/db/models/fields/related.py:

Start line: 1, End line: 34

```python
import functools
import inspect
from functools import partial

from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
    ForeignKeyDeferredAttribute, ForwardManyToOneDescriptor,
    ForwardOneToOneDescriptor, ManyToManyDescriptor,
    ReverseManyToOneDescriptor, ReverseOneToOneDescriptor,
)
from .related_lookups import (
    RelatedExact, RelatedGreaterThan, RelatedGreaterThanOrEqual, RelatedIn,
    RelatedIsNull, RelatedLessThan, RelatedLessThanOrEqual,
)
from .reverse_related import (
    ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel,
)

RECURSIVE_RELATIONSHIP_CONSTANT = 'self'
```
### 58 - django/db/models/fields/related.py:

Start line: 768, End line: 786

```python
class ForeignObject(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal FK's - i.e., those with a related name ending with '+' -
        # and swapped models don't get a related descriptor.
        if not self.remote_field.is_hidden() and not related.related_model._meta.swapped:
            setattr(cls._meta.concrete_model, related.get_accessor_name(), self.related_accessor_class(related))
            # While 'limit_choices_to' might be a callable, simply pass
            # it along for later - this is too early because it's still
            # model load time.
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(self.remote_field.limit_choices_to)


ForeignObject.register_lookup(RelatedIn)
ForeignObject.register_lookup(RelatedExact)
ForeignObject.register_lookup(RelatedLessThan)
ForeignObject.register_lookup(RelatedGreaterThan)
ForeignObject.register_lookup(RelatedGreaterThanOrEqual)
ForeignObject.register_lookup(RelatedLessThanOrEqual)
ForeignObject.register_lookup(RelatedIsNull)
```
### 67 - django/db/models/fields/related.py:

Start line: 296, End line: 330

```python
class RelatedField(FieldCacheMixin, Field):

    def db_type(self, connection):
        # By default related field will not have a column as it relates to
        # columns from another table.
        return None

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):

        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)

        self.opts = cls._meta

        if not cls._meta.abstract:
            if self.remote_field.related_name:
                related_name = self.remote_field.related_name
            else:
                related_name = self.opts.default_related_name
            if related_name:
                related_name = related_name % {
                    'class': cls.__name__.lower(),
                    'model_name': cls._meta.model_name.lower(),
                    'app_label': cls._meta.app_label.lower()
                }
                self.remote_field.related_name = related_name

            if self.remote_field.related_query_name:
                related_query_name = self.remote_field.related_query_name % {
                    'class': cls.__name__.lower(),
                    'app_label': cls._meta.app_label.lower(),
                }
                self.remote_field.related_query_name = related_query_name

            def resolve_related_class(model, related, field):
                field.remote_field.model = related
                field.do_related_class(related, model)
            lazy_related_operation(resolve_related_class, cls, self.remote_field.model, field=self)
```
### 71 - django/db/models/fields/related.py:

Start line: 885, End line: 911

```python
class ForeignKey(ForeignObject):

    def _check_unique(self, **kwargs):
        return [
            checks.Warning(
                'Setting unique=True on a ForeignKey has the same effect as using a OneToOneField.',
                hint='ForeignKey(unique=True) is usually better served by a OneToOneField.',
                obj=self,
                id='fields.W342',
            )
        ] if self.unique else []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['to_fields']
        del kwargs['from_fields']
        # Handle the simpler arguments
        if self.db_index:
            del kwargs['db_index']
        else:
            kwargs['db_index'] = False
        if self.db_constraint is not True:
            kwargs['db_constraint'] = self.db_constraint
        # Rel needs more work.
        to_meta = getattr(self.remote_field.model, "_meta", None)
        if self.remote_field.field_name and (
                not to_meta or (to_meta.pk and self.remote_field.field_name != to_meta.pk.name)):
            kwargs['to_field'] = self.remote_field.field_name
        return name, path, args, kwargs
```
### 92 - django/db/models/fields/related.py:

Start line: 527, End line: 592

```python
class ForeignObject(RelatedField):

    def _check_unique_target(self):
        rel_is_string = isinstance(self.remote_field.model, str)
        if rel_is_string or not self.requires_unique_target:
            return []

        try:
            self.foreign_related_fields
        except exceptions.FieldDoesNotExist:
            return []

        if not self.foreign_related_fields:
            return []

        unique_foreign_fields = {
            frozenset([f.name])
            for f in self.remote_field.model._meta.get_fields()
            if getattr(f, 'unique', False)
        }
        unique_foreign_fields.update({
            frozenset(ut)
            for ut in self.remote_field.model._meta.unique_together
        })
        unique_foreign_fields.update({
            frozenset(uc.fields)
            for uc in self.remote_field.model._meta.total_unique_constraints
        })
        foreign_fields = {f.name for f in self.foreign_related_fields}
        has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)

        if not has_unique_constraint and len(self.foreign_related_fields) > 1:
            field_combination = ', '.join(
                "'%s'" % rel_field.name for rel_field in self.foreign_related_fields
            )
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "No subset of the fields %s on model '%s' is unique."
                    % (field_combination, model_name),
                    hint=(
                        'Mark a single field as unique=True or add a set of '
                        'fields to a unique constraint (via unique_together '
                        'or a UniqueConstraint (without condition) in the '
                        'model Meta.constraints).'
                    ),
                    obj=self,
                    id='fields.E310',
                )
            ]
        elif not has_unique_constraint:
            field_name = self.foreign_related_fields[0].name
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "'%s.%s' must be unique because it is referenced by "
                    "a foreign key." % (model_name, field_name),
                    hint=(
                        'Add unique=True to this field or add a '
                        'UniqueConstraint (without condition) in the model '
                        'Meta.constraints.'
                    ),
                    obj=self,
                    id='fields.E311',
                )
            ]
        else:
            return []
```
### 97 - django/db/models/fields/related.py:

Start line: 648, End line: 668

```python
class ForeignObject(RelatedField):

    @cached_property
    def related_fields(self):
        return self.resolve_related_fields()

    @cached_property
    def reverse_related_fields(self):
        return [(rhs_field, lhs_field) for lhs_field, rhs_field in self.related_fields]

    @cached_property
    def local_related_fields(self):
        return tuple(lhs_field for lhs_field, rhs_field in self.related_fields)

    @cached_property
    def foreign_related_fields(self):
        return tuple(rhs_field for lhs_field, rhs_field in self.related_fields if rhs_field)

    def get_local_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.local_related_fields)

    def get_foreign_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.foreign_related_fields)
```
### 114 - django/db/models/fields/related.py:

Start line: 956, End line: 969

```python
class ForeignKey(ForeignObject):

    def resolve_related_fields(self):
        related_fields = super().resolve_related_fields()
        for from_field, to_field in related_fields:
            if to_field and to_field.model != self.remote_field.model._meta.concrete_model:
                raise exceptions.FieldError(
                    "'%s.%s' refers to field '%s' which is not local to model "
                    "'%s'." % (
                        self.model._meta.label,
                        self.name,
                        to_field.name,
                        self.remote_field.model._meta.concrete_model._meta.label,
                    )
                )
        return related_fields
```
### 129 - django/db/models/fields/related.py:

Start line: 688, End line: 712

```python
class ForeignObject(RelatedField):

    def get_attname_column(self):
        attname, column = super().get_attname_column()
        return attname, None

    def get_joining_columns(self, reverse_join=False):
        source = self.reverse_related_fields if reverse_join else self.related_fields
        return tuple((lhs_field.column, rhs_field.column) for lhs_field, rhs_field in source)

    def get_reverse_joining_columns(self):
        return self.get_joining_columns(reverse_join=True)

    def get_extra_descriptor_filter(self, instance):
        """
        Return an extra filter condition for related object fetching when
        user does 'instance.fieldname', that is the extra filter is used in
        the descriptor of the field.

        The filter should be either a dict usable in .filter(**kwargs) call or
        a Q-object. The condition will be ANDed together with the relation's
        joining columns.

        A parallel method is get_extra_restriction() which is used in
        JOIN and subquery conditions.
        """
        return {}
```
### 133 - django/db/models/fields/related.py:

Start line: 1498, End line: 1532

```python
class ManyToManyField(RelatedField):

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        # Handle the simpler arguments.
        if self.db_table is not None:
            kwargs['db_table'] = self.db_table
        if self.remote_field.db_constraint is not True:
            kwargs['db_constraint'] = self.remote_field.db_constraint
        # Rel needs more work.
        if isinstance(self.remote_field.model, str):
            kwargs['to'] = self.remote_field.model
        else:
            kwargs['to'] = self.remote_field.model._meta.label
        if getattr(self.remote_field, 'through', None) is not None:
            if isinstance(self.remote_field.through, str):
                kwargs['through'] = self.remote_field.through
            elif not self.remote_field.through._meta.auto_created:
                kwargs['through'] = self.remote_field.through._meta.label
        # If swappable is True, then see if we're actually pointing to the target
        # of a swap.
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            # If it's already a settings reference, error.
            if hasattr(kwargs['to'], "setting_name"):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError(
                        "Cannot deconstruct a ManyToManyField pointing to a "
                        "model that is swapped in place of more than one model "
                        "(%s and %s)" % (kwargs['to'].setting_name, swappable_setting)
                    )

            kwargs['to'] = SettingsReference(
                kwargs['to'],
                swappable_setting,
            )
        return name, path, args, kwargs
```
