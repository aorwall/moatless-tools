# django__django-12477

| **django/django** | `41ebe60728a15aa273f4d70de92f5246a89c3d4e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1270 |
| **Any found context length** | 1270 |
| **Avg pos** | 12.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 4 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -528,6 +528,10 @@ def _check_unique_target(self):
             frozenset(ut)
             for ut in self.remote_field.model._meta.unique_together
         })
+        unique_foreign_fields.update({
+            frozenset(uc.fields)
+            for uc in self.remote_field.model._meta.total_unique_constraints
+        })
         foreign_fields = {f.name for f in self.foreign_related_fields}
         has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)
 
@@ -541,8 +545,10 @@ def _check_unique_target(self):
                     "No subset of the fields %s on model '%s' is unique."
                     % (field_combination, model_name),
                     hint=(
-                        "Add unique=True on any of those fields or add at "
-                        "least a subset of them to a unique_together constraint."
+                        'Mark a single field as unique=True or add a set of '
+                        'fields to a unique constraint (via unique_together '
+                        'or a UniqueConstraint (without condition) in the '
+                        'model Meta.constraints).'
                     ),
                     obj=self,
                     id='fields.E310',
@@ -553,8 +559,13 @@ def _check_unique_target(self):
             model_name = self.remote_field.model.__name__
             return [
                 checks.Error(
-                    "'%s.%s' must set unique=True because it is referenced by "
+                    "'%s.%s' must be unique because it is referenced by "
                     "a foreign key." % (model_name, field_name),
+                    hint=(
+                        'Add unique=True to this field or add a '
+                        'UniqueConstraint (without condition) in the model '
+                        'Meta.constraints.'
+                    ),
                     obj=self,
                     id='fields.E311',
                 )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related.py | 531 | 531 | 4 | 4 | 1270
| django/db/models/fields/related.py | 544 | 545 | 4 | 4 | 1270
| django/db/models/fields/related.py | 556 | 556 | 4 | 4 | 1270


## Problem Statement

```
fields.E310-E311 should take into account UniqueConstraints without conditions.
Description
	
Hello, 
I'm trying to create migration with this kind of model.
class AppUsers(models.Model):
	name = models.CharField(...)
	uid = models.CharField(...)
	source = models.ForeignKey(...)
	class Meta:
		constraints = [models.UniqueConstraint(fields=['uid', 'source'], name='appusers_uniqueness')]
When I start makemigrations command in manage.py I've faced fields.E310 ​https://docs.djangoproject.com/en/2.2/ref/checks/#related-fields error 
It says that I should add unique_together field in Meta options:
app_name.AppUsers.field: (fields.E310) No subset of the fields 'uid', 'source' on model 'AppUsers' is unique.
HINT: Add unique=True on any of those fields or add at least a subset of them to a unique_together constraint.
If I change Meta options to unique_together constraint migration passes with no errors.
class AppUsers(models.Model):
	name = models.CharField(...)
	uid = models.CharField(...)
	source = models.ForeignKey(...)
	class Meta:
		unique_together = [['uid', 'source']]
As mentioned in docs ​https://docs.djangoproject.com/en/2.2/ref/models/options/#unique-together unique_together may be deprecated in the future. So I think nobody wants to face this issue when this will be deprecated :) 
Thanks,
Pavel

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/constraints.py | 72 | 126| 496 | 496 | 1049 | 
| 2 | 2 django/db/models/base.py | 1559 | 1584| 183 | 679 | 16399 | 
| 3 | 3 django/db/backends/base/schema.py | 370 | 384| 182 | 861 | 27714 | 
| **-> 4 <-** | **4 django/db/models/fields/related.py** | 509 | 563| 409 | 1270 | 41496 | 
| 5 | 4 django/db/backends/base/schema.py | 1064 | 1078| 126 | 1396 | 41496 | 
| 6 | 4 django/db/models/base.py | 1071 | 1114| 404 | 1800 | 41496 | 
| 7 | 4 django/db/models/base.py | 1162 | 1190| 213 | 2013 | 41496 | 
| 8 | **4 django/db/models/fields/related.py** | 1224 | 1341| 967 | 2980 | 41496 | 
| 9 | 5 django/db/migrations/operations/models.py | 807 | 836| 278 | 3258 | 48192 | 
| 10 | 5 django/db/models/base.py | 1532 | 1557| 183 | 3441 | 48192 | 
| 11 | 6 django/forms/models.py | 679 | 750| 732 | 4173 | 59819 | 
| 12 | 6 django/db/backends/base/schema.py | 1106 | 1136| 240 | 4413 | 59819 | 
| 13 | 6 django/db/migrations/operations/models.py | 544 | 563| 148 | 4561 | 59819 | 
| 14 | **6 django/db/models/fields/related.py** | 853 | 879| 240 | 4801 | 59819 | 
| 15 | 6 django/db/backends/base/schema.py | 1080 | 1104| 193 | 4994 | 59819 | 
| 16 | 7 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 5225 | 71554 | 
| 17 | 7 django/db/models/base.py | 1498 | 1530| 231 | 5456 | 71554 | 
| 18 | 7 django/db/migrations/operations/models.py | 839 | 874| 347 | 5803 | 71554 | 
| 19 | 7 django/db/models/base.py | 1615 | 1663| 348 | 6151 | 71554 | 
| 20 | 7 django/db/models/base.py | 983 | 1011| 230 | 6381 | 71554 | 
| 21 | 7 django/db/models/constraints.py | 1 | 27| 203 | 6584 | 71554 | 
| 22 | 7 django/db/migrations/autodetector.py | 1027 | 1043| 188 | 6772 | 71554 | 
| 23 | 8 django/db/models/options.py | 830 | 862| 225 | 6997 | 78650 | 
| 24 | 9 django/db/migrations/questioner.py | 56 | 81| 220 | 7217 | 80723 | 
| 25 | 9 django/db/models/base.py | 1838 | 1862| 174 | 7391 | 80723 | 
| 26 | **9 django/db/models/fields/related.py** | 1417 | 1458| 418 | 7809 | 80723 | 
| 27 | **9 django/db/models/fields/related.py** | 1343 | 1415| 616 | 8425 | 80723 | 
| 28 | **9 django/db/models/fields/related.py** | 487 | 507| 138 | 8563 | 80723 | 
| 29 | 9 django/forms/models.py | 752 | 773| 194 | 8757 | 80723 | 
| 30 | 9 django/db/models/base.py | 1392 | 1447| 491 | 9248 | 80723 | 
| 31 | 10 django/db/backends/mysql/schema.py | 106 | 120| 201 | 9449 | 82118 | 
| 32 | 10 django/db/migrations/operations/models.py | 528 | 541| 139 | 9588 | 82118 | 
| 33 | 10 django/forms/models.py | 309 | 348| 387 | 9975 | 82118 | 
| 34 | 10 django/db/migrations/operations/models.py | 1 | 38| 238 | 10213 | 82118 | 
| 35 | 10 django/db/backends/base/schema.py | 402 | 416| 174 | 10387 | 82118 | 
| 36 | **10 django/db/models/fields/related.py** | 1191 | 1222| 180 | 10567 | 82118 | 
| 37 | **10 django/db/models/fields/related.py** | 255 | 282| 269 | 10836 | 82118 | 
| 38 | 11 django/db/models/fields/__init__.py | 2289 | 2339| 339 | 11175 | 99705 | 
| 39 | 11 django/db/models/base.py | 1449 | 1472| 176 | 11351 | 99705 | 
| 40 | 11 django/db/migrations/autodetector.py | 1045 | 1065| 136 | 11487 | 99705 | 
| 41 | 12 django/contrib/auth/migrations/0001_initial.py | 1 | 105| 849 | 12336 | 100554 | 
| 42 | 12 django/db/migrations/questioner.py | 227 | 240| 123 | 12459 | 100554 | 
| 43 | 12 django/db/migrations/operations/models.py | 517 | 526| 129 | 12588 | 100554 | 
| 44 | 12 django/forms/models.py | 412 | 442| 243 | 12831 | 100554 | 
| 45 | 13 django/db/migrations/state.py | 600 | 611| 136 | 12967 | 105762 | 
| 46 | 13 django/db/models/base.py | 1145 | 1160| 138 | 13105 | 105762 | 
| 47 | 14 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 13242 | 105899 | 
| 48 | 14 django/db/migrations/autodetector.py | 1086 | 1121| 312 | 13554 | 105899 | 
| 49 | 14 django/db/models/base.py | 1765 | 1836| 565 | 14119 | 105899 | 
| 50 | 14 django/db/migrations/state.py | 579 | 598| 188 | 14307 | 105899 | 
| 51 | 15 django/db/models/lookups.py | 595 | 628| 141 | 14448 | 110722 | 
| 52 | **15 django/db/models/fields/related.py** | 127 | 154| 202 | 14650 | 110722 | 
| 53 | 15 django/db/backends/base/schema.py | 1048 | 1062| 123 | 14773 | 110722 | 
| 54 | 15 django/db/models/base.py | 1013 | 1069| 560 | 15333 | 110722 | 
| 55 | 15 django/forms/models.py | 382 | 410| 240 | 15573 | 110722 | 
| 56 | 15 django/db/migrations/questioner.py | 143 | 160| 183 | 15756 | 110722 | 
| 57 | **15 django/db/models/fields/related.py** | 156 | 169| 144 | 15900 | 110722 | 
| 58 | **15 django/db/models/fields/related.py** | 830 | 851| 169 | 16069 | 110722 | 
| 59 | **15 django/db/models/fields/related.py** | 108 | 125| 155 | 16224 | 110722 | 
| 60 | 15 django/db/migrations/operations/models.py | 591 | 607| 215 | 16439 | 110722 | 
| 61 | 15 django/db/models/fields/__init__.py | 1059 | 1088| 218 | 16657 | 110722 | 
| 62 | 16 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 35| 207 | 16864 | 110929 | 
| 63 | **16 django/db/models/fields/related.py** | 902 | 922| 178 | 17042 | 110929 | 
| 64 | 16 django/db/backends/base/schema.py | 1168 | 1197| 268 | 17310 | 110929 | 
| 65 | 16 django/db/models/base.py | 1253 | 1283| 255 | 17565 | 110929 | 
| 66 | 16 django/db/migrations/operations/models.py | 120 | 239| 827 | 18392 | 110929 | 
| 67 | 16 django/db/models/base.py | 1343 | 1373| 244 | 18636 | 110929 | 
| 68 | 16 django/db/models/constraints.py | 30 | 69| 348 | 18984 | 110929 | 
| 69 | 16 django/db/models/base.py | 1474 | 1496| 171 | 19155 | 110929 | 
| 70 | 17 django/db/backends/sqlite3/schema.py | 1 | 37| 317 | 19472 | 114893 | 
| 71 | **17 django/db/models/fields/related.py** | 973 | 984| 128 | 19600 | 114893 | 
| 72 | 17 django/db/backends/base/schema.py | 386 | 400| 185 | 19785 | 114893 | 
| 73 | 18 django/contrib/admin/checks.py | 231 | 261| 229 | 20014 | 123900 | 
| 74 | 19 django/db/migrations/operations/__init__.py | 1 | 18| 195 | 20209 | 124095 | 
| 75 | 19 django/db/models/fields/__init__.py | 2342 | 2391| 311 | 20520 | 124095 | 
| 76 | 19 django/db/models/options.py | 147 | 206| 587 | 21107 | 124095 | 
| 77 | 19 django/db/backends/sqlite3/schema.py | 367 | 413| 444 | 21551 | 124095 | 
| 78 | 20 django/contrib/contenttypes/fields.py | 332 | 354| 185 | 21736 | 129528 | 
| 79 | 20 django/db/models/fields/__init__.py | 308 | 336| 205 | 21941 | 129528 | 
| 80 | 21 django/db/models/deletion.py | 1 | 76| 566 | 22507 | 133343 | 
| 81 | 22 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 22657 | 133493 | 
| 82 | **22 django/db/models/fields/related.py** | 171 | 188| 166 | 22823 | 133493 | 
| 83 | 22 django/db/backends/mysql/schema.py | 91 | 104| 148 | 22971 | 133493 | 
| 84 | **22 django/db/models/fields/related.py** | 739 | 757| 222 | 23193 | 133493 | 
| 85 | 22 django/db/migrations/state.py | 348 | 398| 471 | 23664 | 133493 | 
| 86 | 22 django/forms/models.py | 952 | 985| 367 | 24031 | 133493 | 
| 87 | 22 django/db/models/fields/__init__.py | 338 | 365| 203 | 24234 | 133493 | 
| 88 | 22 django/db/models/base.py | 1375 | 1390| 153 | 24387 | 133493 | 
| 89 | 23 django/contrib/postgres/constraints.py | 53 | 63| 127 | 24514 | 134339 | 
| 90 | 23 django/db/migrations/operations/models.py | 488 | 515| 181 | 24695 | 134339 | 
| 91 | 23 django/db/models/base.py | 1192 | 1226| 230 | 24925 | 134339 | 
| 92 | 23 django/contrib/postgres/constraints.py | 65 | 105| 306 | 25231 | 134339 | 
| 93 | 23 django/db/backends/base/schema.py | 31 | 41| 120 | 25351 | 134339 | 
| 94 | **23 django/db/models/fields/related.py** | 190 | 254| 673 | 26024 | 134339 | 
| 95 | 23 django/db/migrations/state.py | 495 | 527| 250 | 26274 | 134339 | 
| 96 | 23 django/db/migrations/operations/models.py | 102 | 118| 147 | 26421 | 134339 | 
| 97 | 23 django/contrib/contenttypes/fields.py | 677 | 702| 254 | 26675 | 134339 | 
| 98 | 24 django/db/migrations/operations/utils.py | 1 | 14| 138 | 26813 | 134819 | 
| 99 | 24 django/db/backends/base/schema.py | 526 | 565| 470 | 27283 | 134819 | 
| 100 | 24 django/db/backends/sqlite3/schema.py | 101 | 138| 486 | 27769 | 134819 | 
| 101 | 24 django/db/backends/sqlite3/schema.py | 223 | 305| 731 | 28500 | 134819 | 
| 102 | 25 django/db/models/query_utils.py | 284 | 309| 293 | 28793 | 137531 | 
| 103 | 25 django/db/backends/base/schema.py | 1031 | 1046| 170 | 28963 | 137531 | 
| 104 | **25 django/db/models/fields/related.py** | 1016 | 1063| 368 | 29331 | 137531 | 
| 105 | 25 django/db/migrations/autodetector.py | 525 | 671| 1109 | 30440 | 137531 | 
| 106 | 26 django/contrib/auth/migrations/0008_alter_user_username_max_length.py | 1 | 25| 138 | 30578 | 137669 | 
| 107 | 26 django/db/migrations/autodetector.py | 1067 | 1084| 180 | 30758 | 137669 | 
| 108 | **26 django/db/models/fields/related.py** | 600 | 617| 197 | 30955 | 137669 | 
| 109 | **26 django/db/models/fields/related.py** | 284 | 318| 293 | 31248 | 137669 | 
| 110 | **26 django/db/models/fields/related.py** | 1595 | 1632| 484 | 31732 | 137669 | 
| 111 | 26 django/db/models/base.py | 1665 | 1763| 717 | 32449 | 137669 | 
| 112 | 26 django/db/models/fields/__init__.py | 367 | 393| 199 | 32648 | 137669 | 
| 113 | 26 django/db/migrations/questioner.py | 162 | 185| 246 | 32894 | 137669 | 
| 114 | **26 django/db/models/fields/related.py** | 924 | 937| 126 | 33020 | 137669 | 
| 115 | 26 django/db/backends/base/schema.py | 626 | 698| 792 | 33812 | 137669 | 
| 116 | 26 django/db/backends/sqlite3/schema.py | 140 | 221| 820 | 34632 | 137669 | 
| 117 | 26 django/db/models/fields/__init__.py | 208 | 242| 235 | 34867 | 137669 | 
| 118 | 27 django/db/models/fields/related_lookups.py | 62 | 99| 447 | 35314 | 139118 | 
| 119 | 27 django/db/models/base.py | 404 | 503| 856 | 36170 | 139118 | 
| 120 | 27 django/db/migrations/autodetector.py | 89 | 101| 118 | 36288 | 139118 | 
| 121 | 28 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 36903 | 140143 | 
| 122 | 28 django/db/migrations/autodetector.py | 437 | 463| 256 | 37159 | 140143 | 
| 123 | 28 django/db/backends/sqlite3/schema.py | 86 | 99| 181 | 37340 | 140143 | 
| 124 | 28 django/db/models/fields/__init__.py | 1 | 81| 633 | 37973 | 140143 | 
| 125 | 28 django/db/migrations/operations/models.py | 345 | 394| 493 | 38466 | 140143 | 
| 126 | 28 django/db/migrations/operations/models.py | 609 | 622| 137 | 38603 | 140143 | 
| 127 | 28 django/contrib/contenttypes/fields.py | 430 | 451| 248 | 38851 | 140143 | 
| 128 | 28 django/db/migrations/autodetector.py | 707 | 794| 789 | 39640 | 140143 | 
| 129 | 28 django/db/models/options.py | 256 | 288| 331 | 39971 | 140143 | 
| 130 | 28 django/contrib/contenttypes/fields.py | 598 | 629| 278 | 40249 | 140143 | 
| 131 | 28 django/contrib/contenttypes/fields.py | 110 | 158| 328 | 40577 | 140143 | 
| 132 | 28 django/db/backends/base/schema.py | 769 | 809| 511 | 41088 | 140143 | 
| 133 | 28 django/db/models/options.py | 1 | 34| 285 | 41373 | 140143 | 
| 134 | 29 django/contrib/admin/options.py | 421 | 464| 350 | 41723 | 158631 | 
| 135 | 29 django/db/models/fields/__init__.py | 244 | 306| 448 | 42171 | 158631 | 
| 136 | 30 django/db/migrations/operations/fields.py | 91 | 101| 125 | 42296 | 161897 | 
| 137 | 30 django/db/migrations/operations/fields.py | 302 | 355| 535 | 42831 | 161897 | 
| 138 | 31 django/db/models/__init__.py | 1 | 52| 605 | 43436 | 162502 | 
| 139 | 32 django/db/migrations/exceptions.py | 1 | 55| 249 | 43685 | 162752 | 
| 140 | 32 django/db/backends/base/schema.py | 1 | 28| 194 | 43879 | 162752 | 
| 141 | 32 django/contrib/admin/options.py | 367 | 419| 504 | 44383 | 162752 | 
| 142 | 33 django/contrib/admin/migrations/0001_initial.py | 1 | 48| 320 | 44703 | 163072 | 
| 143 | 33 django/forms/models.py | 279 | 307| 288 | 44991 | 163072 | 
| 144 | 33 django/db/backends/mysql/schema.py | 79 | 89| 138 | 45129 | 163072 | 
| 145 | 34 django/db/models/fields/related_descriptors.py | 672 | 730| 548 | 45677 | 173467 | 
| 146 | 35 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 40| 217 | 45894 | 173684 | 
| 147 | 35 django/contrib/admin/checks.py | 1020 | 1047| 194 | 46088 | 173684 | 
| 148 | 36 django/core/checks/model_checks.py | 1 | 86| 667 | 46755 | 175471 | 
| 149 | 36 django/db/models/options.py | 37 | 60| 203 | 46958 | 175471 | 
| 150 | 36 django/forms/models.py | 1256 | 1286| 242 | 47200 | 175471 | 
| 151 | 36 django/core/checks/model_checks.py | 178 | 211| 332 | 47532 | 175471 | 
| 152 | 36 django/forms/models.py | 350 | 380| 233 | 47765 | 175471 | 
| 153 | 36 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 48176 | 175471 | 
| 154 | 37 django/forms/fields.py | 1184 | 1214| 182 | 48358 | 184484 | 
| 155 | 37 django/db/models/deletion.py | 269 | 344| 798 | 49156 | 184484 | 
| 156 | 37 django/db/backends/base/schema.py | 699 | 768| 740 | 49896 | 184484 | 
| 157 | 37 django/db/migrations/questioner.py | 207 | 224| 171 | 50067 | 184484 | 
| 158 | 38 django/contrib/auth/checks.py | 102 | 208| 776 | 50843 | 185956 | 
| 159 | 38 django/db/models/fields/related_lookups.py | 119 | 155| 244 | 51087 | 185956 | 
| 160 | 39 django/contrib/sites/migrations/0001_initial.py | 1 | 32| 191 | 51278 | 186147 | 
| 161 | **39 django/db/models/fields/related.py** | 986 | 1013| 215 | 51493 | 186147 | 
| 162 | 39 django/db/migrations/operations/models.py | 460 | 485| 279 | 51772 | 186147 | 
| 163 | 39 django/db/models/fields/related_lookups.py | 102 | 117| 215 | 51987 | 186147 | 
| 164 | **39 django/db/models/fields/related.py** | 1652 | 1686| 266 | 52253 | 186147 | 
| 165 | 40 django/contrib/flatpages/migrations/0001_initial.py | 1 | 40| 307 | 52560 | 186454 | 
| 166 | 40 django/db/migrations/operations/fields.py | 357 | 384| 289 | 52849 | 186454 | 
| 167 | **40 django/db/models/fields/related.py** | 1066 | 1110| 407 | 53256 | 186454 | 
| 168 | 40 django/db/migrations/operations/fields.py | 103 | 123| 223 | 53479 | 186454 | 
| 169 | 40 django/db/migrations/autodetector.py | 1182 | 1207| 245 | 53724 | 186454 | 
| 170 | 41 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 53846 | 186703 | 
| 171 | 41 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 54251 | 186703 | 
| 172 | 41 django/db/migrations/autodetector.py | 358 | 372| 141 | 54392 | 186703 | 
| 173 | 42 django/db/models/indexes.py | 79 | 118| 411 | 54803 | 187876 | 
| 174 | 42 django/contrib/admin/checks.py | 1049 | 1091| 343 | 55146 | 187876 | 
| 175 | 42 django/db/migrations/autodetector.py | 796 | 845| 570 | 55716 | 187876 | 
| 176 | 42 django/db/models/fields/related_descriptors.py | 309 | 323| 182 | 55898 | 187876 | 
| 177 | 42 django/db/models/base.py | 1 | 50| 327 | 56225 | 187876 | 
| 178 | 42 django/db/migrations/autodetector.py | 904 | 985| 876 | 57101 | 187876 | 
| 179 | 43 django/db/backends/mysql/introspection.py | 167 | 253| 729 | 57830 | 190104 | 
| 180 | 43 django/contrib/postgres/constraints.py | 1 | 51| 427 | 58257 | 190104 | 
| 181 | 43 django/db/backends/mysql/introspection.py | 254 | 270| 184 | 58441 | 190104 | 
| 182 | 43 django/db/migrations/operations/models.py | 304 | 343| 406 | 58847 | 190104 | 
| 183 | 43 django/db/models/fields/related_lookups.py | 46 | 60| 224 | 59071 | 190104 | 
| 184 | 43 django/db/migrations/operations/fields.py | 241 | 251| 146 | 59217 | 190104 | 
| 185 | 43 django/db/migrations/operations/fields.py | 39 | 67| 285 | 59502 | 190104 | 
| 186 | 43 django/db/models/base.py | 1116 | 1143| 286 | 59788 | 190104 | 
| 187 | 43 django/db/migrations/operations/fields.py | 1 | 37| 236 | 60024 | 190104 | 
| 188 | 43 django/db/models/base.py | 567 | 586| 170 | 60194 | 190104 | 
| 189 | 44 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 60982 | 192853 | 
| 190 | 44 django/db/backends/base/schema.py | 1138 | 1166| 284 | 61266 | 192853 | 
| 191 | 44 django/db/models/fields/__init__.py | 2422 | 2447| 143 | 61409 | 192853 | 
| 192 | 44 django/db/backends/base/schema.py | 567 | 625| 676 | 62085 | 192853 | 
| 193 | 44 django/contrib/admin/checks.py | 161 | 202| 325 | 62410 | 192853 | 
| 194 | 45 django/contrib/redirects/migrations/0001_initial.py | 1 | 41| 274 | 62684 | 193127 | 
| 195 | 46 django/db/backends/oracle/schema.py | 57 | 77| 249 | 62933 | 194882 | 
| 196 | 46 django/db/backends/base/schema.py | 317 | 332| 154 | 63087 | 194882 | 
| 197 | 46 django/contrib/contenttypes/fields.py | 20 | 108| 571 | 63658 | 194882 | 
| 198 | 46 django/db/models/base.py | 1312 | 1341| 205 | 63863 | 194882 | 
| 199 | 46 django/db/migrations/operations/fields.py | 175 | 193| 236 | 64099 | 194882 | 
| 200 | 46 django/db/models/base.py | 1285 | 1310| 184 | 64283 | 194882 | 
| 201 | 46 django/contrib/admin/checks.py | 204 | 214| 127 | 64410 | 194882 | 
| 202 | 46 django/forms/models.py | 1359 | 1386| 209 | 64619 | 194882 | 
| 203 | 46 django/db/migrations/operations/models.py | 625 | 677| 342 | 64961 | 194882 | 
| 204 | 46 django/db/migrations/operations/utils.py | 17 | 54| 340 | 65301 | 194882 | 
| 205 | 47 django/db/backends/sqlite3/introspection.py | 206 | 220| 146 | 65447 | 198591 | 
| 206 | 48 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 65447 | 198688 | 
| 207 | 48 django/db/backends/base/schema.py | 906 | 933| 327 | 65774 | 198688 | 
| 208 | 48 django/db/models/fields/__init__.py | 1738 | 1765| 215 | 65989 | 198688 | 
| 209 | **48 django/db/models/fields/related.py** | 1634 | 1650| 286 | 66275 | 198688 | 
| 210 | 48 django/db/migrations/autodetector.py | 987 | 1003| 188 | 66463 | 198688 | 
| 211 | 48 django/db/migrations/operations/fields.py | 220 | 239| 205 | 66668 | 198688 | 
| 212 | 48 django/db/models/fields/__init__.py | 1713 | 1736| 146 | 66814 | 198688 | 
| 213 | 48 django/contrib/admin/checks.py | 1094 | 1123| 178 | 66992 | 198688 | 
| 214 | 48 django/contrib/contenttypes/fields.py | 630 | 654| 214 | 67206 | 198688 | 
| 215 | 48 django/contrib/admin/checks.py | 354 | 370| 134 | 67340 | 198688 | 
| 216 | 49 django/contrib/sessions/migrations/0001_initial.py | 1 | 31| 162 | 67502 | 198850 | 


### Hint

```
Agreed, both checks should take into UniqueConstraint's without condition's.
Posting a patch soon
```

## Patch

```diff
diff --git a/django/db/models/fields/related.py b/django/db/models/fields/related.py
--- a/django/db/models/fields/related.py
+++ b/django/db/models/fields/related.py
@@ -528,6 +528,10 @@ def _check_unique_target(self):
             frozenset(ut)
             for ut in self.remote_field.model._meta.unique_together
         })
+        unique_foreign_fields.update({
+            frozenset(uc.fields)
+            for uc in self.remote_field.model._meta.total_unique_constraints
+        })
         foreign_fields = {f.name for f in self.foreign_related_fields}
         has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)
 
@@ -541,8 +545,10 @@ def _check_unique_target(self):
                     "No subset of the fields %s on model '%s' is unique."
                     % (field_combination, model_name),
                     hint=(
-                        "Add unique=True on any of those fields or add at "
-                        "least a subset of them to a unique_together constraint."
+                        'Mark a single field as unique=True or add a set of '
+                        'fields to a unique constraint (via unique_together '
+                        'or a UniqueConstraint (without condition) in the '
+                        'model Meta.constraints).'
                     ),
                     obj=self,
                     id='fields.E310',
@@ -553,8 +559,13 @@ def _check_unique_target(self):
             model_name = self.remote_field.model.__name__
             return [
                 checks.Error(
-                    "'%s.%s' must set unique=True because it is referenced by "
+                    "'%s.%s' must be unique because it is referenced by "
                     "a foreign key." % (model_name, field_name),
+                    hint=(
+                        'Add unique=True to this field or add a '
+                        'UniqueConstraint (without condition) in the model '
+                        'Meta.constraints.'
+                    ),
                     obj=self,
                     id='fields.E311',
                 )

```

## Test Patch

```diff
diff --git a/tests/invalid_models_tests/test_relative_fields.py b/tests/invalid_models_tests/test_relative_fields.py
--- a/tests/invalid_models_tests/test_relative_fields.py
+++ b/tests/invalid_models_tests/test_relative_fields.py
@@ -352,7 +352,11 @@ class Model(models.Model):
         field = Model._meta.get_field('foreign_key')
         self.assertEqual(field.check(), [
             Error(
-                "'Target.bad' must set unique=True because it is referenced by a foreign key.",
+                "'Target.bad' must be unique because it is referenced by a foreign key.",
+                hint=(
+                    'Add unique=True to this field or add a UniqueConstraint '
+                    '(without condition) in the model Meta.constraints.'
+                ),
                 obj=field,
                 id='fields.E311',
             ),
@@ -368,12 +372,64 @@ class Model(models.Model):
         field = Model._meta.get_field('field')
         self.assertEqual(field.check(), [
             Error(
-                "'Target.bad' must set unique=True because it is referenced by a foreign key.",
+                "'Target.bad' must be unique because it is referenced by a foreign key.",
+                hint=(
+                    'Add unique=True to this field or add a UniqueConstraint '
+                    '(without condition) in the model Meta.constraints.'
+                ),
                 obj=field,
                 id='fields.E311',
             ),
         ])
 
+    def test_foreign_key_to_partially_unique_field(self):
+        class Target(models.Model):
+            source = models.IntegerField()
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(
+                        fields=['source'],
+                        name='tfktpuf_partial_unique',
+                        condition=models.Q(pk__gt=2),
+                    ),
+                ]
+
+        class Model(models.Model):
+            field = models.ForeignKey(Target, models.CASCADE, to_field='source')
+
+        field = Model._meta.get_field('field')
+        self.assertEqual(field.check(), [
+            Error(
+                "'Target.source' must be unique because it is referenced by a "
+                "foreign key.",
+                hint=(
+                    'Add unique=True to this field or add a UniqueConstraint '
+                    '(without condition) in the model Meta.constraints.'
+                ),
+                obj=field,
+                id='fields.E311',
+            ),
+        ])
+
+    def test_foreign_key_to_unique_field_with_meta_constraint(self):
+        class Target(models.Model):
+            source = models.IntegerField()
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(
+                        fields=['source'],
+                        name='tfktufwmc_unique',
+                    ),
+                ]
+
+        class Model(models.Model):
+            field = models.ForeignKey(Target, models.CASCADE, to_field='source')
+
+        field = Model._meta.get_field('field')
+        self.assertEqual(field.check(), [])
+
     def test_foreign_object_to_non_unique_fields(self):
         class Person(models.Model):
             # Note that both fields are not unique.
@@ -396,14 +452,82 @@ class MMembership(models.Model):
             Error(
                 "No subset of the fields 'country_id', 'city_id' on model 'Person' is unique.",
                 hint=(
-                    "Add unique=True on any of those fields or add at least "
-                    "a subset of them to a unique_together constraint."
+                    'Mark a single field as unique=True or add a set of '
+                    'fields to a unique constraint (via unique_together or a '
+                    'UniqueConstraint (without condition) in the model '
+                    'Meta.constraints).'
                 ),
                 obj=field,
                 id='fields.E310',
             )
         ])
 
+    def test_foreign_object_to_partially_unique_field(self):
+        class Person(models.Model):
+            country_id = models.IntegerField()
+            city_id = models.IntegerField()
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(
+                        fields=['country_id', 'city_id'],
+                        name='tfotpuf_partial_unique',
+                        condition=models.Q(pk__gt=2),
+                    ),
+                ]
+
+        class MMembership(models.Model):
+            person_country_id = models.IntegerField()
+            person_city_id = models.IntegerField()
+            person = models.ForeignObject(
+                Person,
+                on_delete=models.CASCADE,
+                from_fields=['person_country_id', 'person_city_id'],
+                to_fields=['country_id', 'city_id'],
+            )
+
+        field = MMembership._meta.get_field('person')
+        self.assertEqual(field.check(), [
+            Error(
+                "No subset of the fields 'country_id', 'city_id' on model "
+                "'Person' is unique.",
+                hint=(
+                    'Mark a single field as unique=True or add a set of '
+                    'fields to a unique constraint (via unique_together or a '
+                    'UniqueConstraint (without condition) in the model '
+                    'Meta.constraints).'
+                ),
+                obj=field,
+                id='fields.E310',
+            ),
+        ])
+
+    def test_foreign_object_to_unique_field_with_meta_constraint(self):
+        class Person(models.Model):
+            country_id = models.IntegerField()
+            city_id = models.IntegerField()
+
+            class Meta:
+                constraints = [
+                    models.UniqueConstraint(
+                        fields=['country_id', 'city_id'],
+                        name='tfotpuf_unique',
+                    ),
+                ]
+
+        class MMembership(models.Model):
+            person_country_id = models.IntegerField()
+            person_city_id = models.IntegerField()
+            person = models.ForeignObject(
+                Person,
+                on_delete=models.CASCADE,
+                from_fields=['person_country_id', 'person_city_id'],
+                to_fields=['country_id', 'city_id'],
+            )
+
+        field = MMembership._meta.get_field('person')
+        self.assertEqual(field.check(), [])
+
     def test_on_delete_set_null_on_non_nullable_field(self):
         class Person(models.Model):
             pass
@@ -1453,8 +1577,10 @@ class Child(models.Model):
             Error(
                 "No subset of the fields 'a', 'b' on model 'Parent' is unique.",
                 hint=(
-                    "Add unique=True on any of those fields or add at least "
-                    "a subset of them to a unique_together constraint."
+                    'Mark a single field as unique=True or add a set of '
+                    'fields to a unique constraint (via unique_together or a '
+                    'UniqueConstraint (without condition) in the model '
+                    'Meta.constraints).'
                 ),
                 obj=field,
                 id='fields.E310',
@@ -1489,8 +1615,10 @@ class Child(models.Model):
             Error(
                 "No subset of the fields 'a', 'b', 'd' on model 'Parent' is unique.",
                 hint=(
-                    "Add unique=True on any of those fields or add at least "
-                    "a subset of them to a unique_together constraint."
+                    'Mark a single field as unique=True or add a set of '
+                    'fields to a unique constraint (via unique_together or a '
+                    'UniqueConstraint (without condition) in the model '
+                    'Meta.constraints).'
                 ),
                 obj=field,
                 id='fields.E310',

```


## Code snippets

### 1 - django/db/models/constraints.py:

Start line: 72, End line: 126

```python
class UniqueConstraint(BaseConstraint):
    def __init__(self, *, fields, name, condition=None):
        if not fields:
            raise ValueError('At least one field is required to define a unique constraint.')
        if not isinstance(condition, (type(None), Q)):
            raise ValueError('UniqueConstraint.condition must be a Q instance.')
        self.fields = tuple(fields)
        self.condition = condition
        super().__init__(name)

    def _get_condition_sql(self, model, schema_editor):
        if self.condition is None:
            return None
        query = Query(model=model, alias_cols=False)
        where = query.build_where(self.condition)
        compiler = query.get_compiler(connection=schema_editor.connection)
        sql, params = where.as_sql(compiler, schema_editor.connection)
        return sql % tuple(schema_editor.quote_value(p) for p in params)

    def constraint_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name).column for field_name in self.fields]
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._unique_sql(model, fields, self.name, condition=condition)

    def create_sql(self, model, schema_editor):
        fields = [model._meta.get_field(field_name).column for field_name in self.fields]
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._create_unique_sql(model, fields, self.name, condition=condition)

    def remove_sql(self, model, schema_editor):
        condition = self._get_condition_sql(model, schema_editor)
        return schema_editor._delete_unique_sql(model, self.name, condition=condition)

    def __repr__(self):
        return '<%s: fields=%r name=%r%s>' % (
            self.__class__.__name__, self.fields, self.name,
            '' if self.condition is None else ' condition=%s' % self.condition,
        )

    def __eq__(self, other):
        if isinstance(other, UniqueConstraint):
            return (
                self.name == other.name and
                self.fields == other.fields and
                self.condition == other.condition
            )
        return super().__eq__(other)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs['fields'] = self.fields
        if self.condition:
            kwargs['condition'] = self.condition
        return path, args, kwargs
```
### 2 - django/db/models/base.py:

Start line: 1559, End line: 1584

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_unique_together(cls):
        """Check the value of "unique_together" option."""
        if not isinstance(cls._meta.unique_together, (tuple, list)):
            return [
                checks.Error(
                    "'unique_together' must be a list or tuple.",
                    obj=cls,
                    id='models.E010',
                )
            ]

        elif any(not isinstance(fields, (tuple, list)) for fields in cls._meta.unique_together):
            return [
                checks.Error(
                    "All 'unique_together' elements must be lists or tuples.",
                    obj=cls,
                    id='models.E011',
                )
            ]

        else:
            errors = []
            for fields in cls._meta.unique_together:
                errors.extend(cls._check_local_fields(fields, "unique_together"))
            return errors
```
### 3 - django/db/backends/base/schema.py:

Start line: 370, End line: 384

```python
class BaseDatabaseSchemaEditor:

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for fields in news.difference(olds):
            columns = [model._meta.get_field(field).column for field in fields]
            self.execute(self._create_unique_sql(model, columns))
```
### 4 - django/db/models/fields/related.py:

Start line: 509, End line: 563

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
                        "Add unique=True on any of those fields or add at "
                        "least a subset of them to a unique_together constraint."
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
                    "'%s.%s' must set unique=True because it is referenced by "
                    "a foreign key." % (model_name, field_name),
                    obj=self,
                    id='fields.E311',
                )
            ]
        else:
            return []
```
### 5 - django/db/backends/base/schema.py:

Start line: 1064, End line: 1078

```python
class BaseDatabaseSchemaEditor:

    def _unique_sql(self, model, fields, name, condition=None):
        if condition:
            # Databases support conditional unique constraints via a unique
            # index.
            sql = self._create_unique_sql(model, fields, name=name, condition=condition)
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            'columns': ', '.join(map(self.quote_name, fields)),
        }
        return self.sql_constraint % {
            'name': self.quote_name(name),
            'constraint': constraint,
        }
```
### 6 - django/db/models/base.py:

Start line: 1071, End line: 1114

```python
class Model(metaclass=ModelBase):

    def _perform_unique_checks(self, unique_checks):
        errors = {}

        for model_class, unique_check in unique_checks:
            # Try to look up an existing object with the same values as this
            # object's values for all the unique field.

            lookup_kwargs = {}
            for field_name in unique_check:
                f = self._meta.get_field(field_name)
                lookup_value = getattr(self, f.attname)
                # TODO: Handle multiple backends with different feature flags.
                if (lookup_value is None or
                        (lookup_value == '' and connection.features.interprets_empty_strings_as_nulls)):
                    # no value, skip the lookup
                    continue
                if f.primary_key and not self._state.adding:
                    # no need to check for unique primary key when editing
                    continue
                lookup_kwargs[str(field_name)] = lookup_value

            # some fields were skipped, no reason to do the check
            if len(unique_check) != len(lookup_kwargs):
                continue

            qs = model_class._default_manager.filter(**lookup_kwargs)

            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            # Note that we need to use the pk as defined by model_class, not
            # self.pk. These can be different fields because model inheritance
            # allows single model to have effectively multiple primary keys.
            # Refs #17615.
            model_class_pk = self._get_pk_val(model_class._meta)
            if not self._state.adding and model_class_pk is not None:
                qs = qs.exclude(pk=model_class_pk)
            if qs.exists():
                if len(unique_check) == 1:
                    key = unique_check[0]
                else:
                    key = NON_FIELD_ERRORS
                errors.setdefault(key, []).append(self.unique_error_message(model_class, unique_check))

        return errors
```
### 7 - django/db/models/base.py:

Start line: 1162, End line: 1190

```python
class Model(metaclass=ModelBase):

    def unique_error_message(self, model_class, unique_check):
        opts = model_class._meta

        params = {
            'model': self,
            'model_class': model_class,
            'model_name': capfirst(opts.verbose_name),
            'unique_check': unique_check,
        }

        # A unique field
        if len(unique_check) == 1:
            field = opts.get_field(unique_check[0])
            params['field_label'] = capfirst(field.verbose_name)
            return ValidationError(
                message=field.error_messages['unique'],
                code='unique',
                params=params,
            )

        # unique_together
        else:
            field_labels = [capfirst(opts.get_field(f).verbose_name) for f in unique_check]
            params['field_labels'] = get_text_list(field_labels, _('and'))
            return ValidationError(
                message=_("%(model_name)s with this %(field_labels)s already exists."),
                code='unique_together',
                params=params,
            )
```
### 8 - django/db/models/fields/related.py:

Start line: 1224, End line: 1341

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
                                'use ForeignKey("%s", symmetrical=False, through="%s").'
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
                                'use ForeignKey("%s", symmetrical=False, through="%s").'
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
### 9 - django/db/migrations/operations/models.py:

Start line: 807, End line: 836

```python
class AddConstraint(IndexOperation):
    option_name = 'constraints'

    def __init__(self, model_name, constraint):
        self.model_name = model_name
        self.constraint = constraint

    def state_forwards(self, app_label, state):
        model_state = state.models[app_label, self.model_name_lower]
        model_state.options[self.option_name] = [*model_state.options[self.option_name], self.constraint]
        state.reload_model(app_label, self.model_name_lower, delay=True)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.add_constraint(model, self.constraint)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.remove_constraint(model, self.constraint)

    def deconstruct(self):
        return self.__class__.__name__, [], {
            'model_name': self.model_name,
            'constraint': self.constraint,
        }

    def describe(self):
        return 'Create constraint %s on model %s' % (self.constraint.name, self.model_name)
```
### 10 - django/db/models/base.py:

Start line: 1532, End line: 1557

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_index_together(cls):
        """Check the value of "index_together" option."""
        if not isinstance(cls._meta.index_together, (tuple, list)):
            return [
                checks.Error(
                    "'index_together' must be a list or tuple.",
                    obj=cls,
                    id='models.E008',
                )
            ]

        elif any(not isinstance(fields, (tuple, list)) for fields in cls._meta.index_together):
            return [
                checks.Error(
                    "All 'index_together' elements must be lists or tuples.",
                    obj=cls,
                    id='models.E009',
                )
            ]

        else:
            errors = []
            for fields in cls._meta.index_together:
                errors.extend(cls._check_local_fields(fields, "index_together"))
            return errors
```
### 14 - django/db/models/fields/related.py:

Start line: 853, End line: 879

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
### 26 - django/db/models/fields/related.py:

Start line: 1417, End line: 1458

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
### 27 - django/db/models/fields/related.py:

Start line: 1343, End line: 1415

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
### 28 - django/db/models/fields/related.py:

Start line: 487, End line: 507

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
### 36 - django/db/models/fields/related.py:

Start line: 1191, End line: 1222

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
        if (self.remote_field.limit_choices_to and self.remote_field.through and
                not self.remote_field.through._meta.auto_created):
            warnings.append(
                checks.Warning(
                    'limit_choices_to has no effect on ManyToManyField '
                    'with a through model.',
                    obj=self,
                    id='fields.W343',
                )
            )

        return warnings
```
### 37 - django/db/models/fields/related.py:

Start line: 255, End line: 282

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            clash_name = "%s.%s" % (  # i. e. "Model.m2m"
                clash_field.related_model._meta.object_name,
                clash_field.field.name)
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
### 52 - django/db/models/fields/related.py:

Start line: 127, End line: 154

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
                    % (rel_query_name,),
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
### 57 - django/db/models/fields/related.py:

Start line: 156, End line: 169

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
### 58 - django/db/models/fields/related.py:

Start line: 830, End line: 851

```python
class ForeignKey(ForeignObject):

    def _check_on_delete(self):
        on_delete = getattr(self.remote_field, 'on_delete', None)
        if on_delete == SET_NULL and not self.null:
            return [
                checks.Error(
                    'Field specifies on_delete=SET_NULL, but cannot be null.',
                    hint='Set null=True argument on the field, or change the on_delete rule.',
                    obj=self,
                    id='fields.E320',
                )
            ]
        elif on_delete == SET_DEFAULT and not self.has_default():
            return [
                checks.Error(
                    'Field specifies on_delete=SET_DEFAULT, but has no default value.',
                    hint='Set a default value, or change the on_delete rule.',
                    obj=self,
                    id='fields.E321',
                )
            ]
        else:
            return []
```
### 59 - django/db/models/fields/related.py:

Start line: 108, End line: 125

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
### 63 - django/db/models/fields/related.py:

Start line: 902, End line: 922

```python
class ForeignKey(ForeignObject):

    def validate(self, value, model_instance):
        if self.remote_field.parent_link:
            return
        super().validate(value, model_instance)
        if value is None:
            return

        using = router.db_for_read(self.remote_field.model, instance=model_instance)
        qs = self.remote_field.model._default_manager.using(using).filter(
            **{self.remote_field.field_name: value}
        )
        qs = qs.complex_filter(self.get_limit_choices_to())
        if not qs.exists():
            raise exceptions.ValidationError(
                self.error_messages['invalid'],
                code='invalid',
                params={
                    'model': self.remote_field.model._meta.verbose_name, 'pk': value,
                    'field': self.remote_field.field_name, 'value': value,
                },  # 'pk' is included for backwards compatibility
            )
```
### 71 - django/db/models/fields/related.py:

Start line: 973, End line: 984

```python
class ForeignKey(ForeignObject):

    def formfield(self, *, using=None, **kwargs):
        if isinstance(self.remote_field.model, str):
            raise ValueError("Cannot create form field for %r yet, because "
                             "its related model %r has not been loaded yet" %
                             (self.name, self.remote_field.model))
        return super().formfield(**{
            'form_class': forms.ModelChoiceField,
            'queryset': self.remote_field.model._default_manager.using(using),
            'to_field_name': self.remote_field.field_name,
            **kwargs,
            'blank': self.blank,
        })
```
### 82 - django/db/models/fields/related.py:

Start line: 171, End line: 188

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (self.remote_field.model not in self.opts.apps.get_models() and
                not isinstance(self.remote_field.model, str) and
                self.remote_field.model._meta.swapped):
            model = "%s.%s" % (
                self.remote_field.model._meta.app_label,
                self.remote_field.model._meta.object_name
            )
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % model,
                    hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable,
                    obj=self,
                    id='fields.E301',
                )
            ]
        return []
```
### 84 - django/db/models/fields/related.py:

Start line: 739, End line: 757

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
### 94 - django/db/models/fields/related.py:

Start line: 190, End line: 254

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
        field_name = "%s.%s" % (opts.object_name, self.name)  # i. e. "Model.field"

        # Check clashes between accessor or reverse query name of `field`
        # and any other field name -- i.e. accessor for Model.foreign is
        # model_set and it clashes with Target.model_set.
        potential_clashes = rel_opts.fields + rel_opts.many_to_many
        for clash_field in potential_clashes:
            clash_name = "%s.%s" % (rel_opts.object_name, clash_field.name)  # i.e. "Target.model_set"
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
### 104 - django/db/models/fields/related.py:

Start line: 1016, End line: 1063

```python
class OneToOneField(ForeignKey):
    """
    A OneToOneField is essentially the same as a ForeignKey, with the exception
    that it always carries a "unique" constraint with it and the reverse
    relation always returns the object pointed to (since there will only ever
    be one), rather than returning a list.
    """

    # Field flags
    many_to_many = False
    many_to_one = False
    one_to_many = False
    one_to_one = True

    related_accessor_class = ReverseOneToOneDescriptor
    forward_related_accessor_class = ForwardOneToOneDescriptor
    rel_class = OneToOneRel

    description = _("One-to-one relationship")

    def __init__(self, to, on_delete, to_field=None, **kwargs):
        kwargs['unique'] = True
        super().__init__(to, on_delete, to_field=to_field, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if "unique" in kwargs:
            del kwargs['unique']
        return name, path, args, kwargs

    def formfield(self, **kwargs):
        if self.remote_field.parent_link:
            return None
        return super().formfield(**kwargs)

    def save_form_data(self, instance, data):
        if isinstance(data, self.remote_field.model):
            setattr(instance, self.name, data)
        else:
            setattr(instance, self.attname, data)
            # Remote field object must be cleared otherwise Model.save()
            # will reassign attname using the related object pk.
            if data is None:
                setattr(instance, self.name, data)

    def _check_unique(self, **kwargs):
        # Override ForeignKey since check isn't applicable here.
        return []
```
### 108 - django/db/models/fields/related.py:

Start line: 600, End line: 617

```python
class ForeignObject(RelatedField):

    def resolve_related_fields(self):
        if not self.from_fields or len(self.from_fields) != len(self.to_fields):
            raise ValueError('Foreign Object from and to fields must be the same non-zero length')
        if isinstance(self.remote_field.model, str):
            raise ValueError('Related model %r cannot be resolved' % self.remote_field.model)
        related_fields = []
        for index in range(len(self.from_fields)):
            from_field_name = self.from_fields[index]
            to_field_name = self.to_fields[index]
            from_field = (
                self
                if from_field_name == RECURSIVE_RELATIONSHIP_CONSTANT
                else self.opts.get_field(from_field_name)
            )
            to_field = (self.remote_field.model._meta.pk if to_field_name is None
                        else self.remote_field.model._meta.get_field(to_field_name))
            related_fields.append((from_field, to_field))
        return related_fields
```
### 109 - django/db/models/fields/related.py:

Start line: 284, End line: 318

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
### 110 - django/db/models/fields/related.py:

Start line: 1595, End line: 1632

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
            self.remote_field.related_name = "_%s_%s_+" % (cls.__name__.lower(), name)

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
### 114 - django/db/models/fields/related.py:

Start line: 924, End line: 937

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
### 161 - django/db/models/fields/related.py:

Start line: 986, End line: 1013

```python
class ForeignKey(ForeignObject):

    def db_check(self, connection):
        return []

    def db_type(self, connection):
        return self.target_field.rel_db_type(connection=connection)

    def db_parameters(self, connection):
        return {"type": self.db_type(connection), "check": self.db_check(connection)}

    def convert_empty_strings(self, value, expression, connection):
        if (not value) and isinstance(value, str):
            return None
        return value

    def get_db_converters(self, connection):
        converters = super().get_db_converters(connection)
        if connection.features.interprets_empty_strings_as_nulls:
            converters += [self.convert_empty_strings]
        return converters

    def get_col(self, alias, output_field=None):
        if output_field is None:
            output_field = self.target_field
            while isinstance(output_field, ForeignKey):
                output_field = output_field.target_field
                if output_field is self:
                    raise ValueError('Cannot resolve output_field.')
        return super().get_col(alias, output_field)
```
### 164 - django/db/models/fields/related.py:

Start line: 1652, End line: 1686

```python
class ManyToManyField(RelatedField):

    def set_attributes_from_rel(self):
        pass

    def value_from_object(self, obj):
        return [] if obj.pk is None else list(getattr(obj, self.attname).all())

    def save_form_data(self, instance, data):
        getattr(instance, self.attname).set(data)

    def formfield(self, *, using=None, **kwargs):
        defaults = {
            'form_class': forms.ModelMultipleChoiceField,
            'queryset': self.remote_field.model._default_manager.using(using),
            **kwargs,
        }
        # If initial is passed in, it's a list of related objects, but the
        # MultipleChoiceField takes a list of IDs.
        if defaults.get('initial') is not None:
            initial = defaults['initial']
            if callable(initial):
                initial = initial()
            defaults['initial'] = [i.pk for i in initial]
        return super().formfield(**defaults)

    def db_check(self, connection):
        return None

    def db_type(self, connection):
        # A ManyToManyField is not represented by a single column,
        # so return None.
        return None

    def db_parameters(self, connection):
        return {"type": None, "check": None}
```
### 167 - django/db/models/fields/related.py:

Start line: 1066, End line: 1110

```python
def create_many_to_many_intermediary_model(field, klass):
    from django.db import models

    def set_managed(model, related, through):
        through._meta.managed = model._meta.managed or related._meta.managed

    to_model = resolve_relation(klass, field.remote_field.model)
    name = '%s_%s' % (klass._meta.object_name, field.name)
    lazy_related_operation(set_managed, klass, to_model, name)

    to = make_model_tuple(to_model)[1]
    from_ = klass._meta.model_name
    if to == from_:
        to = 'to_%s' % to
        from_ = 'from_%s' % from_

    meta = type('Meta', (), {
        'db_table': field._get_m2m_db_table(klass._meta),
        'auto_created': klass,
        'app_label': klass._meta.app_label,
        'db_tablespace': klass._meta.db_tablespace,
        'unique_together': (from_, to),
        'verbose_name': _('%(from)s-%(to)s relationship') % {'from': from_, 'to': to},
        'verbose_name_plural': _('%(from)s-%(to)s relationships') % {'from': from_, 'to': to},
        'apps': field.model._meta.apps,
    })
    # Construct and return the new class.
    return type(name, (models.Model,), {
        'Meta': meta,
        '__module__': klass.__module__,
        from_: models.ForeignKey(
            klass,
            related_name='%s+' % name,
            db_tablespace=field.db_tablespace,
            db_constraint=field.remote_field.db_constraint,
            on_delete=CASCADE,
        ),
        to: models.ForeignKey(
            to_model,
            related_name='%s+' % name,
            db_tablespace=field.db_tablespace,
            db_constraint=field.remote_field.db_constraint,
            on_delete=CASCADE,
        )
    })
```
### 209 - django/db/models/fields/related.py:

Start line: 1634, End line: 1650

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
