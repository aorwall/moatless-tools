# django__django-16281

| **django/django** | `2848e5d0ce5cf3c31fe87525536093b21d570f69` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 36139 |
| **Any found context length** | 2198 |
| **Avg pos** | 287.0 |
| **Min pos** | 4 |
| **Max pos** | 100 |
| **Top file pos** | 4 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py
--- a/django/db/backends/sqlite3/schema.py
+++ b/django/db/backends/sqlite3/schema.py
@@ -174,7 +174,7 @@ def alter_field(self, model, old_field, new_field, strict=False):
             super().alter_field(model, old_field, new_field, strict=strict)
 
     def _remake_table(
-        self, model, create_field=None, delete_field=None, alter_field=None
+        self, model, create_field=None, delete_field=None, alter_fields=None
     ):
         """
         Shortcut to transform a model from old_model into new_model
@@ -213,15 +213,16 @@ def is_self_referential(f):
         # If any of the new or altered fields is introducing a new PK,
         # remove the old one
         restore_pk_field = None
-        if getattr(create_field, "primary_key", False) or (
-            alter_field and getattr(alter_field[1], "primary_key", False)
+        alter_fields = alter_fields or []
+        if getattr(create_field, "primary_key", False) or any(
+            getattr(new_field, "primary_key", False) for _, new_field in alter_fields
         ):
             for name, field in list(body.items()):
-                if field.primary_key and not (
+                if field.primary_key and not any(
                     # Do not remove the old primary key when an altered field
                     # that introduces a primary key is the same field.
-                    alter_field
-                    and name == alter_field[1].name
+                    name == new_field.name
+                    for _, new_field in alter_fields
                 ):
                     field.primary_key = False
                     restore_pk_field = field
@@ -237,7 +238,7 @@ def is_self_referential(f):
                     self.effective_default(create_field),
                 )
         # Add in any altered fields
-        if alter_field:
+        for alter_field in alter_fields:
             old_field, new_field = alter_field
             body.pop(old_field.name, None)
             mapping.pop(old_field.column, None)
@@ -457,7 +458,7 @@ def _alter_field(
                 )
             )
         # Alter by remaking table
-        self._remake_table(model, alter_field=(old_field, new_field))
+        self._remake_table(model, alter_fields=[(old_field, new_field)])
         # Rebuild tables with FKs pointing to this field.
         old_collation = old_db_params.get("collation")
         new_collation = new_db_params.get("collation")
@@ -495,18 +496,30 @@ def _alter_many_to_many(self, model, old_field, new_field, strict):
             # propagate this altering.
             self._remake_table(
                 old_field.remote_field.through,
-                alter_field=(
-                    # The field that points to the target model is needed, so
-                    # we can tell alter_field to change it - this is
-                    # m2m_reverse_field_name() (as opposed to m2m_field_name(),
-                    # which points to our model).
-                    old_field.remote_field.through._meta.get_field(
-                        old_field.m2m_reverse_field_name()
+                alter_fields=[
+                    (
+                        # The field that points to the target model is needed,
+                        # so that table can be remade with the new m2m field -
+                        # this is m2m_reverse_field_name().
+                        old_field.remote_field.through._meta.get_field(
+                            old_field.m2m_reverse_field_name()
+                        ),
+                        new_field.remote_field.through._meta.get_field(
+                            new_field.m2m_reverse_field_name()
+                        ),
                     ),
-                    new_field.remote_field.through._meta.get_field(
-                        new_field.m2m_reverse_field_name()
+                    (
+                        # The field that points to the model itself is needed,
+                        # so that table can be remade with the new self field -
+                        # this is m2m_field_name().
+                        old_field.remote_field.through._meta.get_field(
+                            old_field.m2m_field_name()
+                        ),
+                        new_field.remote_field.through._meta.get_field(
+                            new_field.m2m_field_name()
+                        ),
                     ),
-                ),
+                ],
             )
             return
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/sqlite3/schema.py | 177 | 177 | 61 | 4 | 22633
| django/db/backends/sqlite3/schema.py | 216 | 224 | 61 | 4 | 22633
| django/db/backends/sqlite3/schema.py | 240 | 240 | 61 | 4 | 22633
| django/db/backends/sqlite3/schema.py | 460 | 460 | 100 | 4 | 36139
| django/db/backends/sqlite3/schema.py | 498 | 509 | 4 | 4 | 2198


## Problem Statement

```
Migration changing ManyToManyField target to 'self' doesn't work correctly
Description
	
Steps to reproduce:
Create Models:
class Bar(models.Model):
	pass
class Foo(models.Model):
	bar = models.ManyToManyField('Bar', blank=True)
Migrate:
./manage.py makemigrations app
./manage.py migrate
Change type of the ManyToManyField to Foo:
class Bar(models.Model):
	pass
class Foo(models.Model):
	bar = models.ManyToManyField('Foo', blank=True)
Migrate (see above)
In the admin page, navigate to "add Foo", click save
You should see an OperationalError, "no such column: app_foo_bar.from_foo_id"

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/migrations/operations/models.py | 370 | 424| 501 | 501 | 7713 | 
| 2 | 2 django/db/models/fields/related.py | 1450 | 1579| 984 | 1485 | 22320 | 
| 3 | 3 django/db/backends/base/schema.py | 1249 | 1279| 318 | 1803 | 36184 | 
| **-> 4 <-** | **4 django/db/backends/sqlite3/schema.py** | 488 | 538| 395 | 2198 | 40805 | 
| 5 | 4 django/db/models/fields/related.py | 1681 | 1731| 431 | 2629 | 40805 | 
| 6 | 4 django/db/models/fields/related.py | 1581 | 1679| 655 | 3284 | 40805 | 
| 7 | 5 django/db/migrations/operations/fields.py | 227 | 237| 146 | 3430 | 43317 | 
| 8 | 5 django/db/models/fields/related.py | 1882 | 1929| 505 | 3935 | 43317 | 
| 9 | 5 django/db/models/fields/related.py | 1931 | 1958| 305 | 4240 | 43317 | 
| 10 | 5 django/db/migrations/operations/models.py | 426 | 439| 127 | 4367 | 43317 | 
| 11 | 6 django/db/migrations/autodetector.py | 1432 | 1477| 318 | 4685 | 56778 | 
| 12 | 6 django/db/models/fields/related.py | 1960 | 1994| 266 | 4951 | 56778 | 
| 13 | 7 django/core/management/commands/migrate.py | 96 | 189| 765 | 5716 | 60704 | 
| 14 | 7 django/db/migrations/operations/models.py | 644 | 668| 231 | 5947 | 60704 | 
| 15 | 7 django/db/models/fields/related.py | 1733 | 1771| 413 | 6360 | 60704 | 
| 16 | 7 django/db/migrations/operations/fields.py | 101 | 113| 130 | 6490 | 60704 | 
| 17 | 8 django/db/models/fields/related_descriptors.py | 1457 | 1507| 407 | 6897 | 72362 | 
| 18 | 8 django/core/management/commands/migrate.py | 270 | 368| 813 | 7710 | 72362 | 
| 19 | 9 django/db/migrations/state.py | 291 | 345| 476 | 8186 | 80530 | 
| 20 | 10 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 112 | 8298 | 80642 | 
| 21 | 10 django/db/models/fields/related_descriptors.py | 1386 | 1455| 526 | 8824 | 80642 | 
| 22 | 10 django/db/migrations/operations/models.py | 113 | 134| 164 | 8988 | 80642 | 
| 23 | 10 django/db/migrations/operations/models.py | 670 | 686| 159 | 9147 | 80642 | 
| 24 | 10 django/db/migrations/operations/models.py | 441 | 478| 267 | 9414 | 80642 | 
| 25 | 10 django/db/migrations/autodetector.py | 1101 | 1218| 982 | 10396 | 80642 | 
| 26 | 10 django/db/migrations/operations/models.py | 559 | 568| 129 | 10525 | 80642 | 
| 27 | 10 django/db/models/fields/related.py | 1262 | 1316| 421 | 10946 | 80642 | 
| 28 | 10 django/db/migrations/operations/fields.py | 115 | 127| 131 | 11077 | 80642 | 
| 29 | 10 django/db/models/fields/related.py | 901 | 990| 583 | 11660 | 80642 | 
| 30 | 10 django/db/migrations/operations/fields.py | 239 | 267| 206 | 11866 | 80642 | 
| 31 | 11 django/core/management/commands/sqlmigrate.py | 40 | 84| 395 | 12261 | 81308 | 
| 32 | 12 django/contrib/admin/migrations/0001_initial.py | 1 | 77| 363 | 12624 | 81671 | 
| 33 | 12 django/db/migrations/autodetector.py | 1479 | 1502| 161 | 12785 | 81671 | 
| 34 | 12 django/db/models/fields/related_descriptors.py | 1133 | 1151| 174 | 12959 | 81671 | 
| 35 | 13 django/core/management/commands/makemigrations.py | 104 | 194| 791 | 13750 | 85619 | 
| 36 | 13 django/core/management/commands/makemigrations.py | 196 | 259| 458 | 14208 | 85619 | 
| 37 | 13 django/db/models/fields/related_descriptors.py | 1238 | 1300| 544 | 14752 | 85619 | 
| 38 | 13 django/core/management/commands/migrate.py | 369 | 390| 204 | 14956 | 85619 | 
| 39 | 13 django/db/migrations/operations/fields.py | 270 | 337| 521 | 15477 | 85619 | 
| 40 | 13 django/core/management/commands/makemigrations.py | 261 | 331| 572 | 16049 | 85619 | 
| 41 | 13 django/db/models/fields/related.py | 1177 | 1209| 246 | 16295 | 85619 | 
| 42 | 13 django/db/migrations/state.py | 437 | 458| 204 | 16499 | 85619 | 
| 43 | 14 django/db/models/base.py | 1653 | 1687| 252 | 16751 | 104193 | 
| 44 | 14 django/db/migrations/operations/models.py | 570 | 594| 213 | 16964 | 104193 | 
| 45 | 14 django/db/models/fields/related_descriptors.py | 1206 | 1236| 260 | 17224 | 104193 | 
| 46 | 14 django/db/models/base.py | 1069 | 1121| 520 | 17744 | 104193 | 
| 47 | 15 django/contrib/contenttypes/migrations/0002_remove_content_type_name.py | 1 | 44| 232 | 17976 | 104425 | 
| 48 | 15 django/db/migrations/state.py | 142 | 179| 395 | 18371 | 104425 | 
| 49 | 15 django/db/models/fields/related_descriptors.py | 1302 | 1332| 281 | 18652 | 104425 | 
| 50 | 15 django/db/models/fields/related_descriptors.py | 1334 | 1351| 154 | 18806 | 104425 | 
| 51 | 15 django/db/migrations/operations/models.py | 136 | 306| 968 | 19774 | 104425 | 
| 52 | 15 django/db/migrations/operations/models.py | 1 | 18| 137 | 19911 | 104425 | 
| 53 | 15 django/db/migrations/operations/models.py | 933 | 968| 319 | 20230 | 104425 | 
| 54 | 15 django/db/models/fields/related.py | 1103 | 1120| 133 | 20363 | 104425 | 
| 55 | 15 django/core/management/commands/migrate.py | 191 | 269| 678 | 21041 | 104425 | 
| 56 | 15 django/db/migrations/operations/models.py | 21 | 38| 116 | 21157 | 104425 | 
| 57 | 16 django/core/management/commands/showmigrations.py | 56 | 77| 158 | 21315 | 105721 | 
| 58 | 17 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 59 | 77| 113 | 21428 | 106286 | 
| 59 | 17 django/db/models/fields/related_descriptors.py | 1153 | 1171| 155 | 21583 | 106286 | 
| 60 | 17 django/db/models/fields/related_descriptors.py | 1173 | 1204| 228 | 21811 | 106286 | 
| **-> 61 <-** | **17 django/db/backends/sqlite3/schema.py** | 176 | 264| 822 | 22633 | 106286 | 
| 62 | 17 django/db/migrations/autodetector.py | 1629 | 1643| 135 | 22768 | 106286 | 
| 63 | 17 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 23576 | 106286 | 
| 64 | 17 django/db/models/fields/related.py | 208 | 224| 142 | 23718 | 106286 | 
| 65 | 17 django/db/models/fields/related.py | 604 | 670| 497 | 24215 | 106286 | 
| 66 | 17 django/db/models/fields/related.py | 1160 | 1175| 136 | 24351 | 106286 | 
| 67 | 18 django/contrib/redirects/migrations/0002_alter_redirect_new_path_help_text.py | 1 | 25| 117 | 24468 | 106403 | 
| 68 | 18 django/db/migrations/operations/models.py | 344 | 368| 157 | 24625 | 106403 | 
| 69 | 18 django/db/models/base.py | 1123 | 1150| 238 | 24863 | 106403 | 
| 70 | 18 django/db/models/fields/related_descriptors.py | 751 | 786| 268 | 25131 | 106403 | 
| 71 | 19 django/contrib/redirects/migrations/0001_initial.py | 1 | 66| 309 | 25440 | 106712 | 
| 72 | 19 django/core/management/commands/makemigrations.py | 26 | 102| 446 | 25886 | 106712 | 
| 73 | 19 django/db/models/fields/related_descriptors.py | 895 | 928| 279 | 26165 | 106712 | 
| 74 | 19 django/contrib/auth/migrations/0011_update_proxy_permissions.py | 1 | 56| 452 | 26617 | 106712 | 
| 75 | 19 django/db/models/fields/related.py | 672 | 705| 335 | 26952 | 106712 | 
| 76 | 19 django/db/migrations/autodetector.py | 1595 | 1627| 258 | 27210 | 106712 | 
| 77 | 19 django/db/models/base.py | 1812 | 1847| 246 | 27456 | 106712 | 
| 78 | 19 django/db/models/base.py | 1998 | 2051| 359 | 27815 | 106712 | 
| 79 | 19 django/db/models/fields/related.py | 1018 | 1054| 261 | 28076 | 106712 | 
| 80 | 19 django/db/migrations/autodetector.py | 1534 | 1553| 187 | 28263 | 106712 | 
| 81 | 19 django/db/models/fields/related_descriptors.py | 978 | 1044| 592 | 28855 | 106712 | 
| 82 | 20 django/contrib/auth/migrations/0001_initial.py | 1 | 206| 1007 | 29862 | 107719 | 
| 83 | 20 django/db/migrations/operations/models.py | 970 | 987| 149 | 30011 | 107719 | 
| 84 | 20 django/db/models/fields/related.py | 1319 | 1416| 574 | 30585 | 107719 | 
| 85 | 20 django/core/management/commands/migrate.py | 1 | 14| 134 | 30719 | 107719 | 
| 86 | 20 django/db/migrations/operations/models.py | 481 | 529| 409 | 31128 | 107719 | 
| 87 | 21 django/db/migrations/questioner.py | 291 | 342| 367 | 31495 | 110415 | 
| 88 | 21 django/db/models/fields/related.py | 582 | 602| 138 | 31633 | 110415 | 
| 89 | 21 django/db/models/fields/related_descriptors.py | 1089 | 1131| 381 | 32014 | 110415 | 
| 90 | 21 django/db/models/fields/related.py | 871 | 898| 237 | 32251 | 110415 | 
| 91 | 21 django/db/migrations/autodetector.py | 1504 | 1532| 235 | 32486 | 110415 | 
| 92 | 21 django/db/migrations/state.py | 265 | 289| 238 | 32724 | 110415 | 
| 93 | 22 django/contrib/contenttypes/migrations/0001_initial.py | 1 | 47| 225 | 32949 | 110640 | 
| 94 | 22 django/db/migrations/autodetector.py | 908 | 981| 623 | 33572 | 110640 | 
| 95 | 23 django/db/migrations/exceptions.py | 1 | 61| 249 | 33821 | 110890 | 
| 96 | 23 django/db/models/base.py | 1526 | 1561| 277 | 34098 | 110890 | 
| 97 | 23 django/db/migrations/autodetector.py | 600 | 776| 1231 | 35329 | 110890 | 
| 98 | 23 django/db/migrations/state.py | 460 | 486| 150 | 35479 | 110890 | 
| 99 | 23 django/db/models/fields/related.py | 1418 | 1448| 172 | 35651 | 110890 | 
| **-> 100 <-** | **23 django/db/backends/sqlite3/schema.py** | 429 | 486| 488 | 36139 | 110890 | 
| 101 | 23 django/db/backends/base/schema.py | 523 | 542| 196 | 36335 | 110890 | 
| 102 | 24 django/core/management/commands/squashmigrations.py | 162 | 254| 766 | 37101 | 112930 | 
| 103 | 25 django/db/migrations/operations/special.py | 182 | 209| 254 | 37355 | 114503 | 
| 104 | 25 django/db/backends/base/schema.py | 39 | 72| 214 | 37569 | 114503 | 
| 105 | 25 django/db/models/fields/related.py | 302 | 339| 296 | 37865 | 114503 | 
| 106 | 26 django/contrib/flatpages/migrations/0001_initial.py | 1 | 70| 355 | 38220 | 114858 | 
| 107 | 27 django/db/utils.py | 237 | 279| 322 | 38542 | 116757 | 
| 108 | 27 django/db/models/base.py | 2160 | 2241| 592 | 39134 | 116757 | 
| 109 | 27 django/db/models/fields/related.py | 707 | 731| 210 | 39344 | 116757 | 
| 110 | 27 django/core/management/commands/squashmigrations.py | 256 | 269| 112 | 39456 | 116757 | 
| 111 | 27 django/db/migrations/autodetector.py | 1346 | 1367| 197 | 39653 | 116757 | 
| 112 | 27 django/db/models/base.py | 459 | 572| 957 | 40610 | 116757 | 
| 113 | 27 django/db/models/base.py | 1595 | 1620| 188 | 40798 | 116757 | 
| 114 | 27 django/db/migrations/autodetector.py | 811 | 906| 712 | 41510 | 116757 | 
| 115 | 27 django/db/migrations/state.py | 240 | 263| 247 | 41757 | 116757 | 
| 116 | 28 django/db/migrations/operations/__init__.py | 1 | 43| 227 | 41984 | 116984 | 
| 117 | **28 django/db/backends/sqlite3/schema.py** | 123 | 174| 527 | 42511 | 116984 | 
| 118 | 28 django/db/migrations/autodetector.py | 484 | 514| 267 | 42778 | 116984 | 
| 119 | 28 django/db/models/fields/related_descriptors.py | 730 | 749| 231 | 43009 | 116984 | 
| 120 | 29 django/db/migrations/recorder.py | 1 | 22| 148 | 43157 | 117671 | 
| 121 | 29 django/db/models/fields/related_descriptors.py | 632 | 663| 243 | 43400 | 117671 | 
| 122 | 29 django/db/migrations/operations/fields.py | 339 | 358| 153 | 43553 | 117671 | 
| 123 | 30 django/contrib/admin/options.py | 1748 | 1850| 780 | 44333 | 136910 | 
| 124 | 30 django/core/management/commands/makemigrations.py | 1 | 23| 185 | 44518 | 136910 | 
| 125 | 30 django/db/migrations/operations/models.py | 689 | 741| 320 | 44838 | 136910 | 
| 126 | 31 django/db/backends/oracle/schema.py | 105 | 170| 740 | 45578 | 139252 | 
| 127 | 31 django/db/migrations/operations/models.py | 989 | 1024| 249 | 45827 | 139252 | 
| 128 | **31 django/db/backends/sqlite3/schema.py** | 100 | 121| 195 | 46022 | 139252 | 
| 129 | 32 django/contrib/contenttypes/management/__init__.py | 1 | 43| 358 | 46380 | 140240 | 
| 130 | 32 django/core/management/commands/migrate.py | 17 | 94| 487 | 46867 | 140240 | 
| 131 | 32 django/core/management/commands/migrate.py | 392 | 430| 361 | 47228 | 140240 | 
| 132 | 32 django/db/migrations/operations/fields.py | 154 | 195| 348 | 47576 | 140240 | 
| 133 | 32 django/db/models/base.py | 1788 | 1810| 175 | 47751 | 140240 | 
| 134 | 32 django/db/migrations/autodetector.py | 516 | 581| 482 | 48233 | 140240 | 
| 135 | 32 django/core/management/commands/sqlmigrate.py | 1 | 38| 276 | 48509 | 140240 | 
| 136 | 32 django/db/models/base.py | 1763 | 1786| 180 | 48689 | 140240 | 
| 137 | 32 django/core/management/commands/squashmigrations.py | 62 | 160| 809 | 49498 | 140240 | 
| 138 | 32 django/db/migrations/autodetector.py | 1027 | 1076| 384 | 49882 | 140240 | 
| 139 | 32 django/db/models/fields/related_descriptors.py | 1046 | 1064| 199 | 50081 | 140240 | 
| 140 | 33 django/db/models/options.py | 369 | 387| 124 | 50205 | 147923 | 
| 141 | 33 django/db/migrations/questioner.py | 57 | 87| 255 | 50460 | 147923 | 
| 142 | 33 django/db/models/fields/related_descriptors.py | 705 | 728| 227 | 50687 | 147923 | 
| 143 | 33 django/db/migrations/autodetector.py | 1220 | 1307| 719 | 51406 | 147923 | 
| 144 | 33 django/db/migrations/operations/fields.py | 129 | 151| 133 | 51539 | 147923 | 
| 145 | 33 django/db/migrations/operations/fields.py | 45 | 72| 192 | 51731 | 147923 | 
| 146 | 34 django/contrib/sites/migrations/0001_initial.py | 1 | 45| 210 | 51941 | 148133 | 
| 147 | 34 django/db/migrations/operations/models.py | 1068 | 1108| 337 | 52278 | 148133 | 
| 148 | 35 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 28| 143 | 52421 | 148276 | 
| 149 | 35 django/db/migrations/state.py | 181 | 238| 598 | 53019 | 148276 | 
| 150 | 35 django/core/management/commands/migrate.py | 432 | 487| 409 | 53428 | 148276 | 
| 151 | 35 django/db/models/fields/related.py | 514 | 580| 367 | 53795 | 148276 | 
| 152 | 36 django/db/migrations/migration.py | 200 | 240| 292 | 54087 | 150183 | 
| 153 | 37 django/core/serializers/xml_serializer.py | 127 | 179| 404 | 54491 | 153817 | 
| 154 | 37 django/db/backends/oracle/schema.py | 172 | 181| 142 | 54633 | 153817 | 
| 155 | 37 django/db/backends/base/schema.py | 599 | 627| 245 | 54878 | 153817 | 
| 156 | 38 django/db/backends/mysql/operations.py | 436 | 465| 274 | 55152 | 157995 | 
| 157 | 38 django/db/migrations/operations/models.py | 597 | 618| 148 | 55300 | 157995 | 
| 158 | 38 django/db/models/base.py | 574 | 612| 326 | 55626 | 157995 | 
| 159 | 38 django/db/migrations/autodetector.py | 280 | 378| 806 | 56432 | 157995 | 
| 160 | 38 django/db/models/fields/related_descriptors.py | 1353 | 1384| 348 | 56780 | 157995 | 
| 161 | 38 django/db/migrations/autodetector.py | 1078 | 1099| 188 | 56968 | 157995 | 
| 162 | 39 django/contrib/contenttypes/fields.py | 25 | 114| 560 | 57528 | 163826 | 
| 163 | 39 django/db/backends/base/schema.py | 799 | 889| 773 | 58301 | 163826 | 
| 164 | 39 django/db/migrations/autodetector.py | 583 | 599| 186 | 58487 | 163826 | 
| 165 | 39 django/db/migrations/operations/models.py | 41 | 111| 524 | 59011 | 163826 | 
| 166 | 40 django/db/backends/sqlite3/operations.py | 415 | 435| 148 | 59159 | 167298 | 
| 167 | 40 django/db/models/fields/related.py | 1079 | 1101| 180 | 59339 | 167298 | 
| 168 | 40 django/db/models/fields/related.py | 992 | 1016| 176 | 59515 | 167298 | 
| 169 | 40 django/db/migrations/operations/models.py | 744 | 778| 225 | 59740 | 167298 | 
| 170 | 40 django/db/migrations/operations/fields.py | 1 | 43| 238 | 59978 | 167298 | 
| 171 | 40 django/db/migrations/autodetector.py | 1397 | 1430| 289 | 60267 | 167298 | 
| 172 | 40 django/db/models/fields/related.py | 1122 | 1158| 284 | 60551 | 167298 | 
| 173 | 40 django/db/migrations/autodetector.py | 233 | 264| 256 | 60807 | 167298 | 
| 174 | 40 django/db/migrations/state.py | 973 | 989| 140 | 60947 | 167298 | 
| 175 | 40 django/db/models/fields/related_descriptors.py | 406 | 427| 158 | 61105 | 167298 | 
| 176 | 41 django/db/backends/postgresql/schema.py | 140 | 246| 866 | 61971 | 170039 | 
| 177 | 41 django/db/backends/oracle/schema.py | 75 | 103| 344 | 62315 | 170039 | 
| 178 | 41 django/db/migrations/autodetector.py | 983 | 1025| 297 | 62612 | 170039 | 
| 179 | 41 django/db/migrations/operations/models.py | 621 | 642| 162 | 62774 | 170039 | 
| 180 | 41 django/db/models/base.py | 250 | 367| 874 | 63648 | 170039 | 
| 181 | 41 django/db/models/fields/related.py | 1804 | 1829| 222 | 63870 | 170039 | 
| 182 | 41 django/db/models/base.py | 1708 | 1761| 494 | 64364 | 170039 | 
| 183 | 42 django/contrib/auth/admin.py | 28 | 40| 130 | 64494 | 171810 | 
| 184 | 42 django/db/models/fields/related.py | 187 | 206| 155 | 64649 | 171810 | 
| 185 | 42 django/db/models/base.py | 1381 | 1396| 142 | 64791 | 171810 | 
| 186 | 42 django/db/models/fields/related.py | 341 | 378| 294 | 65085 | 171810 | 
| 187 | **42 django/db/backends/sqlite3/schema.py** | 265 | 360| 807 | 65892 | 171810 | 
| 188 | 42 django/db/models/fields/related_descriptors.py | 665 | 703| 354 | 66246 | 171810 | 
| 189 | 42 django/db/backends/base/schema.py | 544 | 563| 199 | 66445 | 171810 | 
| 190 | 43 django/db/models/sql/datastructures.py | 130 | 148| 141 | 66586 | 173320 | 


### Hint

```
I believe it works correctly as long as the new target model isn't 'self'.
Can I check out what you want? You can use 'self' instead of 'Foo' like this : class Foo(models.Model): bar = models.ManyToManyField('self', blank=True) You meant that we should use 'Foo' rather than 'self', right?
Replying to SShayashi: Can I check out what you want? You can use 'self' instead of 'Foo' like this : class Foo(models.Model): bar = models.ManyToManyField('self', blank=True) You meant that we should use 'Foo' rather than 'self', right? Exactly, though i wasn't aware that self would work in that context. I think i like naming the type directly more, since self could be easily confused with the self argument of class methods.
While altering a many to many field , a new table is created (which is later renamed) and data is copied from old table to new table and then old table is deleted. This issue caused while making this new table ​here, we are only altering the m2m_reverse_fields ignoring the m2m_fields that points to our model. But both m2m_reverse_fieldsand m2m_fields needs to be changed. Hope i'm proceeding into the right direction.
It is also failing for the case if process is reversed, like when we migrate passing self/Foo to the m2m field and then change it to Bar.(due to the same reason that we are not altering m2m_fields) I tried to alter self pointing field and with the changes below its working as expected for the above issue and passing all the tests as well. django/db/backends/sqlite3/schema.py diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py index 88fa466f79..970820827e 100644 a b class DatabaseSchemaEditor(BaseDatabaseSchemaEditor): 174174 super().alter_field(model, old_field, new_field, strict=strict) 175175 176176 def _remake_table( 177 self, model, create_field=None, delete_field=None, alter_field=None 177 self, model, create_field=None, delete_field=None, alter_field=None, alter_self_field=None 178178 ): 179179 """ 180180 Shortcut to transform a model from old_model into new_model … … class DatabaseSchemaEditor(BaseDatabaseSchemaEditor): 236236 mapping[create_field.column] = self.prepare_default( 237237 self.effective_default(create_field), 238238 ) 239 240 # Alter field pointing to the model itself. 241 if alter_self_field: 242 old_self_field, new_self_field = alter_self_field 243 body.pop(old_self_field.name, None) 244 mapping.pop(old_self_field.column, None) 245 body[new_self_field.name] = new_self_field 246 mapping[new_self_field.column] = self.quote_name(old_self_field.column) 247 rename_mapping[old_self_field.name] = new_self_field.name 248 239249 # Add in any altered fields 240250 if alter_field: 241251 old_field, new_field = alter_field … … class DatabaseSchemaEditor(BaseDatabaseSchemaEditor): 507517 new_field.m2m_reverse_field_name() 508518 ), 509519 ), 520 alter_self_field=( 521 old_field.remote_field.through._meta.get_field( 522 old_field.m2m_field_name() 523 ), 524 new_field.remote_field.through._meta.get_field( 525 new_field.m2m_field_name() 526 ), 527 ), 510528 ) 511529 return
Hi Bhuvnesh I tried to alter self pointing field and with the changes below its working as expected for the above issue and passing all the tests as well. If the tests are all passing it's worth opening a PR to make review easier. (Make sure to add a new test for the issue here too.) Thanks.
```

## Patch

```diff
diff --git a/django/db/backends/sqlite3/schema.py b/django/db/backends/sqlite3/schema.py
--- a/django/db/backends/sqlite3/schema.py
+++ b/django/db/backends/sqlite3/schema.py
@@ -174,7 +174,7 @@ def alter_field(self, model, old_field, new_field, strict=False):
             super().alter_field(model, old_field, new_field, strict=strict)
 
     def _remake_table(
-        self, model, create_field=None, delete_field=None, alter_field=None
+        self, model, create_field=None, delete_field=None, alter_fields=None
     ):
         """
         Shortcut to transform a model from old_model into new_model
@@ -213,15 +213,16 @@ def is_self_referential(f):
         # If any of the new or altered fields is introducing a new PK,
         # remove the old one
         restore_pk_field = None
-        if getattr(create_field, "primary_key", False) or (
-            alter_field and getattr(alter_field[1], "primary_key", False)
+        alter_fields = alter_fields or []
+        if getattr(create_field, "primary_key", False) or any(
+            getattr(new_field, "primary_key", False) for _, new_field in alter_fields
         ):
             for name, field in list(body.items()):
-                if field.primary_key and not (
+                if field.primary_key and not any(
                     # Do not remove the old primary key when an altered field
                     # that introduces a primary key is the same field.
-                    alter_field
-                    and name == alter_field[1].name
+                    name == new_field.name
+                    for _, new_field in alter_fields
                 ):
                     field.primary_key = False
                     restore_pk_field = field
@@ -237,7 +238,7 @@ def is_self_referential(f):
                     self.effective_default(create_field),
                 )
         # Add in any altered fields
-        if alter_field:
+        for alter_field in alter_fields:
             old_field, new_field = alter_field
             body.pop(old_field.name, None)
             mapping.pop(old_field.column, None)
@@ -457,7 +458,7 @@ def _alter_field(
                 )
             )
         # Alter by remaking table
-        self._remake_table(model, alter_field=(old_field, new_field))
+        self._remake_table(model, alter_fields=[(old_field, new_field)])
         # Rebuild tables with FKs pointing to this field.
         old_collation = old_db_params.get("collation")
         new_collation = new_db_params.get("collation")
@@ -495,18 +496,30 @@ def _alter_many_to_many(self, model, old_field, new_field, strict):
             # propagate this altering.
             self._remake_table(
                 old_field.remote_field.through,
-                alter_field=(
-                    # The field that points to the target model is needed, so
-                    # we can tell alter_field to change it - this is
-                    # m2m_reverse_field_name() (as opposed to m2m_field_name(),
-                    # which points to our model).
-                    old_field.remote_field.through._meta.get_field(
-                        old_field.m2m_reverse_field_name()
+                alter_fields=[
+                    (
+                        # The field that points to the target model is needed,
+                        # so that table can be remade with the new m2m field -
+                        # this is m2m_reverse_field_name().
+                        old_field.remote_field.through._meta.get_field(
+                            old_field.m2m_reverse_field_name()
+                        ),
+                        new_field.remote_field.through._meta.get_field(
+                            new_field.m2m_reverse_field_name()
+                        ),
                     ),
-                    new_field.remote_field.through._meta.get_field(
-                        new_field.m2m_reverse_field_name()
+                    (
+                        # The field that points to the model itself is needed,
+                        # so that table can be remade with the new self field -
+                        # this is m2m_field_name().
+                        old_field.remote_field.through._meta.get_field(
+                            old_field.m2m_field_name()
+                        ),
+                        new_field.remote_field.through._meta.get_field(
+                            new_field.m2m_field_name()
+                        ),
                     ),
-                ),
+                ],
             )
             return
 

```

## Test Patch

```diff
diff --git a/tests/migrations/test_operations.py b/tests/migrations/test_operations.py
--- a/tests/migrations/test_operations.py
+++ b/tests/migrations/test_operations.py
@@ -1796,6 +1796,43 @@ def test_alter_model_table_m2m(self):
         self.assertTableExists(original_m2m_table)
         self.assertTableNotExists(new_m2m_table)
 
+    def test_alter_model_table_m2m_field(self):
+        app_label = "test_talm2mfl"
+        project_state = self.set_up_test_model(app_label, second_model=True)
+        # Add the M2M field.
+        project_state = self.apply_operations(
+            app_label,
+            project_state,
+            operations=[
+                migrations.AddField(
+                    "Pony",
+                    "stables",
+                    models.ManyToManyField("Stable"),
+                )
+            ],
+        )
+        m2m_table = f"{app_label}_pony_stables"
+        self.assertColumnExists(m2m_table, "pony_id")
+        self.assertColumnExists(m2m_table, "stable_id")
+        # Point the M2M field to self.
+        with_field_state = project_state.clone()
+        operations = [
+            migrations.AlterField(
+                model_name="Pony",
+                name="stables",
+                field=models.ManyToManyField("self"),
+            )
+        ]
+        project_state = self.apply_operations(
+            app_label, project_state, operations=operations
+        )
+        self.assertColumnExists(m2m_table, "from_pony_id")
+        self.assertColumnExists(m2m_table, "to_pony_id")
+        # Reversal.
+        self.unapply_operations(app_label, with_field_state, operations=operations)
+        self.assertColumnExists(m2m_table, "pony_id")
+        self.assertColumnExists(m2m_table, "stable_id")
+
     def test_alter_field(self):
         """
         Tests the AlterField operation.

```


## Code snippets

### 1 - django/db/migrations/operations/models.py:

Start line: 370, End line: 424

```python
class RenameModel(ModelOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        new_model = to_state.apps.get_model(app_label, self.new_name)
        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
            old_model = from_state.apps.get_model(app_label, self.old_name)
            # Move the main table
            schema_editor.alter_db_table(
                new_model,
                old_model._meta.db_table,
                new_model._meta.db_table,
            )
            # Alter the fields pointing to us
            for related_object in old_model._meta.related_objects:
                if related_object.related_model == old_model:
                    model = new_model
                    related_key = (app_label, self.new_name_lower)
                else:
                    model = related_object.related_model
                    related_key = (
                        related_object.related_model._meta.app_label,
                        related_object.related_model._meta.model_name,
                    )
                to_field = to_state.apps.get_model(*related_key)._meta.get_field(
                    related_object.field.name
                )
                schema_editor.alter_field(
                    model,
                    related_object.field,
                    to_field,
                )
            # Rename M2M fields whose name is based on this model's name.
            fields = zip(
                old_model._meta.local_many_to_many, new_model._meta.local_many_to_many
            )
            for (old_field, new_field) in fields:
                # Skip self-referential fields as these are renamed above.
                if (
                    new_field.model == new_field.related_model
                    or not new_field.remote_field.through._meta.auto_created
                ):
                    continue
                # Rename the M2M table that's based on this model's name.
                old_m2m_model = old_field.remote_field.through
                new_m2m_model = new_field.remote_field.through
                schema_editor.alter_db_table(
                    new_m2m_model,
                    old_m2m_model._meta.db_table,
                    new_m2m_model._meta.db_table,
                )
                # Rename the column in the M2M table that's based on this
                # model's name.
                schema_editor.alter_field(
                    new_m2m_model,
                    old_m2m_model._meta.get_field(old_model._meta.model_name),
                    new_m2m_model._meta.get_field(new_model._meta.model_name),
                )
```
### 2 - django/db/models/fields/related.py:

Start line: 1450, End line: 1579

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, "_meta"):
            qualified_model_name = "%s.%s" % (
                self.remote_field.through._meta.app_label,
                self.remote_field.through.__name__,
            )
        else:
            qualified_model_name = self.remote_field.through

        errors = []

        if self.remote_field.through not in self.opts.apps.get_models(
            include_auto_created=True
        ):
            # The relationship model is not installed.
            errors.append(
                checks.Error(
                    "Field specifies a many-to-many relation through model "
                    "'%s', which has not been installed." % qualified_model_name,
                    obj=self,
                    id="fields.E331",
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
                    from_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_self > 2 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it has more than two foreign keys "
                            "to '%s', which is ambiguous. You must specify "
                            "which two foreign keys Django should use via the "
                            "through_fields keyword argument."
                            % (self, from_model_name),
                            hint=(
                                "Use through_fields to specify which two foreign keys "
                                "Django should use."
                            ),
                            obj=self.remote_field.through,
                            id="fields.E333",
                        )
                    )

            else:
                # Count foreign keys in relationship model
                seen_from = sum(
                    from_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )
                seen_to = sum(
                    to_model == getattr(field.remote_field, "model", None)
                    for field in self.remote_field.through._meta.fields
                )

                if seen_from > 1 and not self.remote_field.through_fields:
                    errors.append(
                        checks.Error(
                            (
                                "The model is used as an intermediate model by "
                                "'%s', but it has more than one foreign key "
                                "from '%s', which is ambiguous. You must specify "
                                "which foreign key Django should use via the "
                                "through_fields keyword argument."
                            )
                            % (self, from_model_name),
                            hint=(
                                "If you want to create a recursive relationship, "
                                'use ManyToManyField("%s", through="%s").'
                            )
                            % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id="fields.E334",
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
                                "If you want to create a recursive relationship, "
                                'use ManyToManyField("%s", through="%s").'
                            )
                            % (
                                RECURSIVE_RELATIONSHIP_CONSTANT,
                                relationship_model_name,
                            ),
                            obj=self,
                            id="fields.E335",
                        )
                    )

                if seen_from == 0 or seen_to == 0:
                    errors.append(
                        checks.Error(
                            "The model is used as an intermediate model by "
                            "'%s', but it does not have a foreign key to '%s' or '%s'."
                            % (self, from_model_name, to_model_name),
                            obj=self.remote_field.through,
                            id="fields.E336",
                        )
                    )
        # ... other code
```
### 3 - django/db/backends/base/schema.py:

Start line: 1249, End line: 1279

```python
class BaseDatabaseSchemaEditor:

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        # Rename the through table
        if (
            old_field.remote_field.through._meta.db_table
            != new_field.remote_field.through._meta.db_table
        ):
            self.alter_db_table(
                old_field.remote_field.through,
                old_field.remote_field.through._meta.db_table,
                new_field.remote_field.through._meta.db_table,
            )
        # Repoint the FK to the other side
        self.alter_field(
            new_field.remote_field.through,
            # The field that points to the target model is needed, so we can
            # tell alter_field to change it - this is m2m_reverse_field_name()
            # (as opposed to m2m_field_name(), which points to our model).
            old_field.remote_field.through._meta.get_field(
                old_field.m2m_reverse_field_name()
            ),
            new_field.remote_field.through._meta.get_field(
                new_field.m2m_reverse_field_name()
            ),
        )
        self.alter_field(
            new_field.remote_field.through,
            # for self-referential models we need to alter field from the other end too
            old_field.remote_field.through._meta.get_field(old_field.m2m_field_name()),
            new_field.remote_field.through._meta.get_field(new_field.m2m_field_name()),
        )
```
### 4 - django/db/backends/sqlite3/schema.py:

Start line: 488, End line: 538

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_many_to_many(self, model, old_field, new_field, strict):
        """Alter M2Ms to repoint their to= endpoints."""
        if (
            old_field.remote_field.through._meta.db_table
            == new_field.remote_field.through._meta.db_table
        ):
            # The field name didn't change, but some options did, so we have to
            # propagate this altering.
            self._remake_table(
                old_field.remote_field.through,
                alter_field=(
                    # The field that points to the target model is needed, so
                    # we can tell alter_field to change it - this is
                    # m2m_reverse_field_name() (as opposed to m2m_field_name(),
                    # which points to our model).
                    old_field.remote_field.through._meta.get_field(
                        old_field.m2m_reverse_field_name()
                    ),
                    new_field.remote_field.through._meta.get_field(
                        new_field.m2m_reverse_field_name()
                    ),
                ),
            )
            return

        # Make a new through table
        self.create_model(new_field.remote_field.through)
        # Copy the data across
        self.execute(
            "INSERT INTO %s (%s) SELECT %s FROM %s"
            % (
                self.quote_name(new_field.remote_field.through._meta.db_table),
                ", ".join(
                    [
                        "id",
                        new_field.m2m_column_name(),
                        new_field.m2m_reverse_name(),
                    ]
                ),
                ", ".join(
                    [
                        "id",
                        old_field.m2m_column_name(),
                        old_field.m2m_reverse_name(),
                    ]
                ),
                self.quote_name(old_field.remote_field.through._meta.db_table),
            )
        )
        # Delete the old through table
        self.delete_model(old_field.remote_field.through)
```
### 5 - django/db/models/fields/related.py:

Start line: 1681, End line: 1731

```python
class ManyToManyField(RelatedField):

    def _check_table_uniqueness(self, **kwargs):
        if (
            isinstance(self.remote_field.through, str)
            or not self.remote_field.through._meta.managed
        ):
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
        if (
            model
            and model._meta.concrete_model
            != self.remote_field.through._meta.concrete_model
        ):
            if model._meta.auto_created:

                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name

                opts = model._meta.auto_created._meta
                clashing_obj = "%s.%s" % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = checks.Warning, "fields.W344"
                error_hint = (
                    "You have configured settings.DATABASE_ROUTERS. Verify "
                    "that the table of %r is correctly routed to a separate "
                    "database." % clashing_obj
                )
            else:
                error_class, error_id = checks.Error, "fields.E340"
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
### 6 - django/db/models/fields/related.py:

Start line: 1581, End line: 1679

```python
class ManyToManyField(RelatedField):

    def _check_relationship_model(self, from_model=None, **kwargs):

        # Validate `through_fields`.
        if self.remote_field.through_fields is not None:
            # Validate that we're given an iterable of at least two items
            # and that none of them is "falsy".
            if not (
                len(self.remote_field.through_fields) >= 2
                and self.remote_field.through_fields[0]
                and self.remote_field.through_fields[1]
            ):
                errors.append(
                    checks.Error(
                        "Field specifies 'through_fields' but does not provide "
                        "the names of the two link fields that should be used "
                        "for the relation through model '%s'." % qualified_model_name,
                        hint=(
                            "Make sure you specify 'through_fields' as "
                            "through_fields=('field1', 'field2')"
                        ),
                        obj=self,
                        id="fields.E337",
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

                source, through, target = (
                    from_model,
                    self.remote_field.through,
                    self.remote_field.model,
                )
                source_field_name, target_field_name = self.remote_field.through_fields[
                    :2
                ]

                for field_name, related_model in (
                    (source_field_name, source),
                    (target_field_name, target),
                ):

                    possible_field_names = []
                    for f in through._meta.fields:
                        if (
                            hasattr(f, "remote_field")
                            and getattr(f.remote_field, "model", None) == related_model
                        ):
                            possible_field_names.append(f.name)
                    if possible_field_names:
                        hint = (
                            "Did you mean one of the following foreign keys to '%s': "
                            "%s?"
                            % (
                                related_model._meta.object_name,
                                ", ".join(possible_field_names),
                            )
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
                                id="fields.E338",
                            )
                        )
                    else:
                        if not (
                            hasattr(field, "remote_field")
                            and getattr(field.remote_field, "model", None)
                            == related_model
                        ):
                            errors.append(
                                checks.Error(
                                    "'%s.%s' is not a foreign key to '%s'."
                                    % (
                                        through._meta.object_name,
                                        field_name,
                                        related_model._meta.object_name,
                                    ),
                                    hint=hint,
                                    obj=self,
                                    id="fields.E339",
                                )
                            )

        return errors
```
### 7 - django/db/migrations/operations/fields.py:

Start line: 227, End line: 237

```python
class AlterField(FieldOperation):

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        to_model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, to_model):
            from_model = from_state.apps.get_model(app_label, self.model_name)
            from_field = from_model._meta.get_field(self.name)
            to_field = to_model._meta.get_field(self.name)
            if not self.preserve_default:
                to_field.default = self.field.default
            schema_editor.alter_field(from_model, from_field, to_field)
            if not self.preserve_default:
                to_field.default = NOT_PROVIDED
```
### 8 - django/db/models/fields/related.py:

Start line: 1882, End line: 1929

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
            self.remote_field.model == RECURSIVE_RELATIONSHIP_CONSTANT
            or self.remote_field.model == cls._meta.object_name
        ):
            self.remote_field.related_name = "%s_rel_+" % name
        elif self.remote_field.is_hidden():
            # If the backwards relation is disabled, replace the original
            # related_name with one generated from the m2m field name. Django
            # still uses backwards relations internally and we need to avoid
            # clashes between multiple m2m fields with related_name == '+'.
            self.remote_field.related_name = "_%s_%s_%s_+" % (
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

                lazy_related_operation(
                    resolve_through_model, cls, self.remote_field.through, field=self
                )
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(
                    self, cls
                )

        # Add the descriptor for the m2m relation.
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))

        # Set up the accessor for the m2m table name for the relation.
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)
```
### 9 - django/db/models/fields/related.py:

Start line: 1931, End line: 1958

```python
class ManyToManyField(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal M2Ms (i.e., those with a related name ending with '+')
        # and swapped models don't get a related descriptor.
        if (
            not self.remote_field.is_hidden()
            and not related.related_model._meta.swapped
        ):
            setattr(
                cls,
                related.get_accessor_name(),
                ManyToManyDescriptor(self.remote_field, reverse=True),
            )

        # Set up the accessors for the column names on the m2m table.
        self.m2m_column_name = partial(self._get_m2m_attr, related, "column")
        self.m2m_reverse_name = partial(self._get_m2m_reverse_attr, related, "column")

        self.m2m_field_name = partial(self._get_m2m_attr, related, "name")
        self.m2m_reverse_field_name = partial(
            self._get_m2m_reverse_attr, related, "name"
        )

        get_m2m_rel = partial(self._get_m2m_attr, related, "remote_field")
        self.m2m_target_field_name = lambda: get_m2m_rel().field_name
        get_m2m_reverse_rel = partial(
            self._get_m2m_reverse_attr, related, "remote_field"
        )
        self.m2m_reverse_target_field_name = lambda: get_m2m_reverse_rel().field_name
```
### 10 - django/db/migrations/operations/models.py:

Start line: 426, End line: 439

```python
class RenameModel(ModelOperation):

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name

        self.database_forwards(app_label, schema_editor, from_state, to_state)

        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name
```
### 61 - django/db/backends/sqlite3/schema.py:

Start line: 176, End line: 264

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(
        self, model, create_field=None, delete_field=None, alter_field=None
    ):
        """
        Shortcut to transform a model from old_model into new_model

        This follows the correct procedure to perform non-rename or column
        addition operations based on SQLite's documentation

        https://www.sqlite.org/lang_altertable.html#caution

        The essential steps are:
          1. Create a table with the updated definition called "new__app_model"
          2. Copy the data from the existing "app_model" table to the new table
          3. Drop the "app_model" table
          4. Rename the "new__app_model" table to "app_model"
          5. Restore any index of the previous "app_model" table.
        """
        # Self-referential fields must be recreated rather than copied from
        # the old model to ensure their remote_field.field_name doesn't refer
        # to an altered field.
        def is_self_referential(f):
            return f.is_relation and f.remote_field.model is model

        # Work out the new fields dict / mapping
        body = {
            f.name: f.clone() if is_self_referential(f) else f
            for f in model._meta.local_concrete_fields
        }
        # Since mapping might mix column names and default values,
        # its values must be already quoted.
        mapping = {
            f.column: self.quote_name(f.column)
            for f in model._meta.local_concrete_fields
        }
        # This maps field names (not columns) for things like unique_together
        rename_mapping = {}
        # If any of the new or altered fields is introducing a new PK,
        # remove the old one
        restore_pk_field = None
        if getattr(create_field, "primary_key", False) or (
            alter_field and getattr(alter_field[1], "primary_key", False)
        ):
            for name, field in list(body.items()):
                if field.primary_key and not (
                    # Do not remove the old primary key when an altered field
                    # that introduces a primary key is the same field.
                    alter_field
                    and name == alter_field[1].name
                ):
                    field.primary_key = False
                    restore_pk_field = field
                    if field.auto_created:
                        del body[name]
                        del mapping[field.column]
        # Add in any created fields
        if create_field:
            body[create_field.name] = create_field
            # Choose a default and insert it into the copy map
            if not create_field.many_to_many and create_field.concrete:
                mapping[create_field.column] = self.prepare_default(
                    self.effective_default(create_field),
                )
        # Add in any altered fields
        if alter_field:
            old_field, new_field = alter_field
            body.pop(old_field.name, None)
            mapping.pop(old_field.column, None)
            body[new_field.name] = new_field
            if old_field.null and not new_field.null:
                case_sql = "coalesce(%(col)s, %(default)s)" % {
                    "col": self.quote_name(old_field.column),
                    "default": self.prepare_default(self.effective_default(new_field)),
                }
                mapping[new_field.column] = case_sql
            else:
                mapping[new_field.column] = self.quote_name(old_field.column)
            rename_mapping[old_field.name] = new_field.name
        # Remove any deleted fields
        if delete_field:
            del body[delete_field.name]
            del mapping[delete_field.column]
            # Remove any implicit M2M tables
            if (
                delete_field.many_to_many
                and delete_field.remote_field.through._meta.auto_created
            ):
                return self.delete_model(delete_field.remote_field.through)
        # Work inside a new app registry
        # ... other code
```
### 100 - django/db/backends/sqlite3/schema.py:

Start line: 429, End line: 486

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        """Perform a "physical" (non-ManyToMany) field update."""
        # Use "ALTER TABLE ... RENAME COLUMN" if only the column name
        # changed and there aren't any constraints.
        if (
            self.connection.features.can_alter_table_rename_column
            and old_field.column != new_field.column
            and self.column_sql(model, old_field) == self.column_sql(model, new_field)
            and not (
                old_field.remote_field
                and old_field.db_constraint
                or new_field.remote_field
                and new_field.db_constraint
            )
        ):
            return self.execute(
                self._rename_field_sql(
                    model._meta.db_table, old_field, new_field, new_type
                )
            )
        # Alter by remaking table
        self._remake_table(model, alter_field=(old_field, new_field))
        # Rebuild tables with FKs pointing to this field.
        old_collation = old_db_params.get("collation")
        new_collation = new_db_params.get("collation")
        if new_field.unique and (
            old_type != new_type or old_collation != new_collation
        ):
            related_models = set()
            opts = new_field.model._meta
            for remote_field in opts.related_objects:
                # Ignore self-relationship since the table was already rebuilt.
                if remote_field.related_model == model:
                    continue
                if not remote_field.many_to_many:
                    if remote_field.field_name == new_field.name:
                        related_models.add(remote_field.related_model)
                elif new_field.primary_key and remote_field.through._meta.auto_created:
                    related_models.add(remote_field.through)
            if new_field.primary_key:
                for many_to_many in opts.many_to_many:
                    # Ignore self-relationship since the table was already rebuilt.
                    if many_to_many.related_model == model:
                        continue
                    if many_to_many.remote_field.through._meta.auto_created:
                        related_models.add(many_to_many.remote_field.through)
            for related_model in related_models:
                self._remake_table(related_model)
```
### 117 - django/db/backends/sqlite3/schema.py:

Start line: 123, End line: 174

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_field(self, model, old_field, new_field, strict=False):
        if not self._field_should_be_altered(old_field, new_field):
            return
        old_field_name = old_field.name
        table_name = model._meta.db_table
        _, old_column_name = old_field.get_attname_column()
        if (
            new_field.name != old_field_name
            and not self.connection.features.supports_atomic_references_rename
            and self._is_referenced_by_fk_constraint(
                table_name, old_column_name, ignore_self=True
            )
        ):
            if self.connection.in_atomic_block:
                raise NotSupportedError(
                    (
                        "Renaming the %r.%r column while in a transaction is not "
                        "supported on SQLite < 3.26 because it would break referential "
                        "integrity. Try adding `atomic = False` to the Migration class."
                    )
                    % (model._meta.db_table, old_field_name)
                )
            with atomic(self.connection.alias):
                super().alter_field(model, old_field, new_field, strict=strict)
                # Follow SQLite's documented procedure for performing changes
                # that don't affect the on-disk content.
                # https://sqlite.org/lang_altertable.html#otheralter
                with self.connection.cursor() as cursor:
                    schema_version = cursor.execute("PRAGMA schema_version").fetchone()[
                        0
                    ]
                    cursor.execute("PRAGMA writable_schema = 1")
                    references_template = ' REFERENCES "%s" ("%%s") ' % table_name
                    new_column_name = new_field.get_attname_column()[1]
                    search = references_template % old_column_name
                    replacement = references_template % new_column_name
                    cursor.execute(
                        "UPDATE sqlite_master SET sql = replace(sql, %s, %s)",
                        (search, replacement),
                    )
                    cursor.execute("PRAGMA schema_version = %d" % (schema_version + 1))
                    cursor.execute("PRAGMA writable_schema = 0")
                    # The integrity check will raise an exception and rollback
                    # the transaction if the sqlite_master updates corrupt the
                    # database.
                    cursor.execute("PRAGMA integrity_check")
            # Perform a VACUUM to refresh the database representation from
            # the sqlite_master table.
            with self.connection.cursor() as cursor:
                cursor.execute("VACUUM")
        else:
            super().alter_field(model, old_field, new_field, strict=strict)
```
### 128 - django/db/backends/sqlite3/schema.py:

Start line: 100, End line: 121

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def alter_db_table(
        self, model, old_db_table, new_db_table, disable_constraints=True
    ):
        if (
            not self.connection.features.supports_atomic_references_rename
            and disable_constraints
            and self._is_referenced_by_fk_constraint(old_db_table)
        ):
            if self.connection.in_atomic_block:
                raise NotSupportedError(
                    (
                        "Renaming the %r table while in a transaction is not "
                        "supported on SQLite < 3.26 because it would break referential "
                        "integrity. Try adding `atomic = False` to the Migration class."
                    )
                    % old_db_table
                )
            self.connection.enable_constraint_checking()
            super().alter_db_table(model, old_db_table, new_db_table)
            self.connection.disable_constraint_checking()
        else:
            super().alter_db_table(model, old_db_table, new_db_table)
```
### 187 - django/db/backends/sqlite3/schema.py:

Start line: 265, End line: 360

```python
class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):

    def _remake_table(
        self, model, create_field=None, delete_field=None, alter_field=None
    ):
        # ... other code
        apps = Apps()

        # Work out the new value of unique_together, taking renames into
        # account
        unique_together = [
            [rename_mapping.get(n, n) for n in unique]
            for unique in model._meta.unique_together
        ]

        # Work out the new value for index_together, taking renames into
        # account
        index_together = [
            [rename_mapping.get(n, n) for n in index]
            for index in model._meta.index_together
        ]

        indexes = model._meta.indexes
        if delete_field:
            indexes = [
                index for index in indexes if delete_field.name not in index.fields
            ]

        constraints = list(model._meta.constraints)

        # Provide isolated instances of the fields to the new model body so
        # that the existing model's internals aren't interfered with when
        # the dummy model is constructed.
        body_copy = copy.deepcopy(body)

        # Construct a new model with the new fields to allow self referential
        # primary key to resolve to. This model won't ever be materialized as a
        # table and solely exists for foreign key reference resolution purposes.
        # This wouldn't be required if the schema editor was operating on model
        # states instead of rendered models.
        meta_contents = {
            "app_label": model._meta.app_label,
            "db_table": model._meta.db_table,
            "unique_together": unique_together,
            "index_together": index_together,
            "indexes": indexes,
            "constraints": constraints,
            "apps": apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy["Meta"] = meta
        body_copy["__module__"] = model.__module__
        type(model._meta.object_name, model.__bases__, body_copy)

        # Construct a model with a renamed table name.
        body_copy = copy.deepcopy(body)
        meta_contents = {
            "app_label": model._meta.app_label,
            "db_table": "new__%s" % strip_quotes(model._meta.db_table),
            "unique_together": unique_together,
            "index_together": index_together,
            "indexes": indexes,
            "constraints": constraints,
            "apps": apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy["Meta"] = meta
        body_copy["__module__"] = model.__module__
        new_model = type("New%s" % model._meta.object_name, model.__bases__, body_copy)

        # Create a new table with the updated schema.
        self.create_model(new_model)

        # Copy data from the old table into the new table
        self.execute(
            "INSERT INTO %s (%s) SELECT %s FROM %s"
            % (
                self.quote_name(new_model._meta.db_table),
                ", ".join(self.quote_name(x) for x in mapping),
                ", ".join(mapping.values()),
                self.quote_name(model._meta.db_table),
            )
        )

        # Delete the old table to make way for the new
        self.delete_model(model, handle_autom2m=False)

        # Rename the new table to take way for the old
        self.alter_db_table(
            new_model,
            new_model._meta.db_table,
            model._meta.db_table,
            disable_constraints=False,
        )

        # Run deferred SQL on correct table
        for sql in self.deferred_sql:
            self.execute(sql)
        self.deferred_sql = []
        # Fix any PK-removed field
        if restore_pk_field:
            restore_pk_field.primary_key = True
```
