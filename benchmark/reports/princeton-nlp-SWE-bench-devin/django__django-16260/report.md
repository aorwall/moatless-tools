# django__django-16260

| **django/django** | `444b6da7cc229a58a2c476a52e45233001dc7073` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2603 |
| **Any found context length** | 2603 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -737,6 +737,11 @@ def refresh_from_db(self, using=None, fields=None):
             if field.is_cached(self):
                 field.delete_cached_value(self)
 
+        # Clear cached private relations.
+        for field in self._meta.private_fields:
+            if field.is_relation and field.is_cached(self):
+                field.delete_cached_value(self)
+
         self._state.db = db_instance._state.db
 
     async def arefresh_from_db(self, using=None, fields=None):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 740 | 740 | 5 | 2 | 2603


## Problem Statement

```
model.refresh_from_db() doesn't clear cached generic foreign keys
Description
	 
		(last modified by pascal chambon)
	 
In my code, Users have a generic foreign key like so: 
	controlled_entity_content_type = models.ForeignKey(ContentType, blank=True, null=True, on_delete=models.CASCADE)
	controlled_entity_object_id = models.PositiveIntegerField(blank=True, null=True)
	controlled_entity = GenericForeignKey("controlled_entity_content_type", "controlled_entity_object_id")
However, in unit-tests, when I refresh a user instance, the controlled_entity relation isn't cleared from cache, as can be seen here with IDs: 
		old_controlled_entity = authenticated_user.controlled_entity
		authenticated_user.refresh_from_db()
		new_controlled_entity = authenticated_user.controlled_entity
		assert id(old_controlled_entity) != id(new_controlled_entity) # FAILS
And this leads to subtle bugs like non-transitive equalities in tests :
		assert authenticated_user.controlled_entity == fixtures.client1_project2_organization3
		assert fixtures.client1_project2_organization3.get_pricing_plan() == pricing_plan
		assert authenticated_user.controlled_entity.get_pricing_plan() == pricing_plan	 # FAILS
Calling "authenticated_user.controlled_entity.refresh_from_db()" solved this particular bug, but "authenticated_user.refresh_from_db()" isn't enough.
Tested under Django3.2.13 but the code of refresh_from_db() hasn't changed since then in Git's main branch (except few cosmetic adjustments on code format).
I'm a bit lost in the code of refresh_from_db(), but I've just seen that the generic relation appears once in this loop, but is not considered as "cached" in the if() branch.
		for field in self._meta.related_objects:
			#print("%% CLEARING RELATED FIELD", field)
			if field.is_cached(self):
				#print("%% DONE") # not called
				field.delete_cached_value(self)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/related.py | 1450 | 1579| 984 | 984 | 14607 | 
| 2 | **2 django/db/models/base.py** | 1064 | 1116| 520 | 1504 | 33146 | 
| 3 | 2 django/db/models/fields/related.py | 302 | 339| 296 | 1800 | 33146 | 
| 4 | 2 django/db/models/fields/related.py | 1018 | 1054| 261 | 2061 | 33146 | 
| **-> 5 <-** | **2 django/db/models/base.py** | 675 | 740| 542 | 2603 | 33146 | 
| 6 | 2 django/db/models/fields/related.py | 208 | 224| 142 | 2745 | 33146 | 
| 7 | 2 django/db/models/fields/related.py | 154 | 185| 209 | 2954 | 33146 | 
| 8 | 3 django/core/cache/backends/db.py | 257 | 295| 356 | 3310 | 35290 | 
| 9 | 4 django/contrib/contenttypes/fields.py | 224 | 266| 384 | 3694 | 40854 | 
| 10 | 4 django/db/models/fields/related.py | 604 | 670| 497 | 4191 | 40854 | 
| 11 | 4 django/db/models/fields/related.py | 187 | 206| 155 | 4346 | 40854 | 
| 12 | 4 django/db/models/fields/related.py | 226 | 301| 696 | 5042 | 40854 | 
| 13 | 4 django/db/models/fields/related.py | 992 | 1016| 176 | 5218 | 40854 | 
| 14 | 5 django/db/backends/base/schema.py | 39 | 72| 214 | 5432 | 54718 | 
| 15 | 6 django/db/models/deletion.py | 314 | 375| 632 | 6064 | 58700 | 
| 16 | 6 django/db/models/fields/related.py | 126 | 152| 171 | 6235 | 58700 | 
| 17 | 6 django/db/models/fields/related.py | 341 | 378| 294 | 6529 | 58700 | 
| 18 | 7 django/db/models/query.py | 2254 | 2388| 1129 | 7658 | 79336 | 
| 19 | 7 django/contrib/contenttypes/fields.py | 364 | 390| 181 | 7839 | 79336 | 
| 20 | 8 django/db/models/fields/reverse_related.py | 153 | 163| 130 | 7969 | 81868 | 
| 21 | 8 django/db/models/fields/related.py | 1103 | 1120| 133 | 8102 | 81868 | 
| 22 | 8 django/db/models/fields/related.py | 871 | 898| 237 | 8339 | 81868 | 
| 23 | **8 django/db/models/base.py** | 2448 | 2500| 341 | 8680 | 81868 | 
| 24 | 8 django/contrib/contenttypes/fields.py | 688 | 715| 218 | 8898 | 81868 | 
| 25 | 8 django/db/models/fields/reverse_related.py | 185 | 203| 167 | 9065 | 81868 | 
| 26 | 9 django/db/models/fields/related_descriptors.py | 404 | 425| 158 | 9223 | 92958 | 
| 27 | 9 django/db/models/fields/related_descriptors.py | 786 | 861| 596 | 9819 | 92958 | 
| 28 | 9 django/db/models/fields/reverse_related.py | 20 | 151| 765 | 10584 | 92958 | 
| 29 | **9 django/db/models/base.py** | 1807 | 1842| 246 | 10830 | 92958 | 
| 30 | 9 django/db/models/fields/related.py | 1581 | 1679| 655 | 11485 | 92958 | 
| 31 | 9 django/contrib/contenttypes/fields.py | 717 | 737| 192 | 11677 | 92958 | 
| 32 | 9 django/db/models/fields/related.py | 1681 | 1731| 431 | 12108 | 92958 | 
| 33 | 9 django/contrib/contenttypes/fields.py | 739 | 767| 258 | 12366 | 92958 | 
| 34 | 9 django/db/models/fields/related.py | 582 | 602| 138 | 12504 | 92958 | 
| 35 | 9 django/contrib/contenttypes/fields.py | 175 | 222| 417 | 12921 | 92958 | 
| 36 | 9 django/db/models/fields/related.py | 1177 | 1209| 246 | 13167 | 92958 | 
| 37 | 9 django/core/cache/backends/db.py | 206 | 233| 280 | 13447 | 92958 | 
| 38 | 9 django/db/models/fields/related.py | 1079 | 1101| 180 | 13627 | 92958 | 
| 39 | 9 django/db/models/fields/related.py | 707 | 731| 210 | 13837 | 92958 | 
| 40 | 9 django/contrib/contenttypes/fields.py | 472 | 499| 258 | 14095 | 92958 | 
| 41 | 9 django/db/models/fields/related.py | 672 | 705| 335 | 14430 | 92958 | 
| 42 | 9 django/db/models/fields/related.py | 733 | 755| 172 | 14602 | 92958 | 
| 43 | 9 django/contrib/contenttypes/fields.py | 652 | 686| 288 | 14890 | 92958 | 
| 44 | 9 django/contrib/contenttypes/fields.py | 616 | 650| 341 | 15231 | 92958 | 
| 45 | 9 django/core/cache/backends/db.py | 113 | 204| 809 | 16040 | 92958 | 
| 46 | 9 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 16160 | 92958 | 
| 47 | 9 django/db/models/fields/related_descriptors.py | 153 | 197| 418 | 16578 | 92958 | 
| 48 | **9 django/db/models/base.py** | 1296 | 1343| 413 | 16991 | 92958 | 
| 49 | 9 django/contrib/contenttypes/fields.py | 23 | 112| 560 | 17551 | 92958 | 
| 50 | 9 django/db/models/deletion.py | 377 | 401| 273 | 17824 | 92958 | 
| 51 | 9 django/db/models/fields/related.py | 1418 | 1448| 172 | 17996 | 92958 | 
| 52 | 9 django/core/cache/backends/db.py | 42 | 99| 430 | 18426 | 92958 | 
| 53 | 9 django/db/models/fields/reverse_related.py | 205 | 238| 306 | 18732 | 92958 | 
| 54 | 10 django/db/models/options.py | 826 | 841| 144 | 18876 | 100641 | 
| 55 | 10 django/db/models/query.py | 2626 | 2649| 201 | 19077 | 100641 | 
| 56 | 10 django/db/models/fields/related_descriptors.py | 1 | 89| 731 | 19808 | 100641 | 
| 57 | 11 django/db/models/fields/related_lookups.py | 75 | 108| 369 | 20177 | 102241 | 
| 58 | 11 django/db/models/fields/related.py | 514 | 580| 367 | 20544 | 102241 | 
| 59 | 11 django/core/cache/backends/db.py | 235 | 255| 246 | 20790 | 102241 | 
| 60 | **11 django/db/models/base.py** | 1345 | 1374| 294 | 21084 | 102241 | 
| 61 | 11 django/contrib/contenttypes/fields.py | 161 | 173| 125 | 21209 | 102241 | 
| 62 | 11 django/db/models/fields/related_descriptors.py | 427 | 450| 194 | 21403 | 102241 | 
| 63 | 11 django/db/models/fields/related.py | 1 | 40| 251 | 21654 | 102241 | 
| 64 | **11 django/db/models/base.py** | 2254 | 2445| 1306 | 22960 | 102241 | 
| 65 | 12 django/db/backends/sqlite3/operations.py | 214 | 239| 218 | 23178 | 105713 | 
| 66 | 12 django/db/models/fields/related.py | 1160 | 1175| 136 | 23314 | 105713 | 
| 67 | **12 django/db/models/base.py** | 459 | 572| 957 | 24271 | 105713 | 
| 68 | 12 django/db/models/fields/related_lookups.py | 151 | 168| 216 | 24487 | 105713 | 
| 69 | 12 django/db/models/fields/related.py | 1882 | 1929| 505 | 24992 | 105713 | 
| 70 | 13 django/db/backends/mysql/features.py | 84 | 158| 597 | 25589 | 108137 | 
| 71 | 13 django/db/models/query.py | 1928 | 1993| 543 | 26132 | 108137 | 
| 72 | **13 django/db/models/base.py** | 1521 | 1556| 277 | 26409 | 108137 | 
| 73 | 14 django/db/migrations/autodetector.py | 811 | 906| 712 | 27121 | 121598 | 
| 74 | **14 django/db/models/base.py** | 1118 | 1145| 238 | 27359 | 121598 | 
| 75 | 14 django/db/models/fields/related_descriptors.py | 703 | 726| 227 | 27586 | 121598 | 
| 76 | 15 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 34 | 111| 598 | 28184 | 122386 | 
| 77 | 15 django/db/models/deletion.py | 1 | 93| 601 | 28785 | 122386 | 
| 78 | 15 django/db/models/fields/related.py | 819 | 869| 378 | 29163 | 122386 | 
| 79 | 15 django/db/backends/base/schema.py | 1 | 36| 223 | 29386 | 122386 | 
| 80 | 16 django/db/migrations/state.py | 437 | 458| 204 | 29590 | 130554 | 
| 81 | **16 django/db/models/base.py** | 1191 | 1231| 310 | 29900 | 130554 | 
| 82 | **16 django/db/models/base.py** | 1993 | 2046| 359 | 30259 | 130554 | 
| 83 | 16 django/db/models/fields/related.py | 1122 | 1158| 284 | 30543 | 130554 | 
| 84 | 16 django/contrib/contenttypes/fields.py | 557 | 614| 443 | 30986 | 130554 | 
| 85 | 17 django/contrib/sites/models.py | 79 | 122| 236 | 31222 | 131343 | 
| 86 | **17 django/db/models/base.py** | 1648 | 1682| 252 | 31474 | 131343 | 
| 87 | 17 django/db/models/fields/related.py | 485 | 511| 177 | 31651 | 131343 | 
| 88 | 18 django/db/backends/oracle/creation.py | 159 | 201| 411 | 32062 | 135358 | 
| 89 | 18 django/db/models/fields/related_lookups.py | 170 | 210| 250 | 32312 | 135358 | 
| 90 | 18 django/db/models/fields/related_descriptors.py | 728 | 747| 231 | 32543 | 135358 | 
| 91 | 19 django/db/models/__init__.py | 1 | 116| 682 | 33225 | 136040 | 
| 92 | 20 django/db/backends/sqlite3/features.py | 68 | 115| 383 | 33608 | 137326 | 
| 93 | 21 django/db/models/sql/compiler.py | 1221 | 1339| 928 | 34536 | 153897 | 
| 94 | 21 django/db/models/fields/related_lookups.py | 1 | 40| 221 | 34757 | 153897 | 
| 95 | 21 django/db/models/fields/related.py | 89 | 124| 232 | 34989 | 153897 | 
| 96 | 22 django/db/backends/oracle/operations.py | 407 | 453| 385 | 35374 | 160117 | 
| 97 | 23 django/contrib/admin/utils.py | 295 | 321| 189 | 35563 | 164385 | 
| 98 | 23 django/db/backends/base/schema.py | 1063 | 1143| 779 | 36342 | 164385 | 
| 99 | 23 django/db/models/fields/related_descriptors.py | 250 | 332| 760 | 37102 | 164385 | 
| 100 | 23 django/core/cache/backends/db.py | 101 | 111| 222 | 37324 | 164385 | 
| 101 | 23 django/db/models/fields/related_lookups.py | 110 | 148| 324 | 37648 | 164385 | 
| 102 | 23 django/db/models/fields/related_descriptors.py | 863 | 891| 233 | 37881 | 164385 | 
| 103 | 23 django/db/models/fields/related.py | 1733 | 1771| 413 | 38294 | 164385 | 
| 104 | 24 django/db/backends/sqlite3/schema.py | 265 | 360| 807 | 39101 | 169006 | 
| 105 | 24 django/db/backends/base/schema.py | 523 | 542| 196 | 39297 | 169006 | 
| 106 | 25 django/contrib/contenttypes/admin.py | 1 | 88| 585 | 39882 | 170011 | 
| 107 | 25 django/db/models/fields/related.py | 1931 | 1958| 305 | 40187 | 170011 | 
| 108 | 25 django/db/migrations/autodetector.py | 600 | 776| 1231 | 41418 | 170011 | 
| 109 | 25 django/db/models/query.py | 2225 | 2253| 246 | 41664 | 170011 | 
| 110 | **25 django/db/models/base.py** | 1703 | 1756| 494 | 42158 | 170011 | 
| 111 | 25 django/db/models/fields/related.py | 1960 | 1994| 266 | 42424 | 170011 | 
| 112 | 25 django/contrib/contenttypes/fields.py | 269 | 289| 107 | 42531 | 170011 | 
| 113 | **25 django/db/models/base.py** | 574 | 612| 326 | 42857 | 170011 | 
| 114 | 25 django/db/models/fields/related_descriptors.py | 1184 | 1218| 345 | 43202 | 170011 | 
| 115 | 25 django/db/backends/mysql/features.py | 160 | 267| 728 | 43930 | 170011 | 
| 116 | 25 django/db/models/fields/related_descriptors.py | 749 | 784| 268 | 44198 | 170011 | 
| 117 | 25 django/contrib/contenttypes/fields.py | 292 | 362| 487 | 44685 | 170011 | 
| 118 | 25 django/db/backends/base/schema.py | 890 | 974| 778 | 45463 | 170011 | 
| 119 | **25 django/db/models/base.py** | 1558 | 1588| 247 | 45710 | 170011 | 
| 120 | 25 django/contrib/contenttypes/fields.py | 521 | 554| 229 | 45939 | 170011 | 
| 121 | 26 django/db/models/query_utils.py | 366 | 392| 289 | 46228 | 173228 | 
| 122 | 26 django/db/models/fields/related.py | 757 | 775| 166 | 46394 | 173228 | 
| 123 | 26 django/db/models/fields/related_descriptors.py | 366 | 383| 188 | 46582 | 173228 | 
| 124 | 26 django/db/models/fields/related.py | 380 | 401| 219 | 46801 | 173228 | 
| 125 | 26 django/db/models/fields/related_descriptors.py | 1124 | 1150| 202 | 47003 | 173228 | 
| 126 | 26 django/contrib/contenttypes/fields.py | 114 | 159| 322 | 47325 | 173228 | 
| 127 | 26 django/db/models/fields/related_descriptors.py | 663 | 701| 354 | 47679 | 173228 | 
| 128 | 26 django/db/models/sql/compiler.py | 1964 | 2025| 588 | 48267 | 173228 | 
| 129 | 26 django/db/backends/base/schema.py | 1720 | 1757| 303 | 48570 | 173228 | 
| 130 | 26 django/db/backends/oracle/creation.py | 29 | 124| 768 | 49338 | 173228 | 
| 131 | 26 django/db/models/options.py | 331 | 367| 338 | 49676 | 173228 | 
| 132 | 26 django/db/models/fields/related_descriptors.py | 1375 | 1425| 407 | 50083 | 173228 | 
| 133 | **26 django/db/models/base.py** | 1 | 66| 361 | 50444 | 173228 | 
| 134 | **26 django/db/models/base.py** | 250 | 367| 874 | 51318 | 173228 | 
| 135 | 26 django/db/models/query.py | 2521 | 2553| 314 | 51632 | 173228 | 
| 136 | 26 django/db/models/fields/reverse_related.py | 165 | 183| 160 | 51792 | 173228 | 
| 137 | 27 django/contrib/admin/options.py | 432 | 490| 509 | 52301 | 192467 | 
| 138 | 27 django/db/models/fields/related.py | 901 | 990| 583 | 52884 | 192467 | 
| 139 | 27 django/db/models/options.py | 786 | 824| 394 | 53278 | 192467 | 
| 140 | 27 django/db/models/deletion.py | 96 | 116| 210 | 53488 | 192467 | 
| 141 | 27 django/db/backends/sqlite3/schema.py | 123 | 174| 527 | 54015 | 192467 | 
| 142 | 27 django/db/migrations/state.py | 460 | 486| 150 | 54165 | 192467 | 
| 143 | 27 django/db/models/fields/related_descriptors.py | 1052 | 1094| 381 | 54546 | 192467 | 


### Hint

```
Thanks for the report.
Bonjour Pascal, It seems to be an oversight in Model.refresh_from_db as it should also consider _meta.private_fields which is where GenericForeignKey and GenericRel end up as opposed to related_objects. Something along these lines should address the issue django/db/models/base.py diff --git a/django/db/models/base.py b/django/db/models/base.py index 2eb7ba7e9b..0f5f8d0881 100644 a b def refresh_from_db(self, using=None, fields=None): 737737 if field.is_cached(self): 738738 field.delete_cached_value(self) 739739 740 # Clear cached private relations. 741 for field in self._meta.private_fields: 742 if field.is_relation and field.is_cached(self): 743 field.delete_cached_value(self) 744 740745 self._state.db = db_instance._state.db 741746 742747 async def arefresh_from_db(self, using=None, fields=None): Would you be interested in submitting a patch whit these changes and ​adding a regression test to the suite?
Hi, I was just working around to write a regression test for the issue, if not pascal I would like to submit a patch with a test and changes proposed by simon. Thanks :)
Please proceed yes I'll be happy to backport/test the PR against my own project to validate it in real conditions
```

## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -737,6 +737,11 @@ def refresh_from_db(self, using=None, fields=None):
             if field.is_cached(self):
                 field.delete_cached_value(self)
 
+        # Clear cached private relations.
+        for field in self._meta.private_fields:
+            if field.is_relation and field.is_cached(self):
+                field.delete_cached_value(self)
+
         self._state.db = db_instance._state.db
 
     async def arefresh_from_db(self, using=None, fields=None):

```

## Test Patch

```diff
diff --git a/tests/contenttypes_tests/test_fields.py b/tests/contenttypes_tests/test_fields.py
--- a/tests/contenttypes_tests/test_fields.py
+++ b/tests/contenttypes_tests/test_fields.py
@@ -43,6 +43,14 @@ def test_get_object_cache_respects_deleted_objects(self):
             self.assertIsNone(post.parent)
             self.assertIsNone(post.parent)
 
+    def test_clear_cached_generic_relation(self):
+        question = Question.objects.create(text="What is your name?")
+        answer = Answer.objects.create(text="Answer", question=question)
+        old_entity = answer.question
+        answer.refresh_from_db()
+        new_entity = answer.question
+        self.assertIsNot(old_entity, new_entity)
+
 
 class GenericRelationTests(TestCase):
     def test_value_to_string(self):

```


## Code snippets

### 1 - django/db/models/fields/related.py:

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
### 2 - django/db/models/base.py:

Start line: 1064, End line: 1116

```python
class Model(AltersData, metaclass=ModelBase):

    def _prepare_related_fields_for_save(self, operation_name, fields=None):
        # Ensure that a model instance without a PK hasn't been assigned to
        # a ForeignKey, GenericForeignKey or OneToOneField on this model. If
        # the field is nullable, allowing the save would result in silent data
        # loss.
        for field in self._meta.concrete_fields:
            if fields and field not in fields:
                continue
            # If the related field isn't cached, then an instance hasn't been
            # assigned and there's no need to worry about this check.
            if field.is_relation and field.is_cached(self):
                obj = getattr(self, field.name, None)
                if not obj:
                    continue
                # A pk may have been assigned manually to a model instance not
                # saved to the database (or auto-generated in a case like
                # UUIDField), but we allow the save to proceed and rely on the
                # database to raise an IntegrityError if applicable. If
                # constraints aren't supported by the database, there's the
                # unavoidable risk of data corruption.
                if obj.pk is None:
                    # Remove the object from a related instance cache.
                    if not field.remote_field.multiple:
                        field.remote_field.delete_cached_value(obj)
                    raise ValueError(
                        "%s() prohibited to prevent data loss due to unsaved "
                        "related object '%s'." % (operation_name, field.name)
                    )
                elif getattr(self, field.attname) in field.empty_values:
                    # Set related object if it has been saved after an
                    # assignment.
                    setattr(self, field.name, obj)
                # If the relationship's pk/to_field was changed, clear the
                # cached relationship.
                if getattr(obj, field.target_field.attname) != getattr(
                    self, field.attname
                ):
                    field.delete_cached_value(self)
        # GenericForeignKeys are private.
        for field in self._meta.private_fields:
            if fields and field not in fields:
                continue
            if (
                field.is_relation
                and field.is_cached(self)
                and hasattr(field, "fk_field")
            ):
                obj = field.get_cached_value(self, default=None)
                if obj and obj.pk is None:
                    raise ValueError(
                        f"{operation_name}() prohibited to prevent data loss due to "
                        f"unsaved related object '{field.name}'."
                    )
```
### 3 - django/db/models/fields/related.py:

Start line: 302, End line: 339

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_clashes(self):
        # ... other code
        for clash_field in potential_clashes:
            # i.e. "app_label.Model.m2m".
            clash_name = "%s.%s" % (
                clash_field.related_model._meta.label,
                clash_field.field.name,
            )
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(
                    checks.Error(
                        f"Reverse accessor '{rel_opts.object_name}.{rel_name}' "
                        f"for '{field_name}' clashes with reverse accessor for "
                        f"'{clash_name}'.",
                        hint=(
                            "Add or change a related_name argument "
                            "to the definition for '%s' or '%s'."
                        )
                        % (field_name, clash_name),
                        obj=self,
                        id="fields.E304",
                    )
                )

            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(
                    checks.Error(
                        "Reverse query name for '%s' clashes with reverse query name "
                        "for '%s'." % (field_name, clash_name),
                        hint=(
                            "Add or change a related_name argument "
                            "to the definition for '%s' or '%s'."
                        )
                        % (field_name, clash_name),
                        obj=self,
                        id="fields.E305",
                    )
                )

        return errors
```
### 4 - django/db/models/fields/related.py:

Start line: 1018, End line: 1054

```python
class ForeignKey(ForeignObject):

    def _check_unique(self, **kwargs):
        return (
            [
                checks.Warning(
                    "Setting unique=True on a ForeignKey has the same effect as using "
                    "a OneToOneField.",
                    hint=(
                        "ForeignKey(unique=True) is usually better served by a "
                        "OneToOneField."
                    ),
                    obj=self,
                    id="fields.W342",
                )
            ]
            if self.unique
            else []
        )

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs["to_fields"]
        del kwargs["from_fields"]
        # Handle the simpler arguments
        if self.db_index:
            del kwargs["db_index"]
        else:
            kwargs["db_index"] = False
        if self.db_constraint is not True:
            kwargs["db_constraint"] = self.db_constraint
        # Rel needs more work.
        to_meta = getattr(self.remote_field.model, "_meta", None)
        if self.remote_field.field_name and (
            not to_meta
            or (to_meta.pk and self.remote_field.field_name != to_meta.pk.name)
        ):
            kwargs["to_field"] = self.remote_field.field_name
        return name, path, args, kwargs
```
### 5 - django/db/models/base.py:

Start line: 675, End line: 740

```python
class Model(AltersData, metaclass=ModelBase):

    def refresh_from_db(self, using=None, fields=None):
        """
        Reload field values from the database.

        By default, the reloading happens from the database this instance was
        loaded from, or by the read router if this instance wasn't loaded from
        any database. The using parameter will override the default.

        Fields can be used to specify which fields to reload. The fields
        should be an iterable of field attnames. If fields is None, then
        all non-deferred fields are reloaded.

        When accessing deferred fields of an instance, the deferred loading
        of the field will call this method.
        """
        if fields is None:
            self._prefetched_objects_cache = {}
        else:
            prefetched_objects_cache = getattr(self, "_prefetched_objects_cache", ())
            for field in fields:
                if field in prefetched_objects_cache:
                    del prefetched_objects_cache[field]
                    fields.remove(field)
            if not fields:
                return
            if any(LOOKUP_SEP in f for f in fields):
                raise ValueError(
                    'Found "%s" in fields argument. Relations and transforms '
                    "are not allowed in fields." % LOOKUP_SEP
                )

        hints = {"instance": self}
        db_instance_qs = self.__class__._base_manager.db_manager(
            using, hints=hints
        ).filter(pk=self.pk)

        # Use provided fields, if not set then reload all non-deferred fields.
        deferred_fields = self.get_deferred_fields()
        if fields is not None:
            fields = list(fields)
            db_instance_qs = db_instance_qs.only(*fields)
        elif deferred_fields:
            fields = [
                f.attname
                for f in self._meta.concrete_fields
                if f.attname not in deferred_fields
            ]
            db_instance_qs = db_instance_qs.only(*fields)

        db_instance = db_instance_qs.get()
        non_loaded_fields = db_instance.get_deferred_fields()
        for field in self._meta.concrete_fields:
            if field.attname in non_loaded_fields:
                # This field wasn't refreshed - skip ahead.
                continue
            setattr(self, field.attname, getattr(db_instance, field.attname))
            # Clear cached foreign keys.
            if field.is_relation and field.is_cached(self):
                field.delete_cached_value(self)

        # Clear cached relations.
        for field in self._meta.related_objects:
            if field.is_cached(self):
                field.delete_cached_value(self)

        self._state.db = db_instance._state.db
```
### 6 - django/db/models/fields/related.py:

Start line: 208, End line: 224

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_referencing_to_swapped_model(self):
        if (
            self.remote_field.model not in self.opts.apps.get_models()
            and not isinstance(self.remote_field.model, str)
            and self.remote_field.model._meta.swapped
        ):
            return [
                checks.Error(
                    "Field defines a relation with the model '%s', which has "
                    "been swapped out." % self.remote_field.model._meta.label,
                    hint="Update the relation to point at 'settings.%s'."
                    % self.remote_field.model._meta.swappable,
                    obj=self,
                    id="fields.E301",
                )
            ]
        return []
```
### 7 - django/db/models/fields/related.py:

Start line: 154, End line: 185

```python
class RelatedField(FieldCacheMixin, Field):

    def _check_related_query_name_is_valid(self):
        if self.remote_field.is_hidden():
            return []
        rel_query_name = self.related_query_name()
        errors = []
        if rel_query_name.endswith("_"):
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not end with an underscore."
                    % rel_query_name,
                    hint=(
                        "Add or change a related_name or related_query_name "
                        "argument for this field."
                    ),
                    obj=self,
                    id="fields.E308",
                )
            )
        if LOOKUP_SEP in rel_query_name:
            errors.append(
                checks.Error(
                    "Reverse query name '%s' must not contain '%s'."
                    % (rel_query_name, LOOKUP_SEP),
                    hint=(
                        "Add or change a related_name or related_query_name "
                        "argument for this field."
                    ),
                    obj=self,
                    id="fields.E309",
                )
            )
        return errors
```
### 8 - django/core/cache/backends/db.py:

Start line: 257, End line: 295

```python
class DatabaseCache(BaseDatabaseCache):

    # This class uses cursors provided by the database connection. This means
    # it reads expiration values as aware or naive datetimes, depending on the
    # value of USE_TZ and whether the database supports time zones. The ORM's
    # conversion and adaptation infrastructure is then used to avoid comparing

    pickle_protocol =
    # ... other code

    def _cull(self, db, cursor, now, num):
        if self._cull_frequency == 0:
            self.clear()
        else:
            connection = connections[db]
            table = connection.ops.quote_name(self._table)
            cursor.execute(
                "DELETE FROM %s WHERE %s < %%s"
                % (
                    table,
                    connection.ops.quote_name("expires"),
                ),
                [connection.ops.adapt_datetimefield_value(now)],
            )
            deleted_count = cursor.rowcount
            remaining_num = num - deleted_count
            if remaining_num > self._max_entries:
                cull_num = remaining_num // self._cull_frequency
                cursor.execute(
                    connection.ops.cache_key_culling_sql() % table, [cull_num]
                )
                last_cache_key = cursor.fetchone()
                if last_cache_key:
                    cursor.execute(
                        "DELETE FROM %s WHERE %s < %%s"
                        % (
                            table,
                            connection.ops.quote_name("cache_key"),
                        ),
                        [last_cache_key[0]],
                    )

    def clear(self):
        db = router.db_for_write(self.cache_model_class)
        connection = connections[db]
        table = connection.ops.quote_name(self._table)
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM %s" % table)
```
### 9 - django/contrib/contenttypes/fields.py:

Start line: 224, End line: 266

```python
class GenericForeignKey(FieldCacheMixin):

    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        # Don't use getattr(instance, self.ct_field) here because that might
        # reload the same ContentType over and over (#5570). Instead, get the
        # content type ID here, and later when the actual instance is needed,
        # use ContentType.objects.get_for_id(), which has a global cache.
        f = self.model._meta.get_field(self.ct_field)
        ct_id = getattr(instance, f.get_attname(), None)
        pk_val = getattr(instance, self.fk_field)

        rel_obj = self.get_cached_value(instance, default=None)
        if rel_obj is None and self.is_cached(instance):
            return rel_obj
        if rel_obj is not None:
            ct_match = (
                ct_id == self.get_content_type(obj=rel_obj, using=instance._state.db).id
            )
            pk_match = rel_obj._meta.pk.to_python(pk_val) == rel_obj.pk
            if ct_match and pk_match:
                return rel_obj
            else:
                rel_obj = None
        if ct_id is not None:
            ct = self.get_content_type(id=ct_id, using=instance._state.db)
            try:
                rel_obj = ct.get_object_for_this_type(pk=pk_val)
            except ObjectDoesNotExist:
                pass
        self.set_cached_value(instance, rel_obj)
        return rel_obj

    def __set__(self, instance, value):
        ct = None
        fk = None
        if value is not None:
            ct = self.get_content_type(obj=value)
            fk = value.pk

        setattr(instance, self.ct_field, ct)
        setattr(instance, self.fk_field, fk)
        self.set_cached_value(instance, value)
```
### 10 - django/db/models/fields/related.py:

Start line: 604, End line: 670

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
            if getattr(f, "unique", False)
        }
        unique_foreign_fields.update(
            {frozenset(ut) for ut in self.remote_field.model._meta.unique_together}
        )
        unique_foreign_fields.update(
            {
                frozenset(uc.fields)
                for uc in self.remote_field.model._meta.total_unique_constraints
            }
        )
        foreign_fields = {f.name for f in self.foreign_related_fields}
        has_unique_constraint = any(u <= foreign_fields for u in unique_foreign_fields)

        if not has_unique_constraint and len(self.foreign_related_fields) > 1:
            field_combination = ", ".join(
                "'%s'" % rel_field.name for rel_field in self.foreign_related_fields
            )
            model_name = self.remote_field.model.__name__
            return [
                checks.Error(
                    "No subset of the fields %s on model '%s' is unique."
                    % (field_combination, model_name),
                    hint=(
                        "Mark a single field as unique=True or add a set of "
                        "fields to a unique constraint (via unique_together "
                        "or a UniqueConstraint (without condition) in the "
                        "model Meta.constraints)."
                    ),
                    obj=self,
                    id="fields.E310",
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
                        "Add unique=True to this field or add a "
                        "UniqueConstraint (without condition) in the model "
                        "Meta.constraints."
                    ),
                    obj=self,
                    id="fields.E311",
                )
            ]
        else:
            return []
```
### 23 - django/db/models/base.py:

Start line: 2448, End line: 2500

```python
############################################
# HELPER FUNCTIONS (CURRIED MODEL METHODS) #
############################################

# ORDERING METHODS #########################


def method_set_order(self, ordered_obj, id_list, using=None):
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    ordered_obj.objects.db_manager(using).filter(**filter_args).bulk_update(
        [ordered_obj(pk=pk, _order=order) for order, pk in enumerate(id_list)],
        ["_order"],
    )


def method_get_order(self, ordered_obj):
    order_wrt = ordered_obj._meta.order_with_respect_to
    filter_args = order_wrt.get_forward_related_filter(self)
    pk_name = ordered_obj._meta.pk.name
    return ordered_obj.objects.filter(**filter_args).values_list(pk_name, flat=True)


def make_foreign_order_accessors(model, related_model):
    setattr(
        related_model,
        "get_%s_order" % model.__name__.lower(),
        partialmethod(method_get_order, model),
    )
    setattr(
        related_model,
        "set_%s_order" % model.__name__.lower(),
        partialmethod(method_set_order, model),
    )


########
# MISC #
########


def model_unpickle(model_id):
    """Used to unpickle Model subclasses with deferred fields."""
    if isinstance(model_id, tuple):
        model = apps.get_model(*model_id)
    else:
        # Backwards compat - the model was cached directly in earlier versions.
        model = model_id
    return model.__new__(model)


model_unpickle.__safe_for_unpickle__ = True
```
### 29 - django/db/models/base.py:

Start line: 1807, End line: 1842

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_property_name_related_field_accessor_clashes(cls):
        errors = []
        property_names = cls._meta._property_names
        related_field_accessors = (
            f.get_attname()
            for f in cls._meta._get_fields(reverse=False)
            if f.is_relation and f.related_model is not None
        )
        for accessor in related_field_accessors:
            if accessor in property_names:
                errors.append(
                    checks.Error(
                        "The property '%s' clashes with a related field "
                        "accessor." % accessor,
                        obj=cls,
                        id="models.E025",
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
                    id="models.E026",
                )
            )
        return errors

    # RemovedInDjango51Warning.
```
### 48 - django/db/models/base.py:

Start line: 1296, End line: 1343

```python
class Model(AltersData, metaclass=ModelBase):

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
                if lookup_value is None or (
                    lookup_value == ""
                    and connection.features.interprets_empty_strings_as_nulls
                ):
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
                errors.setdefault(key, []).append(
                    self.unique_error_message(model_class, unique_check)
                )

        return errors
```
### 60 - django/db/models/base.py:

Start line: 1345, End line: 1374

```python
class Model(AltersData, metaclass=ModelBase):

    def _perform_date_checks(self, date_checks):
        errors = {}
        for model_class, lookup_type, field, unique_for in date_checks:
            lookup_kwargs = {}
            # there's a ticket to add a date lookup, we can remove this special
            # case if that makes it's way in
            date = getattr(self, unique_for)
            if date is None:
                continue
            if lookup_type == "date":
                lookup_kwargs["%s__day" % unique_for] = date.day
                lookup_kwargs["%s__month" % unique_for] = date.month
                lookup_kwargs["%s__year" % unique_for] = date.year
            else:
                lookup_kwargs["%s__%s" % (unique_for, lookup_type)] = getattr(
                    date, lookup_type
                )
            lookup_kwargs[field] = getattr(self, field)

            qs = model_class._default_manager.filter(**lookup_kwargs)
            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            if not self._state.adding and self.pk is not None:
                qs = qs.exclude(pk=self.pk)

            if qs.exists():
                errors.setdefault(field, []).append(
                    self.date_error_message(lookup_type, field, unique_for)
                )
        return errors
```
### 64 - django/db/models/base.py:

Start line: 2254, End line: 2445

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_constraints(cls, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, cls):
                continue
            connection = connections[db]
            if not (
                connection.features.supports_table_check_constraints
                or "supports_table_check_constraints" in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, CheckConstraint)
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        "%s does not support check constraints."
                        % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id="models.W027",
                    )
                )
            if not (
                connection.features.supports_partial_indexes
                or "supports_partial_indexes" in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, UniqueConstraint)
                and constraint.condition is not None
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        "%s does not support unique constraints with "
                        "conditions." % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id="models.W036",
                    )
                )
            if not (
                connection.features.supports_deferrable_unique_constraints
                or "supports_deferrable_unique_constraints"
                in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, UniqueConstraint)
                and constraint.deferrable is not None
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        "%s does not support deferrable unique constraints."
                        % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id="models.W038",
                    )
                )
            if not (
                connection.features.supports_covering_indexes
                or "supports_covering_indexes" in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, UniqueConstraint) and constraint.include
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        "%s does not support unique constraints with non-key "
                        "columns." % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id="models.W039",
                    )
                )
            if not (
                connection.features.supports_expression_indexes
                or "supports_expression_indexes" in cls._meta.required_db_features
            ) and any(
                isinstance(constraint, UniqueConstraint)
                and constraint.contains_expressions
                for constraint in cls._meta.constraints
            ):
                errors.append(
                    checks.Warning(
                        "%s does not support unique constraints on "
                        "expressions." % connection.display_name,
                        hint=(
                            "A constraint won't be created. Silence this "
                            "warning if you don't care about it."
                        ),
                        obj=cls,
                        id="models.W044",
                    )
                )
            fields = set(
                chain.from_iterable(
                    (*constraint.fields, *constraint.include)
                    for constraint in cls._meta.constraints
                    if isinstance(constraint, UniqueConstraint)
                )
            )
            references = set()
            for constraint in cls._meta.constraints:
                if isinstance(constraint, UniqueConstraint):
                    if (
                        connection.features.supports_partial_indexes
                        or "supports_partial_indexes"
                        not in cls._meta.required_db_features
                    ) and isinstance(constraint.condition, Q):
                        references.update(
                            cls._get_expr_references(constraint.condition)
                        )
                    if (
                        connection.features.supports_expression_indexes
                        or "supports_expression_indexes"
                        not in cls._meta.required_db_features
                    ) and constraint.contains_expressions:
                        for expression in constraint.expressions:
                            references.update(cls._get_expr_references(expression))
                elif isinstance(constraint, CheckConstraint):
                    if (
                        connection.features.supports_table_check_constraints
                        or "supports_table_check_constraints"
                        not in cls._meta.required_db_features
                    ):
                        if isinstance(constraint.check, Q):
                            references.update(
                                cls._get_expr_references(constraint.check)
                            )
                        if any(
                            isinstance(expr, RawSQL)
                            for expr in constraint.check.flatten()
                        ):
                            errors.append(
                                checks.Warning(
                                    f"Check constraint {constraint.name!r} contains "
                                    f"RawSQL() expression and won't be validated "
                                    f"during the model full_clean().",
                                    hint=(
                                        "Silence this warning if you don't care about "
                                        "it."
                                    ),
                                    obj=cls,
                                    id="models.W045",
                                ),
                            )
            for field_name, *lookups in references:
                # pk is an alias that won't be found by opts.get_field.
                if field_name != "pk":
                    fields.add(field_name)
                if not lookups:
                    # If it has no lookups it cannot result in a JOIN.
                    continue
                try:
                    if field_name == "pk":
                        field = cls._meta.pk
                    else:
                        field = cls._meta.get_field(field_name)
                    if not field.is_relation or field.many_to_many or field.one_to_many:
                        continue
                except FieldDoesNotExist:
                    continue
                # JOIN must happen at the first lookup.
                first_lookup = lookups[0]
                if (
                    hasattr(field, "get_transform")
                    and hasattr(field, "get_lookup")
                    and field.get_transform(first_lookup) is None
                    and field.get_lookup(first_lookup) is None
                ):
                    errors.append(
                        checks.Error(
                            "'constraints' refers to the joined field '%s'."
                            % LOOKUP_SEP.join([field_name] + lookups),
                            obj=cls,
                            id="models.E041",
                        )
                    )
            errors.extend(cls._check_local_fields(fields, "constraints"))
        return errors
```
### 67 - django/db/models/base.py:

Start line: 459, End line: 572

```python
class Model(AltersData, metaclass=ModelBase):
    def __init__(self, *args, **kwargs):
        # Alias some things as locals to avoid repeat global lookups
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED
        if opts.abstract:
            raise TypeError("Abstract models cannot be instantiated.")

        pre_init.send(sender=cls, args=args, kwargs=kwargs)

        # Set up the storage for instance state
        self._state = ModelState()

        # There is a rather weird disparity here; if kwargs, it's set, then args
        # overrides it. It should be one or the other; don't duplicate the work
        # The reason for the kwargs check is that standard iterator passes in by
        # args, and instantiation for iteration is 33% faster.
        if len(args) > len(opts.concrete_fields):
            # Daft, but matches old exception sans the err msg.
            raise IndexError("Number of args exceeds number of fields")

        if not kwargs:
            fields_iter = iter(opts.concrete_fields)
            # The ordering of the zip calls matter - zip throws StopIteration
            # when an iter throws it. So if the first iter throws it, the second
            # is *not* consumed. We rely on this, so don't change the order
            # without changing the logic.
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
        else:
            # Slower, kwargs-ready version.
            fields_iter = iter(opts.fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
                if kwargs.pop(field.name, NOT_PROVIDED) is not NOT_PROVIDED:
                    raise TypeError(
                        f"{cls.__qualname__}() got both positional and "
                        f"keyword arguments for field '{field.name}'."
                    )

        # Now we're left with the unprocessed fields that *must* come from
        # keywords, or default.

        for field in fields_iter:
            is_related_object = False
            # Virtual field
            if field.attname not in kwargs and field.column is None:
                continue
            if kwargs:
                if isinstance(field.remote_field, ForeignObjectRel):
                    try:
                        # Assume object instance was passed in.
                        rel_obj = kwargs.pop(field.name)
                        is_related_object = True
                    except KeyError:
                        try:
                            # Object instance wasn't passed in -- must be an ID.
                            val = kwargs.pop(field.attname)
                        except KeyError:
                            val = field.get_default()
                else:
                    try:
                        val = kwargs.pop(field.attname)
                    except KeyError:
                        # This is done with an exception rather than the
                        # default argument on pop because we don't want
                        # get_default() to be evaluated, and then not used.
                        # Refs #12057.
                        val = field.get_default()
            else:
                val = field.get_default()

            if is_related_object:
                # If we are passed a related instance, set it using the
                # field.name instead of field.attname (e.g. "user" instead of
                # "user_id") so that the object gets properly cached (and type
                # checked) by the RelatedObjectDescriptor.
                if rel_obj is not _DEFERRED:
                    _setattr(self, field.name, rel_obj)
            else:
                if val is not _DEFERRED:
                    _setattr(self, field.attname, val)

        if kwargs:
            property_names = opts._property_names
            unexpected = ()
            for prop, value in kwargs.items():
                # Any remaining kwargs must correspond to properties or virtual
                # fields.
                if prop in property_names:
                    if value is not _DEFERRED:
                        _setattr(self, prop, value)
                else:
                    try:
                        opts.get_field(prop)
                    except FieldDoesNotExist:
                        unexpected += (prop,)
                    else:
                        if value is not _DEFERRED:
                            _setattr(self, prop, value)
            if unexpected:
                unexpected_names = ", ".join(repr(n) for n in unexpected)
                raise TypeError(
                    f"{cls.__name__}() got unexpected keyword arguments: "
                    f"{unexpected_names}"
                )
        super().__init__()
        post_init.send(sender=cls, instance=self)
```
### 72 - django/db/models/base.py:

Start line: 1521, End line: 1556

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def check(cls, **kwargs):
        errors = [
            *cls._check_swappable(),
            *cls._check_model(),
            *cls._check_managers(**kwargs),
        ]
        if not cls._meta.swapped:
            databases = kwargs.get("databases") or []
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
### 74 - django/db/models/base.py:

Start line: 1118, End line: 1145

```python
class Model(AltersData, metaclass=ModelBase):

    def delete(self, using=None, keep_parents=False):
        if self.pk is None:
            raise ValueError(
                "%s object can't be deleted because its %s attribute is set "
                "to None." % (self._meta.object_name, self._meta.pk.attname)
            )
        using = using or router.db_for_write(self.__class__, instance=self)
        collector = Collector(using=using, origin=self)
        collector.collect([self], keep_parents=keep_parents)
        return collector.delete()

    delete.alters_data = True

    async def adelete(self, using=None, keep_parents=False):
        return await sync_to_async(self.delete)(
            using=using,
            keep_parents=keep_parents,
        )

    adelete.alters_data = True

    def _get_FIELD_display(self, field):
        value = getattr(self, field.attname)
        choices_dict = dict(make_hashable(field.flatchoices))
        # force_str() to coerce lazy strings.
        return force_str(
            choices_dict.get(make_hashable(value), value), strings_only=True
        )
```
### 81 - django/db/models/base.py:

Start line: 1191, End line: 1231

```python
class Model(AltersData, metaclass=ModelBase):

    def _get_field_value_map(self, meta, exclude=None):
        if exclude is None:
            exclude = set()
        meta = meta or self._meta
        return {
            field.name: Value(getattr(self, field.attname), field)
            for field in meta.local_concrete_fields
            if field.name not in exclude
        }

    def prepare_database_save(self, field):
        if self.pk is None:
            raise ValueError(
                "Unsaved model instance %r cannot be used in an ORM query." % self
            )
        return getattr(self, field.remote_field.get_related_field().attname)

    def clean(self):
        """
        Hook for doing any extra model-wide validation after clean() has been
        called on every field by self.clean_fields. Any ValidationError raised
        by this method will not be associated with a particular field; it will
        have a special-case association with the field defined by NON_FIELD_ERRORS.
        """
        pass

    def validate_unique(self, exclude=None):
        """
        Check unique constraints on the model and raise ValidationError if any
        failed.
        """
        unique_checks, date_checks = self._get_unique_checks(exclude=exclude)

        errors = self._perform_unique_checks(unique_checks)
        date_errors = self._perform_date_checks(date_checks)

        for k, v in date_errors.items():
            errors.setdefault(k, []).extend(v)

        if errors:
            raise ValidationError(errors)
```
### 82 - django/db/models/base.py:

Start line: 1993, End line: 2046

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_local_fields(cls, fields, option):
        from django.db import models

        # In order to avoid hitting the relation tree prematurely, we use our
        # own fields_map instead of using get_field()
        forward_fields_map = {}
        for field in cls._meta._get_fields(reverse=False):
            forward_fields_map[field.name] = field
            if hasattr(field, "attname"):
                forward_fields_map[field.attname] = field

        errors = []
        for field_name in fields:
            try:
                field = forward_fields_map[field_name]
            except KeyError:
                errors.append(
                    checks.Error(
                        "'%s' refers to the nonexistent field '%s'."
                        % (
                            option,
                            field_name,
                        ),
                        obj=cls,
                        id="models.E012",
                    )
                )
            else:
                if isinstance(field.remote_field, models.ManyToManyRel):
                    errors.append(
                        checks.Error(
                            "'%s' refers to a ManyToManyField '%s', but "
                            "ManyToManyFields are not permitted in '%s'."
                            % (
                                option,
                                field_name,
                                option,
                            ),
                            obj=cls,
                            id="models.E013",
                        )
                    )
                elif field not in cls._meta.local_fields:
                    errors.append(
                        checks.Error(
                            "'%s' refers to field '%s' which is not local to model "
                            "'%s'." % (option, field_name, cls._meta.object_name),
                            hint="This issue may be caused by multi-table inheritance.",
                            obj=cls,
                            id="models.E016",
                        )
                    )
        return errors
```
### 86 - django/db/models/base.py:

Start line: 1648, End line: 1682

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_m2m_through_same_relationship(cls):
        """Check if no relationship model is used by more than one m2m field."""

        errors = []
        seen_intermediary_signatures = []

        fields = cls._meta.local_many_to_many

        # Skip when the target model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.model, ModelBase))

        # Skip when the relationship model wasn't found.
        fields = (f for f in fields if isinstance(f.remote_field.through, ModelBase))

        for f in fields:
            signature = (
                f.remote_field.model,
                cls,
                f.remote_field.through,
                f.remote_field.through_fields,
            )
            if signature in seen_intermediary_signatures:
                errors.append(
                    checks.Error(
                        "The model has two identical many-to-many relations "
                        "through the intermediate model '%s'."
                        % f.remote_field.through._meta.label,
                        obj=cls,
                        id="models.E003",
                    )
                )
            else:
                seen_intermediary_signatures.append(signature)
        return errors
```
### 110 - django/db/models/base.py:

Start line: 1703, End line: 1756

```python
class Model(AltersData, metaclass=ModelBase):

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
                            "from parent model '%s'."
                            % (clash.name, clash.model._meta, f.name, f.model._meta),
                            obj=cls,
                            id="models.E005",
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
            id_conflict = (
                f.name == "id" and clash and clash.name == "id" and clash.model == cls
            )
            if clash and not id_conflict:
                errors.append(
                    checks.Error(
                        "The field '%s' clashes with the field '%s' "
                        "from model '%s'." % (f.name, clash.name, clash.model._meta),
                        obj=f,
                        id="models.E006",
                    )
                )
            used_fields[f.name] = f
            used_fields[f.attname] = f

        return errors
```
### 113 - django/db/models/base.py:

Start line: 574, End line: 612

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def from_db(cls, db, field_names, values):
        if len(values) != len(cls._meta.concrete_fields):
            values_iter = iter(values)
            values = [
                next(values_iter) if f.attname in field_names else DEFERRED
                for f in cls._meta.concrete_fields
            ]
        new = cls(*values)
        new._state.adding = False
        new._state.db = db
        return new

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    def __str__(self):
        return "%s object (%s)" % (self.__class__.__name__, self.pk)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        if self._meta.concrete_model != other._meta.concrete_model:
            return False
        my_pk = self.pk
        if my_pk is None:
            return self is other
        return my_pk == other.pk

    def __hash__(self):
        if self.pk is None:
            raise TypeError("Model instances without primary key value are unhashable")
        return hash(self.pk)

    def __reduce__(self):
        data = self.__getstate__()
        data[DJANGO_VERSION_PICKLE_KEY] = django.__version__
        class_id = self._meta.app_label, self._meta.object_name
        return model_unpickle, (class_id,), data
```
### 119 - django/db/models/base.py:

Start line: 1558, End line: 1588

```python
class Model(AltersData, metaclass=ModelBase):

    @classmethod
    def _check_default_pk(cls):
        if (
            not cls._meta.abstract
            and cls._meta.pk.auto_created
            and
            # Inherited PKs are checked in parents models.
            not (
                isinstance(cls._meta.pk, OneToOneField)
                and cls._meta.pk.remote_field.parent_link
            )
            and not settings.is_overridden("DEFAULT_AUTO_FIELD")
            and cls._meta.app_config
            and not cls._meta.app_config._is_default_auto_field_overridden
        ):
            return [
                checks.Warning(
                    f"Auto-created primary key used when not defining a "
                    f"primary key type, by default "
                    f"'{settings.DEFAULT_AUTO_FIELD}'.",
                    hint=(
                        f"Configure the DEFAULT_AUTO_FIELD setting or the "
                        f"{cls._meta.app_config.__class__.__qualname__}."
                        f"default_auto_field attribute to point to a subclass "
                        f"of AutoField, e.g. 'django.db.models.BigAutoField'."
                    ),
                    obj=cls,
                    id="models.W042",
                ),
            ]
        return []
```
### 133 - django/db/models/base.py:

Start line: 1, End line: 66

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

from asgiref.sync import sync_to_async

import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
    NON_FIELD_ERRORS,
    FieldDoesNotExist,
    FieldError,
    MultipleObjectsReturned,
    ObjectDoesNotExist,
    ValidationError,
)
from django.db import (
    DJANGO_VERSION_PICKLE_KEY,
    DatabaseError,
    connection,
    connections,
    router,
    transaction,
)
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import RawSQL
from django.db.models.fields.related import (
    ForeignObjectRel,
    OneToOneField,
    lazy_related_operation,
    resolve_relation,
)
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
    class_prepared,
    post_init,
    post_save,
    pre_init,
    pre_save,
)
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _


class Deferred:
    def __repr__(self):
        return "<Deferred field>"

    def __str__(self):
        return "<Deferred field>"


DEFERRED = Deferred()
```
### 134 - django/db/models/base.py:

Start line: 250, End line: 367

```python
class ModelBase(type):

    def __new__(cls, name, bases, attrs, **kwargs):
        # ... other code
        for base in new_class.mro():
            if base not in parents or not hasattr(base, "_meta"):
                # Things without _meta aren't functional models, so they're
                # uninteresting parents.
                inherited_attributes.update(base.__dict__)
                continue

            parent_fields = base._meta.local_fields + base._meta.local_many_to_many
            if not base._meta.abstract:
                # Check for clashes between locally declared fields and those
                # on the base classes.
                for field in parent_fields:
                    if field.name in field_names:
                        raise FieldError(
                            "Local field %r in class %r clashes with field of "
                            "the same name from base class %r."
                            % (
                                field.name,
                                name,
                                base.__name__,
                            )
                        )
                    else:
                        inherited_attributes.add(field.name)

                # Concrete classes...
                base = base._meta.concrete_model
                base_key = make_model_tuple(base)
                if base_key in parent_links:
                    field = parent_links[base_key]
                elif not is_proxy:
                    attr_name = "%s_ptr" % base._meta.model_name
                    field = OneToOneField(
                        base,
                        on_delete=CASCADE,
                        name=attr_name,
                        auto_created=True,
                        parent_link=True,
                    )

                    if attr_name in field_names:
                        raise FieldError(
                            "Auto-generated field '%s' in class %r for "
                            "parent_link to base class %r clashes with "
                            "declared field of the same name."
                            % (
                                attr_name,
                                name,
                                base.__name__,
                            )
                        )

                    # Only add the ptr field if it's not already present;
                    # e.g. migrations will already have it specified
                    if not hasattr(new_class, attr_name):
                        new_class.add_to_class(attr_name, field)
                else:
                    field = None
                new_class._meta.parents[base] = field
            else:
                base_parents = base._meta.parents.copy()

                # Add fields from abstract base class if it wasn't overridden.
                for field in parent_fields:
                    if (
                        field.name not in field_names
                        and field.name not in new_class.__dict__
                        and field.name not in inherited_attributes
                    ):
                        new_field = copy.deepcopy(field)
                        new_class.add_to_class(field.name, new_field)
                        # Replace parent links defined on this base by the new
                        # field. It will be appropriately resolved if required.
                        if field.one_to_one:
                            for parent, parent_link in base_parents.items():
                                if field == parent_link:
                                    base_parents[parent] = new_field

                # Pass any non-abstract parent classes onto child.
                new_class._meta.parents.update(base_parents)

            # Inherit private fields (like GenericForeignKey) from the parent
            # class
            for field in base._meta.private_fields:
                if field.name in field_names:
                    if not base._meta.abstract:
                        raise FieldError(
                            "Local field %r in class %r clashes with field of "
                            "the same name from base class %r."
                            % (
                                field.name,
                                name,
                                base.__name__,
                            )
                        )
                else:
                    field = copy.deepcopy(field)
                    if not base._meta.abstract:
                        field.mti_inherited = True
                    new_class.add_to_class(field.name, field)

        # Copy indexes so that index names are unique when models extend an
        # abstract model.
        new_class._meta.indexes = [
            copy.deepcopy(idx) for idx in new_class._meta.indexes
        ]

        if abstract:
            # Abstract base models can't be instantiated and don't appear in
            # the list of models for an app. We do the final setup for them a
            # little differently from normal models.
            attr_meta.abstract = False
            new_class.Meta = attr_meta
            return new_class

        new_class._prepare()
        new_class._meta.apps.register_model(new_class._meta.app_label, new_class)
        return new_class
```
