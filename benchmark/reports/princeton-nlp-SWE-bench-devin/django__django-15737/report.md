# django__django-15737

| **django/django** | `f1e0fc645bb0b2c15d1510c9a8501743297dec9d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 520 |
| **Any found context length** | 520 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -1100,9 +1100,9 @@ def _prepare_related_fields_for_save(self, operation_name, fields=None):
                         "related object '%s'." % (operation_name, field.name)
                     )
                 elif getattr(self, field.attname) in field.empty_values:
-                    # Use pk from related object if it has been saved after
-                    # an assignment.
-                    setattr(self, field.attname, obj.pk)
+                    # Set related object if it has been saved after an
+                    # assignment.
+                    setattr(self, field.name, obj)
                 # If the relationship's pk/to_field was changed, clear the
                 # cached relationship.
                 if getattr(obj, field.target_field.attname) != getattr(

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/base.py | 1103 | 1105 | 1 | 1 | 520


## Problem Statement

```
Avoid unnecessary clear of cached reference
Description
	 
		(last modified by Barry Johnson)
	 
Consider this case of ORM models "Parent" and "Child", where Child has a foreign key reference to Parent (and the database can return generated IDs following insert operations):
parent = Parent(name='parent_object')
child = Child(parent=parent)
parent.save()
child.save()
print(child.parent.name)
The print statement will cause an unnecessary lazy read of the parent object.
In the application where this behavior was first observed, the application was creating thousands of parent and child objects using bulk_create(). The subsequent lazy reads occurred when creating log entries to record the action, and added thousands of unwanted SELECT queries.
Closed ticket #29497 solved a problem with potential data loss in this situation by essentially executing child.parent_id = child.parent.pk while preparing the child object to be saved. However, when the child's ForeignKeyDeferredAttrbute "parent_id" changes value from None to the parent's ID, the child's internal cache containing the reference to "parent" is cleared. The subsequent reference to child.parent then must do a lazy read and reload parent from the database.
A workaround to avoid this lazy read is to explicitly update both the "parent_id" and "parent" cache entry by adding this non-intuitive statement:
child.parent = child.parent
after executing parent.save()
But it appears that a simple change could avoid clearing the cache in this narrow case.
Within Model._prepare_related_fields_for_save(), replace
setattr(self, field.attname, obj.pk)
with
setattr(self, field.name, obj)
This suggested code has -not- been tested.
This change would set the associated "parent_id" attribute while ensuring that the currently referenced object remains referenced.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/db/models/base.py** | 1074 | 1126| 520 | 520 | 18545 | 
| 2 | 2 django/db/models/query.py | 2188 | 2322| 1129 | 1649 | 38718 | 
| 3 | 3 django/db/models/fields/related.py | 302 | 339| 296 | 1945 | 53330 | 
| 4 | 4 django/db/models/fields/related_descriptors.py | 1 | 81| 687 | 2632 | 64393 | 
| 5 | 4 django/db/models/fields/related.py | 870 | 897| 237 | 2869 | 64393 | 
| 6 | 5 django/db/models/fields/reverse_related.py | 185 | 203| 167 | 3036 | 66925 | 
| 7 | 5 django/db/models/fields/related.py | 154 | 185| 209 | 3245 | 66925 | 
| 8 | 5 django/db/models/fields/related.py | 1017 | 1053| 261 | 3506 | 66925 | 
| 9 | 5 django/db/models/fields/reverse_related.py | 153 | 163| 130 | 3636 | 66925 | 
| 10 | 5 django/db/models/fields/related.py | 341 | 378| 297 | 3933 | 66925 | 
| 11 | 5 django/db/models/fields/related.py | 1449 | 1578| 984 | 4917 | 66925 | 
| 12 | 5 django/db/models/fields/related.py | 991 | 1015| 176 | 5093 | 66925 | 
| 13 | 5 django/db/models/fields/related_descriptors.py | 124 | 168| 418 | 5511 | 66925 | 
| 14 | 5 django/db/models/fields/related.py | 603 | 669| 497 | 6008 | 66925 | 
| 15 | 5 django/db/models/fields/related.py | 732 | 754| 172 | 6180 | 66925 | 
| 16 | 5 django/db/models/fields/related.py | 208 | 224| 142 | 6322 | 66925 | 
| 17 | 5 django/db/models/fields/related.py | 126 | 152| 171 | 6493 | 66925 | 
| 18 | 5 django/db/models/fields/related.py | 1078 | 1100| 180 | 6673 | 66925 | 
| 19 | 6 django/db/models/fields/related_lookups.py | 74 | 108| 379 | 7052 | 68528 | 
| 20 | 6 django/db/models/fields/related.py | 187 | 206| 155 | 7207 | 68528 | 
| 21 | 6 django/db/models/fields/reverse_related.py | 20 | 151| 765 | 7972 | 68528 | 
| 22 | 7 django/core/cache/backends/db.py | 257 | 295| 356 | 8328 | 70672 | 
| 23 | 7 django/db/models/fields/related.py | 756 | 774| 166 | 8494 | 70672 | 
| 24 | **7 django/db/models/base.py** | 2449 | 2501| 341 | 8835 | 70672 | 
| 25 | 7 django/db/models/fields/related.py | 1121 | 1157| 284 | 9119 | 70672 | 
| 26 | 7 django/db/models/fields/related_descriptors.py | 375 | 396| 158 | 9277 | 70672 | 
| 27 | 7 django/db/models/fields/related_descriptors.py | 221 | 303| 760 | 10037 | 70672 | 
| 28 | **7 django/db/models/base.py** | 906 | 941| 318 | 10355 | 70672 | 
| 29 | 7 django/db/models/fields/related.py | 226 | 301| 696 | 11051 | 70672 | 
| 30 | 7 django/db/models/fields/related.py | 706 | 730| 210 | 11261 | 70672 | 
| 31 | 7 django/db/models/fields/related.py | 671 | 704| 335 | 11596 | 70672 | 
| 32 | **7 django/db/models/base.py** | 838 | 904| 475 | 12071 | 70672 | 
| 33 | 7 django/db/models/fields/related_descriptors.py | 398 | 421| 194 | 12265 | 70672 | 
| 34 | 7 django/db/models/fields/related_descriptors.py | 84 | 122| 266 | 12531 | 70672 | 
| 35 | 7 django/db/models/fields/related.py | 1102 | 1119| 133 | 12664 | 70672 | 
| 36 | 7 django/db/models/fields/related_lookups.py | 151 | 168| 216 | 12880 | 70672 | 
| 37 | 7 django/db/models/fields/reverse_related.py | 205 | 238| 306 | 13186 | 70672 | 
| 38 | 7 django/db/models/query.py | 2560 | 2583| 201 | 13387 | 70672 | 
| 39 | 8 django/db/models/options.py | 816 | 831| 144 | 13531 | 78174 | 
| 40 | 8 django/db/models/fields/related.py | 581 | 601| 138 | 13669 | 78174 | 
| 41 | 9 django/db/backends/base/schema.py | 38 | 71| 214 | 13883 | 91971 | 
| 42 | 9 django/db/models/query.py | 2455 | 2487| 314 | 14197 | 91971 | 
| 43 | 10 django/db/models/deletion.py | 310 | 370| 616 | 14813 | 95903 | 
| 44 | 10 django/db/models/fields/related_descriptors.py | 770 | 845| 592 | 15405 | 95903 | 
| 45 | 11 django/core/serializers/base.py | 297 | 322| 224 | 15629 | 98579 | 
| 46 | 12 django/contrib/contenttypes/fields.py | 223 | 265| 384 | 16013 | 104206 | 
| 47 | 12 django/core/cache/backends/db.py | 206 | 233| 280 | 16293 | 104206 | 
| 48 | 12 django/db/models/query.py | 2159 | 2187| 246 | 16539 | 104206 | 
| 49 | 12 django/core/cache/backends/db.py | 113 | 204| 809 | 17348 | 104206 | 
| 50 | **12 django/db/models/base.py** | 477 | 590| 953 | 18301 | 104206 | 
| 51 | 12 django/db/models/fields/reverse_related.py | 1 | 17| 120 | 18421 | 104206 | 
| 52 | 12 django/db/models/fields/related.py | 1417 | 1447| 172 | 18593 | 104206 | 
| 53 | 12 django/db/models/fields/related.py | 1159 | 1174| 136 | 18729 | 104206 | 
| 54 | 12 django/db/models/fields/related.py | 1881 | 1928| 505 | 19234 | 104206 | 
| 55 | 12 django/db/models/fields/related.py | 1176 | 1208| 246 | 19480 | 104206 | 
| 56 | 12 django/db/models/fields/related.py | 513 | 579| 367 | 19847 | 104206 | 
| 57 | 12 django/db/models/fields/related_lookups.py | 110 | 148| 324 | 20171 | 104206 | 
| 58 | 12 django/db/models/fields/related.py | 900 | 989| 583 | 20754 | 104206 | 
| 59 | 12 django/db/models/fields/related_descriptors.py | 170 | 219| 452 | 21206 | 104206 | 
| 60 | **12 django/db/models/base.py** | 1809 | 1842| 232 | 21438 | 104206 | 
| 61 | **12 django/db/models/base.py** | 2255 | 2446| 1302 | 22740 | 104206 | 
| 62 | 12 django/db/models/fields/related_descriptors.py | 337 | 354| 188 | 22928 | 104206 | 
| 63 | 12 django/db/models/fields/related.py | 484 | 510| 177 | 23105 | 104206 | 
| 64 | 12 django/db/models/fields/related_descriptors.py | 544 | 579| 245 | 23350 | 104206 | 
| 65 | 13 django/db/models/sql/compiler.py | 1088 | 1193| 871 | 24221 | 119712 | 
| 66 | 13 django/db/models/fields/related.py | 1580 | 1678| 655 | 24876 | 119712 | 
| 67 | 13 django/db/models/fields/related.py | 1680 | 1730| 431 | 25307 | 119712 | 
| 68 | 13 django/db/models/fields/related.py | 1959 | 1993| 266 | 25573 | 119712 | 
| 69 | 13 django/db/models/fields/related.py | 1 | 40| 251 | 25824 | 119712 | 
| 70 | 13 django/core/cache/backends/db.py | 235 | 255| 246 | 26070 | 119712 | 
| 71 | 13 django/core/cache/backends/db.py | 42 | 99| 430 | 26500 | 119712 | 
| 72 | 14 django/db/models/query_utils.py | 155 | 193| 300 | 26800 | 122558 | 
| 73 | 15 django/db/migrations/autodetector.py | 600 | 774| 1213 | 28013 | 135921 | 
| 74 | 15 django/db/models/query.py | 2384 | 2453| 665 | 28678 | 135921 | 
| 75 | 15 django/contrib/contenttypes/fields.py | 744 | 772| 254 | 28932 | 135921 | 
| 76 | 15 django/db/models/fields/related_descriptors.py | 581 | 609| 220 | 29152 | 135921 | 
| 77 | 15 django/contrib/contenttypes/fields.py | 174 | 221| 417 | 29569 | 135921 | 
| 78 | 15 django/contrib/contenttypes/fields.py | 657 | 691| 284 | 29853 | 135921 | 
| 79 | 15 django/db/models/fields/related_descriptors.py | 694 | 731| 341 | 30194 | 135921 | 
| 80 | 15 django/contrib/contenttypes/fields.py | 693 | 720| 214 | 30408 | 135921 | 
| 81 | 15 django/contrib/contenttypes/fields.py | 722 | 742| 188 | 30596 | 135921 | 
| 82 | 15 django/db/models/options.py | 776 | 814| 394 | 30990 | 135921 | 
| 83 | **15 django/db/models/base.py** | 1 | 64| 346 | 31336 | 135921 | 
| 84 | 16 django/db/models/sql/query.py | 721 | 802| 833 | 32169 | 159068 | 
| 85 | 16 django/contrib/contenttypes/fields.py | 471 | 498| 258 | 32427 | 159068 | 
| 86 | 16 django/db/models/fields/related.py | 818 | 868| 377 | 32804 | 159068 | 
| 87 | 16 django/db/models/deletion.py | 1 | 86| 569 | 33373 | 159068 | 
| 88 | 16 django/db/models/query.py | 2490 | 2558| 785 | 34158 | 159068 | 
| 89 | 16 django/core/cache/backends/db.py | 101 | 111| 222 | 34380 | 159068 | 
| 90 | 16 django/db/models/fields/related_descriptors.py | 1177 | 1211| 341 | 34721 | 159068 | 
| 91 | 16 django/db/models/fields/related.py | 380 | 401| 219 | 34940 | 159068 | 
| 92 | 16 django/db/models/options.py | 321 | 357| 338 | 35278 | 159068 | 
| 93 | 17 django/db/backends/oracle/creation.py | 159 | 201| 411 | 35689 | 163083 | 
| 94 | 17 django/db/models/fields/related_lookups.py | 170 | 210| 250 | 35939 | 163083 | 
| 95 | 18 django/core/serializers/xml_serializer.py | 102 | 125| 196 | 36135 | 166679 | 
| 96 | 19 django/contrib/admin/utils.py | 192 | 228| 251 | 36386 | 170885 | 
| 97 | 20 django/db/backends/oracle/operations.py | 391 | 437| 385 | 36771 | 177012 | 
| 98 | 20 django/contrib/contenttypes/fields.py | 22 | 111| 560 | 37331 | 177012 | 
| 99 | 20 django/db/models/fields/related.py | 776 | 802| 222 | 37553 | 177012 | 
| 100 | 21 django/core/checks/model_checks.py | 93 | 116| 170 | 37723 | 178822 | 
| 101 | **21 django/db/models/base.py** | 1128 | 1147| 188 | 37911 | 178822 | 
| 102 | 21 django/db/models/fields/related_descriptors.py | 847 | 875| 229 | 38140 | 178822 | 
| 103 | 22 django/contrib/sites/models.py | 79 | 122| 236 | 38376 | 179611 | 
| 104 | 22 django/db/models/deletion.py | 372 | 396| 273 | 38649 | 179611 | 
| 105 | 22 django/db/models/fields/related_descriptors.py | 733 | 768| 264 | 38913 | 179611 | 
| 106 | 22 django/db/migrations/autodetector.py | 809 | 902| 694 | 39607 | 179611 | 
| 107 | 22 django/db/models/options.py | 1 | 54| 319 | 39926 | 179611 | 
| 108 | 22 django/db/models/sql/compiler.py | 1828 | 1889| 588 | 40514 | 179611 | 
| 109 | 22 django/db/models/query.py | 2098 | 2156| 487 | 41001 | 179611 | 
| 110 | 22 django/db/migrations/autodetector.py | 904 | 977| 623 | 41624 | 179611 | 
| 111 | 22 django/contrib/contenttypes/fields.py | 363 | 389| 181 | 41805 | 179611 | 
| 112 | 22 django/contrib/contenttypes/fields.py | 623 | 655| 333 | 42138 | 179611 | 
| 113 | 22 django/db/models/fields/reverse_related.py | 165 | 183| 160 | 42298 | 179611 | 
| 114 | 22 django/db/models/query.py | 287 | 348| 469 | 42767 | 179611 | 
| 115 | 23 django/db/backends/sqlite3/operations.py | 216 | 241| 218 | 42985 | 183102 | 
| 116 | 23 django/db/models/fields/related.py | 1930 | 1957| 305 | 43290 | 183102 | 
| 117 | **23 django/db/models/base.py** | 943 | 1031| 686 | 43976 | 183102 | 
| 118 | 23 django/db/models/sql/query.py | 2312 | 2339| 291 | 44267 | 183102 | 
| 119 | 23 django/db/models/fields/related_descriptors.py | 1045 | 1087| 382 | 44649 | 183102 | 
| 120 | 23 django/core/checks/model_checks.py | 187 | 228| 345 | 44994 | 183102 | 
| 121 | 23 django/db/models/fields/related_descriptors.py | 1368 | 1418| 403 | 45397 | 183102 | 
| 122 | 23 django/db/models/options.py | 277 | 319| 355 | 45752 | 183102 | 
| 123 | 23 django/db/models/fields/related_descriptors.py | 1117 | 1143| 198 | 45950 | 183102 | 
| 124 | 23 django/db/models/fields/related_descriptors.py | 1297 | 1366| 522 | 46472 | 183102 | 
| 125 | 23 django/db/models/options.py | 617 | 645| 231 | 46703 | 183102 | 
| 126 | **23 django/db/models/base.py** | 248 | 365| 874 | 47577 | 183102 | 
| 127 | **23 django/db/models/base.py** | 1298 | 1345| 409 | 47986 | 183102 | 
| 128 | 23 django/db/models/fields/related_descriptors.py | 1145 | 1175| 256 | 48242 | 183102 | 
| 129 | **23 django/db/models/base.py** | 1705 | 1758| 490 | 48732 | 183102 | 
| 130 | 24 django/db/migrations/state.py | 347 | 395| 371 | 49103 | 191266 | 
| 131 | 25 django/contrib/contenttypes/management/commands/remove_stale_contenttypes.py | 34 | 111| 540 | 49643 | 192015 | 
| 132 | 25 django/db/models/fields/related.py | 1211 | 1258| 368 | 50011 | 192015 | 
| 133 | 25 django/db/migrations/state.py | 265 | 289| 238 | 50249 | 192015 | 
| 134 | 25 django/db/backends/base/schema.py | 1 | 35| 217 | 50466 | 192015 | 
| 135 | 26 django/db/backends/ddl_references.py | 138 | 181| 297 | 50763 | 193644 | 
| 136 | **26 django/db/models/base.py** | 1994 | 2047| 355 | 51118 | 193644 | 
| 137 | 26 django/db/models/query.py | 1660 | 1675| 149 | 51267 | 193644 | 
| 138 | **26 django/db/models/base.py** | 776 | 836| 508 | 51775 | 193644 | 
| 139 | **26 django/db/models/base.py** | 1347 | 1376| 290 | 52065 | 193644 | 
| 140 | 26 django/db/models/fields/reverse_related.py | 241 | 300| 386 | 52451 | 193644 | 


### Hint

```
Hi Barry. Thanks for the report. For me, this is just how things are expected to work, so I'm going to say wontfix. I'm happy if you want to take it to the DevelopersMailingList for a wider discussion, but (in general) I'm suspicious of a "simple change" adjusting a long-standing behaviour like this… (Perhaps there's a case to be made on the mailing list though.) To address the example, my first thought is that you should be saving parent before creating child: parent = Parent(name='parent_object') parent.save() child = Child(parent=parent) ... My second is that, you have parent, so print the name from there: print(parent.name) I appreciate you've no-doubt reduced this in order to show the issue more clearly than in a real example, but I'd suspect those thoughts would go back to the more realistic case. However, as I say, the DevelopersMailingList is a better venue to pursue this kind of thing. Thanks.
​Follow up discussion on the mailing list. Reopening based on the discussion there.
Changed the suggested correction based on the discussion within the mailing list.
Replying to Barry Johnson: Changed the suggested correction based on the discussion within the mailing list. Would you like to prepare a patch?
```

## Patch

```diff
diff --git a/django/db/models/base.py b/django/db/models/base.py
--- a/django/db/models/base.py
+++ b/django/db/models/base.py
@@ -1100,9 +1100,9 @@ def _prepare_related_fields_for_save(self, operation_name, fields=None):
                         "related object '%s'." % (operation_name, field.name)
                     )
                 elif getattr(self, field.attname) in field.empty_values:
-                    # Use pk from related object if it has been saved after
-                    # an assignment.
-                    setattr(self, field.attname, obj.pk)
+                    # Set related object if it has been saved after an
+                    # assignment.
+                    setattr(self, field.name, obj)
                 # If the relationship's pk/to_field was changed, clear the
                 # cached relationship.
                 if getattr(obj, field.target_field.attname) != getattr(

```

## Test Patch

```diff
diff --git a/tests/many_to_one/tests.py b/tests/many_to_one/tests.py
--- a/tests/many_to_one/tests.py
+++ b/tests/many_to_one/tests.py
@@ -654,6 +654,16 @@ def test_fk_assignment_and_related_object_cache(self):
         self.assertIsNot(c.parent, p)
         self.assertEqual(c.parent, p)
 
+    def test_save_parent_after_assign(self):
+        category = Category(name="cats")
+        record = Record(category=category)
+        category.save()
+        record.save()
+        category.name = "dogs"
+        with self.assertNumQueries(0):
+            self.assertEqual(category.id, record.category_id)
+            self.assertEqual(category.name, record.category.name)
+
     def test_save_nullable_fk_after_parent(self):
         parent = Parent()
         child = ChildNullableParent(parent=parent)

```


## Code snippets

### 1 - django/db/models/base.py:

Start line: 1074, End line: 1126

```python
class Model(metaclass=ModelBase):

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
                    # Use pk from related object if it has been saved after
                    # an assignment.
                    setattr(self, field.attname, obj.pk)
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
### 2 - django/db/models/query.py:

Start line: 2188, End line: 2322

```python
def prefetch_related_objects(model_instances, *related_lookups):
    # ... other code
    while all_lookups:
        lookup = all_lookups.pop()
        if lookup.prefetch_to in done_queries:
            if lookup.queryset is not None:
                raise ValueError(
                    "'%s' lookup was already seen with a different queryset. "
                    "You may need to adjust the ordering of your lookups."
                    % lookup.prefetch_to
                )

            continue

        # Top level, the list of objects to decorate is the result cache
        # from the primary QuerySet. It won't be for deeper levels.
        obj_list = model_instances

        through_attrs = lookup.prefetch_through.split(LOOKUP_SEP)
        for level, through_attr in enumerate(through_attrs):
            # Prepare main instances
            if not obj_list:
                break

            prefetch_to = lookup.get_current_prefetch_to(level)
            if prefetch_to in done_queries:
                # Skip any prefetching, and any object preparation
                obj_list = done_queries[prefetch_to]
                continue

            # Prepare objects:
            good_objects = True
            for obj in obj_list:
                # Since prefetching can re-use instances, it is possible to have
                # the same instance multiple times in obj_list, so obj might
                # already be prepared.
                if not hasattr(obj, "_prefetched_objects_cache"):
                    try:
                        obj._prefetched_objects_cache = {}
                    except (AttributeError, TypeError):
                        # Must be an immutable object from
                        # values_list(flat=True), for example (TypeError) or
                        # a QuerySet subclass that isn't returning Model
                        # instances (AttributeError), either in Django or a 3rd
                        # party. prefetch_related() doesn't make sense, so quit.
                        good_objects = False
                        break
            if not good_objects:
                break

            # Descend down tree

            # We assume that objects retrieved are homogeneous (which is the premise
            # of prefetch_related), so what applies to first object applies to all.
            first_obj = obj_list[0]
            to_attr = lookup.get_current_to_attr(level)[0]
            prefetcher, descriptor, attr_found, is_fetched = get_prefetcher(
                first_obj, through_attr, to_attr
            )

            if not attr_found:
                raise AttributeError(
                    "Cannot find '%s' on %s object, '%s' is an invalid "
                    "parameter to prefetch_related()"
                    % (
                        through_attr,
                        first_obj.__class__.__name__,
                        lookup.prefetch_through,
                    )
                )

            if level == len(through_attrs) - 1 and prefetcher is None:
                # Last one, this *must* resolve to something that supports
                # prefetching, otherwise there is no point adding it and the
                # developer asking for it has made a mistake.
                raise ValueError(
                    "'%s' does not resolve to an item that supports "
                    "prefetching - this is an invalid parameter to "
                    "prefetch_related()." % lookup.prefetch_through
                )

            obj_to_fetch = None
            if prefetcher is not None:
                obj_to_fetch = [obj for obj in obj_list if not is_fetched(obj)]

            if obj_to_fetch:
                obj_list, additional_lookups = prefetch_one_level(
                    obj_to_fetch,
                    prefetcher,
                    lookup,
                    level,
                )
                # We need to ensure we don't keep adding lookups from the
                # same relationships to stop infinite recursion. So, if we
                # are already on an automatically added lookup, don't add
                # the new lookups from relationships we've seen already.
                if not (
                    prefetch_to in done_queries
                    and lookup in auto_lookups
                    and descriptor in followed_descriptors
                ):
                    done_queries[prefetch_to] = obj_list
                    new_lookups = normalize_prefetch_lookups(
                        reversed(additional_lookups), prefetch_to
                    )
                    auto_lookups.update(new_lookups)
                    all_lookups.extend(new_lookups)
                followed_descriptors.add(descriptor)
            else:
                # Either a singly related object that has already been fetched
                # (e.g. via select_related), or hopefully some other property
                # that doesn't support prefetching but needs to be traversed.

                # We replace the current list of parent objects with the list
                # of related objects, filtering out empty or missing values so
                # that we can continue with nullable or reverse relations.
                new_obj_list = []
                for obj in obj_list:
                    if through_attr in getattr(obj, "_prefetched_objects_cache", ()):
                        # If related objects have been prefetched, use the
                        # cache rather than the object's through_attr.
                        new_obj = list(obj._prefetched_objects_cache.get(through_attr))
                    else:
                        try:
                            new_obj = getattr(obj, through_attr)
                        except exceptions.ObjectDoesNotExist:
                            continue
                    if new_obj is None:
                        continue
                    # We special-case `list` rather than something more generic
                    # like `Iterable` because we don't want to accidentally match
                    # user models that define __iter__.
                    if isinstance(new_obj, list):
                        new_obj_list.extend(new_obj)
                    else:
                        new_obj_list.append(new_obj)
                obj_list = new_obj_list
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
### 4 - django/db/models/fields/related_descriptors.py:

Start line: 1, End line: 81

```python
"""
Accessors for related objects.

When a field defines a relation between two models, each model class provides
an attribute to access related instances of the other model class (unless the
reverse accessor has been disabled with related_name='+').

Accessors are implemented as descriptors in order to customize access and
assignment. This module defines the descriptor classes.

Forward accessors follow foreign keys. Reverse accessors trace them back. For
example, with the following models::

    class Parent(Model):
        pass

    class Child(Model):
        parent = ForeignKey(Parent, related_name='children')

 ``child.parent`` is a forward many-to-one relation. ``parent.children`` is a
reverse many-to-one relation.

There are three types of relations (many-to-one, one-to-one, and many-to-many)
and two directions (forward and reverse) for a total of six combinations.

1. Related instance on the forward side of a many-to-one relation:
   ``ForwardManyToOneDescriptor``.

   Uniqueness of foreign key values is irrelevant to accessing the related
   instance, making the many-to-one and one-to-one cases identical as far as
   the descriptor is concerned. The constraint is checked upstream (unicity
   validation in forms) or downstream (unique indexes in the database).

2. Related instance on the forward side of a one-to-one
   relation: ``ForwardOneToOneDescriptor``.

   It avoids querying the database when accessing the parent link field in
   a multi-table inheritance scenario.

3. Related instance on the reverse side of a one-to-one relation:
   ``ReverseOneToOneDescriptor``.

   One-to-one relations are asymmetrical, despite the apparent symmetry of the
   name, because they're implemented in the database with a foreign key from
   one table to another. As a consequence ``ReverseOneToOneDescriptor`` is
   slightly different from ``ForwardManyToOneDescriptor``.

4. Related objects manager for related instances on the reverse side of a
   many-to-one relation: ``ReverseManyToOneDescriptor``.

   Unlike the previous two classes, this one provides access to a collection
   of objects. It returns a manager rather than an instance.

5. Related objects manager for related instances on the forward or reverse
   sides of a many-to-many relation: ``ManyToManyDescriptor``.

   Many-to-many relations are symmetrical. The syntax of Django models
   requires declaring them on one side but that's an implementation detail.
   They could be declared on the other side without any change in behavior.
   Therefore the forward and reverse descriptors can be the same.

   If you're looking for ``ForwardManyToManyDescriptor`` or
   ``ReverseManyToManyDescriptor``, use ``ManyToManyDescriptor`` instead.
"""

from django.core.exceptions import FieldError
from django.db import connections, router, transaction
from django.db.models import Q, signals
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import resolve_callables
from django.utils.functional import cached_property


class ForeignKeyDeferredAttribute(DeferredAttribute):
    def __set__(self, instance, value):
        if instance.__dict__.get(self.field.attname) != value and self.field.is_cached(
            instance
        ):
            self.field.delete_cached_value(instance)
        instance.__dict__[self.field.attname] = value
```
### 5 - django/db/models/fields/related.py:

Start line: 870, End line: 897

```python
class ForeignObject(RelatedField):

    def contribute_to_related_class(self, cls, related):
        # Internal FK's - i.e., those with a related name ending with '+' -
        # and swapped models don't get a related descriptor.
        if (
            not self.remote_field.is_hidden()
            and not related.related_model._meta.swapped
        ):
            setattr(
                cls._meta.concrete_model,
                related.get_accessor_name(),
                self.related_accessor_class(related),
            )
            # While 'limit_choices_to' might be a callable, simply pass
            # it along for later - this is too early because it's still
            # model load time.
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(
                    self.remote_field.limit_choices_to
                )


ForeignObject.register_lookup(RelatedIn)
ForeignObject.register_lookup(RelatedExact)
ForeignObject.register_lookup(RelatedLessThan)
ForeignObject.register_lookup(RelatedGreaterThan)
ForeignObject.register_lookup(RelatedGreaterThanOrEqual)
ForeignObject.register_lookup(RelatedLessThanOrEqual)
ForeignObject.register_lookup(RelatedIsNull)
```
### 6 - django/db/models/fields/reverse_related.py:

Start line: 185, End line: 203

```python
class ForeignObjectRel(FieldCacheMixin):

    def is_hidden(self):
        """Should the related object be hidden?"""
        return bool(self.related_name) and self.related_name[-1] == "+"

    def get_joining_columns(self):
        return self.field.get_reverse_joining_columns()

    def get_extra_restriction(self, alias, related_alias):
        return self.field.get_extra_restriction(related_alias, alias)

    def set_field_name(self):
        """
        Set the related field's name, this is not available until later stages
        of app loading, so set_field_name is called from
        set_attributes_from_rel()
        """
        # By default foreign object doesn't relate to any remote field (for
        # example custom multicolumn joins currently have no remote field).
        self.field_name = None
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
### 8 - django/db/models/fields/related.py:

Start line: 1017, End line: 1053

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
### 9 - django/db/models/fields/reverse_related.py:

Start line: 153, End line: 163

```python
class ForeignObjectRel(FieldCacheMixin):

    def __getstate__(self):
        state = self.__dict__.copy()
        # Delete the path_infos cached property because it can be recalculated
        # at first invocation after deserialization. The attribute must be
        # removed because subclasses like ManyToOneRel may have a PathInfo
        # which contains an intermediate M2M table that's been dynamically
        # created and doesn't exist in the .models module.
        # This is a reverse relation, so there is no reverse_path_infos to
        # delete.
        state.pop("path_infos", None)
        return state
```
### 10 - django/db/models/fields/related.py:

Start line: 341, End line: 378

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
                    "class": cls.__name__.lower(),
                    "model_name": cls._meta.model_name.lower(),
                    "app_label": cls._meta.app_label.lower(),
                }
                self.remote_field.related_name = related_name

            if self.remote_field.related_query_name:
                related_query_name = self.remote_field.related_query_name % {
                    "class": cls.__name__.lower(),
                    "app_label": cls._meta.app_label.lower(),
                }
                self.remote_field.related_query_name = related_query_name

            def resolve_related_class(model, related, field):
                field.remote_field.model = related
                field.do_related_class(related, model)

            lazy_related_operation(
                resolve_related_class, cls, self.remote_field.model, field=self
            )
```
### 24 - django/db/models/base.py:

Start line: 2449, End line: 2501

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
### 28 - django/db/models/base.py:

Start line: 906, End line: 941

```python
class Model(metaclass=ModelBase):

    save_base.alters_data = True

    def _save_parents(self, cls, using, update_fields):
        """Save all the parents of cls using values from self."""
        meta = cls._meta
        inserted = False
        for parent, field in meta.parents.items():
            # Make sure the link fields are synced between parent and self.
            if (
                field
                and getattr(self, parent._meta.pk.attname) is None
                and getattr(self, field.attname) is not None
            ):
                setattr(self, parent._meta.pk.attname, getattr(self, field.attname))
            parent_inserted = self._save_parents(
                cls=parent, using=using, update_fields=update_fields
            )
            updated = self._save_table(
                cls=parent,
                using=using,
                update_fields=update_fields,
                force_insert=parent_inserted,
            )
            if not updated:
                inserted = True
            # Set the parent's PK value to self.
            if field:
                setattr(self, field.attname, self._get_pk_val(parent._meta))
                # Since we didn't have an instance of the parent handy set
                # attname directly, bypassing the descriptor. Invalidate
                # the related object cache, in case it's been accidentally
                # populated. A fresh instance will be re-built from the
                # database if necessary.
                if field.is_cached(self):
                    field.delete_cached_value(self)
        return inserted
```
### 32 - django/db/models/base.py:

Start line: 838, End line: 904

```python
class Model(metaclass=ModelBase):

    save.alters_data = True

    def save_base(
        self,
        raw=False,
        force_insert=False,
        force_update=False,
        using=None,
        update_fields=None,
    ):
        """
        Handle the parts of saving which should be done only once per save,
        yet need to be done in raw saves, too. This includes some sanity
        checks and signal sending.

        The 'raw' argument is telling save_base not to save any parent
        models and not to do any changes to the values before save. This
        is used by fixture loading.
        """
        using = using or router.db_for_write(self.__class__, instance=self)
        assert not (force_insert and (force_update or update_fields))
        assert update_fields is None or update_fields
        cls = origin = self.__class__
        # Skip proxies, but keep the origin as the proxy model.
        if cls._meta.proxy:
            cls = cls._meta.concrete_model
        meta = cls._meta
        if not meta.auto_created:
            pre_save.send(
                sender=origin,
                instance=self,
                raw=raw,
                using=using,
                update_fields=update_fields,
            )
        # A transaction isn't needed if one query is issued.
        if meta.parents:
            context_manager = transaction.atomic(using=using, savepoint=False)
        else:
            context_manager = transaction.mark_for_rollback_on_error(using=using)
        with context_manager:
            parent_inserted = False
            if not raw:
                parent_inserted = self._save_parents(cls, using, update_fields)
            updated = self._save_table(
                raw,
                cls,
                force_insert or parent_inserted,
                force_update,
                using,
                update_fields,
            )
        # Store the database on which the object was saved
        self._state.db = using
        # Once saved, this is no longer a to-be-added instance.
        self._state.adding = False

        # Signal that the save is complete
        if not meta.auto_created:
            post_save.send(
                sender=origin,
                instance=self,
                created=(not updated),
                update_fields=update_fields,
                raw=raw,
                using=using,
            )
```
### 50 - django/db/models/base.py:

Start line: 477, End line: 590

```python
class Model(metaclass=ModelBase):
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
### 60 - django/db/models/base.py:

Start line: 1809, End line: 1842

```python
class Model(metaclass=ModelBase):

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
```
### 61 - django/db/models/base.py:

Start line: 2255, End line: 2446

```python
class Model(metaclass=ModelBase):

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
### 83 - django/db/models/base.py:

Start line: 1, End line: 64

```python
import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain

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
from django.db.models.utils import make_model_tuple
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
### 101 - django/db/models/base.py:

Start line: 1128, End line: 1147

```python
class Model(metaclass=ModelBase):

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

    def _get_FIELD_display(self, field):
        value = getattr(self, field.attname)
        choices_dict = dict(make_hashable(field.flatchoices))
        # force_str() to coerce lazy strings.
        return force_str(
            choices_dict.get(make_hashable(value), value), strings_only=True
        )
```
### 117 - django/db/models/base.py:

Start line: 943, End line: 1031

```python
class Model(metaclass=ModelBase):

    def _save_table(
        self,
        raw=False,
        cls=None,
        force_insert=False,
        force_update=False,
        using=None,
        update_fields=None,
    ):
        """
        Do the heavy-lifting involved in saving. Update or insert the data
        for a single table.
        """
        meta = cls._meta
        non_pks = [f for f in meta.local_concrete_fields if not f.primary_key]

        if update_fields:
            non_pks = [
                f
                for f in non_pks
                if f.name in update_fields or f.attname in update_fields
            ]

        pk_val = self._get_pk_val(meta)
        if pk_val is None:
            pk_val = meta.pk.get_pk_value_on_save(self)
            setattr(self, meta.pk.attname, pk_val)
        pk_set = pk_val is not None
        if not pk_set and (force_update or update_fields):
            raise ValueError("Cannot force an update in save() with no primary key.")
        updated = False
        # Skip an UPDATE when adding an instance and primary key has a default.
        if (
            not raw
            and not force_insert
            and self._state.adding
            and meta.pk.default
            and meta.pk.default is not NOT_PROVIDED
        ):
            force_insert = True
        # If possible, try an UPDATE. If that doesn't update anything, do an INSERT.
        if pk_set and not force_insert:
            base_qs = cls._base_manager.using(using)
            values = [
                (
                    f,
                    None,
                    (getattr(self, f.attname) if raw else f.pre_save(self, False)),
                )
                for f in non_pks
            ]
            forced_update = update_fields or force_update
            updated = self._do_update(
                base_qs, using, pk_val, values, update_fields, forced_update
            )
            if force_update and not updated:
                raise DatabaseError("Forced update did not affect any rows.")
            if update_fields and not updated:
                raise DatabaseError("Save with update_fields did not affect any rows.")
        if not updated:
            if meta.order_with_respect_to:
                # If this is a model with an order_with_respect_to
                # autopopulate the _order field
                field = meta.order_with_respect_to
                filter_args = field.get_filter_kwargs_for_object(self)
                self._order = (
                    cls._base_manager.using(using)
                    .filter(**filter_args)
                    .aggregate(
                        _order__max=Coalesce(
                            ExpressionWrapper(
                                Max("_order") + Value(1), output_field=IntegerField()
                            ),
                            Value(0),
                        ),
                    )["_order__max"]
                )
            fields = meta.local_concrete_fields
            if not pk_set:
                fields = [f for f in fields if f is not meta.auto_field]

            returning_fields = meta.db_returning_fields
            results = self._do_insert(
                cls._base_manager, using, fields, returning_fields, raw
            )
            if results:
                for value, field in zip(results[0], returning_fields):
                    setattr(self, field.attname, value)
        return updated
```
### 126 - django/db/models/base.py:

Start line: 248, End line: 365

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
### 127 - django/db/models/base.py:

Start line: 1298, End line: 1345

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
### 129 - django/db/models/base.py:

Start line: 1705, End line: 1758

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
### 136 - django/db/models/base.py:

Start line: 1994, End line: 2047

```python
class Model(metaclass=ModelBase):

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
### 138 - django/db/models/base.py:

Start line: 776, End line: 836

```python
class Model(metaclass=ModelBase):

    def save(
        self, force_insert=False, force_update=False, using=None, update_fields=None
    ):
        """
        Save the current instance. Override this in a subclass if you want to
        control the saving process.

        The 'force_insert' and 'force_update' parameters can be used to insist
        that the "save" must be an SQL insert or update (or equivalent for
        non-SQL backends), respectively. Normally, they should not be set.
        """
        self._prepare_related_fields_for_save(operation_name="save")

        using = using or router.db_for_write(self.__class__, instance=self)
        if force_insert and (force_update or update_fields):
            raise ValueError("Cannot force both insert and updating in model saving.")

        deferred_fields = self.get_deferred_fields()
        if update_fields is not None:
            # If update_fields is empty, skip the save. We do also check for
            # no-op saves later on for inheritance cases. This bailout is
            # still needed for skipping signal sending.
            if not update_fields:
                return

            update_fields = frozenset(update_fields)
            field_names = set()

            for field in self._meta.concrete_fields:
                if not field.primary_key:
                    field_names.add(field.name)

                    if field.name != field.attname:
                        field_names.add(field.attname)

            non_model_fields = update_fields.difference(field_names)

            if non_model_fields:
                raise ValueError(
                    "The following fields do not exist in this model, are m2m "
                    "fields, or are non-concrete fields: %s"
                    % ", ".join(non_model_fields)
                )

        # If saving to the same database, and this model is deferred, then
        # automatically do an "update_fields" save on the loaded fields.
        elif not force_insert and deferred_fields and using == self._state.db:
            field_names = set()
            for field in self._meta.concrete_fields:
                if not field.primary_key and not hasattr(field, "through"):
                    field_names.add(field.attname)
            loaded_fields = field_names.difference(deferred_fields)
            if loaded_fields:
                update_fields = frozenset(loaded_fields)

        self.save_base(
            using=using,
            force_insert=force_insert,
            force_update=force_update,
            update_fields=update_fields,
        )
```
### 139 - django/db/models/base.py:

Start line: 1347, End line: 1376

```python
class Model(metaclass=ModelBase):

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
