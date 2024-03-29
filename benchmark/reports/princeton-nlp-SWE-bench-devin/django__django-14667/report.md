# django__django-14667

| **django/django** | `6a970a8b4600eb91be25f38caed0a52269d6303d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 996 |
| **Any found context length** | 996 |
| **Avg pos** | 4.0 |
| **Min pos** | 4 |
| **Max pos** | 4 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -2086,7 +2086,12 @@ def add_deferred_loading(self, field_names):
             self.deferred_loading = existing.union(field_names), True
         else:
             # Remove names from the set of any existing "immediate load" names.
-            self.deferred_loading = existing.difference(field_names), False
+            if new_existing := existing.difference(field_names):
+                self.deferred_loading = new_existing, False
+            else:
+                self.clear_deferred_loading()
+                if new_only := set(field_names).difference(existing):
+                    self.deferred_loading = new_only, True
 
     def add_immediate_loading(self, field_names):
         """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/query.py | 2089 | 2089 | 4 | 2 | 996


## Problem Statement

```
QuerySet.defer() doesn't clear deferred field when chaining with only().
Description
	
Considering a simple Company model with four fields: id, name, trade_number and country. If we evaluate a queryset containing a .defer() following a .only(), the generated sql query selects unexpected fields. For example: 
Company.objects.only("name").defer("name")
loads all the fields with the following query:
SELECT "company"."id", "company"."name", "company"."trade_number", "company"."country" FROM "company"
and 
Company.objects.only("name").defer("name").defer("country")
also loads all the fields with the same query:
SELECT "company"."id", "company"."name", "company"."trade_number", "company"."country" FROM "company"
In those two cases, i would expect the sql query to be:
SELECT "company"."id" FROM "company"
In the following example, we get the expected behavior:
Company.objects.only("name", "country").defer("name")
only loads "id" and "country" fields with the following query:
SELECT "company"."id", "company"."country" FROM "company"

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 1227 | 1246| 209 | 209 | 17575 | 
| 2 | 1 django/db/models/query.py | 1210 | 1225| 149 | 358 | 17575 | 
| 3 | **2 django/db/models/sql/query.py** | 715 | 750| 389 | 747 | 39914 | 
| **-> 4 <-** | **2 django/db/models/sql/query.py** | 2067 | 2089| 249 | 996 | 39914 | 
| 5 | **2 django/db/models/sql/query.py** | 2115 | 2132| 156 | 1152 | 39914 | 
| 6 | **2 django/db/models/sql/query.py** | 666 | 713| 511 | 1663 | 39914 | 
| 7 | 2 django/db/models/query.py | 839 | 870| 272 | 1935 | 39914 | 
| 8 | 3 django/db/models/query_utils.py | 94 | 131| 300 | 2235 | 42402 | 
| 9 | **3 django/db/models/sql/query.py** | 2161 | 2208| 394 | 2629 | 42402 | 
| 10 | 3 django/db/models/query.py | 1600 | 1656| 481 | 3110 | 42402 | 
| 11 | **3 django/db/models/sql/query.py** | 2091 | 2113| 229 | 3339 | 42402 | 
| 12 | 4 django/db/models/fields/related_descriptors.py | 120 | 154| 405 | 3744 | 52799 | 
| 13 | **4 django/db/models/sql/query.py** | 2134 | 2159| 214 | 3958 | 52799 | 
| 14 | 4 django/db/models/query.py | 1925 | 1957| 314 | 4272 | 52799 | 
| 15 | 4 django/db/models/query.py | 1688 | 1803| 1098 | 5370 | 52799 | 
| 16 | 5 django/db/models/base.py | 577 | 596| 170 | 5540 | 70123 | 
| 17 | 6 django/db/models/__init__.py | 1 | 53| 619 | 6159 | 70742 | 
| 18 | 6 django/db/models/fields/related_descriptors.py | 365 | 381| 184 | 6343 | 70742 | 
| 19 | **6 django/db/models/sql/query.py** | 1472 | 1557| 801 | 7144 | 70742 | 
| 20 | **6 django/db/models/sql/query.py** | 1843 | 1892| 329 | 7473 | 70742 | 
| 21 | 7 django/db/models/sql/where.py | 252 | 269| 141 | 7614 | 72718 | 
| 22 | 7 django/db/models/query.py | 175 | 234| 469 | 8083 | 72718 | 
| 23 | **7 django/db/models/sql/query.py** | 1701 | 1743| 436 | 8519 | 72718 | 
| 24 | 7 django/db/models/query.py | 1091 | 1112| 214 | 8733 | 72718 | 
| 25 | 8 django/db/models/sql/compiler.py | 1118 | 1147| 253 | 8986 | 87481 | 
| 26 | 9 django/core/serializers/base.py | 232 | 249| 208 | 9194 | 89907 | 
| 27 | 9 django/db/models/sql/compiler.py | 276 | 371| 710 | 9904 | 89907 | 
| 28 | **9 django/db/models/sql/query.py** | 1988 | 2033| 356 | 10260 | 89907 | 
| 29 | 9 django/db/models/fields/related_descriptors.py | 1 | 79| 683 | 10943 | 89907 | 
| 30 | 9 django/db/models/query.py | 1476 | 1514| 308 | 11251 | 89907 | 
| 31 | 9 django/db/models/query.py | 872 | 901| 248 | 11499 | 89907 | 
| 32 | **9 django/db/models/sql/query.py** | 1633 | 1654| 200 | 11699 | 89907 | 
| 33 | 9 django/db/models/base.py | 1 | 50| 328 | 12027 | 89907 | 
| 34 | 9 django/db/models/query.py | 1806 | 1857| 480 | 12507 | 89907 | 
| 35 | 10 django/db/models/aggregates.py | 50 | 68| 278 | 12785 | 91322 | 
| 36 | 10 django/db/models/query.py | 793 | 816| 207 | 12992 | 91322 | 
| 37 | 10 django/db/models/query.py | 1860 | 1923| 658 | 13650 | 91322 | 
| 38 | **10 django/db/models/sql/query.py** | 1894 | 1935| 355 | 14005 | 91322 | 
| 39 | 10 django/db/models/sql/compiler.py | 23 | 51| 302 | 14307 | 91322 | 
| 40 | 10 django/db/models/query.py | 324 | 350| 222 | 14529 | 91322 | 
| 41 | 10 django/db/models/base.py | 2127 | 2178| 351 | 14880 | 91322 | 
| 42 | **10 django/db/models/sql/query.py** | 1295 | 1359| 761 | 15641 | 91322 | 
| 43 | 11 django/db/models/fields/related.py | 139 | 166| 201 | 15842 | 105302 | 
| 44 | 11 django/db/models/query.py | 1351 | 1369| 186 | 16028 | 105302 | 
| 45 | 11 django/db/models/sql/compiler.py | 729 | 751| 202 | 16230 | 105302 | 
| 46 | **11 django/db/models/sql/query.py** | 1974 | 1986| 129 | 16359 | 105302 | 
| 47 | 11 django/db/models/sql/compiler.py | 923 | 1015| 839 | 17198 | 105302 | 
| 48 | 11 django/db/models/query.py | 1016 | 1036| 240 | 17438 | 105302 | 
| 49 | 12 django/db/models/fields/related_lookups.py | 65 | 104| 451 | 17889 | 106771 | 
| 50 | 12 django/db/models/sql/compiler.py | 462 | 515| 569 | 18458 | 106771 | 
| 51 | 12 django/db/models/aggregates.py | 70 | 106| 349 | 18807 | 106771 | 
| 52 | 12 django/db/models/sql/compiler.py | 1076 | 1116| 337 | 19144 | 106771 | 
| 53 | 12 django/db/models/sql/compiler.py | 1602 | 1642| 410 | 19554 | 106771 | 
| 54 | 12 django/db/models/query.py | 1129 | 1170| 323 | 19877 | 106771 | 
| 55 | 13 django/contrib/contenttypes/fields.py | 173 | 217| 411 | 20288 | 112222 | 
| 56 | 14 django/db/migrations/autodetector.py | 89 | 101| 116 | 20404 | 123802 | 
| 57 | **14 django/db/models/sql/query.py** | 1056 | 1074| 162 | 20566 | 123802 | 
| 58 | 14 django/db/models/sql/compiler.py | 1 | 20| 173 | 20739 | 123802 | 
| 59 | 14 django/db/models/query.py | 236 | 263| 221 | 20960 | 123802 | 
| 60 | **14 django/db/models/sql/query.py** | 1 | 62| 450 | 21410 | 123802 | 
| 61 | 14 django/db/models/query.py | 1516 | 1549| 297 | 21707 | 123802 | 
| 62 | 15 django/db/backends/mysql/features.py | 63 | 126| 640 | 22347 | 126044 | 
| 63 | 15 django/db/models/fields/related.py | 413 | 431| 165 | 22512 | 126044 | 
| 64 | 15 django/db/models/sql/compiler.py | 1054 | 1075| 199 | 22711 | 126044 | 
| 65 | 16 django/db/models/fields/__init__.py | 366 | 392| 199 | 22910 | 144403 | 
| 66 | 16 django/db/models/fields/related.py | 1263 | 1380| 963 | 23873 | 144403 | 
| 67 | **16 django/db/models/sql/query.py** | 139 | 233| 833 | 24706 | 144403 | 
| 68 | 17 django/db/backends/sqlite3/operations.py | 135 | 160| 279 | 24985 | 147675 | 
| 69 | 17 django/db/models/query.py | 1248 | 1276| 183 | 25168 | 147675 | 
| 70 | 18 django/db/models/sql/subqueries.py | 137 | 163| 161 | 25329 | 148876 | 
| 71 | 18 django/db/models/query.py | 265 | 285| 180 | 25509 | 148876 | 
| 72 | 18 django/db/models/query.py | 1466 | 1474| 136 | 25645 | 148876 | 
| 73 | 19 django/db/models/deletion.py | 1 | 76| 566 | 26211 | 152704 | 
| 74 | 19 django/db/models/query.py | 949 | 997| 371 | 26582 | 152704 | 
| 75 | 19 django/db/models/sql/compiler.py | 1233 | 1255| 243 | 26825 | 152704 | 
| 76 | 20 django/db/backends/base/features.py | 1 | 112| 895 | 27720 | 155690 | 
| 77 | 20 django/db/models/query.py | 670 | 688| 178 | 27898 | 155690 | 
| 78 | 20 django/db/models/sql/compiler.py | 427 | 435| 133 | 28031 | 155690 | 
| 79 | 20 django/db/models/sql/subqueries.py | 1 | 44| 320 | 28351 | 155690 | 
| 80 | 20 django/db/models/fields/related_lookups.py | 124 | 160| 244 | 28595 | 155690 | 
| 81 | **20 django/db/models/sql/query.py** | 367 | 417| 494 | 29089 | 155690 | 
| 82 | 20 django/db/models/query.py | 1069 | 1089| 155 | 29244 | 155690 | 
| 83 | 20 django/db/models/sql/compiler.py | 794 | 805| 163 | 29407 | 155690 | 
| 84 | 20 django/db/models/sql/compiler.py | 154 | 202| 523 | 29930 | 155690 | 
| 85 | 20 django/db/models/fields/related.py | 1230 | 1261| 180 | 30110 | 155690 | 
| 86 | **20 django/db/models/sql/query.py** | 1812 | 1841| 259 | 30369 | 155690 | 
| 87 | 21 django/db/models/lookups.py | 387 | 426| 329 | 30698 | 161029 | 
| 88 | 21 django/db/models/query.py | 759 | 792| 274 | 30972 | 161029 | 
| 89 | 21 django/db/models/fields/related.py | 527 | 592| 492 | 31464 | 161029 | 
| 90 | 22 django/contrib/admin/filters.py | 424 | 431| 107 | 31571 | 165152 | 
| 91 | 23 django/db/backends/base/schema.py | 1199 | 1230| 233 | 31804 | 177975 | 
| 92 | **23 django/db/models/sql/query.py** | 1427 | 1454| 283 | 32087 | 177975 | 
| 93 | 23 django/db/models/fields/related_descriptors.py | 344 | 363| 156 | 32243 | 177975 | 
| 94 | 24 django/db/models/fields/reverse_related.py | 160 | 178| 172 | 32415 | 180297 | 
| 95 | 25 django/db/backends/base/operations.py | 155 | 228| 615 | 33030 | 186059 | 
| 96 | **25 django/db/models/sql/query.py** | 639 | 664| 277 | 33307 | 186059 | 
| 97 | 25 django/db/models/query.py | 1172 | 1208| 327 | 33634 | 186059 | 
| 98 | 26 django/db/models/options.py | 557 | 585| 231 | 33865 | 193426 | 
| 99 | 26 django/db/models/fields/related.py | 296 | 330| 293 | 34158 | 193426 | 
| 100 | **26 django/db/models/sql/query.py** | 911 | 933| 248 | 34406 | 193426 | 
| 101 | **26 django/db/models/sql/query.py** | 1745 | 1810| 667 | 35073 | 193426 | 
| 102 | 26 django/db/models/query_utils.py | 201 | 235| 303 | 35376 | 193426 | 
| 103 | 26 django/db/models/fields/related.py | 714 | 726| 116 | 35492 | 193426 | 
| 104 | 26 django/db/models/query.py | 729 | 757| 261 | 35753 | 193426 | 
| 105 | **26 django/db/models/sql/query.py** | 2318 | 2334| 177 | 35930 | 193426 | 
| 106 | 26 django/db/models/base.py | 1440 | 1495| 491 | 36421 | 193426 | 
| 107 | 26 django/db/models/query.py | 1659 | 1687| 246 | 36667 | 193426 | 
| 108 | 27 django/db/backends/oracle/features.py | 1 | 120| 1015 | 37682 | 194442 | 
| 109 | 27 django/db/models/deletion.py | 269 | 344| 800 | 38482 | 194442 | 
| 110 | **27 django/db/models/sql/query.py** | 2210 | 2242| 228 | 38710 | 194442 | 
| 111 | 27 django/db/models/query.py | 1382 | 1430| 405 | 39115 | 194442 | 
| 112 | **27 django/db/models/sql/query.py** | 1076 | 1092| 152 | 39267 | 194442 | 


### Hint

```
Replying to Manuel Baclet: Considering a simple Company model with four fields: id, name, trade_number and country. If we evaluate a queryset containing a .defer() following a .only(), the generated sql query selects unexpected fields. For example: Company.objects.only("name").defer("name") loads all the fields with the following query: SELECT "company"."id", "company"."name", "company"."trade_number", "company"."country" FROM "company" This is an expected behavior, defer() removes fields from the list of fields specified by the only() method (i.e. list of fields that should not be deferred). In this example only() adds name to the list, defer() removes name from the list, so you have empty lists and all fields will be loaded. It is also ​documented: # Final result is that everything except "headline" is deferred. Entry.objects.only("headline", "body").defer("body") Company.objects.only("name").defer("name").defer("country") also loads all the fields with the same query: SELECT "company"."id", "company"."name", "company"."trade_number", "company"."country" FROM "company" I agree you shouldn't get all field, but only pk, name, and trade_number: SELECT "ticket_32704_company"."id", "ticket_32704_company"."name", "ticket_32704_company"."trade_number" FROM "ticket_32704_company" this is due to the fact that defer() doesn't clear the list of deferred field when chaining with only(). I attached a proposed patch.
Draft.
After reading the documentation carefully, i cannot say that it is clearly stated that deferring all the fields used in a previous .only() call performs a reset of the deferred set. Moreover, in the .defer() ​section, we have: You can make multiple calls to defer(). Each call adds new fields to the deferred set: and this seems to suggest that: calls to .defer() cannot remove items from the deferred set and evaluating qs.defer("some_field") should never fetch the column "some_field" (since this should add "some_field" to the deferred set) the querysets qs.defer("field1").defer("field2") and qs.defer("field1", "field2") should be equivalent IMHO, there is a mismatch between the doc and the actual implementation.
calls to .defer() cannot remove items from the deferred set and evaluating qs.defer("some_field") should never fetch the column "some_field" (since this should add "some_field" to the deferred set) Feel-free to propose a docs clarification (see also #24048). the querysets qs.defer("field1").defer("field2") and qs.defer("field1", "field2") should be equivalent That's why I accepted Company.objects.only("name").defer("name").defer("country") as a bug.
I think that what is described in the documentation is what users are expected and it is the implementation that should be fixed! With your patch proposal, i do not think that: Company.objects.only("name").defer("name").defer("country") is equivalent to Company.objects.only("name").defer("name", "country")
Replying to Manuel Baclet: I think that what is described in the documentation is what users are expected and it is the implementation that should be fixed! I don't agree, and there is no need to shout. As documented: "The only() method is more or less the opposite of defer(). You call it with the fields that should not be deferred ...", so .only('name').defer('name') should return all fields. You can start a discussion on DevelopersMailingList if you don't agree. With your patch proposal, i do not think that: Company.objects.only("name").defer("name").defer("country") is equivalent to Company.objects.only("name").defer("name", "country") Did you check this? with proposed patch country is the only deferred fields in both cases. As far as I'm aware that's an intended behavior.
With the proposed patch, I think that: Company.objects.only("name").defer("name", "country") loads all fields whereas: Company.objects.only("name").defer("name").defer("country") loads all fields except "country". Why is that? In the first case: existing.difference(field_names) == {"name"}.difference(["name", "country"]) == empty_set and we go into the if branch clearing all the deferred fields and we are done. In the second case: existing.difference(field_names) == {"name"}.difference(["name"]) == empty_set and we go into the if branch clearing all the deferred fields. Then we add "country" to the set of deferred fields.
Hey all, Replying to Mariusz Felisiak: Replying to Manuel Baclet: With your patch proposal, i do not think that: Company.objects.only("name").defer("name").defer("country") is equivalent to Company.objects.only("name").defer("name", "country") Did you check this? with proposed patch country is the only deferred fields in both cases. As far as I'm aware that's an intended behavior. I believe Manuel is right. This happens because the set difference in one direction gives you the empty set that will clear out the deferred fields - but it is missing the fact that we might also be adding more defer fields than we had only fields in the first place, so that we actually switch from an .only() to a .defer() mode. See the corresponding PR that should fix this behaviour ​https://github.com/django/django/pull/14667
```

## Patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -2086,7 +2086,12 @@ def add_deferred_loading(self, field_names):
             self.deferred_loading = existing.union(field_names), True
         else:
             # Remove names from the set of any existing "immediate load" names.
-            self.deferred_loading = existing.difference(field_names), False
+            if new_existing := existing.difference(field_names):
+                self.deferred_loading = new_existing, False
+            else:
+                self.clear_deferred_loading()
+                if new_only := set(field_names).difference(existing):
+                    self.deferred_loading = new_only, True
 
     def add_immediate_loading(self, field_names):
         """

```

## Test Patch

```diff
diff --git a/tests/defer/tests.py b/tests/defer/tests.py
--- a/tests/defer/tests.py
+++ b/tests/defer/tests.py
@@ -49,8 +49,16 @@ def test_defer_only_chaining(self):
         qs = Primary.objects.all()
         self.assert_delayed(qs.only("name", "value").defer("name")[0], 2)
         self.assert_delayed(qs.defer("name").only("value", "name")[0], 2)
+        self.assert_delayed(qs.defer('name').only('name').only('value')[0], 2)
         self.assert_delayed(qs.defer("name").only("value")[0], 2)
         self.assert_delayed(qs.only("name").defer("value")[0], 2)
+        self.assert_delayed(qs.only('name').defer('name').defer('value')[0], 1)
+        self.assert_delayed(qs.only('name').defer('name', 'value')[0], 1)
+
+    def test_defer_only_clear(self):
+        qs = Primary.objects.all()
+        self.assert_delayed(qs.only('name').defer('name')[0], 0)
+        self.assert_delayed(qs.defer('name').only('name')[0], 0)
 
     def test_defer_on_an_already_deferred_field(self):
         qs = Primary.objects.all()

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 1227, End line: 1246

```python
class QuerySet:

    def only(self, *fields):
        """
        Essentially, the opposite of defer(). Only the fields passed into this
        method and that are not already specified as deferred are loaded
        immediately when the queryset is evaluated.
        """
        self._not_support_combined_queries('only')
        if self._fields is not None:
            raise TypeError("Cannot call only() after .values() or .values_list()")
        if fields == (None,):
            # Can only pass None to defer(), not only(), as the rest option.
            # That won't stop people trying to do this, so let's be explicit.
            raise TypeError("Cannot pass None as an argument to only().")
        for field in fields:
            field = field.split(LOOKUP_SEP, 1)[0]
            if field in self.query._filtered_relations:
                raise ValueError('only() is not supported with FilteredRelation.')
        clone = self._chain()
        clone.query.add_immediate_loading(fields)
        return clone
```
### 2 - django/db/models/query.py:

Start line: 1210, End line: 1225

```python
class QuerySet:

    def defer(self, *fields):
        """
        Defer the loading of data for certain fields until they are accessed.
        Add the set of deferred fields to any existing set of deferred fields.
        The only exception to this is if None is passed in as the only
        parameter, in which case removal all deferrals.
        """
        self._not_support_combined_queries('defer')
        if self._fields is not None:
            raise TypeError("Cannot call defer() after .values() or .values_list()")
        clone = self._chain()
        if fields == (None,):
            clone.query.clear_deferred_loading()
        else:
            clone.query.add_deferred_loading(fields)
        return clone
```
### 3 - django/db/models/sql/query.py:

Start line: 715, End line: 750

```python
class Query(BaseExpression):

    def deferred_to_data(self, target, callback):
        # ... other code

        if defer:
            # We need to load all fields for each model, except those that
            # appear in "seen" (for all models that appear in "seen"). The only
            # slight complexity here is handling fields that exist on parent
            # models.
            workset = {}
            for model, values in seen.items():
                for field in model._meta.local_fields:
                    if field not in values:
                        m = field.model._meta.concrete_model
                        add_to_dict(workset, m, field)
            for model, values in must_include.items():
                # If we haven't included a model in workset, we don't add the
                # corresponding must_include fields for that model, since an
                # empty set means "include all fields". That's why there's no
                # "else" branch here.
                if model in workset:
                    workset[model].update(values)
            for model, values in workset.items():
                callback(target, model, values)
        else:
            for model, values in must_include.items():
                if model in seen:
                    seen[model].update(values)
                else:
                    # As we've passed through this model, but not explicitly
                    # included any fields, we have to make sure it's mentioned
                    # so that only the "must include" fields are pulled in.
                    seen[model] = values
            # Now ensure that every model in the inheritance chain is mentioned
            # in the parent list. Again, it must be mentioned to ensure that
            # only "must include" fields are pulled in.
            for model in orig_opts.get_parent_list():
                seen.setdefault(model, set())
            for model, values in seen.items():
                callback(target, model, values)
```
### 4 - django/db/models/sql/query.py:

Start line: 2067, End line: 2089

```python
class Query(BaseExpression):

    def clear_deferred_loading(self):
        """Remove any fields from the deferred loading set."""
        self.deferred_loading = (frozenset(), True)

    def add_deferred_loading(self, field_names):
        """
        Add the given list of model field names to the set of fields to
        exclude from loading from the database when automatic column selection
        is done. Add the new field names to any existing field names that
        are deferred (or removed from any existing field names that are marked
        as the only ones for immediate loading).
        """
        # Fields on related models are stored in the literal double-underscore
        # format, so that we can use a set datastructure. We do the foo__bar
        # splitting and handling when computing the SQL column names (as part of
        # get_columns()).
        existing, defer = self.deferred_loading
        if defer:
            # Add to existing deferred names.
            self.deferred_loading = existing.union(field_names), True
        else:
            # Remove names from the set of any existing "immediate load" names.
            self.deferred_loading = existing.difference(field_names), False
```
### 5 - django/db/models/sql/query.py:

Start line: 2115, End line: 2132

```python
class Query(BaseExpression):

    def get_loaded_field_names(self):
        """
        If any fields are marked to be deferred, return a dictionary mapping
        models to a set of names in those fields that will be loaded. If a
        model is not in the returned dictionary, none of its fields are
        deferred.

        If no fields are marked for deferral, return an empty dictionary.
        """
        # We cache this because we call this function multiple times
        # (compiler.fill_related_selections, query.iterator)
        try:
            return self._loaded_field_names_cache
        except AttributeError:
            collection = {}
            self.deferred_to_data(collection, self.get_loaded_field_names_cb)
            self._loaded_field_names_cache = collection
            return collection
```
### 6 - django/db/models/sql/query.py:

Start line: 666, End line: 713

```python
class Query(BaseExpression):

    def deferred_to_data(self, target, callback):
        """
        Convert the self.deferred_loading data structure to an alternate data
        structure, describing the field that *will* be loaded. This is used to
        compute the columns to select from the database and also by the
        QuerySet class to work out which fields are being initialized on each
        model. Models that have all their fields included aren't mentioned in
        the result, only those that have field restrictions in place.

        The "target" parameter is the instance that is populated (in place).
        The "callback" is a function that is called whenever a (model, field)
        pair need to be added to "target". It accepts three parameters:
        "target", and the model and list of fields being added for that model.
        """
        field_names, defer = self.deferred_loading
        if not field_names:
            return
        orig_opts = self.get_meta()
        seen = {}
        must_include = {orig_opts.concrete_model: {orig_opts.pk}}
        for field_name in field_names:
            parts = field_name.split(LOOKUP_SEP)
            cur_model = self.model._meta.concrete_model
            opts = orig_opts
            for name in parts[:-1]:
                old_model = cur_model
                if name in self._filtered_relations:
                    name = self._filtered_relations[name].relation_name
                source = opts.get_field(name)
                if is_reverse_o2o(source):
                    cur_model = source.related_model
                else:
                    cur_model = source.remote_field.model
                opts = cur_model._meta
                # Even if we're "just passing through" this model, we must add
                # both the current model's pk and the related reference field
                # (if it's not a reverse relation) to the things we select.
                if not is_reverse_o2o(source):
                    must_include[old_model].add(source)
                add_to_dict(must_include, cur_model, opts.pk)
            field = opts.get_field(parts[-1])
            is_reverse_object = field.auto_created and not field.concrete
            model = field.related_model if is_reverse_object else field.model
            model = model._meta.concrete_model
            if model == opts.model:
                model = cur_model
            if not is_reverse_o2o(field):
                add_to_dict(seen, model, field)
        # ... other code
```
### 7 - django/db/models/query.py:

Start line: 839, End line: 870

```python
class QuerySet:

    def _prefetch_related_objects(self):
        # This method can only be called once the result cache has been filled.
        prefetch_related_objects(self._result_cache, *self._prefetch_related_lookups)
        self._prefetch_done = True

    def explain(self, *, format=None, **options):
        return self.query.explain(using=self.db, format=format, **options)

    ##################################################
    # PUBLIC METHODS THAT RETURN A QUERYSET SUBCLASS #
    ##################################################

    def raw(self, raw_query, params=(), translations=None, using=None):
        if using is None:
            using = self.db
        qs = RawQuerySet(raw_query, model=self.model, params=params, translations=translations, using=using)
        qs._prefetch_related_lookups = self._prefetch_related_lookups[:]
        return qs

    def _values(self, *fields, **expressions):
        clone = self._chain()
        if expressions:
            clone = clone.annotate(**expressions)
        clone._fields = fields
        clone.query.set_values(fields)
        return clone

    def values(self, *fields, **expressions):
        fields += tuple(expressions)
        clone = self._values(*fields, **expressions)
        clone._iterable_class = ValuesIterable
        return clone
```
### 8 - django/db/models/query_utils.py:

Start line: 94, End line: 131

```python
class DeferredAttribute:
    """
    A wrapper for a deferred-loading field. When the value is read from this
    object the first time, the query is executed.
    """
    def __init__(self, field):
        self.field = field

    def __get__(self, instance, cls=None):
        """
        Retrieve and caches the value from the datastore on the first lookup.
        Return the cached value.
        """
        if instance is None:
            return self
        data = instance.__dict__
        field_name = self.field.attname
        if field_name not in data:
            # Let's see if the field is part of the parent chain. If so we
            # might be able to reuse the already loaded value. Refs #18343.
            val = self._check_parent_chain(instance)
            if val is None:
                instance.refresh_from_db(fields=[field_name])
            else:
                data[field_name] = val
        return data[field_name]

    def _check_parent_chain(self, instance):
        """
        Check if the field value can be fetched from a parent field already
        loaded in the instance. This can be done if the to-be fetched
        field is a primary key field.
        """
        opts = instance._meta
        link_field = opts.get_ancestor_link(self.field.model)
        if self.field.primary_key and self.field != link_field:
            return getattr(instance, link_field.attname)
        return None
```
### 9 - django/db/models/sql/query.py:

Start line: 2161, End line: 2208

```python
class Query(BaseExpression):

    def set_values(self, fields):
        self.select_related = False
        self.clear_deferred_loading()
        self.clear_select_fields()

        if fields:
            field_names = []
            extra_names = []
            annotation_names = []
            if not self.extra and not self.annotations:
                # Shortcut - if there are no extra or annotations, then
                # the values() clause must be just field names.
                field_names = list(fields)
            else:
                self.default_cols = False
                for f in fields:
                    if f in self.extra_select:
                        extra_names.append(f)
                    elif f in self.annotation_select:
                        annotation_names.append(f)
                    else:
                        field_names.append(f)
            self.set_extra_mask(extra_names)
            self.set_annotation_mask(annotation_names)
            selected = frozenset(field_names + extra_names + annotation_names)
        else:
            field_names = [f.attname for f in self.model._meta.concrete_fields]
            selected = frozenset(field_names)
        # Selected annotations must be known before setting the GROUP BY
        # clause.
        if self.group_by is True:
            self.add_fields((f.attname for f in self.model._meta.concrete_fields), False)
            # Disable GROUP BY aliases to avoid orphaning references to the
            # SELECT clause which is about to be cleared.
            self.set_group_by(allow_aliases=False)
            self.clear_select_fields()
        elif self.group_by:
            # Resolve GROUP BY annotation references if they are not part of
            # the selected fields anymore.
            group_by = []
            for expr in self.group_by:
                if isinstance(expr, Ref) and expr.refs not in selected:
                    expr = self.annotations[expr.refs]
                group_by.append(expr)
            self.group_by = tuple(group_by)

        self.values_select = tuple(field_names)
        self.add_fields(field_names, True)
```
### 10 - django/db/models/query.py:

Start line: 1600, End line: 1656

```python
class Prefetch:
    def __init__(self, lookup, queryset=None, to_attr=None):
        # `prefetch_through` is the path we traverse to perform the prefetch.
        self.prefetch_through = lookup
        # `prefetch_to` is the path to the attribute that stores the result.
        self.prefetch_to = lookup
        if queryset is not None and (
            isinstance(queryset, RawQuerySet) or (
                hasattr(queryset, '_iterable_class') and
                not issubclass(queryset._iterable_class, ModelIterable)
            )
        ):
            raise ValueError(
                'Prefetch querysets cannot use raw(), values(), and '
                'values_list().'
            )
        if to_attr:
            self.prefetch_to = LOOKUP_SEP.join(lookup.split(LOOKUP_SEP)[:-1] + [to_attr])

        self.queryset = queryset
        self.to_attr = to_attr

    def __getstate__(self):
        obj_dict = self.__dict__.copy()
        if self.queryset is not None:
            # Prevent the QuerySet from being evaluated
            obj_dict['queryset'] = self.queryset._chain(
                _result_cache=[],
                _prefetch_done=True,
            )
        return obj_dict

    def add_prefix(self, prefix):
        self.prefetch_through = prefix + LOOKUP_SEP + self.prefetch_through
        self.prefetch_to = prefix + LOOKUP_SEP + self.prefetch_to

    def get_current_prefetch_to(self, level):
        return LOOKUP_SEP.join(self.prefetch_to.split(LOOKUP_SEP)[:level + 1])

    def get_current_to_attr(self, level):
        parts = self.prefetch_to.split(LOOKUP_SEP)
        to_attr = parts[level]
        as_attr = self.to_attr and level == len(parts) - 1
        return to_attr, as_attr

    def get_current_queryset(self, level):
        if self.get_current_prefetch_to(level) == self.prefetch_to:
            return self.queryset
        return None

    def __eq__(self, other):
        if not isinstance(other, Prefetch):
            return NotImplemented
        return self.prefetch_to == other.prefetch_to

    def __hash__(self):
        return hash((self.__class__, self.prefetch_to))
```
### 11 - django/db/models/sql/query.py:

Start line: 2091, End line: 2113

```python
class Query(BaseExpression):

    def add_immediate_loading(self, field_names):
        """
        Add the given list of model field names to the set of fields to
        retrieve when the SQL is executed ("immediate loading" fields). The
        field names replace any existing immediate loading field names. If
        there are field names already specified for deferred loading, remove
        those names from the new field_names before storing the new names
        for immediate loading. (That is, immediate loading overrides any
        existing immediate values, but respects existing deferrals.)
        """
        existing, defer = self.deferred_loading
        field_names = set(field_names)
        if 'pk' in field_names:
            field_names.remove('pk')
            field_names.add(self.get_meta().pk.name)

        if defer:
            # Remove any existing deferred names from the current set before
            # setting the new names.
            self.deferred_loading = field_names.difference(existing), False
        else:
            # Replace any existing "immediate load" field names.
            self.deferred_loading = frozenset(field_names), False
```
### 13 - django/db/models/sql/query.py:

Start line: 2134, End line: 2159

```python
class Query(BaseExpression):

    def get_loaded_field_names_cb(self, target, model, fields):
        """Callback used by get_deferred_field_names()."""
        target[model] = {f.attname for f in fields}

    def set_annotation_mask(self, names):
        """Set the mask of annotations that will be returned by the SELECT."""
        if names is None:
            self.annotation_select_mask = None
        else:
            self.annotation_select_mask = set(names)
        self._annotation_select_cache = None

    def append_annotation_mask(self, names):
        if self.annotation_select_mask is not None:
            self.set_annotation_mask(self.annotation_select_mask.union(names))

    def set_extra_mask(self, names):
        """
        Set the mask of extra select items that will be returned by SELECT.
        Don't remove them from the Query since they might be used later.
        """
        if names is None:
            self.extra_select_mask = None
        else:
            self.extra_select_mask = set(names)
        self._extra_select_cache = None
```
### 19 - django/db/models/sql/query.py:

Start line: 1472, End line: 1557

```python
class Query(BaseExpression):

    def names_to_path(self, names, opts, allow_many=True, fail_on_missing=False):
        # ... other code
        for pos, name in enumerate(names):
            cur_names_with_path = (name, [])
            if name == 'pk':
                name = opts.pk.name

            field = None
            filtered_relation = None
            try:
                field = opts.get_field(name)
            except FieldDoesNotExist:
                if name in self.annotation_select:
                    field = self.annotation_select[name].output_field
                elif name in self._filtered_relations and pos == 0:
                    filtered_relation = self._filtered_relations[name]
                    if LOOKUP_SEP in filtered_relation.relation_name:
                        parts = filtered_relation.relation_name.split(LOOKUP_SEP)
                        filtered_relation_path, field, _, _ = self.names_to_path(
                            parts, opts, allow_many, fail_on_missing,
                        )
                        path.extend(filtered_relation_path[:-1])
                    else:
                        field = opts.get_field(filtered_relation.relation_name)
            if field is not None:
                # Fields that contain one-to-many relations with a generic
                # model (like a GenericForeignKey) cannot generate reverse
                # relations and therefore cannot be used for reverse querying.
                if field.is_relation and not field.related_model:
                    raise FieldError(
                        "Field %r does not generate an automatic reverse "
                        "relation and therefore cannot be used for reverse "
                        "querying. If it is a GenericForeignKey, consider "
                        "adding a GenericRelation." % name
                    )
                try:
                    model = field.model._meta.concrete_model
                except AttributeError:
                    # QuerySet.annotate() may introduce fields that aren't
                    # attached to a model.
                    model = None
            else:
                # We didn't find the current field, so move position back
                # one step.
                pos -= 1
                if pos == -1 or fail_on_missing:
                    available = sorted([
                        *get_field_names_from_opts(opts),
                        *self.annotation_select,
                        *self._filtered_relations,
                    ])
                    raise FieldError("Cannot resolve keyword '%s' into field. "
                                     "Choices are: %s" % (name, ", ".join(available)))
                break
            # Check if we need any joins for concrete inheritance cases (the
            # field lives in parent, but we are currently in one of its
            # children)
            if model is not opts.model:
                path_to_parent = opts.get_path_to_parent(model)
                if path_to_parent:
                    path.extend(path_to_parent)
                    cur_names_with_path[1].extend(path_to_parent)
                    opts = path_to_parent[-1].to_opts
            if hasattr(field, 'get_path_info'):
                pathinfos = field.get_path_info(filtered_relation)
                if not allow_many:
                    for inner_pos, p in enumerate(pathinfos):
                        if p.m2m:
                            cur_names_with_path[1].extend(pathinfos[0:inner_pos + 1])
                            names_with_path.append(cur_names_with_path)
                            raise MultiJoin(pos + 1, names_with_path)
                last = pathinfos[-1]
                path.extend(pathinfos)
                final_field = last.join_field
                opts = last.to_opts
                targets = last.target_fields
                cur_names_with_path[1].extend(pathinfos)
                names_with_path.append(cur_names_with_path)
            else:
                # Local non-relational field.
                final_field = field
                targets = (field,)
                if fail_on_missing and pos + 1 != len(names):
                    raise FieldError(
                        "Cannot resolve keyword %r into field. Join on '%s'"
                        " not permitted." % (names[pos + 1], name))
                break
        return path, final_field, targets, names[pos + 1:]
```
### 20 - django/db/models/sql/query.py:

Start line: 1843, End line: 1892

```python
class Query(BaseExpression):

    def clear_limits(self):
        """Clear any existing limits."""
        self.low_mark, self.high_mark = 0, None

    @property
    def is_sliced(self):
        return self.low_mark != 0 or self.high_mark is not None

    def has_limit_one(self):
        return self.high_mark is not None and (self.high_mark - self.low_mark) == 1

    def can_filter(self):
        """
        Return True if adding filters to this instance is still possible.

        Typically, this means no limits or offsets have been put on the results.
        """
        return not self.is_sliced

    def clear_select_clause(self):
        """Remove all fields from SELECT clause."""
        self.select = ()
        self.default_cols = False
        self.select_related = False
        self.set_extra_mask(())
        self.set_annotation_mask(())

    def clear_select_fields(self):
        """
        Clear the list of fields to select (but not extra_select columns).
        Some queryset types completely replace any existing list of select
        columns.
        """
        self.select = ()
        self.values_select = ()

    def add_select_col(self, col, name):
        self.select += col,
        self.values_select += name,

    def set_select(self, cols):
        self.default_cols = False
        self.select = tuple(cols)

    def add_distinct_fields(self, *field_names):
        """
        Add and resolve the given fields to the query's "distinct on" clause.
        """
        self.distinct_fields = field_names
        self.distinct = True
```
### 23 - django/db/models/sql/query.py:

Start line: 1701, End line: 1743

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False):
        annotation = self.annotations.get(name)
        if annotation is not None:
            if not allow_joins:
                for alias in self._gen_col_aliases([annotation]):
                    if isinstance(self.alias_map[alias], Join):
                        raise FieldError(
                            'Joined field references are not permitted in '
                            'this query'
                        )
            if summarize:
                # Summarize currently means we are doing an aggregate() query
                # which is executed as a wrapped subquery if any of the
                # aggregate() elements reference an existing annotation. In
                # that case we need to return a Ref to the subquery's annotation.
                if name not in self.annotation_select:
                    raise FieldError(
                        "Cannot aggregate over the '%s' alias. Use annotate() "
                        "to promote it." % name
                    )
                return Ref(name, self.annotation_select[name])
            else:
                return annotation
        else:
            field_list = name.split(LOOKUP_SEP)
            annotation = self.annotations.get(field_list[0])
            if annotation is not None:
                for transform in field_list[1:]:
                    annotation = self.try_transform(annotation, transform)
                return annotation
            join_info = self.setup_joins(field_list, self.get_meta(), self.get_initial_alias(), can_reuse=reuse)
            targets, final_alias, join_list = self.trim_joins(join_info.targets, join_info.joins, join_info.path)
            if not allow_joins and len(join_list) > 1:
                raise FieldError('Joined field references are not permitted in this query')
            if len(targets) > 1:
                raise FieldError("Referencing multicolumn fields with F() objects "
                                 "isn't supported")
            # Verify that the last lookup in name is a field or a transform:
            # transform_function() raises FieldError if not.
            transform = join_info.transform_function(targets[0], final_alias)
            if reuse is not None:
                reuse.update(join_list)
            return transform
```
### 28 - django/db/models/sql/query.py:

Start line: 1988, End line: 2033

```python
class Query(BaseExpression):

    def set_group_by(self, allow_aliases=True):
        """
        Expand the GROUP BY clause required by the query.

        This will usually be the set of all non-aggregate fields in the
        return data. If the database backend supports grouping by the
        primary key, and the query would be equivalent, the optimization
        will be made automatically.
        """
        # Column names from JOINs to check collisions with aliases.
        if allow_aliases:
            column_names = set()
            seen_models = set()
            for join in list(self.alias_map.values())[1:]:  # Skip base table.
                model = join.join_field.related_model
                if model not in seen_models:
                    column_names.update({
                        field.column
                        for field in model._meta.local_concrete_fields
                    })
                    seen_models.add(model)

        group_by = list(self.select)
        if self.annotation_select:
            for alias, annotation in self.annotation_select.items():
                if not allow_aliases or alias in column_names:
                    alias = None
                group_by_cols = annotation.get_group_by_cols(alias=alias)
                group_by.extend(group_by_cols)
        self.group_by = tuple(group_by)

    def add_select_related(self, fields):
        """
        Set up the select_related data structure so that we only select
        certain related models (as opposed to all models, when
        self.select_related=True).
        """
        if isinstance(self.select_related, bool):
            field_dict = {}
        else:
            field_dict = self.select_related
        for field in fields:
            d = field_dict
            for part in field.split(LOOKUP_SEP):
                d = d.setdefault(part, {})
        self.select_related = field_dict
```
### 32 - django/db/models/sql/query.py:

Start line: 1633, End line: 1654

```python
class Query(BaseExpression):

    def setup_joins(self, names, opts, alias, can_reuse=None, allow_many=True):
        # ... other code
        for join in path:
            if join.filtered_relation:
                filtered_relation = join.filtered_relation.clone()
                table_alias = filtered_relation.alias
            else:
                filtered_relation = None
                table_alias = None
            opts = join.to_opts
            if join.direct:
                nullable = self.is_nullable(join.join_field)
            else:
                nullable = True
            connection = Join(
                opts.db_table, alias, table_alias, INNER, join.join_field,
                nullable, filtered_relation=filtered_relation,
            )
            reuse = can_reuse if join.m2m else None
            alias = self.join(connection, reuse=reuse)
            joins.append(alias)
            if filtered_relation:
                filtered_relation.path = joins[:]
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 38 - django/db/models/sql/query.py:

Start line: 1894, End line: 1935

```python
class Query(BaseExpression):

    def add_fields(self, field_names, allow_m2m=True):
        """
        Add the given (model) fields to the select set. Add the field names in
        the order specified.
        """
        alias = self.get_initial_alias()
        opts = self.get_meta()

        try:
            cols = []
            for name in field_names:
                # Join promotion note - we must not remove any rows here, so
                # if there is no existing joins, use outer join.
                join_info = self.setup_joins(name.split(LOOKUP_SEP), opts, alias, allow_many=allow_m2m)
                targets, final_alias, joins = self.trim_joins(
                    join_info.targets,
                    join_info.joins,
                    join_info.path,
                )
                for target in targets:
                    cols.append(join_info.transform_function(target, final_alias))
            if cols:
                self.set_select(cols)
        except MultiJoin:
            raise FieldError("Invalid field name: '%s'" % name)
        except FieldError:
            if LOOKUP_SEP in name:
                # For lookups spanning over relationships, show the error
                # from the model on which the lookup failed.
                raise
            elif name in self.annotations:
                raise FieldError(
                    "Cannot select the '%s' alias. Use annotate() to promote "
                    "it." % name
                )
            else:
                names = sorted([
                    *get_field_names_from_opts(opts), *self.extra,
                    *self.annotation_select, *self._filtered_relations
                ])
                raise FieldError("Cannot resolve keyword %r into field. "
                                 "Choices are: %s" % (name, ", ".join(names)))
```
### 42 - django/db/models/sql/query.py:

Start line: 1295, End line: 1359

```python
class Query(BaseExpression):

    def build_filter(self, filter_expr, branch_negated=False, current_negated=False,
                     can_reuse=None, allow_joins=True, split_subq=True,
                     check_filterable=True):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts, opts, alias, can_reuse=can_reuse, allow_many=allow_many,
            )

            # Prevent iterator from being consumed by check_related_objects()
            if isinstance(value, Iterator):
                value = list(value)
            self.check_related_objects(join_info.final_field, value, join_info.opts)

            # split_exclude() needs to know which joins were generated for the
            # lookup parts
            self._lookup_joins = join_info.joins
        except MultiJoin as e:
            return self.split_exclude(filter_expr, can_reuse, e.names_with_path)

        # Update used_joins before trimming since they are reused to determine
        # which joins could be later promoted to INNER.
        used_joins.update(join_info.joins)
        targets, alias, join_list = self.trim_joins(join_info.targets, join_info.joins, join_info.path)
        if can_reuse is not None:
            can_reuse.update(join_list)

        if join_info.final_field.is_relation:
            # No support for transforms for relational fields
            num_lookups = len(lookups)
            if num_lookups > 1:
                raise FieldError('Related Field got invalid lookup: {}'.format(lookups[0]))
            if len(targets) == 1:
                col = self._get_col(targets[0], join_info.final_field, alias)
            else:
                col = MultiColSource(alias, targets, join_info.targets, join_info.final_field)
        else:
            col = self._get_col(targets[0], join_info.final_field, alias)

        condition = self.build_lookup(lookups, col, value)
        lookup_type = condition.lookup_name
        clause = self.where_class([condition], connector=AND)

        require_outer = lookup_type == 'isnull' and condition.rhs is True and not current_negated
        if current_negated and (lookup_type != 'isnull' or condition.rhs is False) and condition.rhs is not None:
            require_outer = True
            if lookup_type != 'isnull':
                # The condition added here will be SQL like this:
                # NOT (col IS NOT NULL), where the first NOT is added in
                # upper layers of code. The reason for addition is that if col
                # is null, then col != someval will result in SQL "unknown"
                # which isn't the same as in Python. The Python None handling
                # is wanted, and it can be gotten by
                # (col IS NULL OR col != someval)
                #   <=>
                # NOT (col IS NOT NULL AND col = someval).
                if (
                    self.is_nullable(targets[0]) or
                    self.alias_map[join_list[-1]].join_type == LOUTER
                ):
                    lookup_class = targets[0].get_lookup('isnull')
                    col = self._get_col(targets[0], join_info.targets[0], alias)
                    clause.add(lookup_class(col, False), AND)
                # If someval is a nullable column, someval IS NOT NULL is
                # added.
                if isinstance(value, Col) and self.is_nullable(value.target):
                    lookup_class = value.target.get_lookup('isnull')
                    clause.add(lookup_class(value, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 46 - django/db/models/sql/query.py:

Start line: 1974, End line: 1986

```python
class Query(BaseExpression):

    def clear_ordering(self, force=False, clear_default=True):
        """
        Remove any ordering settings if the current query allows it without
        side effects, set 'force' to True to clear the ordering regardless.
        If 'clear_default' is True, there will be no ordering in the resulting
        query (not even the model's default).
        """
        if not force and (self.is_sliced or self.distinct_fields or self.select_for_update):
            return
        self.order_by = ()
        self.extra_order_by = ()
        if clear_default:
            self.default_ordering = False
```
### 57 - django/db/models/sql/query.py:

Start line: 1056, End line: 1074

```python
class Query(BaseExpression):

    def get_external_cols(self):
        exprs = chain(self.annotations.values(), self.where.children)
        return [
            col for col in self._gen_cols(exprs, include_external=True)
            if col.alias in self.external_aliases
        ]

    def as_sql(self, compiler, connection):
        # Some backends (e.g. Oracle) raise an error when a subquery contains
        # unnecessary ORDER BY clause.
        if (
            self.subquery and
            not connection.features.ignores_unnecessary_order_by_in_subqueries
        ):
            self.clear_ordering(force=False)
        sql, params = self.get_compiler(connection=connection).as_sql()
        if self.subquery:
            sql = '(%s)' % sql
        return sql, params
```
### 60 - django/db/models/sql/query.py:

Start line: 1, End line: 62

```python
"""
Create SQL statements for QuerySets.

The code in here encapsulates all of the SQL construction so that QuerySets
themselves do not have to (and could be backed by things other than SQL
databases). The abstraction barrier only works one way: this module has to know
all about the internals of models in order to get the information it needs.
"""
import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase

from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
    BaseExpression, Col, Exists, F, OuterRef, Ref, ResolvedOuterRef,
)
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
    Q, check_rel_lookup_compatibility, refs_expression,
)
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import (
    BaseTable, Empty, Join, MultiJoin,
)
from django.db.models.sql.where import (
    AND, OR, ExtraWhere, NothingNode, WhereNode,
)
from django.utils.functional import cached_property
from django.utils.tree import Node

__all__ = ['Query', 'RawQuery']


def get_field_names_from_opts(opts):
    return set(chain.from_iterable(
        (f.name, f.attname) if f.concrete else (f.name,)
        for f in opts.get_fields()
    ))


def get_children_from_q(q):
    for child in q.children:
        if isinstance(child, Node):
            yield from get_children_from_q(child)
        else:
            yield child


JoinInfo = namedtuple(
    'JoinInfo',
    ('final_field', 'targets', 'opts', 'joins', 'path', 'transform_function')
)
```
### 67 - django/db/models/sql/query.py:

Start line: 139, End line: 233

```python
class Query(BaseExpression):
    """A single SQL query."""

    alias_prefix = 'T'
    subq_aliases = frozenset([alias_prefix])

    compiler = 'SQLCompiler'

    def __init__(self, model, where=WhereNode, alias_cols=True):
        self.model = model
        self.alias_refcount = {}
        # alias_map is the most important data structure regarding joins.
        # It's used for recording which joins exist in the query and what
        # types they are. The key is the alias of the joined table (possibly
        # the table name) and the value is a Join-like object (see
        # sql.datastructures.Join for more information).
        self.alias_map = {}
        # Whether to provide alias to columns during reference resolving.
        self.alias_cols = alias_cols
        # Sometimes the query contains references to aliases in outer queries (as
        # a result of split_exclude). Correct alias quoting needs to know these
        # aliases too.
        # Map external tables to whether they are aliased.
        self.external_aliases = {}
        self.table_map = {}     # Maps table names to list of aliases.
        self.default_cols = True
        self.default_ordering = True
        self.standard_ordering = True
        self.used_aliases = set()
        self.filter_is_sticky = False
        self.subquery = False

        # SQL-related attributes
        # Select and related select clauses are expressions to use in the
        # SELECT clause of the query.
        # The select is used for cases where we want to set up the select
        # clause to contain other than default fields (values(), subqueries...)
        # Note that annotations go to annotations dictionary.
        self.select = ()
        self.where = where()
        self.where_class = where
        # The group_by attribute can have one of the following forms:
        #  - None: no group by at all in the query
        #  - A tuple of expressions: group by (at least) those expressions.
        #    String refs are also allowed for now.
        #  - True: group by all select fields of the model
        # See compiler.get_group_by() for details.
        self.group_by = None
        self.order_by = ()
        self.low_mark, self.high_mark = 0, None  # Used for offset/limit
        self.distinct = False
        self.distinct_fields = ()
        self.select_for_update = False
        self.select_for_update_nowait = False
        self.select_for_update_skip_locked = False
        self.select_for_update_of = ()
        self.select_for_no_key_update = False

        self.select_related = False
        # Arbitrary limit for select_related to prevents infinite recursion.
        self.max_depth = 5

        # Holds the selects defined by a call to values() or values_list()
        # excluding annotation_select and extra_select.
        self.values_select = ()

        # SQL annotation-related attributes
        self.annotations = {}  # Maps alias -> Annotation Expression
        self.annotation_select_mask = None
        self._annotation_select_cache = None

        # Set combination attributes
        self.combinator = None
        self.combinator_all = False
        self.combined_queries = ()

        # These are for extensions. The contents are more or less appended
        # verbatim to the appropriate clause.
        self.extra = {}  # Maps col_alias -> (col_sql, params).
        self.extra_select_mask = None
        self._extra_select_cache = None

        self.extra_tables = ()
        self.extra_order_by = ()

        # A tuple that is a set of model field names and either True, if these
        # are the fields to defer, or False if these are the only fields to
        # load.
        self.deferred_loading = (frozenset(), True)

        self._filtered_relations = {}

        self.explain_query = False
        self.explain_format = None
        self.explain_options = {}
```
### 81 - django/db/models/sql/query.py:

Start line: 367, End line: 417

```python
class Query(BaseExpression):

    def rewrite_cols(self, annotation, col_cnt):
        # We must make sure the inner query has the referred columns in it.
        # If we are aggregating over an annotation, then Django uses Ref()
        # instances to note this. However, if we are annotating over a column
        # of a related model, then it might be that column isn't part of the
        # SELECT clause of the inner query, and we must manually make sure
        # the column is selected. An example case is:
        orig_exprs = annotation.get_source_expressions()
        new_exprs = []
        for expr in orig_exprs:
            # FIXME: These conditions are fairly arbitrary. Identify a better
            # method of having expressions decide which code path they should
            # take.
            if isinstance(expr, Ref):
                # Its already a Ref to subquery (see resolve_ref() for
                # details)
                new_exprs.append(expr)
            elif isinstance(expr, (WhereNode, Lookup)):
                # Decompose the subexpressions further. The code here is
                # copied from the else clause, but this condition must appear
                # before the contains_aggregate/is_summary condition below.
                new_expr, col_cnt = self.rewrite_cols(expr, col_cnt)
                new_exprs.append(new_expr)
            else:
                # Reuse aliases of expressions already selected in subquery.
                for col_alias, selected_annotation in self.annotation_select.items():
                    if selected_annotation is expr:
                        new_expr = Ref(col_alias, expr)
                        break
                else:
                    # An expression that is not selected the subquery.
                    if isinstance(expr, Col) or (expr.contains_aggregate and not expr.is_summary):
                        # Reference column or another aggregate. Select it
                        # under a non-conflicting alias.
                        col_cnt += 1
                        col_alias = '__col%d' % col_cnt
                        self.annotations[col_alias] = expr
                        self.append_annotation_mask([col_alias])
                        new_expr = Ref(col_alias, expr)
                    else:
                        # Some other expression not referencing database values
                        # directly. Its subexpression might contain Cols.
                        new_expr, col_cnt = self.rewrite_cols(expr, col_cnt)
                new_exprs.append(new_expr)
        annotation.set_source_expressions(new_exprs)
        return annotation, col_cnt
```
### 86 - django/db/models/sql/query.py:

Start line: 1812, End line: 1841

```python
class Query(BaseExpression):

    def set_empty(self):
        self.where.add(NothingNode(), AND)
        for query in self.combined_queries:
            query.set_empty()

    def is_empty(self):
        return any(isinstance(c, NothingNode) for c in self.where.children)

    def set_limits(self, low=None, high=None):
        """
        Adjust the limits on the rows retrieved. Use low/high to set these,
        as it makes it more Pythonic to read and write. When the SQL query is
        created, convert them to the appropriate offset and limit values.

        Apply any limits passed in here to the existing constraints. Add low
        to the current low value and clamp both to any existing high value.
        """
        if high is not None:
            if self.high_mark is not None:
                self.high_mark = min(self.high_mark, self.low_mark + high)
            else:
                self.high_mark = self.low_mark + high
        if low is not None:
            if self.high_mark is not None:
                self.low_mark = min(self.high_mark, self.low_mark + low)
            else:
                self.low_mark = self.low_mark + low

        if self.low_mark == self.high_mark:
            self.set_empty()
```
### 92 - django/db/models/sql/query.py:

Start line: 1427, End line: 1454

```python
class Query(BaseExpression):

    def add_filtered_relation(self, filtered_relation, alias):
        filtered_relation.alias = alias
        lookups = dict(get_children_from_q(filtered_relation.condition))
        relation_lookup_parts, relation_field_parts, _ = self.solve_lookup_type(filtered_relation.relation_name)
        if relation_lookup_parts:
            raise ValueError(
                "FilteredRelation's relation_name cannot contain lookups "
                "(got %r)." % filtered_relation.relation_name
            )
        for lookup in chain(lookups):
            lookup_parts, lookup_field_parts, _ = self.solve_lookup_type(lookup)
            shift = 2 if not lookup_parts else 1
            lookup_field_path = lookup_field_parts[:-shift]
            for idx, lookup_field_part in enumerate(lookup_field_path):
                if len(relation_field_parts) > idx:
                    if relation_field_parts[idx] != lookup_field_part:
                        raise ValueError(
                            "FilteredRelation's condition doesn't support "
                            "relations outside the %r (got %r)."
                            % (filtered_relation.relation_name, lookup)
                        )
                else:
                    raise ValueError(
                        "FilteredRelation's condition doesn't support nested "
                        "relations deeper than the relation_name (got %r for "
                        "%r)." % (lookup, filtered_relation.relation_name)
                    )
        self._filtered_relations[filtered_relation.alias] = filtered_relation
```
### 96 - django/db/models/sql/query.py:

Start line: 639, End line: 664

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):

        # Selection columns and extra extensions are those provided by 'rhs'.
        if rhs.select:
            self.set_select([col.relabeled_clone(change_map) for col in rhs.select])
        else:
            self.select = ()

        if connector == OR:
            # It would be nice to be able to handle this, but the queries don't
            # really make sense (or return consistent value sets). Not worth
            # the extra complexity when you can write a real query instead.
            if self.extra and rhs.extra:
                raise ValueError("When merging querysets using 'or', you cannot have extra(select=...) on both sides.")
        self.extra.update(rhs.extra)
        extra_select_mask = set()
        if self.extra_select_mask is not None:
            extra_select_mask.update(self.extra_select_mask)
        if rhs.extra_select_mask is not None:
            extra_select_mask.update(rhs.extra_select_mask)
        if extra_select_mask:
            self.set_extra_mask(extra_select_mask)
        self.extra_tables += rhs.extra_tables

        # Ordering uses the 'rhs' ordering, unless it has none, in which case
        # the current ordering is used.
        self.order_by = rhs.order_by or self.order_by
        self.extra_order_by = rhs.extra_order_by or self.extra_order_by
```
### 100 - django/db/models/sql/query.py:

Start line: 911, End line: 933

```python
class Query(BaseExpression):

    def bump_prefix(self, outer_query):
        # ... other code

        if self.alias_prefix != outer_query.alias_prefix:
            # No clashes between self and outer query should be possible.
            return

        # Explicitly avoid infinite loop. The constant divider is based on how
        # much depth recursive subquery references add to the stack. This value
        # might need to be adjusted when adding or removing function calls from
        # the code path in charge of performing these operations.
        local_recursion_limit = sys.getrecursionlimit() // 16
        for pos, prefix in enumerate(prefix_gen()):
            if prefix not in self.subq_aliases:
                self.alias_prefix = prefix
                break
            if pos > local_recursion_limit:
                raise RecursionError(
                    'Maximum recursion depth exceeded: too many subqueries.'
                )
        self.subq_aliases = self.subq_aliases.union([self.alias_prefix])
        outer_query.subq_aliases = outer_query.subq_aliases.union(self.subq_aliases)
        self.change_aliases({
            alias: '%s%d' % (self.alias_prefix, pos)
            for pos, alias in enumerate(self.alias_map)
        })
```
### 101 - django/db/models/sql/query.py:

Start line: 1745, End line: 1810

```python
class Query(BaseExpression):

    def split_exclude(self, filter_expr, can_reuse, names_with_path):
        """
        When doing an exclude against any kind of N-to-many relation, we need
        to use a subquery. This method constructs the nested query, given the
        original exclude filter (filter_expr) and the portion up to the first
        N-to-many relation field.

        For example, if the origin filter is ~Q(child__name='foo'), filter_expr
        is ('child__name', 'foo') and can_reuse is a set of joins usable for
        filters in the original query.

        We will turn this into equivalent of:
            WHERE NOT EXISTS(
                SELECT 1
                FROM child
                WHERE name = 'foo' AND child.parent_id = parent.id
                LIMIT 1
            )
        """
        filter_lhs, filter_rhs = filter_expr
        if isinstance(filter_rhs, OuterRef):
            filter_expr = (filter_lhs, OuterRef(filter_rhs))
        elif isinstance(filter_rhs, F):
            filter_expr = (filter_lhs, OuterRef(filter_rhs.name))
        # Generate the inner query.
        query = Query(self.model)
        query._filtered_relations = self._filtered_relations
        query.add_filter(filter_expr)
        query.clear_ordering(force=True)
        # Try to have as simple as possible subquery -> trim leading joins from
        # the subquery.
        trimmed_prefix, contains_louter = query.trim_start(names_with_path)

        col = query.select[0]
        select_field = col.target
        alias = col.alias
        if alias in can_reuse:
            pk = select_field.model._meta.pk
            # Need to add a restriction so that outer query's filters are in effect for
            # the subquery, too.
            query.bump_prefix(self)
            lookup_class = select_field.get_lookup('exact')
            # Note that the query.select[0].alias is different from alias
            # due to bump_prefix above.
            lookup = lookup_class(pk.get_col(query.select[0].alias),
                                  pk.get_col(alias))
            query.where.add(lookup, AND)
            query.external_aliases[alias] = True

        lookup_class = select_field.get_lookup('exact')
        lookup = lookup_class(col, ResolvedOuterRef(trimmed_prefix))
        query.where.add(lookup, AND)
        condition, needed_inner = self.build_filter(Exists(query))

        if contains_louter:
            or_null_condition, _ = self.build_filter(
                ('%s__isnull' % trimmed_prefix, True),
                current_negated=True, branch_negated=True, can_reuse=can_reuse)
            condition.add(or_null_condition, OR)
            # Note that the end result will be:
            # (outercol NOT IN innerq AND outercol IS NOT NULL) OR outercol IS NULL.
            # This might look crazy but due to how IN works, this seems to be
            # correct. If the IS NOT NULL check is removed then outercol NOT
            # IN will return UNKNOWN. If the IS NULL check is removed, then if
            # outercol IS NULL we will not match the row.
        return condition, needed_inner
```
### 105 - django/db/models/sql/query.py:

Start line: 2318, End line: 2334

```python
class Query(BaseExpression):

    def is_nullable(self, field):
        """
        Check if the given field should be treated as nullable.

        Some backends treat '' as null and Django treats such fields as
        nullable for those backends. In such situations field.null can be
        False even if we should treat the field as nullable.
        """
        # We need to use DEFAULT_DB_ALIAS here, as QuerySet does not have
        # (nor should it have) knowledge of which connection is going to be
        # used. The proper fix would be to defer all decisions where
        # is_nullable() is needed to the compiler stage, but that is not easy
        # to do currently.
        return (
            connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls and
            field.empty_strings_allowed
        ) or field.null
```
### 110 - django/db/models/sql/query.py:

Start line: 2210, End line: 2242

```python
class Query(BaseExpression):

    @property
    def annotation_select(self):
        """
        Return the dictionary of aggregate columns that are not masked and
        should be used in the SELECT clause. Cache this result for performance.
        """
        if self._annotation_select_cache is not None:
            return self._annotation_select_cache
        elif not self.annotations:
            return {}
        elif self.annotation_select_mask is not None:
            self._annotation_select_cache = {
                k: v for k, v in self.annotations.items()
                if k in self.annotation_select_mask
            }
            return self._annotation_select_cache
        else:
            return self.annotations

    @property
    def extra_select(self):
        if self._extra_select_cache is not None:
            return self._extra_select_cache
        if not self.extra:
            return {}
        elif self.extra_select_mask is not None:
            self._extra_select_cache = {
                k: v for k, v in self.extra.items()
                if k in self.extra_select_mask
            }
            return self._extra_select_cache
        else:
            return self.extra
```
### 112 - django/db/models/sql/query.py:

Start line: 1076, End line: 1092

```python
class Query(BaseExpression):

    def resolve_lookup_value(self, value, can_reuse, allow_joins):
        if hasattr(value, 'resolve_expression'):
            value = value.resolve_expression(
                self, reuse=can_reuse, allow_joins=allow_joins,
            )
        elif isinstance(value, (list, tuple)):
            # The items of the iterable may be expressions and therefore need
            # to be resolved independently.
            values = (
                self.resolve_lookup_value(sub_value, can_reuse, allow_joins)
                for sub_value in value
            )
            type_ = type(value)
            if hasattr(type_, '_make'):  # namedtuple
                return type_(*values)
            return type_(values)
        return value
```
