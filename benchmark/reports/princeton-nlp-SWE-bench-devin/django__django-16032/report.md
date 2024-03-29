# django__django-16032

| **django/django** | `0c3981eb5094419fe200eb46c71b5376a2266166` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 16530 |
| **Any found context length** | 2553 |
| **Avg pos** | 145.5 |
| **Min pos** | 9 |
| **Max pos** | 140 |
| **Top file pos** | 2 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/fields/related_lookups.py b/django/db/models/fields/related_lookups.py
--- a/django/db/models/fields/related_lookups.py
+++ b/django/db/models/fields/related_lookups.py
@@ -93,7 +93,6 @@ def get_prep_lookup(self):
             elif not getattr(self.rhs, "has_select_fields", True) and not getattr(
                 self.lhs.field.target_field, "primary_key", False
             ):
-                self.rhs.clear_select_clause()
                 if (
                     getattr(self.lhs.output_field, "primary_key", False)
                     and self.lhs.output_field.model == self.rhs.model
@@ -105,7 +104,7 @@ def get_prep_lookup(self):
                     target_field = self.lhs.field.name
                 else:
                     target_field = self.lhs.field.target_field.name
-                self.rhs.add_fields([target_field], True)
+                self.rhs.set_values([target_field])
         return super().get_prep_lookup()
 
     def as_sql(self, compiler, connection):
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -198,6 +198,7 @@ class Query(BaseExpression):
     select_for_update_of = ()
     select_for_no_key_update = False
     select_related = False
+    has_select_fields = False
     # Arbitrary limit for select_related to prevents infinite recursion.
     max_depth = 5
     # Holds the selects defined by a call to values() or values_list()
@@ -263,12 +264,6 @@ def output_field(self):
         elif len(self.annotation_select) == 1:
             return next(iter(self.annotation_select.values())).output_field
 
-    @property
-    def has_select_fields(self):
-        return bool(
-            self.select or self.annotation_select_mask or self.extra_select_mask
-        )
-
     @cached_property
     def base_table(self):
         for alias in self.alias_map:
@@ -2384,6 +2379,7 @@ def set_values(self, fields):
         self.select_related = False
         self.clear_deferred_loading()
         self.clear_select_fields()
+        self.has_select_fields = True
 
         if fields:
             field_names = []

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/fields/related_lookups.py | 96 | 96 | 50 | 9 | 16530
| django/db/models/fields/related_lookups.py | 108 | 108 | 50 | 9 | 16530
| django/db/models/sql/query.py | 201 | 201 | 42 | 2 | 13958
| django/db/models/sql/query.py | 266 | 271 | 140 | 2 | 46625
| django/db/models/sql/query.py | 2387 | 2387 | 9 | 2 | 2553


## Problem Statement

```
__in doesn't clear selected fields on the RHS when QuerySet.alias() is used after annotate().
Description
	
Here is a test case to reproduce the bug, you can add this in tests/annotations/tests.py
	def test_annotation_and_alias_filter_in_subquery(self):
		long_books_qs = (
			Book.objects.filter(
				pages__gt=400,
			)
			.annotate(book_annotate=Value(1))
			.alias(book_alias=Value(1))
		)
		publisher_books_qs = (
			Publisher.objects.filter(
				book__in=long_books_qs
			)
			.values("name")
		)
		self.assertCountEqual(
			publisher_books_qs,
			[
				{'name': 'Apress'},
				{'name': 'Sams'},
				{'name': 'Prentice Hall'},
				{'name': 'Morgan Kaufmann'}
			]
		)
You should get this error:
django.db.utils.OperationalError: sub-select returns 10 columns - expected 1

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 1593 | 1646| 343 | 343 | 20450 | 
| 2 | **2 django/db/models/sql/query.py** | 1102 | 1119| 147 | 490 | 43357 | 
| 3 | **2 django/db/models/sql/query.py** | 389 | 441| 497 | 987 | 43357 | 
| 4 | 2 django/db/models/query.py | 1578 | 1591| 115 | 1102 | 43357 | 
| 5 | **2 django/db/models/sql/query.py** | 1923 | 1971| 445 | 1547 | 43357 | 
| 6 | **2 django/db/models/sql/query.py** | 2360 | 2381| 172 | 1719 | 43357 | 
| 7 | **2 django/db/models/sql/query.py** | 1601 | 1630| 287 | 2006 | 43357 | 
| 8 | 3 django/db/models/sql/where.py | 325 | 343| 149 | 2155 | 45868 | 
| **-> 9 <-** | **3 django/db/models/sql/query.py** | 2383 | 2432| 398 | 2553 | 45868 | 
| 10 | **3 django/db/models/sql/query.py** | 897 | 940| 445 | 2998 | 45868 | 
| 11 | 4 django/db/models/expressions.py | 1491 | 1510| 159 | 3157 | 59066 | 
| 12 | **4 django/db/models/sql/query.py** | 968 | 995| 272 | 3429 | 59066 | 
| 13 | 5 django/db/models/fields/related.py | 805 | 817| 113 | 3542 | 73676 | 
| 14 | 5 django/db/models/query.py | 1733 | 1762| 185 | 3727 | 73676 | 
| 15 | 5 django/db/models/expressions.py | 1095 | 1132| 298 | 4025 | 73676 | 
| 16 | 6 django/db/models/aggregates.py | 93 | 136| 361 | 4386 | 75151 | 
| 17 | **6 django/db/models/sql/query.py** | 997 | 1017| 158 | 4544 | 75151 | 
| 18 | **6 django/db/models/sql/query.py** | 2434 | 2466| 229 | 4773 | 75151 | 
| 19 | **6 django/db/models/sql/query.py** | 869 | 895| 280 | 5053 | 75151 | 
| 20 | **6 django/db/models/sql/query.py** | 1427 | 1508| 813 | 5866 | 75151 | 
| 21 | **6 django/db/models/sql/query.py** | 1121 | 1148| 247 | 6113 | 75151 | 
| 22 | **6 django/db/models/sql/query.py** | 1840 | 1870| 244 | 6357 | 75151 | 
| 23 | 7 django/db/backends/mysql/compiler.py | 1 | 22| 142 | 6499 | 75787 | 
| 24 | 8 django/db/models/query_utils.py | 391 | 431| 286 | 6785 | 78954 | 
| 25 | **9 django/db/models/fields/related_lookups.py** | 111 | 149| 324 | 7109 | 80564 | 
| 26 | **9 django/db/models/sql/query.py** | 1150 | 1179| 240 | 7349 | 80564 | 
| 27 | **9 django/db/models/sql/query.py** | 1648 | 1746| 846 | 8195 | 80564 | 
| 28 | 10 django/db/models/sql/compiler.py | 235 | 318| 682 | 8877 | 96721 | 
| 29 | 10 django/db/models/sql/compiler.py | 1512 | 1534| 254 | 9131 | 96721 | 
| 30 | 10 django/db/models/sql/where.py | 181 | 251| 501 | 9632 | 96721 | 
| 31 | **10 django/db/models/sql/query.py** | 1973 | 2040| 667 | 10299 | 96721 | 
| 32 | 10 django/db/models/expressions.py | 1135 | 1168| 238 | 10537 | 96721 | 
| 33 | **10 django/db/models/sql/query.py** | 942 | 966| 215 | 10752 | 96721 | 
| 34 | 11 django/db/models/lookups.py | 414 | 452| 336 | 11088 | 102039 | 
| 35 | **11 django/db/models/fields/related_lookups.py** | 1 | 40| 221 | 11309 | 102039 | 
| 36 | 11 django/db/models/query.py | 1311 | 1347| 263 | 11572 | 102039 | 
| 37 | **11 django/db/models/sql/query.py** | 797 | 830| 300 | 11872 | 102039 | 
| 38 | 11 django/db/models/sql/compiler.py | 1771 | 1803| 254 | 12126 | 102039 | 
| 39 | **11 django/db/models/sql/query.py** | 2073 | 2122| 332 | 12458 | 102039 | 
| 40 | **11 django/db/models/sql/query.py** | 2124 | 2173| 369 | 12827 | 102039 | 
| 41 | 11 django/db/models/query.py | 448 | 480| 249 | 13076 | 102039 | 
| **-> 42 <-** | **11 django/db/models/sql/query.py** | 158 | 264| 882 | 13958 | 102039 | 
| 43 | 12 django/db/models/fields/related_descriptors.py | 92 | 110| 196 | 14154 | 113335 | 
| 44 | **12 django/db/models/sql/query.py** | 679 | 719| 383 | 14537 | 113335 | 
| 45 | **12 django/db/models/sql/query.py** | 1904 | 1921| 135 | 14672 | 113335 | 
| 46 | **12 django/db/models/sql/query.py** | 1233 | 1267| 339 | 15011 | 113335 | 
| 47 | 12 django/db/models/expressions.py | 1432 | 1488| 436 | 15447 | 113335 | 
| 48 | 12 django/db/models/query.py | 1112 | 1146| 306 | 15753 | 113335 | 
| 49 | **12 django/db/models/sql/query.py** | 832 | 867| 398 | 16151 | 113335 | 
| **-> 50 <-** | **12 django/db/models/fields/related_lookups.py** | 75 | 109| 379 | 16530 | 113335 | 
| 51 | 12 django/db/models/query.py | 1874 | 1892| 186 | 16716 | 113335 | 
| 52 | **12 django/db/models/sql/query.py** | 2228 | 2272| 355 | 17071 | 113335 | 
| 53 | 12 django/db/models/expressions.py | 1307 | 1335| 266 | 17337 | 113335 | 
| 54 | **12 django/db/models/sql/query.py** | 1 | 79| 576 | 17913 | 113335 | 
| 55 | 12 django/db/models/lookups.py | 454 | 487| 344 | 18257 | 113335 | 
| 56 | 12 django/db/models/sql/compiler.py | 594 | 660| 563 | 18820 | 113335 | 
| 57 | 12 django/db/models/query.py | 482 | 499| 146 | 18966 | 113335 | 
| 58 | **12 django/db/models/sql/query.py** | 721 | 756| 362 | 19328 | 113335 | 
| 59 | **12 django/db/models/sql/query.py** | 1510 | 1536| 263 | 19591 | 113335 | 
| 60 | 12 django/db/models/expressions.py | 1512 | 1539| 224 | 19815 | 113335 | 
| 61 | **12 django/db/models/sql/query.py** | 1069 | 1100| 307 | 20122 | 113335 | 
| 62 | 12 django/db/models/sql/compiler.py | 1352 | 1395| 349 | 20471 | 113335 | 
| 63 | 12 django/db/models/query_utils.py | 349 | 359| 116 | 20587 | 113335 | 
| 64 | 12 django/db/models/lookups.py | 128 | 164| 270 | 20857 | 113335 | 
| 65 | **12 django/db/models/sql/query.py** | 1019 | 1067| 438 | 21295 | 113335 | 
| 66 | **12 django/db/models/sql/query.py** | 2307 | 2334| 291 | 21586 | 113335 | 
| 67 | 12 django/db/models/sql/compiler.py | 1171 | 1289| 928 | 22514 | 113335 | 
| 68 | **12 django/db/models/sql/query.py** | 1872 | 1902| 282 | 22796 | 113335 | 
| 69 | 13 django/db/models/__init__.py | 1 | 116| 682 | 23478 | 114017 | 
| 70 | 14 django/db/models/sql/subqueries.py | 142 | 172| 190 | 23668 | 115245 | 
| 71 | **14 django/db/models/fields/related_lookups.py** | 171 | 211| 250 | 23918 | 115245 | 
| 72 | 14 django/db/models/query.py | 1906 | 1971| 539 | 24457 | 115245 | 
| 73 | 14 django/db/models/query.py | 1262 | 1309| 357 | 24814 | 115245 | 
| 74 | **14 django/db/models/sql/query.py** | 589 | 604| 136 | 24950 | 115245 | 
| 75 | 14 django/db/models/query.py | 350 | 377| 221 | 25171 | 115245 | 
| 76 | 15 django/db/models/fields/__init__.py | 470 | 491| 169 | 25340 | 133980 | 
| 77 | 15 django/db/models/sql/compiler.py | 30 | 67| 376 | 25716 | 133980 | 
| 78 | 15 django/db/models/query.py | 1712 | 1731| 209 | 25925 | 133980 | 
| 79 | 15 django/db/models/query.py | 1894 | 1904| 114 | 26039 | 133980 | 
| 80 | **15 django/db/models/sql/query.py** | 1748 | 1839| 786 | 26825 | 133980 | 
| 81 | 15 django/db/models/query.py | 1487 | 1496| 126 | 26951 | 133980 | 
| 82 | 15 django/db/models/expressions.py | 1397 | 1429| 302 | 27253 | 133980 | 
| 83 | **15 django/db/models/sql/query.py** | 606 | 678| 835 | 28088 | 133980 | 
| 84 | **15 django/db/models/sql/query.py** | 758 | 773| 146 | 28234 | 133980 | 
| 85 | 15 django/db/models/query.py | 1148 | 1170| 156 | 28390 | 133980 | 
| 86 | 15 django/db/models/query.py | 607 | 624| 131 | 28521 | 133980 | 
| 87 | 15 django/db/models/fields/related_descriptors.py | 427 | 450| 194 | 28715 | 133980 | 
| 88 | 15 django/db/models/query.py | 2033 | 2139| 725 | 29440 | 133980 | 
| 89 | 15 django/db/models/query.py | 1474 | 1485| 120 | 29560 | 133980 | 
| 90 | 15 django/db/models/query.py | 1204 | 1225| 188 | 29748 | 133980 | 
| 91 | 16 django/contrib/admin/filters.py | 227 | 250| 202 | 29950 | 138249 | 
| 92 | 16 django/db/models/aggregates.py | 60 | 91| 314 | 30264 | 138249 | 
| 93 | 16 django/db/models/query.py | 1498 | 1511| 126 | 30390 | 138249 | 
| 94 | 16 django/db/models/query.py | 684 | 743| 479 | 30869 | 138249 | 
| 95 | 16 django/db/models/query.py | 2016 | 2031| 147 | 31016 | 138249 | 
| 96 | 17 django/db/backends/ddl_references.py | 115 | 135| 167 | 31183 | 139878 | 
| 97 | 17 django/db/models/fields/related.py | 154 | 185| 209 | 31392 | 139878 | 
| 98 | 17 django/db/models/sql/compiler.py | 927 | 951| 211 | 31603 | 139878 | 
| 99 | 17 django/db/models/sql/subqueries.py | 1 | 45| 311 | 31914 | 139878 | 
| 100 | 18 django/contrib/postgres/constraints.py | 196 | 236| 367 | 32281 | 141835 | 
| 101 | **18 django/db/models/sql/query.py** | 2468 | 2543| 781 | 33062 | 141835 | 
| 102 | 18 django/db/models/sql/compiler.py | 1016 | 1029| 166 | 33228 | 141835 | 
| 103 | 18 django/db/models/sql/compiler.py | 320 | 422| 723 | 33951 | 141835 | 
| 104 | 19 django/contrib/gis/db/models/__init__.py | 1 | 31| 216 | 34167 | 142051 | 
| 105 | **19 django/db/models/sql/query.py** | 775 | 795| 201 | 34368 | 142051 | 
| 106 | 19 django/contrib/admin/filters.py | 457 | 485| 223 | 34591 | 142051 | 
| 107 | 20 django/contrib/postgres/aggregates/general.py | 60 | 95| 233 | 34824 | 142871 | 
| 108 | **20 django/db/models/sql/query.py** | 1538 | 1570| 262 | 35086 | 142871 | 
| 109 | **20 django/db/models/sql/query.py** | 1181 | 1199| 154 | 35240 | 142871 | 
| 110 | 20 django/db/models/fields/related.py | 777 | 803| 222 | 35462 | 142871 | 
| 111 | **20 django/db/models/sql/query.py** | 2612 | 2668| 823 | 36285 | 142871 | 
| 112 | 20 django/contrib/admin/filters.py | 519 | 551| 236 | 36521 | 142871 | 
| 113 | **20 django/db/models/sql/query.py** | 553 | 587| 286 | 36807 | 142871 | 
| 114 | 20 django/db/models/query.py | 1648 | 1693| 340 | 37147 | 142871 | 
| 115 | 20 django/db/models/fields/related.py | 226 | 301| 696 | 37843 | 142871 | 
| 116 | 20 django/db/models/query.py | 1407 | 1455| 371 | 38214 | 142871 | 
| 117 | **20 django/db/models/fields/related_lookups.py** | 152 | 169| 216 | 38430 | 142871 | 
| 118 | 20 django/db/models/lookups.py | 296 | 309| 157 | 38587 | 142871 | 
| 119 | 20 django/db/models/fields/related.py | 1450 | 1579| 984 | 39571 | 142871 | 
| 120 | 21 django/db/backends/postgresql/features.py | 1 | 103| 792 | 40363 | 143663 | 
| 121 | 21 django/db/models/aggregates.py | 25 | 58| 300 | 40663 | 143663 | 
| 122 | 21 django/db/models/query.py | 1529 | 1551| 158 | 40821 | 143663 | 
| 123 | 21 django/db/models/query.py | 501 | 518| 132 | 40953 | 143663 | 
| 124 | 21 django/db/models/sql/compiler.py | 953 | 1014| 508 | 41461 | 143663 | 
| 125 | 21 django/db/models/query.py | 1172 | 1202| 251 | 41712 | 143663 | 
| 126 | **21 django/db/models/sql/query.py** | 2545 | 2561| 177 | 41889 | 143663 | 
| 127 | 21 django/db/models/lookups.py | 166 | 186| 200 | 42089 | 143663 | 
| 128 | 21 django/db/models/sql/compiler.py | 1068 | 1169| 743 | 42832 | 143663 | 
| 129 | 21 django/db/models/sql/compiler.py | 1291 | 1314| 207 | 43039 | 143663 | 
| 130 | 22 django/forms/models.py | 1421 | 1503| 559 | 43598 | 155857 | 
| 131 | 22 django/db/models/fields/related.py | 302 | 339| 296 | 43894 | 155857 | 
| 132 | 23 django/db/models/sql/datastructures.py | 25 | 68| 365 | 44259 | 157350 | 
| 133 | 23 django/db/models/sql/subqueries.py | 48 | 78| 212 | 44471 | 157350 | 
| 134 | 24 django/contrib/admin/views/main.py | 554 | 586| 227 | 44698 | 161881 | 
| 135 | 24 django/db/models/sql/compiler.py | 520 | 592| 642 | 45340 | 161881 | 
| 136 | 24 django/db/models/query.py | 379 | 409| 246 | 45586 | 161881 | 
| 137 | 24 django/db/models/sql/compiler.py | 490 | 518| 242 | 45828 | 161881 | 
| 138 | 24 django/db/models/fields/__init__.py | 443 | 468| 198 | 46026 | 161881 | 
| 139 | 24 django/db/models/expressions.py | 1605 | 1639| 243 | 46269 | 161881 | 
| **-> 140 <-** | **24 django/db/models/sql/query.py** | 266 | 317| 356 | 46625 | 161881 | 
| 141 | 24 django/db/models/fields/related_descriptors.py | 153 | 197| 418 | 47043 | 161881 | 
| 142 | 24 django/db/models/expressions.py | 996 | 1022| 258 | 47301 | 161881 | 
| 143 | 24 django/db/models/query.py | 1695 | 1710| 149 | 47450 | 161881 | 
| 144 | 24 django/db/models/lookups.py | 343 | 357| 170 | 47620 | 161881 | 
| 145 | 24 django/db/models/query_utils.py | 112 | 147| 316 | 47936 | 161881 | 
| 146 | 24 django/db/models/sql/compiler.py | 174 | 233| 543 | 48479 | 161881 | 
| 147 | 24 django/db/models/expressions.py | 1053 | 1067| 120 | 48599 | 161881 | 
| 148 | 24 django/db/models/fields/__init__.py | 376 | 409| 214 | 48813 | 161881 | 
| 149 | 24 django/db/models/query.py | 187 | 207| 142 | 48955 | 161881 | 
| 150 | **24 django/db/models/sql/query.py** | 1572 | 1599| 220 | 49175 | 161881 | 
| 151 | 24 django/db/models/sql/compiler.py | 480 | 488| 133 | 49308 | 161881 | 
| 152 | 24 django/db/models/fields/related_descriptors.py | 674 | 712| 350 | 49658 | 161881 | 
| 153 | 24 django/db/models/query.py | 1 | 39| 274 | 49932 | 161881 | 
| 154 | 24 django/db/models/expressions.py | 830 | 871| 297 | 50229 | 161881 | 
| 155 | 24 django/db/models/sql/compiler.py | 1329 | 1350| 199 | 50428 | 161881 | 
| 156 | 24 django/contrib/admin/filters.py | 130 | 156| 205 | 50633 | 161881 | 
| 157 | 24 django/db/models/sql/compiler.py | 85 | 172| 902 | 51535 | 161881 | 
| 158 | 25 django/db/models/base.py | 1705 | 1758| 490 | 52025 | 180432 | 
| 159 | 25 django/contrib/admin/views/main.py | 495 | 552| 465 | 52490 | 180432 | 
| 160 | 25 django/db/models/base.py | 1760 | 1783| 176 | 52666 | 180432 | 
| 161 | 25 django/db/models/fields/related.py | 1418 | 1448| 172 | 52838 | 180432 | 
| 162 | 25 django/db/models/fields/__init__.py | 600 | 618| 182 | 53020 | 180432 | 
| 163 | 25 django/contrib/admin/filters.py | 180 | 225| 427 | 53447 | 180432 | 
| 164 | 25 django/contrib/postgres/aggregates/general.py | 41 | 57| 164 | 53611 | 180432 | 
| 165 | 25 django/contrib/admin/filters.py | 315 | 345| 229 | 53840 | 180432 | 
| 166 | 25 django/db/models/base.py | 2050 | 2155| 736 | 54576 | 180432 | 
| 167 | 25 django/db/models/base.py | 1845 | 1872| 187 | 54763 | 180432 | 
| 168 | 26 django/db/migrations/autodetector.py | 90 | 102| 119 | 54882 | 193893 | 
| 169 | 26 django/db/backends/ddl_references.py | 223 | 255| 252 | 55134 | 193893 | 
| 170 | 26 django/db/models/base.py | 1904 | 1993| 675 | 55809 | 193893 | 
| 171 | 26 django/db/models/base.py | 1995 | 2048| 355 | 56164 | 193893 | 
| 172 | 26 django/db/models/query.py | 249 | 284| 217 | 56381 | 193893 | 
| 173 | 27 django/db/models/deletion.py | 1 | 93| 601 | 56982 | 197875 | 


### Hint

```
This is a ​documented and expected behavior.
Using either .alias or .annotate works as expected without using .values to limit to 1 column. Why is that? but using both of them doesn't seem work.
Replying to Gabriel Muj: Using either .alias or .annotate works as expected without using .values to limit to 1 column. Why is that? but using both of them doesn't seem work. We have some simplification in the __in lookup that automatically limit selected fields to the pk when Query is on the right-side and the selected fields are undefined. It looks like they don't play nice when .annotate() and .alias() are mixed. Tentatively accepted for the investigation.
Whether or not the In lookups defaults a Query right-hand-side to .values('pk') is based of rhs.has_select_fields (​source) This dynamic property is based of self.select or self.annotation_select_mask so I suspect that there might be some bad logic in Query.add_annotation (​source) from the introduction of QuerySet.alias in f4ac167119e8897c398527c392ed117326496652. The self.set_annotation_mask(set(self.annotation_select).difference({alias})) ​line seems to be what causes the problem here. If annotate and alias are used then annotation_select_mask will be materialized as a mask is required to keep track of which entries in annotations must be selected and whatnot (not necessary if only using alias or annotate as the first doesn't require any selecting and the second selects all annotations) The add_annotation logic relied on to implement QuerySet.alias is not wrong per se it just happens to piggy back on the annotation masking logic that only Query.set_values relied on previously an now happens to break the heuristics in Query.has_select_fields. The proper solutions is likely to replace Query.has_select_fields by a class attribute that defaults to False and have set_values set it to True and make sure Query.clone carries the attribute over. This seems like a more durable options as that avoids trying to peek into internals to determine if a values call was performed. Gabriel, since you've already written a regression test, would you be interested in submitting a patch that takes the approach proposed above?
hi, I am new to Django contribution, i tried to implement the solution proposed by simon, replaced the Query.has_select_fields with has_select_fields attribute with default value False and have set_values set it to true, but now i am stuck at carrying over the attribute to Query.clone , can I get a bit more explanation for the same? Edit: Replacing Query_has_select_fields with attribute and having set_values set it to true passes tests for annotations but giving 2 Failures on overall tests (queries).Failures are: FAIL: test_in_subquery (queries.tests.ToFieldTests) AssertionError: Items in the second set but not the first: <Eaten: apple at lunch> FAIL: test_nested_in_subquery (queries.tests.ToFieldTests) AssertionError: Sequences differ: <QuerySet []> != [<ReportComment: ReportComment object (1)>] Second sequence contains 1 additional elements. First extra element 0: <ReportComment: ReportComment object (1)>
If you're still struggling with the clone part of the patch it basically means that Query.clone must assign obj.has_select_fields = True when if self.has_select_fields This should not be necessary as it's a shallow attribute and it already gets carried over. The tests are failing because RelatedIn also needs adjustments to make sure it goes through Query.set_values instead of calling clear_select_clause and add_fields django/db/models/fields/related_lookups.py diff --git a/django/db/models/fields/related_lookups.py b/django/db/models/fields/related_lookups.py index 1a845a1f7f..afea09b5a9 100644 a b def get_prep_lookup(self): 9393 elif not getattr(self.rhs, "has_select_fields", True) and not getattr( 9494 self.lhs.field.target_field, "primary_key", False 9595 ): 96 self.rhs.clear_select_clause() 9796 if ( 9897 getattr(self.lhs.output_field, "primary_key", False) 9998 and self.lhs.output_field.model == self.rhs.model … … def get_prep_lookup(self): 105104 target_field = self.lhs.field.name 106105 else: 107106 target_field = self.lhs.field.target_field.name 108 self.rhs.add_fields([target_field], True) 107 self.rhs.set_values([target_field]) 109108 return super().get_prep_lookup() 110109 111110 def as_sql(self, compiler, connection):
Thanks, that worked! Opening a PR.
```

## Patch

```diff
diff --git a/django/db/models/fields/related_lookups.py b/django/db/models/fields/related_lookups.py
--- a/django/db/models/fields/related_lookups.py
+++ b/django/db/models/fields/related_lookups.py
@@ -93,7 +93,6 @@ def get_prep_lookup(self):
             elif not getattr(self.rhs, "has_select_fields", True) and not getattr(
                 self.lhs.field.target_field, "primary_key", False
             ):
-                self.rhs.clear_select_clause()
                 if (
                     getattr(self.lhs.output_field, "primary_key", False)
                     and self.lhs.output_field.model == self.rhs.model
@@ -105,7 +104,7 @@ def get_prep_lookup(self):
                     target_field = self.lhs.field.name
                 else:
                     target_field = self.lhs.field.target_field.name
-                self.rhs.add_fields([target_field], True)
+                self.rhs.set_values([target_field])
         return super().get_prep_lookup()
 
     def as_sql(self, compiler, connection):
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -198,6 +198,7 @@ class Query(BaseExpression):
     select_for_update_of = ()
     select_for_no_key_update = False
     select_related = False
+    has_select_fields = False
     # Arbitrary limit for select_related to prevents infinite recursion.
     max_depth = 5
     # Holds the selects defined by a call to values() or values_list()
@@ -263,12 +264,6 @@ def output_field(self):
         elif len(self.annotation_select) == 1:
             return next(iter(self.annotation_select.values())).output_field
 
-    @property
-    def has_select_fields(self):
-        return bool(
-            self.select or self.annotation_select_mask or self.extra_select_mask
-        )
-
     @cached_property
     def base_table(self):
         for alias in self.alias_map:
@@ -2384,6 +2379,7 @@ def set_values(self, fields):
         self.select_related = False
         self.clear_deferred_loading()
         self.clear_select_fields()
+        self.has_select_fields = True
 
         if fields:
             field_names = []

```

## Test Patch

```diff
diff --git a/tests/annotations/tests.py b/tests/annotations/tests.py
--- a/tests/annotations/tests.py
+++ b/tests/annotations/tests.py
@@ -989,6 +989,34 @@ def test_annotation_filter_with_subquery(self):
             publisher_books_qs, [{"name": "Sams"}, {"name": "Morgan Kaufmann"}]
         )
 
+    def test_annotation_and_alias_filter_in_subquery(self):
+        awarded_publishers_qs = (
+            Publisher.objects.filter(num_awards__gt=4)
+            .annotate(publisher_annotate=Value(1))
+            .alias(publisher_alias=Value(1))
+        )
+        qs = Publisher.objects.filter(pk__in=awarded_publishers_qs)
+        self.assertCountEqual(qs, [self.p3, self.p4])
+
+    def test_annotation_and_alias_filter_related_in_subquery(self):
+        long_books_qs = (
+            Book.objects.filter(pages__gt=400)
+            .annotate(book_annotate=Value(1))
+            .alias(book_alias=Value(1))
+        )
+        publisher_books_qs = Publisher.objects.filter(
+            book__in=long_books_qs,
+        ).values("name")
+        self.assertCountEqual(
+            publisher_books_qs,
+            [
+                {"name": "Apress"},
+                {"name": "Sams"},
+                {"name": "Prentice Hall"},
+                {"name": "Morgan Kaufmann"},
+            ],
+        )
+
     def test_annotation_exists_aggregate_values_chaining(self):
         qs = (
             Book.objects.values("publisher")

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 1593, End line: 1646

```python
class QuerySet:

    def _annotate(self, args, kwargs, select=True):
        self._validate_values_are_expressions(
            args + tuple(kwargs.values()), method_name="annotate"
        )
        annotations = {}
        for arg in args:
            # The default_alias property may raise a TypeError.
            try:
                if arg.default_alias in kwargs:
                    raise ValueError(
                        "The named annotation '%s' conflicts with the "
                        "default name for another annotation." % arg.default_alias
                    )
            except TypeError:
                raise TypeError("Complex annotations require an alias")
            annotations[arg.default_alias] = arg
        annotations.update(kwargs)

        clone = self._chain()
        names = self._fields
        if names is None:
            names = set(
                chain.from_iterable(
                    (field.name, field.attname)
                    if hasattr(field, "attname")
                    else (field.name,)
                    for field in self.model._meta.get_fields()
                )
            )

        for alias, annotation in annotations.items():
            if alias in names:
                raise ValueError(
                    "The annotation '%s' conflicts with a field on "
                    "the model." % alias
                )
            if isinstance(annotation, FilteredRelation):
                clone.query.add_filtered_relation(annotation, alias)
            else:
                clone.query.add_annotation(
                    annotation,
                    alias,
                    is_summary=False,
                    select=select,
                )
        for alias, annotation in clone.query.annotations.items():
            if alias in annotations and annotation.contains_aggregate:
                if clone._fields is None:
                    clone.query.group_by = True
                else:
                    clone.query.set_group_by()
                break

        return clone
```
### 2 - django/db/models/sql/query.py:

Start line: 1102, End line: 1119

```python
class Query(BaseExpression):

    def check_alias(self, alias):
        if FORBIDDEN_ALIAS_PATTERN.search(alias):
            raise ValueError(
                "Column aliases cannot contain whitespace characters, quotation marks, "
                "semicolons, or SQL comments."
            )

    def add_annotation(self, annotation, alias, is_summary=False, select=True):
        """Add a single annotation expression to the Query."""
        self.check_alias(alias)
        annotation = annotation.resolve_expression(
            self, allow_joins=True, reuse=None, summarize=is_summary
        )
        if select:
            self.append_annotation_mask([alias])
        else:
            self.set_annotation_mask(set(self.annotation_select).difference({alias}))
        self.annotations[alias] = annotation
```
### 3 - django/db/models/sql/query.py:

Start line: 389, End line: 441

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
                    if isinstance(expr, Col) or (
                        expr.contains_aggregate and not expr.is_summary
                    ):
                        # Reference column or another aggregate. Select it
                        # under a non-conflicting alias.
                        col_cnt += 1
                        col_alias = "__col%d" % col_cnt
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
### 4 - django/db/models/query.py:

Start line: 1578, End line: 1591

```python
class QuerySet:

    def annotate(self, *args, **kwargs):
        """
        Return a query set in which the returned objects have been annotated
        with extra data or aggregations.
        """
        self._not_support_combined_queries("annotate")
        return self._annotate(args, kwargs, select=True)

    def alias(self, *args, **kwargs):
        """
        Return a query set with added aliases for extra data or aggregations.
        """
        self._not_support_combined_queries("alias")
        return self._annotate(args, kwargs, select=False)
```
### 5 - django/db/models/sql/query.py:

Start line: 1923, End line: 1971

```python
class Query(BaseExpression):

    def resolve_ref(self, name, allow_joins=True, reuse=None, summarize=False):
        annotation = self.annotations.get(name)
        if annotation is not None:
            if not allow_joins:
                for alias in self._gen_col_aliases([annotation]):
                    if isinstance(self.alias_map[alias], Join):
                        raise FieldError(
                            "Joined field references are not permitted in this query"
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
            join_info = self.setup_joins(
                field_list, self.get_meta(), self.get_initial_alias(), can_reuse=reuse
            )
            targets, final_alias, join_list = self.trim_joins(
                join_info.targets, join_info.joins, join_info.path
            )
            if not allow_joins and len(join_list) > 1:
                raise FieldError(
                    "Joined field references are not permitted in this query"
                )
            if len(targets) > 1:
                raise FieldError(
                    "Referencing multicolumn fields with F() objects isn't supported"
                )
            # Verify that the last lookup in name is a field or a transform:
            # transform_function() raises FieldError if not.
            transform = join_info.transform_function(targets[0], final_alias)
            if reuse is not None:
                reuse.update(join_list)
            return transform
```
### 6 - django/db/models/sql/query.py:

Start line: 2360, End line: 2381

```python
class Query(BaseExpression):

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
### 7 - django/db/models/sql/query.py:

Start line: 1601, End line: 1630

```python
class Query(BaseExpression):

    def add_filtered_relation(self, filtered_relation, alias):
        filtered_relation.alias = alias
        lookups = dict(get_children_from_q(filtered_relation.condition))
        relation_lookup_parts, relation_field_parts, _ = self.solve_lookup_type(
            filtered_relation.relation_name
        )
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
### 8 - django/db/models/sql/where.py:

Start line: 325, End line: 343

```python
class SubqueryConstraint:
    # Even if aggregates or windows would be used in a subquery,
    # the outer query isn't interested about those.
    contains_aggregate = False
    contains_over_clause = False

    def __init__(self, alias, columns, targets, query_object):
        self.alias = alias
        self.columns = columns
        self.targets = targets
        query_object.clear_ordering(clear_default=True)
        self.query_object = query_object

    def as_sql(self, compiler, connection):
        query = self.query_object
        query.set_values(self.targets)
        query_compiler = query.get_compiler(connection=connection)
        return query_compiler.as_subquery_condition(self.alias, self.columns, compiler)
```
### 9 - django/db/models/sql/query.py:

Start line: 2383, End line: 2432

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
            self.add_fields(
                (f.attname for f in self.model._meta.concrete_fields), False
            )
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
### 10 - django/db/models/sql/query.py:

Start line: 897, End line: 940

```python
class Query(BaseExpression):

    def change_aliases(self, change_map):
        """
        Change the aliases in change_map (which maps old-alias -> new-alias),
        relabelling any references to them in select columns and the where
        clause.
        """
        # If keys and values of change_map were to intersect, an alias might be
        # updated twice (e.g. T4 -> T5, T5 -> T6, so also T4 -> T6) depending
        # on their order in change_map.
        assert set(change_map).isdisjoint(change_map.values())

        # 1. Update references in "select" (normal columns plus aliases),
        # "group by" and "where".
        self.where.relabel_aliases(change_map)
        if isinstance(self.group_by, tuple):
            self.group_by = tuple(
                [col.relabeled_clone(change_map) for col in self.group_by]
            )
        self.select = tuple([col.relabeled_clone(change_map) for col in self.select])
        self.annotations = self.annotations and {
            key: col.relabeled_clone(change_map)
            for key, col in self.annotations.items()
        }

        # 2. Rename the alias in the internal table/alias datastructures.
        for old_alias, new_alias in change_map.items():
            if old_alias not in self.alias_map:
                continue
            alias_data = self.alias_map[old_alias].relabeled_clone(change_map)
            self.alias_map[new_alias] = alias_data
            self.alias_refcount[new_alias] = self.alias_refcount[old_alias]
            del self.alias_refcount[old_alias]
            del self.alias_map[old_alias]

            table_aliases = self.table_map[alias_data.table_name]
            for pos, alias in enumerate(table_aliases):
                if alias == old_alias:
                    table_aliases[pos] = new_alias
                    break
        self.external_aliases = {
            # Table is aliased or it's being changed and thus is aliased.
            change_map.get(alias, alias): (aliased or alias in change_map)
            for alias, aliased in self.external_aliases.items()
        }
```
### 12 - django/db/models/sql/query.py:

Start line: 968, End line: 995

```python
class Query(BaseExpression):

    def bump_prefix(self, other_query, exclude=None):
        # ... other code

        if self.alias_prefix != other_query.alias_prefix:
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
                    "Maximum recursion depth exceeded: too many subqueries."
                )
        self.subq_aliases = self.subq_aliases.union([self.alias_prefix])
        other_query.subq_aliases = other_query.subq_aliases.union(self.subq_aliases)
        if exclude is None:
            exclude = {}
        self.change_aliases(
            {
                alias: "%s%d" % (self.alias_prefix, pos)
                for pos, alias in enumerate(self.alias_map)
                if alias not in exclude
            }
        )
```
### 17 - django/db/models/sql/query.py:

Start line: 997, End line: 1017

```python
class Query(BaseExpression):

    def get_initial_alias(self):
        """
        Return the first alias for this query, after increasing its reference
        count.
        """
        if self.alias_map:
            alias = self.base_table
            self.ref_alias(alias)
        elif self.model:
            alias = self.join(self.base_table_class(self.get_meta().db_table, None))
        else:
            alias = None
        return alias

    def count_active_tables(self):
        """
        Return the number of tables in this query with a non-zero reference
        count. After execution, the reference counts are zeroed, so tables
        added in compiler will not be seen by this method.
        """
        return len([1 for count in self.alias_refcount.values() if count])
```
### 18 - django/db/models/sql/query.py:

Start line: 2434, End line: 2466

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
                k: v
                for k, v in self.annotations.items()
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
                k: v for k, v in self.extra.items() if k in self.extra_select_mask
            }
            return self._extra_select_cache
        else:
            return self.extra
```
### 19 - django/db/models/sql/query.py:

Start line: 869, End line: 895

```python
class Query(BaseExpression):

    def demote_joins(self, aliases):
        """
        Change join type from LOUTER to INNER for all joins in aliases.

        Similarly to promote_joins(), this method must ensure no join chains
        containing first an outer, then an inner join are generated. If we
        are demoting b->c join in chain a LOUTER b LOUTER c then we must
        demote a->b automatically, or otherwise the demotion of b->c doesn't
        actually change anything in the query results. .
        """
        aliases = list(aliases)
        while aliases:
            alias = aliases.pop(0)
            if self.alias_map[alias].join_type == LOUTER:
                self.alias_map[alias] = self.alias_map[alias].demote()
                parent_alias = self.alias_map[alias].parent_alias
                if self.alias_map[parent_alias].join_type == INNER:
                    aliases.append(parent_alias)

    def reset_refcounts(self, to_counts):
        """
        Reset reference counts for aliases so that they match the value passed
        in `to_counts`.
        """
        for alias, cur_refcount in self.alias_refcount.copy().items():
            unref_amount = cur_refcount - to_counts.get(alias, 0)
            self.unref_alias(alias, unref_amount)
```
### 20 - django/db/models/sql/query.py:

Start line: 1427, End line: 1508

```python
class Query(BaseExpression):

    def build_filter(
        self,
        filter_expr,
        branch_negated=False,
        current_negated=False,
        can_reuse=None,
        allow_joins=True,
        split_subq=True,
        reuse_with_filtered_relation=False,
        check_filterable=True,
    ):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts,
                opts,
                alias,
                can_reuse=can_reuse,
                allow_many=allow_many,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
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
        targets, alias, join_list = self.trim_joins(
            join_info.targets, join_info.joins, join_info.path
        )
        if can_reuse is not None:
            can_reuse.update(join_list)

        if join_info.final_field.is_relation:
            # No support for transforms for relational fields
            num_lookups = len(lookups)
            if num_lookups > 1:
                raise FieldError(
                    "Related Field got invalid lookup: {}".format(lookups[0])
                )
            if len(targets) == 1:
                col = self._get_col(targets[0], join_info.final_field, alias)
            else:
                col = MultiColSource(
                    alias, targets, join_info.targets, join_info.final_field
                )
        else:
            col = self._get_col(targets[0], join_info.final_field, alias)

        condition = self.build_lookup(lookups, col, value)
        lookup_type = condition.lookup_name
        clause = WhereNode([condition], connector=AND)

        require_outer = (
            lookup_type == "isnull" and condition.rhs is True and not current_negated
        )
        if (
            current_negated
            and (lookup_type != "isnull" or condition.rhs is False)
            and condition.rhs is not None
        ):
            require_outer = True
            if lookup_type != "isnull":
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
                    self.is_nullable(targets[0])
                    or self.alias_map[join_list[-1]].join_type == LOUTER
                ):
                    lookup_class = targets[0].get_lookup("isnull")
                    col = self._get_col(targets[0], join_info.targets[0], alias)
                    clause.add(lookup_class(col, False), AND)
                # If someval is a nullable column, someval IS NOT NULL is
                # added.
                if isinstance(value, Col) and self.is_nullable(value.target):
                    lookup_class = value.target.get_lookup("isnull")
                    clause.add(lookup_class(value, False), AND)
        return clause, used_joins if not require_outer else ()
```
### 21 - django/db/models/sql/query.py:

Start line: 1121, End line: 1148

```python
class Query(BaseExpression):

    def resolve_expression(self, query, *args, **kwargs):
        clone = self.clone()
        # Subqueries need to use a different set of aliases than the outer query.
        clone.bump_prefix(query)
        clone.subquery = True
        clone.where.resolve_expression(query, *args, **kwargs)
        # Resolve combined queries.
        if clone.combinator:
            clone.combined_queries = tuple(
                [
                    combined_query.resolve_expression(query, *args, **kwargs)
                    for combined_query in clone.combined_queries
                ]
            )
        for key, value in clone.annotations.items():
            resolved = value.resolve_expression(query, *args, **kwargs)
            if hasattr(resolved, "external_aliases"):
                resolved.external_aliases.update(clone.external_aliases)
            clone.annotations[key] = resolved
        # Outer query's aliases are considered external.
        for alias, table in query.alias_map.items():
            clone.external_aliases[alias] = (
                isinstance(table, Join)
                and table.join_field.related_model._meta.db_table != alias
            ) or (
                isinstance(table, BaseTable) and table.table_name != table.table_alias
            )
        return clone
```
### 22 - django/db/models/sql/query.py:

Start line: 1840, End line: 1870

```python
class Query(BaseExpression):

    def setup_joins(
        self,
        names,
        opts,
        alias,
        can_reuse=None,
        allow_many=True,
        reuse_with_filtered_relation=False,
    ):
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
            connection = self.join_class(
                opts.db_table,
                alias,
                table_alias,
                INNER,
                join.join_field,
                nullable,
                filtered_relation=filtered_relation,
            )
            reuse = can_reuse if join.m2m or reuse_with_filtered_relation else None
            alias = self.join(
                connection,
                reuse=reuse,
                reuse_with_filtered_relation=reuse_with_filtered_relation,
            )
            joins.append(alias)
            if filtered_relation:
                filtered_relation.path = joins[:]
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 25 - django/db/models/fields/related_lookups.py:

Start line: 111, End line: 149

```python
class RelatedIn(In):

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            # For multicolumn lookups we need to build a multicolumn where clause.
            # This clause is either a SubqueryConstraint (for values that need
            # to be compiled to SQL) or an OR-combined list of
            # (col1 = val1 AND col2 = val2 AND ...) clauses.
            from django.db.models.sql.where import (
                AND,
                OR,
                SubqueryConstraint,
                WhereNode,
            )

            root_constraint = WhereNode(connector=OR)
            if self.rhs_is_direct_value():
                values = [get_normalized_value(value, self.lhs) for value in self.rhs]
                for value in values:
                    value_constraint = WhereNode()
                    for source, target, val in zip(
                        self.lhs.sources, self.lhs.targets, value
                    ):
                        lookup_class = target.get_lookup("exact")
                        lookup = lookup_class(
                            target.get_col(self.lhs.alias, source), val
                        )
                        value_constraint.add(lookup, AND)
                    root_constraint.add(value_constraint, OR)
            else:
                root_constraint.add(
                    SubqueryConstraint(
                        self.lhs.alias,
                        [target.column for target in self.lhs.targets],
                        [source.name for source in self.lhs.sources],
                        self.rhs,
                    ),
                    AND,
                )
            return root_constraint.as_sql(compiler, connection)
        return super().as_sql(compiler, connection)
```
### 26 - django/db/models/sql/query.py:

Start line: 1150, End line: 1179

```python
class Query(BaseExpression):

    def get_external_cols(self):
        exprs = chain(self.annotations.values(), self.where.children)
        return [
            col
            for col in self._gen_cols(exprs, include_external=True)
            if col.alias in self.external_aliases
        ]

    def get_group_by_cols(self, alias=None):
        if alias:
            return [Ref(alias, self)]
        external_cols = self.get_external_cols()
        if any(col.possibly_multivalued for col in external_cols):
            return [self]
        return external_cols

    def as_sql(self, compiler, connection):
        # Some backends (e.g. Oracle) raise an error when a subquery contains
        # unnecessary ORDER BY clause.
        if (
            self.subquery
            and not connection.features.ignores_unnecessary_order_by_in_subqueries
        ):
            self.clear_ordering(force=False)
            for query in self.combined_queries:
                query.clear_ordering(force=False)
        sql, params = self.get_compiler(connection=connection).as_sql()
        if self.subquery:
            sql = "(%s)" % sql
        return sql, params
```
### 27 - django/db/models/sql/query.py:

Start line: 1648, End line: 1746

```python
class Query(BaseExpression):

    def names_to_path(self, names, opts, allow_many=True, fail_on_missing=False):
        # ... other code
        for pos, name in enumerate(names):
            cur_names_with_path = (name, [])
            if name == "pk":
                name = opts.pk.name

            field = None
            filtered_relation = None
            try:
                if opts is None:
                    raise FieldDoesNotExist
                field = opts.get_field(name)
            except FieldDoesNotExist:
                if name in self.annotation_select:
                    field = self.annotation_select[name].output_field
                elif name in self._filtered_relations and pos == 0:
                    filtered_relation = self._filtered_relations[name]
                    if LOOKUP_SEP in filtered_relation.relation_name:
                        parts = filtered_relation.relation_name.split(LOOKUP_SEP)
                        filtered_relation_path, field, _, _ = self.names_to_path(
                            parts,
                            opts,
                            allow_many,
                            fail_on_missing,
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
                    available = sorted(
                        [
                            *get_field_names_from_opts(opts),
                            *self.annotation_select,
                            *self._filtered_relations,
                        ]
                    )
                    raise FieldError(
                        "Cannot resolve keyword '%s' into field. "
                        "Choices are: %s" % (name, ", ".join(available))
                    )
                break
            # Check if we need any joins for concrete inheritance cases (the
            # field lives in parent, but we are currently in one of its
            # children)
            if opts is not None and model is not opts.model:
                path_to_parent = opts.get_path_to_parent(model)
                if path_to_parent:
                    path.extend(path_to_parent)
                    cur_names_with_path[1].extend(path_to_parent)
                    opts = path_to_parent[-1].to_opts
            if hasattr(field, "path_infos"):
                if filtered_relation:
                    pathinfos = field.get_path_info(filtered_relation)
                else:
                    pathinfos = field.path_infos
                if not allow_many:
                    for inner_pos, p in enumerate(pathinfos):
                        if p.m2m:
                            cur_names_with_path[1].extend(pathinfos[0 : inner_pos + 1])
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
                        " not permitted." % (names[pos + 1], name)
                    )
                break
        return path, final_field, targets, names[pos + 1 :]
```
### 31 - django/db/models/sql/query.py:

Start line: 1973, End line: 2040

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
        # Generate the inner query.
        query = self.__class__(self.model)
        query._filtered_relations = self._filtered_relations
        filter_lhs, filter_rhs = filter_expr
        if isinstance(filter_rhs, OuterRef):
            filter_rhs = OuterRef(filter_rhs)
        elif isinstance(filter_rhs, F):
            filter_rhs = OuterRef(filter_rhs.name)
        query.add_filter(filter_lhs, filter_rhs)
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
            lookup_class = select_field.get_lookup("exact")
            # Note that the query.select[0].alias is different from alias
            # due to bump_prefix above.
            lookup = lookup_class(pk.get_col(query.select[0].alias), pk.get_col(alias))
            query.where.add(lookup, AND)
            query.external_aliases[alias] = True

        lookup_class = select_field.get_lookup("exact")
        lookup = lookup_class(col, ResolvedOuterRef(trimmed_prefix))
        query.where.add(lookup, AND)
        condition, needed_inner = self.build_filter(Exists(query))

        if contains_louter:
            or_null_condition, _ = self.build_filter(
                ("%s__isnull" % trimmed_prefix, True),
                current_negated=True,
                branch_negated=True,
                can_reuse=can_reuse,
            )
            condition.add(or_null_condition, OR)
            # Note that the end result will be:
            # (outercol NOT IN innerq AND outercol IS NOT NULL) OR outercol IS NULL.
            # This might look crazy but due to how IN works, this seems to be
            # correct. If the IS NOT NULL check is removed then outercol NOT
            # IN will return UNKNOWN. If the IS NULL check is removed, then if
            # outercol IS NULL we will not match the row.
        return condition, needed_inner
```
### 33 - django/db/models/sql/query.py:

Start line: 942, End line: 966

```python
class Query(BaseExpression):

    def bump_prefix(self, other_query, exclude=None):
        """
        Change the alias prefix to the next letter in the alphabet in a way
        that the other query's aliases and this query's aliases will not
        conflict. Even tables that previously had no alias will get an alias
        after this call. To prevent changing aliases use the exclude parameter.
        """

        def prefix_gen():
            """
            Generate a sequence of characters in alphabetical order:
                -> 'A', 'B', 'C', ...

            When the alphabet is finished, the sequence will continue with the
            Cartesian product:
                -> 'AA', 'AB', 'AC', ...
            """
            alphabet = ascii_uppercase
            prefix = chr(ord(self.alias_prefix) + 1)
            yield prefix
            for n in count(1):
                seq = alphabet[alphabet.index(prefix) :] if prefix else alphabet
                for s in product(seq, repeat=n):
                    yield "".join(s)
                prefix = None
        # ... other code
```
### 35 - django/db/models/fields/related_lookups.py:

Start line: 1, End line: 40

```python
import warnings

from django.db.models.lookups import (
    Exact,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNull,
    LessThan,
    LessThanOrEqual,
)
from django.utils.deprecation import RemovedInDjango50Warning


class MultiColSource:
    contains_aggregate = False
    contains_over_clause = False

    def __init__(self, alias, targets, sources, field):
        self.targets, self.sources, self.field, self.alias = (
            targets,
            sources,
            field,
            alias,
        )
        self.output_field = self.field

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.alias, self.field)

    def relabeled_clone(self, relabels):
        return self.__class__(
            relabels.get(self.alias, self.alias), self.targets, self.sources, self.field
        )

    def get_lookup(self, lookup):
        return self.output_field.get_lookup(lookup)

    def resolve_expression(self, *args, **kwargs):
        return self
```
### 37 - django/db/models/sql/query.py:

Start line: 797, End line: 830

```python
class Query(BaseExpression):

    def table_alias(self, table_name, create=False, filtered_relation=None):
        """
        Return a table alias for the given table_name and whether this is a
        new alias or not.

        If 'create' is true, a new alias is always created. Otherwise, the
        most recently created alias for the table (if one exists) is reused.
        """
        alias_list = self.table_map.get(table_name)
        if not create and alias_list:
            alias = alias_list[0]
            self.alias_refcount[alias] += 1
            return alias, False

        # Create a new alias for this table.
        if alias_list:
            alias = "%s%d" % (self.alias_prefix, len(self.alias_map) + 1)
            alias_list.append(alias)
        else:
            # The first occurrence of a table uses the table name directly.
            alias = (
                filtered_relation.alias if filtered_relation is not None else table_name
            )
            self.table_map[table_name] = [alias]
        self.alias_refcount[alias] = 1
        return alias, True

    def ref_alias(self, alias):
        """Increases the reference count for this alias."""
        self.alias_refcount[alias] += 1

    def unref_alias(self, alias, amount=1):
        """Decreases the reference count for this alias."""
        self.alias_refcount[alias] -= amount
```
### 39 - django/db/models/sql/query.py:

Start line: 2073, End line: 2122

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
        self.select += (col,)
        self.values_select += (name,)

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
### 40 - django/db/models/sql/query.py:

Start line: 2124, End line: 2173

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
                join_info = self.setup_joins(
                    name.split(LOOKUP_SEP), opts, alias, allow_many=allow_m2m
                )
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
                names = sorted(
                    [
                        *get_field_names_from_opts(opts),
                        *self.extra,
                        *self.annotation_select,
                        *self._filtered_relations,
                    ]
                )
                raise FieldError(
                    "Cannot resolve keyword %r into field. "
                    "Choices are: %s" % (name, ", ".join(names))
                )
```
### 42 - django/db/models/sql/query.py:

Start line: 158, End line: 264

```python
class Query(BaseExpression):
    """A single SQL query."""

    alias_prefix = "T"
    empty_result_set_value = None
    subq_aliases = frozenset([alias_prefix])

    compiler = "SQLCompiler"

    base_table_class = BaseTable
    join_class = Join

    default_cols = True
    default_ordering = True
    standard_ordering = True

    filter_is_sticky = False
    subquery = False

    # SQL-related attributes.
    # Select and related select clauses are expressions to use in the SELECT
    # clause of the query. The select is used for cases where we want to set up
    # the select clause to contain other than default fields (values(),
    # subqueries...). Note that annotations go to annotations dictionary.
    select = ()
    # The group_by attribute can have one of the following forms:
    #  - None: no group by at all in the query
    #  - A tuple of expressions: group by (at least) those expressions.
    #    String refs are also allowed for now.
    #  - True: group by all select fields of the model
    # See compiler.get_group_by() for details.
    group_by = None
    order_by = ()
    low_mark = 0  # Used for offset/limit.
    high_mark = None  # Used for offset/limit.
    distinct = False
    distinct_fields = ()
    select_for_update = False
    select_for_update_nowait = False
    select_for_update_skip_locked = False
    select_for_update_of = ()
    select_for_no_key_update = False
    select_related = False
    # Arbitrary limit for select_related to prevents infinite recursion.
    max_depth = 5
    # Holds the selects defined by a call to values() or values_list()
    # excluding annotation_select and extra_select.
    values_select = ()

    # SQL annotation-related attributes.
    annotation_select_mask = None
    _annotation_select_cache = None

    # Set combination attributes.
    combinator = None
    combinator_all = False
    combined_queries = ()

    # These are for extensions. The contents are more or less appended verbatim
    # to the appropriate clause.
    extra_select_mask = None
    _extra_select_cache = None

    extra_tables = ()
    extra_order_by = ()

    # A tuple that is a set of model field names and either True, if these are
    # the fields to defer, or False if these are the only fields to load.
    deferred_loading = (frozenset(), True)

    explain_info = None

    def __init__(self, model, alias_cols=True):
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
        self.table_map = {}  # Maps table names to list of aliases.
        self.used_aliases = set()

        self.where = WhereNode()
        # Maps alias -> Annotation Expression.
        self.annotations = {}
        # These are for extensions. The contents are more or less appended
        # verbatim to the appropriate clause.
        self.extra = {}  # Maps col_alias -> (col_sql, params).

        self._filtered_relations = {}

    @property
    def output_field(self):
        if len(self.select) == 1:
            select = self.select[0]
            return getattr(select, "target", None) or select.field
        elif len(self.annotation_select) == 1:
            return next(iter(self.annotation_select.values())).output_field
```
### 44 - django/db/models/sql/query.py:

Start line: 679, End line: 719

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):
        # ... other code
        joinpromoter.update_join_types(self)

        # Combine subqueries aliases to ensure aliases relabelling properly
        # handle subqueries when combining where and select clauses.
        self.subq_aliases |= rhs.subq_aliases

        # Now relabel a copy of the rhs where-clause and add it to the current
        # one.
        w = rhs.where.clone()
        w.relabel_aliases(change_map)
        self.where.add(w, connector)

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
                raise ValueError(
                    "When merging querysets using 'or', you cannot have "
                    "extra(select=...) on both sides."
                )
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
### 45 - django/db/models/sql/query.py:

Start line: 1904, End line: 1921

```python
class Query(BaseExpression):

    @classmethod
    def _gen_cols(cls, exprs, include_external=False):
        for expr in exprs:
            if isinstance(expr, Col):
                yield expr
            elif include_external and callable(
                getattr(expr, "get_external_cols", None)
            ):
                yield from expr.get_external_cols()
            elif hasattr(expr, "get_source_expressions"):
                yield from cls._gen_cols(
                    expr.get_source_expressions(),
                    include_external=include_external,
                )

    @classmethod
    def _gen_col_aliases(cls, exprs):
        yield from (expr.alias for expr in cls._gen_cols(exprs))
```
### 46 - django/db/models/sql/query.py:

Start line: 1233, End line: 1267

```python
class Query(BaseExpression):

    def check_related_objects(self, field, value, opts):
        """Check the type of object passed to query relations."""
        if field.is_relation:
            # Check that the field and the queryset use the same model in a
            # query like .filter(author=Author.objects.all()). For example, the
            # opts would be Author's (from the author field) and value.model
            # would be Author.objects.all() queryset's .model (Author also).
            # The field is the related field on the lhs side.
            if (
                isinstance(value, Query)
                and not value.has_select_fields
                and not check_rel_lookup_compatibility(value.model, opts, field)
            ):
                raise ValueError(
                    'Cannot use QuerySet for "%s": Use a QuerySet for "%s".'
                    % (value.model._meta.object_name, opts.object_name)
                )
            elif hasattr(value, "_meta"):
                self.check_query_object_type(value, opts, field)
            elif hasattr(value, "__iter__"):
                for v in value:
                    self.check_query_object_type(v, opts, field)

    def check_filterable(self, expression):
        """Raise an error if expression cannot be used in a WHERE clause."""
        if hasattr(expression, "resolve_expression") and not getattr(
            expression, "filterable", True
        ):
            raise NotSupportedError(
                expression.__class__.__name__ + " is disallowed in the filter "
                "clause."
            )
        if hasattr(expression, "get_source_expressions"):
            for expr in expression.get_source_expressions():
                self.check_filterable(expr)
```
### 49 - django/db/models/sql/query.py:

Start line: 832, End line: 867

```python
class Query(BaseExpression):

    def promote_joins(self, aliases):
        """
        Promote recursively the join type of given aliases and its children to
        an outer join. If 'unconditional' is False, only promote the join if
        it is nullable or the parent join is an outer join.

        The children promotion is done to avoid join chains that contain a LOUTER
        b INNER c. So, if we have currently a INNER b INNER c and a->b is promoted,
        then we must also promote b->c automatically, or otherwise the promotion
        of a->b doesn't actually change anything in the query results.
        """
        aliases = list(aliases)
        while aliases:
            alias = aliases.pop(0)
            if self.alias_map[alias].join_type is None:
                # This is the base table (first FROM entry) - this table
                # isn't really joined at all in the query, so we should not
                # alter its join type.
                continue
            # Only the first alias (skipped above) should have None join_type
            assert self.alias_map[alias].join_type is not None
            parent_alias = self.alias_map[alias].parent_alias
            parent_louter = (
                parent_alias and self.alias_map[parent_alias].join_type == LOUTER
            )
            already_louter = self.alias_map[alias].join_type == LOUTER
            if (self.alias_map[alias].nullable or parent_louter) and not already_louter:
                self.alias_map[alias] = self.alias_map[alias].promote()
                # Join type of 'alias' changed, so re-examine all aliases that
                # refer to this one.
                aliases.extend(
                    join
                    for join in self.alias_map
                    if self.alias_map[join].parent_alias == alias
                    and join not in aliases
                )
```
### 50 - django/db/models/fields/related_lookups.py:

Start line: 75, End line: 109

```python
class RelatedIn(In):
    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource):
            if self.rhs_is_direct_value():
                # If we get here, we are dealing with single-column relations.
                self.rhs = [get_normalized_value(val, self.lhs)[0] for val in self.rhs]
                # We need to run the related field's get_prep_value(). Consider
                # case ForeignKey to IntegerField given value 'abc'. The
                # ForeignKey itself doesn't have validation for non-integers,
                # so we must run validation using the target field.
                if hasattr(self.lhs.output_field, "path_infos"):
                    # Run the target field's get_prep_value. We can safely
                    # assume there is only one as we don't get to the direct
                    # value branch otherwise.
                    target_field = self.lhs.output_field.path_infos[-1].target_fields[
                        -1
                    ]
                    self.rhs = [target_field.get_prep_value(v) for v in self.rhs]
            elif not getattr(self.rhs, "has_select_fields", True) and not getattr(
                self.lhs.field.target_field, "primary_key", False
            ):
                self.rhs.clear_select_clause()
                if (
                    getattr(self.lhs.output_field, "primary_key", False)
                    and self.lhs.output_field.model == self.rhs.model
                ):
                    # A case like
                    # Restaurant.objects.filter(place__in=restaurant_qs), where
                    # place is a OneToOneField and the primary key of
                    # Restaurant.
                    target_field = self.lhs.field.name
                else:
                    target_field = self.lhs.field.target_field.name
                self.rhs.add_fields([target_field], True)
        return super().get_prep_lookup()
```
### 52 - django/db/models/sql/query.py:

Start line: 2228, End line: 2272

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
                    column_names.update(
                        {field.column for field in model._meta.local_concrete_fields}
                    )
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
### 54 - django/db/models/sql/query.py:

Start line: 1, End line: 79

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
    BaseExpression,
    Col,
    Exists,
    F,
    OuterRef,
    Ref,
    ResolvedOuterRef,
    Value,
)
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
    Q,
    check_rel_lookup_compatibility,
    refs_expression,
)
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node

__all__ = ["Query", "RawQuery"]

# Quotation marks ('"`[]), whitespace characters, semicolons, or inline
# SQL comments are forbidden in column aliases.
FORBIDDEN_ALIAS_PATTERN = _lazy_re_compile(r"['`\"\]\[;\s]|--|/\*|\*/")

# Inspired from
# https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
EXPLAIN_OPTIONS_PATTERN = _lazy_re_compile(r"[\w\-]+")


def get_field_names_from_opts(opts):
    if opts is None:
        return set()
    return set(
        chain.from_iterable(
            (f.name, f.attname) if f.concrete else (f.name,) for f in opts.get_fields()
        )
    )


def get_children_from_q(q):
    for child in q.children:
        if isinstance(child, Node):
            yield from get_children_from_q(child)
        else:
            yield child


JoinInfo = namedtuple(
    "JoinInfo",
    ("final_field", "targets", "opts", "joins", "path", "transform_function"),
)
```
### 58 - django/db/models/sql/query.py:

Start line: 721, End line: 756

```python
class Query(BaseExpression):

    def _get_defer_select_mask(self, opts, mask, select_mask=None):
        if select_mask is None:
            select_mask = {}
        select_mask[opts.pk] = {}
        # All concrete fields that are not part of the defer mask must be
        # loaded. If a relational field is encountered it gets added to the
        # mask for it be considered if `select_related` and the cycle continues
        # by recursively caling this function.
        for field in opts.concrete_fields:
            field_mask = mask.pop(field.name, None)
            if field_mask is None:
                select_mask.setdefault(field, {})
            elif field_mask:
                if not field.is_relation:
                    raise FieldError(next(iter(field_mask)))
                field_select_mask = select_mask.setdefault(field, {})
                related_model = field.remote_field.model._meta.concrete_model
                self._get_defer_select_mask(
                    related_model._meta, field_mask, field_select_mask
                )
        # Remaining defer entries must be references to reverse relationships.
        # The following code is expected to raise FieldError if it encounters
        # a malformed defer entry.
        for field_name, field_mask in mask.items():
            if filtered_relation := self._filtered_relations.get(field_name):
                relation = opts.get_field(filtered_relation.relation_name)
                field_select_mask = select_mask.setdefault((field_name, relation), {})
                field = relation.field
            else:
                field = opts.get_field(field_name).field
                field_select_mask = select_mask.setdefault(field, {})
            related_model = field.model._meta.concrete_model
            self._get_defer_select_mask(
                related_model._meta, field_mask, field_select_mask
            )
        return select_mask
```
### 59 - django/db/models/sql/query.py:

Start line: 1510, End line: 1536

```python
class Query(BaseExpression):

    def add_filter(self, filter_lhs, filter_rhs):
        self.add_q(Q((filter_lhs, filter_rhs)))

    def add_q(self, q_object):
        """
        A preprocessor for the internal _add_q(). Responsible for doing final
        join promotion.
        """
        # For join promotion this case is doing an AND for the added q_object
        # and existing conditions. So, any existing inner join forces the join
        # type to remain inner. Existing outer joins can however be demoted.
        # (Consider case where rel_a is LOUTER and rel_a__col=1 is added - if
        # rel_a doesn't produce any rows, then the whole condition must fail.
        # So, demotion is OK.
        existing_inner = {
            a for a in self.alias_map if self.alias_map[a].join_type == INNER
        }
        clause, _ = self._add_q(q_object, self.used_aliases)
        if clause:
            self.where.add(clause, AND)
        self.demote_joins(existing_inner)

    def build_where(self, filter_expr):
        return self.build_filter(filter_expr, allow_joins=False)[0]

    def clear_where(self):
        self.where = WhereNode()
```
### 61 - django/db/models/sql/query.py:

Start line: 1069, End line: 1100

```python
class Query(BaseExpression):

    def join_parent_model(self, opts, model, alias, seen):
        """
        Make sure the given 'model' is joined in the query. If 'model' isn't
        a parent of 'opts' or if it is None this method is a no-op.

        The 'alias' is the root alias for starting the join, 'seen' is a dict
        of model -> alias of existing joins. It must also contain a mapping
        of None -> some alias. This will be returned in the no-op case.
        """
        if model in seen:
            return seen[model]
        chain = opts.get_base_chain(model)
        if not chain:
            return alias
        curr_opts = opts
        for int_model in chain:
            if int_model in seen:
                curr_opts = int_model._meta
                alias = seen[int_model]
                continue
            # Proxy model have elements in base chain
            # with no parents, assign the new options
            # object and skip to the next base in that
            # case
            if not curr_opts.parents[int_model]:
                curr_opts = int_model._meta
                continue
            link_field = curr_opts.get_ancestor_link(int_model)
            join_info = self.setup_joins([link_field.name], curr_opts, alias)
            curr_opts = int_model._meta
            alias = seen[int_model] = join_info.joins[-1]
        return alias or seen[None]
```
### 65 - django/db/models/sql/query.py:

Start line: 1019, End line: 1067

```python
class Query(BaseExpression):

    def join(self, join, reuse=None, reuse_with_filtered_relation=False):
        """
        Return an alias for the 'join', either reusing an existing alias for
        that join or creating a new one. 'join' is either a base_table_class or
        join_class.

        The 'reuse' parameter can be either None which means all joins are
        reusable, or it can be a set containing the aliases that can be reused.

        The 'reuse_with_filtered_relation' parameter is used when computing
        FilteredRelation instances.

        A join is always created as LOUTER if the lhs alias is LOUTER to make
        sure chains like t1 LOUTER t2 INNER t3 aren't generated. All new
        joins are created as LOUTER if the join is nullable.
        """
        if reuse_with_filtered_relation and reuse:
            reuse_aliases = [
                a for a, j in self.alias_map.items() if a in reuse and j.equals(join)
            ]
        else:
            reuse_aliases = [
                a
                for a, j in self.alias_map.items()
                if (reuse is None or a in reuse) and j == join
            ]
        if reuse_aliases:
            if join.table_alias in reuse_aliases:
                reuse_alias = join.table_alias
            else:
                # Reuse the most recent alias of the joined table
                # (a many-to-many relation may be joined multiple times).
                reuse_alias = reuse_aliases[-1]
            self.ref_alias(reuse_alias)
            return reuse_alias

        # No reuse is possible, so we need a new alias.
        alias, _ = self.table_alias(
            join.table_name, create=True, filtered_relation=join.filtered_relation
        )
        if join.join_type:
            if self.alias_map[join.parent_alias].join_type == LOUTER or join.nullable:
                join_type = LOUTER
            else:
                join_type = INNER
            join.join_type = join_type
        join.table_alias = alias
        self.alias_map[alias] = join
        return alias
```
### 66 - django/db/models/sql/query.py:

Start line: 2307, End line: 2334

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
            if new_existing := existing.difference(field_names):
                self.deferred_loading = new_existing, False
            else:
                self.clear_deferred_loading()
                if new_only := set(field_names).difference(existing):
                    self.deferred_loading = new_only, True
```
### 68 - django/db/models/sql/query.py:

Start line: 1872, End line: 1902

```python
class Query(BaseExpression):

    def trim_joins(self, targets, joins, path):
        """
        The 'target' parameter is the final field being joined to, 'joins'
        is the full list of join aliases. The 'path' contain the PathInfos
        used to create the joins.

        Return the final target field and table alias and the new active
        joins.

        Always trim any direct join if the target column is already in the
        previous table. Can't trim reverse joins as it's unknown if there's
        anything on the other side of the join.
        """
        joins = joins[:]
        for pos, info in enumerate(reversed(path)):
            if len(joins) == 1 or not info.direct:
                break
            if info.filtered_relation:
                break
            join_targets = {t.column for t in info.join_field.foreign_related_fields}
            cur_targets = {t.column for t in targets}
            if not cur_targets.issubset(join_targets):
                break
            targets_dict = {
                r[1].column: r[0]
                for r in info.join_field.related_fields
                if r[1].column in cur_targets
            }
            targets = tuple(targets_dict[t.column] for t in targets)
            self.unref_alias(joins.pop())
        return targets, joins[-1], joins
```
### 71 - django/db/models/fields/related_lookups.py:

Start line: 171, End line: 211

```python
class RelatedLookupMixin:

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            assert self.rhs_is_direct_value()
            self.rhs = get_normalized_value(self.rhs, self.lhs)
            from django.db.models.sql.where import AND, WhereNode

            root_constraint = WhereNode()
            for target, source, val in zip(
                self.lhs.targets, self.lhs.sources, self.rhs
            ):
                lookup_class = target.get_lookup(self.lookup_name)
                root_constraint.add(
                    lookup_class(target.get_col(self.lhs.alias, source), val), AND
                )
            return root_constraint.as_sql(compiler, connection)
        return super().as_sql(compiler, connection)


class RelatedExact(RelatedLookupMixin, Exact):
    pass


class RelatedLessThan(RelatedLookupMixin, LessThan):
    pass


class RelatedGreaterThan(RelatedLookupMixin, GreaterThan):
    pass


class RelatedGreaterThanOrEqual(RelatedLookupMixin, GreaterThanOrEqual):
    pass


class RelatedLessThanOrEqual(RelatedLookupMixin, LessThanOrEqual):
    pass


class RelatedIsNull(RelatedLookupMixin, IsNull):
    pass
```
### 74 - django/db/models/sql/query.py:

Start line: 589, End line: 604

```python
class Query(BaseExpression):

    def has_results(self, using):
        q = self.exists(using)
        compiler = q.get_compiler(using=using)
        return compiler.has_results()

    def explain(self, using, format=None, **options):
        q = self.clone()
        for option_name in options:
            if (
                not EXPLAIN_OPTIONS_PATTERN.fullmatch(option_name)
                or "--" in option_name
            ):
                raise ValueError(f"Invalid option name: {option_name!r}.")
        q.explain_info = ExplainInfo(format, options)
        compiler = q.get_compiler(using=using)
        return "\n".join(compiler.explain_query())
```
### 80 - django/db/models/sql/query.py:

Start line: 1748, End line: 1839

```python
class Query(BaseExpression):

    def setup_joins(
        self,
        names,
        opts,
        alias,
        can_reuse=None,
        allow_many=True,
        reuse_with_filtered_relation=False,
    ):
        """
        Compute the necessary table joins for the passage through the fields
        given in 'names'. 'opts' is the Options class for the current model
        (which gives the table we are starting from), 'alias' is the alias for
        the table to start the joining from.

        The 'can_reuse' defines the reverse foreign key joins we can reuse. It
        can be None in which case all joins are reusable or a set of aliases
        that can be reused. Note that non-reverse foreign keys are always
        reusable when using setup_joins().

        The 'reuse_with_filtered_relation' can be used to force 'can_reuse'
        parameter and force the relation on the given connections.

        If 'allow_many' is False, then any reverse foreign key seen will
        generate a MultiJoin exception.

        Return the final field involved in the joins, the target field (used
        for any 'where' constraint), the final 'opts' value, the joins, the
        field path traveled to generate the joins, and a transform function
        that takes a field and alias and is equivalent to `field.get_col(alias)`
        in the simple case but wraps field transforms if they were included in
        names.

        The target field is the field containing the concrete value. Final
        field can be something different, for example foreign key pointing to
        that value. Final field is needed for example in some value
        conversions (convert 'obj' in fk__id=obj to pk val using the foreign
        key field for example).
        """
        joins = [alias]
        # The transform can't be applied yet, as joins must be trimmed later.
        # To avoid making every caller of this method look up transforms
        # directly, compute transforms here and create a partial that converts
        # fields to the appropriate wrapped version.

        def final_transformer(field, alias):
            if not self.alias_cols:
                alias = None
            return field.get_col(alias)

        # Try resolving all the names as fields first. If there's an error,
        # treat trailing names as lookups until a field can be resolved.
        last_field_exception = None
        for pivot in range(len(names), 0, -1):
            try:
                path, final_field, targets, rest = self.names_to_path(
                    names[:pivot],
                    opts,
                    allow_many,
                    fail_on_missing=True,
                )
            except FieldError as exc:
                if pivot == 1:
                    # The first item cannot be a lookup, so it's safe
                    # to raise the field error here.
                    raise
                else:
                    last_field_exception = exc
            else:
                # The transforms are the remaining items that couldn't be
                # resolved into fields.
                transforms = names[pivot:]
                break
        for name in transforms:

            def transform(field, alias, *, name, previous):
                try:
                    wrapped = previous(field, alias)
                    return self.try_transform(wrapped, name)
                except FieldError:
                    # FieldError is raised if the transform doesn't exist.
                    if isinstance(final_field, Field) and last_field_exception:
                        raise last_field_exception
                    else:
                        raise

            final_transformer = functools.partial(
                transform, name=name, previous=final_transformer
            )
        # Then, add the path to the query's joins. Note that we can't trim
        # joins at this stage - we will need the information about join type
        # of the trimmed joins.
        # ... other code
```
### 83 - django/db/models/sql/query.py:

Start line: 606, End line: 678

```python
class Query(BaseExpression):

    def combine(self, rhs, connector):
        """
        Merge the 'rhs' query into the current one (with any 'rhs' effects
        being applied *after* (that is, "to the right of") anything in the
        current query. 'rhs' is not modified during a call to this function.

        The 'connector' parameter describes how to connect filters from the
        'rhs' query.
        """
        if self.model != rhs.model:
            raise TypeError("Cannot combine queries on two different base models.")
        if self.is_sliced:
            raise TypeError("Cannot combine queries once a slice has been taken.")
        if self.distinct != rhs.distinct:
            raise TypeError("Cannot combine a unique query with a non-unique query.")
        if self.distinct_fields != rhs.distinct_fields:
            raise TypeError("Cannot combine queries with different distinct fields.")

        # If lhs and rhs shares the same alias prefix, it is possible to have
        # conflicting alias changes like T4 -> T5, T5 -> T6, which might end up
        # as T4 -> T6 while combining two querysets. To prevent this, change an
        # alias prefix of the rhs and update current aliases accordingly,
        # except if the alias is the base table since it must be present in the
        # query on both sides.
        initial_alias = self.get_initial_alias()
        rhs.bump_prefix(self, exclude={initial_alias})

        # Work out how to relabel the rhs aliases, if necessary.
        change_map = {}
        conjunction = connector == AND

        # Determine which existing joins can be reused. When combining the
        # query with AND we must recreate all joins for m2m filters. When
        # combining with OR we can reuse joins. The reason is that in AND
        # case a single row can't fulfill a condition like:
        #     revrel__col=1 & revrel__col=2
        # But, there might be two different related rows matching this
        # condition. In OR case a single True is enough, so single row is
        # enough, too.
        #
        # Note that we will be creating duplicate joins for non-m2m joins in
        # the AND case. The results will be correct but this creates too many
        # joins. This is something that could be fixed later on.
        reuse = set() if conjunction else set(self.alias_map)
        joinpromoter = JoinPromoter(connector, 2, False)
        joinpromoter.add_votes(
            j for j in self.alias_map if self.alias_map[j].join_type == INNER
        )
        rhs_votes = set()
        # Now, add the joins from rhs query into the new query (skipping base
        # table).
        rhs_tables = list(rhs.alias_map)[1:]
        for alias in rhs_tables:
            join = rhs.alias_map[alias]
            # If the left side of the join was already relabeled, use the
            # updated alias.
            join = join.relabeled_clone(change_map)
            new_alias = self.join(join, reuse=reuse)
            if join.join_type == INNER:
                rhs_votes.add(new_alias)
            # We can't reuse the same join again in the query. If we have two
            # distinct joins for the same connection in rhs query, then the
            # combined query must have two joins, too.
            reuse.discard(new_alias)
            if alias != new_alias:
                change_map[alias] = new_alias
            if not rhs.alias_refcount[alias]:
                # The alias was unused in the rhs query. Unref it so that it
                # will be unused in the new query, too. We have to add and
                # unref the alias so that join promotion has information of
                # the join type for the unused alias.
                self.unref_alias(new_alias)
        joinpromoter.add_votes(rhs_votes)
        # ... other code
```
### 84 - django/db/models/sql/query.py:

Start line: 758, End line: 773

```python
class Query(BaseExpression):

    def _get_only_select_mask(self, opts, mask, select_mask=None):
        if select_mask is None:
            select_mask = {}
        select_mask[opts.pk] = {}
        # Only include fields mentioned in the mask.
        for field_name, field_mask in mask.items():
            field = opts.get_field(field_name)
            field_select_mask = select_mask.setdefault(field, {})
            if field_mask:
                if not field.is_relation:
                    raise FieldError(next(iter(field_mask)))
                related_model = field.remote_field.model._meta.concrete_model
                self._get_only_select_mask(
                    related_model._meta, field_mask, field_select_mask
                )
        return select_mask
```
### 101 - django/db/models/sql/query.py:

Start line: 2468, End line: 2543

```python
class Query(BaseExpression):

    def trim_start(self, names_with_path):
        """
        Trim joins from the start of the join path. The candidates for trim
        are the PathInfos in names_with_path structure that are m2m joins.

        Also set the select column so the start matches the join.

        This method is meant to be used for generating the subquery joins &
        cols in split_exclude().

        Return a lookup usable for doing outerq.filter(lookup=self) and a
        boolean indicating if the joins in the prefix contain a LEFT OUTER join.
        _"""
        all_paths = []
        for _, paths in names_with_path:
            all_paths.extend(paths)
        contains_louter = False
        # Trim and operate only on tables that were generated for
        # the lookup part of the query. That is, avoid trimming
        # joins generated for F() expressions.
        lookup_tables = [
            t for t in self.alias_map if t in self._lookup_joins or t == self.base_table
        ]
        for trimmed_paths, path in enumerate(all_paths):
            if path.m2m:
                break
            if self.alias_map[lookup_tables[trimmed_paths + 1]].join_type == LOUTER:
                contains_louter = True
            alias = lookup_tables[trimmed_paths]
            self.unref_alias(alias)
        # The path.join_field is a Rel, lets get the other side's field
        join_field = path.join_field.field
        # Build the filter prefix.
        paths_in_prefix = trimmed_paths
        trimmed_prefix = []
        for name, path in names_with_path:
            if paths_in_prefix - len(path) < 0:
                break
            trimmed_prefix.append(name)
            paths_in_prefix -= len(path)
        trimmed_prefix.append(join_field.foreign_related_fields[0].name)
        trimmed_prefix = LOOKUP_SEP.join(trimmed_prefix)
        # Lets still see if we can trim the first join from the inner query
        # (that is, self). We can't do this for:
        # - LEFT JOINs because we would miss those rows that have nothing on
        #   the outer side,
        # - INNER JOINs from filtered relations because we would miss their
        #   filters.
        first_join = self.alias_map[lookup_tables[trimmed_paths + 1]]
        if first_join.join_type != LOUTER and not first_join.filtered_relation:
            select_fields = [r[0] for r in join_field.related_fields]
            select_alias = lookup_tables[trimmed_paths + 1]
            self.unref_alias(lookup_tables[trimmed_paths])
            extra_restriction = join_field.get_extra_restriction(
                None, lookup_tables[trimmed_paths + 1]
            )
            if extra_restriction:
                self.where.add(extra_restriction, AND)
        else:
            # TODO: It might be possible to trim more joins from the start of the
            # inner query if it happens to have a longer join chain containing the
            # values in select_fields. Lets punt this one for now.
            select_fields = [r[1] for r in join_field.related_fields]
            select_alias = lookup_tables[trimmed_paths]
        # The found starting point is likely a join_class instead of a
        # base_table_class reference. But the first entry in the query's FROM
        # clause must not be a JOIN.
        for table in self.alias_map:
            if self.alias_refcount[table] > 0:
                self.alias_map[table] = self.base_table_class(
                    self.alias_map[table].table_name,
                    table,
                )
                break
        self.set_select([f.get_col(select_alias) for f in select_fields])
        return trimmed_prefix, contains_louter
```
### 105 - django/db/models/sql/query.py:

Start line: 775, End line: 795

```python
class Query(BaseExpression):

    def get_select_mask(self):
        """
        Convert the self.deferred_loading data structure to an alternate data
        structure, describing the field that *will* be loaded. This is used to
        compute the columns to select from the database and also by the
        QuerySet class to work out which fields are being initialized on each
        model. Models that have all their fields included aren't mentioned in
        the result, only those that have field restrictions in place.
        """
        field_names, defer = self.deferred_loading
        if not field_names:
            return {}
        mask = {}
        for field_name in field_names:
            part_mask = mask
            for part in field_name.split(LOOKUP_SEP):
                part_mask = part_mask.setdefault(part, {})
        opts = self.get_meta()
        if defer:
            return self._get_defer_select_mask(opts, mask)
        return self._get_only_select_mask(opts, mask)
```
### 108 - django/db/models/sql/query.py:

Start line: 1538, End line: 1570

```python
class Query(BaseExpression):

    def _add_q(
        self,
        q_object,
        used_aliases,
        branch_negated=False,
        current_negated=False,
        allow_joins=True,
        split_subq=True,
        check_filterable=True,
    ):
        """Add a Q-object to the current filter."""
        connector = q_object.connector
        current_negated = current_negated ^ q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = WhereNode(connector=connector, negated=q_object.negated)
        joinpromoter = JoinPromoter(
            q_object.connector, len(q_object.children), current_negated
        )
        for child in q_object.children:
            child_clause, needed_inner = self.build_filter(
                child,
                can_reuse=used_aliases,
                branch_negated=branch_negated,
                current_negated=current_negated,
                allow_joins=allow_joins,
                split_subq=split_subq,
                check_filterable=check_filterable,
            )
            joinpromoter.add_votes(needed_inner)
            if child_clause:
                target_clause.add(child_clause, connector)
        needed_inner = joinpromoter.update_join_types(self)
        return target_clause, needed_inner
```
### 109 - django/db/models/sql/query.py:

Start line: 1181, End line: 1199

```python
class Query(BaseExpression):

    def resolve_lookup_value(self, value, can_reuse, allow_joins):
        if hasattr(value, "resolve_expression"):
            value = value.resolve_expression(
                self,
                reuse=can_reuse,
                allow_joins=allow_joins,
            )
        elif isinstance(value, (list, tuple)):
            # The items of the iterable may be expressions and therefore need
            # to be resolved independently.
            values = (
                self.resolve_lookup_value(sub_value, can_reuse, allow_joins)
                for sub_value in value
            )
            type_ = type(value)
            if hasattr(type_, "_make"):  # namedtuple
                return type_(*values)
            return type_(values)
        return value
```
### 111 - django/db/models/sql/query.py:

Start line: 2612, End line: 2668

```python
class JoinPromoter:

    def update_join_types(self, query):
        """
        Change join types so that the generated query is as efficient as
        possible, but still correct. So, change as many joins as possible
        to INNER, but don't make OUTER joins INNER if that could remove
        results from the query.
        """
        to_promote = set()
        to_demote = set()
        # The effective_connector is used so that NOT (a AND b) is treated
        # similarly to (a OR b) for join promotion.
        for table, votes in self.votes.items():
            # We must use outer joins in OR case when the join isn't contained
            # in all of the joins. Otherwise the INNER JOIN itself could remove
            # valid results. Consider the case where a model with rel_a and
            # rel_b relations is queried with rel_a__col=1 | rel_b__col=2. Now,
            # if rel_a join doesn't produce any results is null (for example
            # reverse foreign key or null value in direct foreign key), and
            # there is a matching row in rel_b with col=2, then an INNER join
            # to rel_a would remove a valid match from the query. So, we need
            # to promote any existing INNER to LOUTER (it is possible this
            # promotion in turn will be demoted later on).
            if self.effective_connector == OR and votes < self.num_children:
                to_promote.add(table)
            # If connector is AND and there is a filter that can match only
            # when there is a joinable row, then use INNER. For example, in
            # rel_a__col=1 & rel_b__col=2, if either of the rels produce NULL
            # as join output, then the col=1 or col=2 can't match (as
            # NULL=anything is always false).
            # For the OR case, if all children voted for a join to be inner,
            # then we can use INNER for the join. For example:
            #     (rel_a__col__icontains=Alex | rel_a__col__icontains=Russell)
            # then if rel_a doesn't produce any rows, the whole condition
            # can't match. Hence we can safely use INNER join.
            if self.effective_connector == AND or (
                self.effective_connector == OR and votes == self.num_children
            ):
                to_demote.add(table)
            # Finally, what happens in cases where we have:
            #    (rel_a__col=1|rel_b__col=2) & rel_a__col__gte=0
            # Now, we first generate the OR clause, and promote joins for it
            # in the first if branch above. Both rel_a and rel_b are promoted
            # to LOUTER joins. After that we do the AND case. The OR case
            # voted no inner joins but the rel_a__col__gte=0 votes inner join
            # for rel_a. We demote it back to INNER join (in AND case a single
            # vote is enough). The demotion is OK, if rel_a doesn't produce
            # rows, then the rel_a__col__gte=0 clause can't be true, and thus
            # the whole clause must be false. So, it is safe to use INNER
            # join.
            # Note that in this example we could just as well have the __gte
            # clause and the OR clause swapped. Or we could replace the __gte
            # clause with an OR clause containing rel_a__col=1|rel_a__col=2,
            # and again we could safely demote to INNER.
        query.promote_joins(to_promote)
        query.demote_joins(to_demote)
        return to_demote
```
### 113 - django/db/models/sql/query.py:

Start line: 553, End line: 587

```python
class Query(BaseExpression):

    def get_count(self, using):
        """
        Perform a COUNT() query using the current filter constraints.
        """
        obj = self.clone()
        obj.add_annotation(Count("*"), alias="__count", is_summary=True)
        return obj.get_aggregation(using, ["__count"])["__count"]

    def has_filters(self):
        return self.where

    def exists(self, using, limit=True):
        q = self.clone()
        if not (q.distinct and q.is_sliced):
            if q.group_by is True:
                q.add_fields(
                    (f.attname for f in self.model._meta.concrete_fields), False
                )
                # Disable GROUP BY aliases to avoid orphaning references to the
                # SELECT clause which is about to be cleared.
                q.set_group_by(allow_aliases=False)
            q.clear_select_clause()
        if q.combined_queries and q.combinator == "union":
            limit_combined = connections[
                using
            ].features.supports_slicing_ordering_in_compound
            q.combined_queries = tuple(
                combined_query.exists(using, limit=limit_combined)
                for combined_query in q.combined_queries
            )
        q.clear_ordering(force=True)
        if limit:
            q.set_limits(high=1)
        q.add_annotation(Value(1), "a")
        return q
```
### 117 - django/db/models/fields/related_lookups.py:

Start line: 152, End line: 169

```python
class RelatedLookupMixin:
    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource) and not hasattr(
            self.rhs, "resolve_expression"
        ):
            # If we get here, we are dealing with single-column relations.
            self.rhs = get_normalized_value(self.rhs, self.lhs)[0]
            # We need to run the related field's get_prep_value(). Consider case
            # ForeignKey to IntegerField given value 'abc'. The ForeignKey itself
            # doesn't have validation for non-integers, so we must run validation
            # using the target field.
            if self.prepare_rhs and hasattr(self.lhs.output_field, "path_infos"):
                # Get the target field. We can safely assume there is only one
                # as we don't get to the direct value branch otherwise.
                target_field = self.lhs.output_field.path_infos[-1].target_fields[-1]
                self.rhs = target_field.get_prep_value(self.rhs)

        return super().get_prep_lookup()
```
### 126 - django/db/models/sql/query.py:

Start line: 2545, End line: 2561

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
        return field.null or (
            field.empty_strings_allowed
            and connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls
        )
```
### 140 - django/db/models/sql/query.py:

Start line: 266, End line: 317

```python
class Query(BaseExpression):

    @property
    def has_select_fields(self):
        return bool(
            self.select or self.annotation_select_mask or self.extra_select_mask
        )

    @cached_property
    def base_table(self):
        for alias in self.alias_map:
            return alias

    def __str__(self):
        """
        Return the query as a string of SQL with the parameter values
        substituted in (use sql_with_params() to see the unsubstituted string).

        Parameter values won't necessarily be quoted correctly, since that is
        done by the database interface at execution time.
        """
        sql, params = self.sql_with_params()
        return sql % params

    def sql_with_params(self):
        """
        Return the query as an SQL string and the parameters that will be
        substituted into the query.
        """
        return self.get_compiler(DEFAULT_DB_ALIAS).as_sql()

    def __deepcopy__(self, memo):
        """Limit the amount of work when a Query is deepcopied."""
        result = self.clone()
        memo[id(self)] = result
        return result

    def get_compiler(self, using=None, connection=None, elide_empty=True):
        if using is None and connection is None:
            raise ValueError("Need either using or connection")
        if using:
            connection = connections[using]
        return connection.ops.compiler(self.compiler)(
            self, connection, using, elide_empty
        )

    def get_meta(self):
        """
        Return the Options instance (the model._meta) from which to start
        processing. Normally, this is self.model._meta, but it can be changed
        by subclasses.
        """
        if self.model:
            return self.model._meta
```
### 150 - django/db/models/sql/query.py:

Start line: 1572, End line: 1599

```python
class Query(BaseExpression):

    def build_filtered_relation_q(
        self, q_object, reuse, branch_negated=False, current_negated=False
    ):
        """Add a FilteredRelation object to the current filter."""
        connector = q_object.connector
        current_negated ^= q_object.negated
        branch_negated = branch_negated or q_object.negated
        target_clause = WhereNode(connector=connector, negated=q_object.negated)
        for child in q_object.children:
            if isinstance(child, Node):
                child_clause = self.build_filtered_relation_q(
                    child,
                    reuse=reuse,
                    branch_negated=branch_negated,
                    current_negated=current_negated,
                )
            else:
                child_clause, _ = self.build_filter(
                    child,
                    can_reuse=reuse,
                    branch_negated=branch_negated,
                    current_negated=current_negated,
                    allow_joins=True,
                    split_subq=False,
                    reuse_with_filtered_relation=True,
                )
            target_clause.add(child_clause, connector)
        return target_clause
```
