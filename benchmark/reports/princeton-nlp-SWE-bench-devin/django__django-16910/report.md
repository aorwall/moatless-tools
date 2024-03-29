# django__django-16910

| **django/django** | `4142739af1cda53581af4169dbe16d6cd5e26948` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4081 |
| **Any found context length** | 4081 |
| **Avg pos** | 12.0 |
| **Min pos** | 12 |
| **Max pos** | 12 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -779,7 +779,13 @@ def _get_only_select_mask(self, opts, mask, select_mask=None):
         # Only include fields mentioned in the mask.
         for field_name, field_mask in mask.items():
             field = opts.get_field(field_name)
-            field_select_mask = select_mask.setdefault(field, {})
+            # Retrieve the actual field associated with reverse relationships
+            # as that's what is expected in the select mask.
+            if field in opts.related_objects:
+                field_key = field.field
+            else:
+                field_key = field
+            field_select_mask = select_mask.setdefault(field_key, {})
             if field_mask:
                 if not field.is_relation:
                     raise FieldError(next(iter(field_mask)))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/sql/query.py | 782 | 782 | 12 | 4 | 4081


## Problem Statement

```
QuerySet.only() doesn't work with select_related() on a reverse OneToOneField relation.
Description
	
On Django 4.2 calling only() with select_related() on a query using the reverse lookup for a OneToOne relation does not generate the correct query.
All the fields from the related model are still included in the generated SQL.
Sample models:
class Main(models.Model):
	main_field_1 = models.CharField(blank=True, max_length=45)
	main_field_2 = models.CharField(blank=True, max_length=45)
	main_field_3 = models.CharField(blank=True, max_length=45)
class Secondary(models.Model):
	main = models.OneToOneField(Main, primary_key=True, related_name='secondary', on_delete=models.CASCADE)
	secondary_field_1 = models.CharField(blank=True, max_length=45)
	secondary_field_2 = models.CharField(blank=True, max_length=45)
	secondary_field_3 = models.CharField(blank=True, max_length=45)
Sample code:
Main.objects.select_related('secondary').only('main_field_1', 'secondary__secondary_field_1')
Generated query on Django 4.2.1:
SELECT "bugtest_main"."id", "bugtest_main"."main_field_1", "bugtest_secondary"."main_id", "bugtest_secondary"."secondary_field_1", "bugtest_secondary"."secondary_field_2", "bugtest_secondary"."secondary_field_3" FROM "bugtest_main" LEFT OUTER JOIN "bugtest_secondary" ON ("bugtest_main"."id" = "bugtest_secondary"."main_id")
Generated query on Django 4.1.9:
SELECT "bugtest_main"."id", "bugtest_main"."main_field_1", "bugtest_secondary"."main_id", "bugtest_secondary"."secondary_field_1" FROM "bugtest_main" LEFT OUTER JOIN "bugtest_secondary" ON ("bugtest_main"."id" = "bugtest_secondary"."main_id")

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/query.py | 1546 | 1568| 162 | 162 | 20560 | 
| 2 | 2 django/db/models/fields/related_descriptors.py | 429 | 452| 194 | 356 | 32218 | 
| 3 | 2 django/db/models/query.py | 1728 | 1747| 213 | 569 | 32218 | 
| 4 | 2 django/db/models/fields/related_descriptors.py | 406 | 427| 158 | 727 | 32218 | 
| 5 | 2 django/db/models/query.py | 1570 | 1593| 222 | 949 | 32218 | 
| 6 | 3 django/db/models/sql/compiler.py | 1246 | 1375| 990 | 1939 | 49005 | 
| 7 | **4 django/db/models/sql/query.py** | 2252 | 2299| 376 | 2315 | 71907 | 
| 8 | 4 django/db/models/fields/related_descriptors.py | 155 | 199| 418 | 2733 | 71907 | 
| 9 | 5 django/db/models/fields/related.py | 156 | 187| 209 | 2942 | 86611 | 
| 10 | 5 django/db/models/sql/compiler.py | 1143 | 1244| 743 | 3685 | 86611 | 
| 11 | 6 django/db/models/fields/related_lookups.py | 160 | 200| 250 | 3935 | 88148 | 
| **-> 12 <-** | **6 django/db/models/sql/query.py** | 775 | 790| 146 | 4081 | 88148 | 
| 13 | **6 django/db/models/sql/query.py** | 2377 | 2437| 502 | 4583 | 88148 | 
| 14 | 6 django/db/models/query.py | 2049 | 2155| 723 | 5306 | 88148 | 
| 15 | **6 django/db/models/sql/query.py** | 1651 | 1749| 846 | 6152 | 88148 | 
| 16 | 7 django/db/models/query_utils.py | 315 | 351| 291 | 6443 | 91461 | 
| 17 | 7 django/db/models/fields/related_lookups.py | 100 | 138| 324 | 6767 | 91461 | 
| 18 | 8 django/db/models/fields/reverse_related.py | 193 | 219| 223 | 6990 | 94096 | 
| 19 | 8 django/db/models/query.py | 451 | 483| 253 | 7243 | 94096 | 
| 20 | 8 django/db/models/query.py | 2248 | 2382| 1129 | 8372 | 94096 | 
| 21 | 8 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 9180 | 94096 | 
| 22 | 8 django/db/models/fields/related_descriptors.py | 94 | 112| 196 | 9376 | 94096 | 
| 23 | **8 django/db/models/sql/query.py** | 729 | 773| 460 | 9836 | 94096 | 
| 24 | 8 django/db/models/fields/related.py | 1463 | 1592| 984 | 10820 | 94096 | 
| 25 | 8 django/db/models/fields/related_descriptors.py | 665 | 703| 354 | 11174 | 94096 | 
| 26 | 8 django/db/models/fields/related.py | 304 | 341| 296 | 11470 | 94096 | 
| 27 | **8 django/db/models/sql/query.py** | 1970 | 2037| 667 | 12137 | 94096 | 
| 28 | 8 django/db/models/fields/related.py | 228 | 303| 696 | 12833 | 94096 | 
| 29 | 8 django/db/models/query.py | 485 | 502| 150 | 12983 | 94096 | 
| 30 | 8 django/db/models/fields/related.py | 404 | 422| 161 | 13144 | 94096 | 
| 31 | **8 django/db/models/sql/query.py** | 1262 | 1296| 339 | 13483 | 94096 | 
| 32 | **8 django/db/models/sql/query.py** | 1840 | 1864| 204 | 13687 | 94096 | 
| 33 | **8 django/db/models/sql/query.py** | 1599 | 1633| 314 | 14001 | 94096 | 
| 34 | 9 django/db/backends/base/schema.py | 52 | 72| 152 | 14153 | 109242 | 
| 35 | **9 django/db/models/sql/query.py** | 1 | 92| 651 | 14804 | 109242 | 
| 36 | 9 django/db/models/query.py | 1749 | 1778| 189 | 14993 | 109242 | 
| 37 | 10 django/db/models/deletion.py | 403 | 416| 117 | 15110 | 113224 | 
| 38 | 10 django/db/models/fields/related_lookups.py | 65 | 98| 369 | 15479 | 113224 | 
| 39 | 10 django/db/models/fields/related_descriptors.py | 705 | 728| 227 | 15706 | 113224 | 
| 40 | 10 django/db/models/query.py | 672 | 724| 424 | 16130 | 113224 | 
| 41 | 10 django/db/models/fields/related_descriptors.py | 730 | 749| 231 | 16361 | 113224 | 
| 42 | 10 django/db/models/fields/reverse_related.py | 173 | 191| 160 | 16521 | 113224 | 
| 43 | 10 django/db/models/query.py | 1163 | 1185| 160 | 16681 | 113224 | 
| 44 | 10 django/db/models/fields/related.py | 381 | 402| 219 | 16900 | 113224 | 
| 45 | 10 django/db/models/fields/related.py | 777 | 806| 235 | 17135 | 113224 | 
| 46 | 10 django/db/models/sql/compiler.py | 1438 | 1481| 349 | 17484 | 113224 | 
| 47 | 10 django/db/models/fields/reverse_related.py | 319 | 349| 159 | 17643 | 113224 | 
| 48 | 10 django/db/models/fields/related.py | 1217 | 1264| 368 | 18011 | 113224 | 
| 49 | **10 django/db/models/sql/query.py** | 1920 | 1968| 445 | 18456 | 113224 | 
| 50 | 10 django/db/models/fields/related.py | 733 | 755| 172 | 18628 | 113224 | 
| 51 | 10 django/db/models/query.py | 504 | 521| 136 | 18764 | 113224 | 
| 52 | 11 django/contrib/admin/views/main.py | 595 | 627| 227 | 18991 | 118141 | 
| 53 | 11 django/db/models/query.py | 1610 | 1662| 342 | 19333 | 118141 | 
| 54 | 11 django/db/models/query_utils.py | 396 | 447| 387 | 19720 | 118141 | 
| 55 | 11 django/db/models/fields/related_descriptors.py | 368 | 385| 188 | 19908 | 118141 | 
| 56 | 11 django/db/models/fields/related.py | 464 | 484| 169 | 20077 | 118141 | 
| 57 | 11 django/db/models/fields/related_descriptors.py | 1066 | 1087| 203 | 20280 | 118141 | 
| 58 | 11 django/db/models/fields/related.py | 1693 | 1743| 431 | 20711 | 118141 | 
| 59 | 11 django/db/models/fields/related.py | 1423 | 1461| 213 | 20924 | 118141 | 
| 60 | 11 django/db/models/fields/related.py | 1010 | 1034| 176 | 21100 | 118141 | 
| 61 | 11 django/db/models/fields/related.py | 823 | 835| 113 | 21213 | 118141 | 
| 62 | 11 django/db/models/fields/related.py | 1036 | 1072| 261 | 21474 | 118141 | 
| 63 | 11 django/db/models/query.py | 1922 | 1987| 543 | 22017 | 118141 | 
| 64 | 11 django/db/models/query.py | 1127 | 1161| 310 | 22327 | 118141 | 
| 65 | 12 django/db/models/options.py | 606 | 627| 156 | 22483 | 125834 | 
| 66 | 12 django/db/models/query.py | 1504 | 1513| 130 | 22613 | 125834 | 
| 67 | 12 django/db/models/fields/related_descriptors.py | 1089 | 1131| 381 | 22994 | 125834 | 
| 68 | 12 django/db/models/fields/related.py | 1074 | 1103| 217 | 23211 | 125834 | 
| 69 | 12 django/db/models/fields/related.py | 889 | 916| 237 | 23448 | 125834 | 
| 70 | 12 django/db/models/fields/related_lookups.py | 1 | 37| 205 | 23653 | 125834 | 
| 71 | 12 django/db/models/fields/related.py | 1105 | 1122| 133 | 23786 | 125834 | 
| 72 | 12 django/db/models/fields/related_descriptors.py | 1 | 91| 742 | 24528 | 125834 | 
| 73 | 12 django/db/models/fields/related.py | 604 | 670| 497 | 25025 | 125834 | 
| 74 | 12 django/db/models/query.py | 1515 | 1528| 130 | 25155 | 125834 | 
| 75 | 12 django/db/models/query.py | 2620 | 2643| 201 | 25356 | 125834 | 
| 76 | 12 django/db/models/fields/related_descriptors.py | 1334 | 1351| 154 | 25510 | 125834 | 
| 77 | 12 django/db/models/fields/related.py | 1894 | 1941| 505 | 26015 | 125834 | 
| 78 | 12 django/db/models/fields/related_descriptors.py | 895 | 928| 279 | 26294 | 125834 | 
| 79 | 12 django/db/models/query.py | 1 | 42| 270 | 26564 | 125834 | 
| 80 | 12 django/db/models/fields/related.py | 343 | 379| 294 | 26858 | 125834 | 
| 81 | **12 django/db/models/sql/query.py** | 449 | 544| 829 | 27687 | 125834 | 
| 82 | 13 django/db/migrations/autodetector.py | 90 | 102| 119 | 27806 | 139609 | 
| 83 | 13 django/db/models/query.py | 1890 | 1908| 190 | 27996 | 139609 | 
| 84 | 13 django/db/models/fields/related.py | 1594 | 1691| 655 | 28651 | 139609 | 
| 85 | 13 django/db/models/fields/related.py | 1179 | 1214| 267 | 28918 | 139609 | 
| 86 | 13 django/db/models/fields/related_lookups.py | 141 | 158| 216 | 29134 | 139609 | 
| 87 | 13 django/db/models/fields/related_descriptors.py | 499 | 572| 648 | 29782 | 139609 | 
| 88 | 13 django/db/models/fields/related_descriptors.py | 575 | 629| 338 | 30120 | 139609 | 
| 89 | **13 django/db/models/sql/query.py** | 171 | 270| 825 | 30945 | 139609 | 
| 90 | 13 django/db/models/fields/related.py | 1972 | 2006| 266 | 31211 | 139609 | 
| 91 | 13 django/db/models/fields/related.py | 1943 | 1970| 305 | 31516 | 139609 | 
| 92 | 13 django/db/models/fields/related_descriptors.py | 115 | 153| 266 | 31782 | 139609 | 
| 93 | 13 django/db/models/query.py | 1226 | 1247| 192 | 31974 | 139609 | 
| 94 | 13 django/db/models/fields/reverse_related.py | 1 | 19| 136 | 32110 | 139609 | 
| 95 | 13 django/db/models/query.py | 1284 | 1331| 361 | 32471 | 139609 | 
| 96 | **13 django/db/models/sql/query.py** | 575 | 595| 185 | 32656 | 139609 | 
| 97 | 13 django/db/models/fields/related_descriptors.py | 252 | 334| 760 | 33416 | 139609 | 
| 98 | 13 django/db/models/fields/related_descriptors.py | 1457 | 1507| 407 | 33823 | 139609 | 
| 99 | 13 django/db/backends/base/schema.py | 23 | 49| 180 | 34003 | 139609 | 
| 100 | **13 django/db/models/sql/query.py** | 985 | 1012| 272 | 34275 | 139609 | 
| 101 | 13 django/db/models/fields/reverse_related.py | 257 | 316| 386 | 34661 | 139609 | 
| 102 | 13 django/db/models/fields/reverse_related.py | 22 | 159| 796 | 35457 | 139609 | 
| 103 | 13 django/db/models/query.py | 1491 | 1502| 124 | 35581 | 139609 | 
| 104 | **13 django/db/models/sql/query.py** | 886 | 912| 280 | 35861 | 139609 | 
| 105 | 13 django/db/models/fields/related_descriptors.py | 201 | 250| 452 | 36313 | 139609 | 
| 106 | **13 django/db/models/sql/query.py** | 1454 | 1528| 753 | 37066 | 139609 | 
| 107 | 13 django/db/models/fields/reverse_related.py | 161 | 171| 130 | 37196 | 139609 | 
| 108 | 13 django/db/models/fields/related_descriptors.py | 632 | 663| 243 | 37439 | 139609 | 
| 109 | 13 django/db/models/sql/compiler.py | 1 | 36| 249 | 37688 | 139609 | 
| 110 | 13 django/db/models/query.py | 1333 | 1369| 267 | 37955 | 139609 | 
| 111 | 13 django/db/models/deletion.py | 1 | 93| 601 | 38556 | 139609 | 
| 112 | 14 django/contrib/admin/filters.py | 220 | 280| 541 | 39097 | 145335 | 
| 113 | 14 django/db/models/fields/related_descriptors.py | 454 | 497| 352 | 39449 | 145335 | 
| 114 | 14 django/db/models/query_utils.py | 367 | 393| 289 | 39738 | 145335 | 
| 115 | 14 django/db/models/fields/related_descriptors.py | 1046 | 1064| 199 | 39937 | 145335 | 
| 116 | **14 django/db/models/sql/query.py** | 1090 | 1121| 307 | 40244 | 145335 | 
| 117 | 14 django/db/models/query.py | 2444 | 2513| 665 | 40909 | 145335 | 
| 118 | 14 django/db/models/sql/compiler.py | 1415 | 1436| 199 | 41108 | 145335 | 
| 119 | **14 django/db/models/sql/query.py** | 1145 | 1172| 247 | 41355 | 145335 | 
| 120 | 14 django/db/models/fields/related_descriptors.py | 1238 | 1300| 544 | 41899 | 145335 | 
| 121 | 14 django/db/models/fields/related.py | 582 | 602| 138 | 42037 | 145335 | 
| 122 | 15 django/db/models/sql/datastructures.py | 87 | 151| 590 | 42627 | 147054 | 
| 123 | 15 django/db/models/fields/related.py | 707 | 731| 210 | 42837 | 147054 | 
| 124 | 15 django/db/models/sql/compiler.py | 509 | 517| 133 | 42970 | 147054 | 
| 125 | 15 django/db/models/fields/related.py | 486 | 512| 177 | 43147 | 147054 | 
| 126 | 15 django/db/models/query.py | 2550 | 2618| 785 | 43932 | 147054 | 
| 127 | 15 django/db/models/fields/related.py | 189 | 208| 155 | 44087 | 147054 | 
| 128 | 15 django/db/models/fields/related.py | 210 | 226| 142 | 44229 | 147054 | 
| 129 | 15 django/contrib/admin/filters.py | 632 | 645| 119 | 44348 | 147054 | 
| 130 | **15 django/db/models/sql/query.py** | 1174 | 1190| 154 | 44502 | 147054 | 
| 131 | 15 django/db/models/fields/related_descriptors.py | 1302 | 1332| 281 | 44783 | 147054 | 
| 132 | **15 django/db/models/sql/query.py** | 2617 | 2673| 823 | 45606 | 147054 | 
| 133 | 16 django/db/models/base.py | 1071 | 1123| 520 | 46126 | 165802 | 
| 134 | 16 django/db/models/fields/related.py | 515 | 580| 367 | 46493 | 165802 | 
| 135 | 16 django/db/models/fields/related.py | 128 | 154| 171 | 46664 | 165802 | 
| 136 | 16 django/db/models/fields/reverse_related.py | 221 | 254| 306 | 46970 | 165802 | 
| 137 | **16 django/db/models/sql/query.py** | 2301 | 2328| 291 | 47261 | 165802 | 
| 138 | 16 django/db/models/fields/related.py | 1 | 42| 267 | 47528 | 165802 | 
| 139 | 17 django/db/backends/mysql/features.py | 87 | 170| 670 | 48198 | 168346 | 
| 140 | 17 django/db/models/query.py | 290 | 351| 473 | 48671 | 168346 | 
| 141 | 17 django/db/models/query.py | 2515 | 2547| 314 | 48985 | 168346 | 
| 142 | 17 django/db/models/sql/compiler.py | 343 | 446| 772 | 49757 | 168346 | 
| 143 | 18 django/contrib/contenttypes/fields.py | 177 | 224| 417 | 50174 | 174174 | 
| 144 | 18 django/db/models/fields/reverse_related.py | 352 | 413| 366 | 50540 | 174174 | 
| 145 | 19 django/db/models/sql/where.py | 338 | 356| 149 | 50689 | 176756 | 
| 146 | 19 django/db/models/fields/related.py | 1124 | 1160| 284 | 50973 | 176756 | 
| 147 | 19 django/contrib/contenttypes/fields.py | 474 | 501| 258 | 51231 | 176756 | 
| 148 | 19 django/db/models/fields/related_descriptors.py | 1206 | 1236| 260 | 51491 | 176756 | 
| 149 | 19 django/db/models/fields/related_descriptors.py | 751 | 786| 268 | 51759 | 176756 | 
| 150 | 19 django/db/models/fields/related.py | 1864 | 1892| 275 | 52034 | 176756 | 
| 151 | 19 django/db/models/query.py | 382 | 412| 250 | 52284 | 176756 | 
| 152 | **19 django/db/models/sql/query.py** | 687 | 727| 383 | 52667 | 176756 | 
| 153 | 19 django/db/models/fields/related_descriptors.py | 1386 | 1455| 526 | 53193 | 176756 | 
| 154 | **19 django/db/models/sql/query.py** | 597 | 612| 136 | 53329 | 176756 | 
| 155 | **19 django/db/models/sql/query.py** | 2070 | 2119| 332 | 53661 | 176756 | 
| 156 | **19 django/db/models/sql/query.py** | 849 | 884| 398 | 54059 | 176756 | 
| 157 | 19 django/contrib/contenttypes/fields.py | 523 | 556| 229 | 54288 | 176756 | 
| 158 | 19 django/db/models/fields/related_descriptors.py | 1173 | 1204| 228 | 54516 | 176756 | 
| 159 | 19 django/db/models/sql/compiler.py | 1377 | 1400| 207 | 54723 | 176756 | 
| 160 | 19 django/db/models/query.py | 656 | 670| 138 | 54861 | 176756 | 
| 161 | 19 django/db/models/fields/related.py | 1745 | 1783| 413 | 55274 | 176756 | 
| 162 | **19 django/db/models/sql/query.py** | 148 | 168| 173 | 55447 | 176756 | 
| 163 | 20 django/db/models/sql/subqueries.py | 1 | 45| 311 | 55758 | 177984 | 
| 164 | 20 django/db/models/query.py | 1664 | 1709| 344 | 56102 | 177984 | 
| 165 | 20 django/db/models/fields/related.py | 1324 | 1421| 574 | 56676 | 177984 | 
| 166 | 20 django/db/models/fields/related.py | 1162 | 1177| 136 | 56812 | 177984 | 
| 167 | 20 django/db/models/sql/subqueries.py | 142 | 172| 190 | 57002 | 177984 | 
| 168 | 21 django/db/models/__init__.py | 1 | 116| 682 | 57684 | 178666 | 
| 169 | 21 django/db/models/sql/compiler.py | 39 | 76| 376 | 58060 | 178666 | 
| 170 | **21 django/db/models/sql/query.py** | 2121 | 2165| 332 | 58392 | 178666 | 
| 171 | **21 django/db/models/sql/query.py** | 1192 | 1205| 130 | 58522 | 178666 | 
| 172 | 21 django/db/models/query.py | 353 | 380| 225 | 58747 | 178666 | 
| 173 | 22 django/db/models/fields/__init__.py | 875 | 932| 429 | 59176 | 198036 | 
| 174 | 22 django/db/models/sql/compiler.py | 1000 | 1024| 211 | 59387 | 198036 | 


### Hint

```
Thanks for the report! Regression in b3db6c8dcb5145f7d45eff517bcd96460475c879. Reproduced at 881cc139e2d53cc1d3ccea7f38faa960f9e56597.
```

## Patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -779,7 +779,13 @@ def _get_only_select_mask(self, opts, mask, select_mask=None):
         # Only include fields mentioned in the mask.
         for field_name, field_mask in mask.items():
             field = opts.get_field(field_name)
-            field_select_mask = select_mask.setdefault(field, {})
+            # Retrieve the actual field associated with reverse relationships
+            # as that's what is expected in the select mask.
+            if field in opts.related_objects:
+                field_key = field.field
+            else:
+                field_key = field
+            field_select_mask = select_mask.setdefault(field_key, {})
             if field_mask:
                 if not field.is_relation:
                     raise FieldError(next(iter(field_mask)))

```

## Test Patch

```diff
diff --git a/tests/defer_regress/tests.py b/tests/defer_regress/tests.py
--- a/tests/defer_regress/tests.py
+++ b/tests/defer_regress/tests.py
@@ -178,6 +178,16 @@ def test_reverse_one_to_one_relations(self):
             self.assertEqual(i.one_to_one_item.name, "second")
         with self.assertNumQueries(1):
             self.assertEqual(i.value, 42)
+        with self.assertNumQueries(1):
+            i = Item.objects.select_related("one_to_one_item").only(
+                "name", "one_to_one_item__item"
+            )[0]
+            self.assertEqual(i.one_to_one_item.pk, o2o.pk)
+            self.assertEqual(i.name, "first")
+        with self.assertNumQueries(1):
+            self.assertEqual(i.one_to_one_item.name, "second")
+        with self.assertNumQueries(1):
+            self.assertEqual(i.value, 42)
 
     def test_defer_with_select_related(self):
         item1 = Item.objects.create(name="first", value=47)
@@ -277,6 +287,28 @@ def test_defer_many_to_many_ignored(self):
         with self.assertNumQueries(1):
             self.assertEqual(Request.objects.defer("items").get(), request)
 
+    def test_only_many_to_many_ignored(self):
+        location = Location.objects.create()
+        request = Request.objects.create(location=location)
+        with self.assertNumQueries(1):
+            self.assertEqual(Request.objects.only("items").get(), request)
+
+    def test_defer_reverse_many_to_many_ignored(self):
+        location = Location.objects.create()
+        request = Request.objects.create(location=location)
+        item = Item.objects.create(value=1)
+        request.items.add(item)
+        with self.assertNumQueries(1):
+            self.assertEqual(Item.objects.defer("request").get(), item)
+
+    def test_only_reverse_many_to_many_ignored(self):
+        location = Location.objects.create()
+        request = Request.objects.create(location=location)
+        item = Item.objects.create(value=1)
+        request.items.add(item)
+        with self.assertNumQueries(1):
+            self.assertEqual(Item.objects.only("request").get(), item)
+
 
 class DeferDeletionSignalsTests(TestCase):
     senders = [Item, Proxy]
diff --git a/tests/select_related_onetoone/tests.py b/tests/select_related_onetoone/tests.py
--- a/tests/select_related_onetoone/tests.py
+++ b/tests/select_related_onetoone/tests.py
@@ -249,6 +249,9 @@ def test_inheritance_deferred2(self):
             self.assertEqual(p.child1.name2, "n2")
         p = qs.get(name2="n2")
         with self.assertNumQueries(0):
+            self.assertEqual(p.child1.value, 1)
+            self.assertEqual(p.child1.child4.value4, 4)
+        with self.assertNumQueries(2):
             self.assertEqual(p.child1.name1, "n1")
             self.assertEqual(p.child1.child4.name1, "n1")
 

```


## Code snippets

### 1 - django/db/models/query.py:

Start line: 1546, End line: 1568

```python
class QuerySet(AltersData):

    def select_related(self, *fields):
        """
        Return a new QuerySet instance that will select related objects.

        If fields are specified, they must be ForeignKey fields and only those
        related objects are included in the selection.

        If select_related(None) is called, clear the list.
        """
        self._not_support_combined_queries("select_related")
        if self._fields is not None:
            raise TypeError(
                "Cannot call select_related() after .values() or .values_list()"
            )

        obj = self._chain()
        if fields == (None,):
            obj.query.select_related = False
        elif fields:
            obj.query.add_select_related(fields)
        else:
            obj.query.select_related = True
        return obj
```
### 2 - django/db/models/fields/related_descriptors.py:

Start line: 429, End line: 452

```python
class ReverseOneToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.related.field.get_local_related_value
        instance_attr = self.related.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        query = {"%s__in" % self.related.field.name: instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        for rel_obj in queryset:
            instance = instances_dict[rel_obj_attr(rel_obj)]
            self.related.field.set_cached_value(rel_obj, instance)
        return (
            queryset,
            rel_obj_attr,
            instance_attr,
            True,
            self.related.get_cache_name(),
            False,
        )
```
### 3 - django/db/models/query.py:

Start line: 1728, End line: 1747

```python
class QuerySet(AltersData):

    def only(self, *fields):
        """
        Essentially, the opposite of defer(). Only the fields passed into this
        method and that are not already specified as deferred are loaded
        immediately when the queryset is evaluated.
        """
        self._not_support_combined_queries("only")
        if self._fields is not None:
            raise TypeError("Cannot call only() after .values() or .values_list()")
        if fields == (None,):
            # Can only pass None to defer(), not only(), as the rest option.
            # That won't stop people trying to do this, so let's be explicit.
            raise TypeError("Cannot pass None as an argument to only().")
        for field in fields:
            field = field.split(LOOKUP_SEP, 1)[0]
            if field in self.query._filtered_relations:
                raise ValueError("only() is not supported with FilteredRelation.")
        clone = self._chain()
        clone.query.add_immediate_loading(fields)
        return clone
```
### 4 - django/db/models/fields/related_descriptors.py:

Start line: 406, End line: 427

```python
class ReverseOneToOneDescriptor:

    @cached_property
    def RelatedObjectDoesNotExist(self):
        # The exception isn't created at initialization time for the sake of
        # consistency with `ForwardManyToOneDescriptor`.
        return type(
            "RelatedObjectDoesNotExist",
            (self.related.related_model.DoesNotExist, AttributeError),
            {
                "__module__": self.related.model.__module__,
                "__qualname__": "%s.%s.RelatedObjectDoesNotExist"
                % (
                    self.related.model.__qualname__,
                    self.related.name,
                ),
            },
        )

    def is_cached(self, instance):
        return self.related.is_cached(instance)

    def get_queryset(self, **hints):
        return self.related.related_model._base_manager.db_manager(hints=hints).all()
```
### 5 - django/db/models/query.py:

Start line: 1570, End line: 1593

```python
class QuerySet(AltersData):

    def prefetch_related(self, *lookups):
        """
        Return a new QuerySet instance that will prefetch the specified
        Many-To-One and Many-To-Many related objects when the QuerySet is
        evaluated.

        When prefetch_related() is called more than once, append to the list of
        prefetch lookups. If prefetch_related(None) is called, clear the list.
        """
        self._not_support_combined_queries("prefetch_related")
        clone = self._chain()
        if lookups == (None,):
            clone._prefetch_related_lookups = ()
        else:
            for lookup in lookups:
                if isinstance(lookup, Prefetch):
                    lookup = lookup.prefetch_to
                lookup = lookup.split(LOOKUP_SEP, 1)[0]
                if lookup in self.query._filtered_relations:
                    raise ValueError(
                        "prefetch_related() is not supported with FilteredRelation."
                    )
            clone._prefetch_related_lookups = clone._prefetch_related_lookups + lookups
        return clone
```
### 6 - django/db/models/sql/compiler.py:

Start line: 1246, End line: 1375

```python
class SQLCompiler:
    ordering_parts =

    def get_related_selections(
        self,
        select,
        select_mask,
        opts=None,
        root_alias=None,
        cur_depth=1,
        requested=None,
        restricted=None,
    ):
        # ... other code

        if restricted:
            related_fields = [
                (o.field, o.related_model)
                for o in opts.related_objects
                if o.field.unique and not o.many_to_many
            ]
            for related_field, model in related_fields:
                related_select_mask = select_mask.get(related_field) or {}
                if not select_related_descend(
                    related_field,
                    restricted,
                    requested,
                    related_select_mask,
                    reverse=True,
                ):
                    continue

                related_field_name = related_field.related_query_name()
                fields_found.add(related_field_name)

                join_info = self.query.setup_joins(
                    [related_field_name], opts, root_alias
                )
                alias = join_info.joins[-1]
                from_parent = issubclass(model, opts.model) and model is not opts.model
                klass_info = {
                    "model": model,
                    "field": related_field,
                    "reverse": True,
                    "local_setter": related_field.remote_field.set_cached_value,
                    "remote_setter": related_field.set_cached_value,
                    "from_parent": from_parent,
                }
                related_klass_infos.append(klass_info)
                select_fields = []
                columns = self.get_default_columns(
                    related_select_mask,
                    start_alias=alias,
                    opts=model._meta,
                    from_parent=opts.model,
                )
                for col in columns:
                    select_fields.append(len(select))
                    select.append((col, None))
                klass_info["select_fields"] = select_fields
                next = requested.get(related_field.related_query_name(), {})
                next_klass_infos = self.get_related_selections(
                    select,
                    related_select_mask,
                    model._meta,
                    alias,
                    cur_depth + 1,
                    next,
                    restricted,
                )
                get_related_klass_infos(klass_info, next_klass_infos)

            def local_setter(final_field, obj, from_obj):
                # Set a reverse fk object when relation is non-empty.
                if from_obj:
                    final_field.remote_field.set_cached_value(from_obj, obj)

            def local_setter_noop(obj, from_obj):
                pass

            def remote_setter(name, obj, from_obj):
                setattr(from_obj, name, obj)

            for name in list(requested):
                # Filtered relations work only on the topmost level.
                if cur_depth > 1:
                    break
                if name in self.query._filtered_relations:
                    fields_found.add(name)
                    final_field, _, join_opts, joins, _, _ = self.query.setup_joins(
                        [name], opts, root_alias
                    )
                    model = join_opts.model
                    alias = joins[-1]
                    from_parent = (
                        issubclass(model, opts.model) and model is not opts.model
                    )
                    klass_info = {
                        "model": model,
                        "field": final_field,
                        "reverse": True,
                        "local_setter": (
                            partial(local_setter, final_field)
                            if len(joins) <= 2
                            else local_setter_noop
                        ),
                        "remote_setter": partial(remote_setter, name),
                        "from_parent": from_parent,
                    }
                    related_klass_infos.append(klass_info)
                    select_fields = []
                    field_select_mask = select_mask.get((name, final_field)) or {}
                    columns = self.get_default_columns(
                        field_select_mask,
                        start_alias=alias,
                        opts=model._meta,
                        from_parent=opts.model,
                    )
                    for col in columns:
                        select_fields.append(len(select))
                        select.append((col, None))
                    klass_info["select_fields"] = select_fields
                    next_requested = requested.get(name, {})
                    next_klass_infos = self.get_related_selections(
                        select,
                        field_select_mask,
                        opts=model._meta,
                        root_alias=alias,
                        cur_depth=cur_depth + 1,
                        requested=next_requested,
                        restricted=restricted,
                    )
                    get_related_klass_infos(klass_info, next_klass_infos)
            fields_not_found = set(requested).difference(fields_found)
            if fields_not_found:
                invalid_fields = ("'%s'" % s for s in fields_not_found)
                raise FieldError(
                    "Invalid field name(s) given in select_related: %s. "
                    "Choices are: %s"
                    % (
                        ", ".join(invalid_fields),
                        ", ".join(_get_field_choices()) or "(none)",
                    )
                )
        return related_klass_infos
    # ... other code
```
### 7 - django/db/models/sql/query.py:

Start line: 2252, End line: 2299

```python
class Query(BaseExpression):

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

    def add_extra(self, select, select_params, where, params, tables, order_by):
        """
        Add data to the various extra_* attributes for user-created additions
        to the query.
        """
        if select:
            # We need to pair any placeholder markers in the 'select'
            # dictionary with their parameters in 'select_params' so that
            # subsequent updates to the select dictionary also adjust the
            # parameters appropriately.
            select_pairs = {}
            if select_params:
                param_iter = iter(select_params)
            else:
                param_iter = iter([])
            for name, entry in select.items():
                self.check_alias(name)
                entry = str(entry)
                entry_params = []
                pos = entry.find("%s")
                while pos != -1:
                    if pos == 0 or entry[pos - 1] != "%":
                        entry_params.append(next(param_iter))
                    pos = entry.find("%s", pos + 2)
                select_pairs[name] = (entry, entry_params)
            self.extra.update(select_pairs)
        if where or params:
            self.where.add(ExtraWhere(where, params), AND)
        if tables:
            self.extra_tables += tuple(tables)
        if order_by:
            self.extra_order_by = order_by
```
### 8 - django/db/models/fields/related_descriptors.py:

Start line: 155, End line: 199

```python
class ForwardManyToOneDescriptor:

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
        queryset._add_hints(instance=instances[0])

        rel_obj_attr = self.field.get_foreign_related_value
        instance_attr = self.field.get_local_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        related_field = self.field.foreign_related_fields[0]
        remote_field = self.field.remote_field

        # FIXME: This will need to be revisited when we introduce support for
        # composite fields. In the meantime we take this practical approach to
        # solve a regression on 1.6 when the reverse manager in hidden
        # (related_name ends with a '+'). Refs #21410.
        # The check for len(...) == 1 is a special case that allows the query
        # to be join-less and smaller. Refs #21760.
        if remote_field.is_hidden() or len(self.field.foreign_related_fields) == 1:
            query = {
                "%s__in"
                % related_field.name: {instance_attr(inst)[0] for inst in instances}
            }
        else:
            query = {"%s__in" % self.field.related_query_name(): instances}
        queryset = queryset.filter(**query)

        # Since we're going to assign directly in the cache,
        # we must manage the reverse relation cache manually.
        if not remote_field.multiple:
            for rel_obj in queryset:
                instance = instances_dict[rel_obj_attr(rel_obj)]
                remote_field.set_cached_value(rel_obj, instance)
        return (
            queryset,
            rel_obj_attr,
            instance_attr,
            True,
            self.field.get_cache_name(),
            False,
        )

    def get_object(self, instance):
        qs = self.get_queryset(instance=instance)
        # Assuming the database enforces foreign keys, this won't fail.
        return qs.get(self.field.get_reverse_related_filter(instance))
```
### 9 - django/db/models/fields/related.py:

Start line: 156, End line: 187

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
### 10 - django/db/models/sql/compiler.py:

Start line: 1143, End line: 1244

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def get_related_selections(
        self,
        select,
        select_mask,
        opts=None,
        root_alias=None,
        cur_depth=1,
        requested=None,
        restricted=None,
    ):
        """
        Fill in the information needed for a select_related query. The current
        depth is measured as the number of connections away from the root model
        (for example, cur_depth=1 means we are looking at models with direct
        connections to the root model).
        """

        def _get_field_choices():
            direct_choices = (f.name for f in opts.fields if f.is_relation)
            reverse_choices = (
                f.field.related_query_name()
                for f in opts.related_objects
                if f.field.unique
            )
            return chain(
                direct_choices, reverse_choices, self.query._filtered_relations
            )

        related_klass_infos = []
        if not restricted and cur_depth > self.query.max_depth:
            # We've recursed far enough; bail out.
            return related_klass_infos

        if not opts:
            opts = self.query.get_meta()
            root_alias = self.query.get_initial_alias()

        # Setup for the case when only particular related fields should be
        # included in the related selection.
        fields_found = set()
        if requested is None:
            restricted = isinstance(self.query.select_related, dict)
            if restricted:
                requested = self.query.select_related

        def get_related_klass_infos(klass_info, related_klass_infos):
            klass_info["related_klass_infos"] = related_klass_infos

        for f in opts.fields:
            fields_found.add(f.name)

            if restricted:
                next = requested.get(f.name, {})
                if not f.is_relation:
                    # If a non-related field is used like a relation,
                    # or if a single non-relational field is given.
                    if next or f.name in requested:
                        raise FieldError(
                            "Non-relational field given in select_related: '%s'. "
                            "Choices are: %s"
                            % (
                                f.name,
                                ", ".join(_get_field_choices()) or "(none)",
                            )
                        )
            else:
                next = False

            if not select_related_descend(f, restricted, requested, select_mask):
                continue
            related_select_mask = select_mask.get(f) or {}
            klass_info = {
                "model": f.remote_field.model,
                "field": f,
                "reverse": False,
                "local_setter": f.set_cached_value,
                "remote_setter": f.remote_field.set_cached_value
                if f.unique
                else lambda x, y: None,
                "from_parent": False,
            }
            related_klass_infos.append(klass_info)
            select_fields = []
            _, _, _, joins, _, _ = self.query.setup_joins([f.name], opts, root_alias)
            alias = joins[-1]
            columns = self.get_default_columns(
                related_select_mask, start_alias=alias, opts=f.remote_field.model._meta
            )
            for col in columns:
                select_fields.append(len(select))
                select.append((col, None))
            klass_info["select_fields"] = select_fields
            next_klass_infos = self.get_related_selections(
                select,
                related_select_mask,
                f.remote_field.model._meta,
                alias,
                cur_depth + 1,
                next,
                restricted,
            )
            get_related_klass_infos(klass_info, next_klass_infos)
        # ... other code
    # ... other code
```
### 12 - django/db/models/sql/query.py:

Start line: 775, End line: 790

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
### 13 - django/db/models/sql/query.py:

Start line: 2377, End line: 2437

```python
class Query(BaseExpression):

    def set_values(self, fields):
        self.select_related = False
        self.clear_deferred_loading()
        self.clear_select_fields()
        self.has_select_fields = True

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
                    elif f in self.annotations:
                        raise FieldError(
                            f"Cannot select the '{f}' alias. Use annotate() to "
                            "promote it."
                        )
                    else:
                        # Call `names_to_path` to ensure a FieldError including
                        # annotations about to be masked as valid choices if
                        # `f` is not resolvable.
                        if self.annotation_select:
                            self.names_to_path(f.split(LOOKUP_SEP), self.model._meta)
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
### 15 - django/db/models/sql/query.py:

Start line: 1651, End line: 1749

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
### 23 - django/db/models/sql/query.py:

Start line: 729, End line: 773

```python
class Query(BaseExpression):

    def _get_defer_select_mask(self, opts, mask, select_mask=None):
        if select_mask is None:
            select_mask = {}
        select_mask[opts.pk] = {}
        # All concrete fields that are not part of the defer mask must be
        # loaded. If a relational field is encountered it gets added to the
        # mask for it be considered if `select_related` and the cycle continues
        # by recursively calling this function.
        for field in opts.concrete_fields:
            field_mask = mask.pop(field.name, None)
            field_att_mask = mask.pop(field.attname, None)
            if field_mask is None and field_att_mask is None:
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
                reverse_rel = opts.get_field(field_name)
                # While virtual fields such as many-to-many and generic foreign
                # keys cannot be effectively deferred we've historically
                # allowed them to be passed to QuerySet.defer(). Ignore such
                # field references until a layer of validation at mask
                # alteration time will be implemented eventually.
                if not hasattr(reverse_rel, "field"):
                    continue
                field = reverse_rel.field
                field_select_mask = select_mask.setdefault(field, {})
            related_model = field.model._meta.concrete_model
            self._get_defer_select_mask(
                related_model._meta, field_mask, field_select_mask
            )
        return select_mask
```
### 27 - django/db/models/sql/query.py:

Start line: 1970, End line: 2037

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
### 31 - django/db/models/sql/query.py:

Start line: 1262, End line: 1296

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
### 32 - django/db/models/sql/query.py:

Start line: 1840, End line: 1864

```python
class Query(BaseExpression):

    def setup_joins(
        self,
        names,
        opts,
        alias,
        can_reuse=None,
        allow_many=True,
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
            reuse = can_reuse if join.m2m else None
            alias = self.join(connection, reuse=reuse)
            joins.append(alias)
        return JoinInfo(final_field, targets, opts, joins, path, final_transformer)
```
### 33 - django/db/models/sql/query.py:

Start line: 1599, End line: 1633

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
        filtered_relation.condition = rename_prefix_from_q(
            filtered_relation.relation_name,
            alias,
            filtered_relation.condition,
        )
        self._filtered_relations[filtered_relation.alias] = filtered_relation
```
### 35 - django/db/models/sql/query.py:

Start line: 1, End line: 92

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


def rename_prefix_from_q(prefix, replacement, q):
    return Q.create(
        [
            rename_prefix_from_q(prefix, replacement, c)
            if isinstance(c, Node)
            else (c[0].replace(prefix, replacement, 1), c[1])
            for c in q.children
        ],
        q.connector,
        q.negated,
    )


JoinInfo = namedtuple(
    "JoinInfo",
    ("final_field", "targets", "opts", "joins", "path", "transform_function"),
)
```
### 49 - django/db/models/sql/query.py:

Start line: 1920, End line: 1968

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
### 81 - django/db/models/sql/query.py:

Start line: 449, End line: 544

```python
class Query(BaseExpression):

    def get_aggregation(self, using, aggregate_exprs):
        # ... other code
        if (
            isinstance(self.group_by, tuple)
            or self.is_sliced
            or has_existing_aggregation
            or refs_subquery
            or qualify
            or self.distinct
            or self.combinator
        ):
            from django.db.models.sql.subqueries import AggregateQuery

            inner_query = self.clone()
            inner_query.subquery = True
            outer_query = AggregateQuery(self.model, inner_query)
            inner_query.select_for_update = False
            inner_query.select_related = False
            inner_query.set_annotation_mask(self.annotation_select)
            # Queries with distinct_fields need ordering and when a limit is
            # applied we must take the slice from the ordered query. Otherwise
            # no need for ordering.
            inner_query.clear_ordering(force=False)
            if not inner_query.distinct:
                # If the inner query uses default select and it has some
                # aggregate annotations, then we must make sure the inner
                # query is grouped by the main model's primary key. However,
                # clearing the select clause can alter results if distinct is
                # used.
                if inner_query.default_cols and has_existing_aggregation:
                    inner_query.group_by = (
                        self.model._meta.pk.get_col(inner_query.get_initial_alias()),
                    )
                inner_query.default_cols = False
                if not qualify:
                    # Mask existing annotations that are not referenced by
                    # aggregates to be pushed to the outer query unless
                    # filtering against window functions is involved as it
                    # requires complex realising.
                    annotation_mask = set()
                    if isinstance(self.group_by, tuple):
                        for expr in self.group_by:
                            annotation_mask |= expr.get_refs()
                    for aggregate in aggregates.values():
                        annotation_mask |= aggregate.get_refs()
                    inner_query.set_annotation_mask(annotation_mask)

            # Add aggregates to the outer AggregateQuery. This requires making
            # sure all columns referenced by the aggregates are selected in the
            # inner query. It is achieved by retrieving all column references
            # by the aggregates, explicitly selecting them in the inner query,
            # and making sure the aggregates are repointed to them.
            col_refs = {}
            for alias, aggregate in aggregates.items():
                replacements = {}
                for col in self._gen_cols([aggregate], resolve_refs=False):
                    if not (col_ref := col_refs.get(col)):
                        index = len(col_refs) + 1
                        col_alias = f"__col{index}"
                        col_ref = Ref(col_alias, col)
                        col_refs[col] = col_ref
                        inner_query.annotations[col_alias] = col
                        inner_query.append_annotation_mask([col_alias])
                    replacements[col] = col_ref
                outer_query.annotations[alias] = aggregate.replace_expressions(
                    replacements
                )
            if (
                inner_query.select == ()
                and not inner_query.default_cols
                and not inner_query.annotation_select_mask
            ):
                # In case of Model.objects[0:3].count(), there would be no
                # field selected in the inner query, yet we must use a subquery.
                # So, make sure at least one field is selected.
                inner_query.select = (
                    self.model._meta.pk.get_col(inner_query.get_initial_alias()),
                )
        else:
            outer_query = self
            self.select = ()
            self.default_cols = False
            self.extra = {}
            if self.annotations:
                # Inline reference to existing annotations and mask them as
                # they are unnecessary given only the summarized aggregations
                # are requested.
                replacements = {
                    Ref(alias, annotation): annotation
                    for alias, annotation in self.annotations.items()
                }
                self.annotations = {
                    alias: aggregate.replace_expressions(replacements)
                    for alias, aggregate in aggregates.items()
                }
            else:
                self.annotations = aggregates
            self.set_annotation_mask(aggregates)
        # ... other code
```
### 89 - django/db/models/sql/query.py:

Start line: 171, End line: 270

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
    has_select_fields = False
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
```
### 96 - django/db/models/sql/query.py:

Start line: 575, End line: 595

```python
class Query(BaseExpression):

    def exists(self, limit=True):
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
            q.combined_queries = tuple(
                combined_query.exists(limit=False)
                for combined_query in q.combined_queries
            )
        q.clear_ordering(force=True)
        if limit:
            q.set_limits(high=1)
        q.add_annotation(Value(1), "a")
        return q
```
### 100 - django/db/models/sql/query.py:

Start line: 985, End line: 1012

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
### 104 - django/db/models/sql/query.py:

Start line: 886, End line: 912

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
### 106 - django/db/models/sql/query.py:

Start line: 1454, End line: 1528

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
        check_filterable=True,
        summarize=False,
        update_join_types=True,
    ):
        # ... other code

        try:
            join_info = self.setup_joins(
                parts,
                opts,
                alias,
                can_reuse=can_reuse,
                allow_many=allow_many,
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
### 116 - django/db/models/sql/query.py:

Start line: 1090, End line: 1121

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
### 119 - django/db/models/sql/query.py:

Start line: 1145, End line: 1172

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
### 130 - django/db/models/sql/query.py:

Start line: 1174, End line: 1190

```python
class Query(BaseExpression):

    def get_external_cols(self):
        exprs = chain(self.annotations.values(), self.where.children)
        return [
            col
            for col in self._gen_cols(exprs, include_external=True)
            if col.alias in self.external_aliases
        ]

    def get_group_by_cols(self, wrapper=None):
        # If wrapper is referenced by an alias for an explicit GROUP BY through
        # values() a reference to this expression and not the self must be
        # returned to ensure external column references are not grouped against
        # as well.
        external_cols = self.get_external_cols()
        if any(col.possibly_multivalued for col in external_cols):
            return [wrapper or self]
        return external_cols
```
### 132 - django/db/models/sql/query.py:

Start line: 2617, End line: 2673

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
### 137 - django/db/models/sql/query.py:

Start line: 2301, End line: 2328

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
### 152 - django/db/models/sql/query.py:

Start line: 687, End line: 727

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
### 154 - django/db/models/sql/query.py:

Start line: 597, End line: 612

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
### 155 - django/db/models/sql/query.py:

Start line: 2070, End line: 2119

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
### 156 - django/db/models/sql/query.py:

Start line: 849, End line: 884

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
### 162 - django/db/models/sql/query.py:

Start line: 148, End line: 168

```python
class RawQuery:

    def _execute_query(self):
        connection = connections[self.using]

        # Adapt parameters to the database, as much as possible considering
        # that the target type isn't known. See #17755.
        params_type = self.params_type
        adapter = connection.ops.adapt_unknown_value
        if params_type is tuple:
            params = tuple(adapter(val) for val in self.params)
        elif params_type is dict:
            params = {key: adapter(val) for key, val in self.params.items()}
        elif params_type is None:
            params = None
        else:
            raise RuntimeError("Unexpected params type: %s" % params_type)

        self.cursor = connection.cursor()
        self.cursor.execute(self.sql, params)


ExplainInfo = namedtuple("ExplainInfo", ("format", "options"))
```
### 170 - django/db/models/sql/query.py:

Start line: 2121, End line: 2165

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
### 171 - django/db/models/sql/query.py:

Start line: 1192, End line: 1205

```python
class Query(BaseExpression):

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
