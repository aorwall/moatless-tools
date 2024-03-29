# django__django-13689

| **django/django** | `ead37dfb580136cc27dbd487a1f1ad90c9235d15` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2009 |
| **Any found context length** | 2009 |
| **Avg pos** | 7.0 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -915,9 +915,13 @@ def get_source_expressions(self):
         return [self.expression]
 
     def get_group_by_cols(self, alias=None):
-        expression = self.expression.copy()
-        expression.output_field = self.output_field
-        return expression.get_group_by_cols(alias=alias)
+        if isinstance(self.expression, Expression):
+            expression = self.expression.copy()
+            expression.output_field = self.output_field
+            return expression.get_group_by_cols(alias=alias)
+        # For non-expressions e.g. an SQL WHERE clause, the entire
+        # `expression` must be included in the GROUP BY clause.
+        return super().get_group_by_cols()
 
     def as_sql(self, compiler, connection):
         return compiler.compile(self.expression)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/models/expressions.py | 918 | 920 | 7 | 2 | 2009


## Problem Statement

```
Aggregating when grouping on an ExpressionWrapper omits the expression from the group by
Description
	
I ran into this with Postgres on Django 3.1.3, I'm not sure what other versions it exists on.
print(
	Fred.objects.annotate(
		bob_id__is_null=ExpressionWrapper(
			Q(bob_id=None), 
			output_field=BooleanField()
		)
	).values(
		"bob_id__is_null"
	).annotate(
		id__count=Count("id", distinct=True)
	).values(
		"bob_id__is_null", 
		"id__count"
	).query
)
SELECT 
	"main_fred"."bob_id" IS NULL AS "bob_id__is_null", 
	COUNT(DISTINCT "main_fred"."id") AS "id__count" 
FROM "main_fred"
GROUP BY "main_fred"."bob_id"
On the last line there the group by has dropped the "IS NULL"

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/aggregates.py | 45 | 68| 294 | 294 | 1301 | 
| 2 | **2 django/db/models/expressions.py** | 929 | 993| 591 | 885 | 12174 | 
| 3 | **2 django/db/models/expressions.py** | 814 | 848| 292 | 1177 | 12174 | 
| 4 | **2 django/db/models/expressions.py** | 746 | 776| 233 | 1410 | 12174 | 
| 5 | **2 django/db/models/expressions.py** | 1139 | 1171| 254 | 1664 | 12174 | 
| 6 | **2 django/db/models/expressions.py** | 728 | 744| 168 | 1832 | 12174 | 
| **-> 7 <-** | **2 django/db/models/expressions.py** | 901 | 926| 177 | 2009 | 12174 | 
| 8 | 3 django/db/models/sql/compiler.py | 149 | 197| 523 | 2532 | 26565 | 
| 9 | **3 django/db/models/expressions.py** | 332 | 387| 368 | 2900 | 26565 | 
| 10 | 3 django/db/models/aggregates.py | 70 | 96| 266 | 3166 | 26565 | 
| 11 | **3 django/db/models/expressions.py** | 1198 | 1223| 248 | 3414 | 26565 | 
| 12 | **3 django/db/models/expressions.py** | 1044 | 1074| 281 | 3695 | 26565 | 
| 13 | 4 django/contrib/postgres/aggregates/general.py | 1 | 69| 423 | 4118 | 26989 | 
| 14 | **4 django/db/models/expressions.py** | 1336 | 1358| 215 | 4333 | 26989 | 
| 15 | **4 django/db/models/expressions.py** | 1257 | 1300| 378 | 4711 | 26989 | 
| 16 | 5 django/db/models/sql/query.py | 373 | 423| 494 | 5205 | 49525 | 
| 17 | **5 django/db/models/expressions.py** | 1077 | 1136| 442 | 5647 | 49525 | 
| 18 | 6 django/contrib/postgres/aggregates/mixins.py | 1 | 20| 182 | 5829 | 49963 | 
| 19 | **6 django/db/models/expressions.py** | 779 | 793| 120 | 5949 | 49963 | 
| 20 | **6 django/db/models/expressions.py** | 1174 | 1196| 188 | 6137 | 49963 | 
| 21 | 7 django/contrib/postgres/aggregates/statistics.py | 1 | 66| 419 | 6556 | 50382 | 
| 22 | **7 django/db/models/expressions.py** | 1225 | 1254| 211 | 6767 | 50382 | 
| 23 | 7 django/db/models/sql/query.py | 1308 | 1373| 772 | 7539 | 50382 | 
| 24 | 7 django/db/models/aggregates.py | 99 | 119| 158 | 7697 | 50382 | 
| 25 | **7 django/db/models/expressions.py** | 489 | 514| 303 | 8000 | 50382 | 
| 26 | 7 django/db/models/sql/query.py | 1716 | 1755| 439 | 8439 | 50382 | 
| 27 | **7 django/db/models/expressions.py** | 219 | 253| 285 | 8724 | 50382 | 
| 28 | 8 django/db/backends/sqlite3/operations.py | 40 | 66| 232 | 8956 | 53472 | 
| 29 | 9 django/contrib/gis/db/models/aggregates.py | 29 | 46| 216 | 9172 | 54089 | 
| 30 | 9 django/db/models/sql/query.py | 2188 | 2235| 394 | 9566 | 54089 | 
| 31 | 9 django/db/models/aggregates.py | 1 | 43| 344 | 9910 | 54089 | 
| 32 | **9 django/db/models/expressions.py** | 795 | 811| 153 | 10063 | 54089 | 
| 33 | **9 django/db/models/expressions.py** | 996 | 1042| 377 | 10440 | 54089 | 
| 34 | **9 django/db/models/expressions.py** | 1302 | 1334| 251 | 10691 | 54089 | 
| 35 | **9 django/db/models/expressions.py** | 442 | 487| 314 | 11005 | 54089 | 
| 36 | 9 django/db/models/sql/compiler.py | 63 | 147| 881 | 11886 | 54089 | 
| 37 | 9 django/db/models/aggregates.py | 122 | 158| 245 | 12131 | 54089 | 
| 38 | **9 django/db/models/expressions.py** | 389 | 414| 186 | 12317 | 54089 | 
| 39 | 9 django/contrib/gis/db/models/aggregates.py | 49 | 84| 207 | 12524 | 54089 | 
| 40 | **9 django/db/models/expressions.py** | 311 | 330| 184 | 12708 | 54089 | 
| 41 | 10 django/db/models/sql/where.py | 233 | 249| 130 | 12838 | 55903 | 
| 42 | **10 django/db/models/expressions.py** | 591 | 630| 290 | 13128 | 55903 | 
| 43 | 11 django/db/models/functions/comparison.py | 57 | 75| 205 | 13333 | 57307 | 
| 44 | **11 django/db/models/expressions.py** | 1 | 30| 204 | 13537 | 57307 | 
| 45 | 11 django/db/models/sql/query.py | 1995 | 2044| 420 | 13957 | 57307 | 
| 46 | 12 django/db/models/query.py | 1098 | 1139| 323 | 14280 | 74541 | 
| 47 | **12 django/db/models/expressions.py** | 150 | 189| 263 | 14543 | 74541 | 
| 48 | 12 django/db/models/sql/query.py | 2237 | 2269| 228 | 14771 | 74541 | 
| 49 | 12 django/db/models/sql/where.py | 212 | 230| 131 | 14902 | 74541 | 
| 50 | 12 django/db/models/sql/query.py | 713 | 748| 389 | 15291 | 74541 | 
| 51 | 12 django/contrib/postgres/aggregates/mixins.py | 36 | 49| 145 | 15436 | 74541 | 
| 52 | 12 django/db/models/sql/query.py | 1069 | 1098| 245 | 15681 | 74541 | 
| 53 | **12 django/db/models/expressions.py** | 679 | 704| 268 | 15949 | 74541 | 
| 54 | 13 django/contrib/postgres/constraints.py | 69 | 91| 213 | 16162 | 75966 | 
| 55 | 13 django/db/models/sql/query.py | 1650 | 1674| 227 | 16389 | 75966 | 
| 56 | **13 django/db/models/expressions.py** | 547 | 588| 298 | 16687 | 75966 | 
| 57 | 13 django/db/models/sql/query.py | 514 | 561| 403 | 17090 | 75966 | 
| 58 | 13 django/db/models/sql/query.py | 1487 | 1572| 801 | 17891 | 75966 | 
| 59 | 13 django/db/models/sql/query.py | 425 | 512| 890 | 18781 | 75966 | 
| 60 | 13 django/db/models/sql/query.py | 1757 | 1822| 666 | 19447 | 75966 | 
| 61 | **13 django/db/models/expressions.py** | 417 | 439| 204 | 19651 | 75966 | 
| 62 | 13 django/db/models/sql/query.py | 1442 | 1469| 283 | 19934 | 75966 | 
| 63 | 13 django/db/models/functions/comparison.py | 78 | 92| 175 | 20109 | 75966 | 
| 64 | 13 django/db/models/query.py | 842 | 871| 248 | 20357 | 75966 | 
| 65 | 13 django/contrib/postgres/aggregates/mixins.py | 22 | 34| 124 | 20481 | 75966 | 
| 66 | 13 django/contrib/gis/db/models/aggregates.py | 1 | 27| 199 | 20680 | 75966 | 
| 67 | 14 django/template/defaulttags.py | 338 | 362| 231 | 20911 | 87109 | 
| 68 | **14 django/db/models/expressions.py** | 517 | 544| 214 | 21125 | 87109 | 
| 69 | **14 django/db/models/expressions.py** | 884 | 898| 123 | 21248 | 87109 | 
| 70 | 15 django/db/models/__init__.py | 1 | 53| 619 | 21867 | 87728 | 
| 71 | 15 django/db/models/sql/query.py | 909 | 931| 248 | 22115 | 87728 | 
| 72 | 15 django/db/models/sql/query.py | 638 | 662| 269 | 22384 | 87728 | 
| 73 | 15 django/db/models/sql/query.py | 2345 | 2361| 177 | 22561 | 87728 | 
| 74 | **15 django/db/models/expressions.py** | 707 | 726| 170 | 22731 | 87728 | 
| 75 | 15 django/db/models/query.py | 801 | 840| 322 | 23053 | 87728 | 
| 76 | 16 django/db/models/lookups.py | 504 | 525| 172 | 23225 | 92755 | 
| 77 | **16 django/db/models/expressions.py** | 633 | 677| 419 | 23644 | 92755 | 
| 78 | **16 django/db/models/expressions.py** | 255 | 282| 184 | 23828 | 92755 | 
| 79 | 16 django/db/models/lookups.py | 103 | 157| 427 | 24255 | 92755 | 
| 80 | 16 django/db/models/sql/query.py | 1129 | 1161| 338 | 24593 | 92755 | 
| 81 | 17 django/db/models/functions/text.py | 191 | 211| 190 | 24783 | 95091 | 
| 82 | 17 django/db/models/sql/query.py | 1855 | 1904| 329 | 25112 | 95091 | 
| 83 | 17 django/db/models/sql/query.py | 137 | 231| 833 | 25945 | 95091 | 
| 84 | 17 django/db/backends/sqlite3/operations.py | 304 | 318| 148 | 26093 | 95091 | 
| 85 | 18 django/contrib/gis/db/backends/oracle/operations.py | 175 | 225| 358 | 26451 | 97173 | 
| 86 | **18 django/db/models/expressions.py** | 191 | 217| 223 | 26674 | 97173 | 
| 87 | 18 django/db/models/sql/query.py | 1034 | 1067| 349 | 27023 | 97173 | 
| 88 | **18 django/db/models/expressions.py** | 33 | 147| 836 | 27859 | 97173 | 
| 89 | 18 django/contrib/postgres/constraints.py | 157 | 167| 132 | 27991 | 97173 | 
| 90 | 18 django/db/models/sql/where.py | 157 | 193| 243 | 28234 | 97173 | 
| 91 | 19 django/contrib/postgres/search.py | 160 | 195| 313 | 28547 | 99395 | 
| 92 | 19 django/db/models/functions/text.py | 95 | 116| 202 | 28749 | 99395 | 
| 93 | 19 django/db/models/query.py | 365 | 399| 314 | 29063 | 99395 | 
| 94 | 19 django/contrib/postgres/constraints.py | 129 | 155| 231 | 29294 | 99395 | 
| 95 | 20 django/db/backends/mysql/operations.py | 278 | 289| 165 | 29459 | 103099 | 
| 96 | 20 django/template/defaulttags.py | 1166 | 1232| 648 | 30107 | 103099 | 
| 97 | 20 django/db/models/query.py | 1351 | 1399| 405 | 30512 | 103099 | 
| 98 | 20 django/db/models/functions/comparison.py | 115 | 144| 244 | 30756 | 103099 | 
| 99 | 20 django/db/models/query.py | 320 | 346| 222 | 30978 | 103099 | 
| 100 | 21 django/db/models/indexes.py | 1 | 61| 526 | 31504 | 104422 | 
| 101 | 22 django/db/models/fields/json.py | 364 | 374| 131 | 31635 | 108523 | 
| 102 | 22 django/db/models/sql/compiler.py | 1208 | 1229| 223 | 31858 | 108523 | 
| 103 | 23 django/db/backends/postgresql/base.py | 65 | 132| 698 | 32556 | 111372 | 
| 104 | 23 django/contrib/postgres/search.py | 265 | 303| 248 | 32804 | 111372 | 
| 105 | 23 django/db/models/sql/query.py | 1375 | 1396| 250 | 33054 | 111372 | 
| 106 | 23 django/db/backends/sqlite3/operations.py | 320 | 365| 453 | 33507 | 111372 | 
| 107 | 23 django/db/models/functions/text.py | 119 | 141| 209 | 33716 | 111372 | 
| 108 | 24 django/db/backends/postgresql/features.py | 1 | 104| 844 | 34560 | 112216 | 
| 109 | 24 django/db/models/sql/compiler.py | 199 | 269| 580 | 35140 | 112216 | 
| 110 | 24 django/db/models/query.py | 916 | 966| 371 | 35511 | 112216 | 
| 111 | 24 django/db/models/sql/compiler.py | 271 | 363| 753 | 36264 | 112216 | 
| 112 | 25 django/db/models/fields/related_lookups.py | 62 | 101| 451 | 36715 | 113669 | 
| 113 | 25 django/contrib/postgres/constraints.py | 93 | 106| 179 | 36894 | 113669 | 
| 114 | 25 django/contrib/postgres/constraints.py | 1 | 67| 550 | 37444 | 113669 | 
| 115 | 26 django/db/backends/base/features.py | 1 | 112| 895 | 38339 | 116500 | 
| 116 | **26 django/db/models/expressions.py** | 284 | 309| 257 | 38596 | 116500 | 
| 117 | 26 django/db/models/fields/related_lookups.py | 121 | 157| 244 | 38840 | 116500 | 
| 118 | 27 django/db/backends/oracle/operations.py | 178 | 205| 319 | 39159 | 122471 | 
| 119 | 27 django/db/models/lookups.py | 371 | 403| 294 | 39453 | 122471 | 
| 120 | 28 django/db/models/fields/related.py | 997 | 1024| 215 | 39668 | 136347 | 
| 121 | 28 django/db/models/lookups.py | 303 | 315| 168 | 39836 | 136347 | 
| 122 | 28 django/db/models/query.py | 985 | 1005| 239 | 40075 | 136347 | 
| 123 | 28 django/db/models/functions/comparison.py | 95 | 112| 158 | 40233 | 136347 | 
| 124 | 28 django/contrib/postgres/search.py | 1 | 24| 205 | 40438 | 136347 | 
| 125 | 29 django/template/base.py | 705 | 724| 179 | 40617 | 144225 | 
| 126 | 30 django/db/backends/mysql/base.py | 98 | 168| 736 | 41353 | 147601 | 
| 127 | 30 django/db/models/sql/query.py | 2094 | 2116| 249 | 41602 | 147601 | 
| 128 | 30 django/contrib/postgres/search.py | 198 | 230| 243 | 41845 | 147601 | 
| 129 | 30 django/db/models/fields/json.py | 148 | 160| 125 | 41970 | 147601 | 
| 130 | 30 django/db/models/query.py | 1007 | 1020| 126 | 42096 | 147601 | 
| 131 | 30 django/db/models/functions/text.py | 291 | 324| 250 | 42346 | 147601 | 
| 132 | 31 django/db/models/deletion.py | 1 | 76| 566 | 42912 | 151429 | 
| 133 | 31 django/db/backends/mysql/operations.py | 291 | 319| 248 | 43160 | 151429 | 
| 134 | 31 django/db/models/sql/query.py | 233 | 295| 446 | 43606 | 151429 | 
| 135 | 31 django/db/models/sql/compiler.py | 700 | 722| 202 | 43808 | 151429 | 
| 136 | 32 django/db/backends/dummy/base.py | 50 | 74| 173 | 43981 | 151874 | 
| 137 | 32 django/db/models/sql/query.py | 2142 | 2159| 156 | 44137 | 151874 | 
| 138 | 32 django/db/backends/postgresql/base.py | 133 | 151| 177 | 44314 | 151874 | 
| 139 | **32 django/db/models/expressions.py** | 851 | 881| 233 | 44547 | 151874 | 
| 140 | 32 django/db/models/sql/compiler.py | 1423 | 1437| 127 | 44674 | 151874 | 
| 141 | 32 django/db/models/sql/query.py | 1676 | 1714| 356 | 45030 | 151874 | 
| 142 | 33 django/contrib/gis/db/models/functions.py | 462 | 490| 225 | 45255 | 155819 | 
| 143 | 33 django/db/backends/oracle/operations.py | 563 | 579| 290 | 45545 | 155819 | 
| 144 | 33 django/db/backends/sqlite3/operations.py | 267 | 282| 156 | 45701 | 155819 | 
| 145 | 33 django/db/models/fields/related_lookups.py | 1 | 23| 170 | 45871 | 155819 | 
| 146 | 34 django/db/models/functions/window.py | 1 | 25| 153 | 46024 | 156462 | 
| 147 | 34 django/db/models/functions/text.py | 1 | 39| 266 | 46290 | 156462 | 
| 148 | 35 django/contrib/postgres/lookups.py | 1 | 61| 337 | 46627 | 156799 | 
| 149 | 35 django/db/models/lookups.py | 318 | 368| 306 | 46933 | 156799 | 
| 150 | **35 django/db/models/expressions.py** | 1396 | 1432| 305 | 47238 | 156799 | 
| 151 | 36 django/contrib/postgres/indexes.py | 83 | 111| 311 | 47549 | 158595 | 
| 152 | 36 django/db/models/lookups.py | 62 | 87| 218 | 47767 | 158595 | 
| 153 | 36 django/db/models/sql/query.py | 1 | 66| 479 | 48246 | 158595 | 
| 154 | 36 django/db/models/sql/compiler.py | 442 | 495| 564 | 48810 | 158595 | 
| 155 | 37 django/db/backends/oracle/features.py | 1 | 97| 778 | 49588 | 159374 | 
| 156 | 37 django/db/models/sql/query.py | 1163 | 1206| 469 | 50057 | 159374 | 
| 157 | 37 django/db/models/sql/compiler.py | 365 | 405| 482 | 50539 | 159374 | 
| 158 | 37 django/db/models/fields/related.py | 1202 | 1233| 180 | 50719 | 159374 | 
| 159 | 37 django/db/models/sql/query.py | 933 | 951| 146 | 50865 | 159374 | 
| 160 | 37 django/template/base.py | 668 | 703| 272 | 51137 | 159374 | 
| 161 | 37 django/db/models/fields/related.py | 696 | 708| 116 | 51253 | 159374 | 
| 162 | 37 django/db/models/sql/where.py | 65 | 115| 396 | 51649 | 159374 | 
| 163 | 38 django/db/models/sql/subqueries.py | 1 | 44| 320 | 51969 | 160575 | 
| 164 | 38 django/db/models/lookups.py | 223 | 258| 304 | 52273 | 160575 | 
| 165 | 38 django/db/models/fields/related.py | 1235 | 1352| 963 | 53236 | 160575 | 
| 166 | 38 django/db/models/sql/compiler.py | 1268 | 1304| 341 | 53577 | 160575 | 
| 167 | 38 django/db/models/query.py | 1083 | 1096| 115 | 53692 | 160575 | 
| 168 | 38 django/db/models/sql/compiler.py | 1 | 19| 170 | 53862 | 160575 | 
| 169 | 38 django/db/models/fields/json.py | 194 | 212| 232 | 54094 | 160575 | 
| 170 | 38 django/contrib/gis/db/models/functions.py | 443 | 459| 180 | 54274 | 160575 | 
| 171 | 38 django/db/backends/postgresql/base.py | 298 | 341| 343 | 54617 | 160575 | 
| 172 | 38 django/contrib/gis/db/models/functions.py | 18 | 53| 312 | 54929 | 160575 | 
| 173 | 38 django/db/backends/oracle/operations.py | 207 | 255| 411 | 55340 | 160575 | 
| 174 | 39 django/db/models/base.py | 1891 | 1905| 136 | 55476 | 177248 | 
| 175 | 39 django/db/models/sql/compiler.py | 1047 | 1087| 337 | 55813 | 177248 | 
| 176 | 40 django/contrib/postgres/aggregates/__init__.py | 1 | 3| 0 | 55813 | 177268 | 
| 177 | 40 django/db/models/fields/related.py | 509 | 574| 492 | 56305 | 177268 | 
| 178 | 40 django/db/models/query.py | 401 | 444| 370 | 56675 | 177268 | 
| 179 | 40 django/db/models/sql/where.py | 117 | 140| 185 | 56860 | 177268 | 
| 180 | 40 django/db/models/sql/where.py | 195 | 209| 156 | 57016 | 177268 | 
| 181 | 40 django/db/models/sql/query.py | 817 | 843| 280 | 57296 | 177268 | 
| 182 | 40 django/db/models/lookups.py | 274 | 301| 236 | 57532 | 177268 | 
| 183 | 40 django/db/models/fields/related.py | 841 | 862| 169 | 57701 | 177268 | 
| 184 | **40 django/db/models/expressions.py** | 1361 | 1394| 276 | 57977 | 177268 | 
| 185 | 40 django/contrib/postgres/search.py | 95 | 127| 277 | 58254 | 177268 | 
| 186 | 40 django/db/models/sql/query.py | 563 | 637| 809 | 59063 | 177268 | 
| 187 | 41 django/db/models/sql/datastructures.py | 117 | 148| 170 | 59233 | 178736 | 
| 188 | 41 django/db/models/deletion.py | 346 | 359| 116 | 59349 | 178736 | 
| 189 | 42 django/db/models/fields/__init__.py | 1 | 81| 633 | 59982 | 197182 | 
| 190 | 42 django/db/models/sql/query.py | 1949 | 1993| 364 | 60346 | 197182 | 
| 191 | 43 django/contrib/postgres/fields/ranges.py | 231 | 321| 479 | 60825 | 199274 | 
| 192 | 43 django/db/models/sql/subqueries.py | 137 | 163| 161 | 60986 | 199274 | 
| 193 | 43 django/contrib/postgres/indexes.py | 114 | 131| 133 | 61119 | 199274 | 
| 194 | 43 django/db/backends/mysql/base.py | 169 | 193| 199 | 61318 | 199274 | 
| 195 | 43 django/db/models/fields/related.py | 750 | 768| 222 | 61540 | 199274 | 
| 196 | 43 django/db/models/sql/query.py | 783 | 815| 392 | 61932 | 199274 | 
| 197 | 43 django/db/models/fields/related_lookups.py | 104 | 119| 215 | 62147 | 199274 | 


### Hint

```
For anyone else who ends up here you can Subquery your way out of this, although I've no idea what the performance implications of that are.
I did some further digging and this only occurs on 3.1, it works fine on 2.0, 2.1, 2.2 and 3.0
I think this is related to https://code.djangoproject.com/ticket/32007
To test this, given the implied model above, you can create 3 Fred objects, 2 with one value for bob_id and the third with a different value. When you do the select on that you should see [{"bob_id__is_null": False, "id_count": 3}] But instead you will get [{"bob_id__is_null": False, "id_count": 1}, {"bob_id__is_null": False, "id_count": 2}]
Regression in df32fd42b84cc6dbba173201f244491b0d154a63 (backported in fdd2b01e8e12857aad2219a46a41bd9051ec8f8d). Reproduced at 4cce1d13cfe9d8e56921c5fa8c61e3034dc8e20c.
```

## Patch

```diff
diff --git a/django/db/models/expressions.py b/django/db/models/expressions.py
--- a/django/db/models/expressions.py
+++ b/django/db/models/expressions.py
@@ -915,9 +915,13 @@ def get_source_expressions(self):
         return [self.expression]
 
     def get_group_by_cols(self, alias=None):
-        expression = self.expression.copy()
-        expression.output_field = self.output_field
-        return expression.get_group_by_cols(alias=alias)
+        if isinstance(self.expression, Expression):
+            expression = self.expression.copy()
+            expression.output_field = self.output_field
+            return expression.get_group_by_cols(alias=alias)
+        # For non-expressions e.g. an SQL WHERE clause, the entire
+        # `expression` must be included in the GROUP BY clause.
+        return super().get_group_by_cols()
 
     def as_sql(self, compiler, connection):
         return compiler.compile(self.expression)

```

## Test Patch

```diff
diff --git a/tests/annotations/tests.py b/tests/annotations/tests.py
--- a/tests/annotations/tests.py
+++ b/tests/annotations/tests.py
@@ -195,6 +195,18 @@ def test_q_expression_annotation_with_aggregation(self):
         self.assertEqual(book.isnull_pubdate, False)
         self.assertEqual(book.rating_count, 1)
 
+    @skipUnlessDBFeature('supports_boolean_expr_in_select_clause')
+    def test_grouping_by_q_expression_annotation(self):
+        authors = Author.objects.annotate(
+            under_40=ExpressionWrapper(Q(age__lt=40), output_field=BooleanField()),
+        ).values('under_40').annotate(
+            count_id=Count('id'),
+        ).values('under_40', 'count_id')
+        self.assertCountEqual(authors, [
+            {'under_40': False, 'count_id': 3},
+            {'under_40': True, 'count_id': 6},
+        ])
+
     def test_aggregate_over_annotation(self):
         agg = Author.objects.annotate(other_age=F('age')).aggregate(otherage_sum=Sum('other_age'))
         other_agg = Author.objects.aggregate(age_sum=Sum('age'))

```


## Code snippets

### 1 - django/db/models/aggregates.py:

Start line: 45, End line: 68

```python
class Aggregate(Func):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # Aggregates are not allowed in UPDATE queries, so ignore for_save
        c = super().resolve_expression(query, allow_joins, reuse, summarize)
        c.filter = c.filter and c.filter.resolve_expression(query, allow_joins, reuse, summarize)
        if not summarize:
            # Call Aggregate.get_source_expressions() to avoid
            # returning self.filter and including that in this loop.
            expressions = super(Aggregate, c).get_source_expressions()
            for index, expr in enumerate(expressions):
                if expr.contains_aggregate:
                    before_resolved = self.get_source_expressions()[index]
                    name = before_resolved.name if hasattr(before_resolved, 'name') else repr(before_resolved)
                    raise FieldError("Cannot compute %s('%s'): '%s' is an aggregate" % (c.name, name, name))
        return c

    @property
    def default_alias(self):
        expressions = self.get_source_expressions()
        if len(expressions) == 1 and hasattr(expressions[0], 'name'):
            return '%s__%s' % (expressions[0].name, self.name.lower())
        raise TypeError("Complex expressions require an alias")

    def get_group_by_cols(self, alias=None):
        return []
```
### 2 - django/db/models/expressions.py:

Start line: 929, End line: 993

```python
class When(Expression):
    template = 'WHEN %(condition)s THEN %(result)s'
    # This isn't a complete conditional expression, must be used in Case().
    conditional = False

    def __init__(self, condition=None, then=None, **lookups):
        if lookups:
            if condition is None:
                condition, lookups = Q(**lookups), None
            elif getattr(condition, 'conditional', False):
                condition, lookups = Q(condition, **lookups), None
        if condition is None or not getattr(condition, 'conditional', False) or lookups:
            raise TypeError(
                'When() supports a Q object, a boolean expression, or lookups '
                'as a condition.'
            )
        if isinstance(condition, Q) and not condition:
            raise ValueError("An empty Q() can't be used as a When() condition.")
        super().__init__(output_field=None)
        self.condition = condition
        self.result = self._parse_expressions(then)[0]

    def __str__(self):
        return "WHEN %r THEN %r" % (self.condition, self.result)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    def get_source_expressions(self):
        return [self.condition, self.result]

    def set_source_expressions(self, exprs):
        self.condition, self.result = exprs

    def get_source_fields(self):
        # We're only interested in the fields of the result expressions.
        return [self.result._output_field_or_none]

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        if hasattr(c.condition, 'resolve_expression'):
            c.condition = c.condition.resolve_expression(query, allow_joins, reuse, summarize, False)
        c.result = c.result.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c

    def as_sql(self, compiler, connection, template=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = extra_context
        sql_params = []
        condition_sql, condition_params = compiler.compile(self.condition)
        template_params['condition'] = condition_sql
        sql_params.extend(condition_params)
        result_sql, result_params = compiler.compile(self.result)
        template_params['result'] = result_sql
        sql_params.extend(result_params)
        template = template or self.template
        return template % template_params, sql_params

    def get_group_by_cols(self, alias=None):
        # This is not a complete expression and cannot be used in GROUP BY.
        cols = []
        for source in self.get_source_expressions():
            cols.extend(source.get_group_by_cols())
        return cols
```
### 3 - django/db/models/expressions.py:

Start line: 814, End line: 848

```python
class Col(Expression):

    contains_column_references = True
    possibly_multivalued = False

    def __init__(self, alias, target, output_field=None):
        if output_field is None:
            output_field = target
        super().__init__(output_field=output_field)
        self.alias, self.target = alias, target

    def __repr__(self):
        alias, target = self.alias, self.target
        identifiers = (alias, str(target)) if alias else (str(target),)
        return '{}({})'.format(self.__class__.__name__, ', '.join(identifiers))

    def as_sql(self, compiler, connection):
        alias, column = self.alias, self.target.column
        identifiers = (alias, column) if alias else (column,)
        sql = '.'.join(map(compiler.quote_name_unless_alias, identifiers))
        return sql, []

    def relabeled_clone(self, relabels):
        if self.alias is None:
            return self
        return self.__class__(relabels.get(self.alias, self.alias), self.target, self.output_field)

    def get_group_by_cols(self, alias=None):
        return [self]

    def get_db_converters(self, connection):
        if self.target == self.output_field:
            return self.output_field.get_db_converters(connection)
        return (self.output_field.get_db_converters(connection) +
                self.target.get_db_converters(connection))
```
### 4 - django/db/models/expressions.py:

Start line: 746, End line: 776

```python
class Value(Expression):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c.for_save = for_save
        return c

    def get_group_by_cols(self, alias=None):
        return []

    def _resolve_output_field(self):
        if isinstance(self.value, str):
            return fields.CharField()
        if isinstance(self.value, bool):
            return fields.BooleanField()
        if isinstance(self.value, int):
            return fields.IntegerField()
        if isinstance(self.value, float):
            return fields.FloatField()
        if isinstance(self.value, datetime.datetime):
            return fields.DateTimeField()
        if isinstance(self.value, datetime.date):
            return fields.DateField()
        if isinstance(self.value, datetime.time):
            return fields.TimeField()
        if isinstance(self.value, datetime.timedelta):
            return fields.DurationField()
        if isinstance(self.value, Decimal):
            return fields.DecimalField()
        if isinstance(self.value, bytes):
            return fields.BinaryField()
        if isinstance(self.value, UUID):
            return fields.UUIDField()
```
### 5 - django/db/models/expressions.py:

Start line: 1139, End line: 1171

```python
class Exists(Subquery):
    template = 'EXISTS(%(subquery)s)'
    output_field = fields.BooleanField()

    def __init__(self, queryset, negated=False, **kwargs):
        self.negated = negated
        super().__init__(queryset, **kwargs)

    def __invert__(self):
        clone = self.copy()
        clone.negated = not self.negated
        return clone

    def as_sql(self, compiler, connection, template=None, **extra_context):
        query = self.query.exists(using=connection.alias)
        sql, params = super().as_sql(
            compiler,
            connection,
            template=template,
            query=query,
            **extra_context,
        )
        if self.negated:
            sql = 'NOT {}'.format(sql)
        return sql, params

    def select_format(self, compiler, sql, params):
        # Wrap EXISTS() with a CASE WHEN expression if a database backend
        # (e.g. Oracle) doesn't support boolean expression in SELECT or GROUP
        # BY list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = 'CASE WHEN {} THEN 1 ELSE 0 END'.format(sql)
        return sql, params
```
### 6 - django/db/models/expressions.py:

Start line: 728, End line: 744

```python
class Value(Expression):

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        val = self.value
        output_field = self._output_field_or_none
        if output_field is not None:
            if self.for_save:
                val = output_field.get_db_prep_save(val, connection=connection)
            else:
                val = output_field.get_db_prep_value(val, connection=connection)
            if hasattr(output_field, 'get_placeholder'):
                return output_field.get_placeholder(val, compiler, connection), [val]
        if val is None:
            # cx_Oracle does not always convert None to the appropriate
            # NULL type (like in case expressions using numbers), so we
            # use a literal SQL NULL
            return 'NULL', []
        return '%s', [val]
```
### 7 - django/db/models/expressions.py:

Start line: 901, End line: 926

```python
class ExpressionWrapper(Expression):
    """
    An expression that can wrap another expression so that it can provide
    extra context to the inner expression, such as the output_field.
    """

    def __init__(self, expression, output_field):
        super().__init__(output_field=output_field)
        self.expression = expression

    def set_source_expressions(self, exprs):
        self.expression = exprs[0]

    def get_source_expressions(self):
        return [self.expression]

    def get_group_by_cols(self, alias=None):
        expression = self.expression.copy()
        expression.output_field = self.output_field
        return expression.get_group_by_cols(alias=alias)

    def as_sql(self, compiler, connection):
        return compiler.compile(self.expression)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.expression)
```
### 8 - django/db/models/sql/compiler.py:

Start line: 149, End line: 197

```python
class SQLCompiler:
    ordering_parts =
    # ... other code

    def collapse_group_by(self, expressions, having):
        # If the DB can group by primary key, then group by the primary key of
        # query's main model. Note that for PostgreSQL the GROUP BY clause must
        # include the primary key of every table, but for MySQL it is enough to
        # have the main table's primary key.
        if self.connection.features.allows_group_by_pk:
            # Determine if the main model's primary key is in the query.
            pk = None
            for expr in expressions:
                # Is this a reference to query's base table primary key? If the
                # expression isn't a Col-like, then skip the expression.
                if (getattr(expr, 'target', None) == self.query.model._meta.pk and
                        getattr(expr, 'alias', None) == self.query.base_table):
                    pk = expr
                    break
            # If the main model's primary key is in the query, group by that
            # field, HAVING expressions, and expressions associated with tables
            # that don't have a primary key included in the grouped columns.
            if pk:
                pk_aliases = {
                    expr.alias for expr in expressions
                    if hasattr(expr, 'target') and expr.target.primary_key
                }
                expressions = [pk] + [
                    expr for expr in expressions
                    if expr in having or (
                        getattr(expr, 'alias', None) is not None and expr.alias not in pk_aliases
                    )
                ]
        elif self.connection.features.allows_group_by_selected_pks:
            # Filter out all expressions associated with a table's primary key
            # present in the grouped columns. This is done by identifying all
            # tables that have their primary key included in the grouped
            # columns and removing non-primary key columns referring to them.
            # Unmanaged models are excluded because they could be representing
            # database views on which the optimization might not be allowed.
            pks = {
                expr for expr in expressions
                if (
                    hasattr(expr, 'target') and
                    expr.target.primary_key and
                    self.connection.features.allows_group_by_selected_pks_on_model(expr.target.model)
                )
            }
            aliases = {expr.alias for expr in pks}
            expressions = [
                expr for expr in expressions if expr in pks or getattr(expr, 'alias', None) not in aliases
            ]
        return expressions
    # ... other code
```
### 9 - django/db/models/expressions.py:

Start line: 332, End line: 387

```python
@deconstructible
class BaseExpression:

    def get_lookup(self, lookup):
        return self.output_field.get_lookup(lookup)

    def get_transform(self, name):
        return self.output_field.get_transform(name)

    def relabeled_clone(self, change_map):
        clone = self.copy()
        clone.set_source_expressions([
            e.relabeled_clone(change_map) if e is not None else None
            for e in self.get_source_expressions()
        ])
        return clone

    def copy(self):
        return copy.copy(self)

    def get_group_by_cols(self, alias=None):
        if not self.contains_aggregate:
            return [self]
        cols = []
        for source in self.get_source_expressions():
            cols.extend(source.get_group_by_cols())
        return cols

    def get_source_fields(self):
        """Return the underlying field types used by this aggregate."""
        return [e._output_field_or_none for e in self.get_source_expressions()]

    def asc(self, **kwargs):
        return OrderBy(self, **kwargs)

    def desc(self, **kwargs):
        return OrderBy(self, descending=True, **kwargs)

    def reverse_ordering(self):
        return self

    def flatten(self):
        """
        Recursively yield this expression and all subexpressions, in
        depth-first order.
        """
        yield self
        for expr in self.get_source_expressions():
            if expr:
                yield from expr.flatten()

    def select_format(self, compiler, sql, params):
        """
        Custom format for select clauses. For example, EXISTS expressions need
        to be wrapped in CASE WHEN on Oracle.
        """
        if hasattr(self.output_field, 'select_format'):
            return self.output_field.select_format(compiler, sql, params)
        return sql, params
```
### 10 - django/db/models/aggregates.py:

Start line: 70, End line: 96

```python
class Aggregate(Func):

    def as_sql(self, compiler, connection, **extra_context):
        extra_context['distinct'] = 'DISTINCT ' if self.distinct else ''
        if self.filter:
            if connection.features.supports_aggregate_filter_clause:
                filter_sql, filter_params = self.filter.as_sql(compiler, connection)
                template = self.filter_template % extra_context.get('template', self.template)
                sql, params = super().as_sql(
                    compiler, connection, template=template, filter=filter_sql,
                    **extra_context
                )
                return sql, params + filter_params
            else:
                copy = self.copy()
                copy.filter = None
                source_expressions = copy.get_source_expressions()
                condition = When(self.filter, then=source_expressions[0])
                copy.set_source_expressions([Case(condition)] + source_expressions[1:])
                return super(Aggregate, copy).as_sql(compiler, connection, **extra_context)
        return super().as_sql(compiler, connection, **extra_context)

    def _get_repr_options(self):
        options = super()._get_repr_options()
        if self.distinct:
            options['distinct'] = self.distinct
        if self.filter:
            options['filter'] = self.filter
        return options
```
### 11 - django/db/models/expressions.py:

Start line: 1198, End line: 1223

```python
class OrderBy(BaseExpression):

    def as_sql(self, compiler, connection, template=None, **extra_context):
        template = template or self.template
        if connection.features.supports_order_by_nulls_modifier:
            if self.nulls_last:
                template = '%s NULLS LAST' % template
            elif self.nulls_first:
                template = '%s NULLS FIRST' % template
        else:
            if self.nulls_last and not (
                self.descending and connection.features.order_by_nulls_first
            ):
                template = '%%(expression)s IS NULL, %s' % template
            elif self.nulls_first and not (
                not self.descending and connection.features.order_by_nulls_first
            ):
                template = '%%(expression)s IS NOT NULL, %s' % template
        connection.ops.check_expression_support(self)
        expression_sql, params = compiler.compile(self.expression)
        placeholders = {
            'expression': expression_sql,
            'ordering': 'DESC' if self.descending else 'ASC',
            **extra_context,
        }
        template = template or self.template
        params *= template.count('%(expression)s')
        return (template % placeholders).rstrip(), params
```
### 12 - django/db/models/expressions.py:

Start line: 1044, End line: 1074

```python
class Case(Expression):

    def as_sql(self, compiler, connection, template=None, case_joiner=None, **extra_context):
        connection.ops.check_expression_support(self)
        if not self.cases:
            return compiler.compile(self.default)
        template_params = {**self.extra, **extra_context}
        case_parts = []
        sql_params = []
        for case in self.cases:
            try:
                case_sql, case_params = compiler.compile(case)
            except EmptyResultSet:
                continue
            case_parts.append(case_sql)
            sql_params.extend(case_params)
        default_sql, default_params = compiler.compile(self.default)
        if not case_parts:
            return default_sql, default_params
        case_joiner = case_joiner or self.case_joiner
        template_params['cases'] = case_joiner.join(case_parts)
        template_params['default'] = default_sql
        sql_params.extend(default_params)
        template = template or template_params.get('template', self.template)
        sql = template % template_params
        if self._output_field_or_none is not None:
            sql = connection.ops.unification_cast_sql(self.output_field) % sql
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        if not self.cases:
            return self.default.get_group_by_cols(alias)
        return super().get_group_by_cols(alias)
```
### 14 - django/db/models/expressions.py:

Start line: 1336, End line: 1358

```python
class Window(SQLiteNumericMixin, Expression):

    def as_sqlite(self, compiler, connection):
        if isinstance(self.output_field, fields.DecimalField):
            # Casting to numeric must be outside of the window expression.
            copy = self.copy()
            source_expressions = copy.get_source_expressions()
            source_expressions[0].output_field = fields.FloatField()
            copy.set_source_expressions(source_expressions)
            return super(Window, copy).as_sqlite(compiler, connection)
        return self.as_sql(compiler, connection)

    def __str__(self):
        return '{} OVER ({}{}{})'.format(
            str(self.source_expression),
            'PARTITION BY ' + str(self.partition_by) if self.partition_by else '',
            'ORDER BY ' + str(self.order_by) if self.order_by else '',
            str(self.frame or ''),
        )

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def get_group_by_cols(self, alias=None):
        return []
```
### 15 - django/db/models/expressions.py:

Start line: 1257, End line: 1300

```python
class Window(SQLiteNumericMixin, Expression):
    template = '%(expression)s OVER (%(window)s)'
    # Although the main expression may either be an aggregate or an
    # expression with an aggregate function, the GROUP BY that will
    # be introduced in the query as a result is not desired.
    contains_aggregate = False
    contains_over_clause = True
    filterable = False

    def __init__(self, expression, partition_by=None, order_by=None, frame=None, output_field=None):
        self.partition_by = partition_by
        self.order_by = order_by
        self.frame = frame

        if not getattr(expression, 'window_compatible', False):
            raise ValueError(
                "Expression '%s' isn't compatible with OVER clauses." %
                expression.__class__.__name__
            )

        if self.partition_by is not None:
            if not isinstance(self.partition_by, (tuple, list)):
                self.partition_by = (self.partition_by,)
            self.partition_by = ExpressionList(*self.partition_by)

        if self.order_by is not None:
            if isinstance(self.order_by, (list, tuple)):
                self.order_by = ExpressionList(*self.order_by)
            elif not isinstance(self.order_by, BaseExpression):
                raise ValueError(
                    'order_by must be either an Expression or a sequence of '
                    'expressions.'
                )
        super().__init__(output_field=output_field)
        self.source_expression = self._parse_expressions(expression)[0]

    def _resolve_output_field(self):
        return self.source_expression.output_field

    def get_source_expressions(self):
        return [self.source_expression, self.partition_by, self.order_by, self.frame]

    def set_source_expressions(self, exprs):
        self.source_expression, self.partition_by, self.order_by, self.frame = exprs
```
### 17 - django/db/models/expressions.py:

Start line: 1077, End line: 1136

```python
class Subquery(Expression):
    """
    An explicit subquery. It may contain OuterRef() references to the outer
    query which will be resolved when it is applied to that query.
    """
    template = '(%(subquery)s)'
    contains_aggregate = False

    def __init__(self, queryset, output_field=None, **extra):
        # Allow the usage of both QuerySet and sql.Query objects.
        self.query = getattr(queryset, 'query', queryset)
        self.extra = extra
        super().__init__(output_field)

    def __getstate__(self):
        state = super().__getstate__()
        args, kwargs = state['_constructor_args']
        if args:
            args = (self.query, *args[1:])
        else:
            kwargs['queryset'] = self.query
        state['_constructor_args'] = args, kwargs
        return state

    def get_source_expressions(self):
        return [self.query]

    def set_source_expressions(self, exprs):
        self.query = exprs[0]

    def _resolve_output_field(self):
        return self.query.output_field

    def copy(self):
        clone = super().copy()
        clone.query = clone.query.clone()
        return clone

    @property
    def external_aliases(self):
        return self.query.external_aliases

    def as_sql(self, compiler, connection, template=None, query=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = {**self.extra, **extra_context}
        query = query or self.query
        subquery_sql, sql_params = query.as_sql(compiler, connection)
        template_params['subquery'] = subquery_sql[1:-1]

        template = template or template_params.get('template', self.template)
        sql = template % template_params
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        if alias:
            return [Ref(alias, self)]
        external_cols = self.query.get_external_cols()
        if any(col.possibly_multivalued for col in external_cols):
            return [self]
        return external_cols
```
### 19 - django/db/models/expressions.py:

Start line: 779, End line: 793

```python
class RawSQL(Expression):
    def __init__(self, sql, params, output_field=None):
        if output_field is None:
            output_field = fields.Field()
        self.sql, self.params = sql, params
        super().__init__(output_field=output_field)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.sql, self.params)

    def as_sql(self, compiler, connection):
        return '(%s)' % self.sql, self.params

    def get_group_by_cols(self, alias=None):
        return [self]
```
### 20 - django/db/models/expressions.py:

Start line: 1174, End line: 1196

```python
class OrderBy(BaseExpression):
    template = '%(expression)s %(ordering)s'
    conditional = False

    def __init__(self, expression, descending=False, nulls_first=False, nulls_last=False):
        if nulls_first and nulls_last:
            raise ValueError('nulls_first and nulls_last are mutually exclusive')
        self.nulls_first = nulls_first
        self.nulls_last = nulls_last
        self.descending = descending
        if not hasattr(expression, 'resolve_expression'):
            raise ValueError('expression must be an expression type')
        self.expression = expression

    def __repr__(self):
        return "{}({}, descending={})".format(
            self.__class__.__name__, self.expression, self.descending)

    def set_source_expressions(self, exprs):
        self.expression = exprs[0]

    def get_source_expressions(self):
        return [self.expression]
```
### 22 - django/db/models/expressions.py:

Start line: 1225, End line: 1254

```python
class OrderBy(BaseExpression):

    def as_oracle(self, compiler, connection):
        # Oracle doesn't allow ORDER BY EXISTS() unless it's wrapped in
        # a CASE WHEN.
        if isinstance(self.expression, Exists):
            copy = self.copy()
            copy.expression = Case(
                When(self.expression, then=True),
                default=False,
            )
            return copy.as_sql(compiler, connection)
        return self.as_sql(compiler, connection)

    def get_group_by_cols(self, alias=None):
        cols = []
        for source in self.get_source_expressions():
            cols.extend(source.get_group_by_cols())
        return cols

    def reverse_ordering(self):
        self.descending = not self.descending
        if self.nulls_first or self.nulls_last:
            self.nulls_first = not self.nulls_first
            self.nulls_last = not self.nulls_last
        return self

    def asc(self):
        self.descending = False

    def desc(self):
        self.descending = True
```
### 25 - django/db/models/expressions.py:

Start line: 489, End line: 514

```python
class CombinedExpression(SQLiteNumericMixin, Expression):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        lhs = self.lhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        rhs = self.rhs.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        if not isinstance(self, (DurationExpression, TemporalSubtraction)):
            try:
                lhs_type = lhs.output_field.get_internal_type()
            except (AttributeError, FieldError):
                lhs_type = None
            try:
                rhs_type = rhs.output_field.get_internal_type()
            except (AttributeError, FieldError):
                rhs_type = None
            if 'DurationField' in {lhs_type, rhs_type} and lhs_type != rhs_type:
                return DurationExpression(self.lhs, self.connector, self.rhs).resolve_expression(
                    query, allow_joins, reuse, summarize, for_save,
                )
            datetime_fields = {'DateField', 'DateTimeField', 'TimeField'}
            if self.connector == self.SUB and lhs_type in datetime_fields and lhs_type == rhs_type:
                return TemporalSubtraction(self.lhs, self.rhs).resolve_expression(
                    query, allow_joins, reuse, summarize, for_save,
                )
        c = self.copy()
        c.is_summary = summarize
        c.lhs = lhs
        c.rhs = rhs
        return c
```
### 27 - django/db/models/expressions.py:

Start line: 219, End line: 253

```python
@deconstructible
class BaseExpression:

    @cached_property
    def contains_aggregate(self):
        return any(expr and expr.contains_aggregate for expr in self.get_source_expressions())

    @cached_property
    def contains_over_clause(self):
        return any(expr and expr.contains_over_clause for expr in self.get_source_expressions())

    @cached_property
    def contains_column_references(self):
        return any(expr and expr.contains_column_references for expr in self.get_source_expressions())

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        """
        Provide the chance to do any preprocessing or validation before being
        added to the query.

        Arguments:
         * query: the backend query implementation
         * allow_joins: boolean allowing or denying use of joins
           in this query
         * reuse: a set of reusable joins for multijoins
         * summarize: a terminal aggregate clause
         * for_save: whether this expression about to be used in a save or update

        Return: an Expression to be added to the query.
        """
        c = self.copy()
        c.is_summary = summarize
        c.set_source_expressions([
            expr.resolve_expression(query, allow_joins, reuse, summarize)
            if expr else None
            for expr in c.get_source_expressions()
        ])
        return c
```
### 32 - django/db/models/expressions.py:

Start line: 795, End line: 811

```python
class RawSQL(Expression):

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # Resolve parents fields used in raw SQL.
        for parent in query.model._meta.get_parent_list():
            for parent_field in parent._meta.local_fields:
                _, column_name = parent_field.get_attname_column()
                if column_name.lower() in self.sql.lower():
                    query.resolve_ref(parent_field.name, allow_joins, reuse, summarize)
                    break
        return super().resolve_expression(query, allow_joins, reuse, summarize, for_save)


class Star(Expression):
    def __repr__(self):
        return "'*'"

    def as_sql(self, compiler, connection):
        return '*', []
```
### 33 - django/db/models/expressions.py:

Start line: 996, End line: 1042

```python
class Case(Expression):
    """
    An SQL searched CASE expression:

        CASE
            WHEN n > 0
                THEN 'positive'
            WHEN n < 0
                THEN 'negative'
            ELSE 'zero'
        END
    """
    template = 'CASE %(cases)s ELSE %(default)s END'
    case_joiner = ' '

    def __init__(self, *cases, default=None, output_field=None, **extra):
        if not all(isinstance(case, When) for case in cases):
            raise TypeError("Positional arguments must all be When objects.")
        super().__init__(output_field)
        self.cases = list(cases)
        self.default = self._parse_expressions(default)[0]
        self.extra = extra

    def __str__(self):
        return "CASE %s, ELSE %r" % (', '.join(str(c) for c in self.cases), self.default)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self)

    def get_source_expressions(self):
        return self.cases + [self.default]

    def set_source_expressions(self, exprs):
        *self.cases, self.default = exprs

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        for pos, case in enumerate(c.cases):
            c.cases[pos] = case.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        c.default = c.default.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c

    def copy(self):
        c = super().copy()
        c.cases = c.cases[:]
        return c
```
### 34 - django/db/models/expressions.py:

Start line: 1302, End line: 1334

```python
class Window(SQLiteNumericMixin, Expression):

    def as_sql(self, compiler, connection, template=None):
        connection.ops.check_expression_support(self)
        if not connection.features.supports_over_clause:
            raise NotSupportedError('This backend does not support window expressions.')
        expr_sql, params = compiler.compile(self.source_expression)
        window_sql, window_params = [], []

        if self.partition_by is not None:
            sql_expr, sql_params = self.partition_by.as_sql(
                compiler=compiler, connection=connection,
                template='PARTITION BY %(expressions)s',
            )
            window_sql.extend(sql_expr)
            window_params.extend(sql_params)

        if self.order_by is not None:
            window_sql.append(' ORDER BY ')
            order_sql, order_params = compiler.compile(self.order_by)
            window_sql.extend(order_sql)
            window_params.extend(order_params)

        if self.frame:
            frame_sql, frame_params = compiler.compile(self.frame)
            window_sql.append(' ' + frame_sql)
            window_params.extend(frame_params)

        params.extend(window_params)
        template = template or self.template

        return template % {
            'expression': expr_sql,
            'window': ''.join(window_sql).strip()
        }, params
```
### 35 - django/db/models/expressions.py:

Start line: 442, End line: 487

```python
class CombinedExpression(SQLiteNumericMixin, Expression):

    def __init__(self, lhs, connector, rhs, output_field=None):
        super().__init__(output_field=output_field)
        self.connector = connector
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self)

    def __str__(self):
        return "{} {} {}".format(self.lhs, self.connector, self.rhs)

    def get_source_expressions(self):
        return [self.lhs, self.rhs]

    def set_source_expressions(self, exprs):
        self.lhs, self.rhs = exprs

    def _resolve_output_field(self):
        try:
            return super()._resolve_output_field()
        except FieldError:
            combined_type = _resolve_combined_type(
                self.connector,
                type(self.lhs.output_field),
                type(self.rhs.output_field),
            )
            if combined_type is None:
                raise
            return combined_type()

    def as_sql(self, compiler, connection):
        expressions = []
        expression_params = []
        sql, params = compiler.compile(self.lhs)
        expressions.append(sql)
        expression_params.extend(params)
        sql, params = compiler.compile(self.rhs)
        expressions.append(sql)
        expression_params.extend(params)
        # order of precedence
        expression_wrapper = '(%s)'
        sql = connection.ops.combine_expression(self.connector, expressions)
        return expression_wrapper % sql, expression_params
```
### 38 - django/db/models/expressions.py:

Start line: 389, End line: 414

```python
@deconstructible
class BaseExpression:

    @cached_property
    def identity(self):
        constructor_signature = inspect.signature(self.__init__)
        args, kwargs = self._constructor_args
        signature = constructor_signature.bind_partial(*args, **kwargs)
        signature.apply_defaults()
        arguments = signature.arguments.items()
        identity = [self.__class__]
        for arg, value in arguments:
            if isinstance(value, fields.Field):
                if value.name and value.model:
                    value = (value.model._meta.label, value.name)
                else:
                    value = type(value)
            else:
                value = make_hashable(value)
            identity.append((arg, value))
        return tuple(identity)

    def __eq__(self, other):
        if not isinstance(other, BaseExpression):
            return NotImplemented
        return other.identity == self.identity

    def __hash__(self):
        return hash(self.identity)
```
### 40 - django/db/models/expressions.py:

Start line: 311, End line: 330

```python
@deconstructible
class BaseExpression:

    @staticmethod
    def _convert_value_noop(value, expression, connection):
        return value

    @cached_property
    def convert_value(self):
        """
        Expressions provide their own converters because users have the option
        of manually specifying the output_field which may be a different type
        from the one the database returns.
        """
        field = self.output_field
        internal_type = field.get_internal_type()
        if internal_type == 'FloatField':
            return lambda value, expression, connection: None if value is None else float(value)
        elif internal_type.endswith('IntegerField'):
            return lambda value, expression, connection: None if value is None else int(value)
        elif internal_type == 'DecimalField':
            return lambda value, expression, connection: None if value is None else Decimal(value)
        return self._convert_value_noop
```
### 42 - django/db/models/expressions.py:

Start line: 591, End line: 630

```python
class ResolvedOuterRef(F):
    """
    An object that contains a reference to an outer query.

    In this case, the reference to the outer query has been resolved because
    the inner query has been used as a subquery.
    """
    contains_aggregate = False

    def as_sql(self, *args, **kwargs):
        raise ValueError(
            'This queryset contains a reference to an outer query and may '
            'only be used in a subquery.'
        )

    def resolve_expression(self, *args, **kwargs):
        col = super().resolve_expression(*args, **kwargs)
        # FIXME: Rename possibly_multivalued to multivalued and fix detection
        # for non-multivalued JOINs (e.g. foreign key fields). This should take
        # into account only many-to-many and one-to-many relationships.
        col.possibly_multivalued = LOOKUP_SEP in self.name
        return col

    def relabeled_clone(self, relabels):
        return self

    def get_group_by_cols(self, alias=None):
        return []


class OuterRef(F):
    contains_aggregate = False

    def resolve_expression(self, *args, **kwargs):
        if isinstance(self.name, self.__class__):
            return self.name
        return ResolvedOuterRef(self.name)

    def relabeled_clone(self, relabels):
        return self
```
### 44 - django/db/models/expressions.py:

Start line: 1, End line: 30

```python
import copy
import datetime
import functools
import inspect
from decimal import Decimal
from uuid import UUID

from django.core.exceptions import EmptyResultSet, FieldError
from django.db import NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable


class SQLiteNumericMixin:
    """
    Some expressions with output_field=DecimalField() must be cast to
    numeric to be properly filtered.
    """
    def as_sqlite(self, compiler, connection, **extra_context):
        sql, params = self.as_sql(compiler, connection, **extra_context)
        try:
            if self.output_field.get_internal_type() == 'DecimalField':
                sql = 'CAST(%s AS NUMERIC)' % sql
        except FieldError:
            pass
        return sql, params
```
### 47 - django/db/models/expressions.py:

Start line: 150, End line: 189

```python
@deconstructible
class BaseExpression:
    """Base class for all query expressions."""

    # aggregate specific fields
    is_summary = False
    _output_field_resolved_to_none = False
    # Can the expression be used in a WHERE clause?
    filterable = True
    # Can the expression can be used as a source expression in Window?
    window_compatible = False

    def __init__(self, output_field=None):
        if output_field is not None:
            self.output_field = output_field

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('convert_value', None)
        return state

    def get_db_converters(self, connection):
        return (
            []
            if self.convert_value is self._convert_value_noop else
            [self.convert_value]
        ) + self.output_field.get_db_converters(connection)

    def get_source_expressions(self):
        return []

    def set_source_expressions(self, exprs):
        assert not exprs

    def _parse_expressions(self, *expressions):
        return [
            arg if hasattr(arg, 'resolve_expression') else (
                F(arg) if isinstance(arg, str) else Value(arg)
            ) for arg in expressions
        ]
```
### 53 - django/db/models/expressions.py:

Start line: 679, End line: 704

```python
class Func(SQLiteNumericMixin, Expression):

    def as_sql(self, compiler, connection, function=None, template=None, arg_joiner=None, **extra_context):
        connection.ops.check_expression_support(self)
        sql_parts = []
        params = []
        for arg in self.source_expressions:
            arg_sql, arg_params = compiler.compile(arg)
            sql_parts.append(arg_sql)
            params.extend(arg_params)
        data = {**self.extra, **extra_context}
        # Use the first supplied value in this order: the parameter to this
        # method, a value supplied in __init__()'s **extra (the value in
        # `data`), or the value defined on the class.
        if function is not None:
            data['function'] = function
        else:
            data.setdefault('function', self.function)
        template = template or data.get('template', self.template)
        arg_joiner = arg_joiner or data.get('arg_joiner', self.arg_joiner)
        data['expressions'] = data['field'] = arg_joiner.join(sql_parts)
        return template % data, params

    def copy(self):
        copy = super().copy()
        copy.source_expressions = self.source_expressions[:]
        copy.extra = self.extra.copy()
        return copy
```
### 56 - django/db/models/expressions.py:

Start line: 547, End line: 588

```python
class TemporalSubtraction(CombinedExpression):
    output_field = fields.DurationField()

    def __init__(self, lhs, rhs):
        super().__init__(lhs, self.SUB, rhs)

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        lhs = compiler.compile(self.lhs)
        rhs = compiler.compile(self.rhs)
        return connection.ops.subtract_temporals(self.lhs.output_field.get_internal_type(), lhs, rhs)


@deconstructible
class F(Combinable):
    """An object capable of resolving references to existing query objects."""

    def __init__(self, name):
        """
        Arguments:
         * name: the name of the field this expression references
        """
        self.name = name

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)

    def resolve_expression(self, query=None, allow_joins=True, reuse=None,
                           summarize=False, for_save=False):
        return query.resolve_ref(self.name, allow_joins, reuse, summarize)

    def asc(self, **kwargs):
        return OrderBy(self, **kwargs)

    def desc(self, **kwargs):
        return OrderBy(self, descending=True, **kwargs)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __hash__(self):
        return hash(self.name)
```
### 61 - django/db/models/expressions.py:

Start line: 417, End line: 439

```python
class Expression(BaseExpression, Combinable):
    """An expression that can be combined with other expressions."""
    pass


_connector_combinators = {
    connector: [
        (fields.IntegerField, fields.IntegerField, fields.IntegerField),
        (fields.IntegerField, fields.DecimalField, fields.DecimalField),
        (fields.DecimalField, fields.IntegerField, fields.DecimalField),
        (fields.IntegerField, fields.FloatField, fields.FloatField),
        (fields.FloatField, fields.IntegerField, fields.FloatField),
    ]
    for connector in (Combinable.ADD, Combinable.SUB, Combinable.MUL, Combinable.DIV)
}


@functools.lru_cache(maxsize=128)
def _resolve_combined_type(connector, lhs_type, rhs_type):
    combinators = _connector_combinators.get(connector, ())
    for combinator_lhs_type, combinator_rhs_type, combined_type in combinators:
        if issubclass(lhs_type, combinator_lhs_type) and issubclass(rhs_type, combinator_rhs_type):
            return combined_type
```
### 68 - django/db/models/expressions.py:

Start line: 517, End line: 544

```python
class DurationExpression(CombinedExpression):
    def compile(self, side, compiler, connection):
        try:
            output = side.output_field
        except FieldError:
            pass
        else:
            if output.get_internal_type() == 'DurationField':
                sql, params = compiler.compile(side)
                return connection.ops.format_for_duration_arithmetic(sql), params
        return compiler.compile(side)

    def as_sql(self, compiler, connection):
        if connection.features.has_native_duration_field:
            return super().as_sql(compiler, connection)
        connection.ops.check_expression_support(self)
        expressions = []
        expression_params = []
        sql, params = self.compile(self.lhs, compiler, connection)
        expressions.append(sql)
        expression_params.extend(params)
        sql, params = self.compile(self.rhs, compiler, connection)
        expressions.append(sql)
        expression_params.extend(params)
        # order of precedence
        expression_wrapper = '(%s)'
        sql = connection.ops.combine_duration_expression(self.connector, expressions)
        return expression_wrapper % sql, expression_params
```
### 69 - django/db/models/expressions.py:

Start line: 884, End line: 898

```python
class ExpressionList(Func):
    """
    An expression containing multiple expressions. Can be used to provide a
    list of expressions as an argument to another expression, like an
    ordering clause.
    """
    template = '%(expressions)s'

    def __init__(self, *expressions, **extra):
        if not expressions:
            raise ValueError('%s requires at least one expression.' % self.__class__.__name__)
        super().__init__(*expressions, **extra)

    def __str__(self):
        return self.arg_joiner.join(str(arg) for arg in self.source_expressions)
```
### 74 - django/db/models/expressions.py:

Start line: 707, End line: 726

```python
class Value(Expression):
    """Represent a wrapped value as a node within an expression."""
    # Provide a default value for `for_save` in order to allow unresolved
    # instances to be compiled until a decision is taken in #25425.
    for_save = False

    def __init__(self, value, output_field=None):
        """
        Arguments:
         * value: the value this expression represents. The value will be
           added into the sql parameter list and properly quoted.

         * output_field: an instance of the model field type that this
           expression will return, such as IntegerField() or CharField().
        """
        super().__init__(output_field=output_field)
        self.value = value

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.value)
```
### 77 - django/db/models/expressions.py:

Start line: 633, End line: 677

```python
class Func(SQLiteNumericMixin, Expression):
    """An SQL function call."""
    function = None
    template = '%(function)s(%(expressions)s)'
    arg_joiner = ', '
    arity = None  # The number of arguments the function accepts.

    def __init__(self, *expressions, output_field=None, **extra):
        if self.arity is not None and len(expressions) != self.arity:
            raise TypeError(
                "'%s' takes exactly %s %s (%s given)" % (
                    self.__class__.__name__,
                    self.arity,
                    "argument" if self.arity == 1 else "arguments",
                    len(expressions),
                )
            )
        super().__init__(output_field=output_field)
        self.source_expressions = self._parse_expressions(*expressions)
        self.extra = extra

    def __repr__(self):
        args = self.arg_joiner.join(str(arg) for arg in self.source_expressions)
        extra = {**self.extra, **self._get_repr_options()}
        if extra:
            extra = ', '.join(str(key) + '=' + str(val) for key, val in sorted(extra.items()))
            return "{}({}, {})".format(self.__class__.__name__, args, extra)
        return "{}({})".format(self.__class__.__name__, args)

    def _get_repr_options(self):
        """Return a dict of extra __init__() options to include in the repr."""
        return {}

    def get_source_expressions(self):
        return self.source_expressions

    def set_source_expressions(self, exprs):
        self.source_expressions = exprs

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        c = self.copy()
        c.is_summary = summarize
        for pos, arg in enumerate(c.source_expressions):
            c.source_expressions[pos] = arg.resolve_expression(query, allow_joins, reuse, summarize, for_save)
        return c
```
### 78 - django/db/models/expressions.py:

Start line: 255, End line: 282

```python
@deconstructible
class BaseExpression:

    @property
    def conditional(self):
        return isinstance(self.output_field, fields.BooleanField)

    @property
    def field(self):
        return self.output_field

    @cached_property
    def output_field(self):
        """Return the output type of this expressions."""
        output_field = self._resolve_output_field()
        if output_field is None:
            self._output_field_resolved_to_none = True
            raise FieldError('Cannot resolve expression type, unknown output_field')
        return output_field

    @cached_property
    def _output_field_or_none(self):
        """
        Return the output field of this expression, or None if
        _resolve_output_field() didn't return an output type.
        """
        try:
            return self.output_field
        except FieldError:
            if not self._output_field_resolved_to_none:
                raise
```
### 86 - django/db/models/expressions.py:

Start line: 191, End line: 217

```python
@deconstructible
class BaseExpression:

    def as_sql(self, compiler, connection):
        """
        Responsible for returning a (sql, [params]) tuple to be included
        in the current query.

        Different backends can provide their own implementation, by
        providing an `as_{vendor}` method and patching the Expression:

        ```
        def override_as_sql(self, compiler, connection):
            # custom logic
            return super().as_sql(compiler, connection)
        setattr(Expression, 'as_' + connection.vendor, override_as_sql)
        ```

        Arguments:
         * compiler: the query compiler responsible for generating the query.
           Must have a compile method, returning a (sql, [params]) tuple.
           Calling compiler(value) will return a quoted `value`.

         * connection: the database connection used for the current query.

        Return: (sql, params)
          Where `sql` is a string containing ordered sql parameters to be
          replaced with the elements of the list `params`.
        """
        raise NotImplementedError("Subclasses must implement as_sql()")
```
### 88 - django/db/models/expressions.py:

Start line: 33, End line: 147

```python
class Combinable:
    """
    Provide the ability to combine one or two objects with
    some connector. For example F('foo') + F('bar').
    """

    # Arithmetic connectors
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    POW = '^'
    # The following is a quoted % operator - it is quoted because it can be
    # used in strings that also have parameter substitution.
    MOD = '%%'

    # Bitwise operators - note that these are generated by .bitand()
    # and .bitor(), the '&' and '|' are reserved for boolean operator
    # usage.
    BITAND = '&'
    BITOR = '|'
    BITLEFTSHIFT = '<<'
    BITRIGHTSHIFT = '>>'
    BITXOR = '#'

    def _combine(self, other, connector, reversed):
        if not hasattr(other, 'resolve_expression'):
            # everything must be resolvable to an expression
            other = Value(other)

        if reversed:
            return CombinedExpression(other, connector, self)
        return CombinedExpression(self, connector, other)

    #############
    # OPERATORS #
    #############

    def __neg__(self):
        return self._combine(-1, self.MUL, False)

    def __add__(self, other):
        return self._combine(other, self.ADD, False)

    def __sub__(self, other):
        return self._combine(other, self.SUB, False)

    def __mul__(self, other):
        return self._combine(other, self.MUL, False)

    def __truediv__(self, other):
        return self._combine(other, self.DIV, False)

    def __mod__(self, other):
        return self._combine(other, self.MOD, False)

    def __pow__(self, other):
        return self._combine(other, self.POW, False)

    def __and__(self, other):
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
            return Q(self) & Q(other)
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )

    def bitand(self, other):
        return self._combine(other, self.BITAND, False)

    def bitleftshift(self, other):
        return self._combine(other, self.BITLEFTSHIFT, False)

    def bitrightshift(self, other):
        return self._combine(other, self.BITRIGHTSHIFT, False)

    def bitxor(self, other):
        return self._combine(other, self.BITXOR, False)

    def __or__(self, other):
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
            return Q(self) | Q(other)
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )

    def bitor(self, other):
        return self._combine(other, self.BITOR, False)

    def __radd__(self, other):
        return self._combine(other, self.ADD, True)

    def __rsub__(self, other):
        return self._combine(other, self.SUB, True)

    def __rmul__(self, other):
        return self._combine(other, self.MUL, True)

    def __rtruediv__(self, other):
        return self._combine(other, self.DIV, True)

    def __rmod__(self, other):
        return self._combine(other, self.MOD, True)

    def __rpow__(self, other):
        return self._combine(other, self.POW, True)

    def __rand__(self, other):
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )

    def __ror__(self, other):
        raise NotImplementedError(
            "Use .bitand() and .bitor() for bitwise logical operations."
        )
```
### 116 - django/db/models/expressions.py:

Start line: 284, End line: 309

```python
@deconstructible
class BaseExpression:

    def _resolve_output_field(self):
        """
        Attempt to infer the output type of the expression. If the output
        fields of all source fields match then, simply infer the same type
        here. This isn't always correct, but it makes sense most of the time.

        Consider the difference between `2 + 2` and `2 / 3`. Inferring
        the type here is a convenience for the common case. The user should
        supply their own output_field with more complex computations.

        If a source's output field resolves to None, exclude it from this check.
        If all sources are None, then an error is raised higher up the stack in
        the output_field property.
        """
        sources_iter = (source for source in self.get_source_fields() if source is not None)
        for output_field in sources_iter:
            for source in sources_iter:
                if not isinstance(output_field, source.__class__):
                    raise FieldError(
                        'Expression contains mixed types: %s, %s. You must '
                        'set output_field.' % (
                            output_field.__class__.__name__,
                            source.__class__.__name__,
                        )
                    )
            return output_field
```
### 139 - django/db/models/expressions.py:

Start line: 851, End line: 881

```python
class Ref(Expression):
    """
    Reference to column alias of the query. For example, Ref('sum_cost') in
    qs.annotate(sum_cost=Sum('cost')) query.
    """
    def __init__(self, refs, source):
        super().__init__()
        self.refs, self.source = refs, source

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.refs, self.source)

    def get_source_expressions(self):
        return [self.source]

    def set_source_expressions(self, exprs):
        self.source, = exprs

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # The sub-expression `source` has already been resolved, as this is
        # just a reference to the name of `source`.
        return self

    def relabeled_clone(self, relabels):
        return self

    def as_sql(self, compiler, connection):
        return connection.ops.quote_name(self.refs), []

    def get_group_by_cols(self, alias=None):
        return [self]
```
### 150 - django/db/models/expressions.py:

Start line: 1396, End line: 1432

```python
class WindowFrame(Expression):

    def __str__(self):
        if self.start.value is not None and self.start.value < 0:
            start = '%d %s' % (abs(self.start.value), connection.ops.PRECEDING)
        elif self.start.value is not None and self.start.value == 0:
            start = connection.ops.CURRENT_ROW
        else:
            start = connection.ops.UNBOUNDED_PRECEDING

        if self.end.value is not None and self.end.value > 0:
            end = '%d %s' % (self.end.value, connection.ops.FOLLOWING)
        elif self.end.value is not None and self.end.value == 0:
            end = connection.ops.CURRENT_ROW
        else:
            end = connection.ops.UNBOUNDED_FOLLOWING
        return self.template % {
            'frame_type': self.frame_type,
            'start': start,
            'end': end,
        }

    def window_frame_start_end(self, connection, start, end):
        raise NotImplementedError('Subclasses must implement window_frame_start_end().')


class RowRange(WindowFrame):
    frame_type = 'ROWS'

    def window_frame_start_end(self, connection, start, end):
        return connection.ops.window_frame_rows_start_end(start, end)


class ValueRange(WindowFrame):
    frame_type = 'RANGE'

    def window_frame_start_end(self, connection, start, end):
        return connection.ops.window_frame_range_start_end(start, end)
```
### 184 - django/db/models/expressions.py:

Start line: 1361, End line: 1394

```python
class WindowFrame(Expression):
    """
    Model the frame clause in window expressions. There are two types of frame
    clauses which are subclasses, however, all processing and validation (by no
    means intended to be complete) is done here. Thus, providing an end for a
    frame is optional (the default is UNBOUNDED FOLLOWING, which is the last
    row in the frame).
    """
    template = '%(frame_type)s BETWEEN %(start)s AND %(end)s'

    def __init__(self, start=None, end=None):
        self.start = Value(start)
        self.end = Value(end)

    def set_source_expressions(self, exprs):
        self.start, self.end = exprs

    def get_source_expressions(self):
        return [self.start, self.end]

    def as_sql(self, compiler, connection):
        connection.ops.check_expression_support(self)
        start, end = self.window_frame_start_end(connection, self.start.value, self.end.value)
        return self.template % {
            'frame_type': self.frame_type,
            'start': start,
            'end': end,
        }, []

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    def get_group_by_cols(self, alias=None):
        return []
```
