# django__django-16757

| **django/django** | `83c9765f45e4622e4a5af3adcd92263a28b13624` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6237 |
| **Any found context length** | 6237 |
| **Avg pos** | 11.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -916,10 +916,13 @@ def _check_list_display_item(self, obj, item, label):
                         id="admin.E108",
                     )
                 ]
-        if isinstance(field, models.ManyToManyField):
+        if isinstance(field, models.ManyToManyField) or (
+            getattr(field, "rel", None) and field.rel.field.many_to_one
+        ):
             return [
                 checks.Error(
-                    "The value of '%s' must not be a ManyToManyField." % label,
+                    f"The value of '{label}' must not be a many-to-many field or a "
+                    f"reverse foreign key.",
                     obj=obj.__class__,
                     id="admin.E109",
                 )

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/checks.py | 919 | 922 | 11 | 4 | 6237


## Problem Statement

```
Admin check for reversed foreign key used in "list_display"
Description
	
Currently the admin site checks and reports an admin.E109 system check error when a ManyToManyField is declared within the list_display values.
But, when a reversed foreign key is used there is no system error reported:
System check identified no issues (0 silenced).
April 10, 2023 - 19:04:29
Django version 5.0.dev20230410064954, using settings 'projectfromrepo.settings'
Starting development server at http://0.0.0.0:8000/
Quit the server with CONTROL-C.
and visiting the admin site for the relevant model results in the following error:
TypeError: create_reverse_many_to_one_manager.<locals>.RelatedManager.__call__() missing 1 required keyword-only argument: 'manager'
[10/Apr/2023 19:30:40] "GET /admin/testapp/question/ HTTP/1.1" 500 415926
Ideally, using a reversed foreign key would also result in a system check error instead of a 500 response.
To reproduce, follow the Question and Choice models from the Django Tutorial and add the following to the QuestionAdmin:
list_display = ["question_text", "choice_set"]
Then, visit /admin/testapp/question/ and the following traceback is returned:
Internal Server Error: /admin/testapp/question/
Traceback (most recent call last):
 File "/some/path/django/django/db/models/options.py", line 681, in get_field
	return self.fields_map[field_name]
		 ~~~~~~~~~~~~~~~^^^^^^^^^^^^
KeyError: 'choice_set'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
 File "/some/path/django/django/contrib/admin/utils.py", line 288, in lookup_field
	f = _get_non_gfk_field(opts, name)
		^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/contrib/admin/utils.py", line 319, in _get_non_gfk_field
	field = opts.get_field(name)
			^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/db/models/options.py", line 683, in get_field
	raise FieldDoesNotExist(
django.core.exceptions.FieldDoesNotExist: Question has no field named 'choice_set'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
 File "/some/path/django/django/core/handlers/exception.py", line 55, in inner
	response = get_response(request)
			 ^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/core/handlers/base.py", line 220, in _get_response
	response = response.render()
			 ^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/response.py", line 111, in render
	self.content = self.rendered_content
				 ^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/response.py", line 89, in rendered_content
	return template.render(context, self._request)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/backends/django.py", line 61, in render
	return self.template.render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 171, in render
	return self._render(context)
		 ^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 163, in _render
	return self.nodelist.render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 962, in render_annotated
	return self.render(context)
		 ^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/loader_tags.py", line 157, in render
	return compiled_parent._render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 163, in _render
	return self.nodelist.render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 962, in render_annotated
	return self.render(context)
		 ^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/loader_tags.py", line 157, in render
	return compiled_parent._render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 163, in _render
	return self.nodelist.render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 962, in render_annotated
	return self.render(context)
		 ^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/loader_tags.py", line 63, in render
	result = block.nodelist.render(context)
			 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 962, in render_annotated
	return self.render(context)
		 ^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/loader_tags.py", line 63, in render
	result = block.nodelist.render(context)
			 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 1001, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
							 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/base.py", line 962, in render_annotated
	return self.render(context)
		 ^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/contrib/admin/templatetags/base.py", line 45, in render
	return super().render(context)
		 ^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/template/library.py", line 258, in render
	_dict = self.func(*resolved_args, **resolved_kwargs)
			^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/contrib/admin/templatetags/admin_list.py", line 340, in result_list
	"results": list(results(cl)),
			 ^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/contrib/admin/templatetags/admin_list.py", line 316, in results
	yield ResultList(None, items_for_result(cl, res, None))
		 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/contrib/admin/templatetags/admin_list.py", line 307, in __init__
	super().__init__(*items)
 File "/some/path/django/django/contrib/admin/templatetags/admin_list.py", line 217, in items_for_result
	f, attr, value = lookup_field(field_name, result, cl.model_admin)
					 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/some/path/django/django/contrib/admin/utils.py", line 301, in lookup_field
	value = attr()
			^^^^^^
TypeError: create_reverse_many_to_one_manager.<locals>.RelatedManager.__call__() missing 1 required keyword-only argument: 'manager'
[10/Apr/2023 19:30:40] "GET /admin/testapp/question/ HTTP/1.1" 500 415926

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/db/models/fields/related.py | 1463 | 1592| 984 | 984 | 14704 | 
| 2 | 2 django/contrib/admin/options.py | 1935 | 2023| 676 | 1660 | 34086 | 
| 3 | 3 django/contrib/admin/views/main.py | 592 | 624| 227 | 1887 | 38959 | 
| 4 | **4 django/contrib/admin/checks.py** | 1090 | 1142| 430 | 2317 | 48485 | 
| 5 | 4 django/contrib/admin/options.py | 2024 | 2115| 784 | 3101 | 48485 | 
| 6 | 5 django/db/models/fields/related_descriptors.py | 788 | 893| 808 | 3909 | 60143 | 
| 7 | **5 django/contrib/admin/checks.py** | 954 | 978| 197 | 4106 | 60143 | 
| 8 | **5 django/contrib/admin/checks.py** | 217 | 264| 333 | 4439 | 60143 | 
| 9 | 5 django/contrib/admin/options.py | 1 | 119| 796 | 5235 | 60143 | 
| 10 | 5 django/contrib/admin/options.py | 1770 | 1872| 780 | 6015 | 60143 | 
| **-> 11 <-** | **5 django/contrib/admin/checks.py** | 893 | 927| 222 | 6237 | 60143 | 
| 12 | 5 django/db/models/fields/related_descriptors.py | 632 | 663| 243 | 6480 | 60143 | 
| 13 | 5 django/db/models/fields/related.py | 228 | 303| 696 | 7176 | 60143 | 
| 14 | 5 django/contrib/admin/options.py | 364 | 440| 531 | 7707 | 60143 | 
| 15 | 5 django/db/models/fields/related_descriptors.py | 730 | 749| 231 | 7938 | 60143 | 
| 16 | **5 django/contrib/admin/checks.py** | 789 | 807| 183 | 8121 | 60143 | 
| 17 | 5 django/db/models/fields/related_descriptors.py | 978 | 1044| 592 | 8713 | 60143 | 
| 18 | 5 django/contrib/admin/options.py | 441 | 506| 550 | 9263 | 60143 | 
| 19 | 6 django/contrib/sites/managers.py | 1 | 46| 277 | 9540 | 60538 | 
| 20 | **6 django/contrib/admin/checks.py** | 1039 | 1070| 229 | 9769 | 60538 | 
| 21 | 6 django/db/models/fields/related_descriptors.py | 705 | 728| 227 | 9996 | 60538 | 
| 22 | 6 django/db/models/fields/related_descriptors.py | 895 | 928| 279 | 10275 | 60538 | 
| 23 | 6 django/db/models/fields/related.py | 1594 | 1691| 655 | 10930 | 60538 | 
| 24 | 6 django/contrib/admin/views/main.py | 1 | 64| 383 | 11313 | 60538 | 
| 25 | 6 django/db/models/fields/related.py | 304 | 341| 296 | 11609 | 60538 | 
| 26 | 6 django/db/models/fields/related_descriptors.py | 1238 | 1300| 544 | 12153 | 60538 | 
| 27 | 6 django/db/models/fields/related_descriptors.py | 665 | 703| 354 | 12507 | 60538 | 
| 28 | 6 django/contrib/admin/views/main.py | 307 | 342| 312 | 12819 | 60538 | 
| 29 | 7 django/contrib/admin/filters.py | 220 | 280| 541 | 13360 | 66264 | 
| 30 | 8 django/db/models/base.py | 1839 | 1874| 246 | 13606 | 84991 | 
| 31 | **8 django/contrib/admin/checks.py** | 55 | 173| 772 | 14378 | 84991 | 
| 32 | **8 django/contrib/admin/checks.py** | 176 | 192| 155 | 14533 | 84991 | 
| 33 | **8 django/contrib/admin/checks.py** | 878 | 891| 121 | 14654 | 84991 | 
| 34 | 8 django/db/models/fields/related.py | 1423 | 1461| 213 | 14867 | 84991 | 
| 35 | 9 django/contrib/admin/__init__.py | 1 | 53| 292 | 15159 | 85283 | 
| 36 | 9 django/db/models/base.py | 1649 | 1678| 209 | 15368 | 85283 | 
| 37 | 9 django/contrib/admin/options.py | 508 | 552| 348 | 15716 | 85283 | 
| 38 | 9 django/db/models/fields/related_descriptors.py | 751 | 786| 268 | 15984 | 85283 | 
| 39 | 9 django/db/models/fields/related_descriptors.py | 1457 | 1507| 407 | 16391 | 85283 | 
| 40 | 9 django/db/models/base.py | 2025 | 2078| 359 | 16750 | 85283 | 
| 41 | 9 django/db/models/fields/related_descriptors.py | 1386 | 1455| 526 | 17276 | 85283 | 
| 42 | **9 django/contrib/admin/checks.py** | 1274 | 1318| 343 | 17619 | 85283 | 
| 43 | 9 django/contrib/admin/options.py | 2488 | 2523| 315 | 17934 | 85283 | 
| 44 | **9 django/contrib/admin/checks.py** | 840 | 876| 268 | 18202 | 85283 | 
| 45 | 10 django/contrib/admin/utils.py | 454 | 483| 200 | 18402 | 89673 | 
| 46 | 10 django/contrib/admin/options.py | 1531 | 1557| 233 | 18635 | 89673 | 
| 47 | 10 django/db/models/fields/related.py | 156 | 187| 209 | 18844 | 89673 | 
| 48 | 10 django/contrib/admin/views/main.py | 175 | 270| 828 | 19672 | 89673 | 
| 49 | **10 django/contrib/admin/checks.py** | 284 | 312| 218 | 19890 | 89673 | 
| 50 | 11 django/contrib/admin/templatetags/admin_list.py | 459 | 531| 385 | 20275 | 93479 | 
| 51 | 11 django/db/models/base.py | 1529 | 1565| 288 | 20563 | 93479 | 
| 52 | **11 django/contrib/admin/checks.py** | 1243 | 1272| 196 | 20759 | 93479 | 
| 53 | 11 django/db/models/fields/related.py | 1693 | 1743| 431 | 21190 | 93479 | 
| 54 | 11 django/db/models/fields/related_descriptors.py | 1153 | 1171| 155 | 21345 | 93479 | 
| 55 | 11 django/contrib/admin/options.py | 1873 | 1904| 303 | 21648 | 93479 | 
| 56 | 11 django/db/models/fields/related_descriptors.py | 1173 | 1204| 228 | 21876 | 93479 | 
| 57 | 11 django/db/models/fields/related_descriptors.py | 1133 | 1151| 174 | 22050 | 93479 | 
| 58 | 11 django/contrib/admin/options.py | 2194 | 2252| 444 | 22494 | 93479 | 
| 59 | 11 django/db/models/fields/related_descriptors.py | 1334 | 1351| 154 | 22648 | 93479 | 
| 60 | **11 django/contrib/admin/checks.py** | 929 | 952| 195 | 22843 | 93479 | 
| 61 | 11 django/contrib/admin/templatetags/admin_list.py | 199 | 295| 817 | 23660 | 93479 | 
| 62 | 12 django/contrib/admin/models.py | 1 | 21| 123 | 23783 | 94672 | 
| 63 | **12 django/contrib/admin/checks.py** | 430 | 458| 224 | 24007 | 94672 | 
| 64 | 12 django/contrib/admin/options.py | 1278 | 1340| 511 | 24518 | 94672 | 
| 65 | 12 django/db/models/fields/related_descriptors.py | 1206 | 1236| 260 | 24778 | 94672 | 
| 66 | 12 django/db/models/fields/related_descriptors.py | 1089 | 1131| 381 | 25159 | 94672 | 
| 67 | 12 django/db/models/fields/related_descriptors.py | 406 | 427| 158 | 25317 | 94672 | 
| 68 | 12 django/contrib/admin/templatetags/admin_list.py | 1 | 33| 194 | 25511 | 94672 | 
| 69 | **12 django/contrib/admin/checks.py** | 761 | 786| 158 | 25669 | 94672 | 
| 70 | 13 django/contrib/admin/sites.py | 1 | 33| 224 | 25893 | 99169 | 
| 71 | **13 django/contrib/admin/checks.py** | 1230 | 1241| 116 | 26009 | 99169 | 
| 72 | 13 django/contrib/admin/options.py | 2459 | 2486| 254 | 26263 | 99169 | 
| 73 | 13 django/contrib/admin/options.py | 2254 | 2303| 450 | 26713 | 99169 | 
| 74 | 13 django/db/models/fields/related_descriptors.py | 1046 | 1064| 199 | 26912 | 99169 | 
| 75 | 13 django/db/models/fields/related_descriptors.py | 1066 | 1087| 203 | 27115 | 99169 | 
| 76 | **13 django/contrib/admin/checks.py** | 521 | 537| 144 | 27259 | 99169 | 
| 77 | **13 django/contrib/admin/checks.py** | 742 | 759| 139 | 27398 | 99169 | 
| 78 | **13 django/contrib/admin/checks.py** | 194 | 215| 141 | 27539 | 99169 | 
| 79 | 13 django/contrib/admin/options.py | 121 | 154| 235 | 27774 | 99169 | 
| 80 | **13 django/contrib/admin/checks.py** | 1321 | 1350| 176 | 27950 | 99169 | 
| 81 | 14 django/contrib/contenttypes/admin.py | 1 | 88| 585 | 28535 | 100174 | 
| 82 | **14 django/contrib/admin/checks.py** | 980 | 1037| 457 | 28992 | 100174 | 
| 83 | 14 django/db/models/base.py | 2080 | 2185| 733 | 29725 | 100174 | 
| 84 | 14 django/contrib/admin/templatetags/admin_list.py | 298 | 352| 351 | 30076 | 100174 | 
| 85 | **14 django/contrib/admin/checks.py** | 704 | 740| 301 | 30377 | 100174 | 
| 86 | 15 django/contrib/admindocs/views.py | 213 | 297| 615 | 30992 | 103662 | 
| 87 | 15 django/contrib/admin/options.py | 1559 | 1626| 586 | 31578 | 103662 | 
| 88 | 16 django/contrib/auth/admin.py | 28 | 40| 130 | 31708 | 105469 | 
| 89 | **16 django/contrib/admin/checks.py** | 266 | 282| 138 | 31846 | 105469 | 
| 90 | 16 django/contrib/admin/filters.py | 632 | 645| 119 | 31965 | 105469 | 
| 91 | **16 django/contrib/admin/checks.py** | 1144 | 1180| 252 | 32217 | 105469 | 
| 92 | 17 django/contrib/contenttypes/fields.py | 756 | 804| 407 | 32624 | 111297 | 
| 93 | **17 django/contrib/admin/checks.py** | 673 | 702| 241 | 32865 | 111297 | 
| 94 | 18 django/db/models/options.py | 1 | 58| 353 | 33218 | 118993 | 
| 95 | 18 django/contrib/admin/options.py | 833 | 868| 251 | 33469 | 118993 | 
| 96 | 18 django/db/models/fields/related.py | 582 | 602| 138 | 33607 | 118993 | 
| 97 | 19 django/db/migrations/questioner.py | 291 | 342| 367 | 33974 | 121689 | 
| 98 | 20 django/contrib/admin/widgets.py | 297 | 343| 439 | 34413 | 125879 | 
| 99 | 21 django/db/models/__init__.py | 1 | 116| 682 | 35095 | 126561 | 
| 100 | 21 django/db/models/fields/related.py | 1 | 42| 267 | 35362 | 126561 | 
| 101 | 22 django/contrib/admin/helpers.py | 1 | 36| 209 | 35571 | 130173 | 
| 102 | 22 django/contrib/admin/views/main.py | 528 | 590| 504 | 36075 | 130173 | 
| 103 | 22 django/contrib/admin/views/main.py | 271 | 287| 158 | 36233 | 130173 | 
| 104 | 22 django/contrib/admin/options.py | 2400 | 2457| 465 | 36698 | 130173 | 
| 105 | 22 django/db/models/fields/related_descriptors.py | 1302 | 1332| 281 | 36979 | 130173 | 
| 106 | 22 django/db/models/base.py | 1 | 66| 361 | 37340 | 130173 | 
| 107 | 22 django/contrib/admin/sites.py | 570 | 597| 196 | 37536 | 130173 | 
| 108 | **22 django/contrib/admin/checks.py** | 556 | 578| 194 | 37730 | 130173 | 
| 109 | 22 django/db/models/fields/related.py | 1972 | 2006| 266 | 37996 | 130173 | 
| 110 | 22 django/db/models/fields/related.py | 1894 | 1941| 505 | 38501 | 130173 | 
| 111 | 23 django/db/models/manager.py | 1 | 173| 1241 | 39742 | 131621 | 
| 112 | 23 django/contrib/admin/options.py | 1081 | 1102| 153 | 39895 | 131621 | 
| 113 | 23 django/contrib/admin/helpers.py | 39 | 96| 322 | 40217 | 131621 | 
| 114 | 23 django/contrib/admin/options.py | 700 | 740| 280 | 40497 | 131621 | 
| 115 | 23 django/contrib/admindocs/views.py | 1 | 36| 252 | 40749 | 131621 | 
| 116 | 23 django/contrib/admin/filters.py | 24 | 87| 418 | 41167 | 131621 | 
| 117 | 23 django/db/models/fields/related.py | 210 | 226| 142 | 41309 | 131621 | 
| 118 | 23 django/contrib/admin/utils.py | 311 | 337| 189 | 41498 | 131621 | 
| 119 | 23 django/contrib/admin/options.py | 2117 | 2192| 599 | 42097 | 131621 | 
| 120 | 23 django/db/models/fields/related_descriptors.py | 575 | 629| 338 | 42435 | 131621 | 
| 121 | 23 django/contrib/admin/widgets.py | 171 | 204| 244 | 42679 | 131621 | 
| 122 | 24 django/db/models/fields/__init__.py | 1280 | 1317| 245 | 42924 | 150625 | 
| 123 | 24 django/db/models/base.py | 1815 | 1837| 175 | 43099 | 150625 | 
| 124 | **24 django/contrib/admin/checks.py** | 825 | 838| 117 | 43216 | 150625 | 
| 125 | 24 django/contrib/admin/options.py | 1161 | 1181| 201 | 43417 | 150625 | 
| 126 | 24 django/db/models/base.py | 1680 | 1714| 252 | 43669 | 150625 | 
| 127 | 25 django/contrib/admin/views/autocomplete.py | 66 | 123| 425 | 44094 | 151466 | 
| 128 | 25 django/contrib/contenttypes/fields.py | 690 | 732| 325 | 44419 | 151466 | 
| 129 | 25 django/contrib/admin/filters.py | 1 | 21| 144 | 44563 | 151466 | 
| 130 | 26 django/core/management/commands/inspectdb.py | 54 | 263| 1616 | 46179 | 154537 | 
| 131 | 26 django/contrib/admin/options.py | 340 | 362| 169 | 46348 | 154537 | 
| 132 | 26 django/db/models/options.py | 333 | 369| 338 | 46686 | 154537 | 
| 133 | 27 django/core/checks/model_checks.py | 187 | 228| 345 | 47031 | 156347 | 
| 134 | 28 django/contrib/auth/checks.py | 107 | 221| 786 | 47817 | 157863 | 
| 135 | 29 django/views/generic/__init__.py | 1 | 40| 204 | 48021 | 158068 | 
| 136 | 29 django/contrib/admindocs/views.py | 298 | 392| 640 | 48661 | 158068 | 
| 137 | 29 django/contrib/admin/options.py | 683 | 698| 123 | 48784 | 158068 | 
| 138 | 29 django/db/models/fields/related.py | 1162 | 1177| 136 | 48920 | 158068 | 
| 139 | 29 django/db/models/fields/related.py | 1105 | 1122| 133 | 49053 | 158068 | 
| 140 | 29 django/db/models/fields/related_descriptors.py | 1353 | 1384| 348 | 49401 | 158068 | 
| 141 | 29 django/contrib/admin/sites.py | 228 | 249| 221 | 49622 | 158068 | 
| 142 | 29 django/db/models/manager.py | 176 | 214| 207 | 49829 | 158068 | 
| 143 | 29 django/contrib/admin/utils.py | 405 | 451| 402 | 50231 | 158068 | 
| 144 | **29 django/contrib/admin/checks.py** | 505 | 519| 122 | 50353 | 158068 | 
| 145 | 29 django/contrib/contenttypes/fields.py | 734 | 754| 192 | 50545 | 158068 | 
| 146 | 29 django/contrib/admin/options.py | 243 | 256| 139 | 50684 | 158068 | 
| 147 | **29 django/contrib/admin/checks.py** | 480 | 503| 188 | 50872 | 158068 | 
| 148 | 30 django/contrib/gis/admin/__init__.py | 1 | 30| 130 | 51002 | 158198 | 
| 149 | 30 django/db/models/fields/related.py | 1036 | 1072| 261 | 51263 | 158198 | 
| 150 | 30 django/core/checks/model_checks.py | 1 | 90| 671 | 51934 | 158198 | 
| 151 | 30 django/db/models/fields/related.py | 128 | 154| 171 | 52105 | 158198 | 
| 152 | 30 django/contrib/admin/views/main.py | 67 | 173| 866 | 52971 | 158198 | 
| 153 | 30 django/contrib/admin/templatetags/admin_list.py | 180 | 196| 140 | 53111 | 158198 | 
| 154 | 30 django/db/models/fields/related.py | 1179 | 1214| 267 | 53378 | 158198 | 
| 155 | 30 django/contrib/admin/templatetags/admin_list.py | 355 | 456| 760 | 54138 | 158198 | 
| 156 | 30 django/db/models/fields/related.py | 1943 | 1970| 305 | 54443 | 158198 | 
| 157 | 30 django/contrib/auth/checks.py | 1 | 104| 728 | 55171 | 158198 | 
| 158 | 30 django/db/models/base.py | 2480 | 2532| 341 | 55512 | 158198 | 
| 159 | 30 django/contrib/admin/filters.py | 555 | 592| 371 | 55883 | 158198 | 
| 160 | 30 django/contrib/admin/options.py | 1699 | 1735| 337 | 56220 | 158198 | 
| 161 | 31 django/core/management/base.py | 1 | 44| 272 | 56492 | 163058 | 
| 162 | 31 django/db/models/base.py | 1398 | 1428| 220 | 56712 | 163058 | 
| 163 | 32 django/db/models/fields/reverse_related.py | 1 | 19| 136 | 56848 | 165693 | 
| 164 | 32 django/db/models/fields/related.py | 189 | 208| 155 | 57003 | 165693 | 
| 165 | 32 django/contrib/admin/options.py | 1676 | 1697| 133 | 57136 | 165693 | 
| 166 | 32 django/contrib/admin/sites.py | 443 | 459| 136 | 57272 | 165693 | 
| 167 | **32 django/contrib/admin/checks.py** | 539 | 554| 136 | 57408 | 165693 | 
| 168 | 32 django/contrib/admin/sites.py | 81 | 97| 132 | 57540 | 165693 | 
| 169 | 32 django/db/models/base.py | 1381 | 1396| 142 | 57682 | 165693 | 
| 170 | 32 django/contrib/admin/widgets.py | 207 | 234| 217 | 57899 | 165693 | 
| 171 | 32 django/contrib/admin/filters.py | 282 | 316| 311 | 58210 | 165693 | 
| 172 | 32 django/contrib/contenttypes/admin.py | 91 | 144| 420 | 58630 | 165693 | 
| 173 | 32 django/contrib/admin/utils.py | 140 | 178| 310 | 58940 | 165693 | 
| 174 | **32 django/contrib/admin/checks.py** | 580 | 608| 195 | 59135 | 165693 | 
| 175 | 32 django/contrib/admin/options.py | 1104 | 1123| 131 | 59266 | 165693 | 


### Hint

```
Thanks for the report.
```

## Patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -916,10 +916,13 @@ def _check_list_display_item(self, obj, item, label):
                         id="admin.E108",
                     )
                 ]
-        if isinstance(field, models.ManyToManyField):
+        if isinstance(field, models.ManyToManyField) or (
+            getattr(field, "rel", None) and field.rel.field.many_to_one
+        ):
             return [
                 checks.Error(
-                    "The value of '%s' must not be a ManyToManyField." % label,
+                    f"The value of '{label}' must not be a many-to-many field or a "
+                    f"reverse foreign key.",
                     obj=obj.__class__,
                     id="admin.E109",
                 )

```

## Test Patch

```diff
diff --git a/tests/modeladmin/test_checks.py b/tests/modeladmin/test_checks.py
--- a/tests/modeladmin/test_checks.py
+++ b/tests/modeladmin/test_checks.py
@@ -537,7 +537,20 @@ class TestModelAdmin(ModelAdmin):
         self.assertIsInvalid(
             TestModelAdmin,
             ValidationTestModel,
-            "The value of 'list_display[0]' must not be a ManyToManyField.",
+            "The value of 'list_display[0]' must not be a many-to-many field or a "
+            "reverse foreign key.",
+            "admin.E109",
+        )
+
+    def test_invalid_reverse_related_field(self):
+        class TestModelAdmin(ModelAdmin):
+            list_display = ["song_set"]
+
+        self.assertIsInvalid(
+            TestModelAdmin,
+            Band,
+            "The value of 'list_display[0]' must not be a many-to-many field or a "
+            "reverse foreign key.",
             "admin.E109",
         )
 

```


## Code snippets

### 1 - django/db/models/fields/related.py:

Start line: 1463, End line: 1592

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
### 2 - django/contrib/admin/options.py:

Start line: 1935, End line: 2023

```python
class ModelAdmin(BaseModelAdmin):

    @csrf_protect_m
    def changelist_view(self, request, extra_context=None):
        """
        The 'change list' admin view for this model.
        """
        from django.contrib.admin.views.main import ERROR_FLAG

        app_label = self.opts.app_label
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied

        try:
            cl = self.get_changelist_instance(request)
        except IncorrectLookupParameters:
            # Wacky lookup parameters were given, so redirect to the main
            # changelist page, without parameters, and pass an 'invalid=1'
            # parameter via the query string. If wacky parameters were given
            # and the 'invalid=1' parameter was already in the query string,
            # something is screwed up with the database, so display an error
            # page.
            if ERROR_FLAG in request.GET:
                return SimpleTemplateResponse(
                    "admin/invalid_setup.html",
                    {
                        "title": _("Database error"),
                    },
                )
            return HttpResponseRedirect(request.path + "?" + ERROR_FLAG + "=1")

        # If the request was POSTed, this might be a bulk action or a bulk
        # edit. Try to look up an action or confirmation first, but if this
        # isn't an action the POST will fall through to the bulk edit check,
        # below.
        action_failed = False
        selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)

        actions = self.get_actions(request)
        # Actions with no confirmation
        if (
            actions
            and request.method == "POST"
            and "index" in request.POST
            and "_save" not in request.POST
        ):
            if selected:
                response = self.response_action(
                    request, queryset=cl.get_queryset(request)
                )
                if response:
                    return response
                else:
                    action_failed = True
            else:
                msg = _(
                    "Items must be selected in order to perform "
                    "actions on them. No items have been changed."
                )
                self.message_user(request, msg, messages.WARNING)
                action_failed = True

        # Actions with confirmation
        if (
            actions
            and request.method == "POST"
            and helpers.ACTION_CHECKBOX_NAME in request.POST
            and "index" not in request.POST
            and "_save" not in request.POST
        ):
            if selected:
                response = self.response_action(
                    request, queryset=cl.get_queryset(request)
                )
                if response:
                    return response
                else:
                    action_failed = True

        if action_failed:
            # Redirect back to the changelist page to avoid resubmitting the
            # form if the user refreshes the browser or uses the "No, take
            # me back" button on the action confirmation page.
            return HttpResponseRedirect(request.get_full_path())

        # If we're allowing changelist editing, we need to construct a formset
        # for the changelist given all the fields to be edited. Then we'll
        # use the formset to validate/process POSTed data.
        formset = cl.formset = None

        # Handle POSTed bulk-edit data.
        # ... other code
```
### 3 - django/contrib/admin/views/main.py:

Start line: 592, End line: 624

```python
class ChangeList:

    def apply_select_related(self, qs):
        if self.list_select_related is True:
            return qs.select_related()

        if self.list_select_related is False:
            if self.has_related_field_in_list_display():
                return qs.select_related()

        if self.list_select_related:
            return qs.select_related(*self.list_select_related)
        return qs

    def has_related_field_in_list_display(self):
        for field_name in self.list_display:
            try:
                field = self.lookup_opts.get_field(field_name)
            except FieldDoesNotExist:
                pass
            else:
                if isinstance(field.remote_field, ManyToOneRel):
                    # <FK>_id field names don't require a join.
                    if field_name != field.get_attname():
                        return True
        return False

    def url_for_result(self, result):
        pk = getattr(result, self.pk_attname)
        return reverse(
            "admin:%s_%s_change" % (self.opts.app_label, self.opts.model_name),
            args=(quote(pk),),
            current_app=self.model_admin.admin_site.name,
        )
```
### 4 - django/contrib/admin/checks.py:

Start line: 1090, End line: 1142

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_editable_item(self, obj, field_name, label):
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E121"
            )
        else:
            if field_name not in obj.list_display:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not "
                        "contained in 'list_display'." % (label, field_name),
                        obj=obj.__class__,
                        id="admin.E122",
                    )
                ]
            elif obj.list_display_links and field_name in obj.list_display_links:
                return [
                    checks.Error(
                        "The value of '%s' cannot be in both 'list_editable' and "
                        "'list_display_links'." % field_name,
                        obj=obj.__class__,
                        id="admin.E123",
                    )
                ]
            # If list_display[0] is in list_editable, check that
            # list_display_links is set. See #22792 and #26229 for use cases.
            elif (
                obj.list_display[0] == field_name
                and not obj.list_display_links
                and obj.list_display_links is not None
            ):
                return [
                    checks.Error(
                        "The value of '%s' refers to the first field in 'list_display' "
                        "('%s'), which cannot be used unless 'list_display_links' is "
                        "set." % (label, obj.list_display[0]),
                        obj=obj.__class__,
                        id="admin.E124",
                    )
                ]
            elif not field.editable or field.primary_key:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not editable "
                        "through the admin." % (label, field_name),
                        obj=obj.__class__,
                        id="admin.E125",
                    )
                ]
            else:
                return []
```
### 5 - django/contrib/admin/options.py:

Start line: 2024, End line: 2115

```python
class ModelAdmin(BaseModelAdmin):

    @csrf_protect_m
    def changelist_view(self, request, extra_context=None):
        # ... other code
        if request.method == "POST" and cl.list_editable and "_save" in request.POST:
            if not self.has_change_permission(request):
                raise PermissionDenied
            FormSet = self.get_changelist_formset(request)
            modified_objects = self._get_list_editable_queryset(
                request, FormSet.get_default_prefix()
            )
            formset = cl.formset = FormSet(
                request.POST, request.FILES, queryset=modified_objects
            )
            if formset.is_valid():
                changecount = 0
                with transaction.atomic(using=router.db_for_write(self.model)):
                    for form in formset.forms:
                        if form.has_changed():
                            obj = self.save_form(request, form, change=True)
                            self.save_model(request, obj, form, change=True)
                            self.save_related(request, form, formsets=[], change=True)
                            change_msg = self.construct_change_message(
                                request, form, None
                            )
                            self.log_change(request, obj, change_msg)
                            changecount += 1
                if changecount:
                    msg = ngettext(
                        "%(count)s %(name)s was changed successfully.",
                        "%(count)s %(name)s were changed successfully.",
                        changecount,
                    ) % {
                        "count": changecount,
                        "name": model_ngettext(self.opts, changecount),
                    }
                    self.message_user(request, msg, messages.SUCCESS)

                return HttpResponseRedirect(request.get_full_path())

        # Handle GET -- construct a formset for display.
        elif cl.list_editable and self.has_change_permission(request):
            FormSet = self.get_changelist_formset(request)
            formset = cl.formset = FormSet(queryset=cl.result_list)

        # Build the list of media to be used by the formset.
        if formset:
            media = self.media + formset.media
        else:
            media = self.media

        # Build the action form and populate it with available actions.
        if actions:
            action_form = self.action_form(auto_id=None)
            action_form.fields["action"].choices = self.get_action_choices(request)
            media += action_form.media
        else:
            action_form = None

        selection_note_all = ngettext(
            "%(total_count)s selected", "All %(total_count)s selected", cl.result_count
        )

        context = {
            **self.admin_site.each_context(request),
            "module_name": str(self.opts.verbose_name_plural),
            "selection_note": _("0 of %(cnt)s selected") % {"cnt": len(cl.result_list)},
            "selection_note_all": selection_note_all % {"total_count": cl.result_count},
            "title": cl.title,
            "subtitle": None,
            "is_popup": cl.is_popup,
            "to_field": cl.to_field,
            "cl": cl,
            "media": media,
            "has_add_permission": self.has_add_permission(request),
            "opts": cl.opts,
            "action_form": action_form,
            "actions_on_top": self.actions_on_top,
            "actions_on_bottom": self.actions_on_bottom,
            "actions_selection_counter": self.actions_selection_counter,
            "preserved_filters": self.get_preserved_filters(request),
            **(extra_context or {}),
        }

        request.current_app = self.admin_site.name

        return TemplateResponse(
            request,
            self.change_list_template
            or [
                "admin/%s/%s/change_list.html" % (app_label, self.opts.model_name),
                "admin/%s/change_list.html" % app_label,
                "admin/change_list.html",
            ],
            context,
        )
```
### 6 - django/db/models/fields/related_descriptors.py:

Start line: 788, End line: 893

```python
def create_reverse_many_to_one_manager(superclass, rel):

    class RelatedManager(superclass, AltersData):

        add.alters_data = True

        async def aadd(self, *objs, bulk=True):
            return await sync_to_async(self.add)(*objs, bulk=bulk)

        aadd.alters_data = True

        def create(self, **kwargs):
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).create(**kwargs)

        create.alters_data = True

        async def acreate(self, **kwargs):
            return await sync_to_async(self.create)(**kwargs)

        acreate.alters_data = True

        def get_or_create(self, **kwargs):
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).get_or_create(**kwargs)

        get_or_create.alters_data = True

        async def aget_or_create(self, **kwargs):
            return await sync_to_async(self.get_or_create)(**kwargs)

        aget_or_create.alters_data = True

        def update_or_create(self, **kwargs):
            self._check_fk_val()
            kwargs[self.field.name] = self.instance
            db = router.db_for_write(self.model, instance=self.instance)
            return super(RelatedManager, self.db_manager(db)).update_or_create(**kwargs)

        update_or_create.alters_data = True

        async def aupdate_or_create(self, **kwargs):
            return await sync_to_async(self.update_or_create)(**kwargs)

        aupdate_or_create.alters_data = True

        # remove() and clear() are only provided if the ForeignKey can have a
        # value of null.
        if rel.field.null:

            def remove(self, *objs, bulk=True):
                if not objs:
                    return
                self._check_fk_val()
                val = self.field.get_foreign_related_value(self.instance)
                old_ids = set()
                for obj in objs:
                    if not isinstance(obj, self.model):
                        raise TypeError(
                            "'%s' instance expected, got %r"
                            % (
                                self.model._meta.object_name,
                                obj,
                            )
                        )
                    # Is obj actually part of this descriptor set?
                    if self.field.get_local_related_value(obj) == val:
                        old_ids.add(obj.pk)
                    else:
                        raise self.field.remote_field.model.DoesNotExist(
                            "%r is not related to %r." % (obj, self.instance)
                        )
                self._clear(self.filter(pk__in=old_ids), bulk)

            remove.alters_data = True

            async def aremove(self, *objs, bulk=True):
                return await sync_to_async(self.remove)(*objs, bulk=bulk)

            aremove.alters_data = True

            def clear(self, *, bulk=True):
                self._check_fk_val()
                self._clear(self, bulk)

            clear.alters_data = True

            async def aclear(self, *, bulk=True):
                return await sync_to_async(self.clear)(bulk=bulk)

            aclear.alters_data = True

            def _clear(self, queryset, bulk):
                self._remove_prefetched_objects()
                db = router.db_for_write(self.model, instance=self.instance)
                queryset = queryset.using(db)
                if bulk:
                    # `QuerySet.update()` is intrinsically atomic.
                    queryset.update(**{self.field.name: None})
                else:
                    with transaction.atomic(using=db, savepoint=False):
                        for obj in queryset:
                            setattr(obj, self.field.name, None)
                            obj.save(update_fields=[self.field.name])

            _clear.alters_data = True
    # ... other code
```
### 7 - django/contrib/admin/checks.py:

Start line: 954, End line: 978

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links_item(self, obj, field_name, label):
        if field_name not in obj.list_display:
            return [
                checks.Error(
                    "The value of '%s' refers to '%s', which is not defined in "
                    "'list_display'." % (label, field_name),
                    obj=obj.__class__,
                    id="admin.E111",
                )
            ]
        else:
            return []

    def _check_list_filter(self, obj):
        if not isinstance(obj.list_filter, (list, tuple)):
            return must_be(
                "a list or tuple", option="list_filter", obj=obj, id="admin.E112"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_list_filter_item(obj, item, "list_filter[%d]" % index)
                    for index, item in enumerate(obj.list_filter)
                )
            )
```
### 8 - django/contrib/admin/checks.py:

Start line: 217, End line: 264

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields_item(self, obj, field_name, label):
        """
        Check that an item in `autocomplete_fields` is a ForeignKey or a
        ManyToManyField and that the item has a related ModelAdmin with
        search_fields defined.
        """
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E037"
            )
        else:
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be(
                    "a foreign key or a many-to-many field",
                    option=label,
                    obj=obj,
                    id="admin.E038",
                )
            related_admin = obj.admin_site._registry.get(field.remote_field.model)
            if related_admin is None:
                return [
                    checks.Error(
                        'An admin for model "%s" has to be registered '
                        "to be referenced by %s.autocomplete_fields."
                        % (
                            field.remote_field.model.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id="admin.E039",
                    )
                ]
            elif not related_admin.search_fields:
                return [
                    checks.Error(
                        '%s must define "search_fields", because it\'s '
                        "referenced by %s.autocomplete_fields."
                        % (
                            related_admin.__class__.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id="admin.E040",
                    )
                ]
            return []
```
### 9 - django/contrib/admin/options.py:

Start line: 1, End line: 119

```python
import copy
import enum
import json
import re
from functools import partial, update_wrapper
from urllib.parse import quote as urlquote

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
    BaseModelAdminChecks,
    InlineModelAdminChecks,
    ModelAdminChecks,
)
from django.contrib.admin.exceptions import DisallowedModelAdminToField
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
    NestedObjects,
    construct_change_message,
    flatten_fieldsets,
    get_deleted_objects,
    lookup_spawns_duplicates,
    model_format_dict,
    model_ngettext,
    quote,
    unquote,
)
from django.contrib.admin.widgets import AutocompleteSelect, AutocompleteSelectMultiple
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
    FieldDoesNotExist,
    FieldError,
    PermissionDenied,
    ValidationError,
)
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
    BaseInlineFormSet,
    inlineformset_factory,
    modelform_defines_fields,
    modelform_factory,
    modelformset_factory,
)
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
    capfirst,
    format_lazy,
    get_text_list,
    smart_split,
    unescape_string_literal,
)
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView

IS_POPUP_VAR = "_popup"
TO_FIELD_VAR = "_to_field"
IS_FACETS_VAR = "_facets"


class ShowFacets(enum.Enum):
    NEVER = "NEVER"
    ALLOW = "ALLOW"
    ALWAYS = "ALWAYS"


HORIZONTAL, VERTICAL = 1, 2


def get_content_type_for_model(obj):
    # Since this module gets imported in the application's root package,
    # it cannot import models from other applications at the module level.
    from django.contrib.contenttypes.models import ContentType

    return ContentType.objects.get_for_model(obj, for_concrete_model=False)


def get_ul_class(radio_style):
    return "radiolist" if radio_style == VERTICAL else "radiolist inline"


class IncorrectLookupParameters(Exception):
    pass


# Defaults for formfield_overrides. ModelAdmin subclasses can change this
# by adding to ModelAdmin.formfield_overrides.

FORMFIELD_FOR_DBFIELD_DEFAULTS = {
    models.DateTimeField: {
        "form_class": forms.SplitDateTimeField,
        "widget": widgets.AdminSplitDateTime,
    },
    models.DateField: {"widget": widgets.AdminDateWidget},
    models.TimeField: {"widget": widgets.AdminTimeWidget},
    models.TextField: {"widget": widgets.AdminTextareaWidget},
    models.URLField: {"widget": widgets.AdminURLFieldWidget},
    models.IntegerField: {"widget": widgets.AdminIntegerFieldWidget},
    models.BigIntegerField: {"widget": widgets.AdminBigIntegerFieldWidget},
    models.CharField: {"widget": widgets.AdminTextInputWidget},
    models.ImageField: {"widget": widgets.AdminFileWidget},
    models.FileField: {"widget": widgets.AdminFileWidget},
    models.EmailField: {"widget": widgets.AdminEmailInputWidget},
    models.UUIDField: {"widget": widgets.AdminUUIDInputWidget},
}
```
### 10 - django/contrib/admin/options.py:

Start line: 1770, End line: 1872

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField(
                "The field %s cannot be referenced." % to_field
            )

        if request.method == "POST" and "_saveasnew" in request.POST:
            object_id = None

        add = object_id is None

        if add:
            if not self.has_add_permission(request):
                raise PermissionDenied
            obj = None

        else:
            obj = self.get_object(request, unquote(object_id), to_field)

            if request.method == "POST":
                if not self.has_change_permission(request, obj):
                    raise PermissionDenied
            else:
                if not self.has_view_or_change_permission(request, obj):
                    raise PermissionDenied

            if obj is None:
                return self._get_obj_does_not_exist_redirect(
                    request, self.opts, object_id
                )

        fieldsets = self.get_fieldsets(request, obj)
        ModelForm = self.get_form(
            request, obj, change=not add, fields=flatten_fieldsets(fieldsets)
        )
        if request.method == "POST":
            form = ModelForm(request.POST, request.FILES, instance=obj)
            formsets, inline_instances = self._create_formsets(
                request,
                form.instance,
                change=not add,
            )
            form_validated = form.is_valid()
            if form_validated:
                new_object = self.save_form(request, form, change=not add)
            else:
                new_object = form.instance
            if all_valid(formsets) and form_validated:
                self.save_model(request, new_object, form, not add)
                self.save_related(request, form, formsets, not add)
                change_message = self.construct_change_message(
                    request, form, formsets, add
                )
                if add:
                    self.log_addition(request, new_object, change_message)
                    return self.response_add(request, new_object)
                else:
                    self.log_change(request, new_object, change_message)
                    return self.response_change(request, new_object)
            else:
                form_validated = False
        else:
            if add:
                initial = self.get_changeform_initial_data(request)
                form = ModelForm(initial=initial)
                formsets, inline_instances = self._create_formsets(
                    request, form.instance, change=False
                )
            else:
                form = ModelForm(instance=obj)
                formsets, inline_instances = self._create_formsets(
                    request, obj, change=True
                )

        if not add and not self.has_change_permission(request, obj):
            readonly_fields = flatten_fieldsets(fieldsets)
        else:
            readonly_fields = self.get_readonly_fields(request, obj)
        admin_form = helpers.AdminForm(
            form,
            list(fieldsets),
            # Clear prepopulated fields on a view-only form to avoid a crash.
            self.get_prepopulated_fields(request, obj)
            if add or self.has_change_permission(request, obj)
            else {},
            readonly_fields,
            model_admin=self,
        )
        media = self.media + admin_form.media

        inline_formsets = self.get_inline_formsets(
            request, formsets, inline_instances, obj
        )
        for inline_formset in inline_formsets:
            media += inline_formset.media

        if add:
            title = _("Add %s")
        elif self.has_change_permission(request, obj):
            title = _("Change %s")
        else:
            title = _("View %s")
        # ... other code
```
### 11 - django/contrib/admin/checks.py:

Start line: 893, End line: 927

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_item(self, obj, item, label):
        if callable(item):
            return []
        elif hasattr(obj, item):
            return []
        try:
            field = obj.model._meta.get_field(item)
        except FieldDoesNotExist:
            try:
                field = getattr(obj.model, item)
            except AttributeError:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not a "
                        "callable, an attribute of '%s', or an attribute or "
                        "method on '%s'."
                        % (
                            label,
                            item,
                            obj.__class__.__name__,
                            obj.model._meta.label,
                        ),
                        obj=obj.__class__,
                        id="admin.E108",
                    )
                ]
        if isinstance(field, models.ManyToManyField):
            return [
                checks.Error(
                    "The value of '%s' must not be a ManyToManyField." % label,
                    obj=obj.__class__,
                    id="admin.E109",
                )
            ]
        return []
```
### 16 - django/contrib/admin/checks.py:

Start line: 789, End line: 807

```python
class ModelAdminChecks(BaseModelAdminChecks):
    def check(self, admin_obj, **kwargs):
        return [
            *super().check(admin_obj),
            *self._check_save_as(admin_obj),
            *self._check_save_on_top(admin_obj),
            *self._check_inlines(admin_obj),
            *self._check_list_display(admin_obj),
            *self._check_list_display_links(admin_obj),
            *self._check_list_filter(admin_obj),
            *self._check_list_select_related(admin_obj),
            *self._check_list_per_page(admin_obj),
            *self._check_list_max_show_all(admin_obj),
            *self._check_list_editable(admin_obj),
            *self._check_search_fields(admin_obj),
            *self._check_date_hierarchy(admin_obj),
            *self._check_action_permission_methods(admin_obj),
            *self._check_actions_uniqueness(admin_obj),
        ]
```
### 20 - django/contrib/admin/checks.py:

Start line: 1039, End line: 1070

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_select_related(self, obj):
        """Check that list_select_related is a boolean, a list or a tuple."""

        if not isinstance(obj.list_select_related, (bool, list, tuple)):
            return must_be(
                "a boolean, tuple or list",
                option="list_select_related",
                obj=obj,
                id="admin.E117",
            )
        else:
            return []

    def _check_list_per_page(self, obj):
        """Check that list_per_page is an integer."""

        if not isinstance(obj.list_per_page, int):
            return must_be(
                "an integer", option="list_per_page", obj=obj, id="admin.E118"
            )
        else:
            return []

    def _check_list_max_show_all(self, obj):
        """Check that list_max_show_all is an integer."""

        if not isinstance(obj.list_max_show_all, int):
            return must_be(
                "an integer", option="list_max_show_all", obj=obj, id="admin.E119"
            )
        else:
            return []
```
### 31 - django/contrib/admin/checks.py:

Start line: 55, End line: 173

```python
def check_dependencies(**kwargs):
    """
    Check that the admin's dependencies are correctly installed.
    """
    from django.contrib.admin.sites import all_sites

    if not apps.is_installed("django.contrib.admin"):
        return []
    errors = []
    app_dependencies = (
        ("django.contrib.contenttypes", 401),
        ("django.contrib.auth", 405),
        ("django.contrib.messages", 406),
    )
    for app_name, error_code in app_dependencies:
        if not apps.is_installed(app_name):
            errors.append(
                checks.Error(
                    "'%s' must be in INSTALLED_APPS in order to use the admin "
                    "application." % app_name,
                    id="admin.E%d" % error_code,
                )
            )
    for engine in engines.all():
        if isinstance(engine, DjangoTemplates):
            django_templates_instance = engine.engine
            break
    else:
        django_templates_instance = None
    if not django_templates_instance:
        errors.append(
            checks.Error(
                "A 'django.template.backends.django.DjangoTemplates' instance "
                "must be configured in TEMPLATES in order to use the admin "
                "application.",
                id="admin.E403",
            )
        )
    else:
        if (
            "django.contrib.auth.context_processors.auth"
            not in django_templates_instance.context_processors
            and _contains_subclass(
                "django.contrib.auth.backends.ModelBackend",
                settings.AUTHENTICATION_BACKENDS,
            )
        ):
            errors.append(
                checks.Error(
                    "'django.contrib.auth.context_processors.auth' must be "
                    "enabled in DjangoTemplates (TEMPLATES) if using the default "
                    "auth backend in order to use the admin application.",
                    id="admin.E402",
                )
            )
        if (
            "django.contrib.messages.context_processors.messages"
            not in django_templates_instance.context_processors
        ):
            errors.append(
                checks.Error(
                    "'django.contrib.messages.context_processors.messages' must "
                    "be enabled in DjangoTemplates (TEMPLATES) in order to use "
                    "the admin application.",
                    id="admin.E404",
                )
            )
        sidebar_enabled = any(site.enable_nav_sidebar for site in all_sites)
        if (
            sidebar_enabled
            and "django.template.context_processors.request"
            not in django_templates_instance.context_processors
        ):
            errors.append(
                checks.Warning(
                    "'django.template.context_processors.request' must be enabled "
                    "in DjangoTemplates (TEMPLATES) in order to use the admin "
                    "navigation sidebar.",
                    id="admin.W411",
                )
            )

    if not _contains_subclass(
        "django.contrib.auth.middleware.AuthenticationMiddleware", settings.MIDDLEWARE
    ):
        errors.append(
            checks.Error(
                "'django.contrib.auth.middleware.AuthenticationMiddleware' must "
                "be in MIDDLEWARE in order to use the admin application.",
                id="admin.E408",
            )
        )
    if not _contains_subclass(
        "django.contrib.messages.middleware.MessageMiddleware", settings.MIDDLEWARE
    ):
        errors.append(
            checks.Error(
                "'django.contrib.messages.middleware.MessageMiddleware' must "
                "be in MIDDLEWARE in order to use the admin application.",
                id="admin.E409",
            )
        )
    if not _contains_subclass(
        "django.contrib.sessions.middleware.SessionMiddleware", settings.MIDDLEWARE
    ):
        errors.append(
            checks.Error(
                "'django.contrib.sessions.middleware.SessionMiddleware' must "
                "be in MIDDLEWARE in order to use the admin application.",
                hint=(
                    "Insert "
                    "'django.contrib.sessions.middleware.SessionMiddleware' "
                    "before "
                    "'django.contrib.auth.middleware.AuthenticationMiddleware'."
                ),
                id="admin.E410",
            )
        )
    return errors
```
### 32 - django/contrib/admin/checks.py:

Start line: 176, End line: 192

```python
class BaseModelAdminChecks:
    def check(self, admin_obj, **kwargs):
        return [
            *self._check_autocomplete_fields(admin_obj),
            *self._check_raw_id_fields(admin_obj),
            *self._check_fields(admin_obj),
            *self._check_fieldsets(admin_obj),
            *self._check_exclude(admin_obj),
            *self._check_form(admin_obj),
            *self._check_filter_vertical(admin_obj),
            *self._check_filter_horizontal(admin_obj),
            *self._check_radio_fields(admin_obj),
            *self._check_prepopulated_fields(admin_obj),
            *self._check_view_on_site_url(admin_obj),
            *self._check_ordering(admin_obj),
            *self._check_readonly_fields(admin_obj),
        ]
```
### 33 - django/contrib/admin/checks.py:

Start line: 878, End line: 891

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display(self, obj):
        """Check that list_display only contains fields or usable attributes."""

        if not isinstance(obj.list_display, (list, tuple)):
            return must_be(
                "a list or tuple", option="list_display", obj=obj, id="admin.E107"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_list_display_item(obj, item, "list_display[%d]" % index)
                    for index, item in enumerate(obj.list_display)
                )
            )
```
### 42 - django/contrib/admin/checks.py:

Start line: 1274, End line: 1318

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_relation(self, obj, parent_model):
        try:
            _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        except ValueError as e:
            return [checks.Error(e.args[0], obj=obj.__class__, id="admin.E202")]
        else:
            return []

    def _check_extra(self, obj):
        """Check that extra is an integer."""

        if not isinstance(obj.extra, int):
            return must_be("an integer", option="extra", obj=obj, id="admin.E203")
        else:
            return []

    def _check_max_num(self, obj):
        """Check that max_num is an integer."""

        if obj.max_num is None:
            return []
        elif not isinstance(obj.max_num, int):
            return must_be("an integer", option="max_num", obj=obj, id="admin.E204")
        else:
            return []

    def _check_min_num(self, obj):
        """Check that min_num is an integer."""

        if obj.min_num is None:
            return []
        elif not isinstance(obj.min_num, int):
            return must_be("an integer", option="min_num", obj=obj, id="admin.E205")
        else:
            return []

    def _check_formset(self, obj):
        """Check formset is a subclass of BaseModelFormSet."""

        if not _issubclass(obj.formset, BaseModelFormSet):
            return must_inherit_from(
                parent="BaseModelFormSet", option="formset", obj=obj, id="admin.E206"
            )
        else:
            return []
```
### 44 - django/contrib/admin/checks.py:

Start line: 840, End line: 876

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_inlines_item(self, obj, inline, label):
        """Check one inline model admin."""
        try:
            inline_label = inline.__module__ + "." + inline.__name__
        except AttributeError:
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % obj,
                    obj=obj.__class__,
                    id="admin.E104",
                )
            ]

        from django.contrib.admin.options import InlineModelAdmin

        if not _issubclass(inline, InlineModelAdmin):
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % inline_label,
                    obj=obj.__class__,
                    id="admin.E104",
                )
            ]
        elif not inline.model:
            return [
                checks.Error(
                    "'%s' must have a 'model' attribute." % inline_label,
                    obj=obj.__class__,
                    id="admin.E105",
                )
            ]
        elif not _issubclass(inline.model, models.Model):
            return must_be(
                "a Model", option="%s.model" % inline_label, obj=obj, id="admin.E106"
            )
        else:
            return inline(obj.model, obj.admin_site).check()
```
### 49 - django/contrib/admin/checks.py:

Start line: 284, End line: 312

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields_item(self, obj, field_name, label):
        """Check an item of `raw_id_fields`, i.e. check that field named
        `field_name` exists in model `model` and is a ForeignKey or a
        ManyToManyField."""

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E002"
            )
        else:
            # Using attname is not supported.
            if field.name != field_name:
                return refer_to_missing_field(
                    field=field_name,
                    option=label,
                    obj=obj,
                    id="admin.E002",
                )
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be(
                    "a foreign key or a many-to-many field",
                    option=label,
                    obj=obj,
                    id="admin.E003",
                )
            else:
                return []
```
### 52 - django/contrib/admin/checks.py:

Start line: 1243, End line: 1272

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_exclude_of_parent_model(self, obj, parent_model):
        # Do not perform more specific checks if the base checks result in an
        # error.
        errors = super()._check_exclude(obj)
        if errors:
            return []

        # Skip if `fk_name` is invalid.
        if self._check_relation(obj, parent_model):
            return []

        if obj.exclude is None:
            return []

        fk = _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        if fk.name in obj.exclude:
            return [
                checks.Error(
                    "Cannot exclude the field '%s', because it is the foreign key "
                    "to the parent model '%s'."
                    % (
                        fk.name,
                        parent_model._meta.label,
                    ),
                    obj=obj.__class__,
                    id="admin.E201",
                )
            ]
        else:
            return []
```
### 60 - django/contrib/admin/checks.py:

Start line: 929, End line: 952

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links(self, obj):
        """Check that list_display_links is a unique subset of list_display."""
        from django.contrib.admin.options import ModelAdmin

        if obj.list_display_links is None:
            return []
        elif not isinstance(obj.list_display_links, (list, tuple)):
            return must_be(
                "a list, a tuple, or None",
                option="list_display_links",
                obj=obj,
                id="admin.E110",
            )
        # Check only if ModelAdmin.get_list_display() isn't overridden.
        elif obj.get_list_display.__func__ is ModelAdmin.get_list_display:
            return list(
                chain.from_iterable(
                    self._check_list_display_links_item(
                        obj, field_name, "list_display_links[%d]" % index
                    )
                    for index, field_name in enumerate(obj.list_display_links)
                )
            )
        return []
```
### 63 - django/contrib/admin/checks.py:

Start line: 430, End line: 458

```python
class BaseModelAdminChecks:

    def _check_field_spec_item(self, obj, field_name, label):
        if field_name in obj.readonly_fields:
            # Stuff can be put in fields that isn't actually a model field if
            # it's in readonly_fields, readonly_fields will handle the
            # validation of such things.
            return []
        else:
            try:
                field = obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                # If we can't find a field on the model that matches, it could
                # be an extra field on the form.
                return []
            else:
                if (
                    isinstance(field, models.ManyToManyField)
                    and not field.remote_field.through._meta.auto_created
                ):
                    return [
                        checks.Error(
                            "The value of '%s' cannot include the ManyToManyField "
                            "'%s', because that field manually specifies a "
                            "relationship model." % (label, field_name),
                            obj=obj.__class__,
                            id="admin.E013",
                        )
                    ]
                else:
                    return []
```
### 69 - django/contrib/admin/checks.py:

Start line: 761, End line: 786

```python
class BaseModelAdminChecks:

    def _check_readonly_fields_item(self, obj, field_name, label):
        if callable(field_name):
            return []
        elif hasattr(obj, field_name):
            return []
        elif hasattr(obj.model, field_name):
            return []
        else:
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return [
                    checks.Error(
                        "The value of '%s' is not a callable, an attribute of "
                        "'%s', or an attribute of '%s'."
                        % (
                            label,
                            obj.__class__.__name__,
                            obj.model._meta.label,
                        ),
                        obj=obj.__class__,
                        id="admin.E035",
                    )
                ]
            else:
                return []
```
### 71 - django/contrib/admin/checks.py:

Start line: 1230, End line: 1241

```python
class InlineModelAdminChecks(BaseModelAdminChecks):
    def check(self, inline_obj, **kwargs):
        parent_model = inline_obj.parent_model
        return [
            *super().check(inline_obj),
            *self._check_relation(inline_obj, parent_model),
            *self._check_exclude_of_parent_model(inline_obj, parent_model),
            *self._check_extra(inline_obj),
            *self._check_max_num(inline_obj),
            *self._check_min_num(inline_obj),
            *self._check_formset(inline_obj),
        ]
```
### 76 - django/contrib/admin/checks.py:

Start line: 521, End line: 537

```python
class BaseModelAdminChecks:

    def _check_filter_item(self, obj, field_name, label):
        """Check one item of `filter_vertical` or `filter_horizontal`, i.e.
        check that given field exists and is a ManyToManyField."""

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E019"
            )
        else:
            if not field.many_to_many:
                return must_be(
                    "a many-to-many field", option=label, obj=obj, id="admin.E020"
                )
            else:
                return []
```
### 77 - django/contrib/admin/checks.py:

Start line: 742, End line: 759

```python
class BaseModelAdminChecks:

    def _check_readonly_fields(self, obj):
        """Check that readonly_fields refers to proper attribute or field."""

        if obj.readonly_fields == ():
            return []
        elif not isinstance(obj.readonly_fields, (list, tuple)):
            return must_be(
                "a list or tuple", option="readonly_fields", obj=obj, id="admin.E034"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_readonly_fields_item(
                        obj, field_name, "readonly_fields[%d]" % index
                    )
                    for index, field_name in enumerate(obj.readonly_fields)
                )
            )
```
### 78 - django/contrib/admin/checks.py:

Start line: 194, End line: 215

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields(self, obj):
        """
        Check that `autocomplete_fields` is a list or tuple of model fields.
        """
        if not isinstance(obj.autocomplete_fields, (list, tuple)):
            return must_be(
                "a list or tuple",
                option="autocomplete_fields",
                obj=obj,
                id="admin.E036",
            )
        else:
            return list(
                chain.from_iterable(
                    [
                        self._check_autocomplete_fields_item(
                            obj, field_name, "autocomplete_fields[%d]" % index
                        )
                        for index, field_name in enumerate(obj.autocomplete_fields)
                    ]
                )
            )
```
### 80 - django/contrib/admin/checks.py:

Start line: 1321, End line: 1350

```python
def must_be(type, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' must be %s." % (option, type),
            obj=obj.__class__,
            id=id,
        ),
    ]


def must_inherit_from(parent, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' must inherit from '%s'." % (option, parent),
            obj=obj.__class__,
            id=id,
        ),
    ]


def refer_to_missing_field(field, option, obj, id):
    return [
        checks.Error(
            "The value of '%s' refers to '%s', which is not a field of '%s'."
            % (option, field, obj.model._meta.label),
            obj=obj.__class__,
            id=id,
        ),
    ]
```
### 82 - django/contrib/admin/checks.py:

Start line: 980, End line: 1037

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_filter_item(self, obj, item, label):
        """
        Check one item of `list_filter`, i.e. check if it is one of three options:
        1. 'field' -- a basic field filter, possibly w/ relationships (e.g.
           'field__rel')
        2. ('field', SomeFieldListFilter) - a field-based list filter class
        3. SomeListFilter - a non-field list filter class
        """
        from django.contrib.admin import FieldListFilter, ListFilter

        if callable(item) and not isinstance(item, models.Field):
            # If item is option 3, it should be a ListFilter...
            if not _issubclass(item, ListFilter):
                return must_inherit_from(
                    parent="ListFilter", option=label, obj=obj, id="admin.E113"
                )
            # ...  but not a FieldListFilter.
            elif issubclass(item, FieldListFilter):
                return [
                    checks.Error(
                        "The value of '%s' must not inherit from 'FieldListFilter'."
                        % label,
                        obj=obj.__class__,
                        id="admin.E114",
                    )
                ]
            else:
                return []
        elif isinstance(item, (tuple, list)):
            # item is option #2
            field, list_filter_class = item
            if not _issubclass(list_filter_class, FieldListFilter):
                return must_inherit_from(
                    parent="FieldListFilter",
                    option="%s[1]" % label,
                    obj=obj,
                    id="admin.E115",
                )
            else:
                return []
        else:
            # item is option #1
            field = item

            # Validate the field string
            try:
                get_fields_from_path(obj.model, field)
            except (NotRelationField, FieldDoesNotExist):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which does not refer to a "
                        "Field." % (label, field),
                        obj=obj.__class__,
                        id="admin.E116",
                    )
                ]
            else:
                return []
```
### 85 - django/contrib/admin/checks.py:

Start line: 704, End line: 740

```python
class BaseModelAdminChecks:

    def _check_ordering_item(self, obj, field_name, label):
        """Check that `ordering` refers to existing fields."""
        if isinstance(field_name, (Combinable, models.OrderBy)):
            if not isinstance(field_name, models.OrderBy):
                field_name = field_name.asc()
            if isinstance(field_name.expression, models.F):
                field_name = field_name.expression.name
            else:
                return []
        if field_name == "?" and len(obj.ordering) != 1:
            return [
                checks.Error(
                    "The value of 'ordering' has the random ordering marker '?', "
                    "but contains other fields as well.",
                    hint='Either remove the "?", or remove the other fields.',
                    obj=obj.__class__,
                    id="admin.E032",
                )
            ]
        elif field_name == "?":
            return []
        elif LOOKUP_SEP in field_name:
            # Skip ordering in the format field1__field2 (FIXME: checking
            # this format would be nice, but it's a little fiddly).
            return []
        else:
            field_name = field_name.removeprefix("-")
            if field_name == "pk":
                return []
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return refer_to_missing_field(
                    field=field_name, option=label, obj=obj, id="admin.E033"
                )
            else:
                return []
```
### 89 - django/contrib/admin/checks.py:

Start line: 266, End line: 282

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields(self, obj):
        """Check that `raw_id_fields` only contains field names that are listed
        on the model."""

        if not isinstance(obj.raw_id_fields, (list, tuple)):
            return must_be(
                "a list or tuple", option="raw_id_fields", obj=obj, id="admin.E001"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_raw_id_fields_item(
                        obj, field_name, "raw_id_fields[%d]" % index
                    )
                    for index, field_name in enumerate(obj.raw_id_fields)
                )
            )
```
### 91 - django/contrib/admin/checks.py:

Start line: 1144, End line: 1180

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_search_fields(self, obj):
        """Check search_fields is a sequence."""

        if not isinstance(obj.search_fields, (list, tuple)):
            return must_be(
                "a list or tuple", option="search_fields", obj=obj, id="admin.E126"
            )
        else:
            return []

    def _check_date_hierarchy(self, obj):
        """Check that date_hierarchy refers to DateField or DateTimeField."""

        if obj.date_hierarchy is None:
            return []
        else:
            try:
                field = get_fields_from_path(obj.model, obj.date_hierarchy)[-1]
            except (NotRelationField, FieldDoesNotExist):
                return [
                    checks.Error(
                        "The value of 'date_hierarchy' refers to '%s', which "
                        "does not refer to a Field." % obj.date_hierarchy,
                        obj=obj.__class__,
                        id="admin.E127",
                    )
                ]
            else:
                if not isinstance(field, (models.DateField, models.DateTimeField)):
                    return must_be(
                        "a DateField or DateTimeField",
                        option="date_hierarchy",
                        obj=obj,
                        id="admin.E128",
                    )
                else:
                    return []
```
### 93 - django/contrib/admin/checks.py:

Start line: 673, End line: 702

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_value_item(self, obj, field_name, label):
        """For `prepopulated_fields` equal to {"slug": ("title",)},
        `field_name` is "title"."""

        try:
            obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E030"
            )
        else:
            return []

    def _check_ordering(self, obj):
        """Check that ordering refers to existing fields or is random."""

        # ordering = None
        if obj.ordering is None:  # The default value is None
            return []
        elif not isinstance(obj.ordering, (list, tuple)):
            return must_be(
                "a list or tuple", option="ordering", obj=obj, id="admin.E031"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_ordering_item(obj, field_name, "ordering[%d]" % index)
                    for index, field_name in enumerate(obj.ordering)
                )
            )
```
### 108 - django/contrib/admin/checks.py:

Start line: 556, End line: 578

```python
class BaseModelAdminChecks:

    def _check_radio_fields_key(self, obj, field_name, label):
        """Check that a key of `radio_fields` dictionary is name of existing
        field and that the field is a ForeignKey or has `choices` defined."""

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(
                field=field_name, option=label, obj=obj, id="admin.E022"
            )
        else:
            if not (isinstance(field, models.ForeignKey) or field.choices):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not an "
                        "instance of ForeignKey, and does not have a 'choices' "
                        "definition." % (label, field_name),
                        obj=obj.__class__,
                        id="admin.E023",
                    )
                ]
            else:
                return []
```
### 124 - django/contrib/admin/checks.py:

Start line: 825, End line: 838

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_inlines(self, obj):
        """Check all inline model admin classes."""

        if not isinstance(obj.inlines, (list, tuple)):
            return must_be(
                "a list or tuple", option="inlines", obj=obj, id="admin.E103"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_inlines_item(obj, item, "inlines[%d]" % index)
                    for index, item in enumerate(obj.inlines)
                )
            )
```
### 144 - django/contrib/admin/checks.py:

Start line: 505, End line: 519

```python
class BaseModelAdminChecks:

    def _check_filter_horizontal(self, obj):
        """Check that filter_horizontal is a sequence of field names."""
        if not isinstance(obj.filter_horizontal, (list, tuple)):
            return must_be(
                "a list or tuple", option="filter_horizontal", obj=obj, id="admin.E018"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_filter_item(
                        obj, field_name, "filter_horizontal[%d]" % index
                    )
                    for index, field_name in enumerate(obj.filter_horizontal)
                )
            )
```
### 147 - django/contrib/admin/checks.py:

Start line: 480, End line: 503

```python
class BaseModelAdminChecks:

    def _check_form(self, obj):
        """Check that form subclasses BaseModelForm."""
        if not _issubclass(obj.form, BaseModelForm):
            return must_inherit_from(
                parent="BaseModelForm", option="form", obj=obj, id="admin.E016"
            )
        else:
            return []

    def _check_filter_vertical(self, obj):
        """Check that filter_vertical is a sequence of field names."""
        if not isinstance(obj.filter_vertical, (list, tuple)):
            return must_be(
                "a list or tuple", option="filter_vertical", obj=obj, id="admin.E017"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_filter_item(
                        obj, field_name, "filter_vertical[%d]" % index
                    )
                    for index, field_name in enumerate(obj.filter_vertical)
                )
            )
```
### 167 - django/contrib/admin/checks.py:

Start line: 539, End line: 554

```python
class BaseModelAdminChecks:

    def _check_radio_fields(self, obj):
        """Check that `radio_fields` is a dictionary."""
        if not isinstance(obj.radio_fields, dict):
            return must_be(
                "a dictionary", option="radio_fields", obj=obj, id="admin.E021"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_radio_fields_key(obj, field_name, "radio_fields")
                    + self._check_radio_fields_value(
                        obj, val, 'radio_fields["%s"]' % field_name
                    )
                    for field_name, val in obj.radio_fields.items()
                )
            )
```
### 174 - django/contrib/admin/checks.py:

Start line: 580, End line: 608

```python
class BaseModelAdminChecks:

    def _check_radio_fields_value(self, obj, val, label):
        """Check type of a value of `radio_fields` dictionary."""

        from django.contrib.admin.options import HORIZONTAL, VERTICAL

        if val not in (HORIZONTAL, VERTICAL):
            return [
                checks.Error(
                    "The value of '%s' must be either admin.HORIZONTAL or "
                    "admin.VERTICAL." % label,
                    obj=obj.__class__,
                    id="admin.E024",
                )
            ]
        else:
            return []

    def _check_view_on_site_url(self, obj):
        if not callable(obj.view_on_site) and not isinstance(obj.view_on_site, bool):
            return [
                checks.Error(
                    "The value of 'view_on_site' must be a callable or a boolean "
                    "value.",
                    obj=obj.__class__,
                    id="admin.E025",
                )
            ]
        else:
            return []
```
