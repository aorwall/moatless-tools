# django__django-16816

| **django/django** | `191f6a9a4586b5e5f79f4f42f190e7ad4bbacc84` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 5228 |
| **Any found context length** | 5228 |
| **Avg pos** | 10.0 |
| **Min pos** | 10 |
| **Max pos** | 10 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -916,9 +916,10 @@ def _check_list_display_item(self, obj, item, label):
                         id="admin.E108",
                     )
                 ]
-        if isinstance(field, models.ManyToManyField) or (
-            getattr(field, "rel", None) and field.rel.field.many_to_one
-        ):
+        if (
+            getattr(field, "is_relation", False)
+            and (field.many_to_many or field.one_to_many)
+        ) or (getattr(field, "rel", None) and field.rel.field.many_to_one):
             return [
                 checks.Error(
                     f"The value of '{label}' must not be a many-to-many field or a "

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/checks.py | 919 | 921 | 10 | 2 | 5228


## Problem Statement

```
Error E108 does not cover some cases
Description
	 
		(last modified by Baha Sdtbekov)
	 
I have two models, Question and Choice. And if I write list_display = ["choice"] in QuestionAdmin, I get no errors.
But when I visit /admin/polls/question/, the following trace is returned:
Internal Server Error: /admin/polls/question/
Traceback (most recent call last):
 File "/some/path/django/contrib/admin/utils.py", line 334, in label_for_field
	field = _get_non_gfk_field(model._meta, name)
 File "/some/path/django/contrib/admin/utils.py", line 310, in _get_non_gfk_field
	raise FieldDoesNotExist()
django.core.exceptions.FieldDoesNotExist
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
 File "/some/path/django/core/handlers/exception.py", line 55, in inner
	response = get_response(request)
 File "/some/path/django/core/handlers/base.py", line 220, in _get_response
	response = response.render()
 File "/some/path/django/template/response.py", line 111, in render
	self.content = self.rendered_content
 File "/some/path/django/template/response.py", line 89, in rendered_content
	return template.render(context, self._request)
 File "/some/path/django/template/backends/django.py", line 61, in render
	return self.template.render(context)
 File "/some/path/django/template/base.py", line 175, in render
	return self._render(context)
 File "/some/path/django/template/base.py", line 167, in _render
	return self.nodelist.render(context)
 File "/some/path/django/template/base.py", line 1005, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 1005, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 966, in render_annotated
	return self.render(context)
 File "/some/path/django/template/loader_tags.py", line 157, in render
	return compiled_parent._render(context)
 File "/some/path/django/template/base.py", line 167, in _render
	return self.nodelist.render(context)
 File "/some/path/django/template/base.py", line 1005, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 1005, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 966, in render_annotated
	return self.render(context)
 File "/some/path/django/template/loader_tags.py", line 157, in render
	return compiled_parent._render(context)
 File "/some/path/django/template/base.py", line 167, in _render
	return self.nodelist.render(context)
 File "/some/path/django/template/base.py", line 1005, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 1005, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 966, in render_annotated
	return self.render(context)
 File "/some/path/django/template/loader_tags.py", line 63, in render
	result = block.nodelist.render(context)
 File "/some/path/django/template/base.py", line 1005, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 1005, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 966, in render_annotated
	return self.render(context)
 File "/some/path/django/template/loader_tags.py", line 63, in render
	result = block.nodelist.render(context)
 File "/some/path/django/template/base.py", line 1005, in render
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 1005, in <listcomp>
	return SafeString("".join([node.render_annotated(context) for node in self]))
 File "/some/path/django/template/base.py", line 966, in render_annotated
	return self.render(context)
 File "/some/path/django/contrib/admin/templatetags/base.py", line 45, in render
	return super().render(context)
 File "/some/path/django/template/library.py", line 258, in render
	_dict = self.func(*resolved_args, **resolved_kwargs)
 File "/some/path/django/contrib/admin/templatetags/admin_list.py", line 326, in result_list
	headers = list(result_headers(cl))
 File "/some/path/django/contrib/admin/templatetags/admin_list.py", line 90, in result_headers
	text, attr = label_for_field(
 File "/some/path/django/contrib/admin/utils.py", line 362, in label_for_field
	raise AttributeError(message)
AttributeError: Unable to lookup 'choice' on Question or QuestionAdmin
[24/Apr/2023 15:43:32] "GET /admin/polls/question/ HTTP/1.1" 500 349913
I suggest that error E108 be updated to cover this case as well
For reproduce see ​github

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/options.py | 2024 | 2115| 784 | 784 | 19382 | 
| 2 | **2 django/contrib/admin/checks.py** | 1093 | 1145| 430 | 1214 | 28937 | 
| 3 | 2 django/contrib/admin/options.py | 1 | 119| 796 | 2010 | 28937 | 
| 4 | 2 django/contrib/admin/options.py | 1935 | 2023| 676 | 2686 | 28937 | 
| 5 | 2 django/contrib/admin/options.py | 1770 | 1872| 780 | 3466 | 28937 | 
| 6 | 3 django/contrib/admin/filters.py | 24 | 87| 418 | 3884 | 34663 | 
| 7 | **3 django/contrib/admin/checks.py** | 957 | 981| 197 | 4081 | 34663 | 
| 8 | 3 django/contrib/admin/options.py | 1278 | 1340| 511 | 4592 | 34663 | 
| 9 | 4 django/contrib/admin/templatetags/admin_list.py | 459 | 531| 385 | 4977 | 38469 | 
| **-> 10 <-** | **4 django/contrib/admin/checks.py** | 893 | 930| 251 | 5228 | 38469 | 
| 11 | 4 django/contrib/admin/options.py | 1873 | 1904| 303 | 5531 | 38469 | 
| 12 | 4 django/contrib/admin/templatetags/admin_list.py | 199 | 295| 817 | 6348 | 38469 | 
| 13 | 4 django/contrib/admin/templatetags/admin_list.py | 298 | 352| 351 | 6699 | 38469 | 
| 14 | 5 django/contrib/admin/utils.py | 405 | 451| 402 | 7101 | 42859 | 
| 15 | 5 django/contrib/admin/templatetags/admin_list.py | 84 | 177| 831 | 7932 | 42859 | 
| 16 | 6 django/contrib/admin/views/main.py | 592 | 624| 227 | 8159 | 47732 | 
| 17 | 6 django/contrib/admin/views/main.py | 307 | 342| 312 | 8471 | 47732 | 
| 18 | 6 django/contrib/admin/options.py | 2488 | 2523| 315 | 8786 | 47732 | 
| 19 | 6 django/contrib/admin/options.py | 441 | 506| 550 | 9336 | 47732 | 
| 20 | **6 django/contrib/admin/checks.py** | 878 | 891| 121 | 9457 | 47732 | 
| 21 | 6 django/contrib/admin/templatetags/admin_list.py | 1 | 33| 194 | 9651 | 47732 | 
| 22 | 6 django/contrib/admin/filters.py | 388 | 417| 251 | 9902 | 47732 | 
| 23 | 7 django/db/migrations/questioner.py | 109 | 124| 135 | 10037 | 50428 | 
| 24 | 7 django/contrib/admin/options.py | 364 | 440| 531 | 10568 | 50428 | 
| 25 | 7 django/contrib/admin/views/main.py | 1 | 64| 383 | 10951 | 50428 | 
| 26 | 8 django/contrib/admin/models.py | 1 | 21| 123 | 11074 | 51621 | 
| 27 | 8 django/contrib/admin/options.py | 1432 | 1529| 710 | 11784 | 51621 | 
| 28 | **8 django/contrib/admin/checks.py** | 983 | 1040| 457 | 12241 | 51621 | 
| 29 | 8 django/contrib/admin/options.py | 1559 | 1626| 586 | 12827 | 51621 | 
| 30 | 8 django/contrib/admin/options.py | 2194 | 2252| 444 | 13271 | 51621 | 
| 31 | 8 django/contrib/admin/templatetags/admin_list.py | 180 | 196| 140 | 13411 | 51621 | 
| 32 | 8 django/contrib/admin/options.py | 1531 | 1557| 233 | 13644 | 51621 | 
| 33 | 8 django/contrib/admin/filters.py | 220 | 280| 541 | 14185 | 51621 | 
| 34 | **8 django/contrib/admin/checks.py** | 789 | 807| 183 | 14368 | 51621 | 
| 35 | **8 django/contrib/admin/checks.py** | 840 | 876| 268 | 14636 | 51621 | 
| 36 | 9 django/contrib/admin/helpers.py | 1 | 36| 209 | 14845 | 55233 | 
| 37 | 9 django/contrib/admin/options.py | 224 | 241| 174 | 15019 | 55233 | 
| 38 | 9 django/contrib/admin/filters.py | 419 | 455| 302 | 15321 | 55233 | 
| 39 | **9 django/contrib/admin/checks.py** | 1042 | 1073| 229 | 15550 | 55233 | 
| 40 | 9 django/contrib/admin/utils.py | 454 | 483| 200 | 15750 | 55233 | 
| 41 | 9 django/contrib/admin/views/main.py | 175 | 270| 828 | 16578 | 55233 | 
| 42 | 10 django/forms/fields.py | 867 | 930| 435 | 17013 | 65025 | 
| 43 | **10 django/contrib/admin/checks.py** | 1324 | 1353| 176 | 17189 | 65025 | 
| 44 | 10 django/contrib/admin/options.py | 683 | 698| 123 | 17312 | 65025 | 
| 45 | 10 django/contrib/admin/options.py | 508 | 552| 348 | 17660 | 65025 | 
| 46 | 10 django/contrib/admin/views/main.py | 271 | 287| 158 | 17818 | 65025 | 
| 47 | 11 django/contrib/admin/templatetags/base.py | 32 | 46| 138 | 17956 | 65340 | 
| 48 | 11 django/contrib/admin/options.py | 1161 | 1181| 201 | 18157 | 65340 | 
| 49 | 11 django/contrib/admin/views/main.py | 528 | 590| 504 | 18661 | 65340 | 
| 50 | 12 django/contrib/admin/views/autocomplete.py | 66 | 123| 425 | 19086 | 66181 | 
| 51 | **12 django/contrib/admin/checks.py** | 556 | 578| 194 | 19280 | 66181 | 
| 52 | 12 django/contrib/admin/options.py | 1699 | 1735| 337 | 19617 | 66181 | 
| 53 | 12 django/contrib/admin/helpers.py | 39 | 96| 322 | 19939 | 66181 | 
| 54 | 12 django/forms/fields.py | 960 | 1003| 307 | 20246 | 66181 | 
| 55 | 12 django/contrib/admin/options.py | 1081 | 1102| 153 | 20399 | 66181 | 
| 56 | 12 django/db/migrations/questioner.py | 291 | 342| 367 | 20766 | 66181 | 
| 57 | 13 django/contrib/admin/__init__.py | 1 | 53| 292 | 21058 | 66473 | 
| 58 | 13 django/contrib/admin/options.py | 982 | 995| 123 | 21181 | 66473 | 
| 59 | **13 django/contrib/admin/checks.py** | 176 | 192| 155 | 21336 | 66473 | 
| 60 | 13 django/contrib/admin/options.py | 2254 | 2303| 450 | 21786 | 66473 | 
| 61 | 13 django/contrib/admin/filters.py | 555 | 592| 371 | 22157 | 66473 | 
| 62 | 13 django/contrib/admin/options.py | 121 | 154| 235 | 22392 | 66473 | 
| 63 | 14 django/forms/models.py | 1529 | 1563| 254 | 22646 | 78746 | 
| 64 | 14 django/contrib/admin/options.py | 833 | 868| 251 | 22897 | 78746 | 
| 65 | 14 django/contrib/admin/options.py | 700 | 740| 280 | 23177 | 78746 | 
| 66 | **14 django/contrib/admin/checks.py** | 1277 | 1321| 343 | 23520 | 78746 | 
| 67 | 14 django/contrib/admin/options.py | 2400 | 2457| 465 | 23985 | 78746 | 
| 68 | 14 django/contrib/admin/options.py | 1342 | 1430| 689 | 24674 | 78746 | 
| 69 | **14 django/contrib/admin/checks.py** | 761 | 786| 158 | 24832 | 78746 | 
| 70 | **14 django/contrib/admin/checks.py** | 932 | 955| 195 | 25027 | 78746 | 
| 71 | 15 django/forms/utils.py | 200 | 207| 133 | 25160 | 80394 | 
| 72 | **15 django/contrib/admin/checks.py** | 194 | 215| 141 | 25301 | 80394 | 
| 73 | 15 django/contrib/admin/options.py | 1676 | 1697| 133 | 25434 | 80394 | 
| 74 | **15 django/contrib/admin/checks.py** | 673 | 702| 241 | 25675 | 80394 | 
| 75 | **15 django/contrib/admin/checks.py** | 217 | 264| 333 | 26008 | 80394 | 
| 76 | 16 django/contrib/admin/sites.py | 570 | 597| 196 | 26204 | 84891 | 
| 77 | 17 django/db/models/fields/__init__.py | 310 | 380| 462 | 26666 | 103895 | 
| 78 | 17 django/contrib/admin/filters.py | 282 | 316| 311 | 26977 | 103895 | 
| 79 | 18 django/contrib/admin/templatetags/admin_modify.py | 61 | 112| 414 | 27391 | 104939 | 
| 80 | 18 django/contrib/admin/helpers.py | 527 | 553| 199 | 27590 | 104939 | 
| 81 | 19 django/contrib/admindocs/views.py | 213 | 297| 615 | 28205 | 108427 | 
| 82 | 20 django/db/models/base.py | 1398 | 1428| 220 | 28425 | 127154 | 
| 83 | **20 django/contrib/admin/checks.py** | 430 | 458| 224 | 28649 | 127154 | 
| 84 | **20 django/contrib/admin/checks.py** | 1075 | 1091| 140 | 28789 | 127154 | 
| 85 | 21 django/contrib/contenttypes/admin.py | 91 | 144| 420 | 29209 | 128159 | 
| 86 | 21 django/contrib/admin/options.py | 243 | 256| 139 | 29348 | 128159 | 
| 87 | **21 django/contrib/admin/checks.py** | 704 | 740| 301 | 29649 | 128159 | 
| 88 | **21 django/contrib/admin/checks.py** | 580 | 608| 195 | 29844 | 128159 | 
| 89 | **21 django/contrib/admin/checks.py** | 55 | 173| 772 | 30616 | 128159 | 
| 90 | 22 django/contrib/gis/admin/options.py | 1 | 22| 160 | 30776 | 128320 | 
| 91 | 22 django/contrib/admin/templatetags/admin_modify.py | 115 | 130| 103 | 30879 | 128320 | 
| 92 | 22 django/contrib/admin/templatetags/admin_list.py | 355 | 456| 760 | 31639 | 128320 | 
| 93 | **22 django/contrib/admin/checks.py** | 539 | 554| 136 | 31775 | 128320 | 
| 94 | 23 django/contrib/auth/checks.py | 107 | 221| 786 | 32561 | 129836 | 
| 95 | 23 django/contrib/admin/filters.py | 632 | 645| 119 | 32680 | 129836 | 
| 96 | 24 django/contrib/admin/templatetags/admin_urls.py | 1 | 67| 419 | 33099 | 130255 | 
| 97 | 24 django/contrib/admin/filters.py | 1 | 21| 144 | 33243 | 130255 | 
| 98 | **24 django/contrib/admin/checks.py** | 742 | 759| 139 | 33382 | 130255 | 
| 99 | 25 django/contrib/auth/admin.py | 1 | 25| 195 | 33577 | 132062 | 
| 100 | **25 django/contrib/admin/checks.py** | 1147 | 1183| 252 | 33829 | 132062 | 
| 101 | 25 django/contrib/admin/helpers.py | 261 | 299| 323 | 34152 | 132062 | 
| 102 | **25 django/contrib/admin/checks.py** | 1246 | 1275| 196 | 34348 | 132062 | 
| 103 | 25 django/contrib/admin/options.py | 2459 | 2486| 254 | 34602 | 132062 | 
| 104 | 25 django/contrib/admin/filters.py | 90 | 166| 628 | 35230 | 132062 | 
| 105 | 25 django/contrib/admin/options.py | 630 | 681| 360 | 35590 | 132062 | 
| 106 | 26 django/contrib/admin/widgets.py | 384 | 453| 373 | 35963 | 136252 | 
| 107 | 26 django/forms/models.py | 892 | 915| 202 | 36165 | 136252 | 
| 108 | 27 django/core/checks/templates.py | 1 | 47| 313 | 36478 | 136734 | 
| 109 | 27 django/contrib/admin/helpers.py | 165 | 192| 221 | 36699 | 136734 | 
| 110 | 27 django/db/models/base.py | 1529 | 1565| 288 | 36987 | 136734 | 
| 111 | **27 django/contrib/admin/checks.py** | 284 | 312| 218 | 37205 | 136734 | 
| 112 | 27 django/db/models/base.py | 2025 | 2078| 359 | 37564 | 136734 | 
| 113 | 27 django/contrib/admin/filters.py | 458 | 533| 631 | 38195 | 136734 | 
| 114 | **27 django/contrib/admin/checks.py** | 480 | 503| 188 | 38383 | 136734 | 
| 115 | 27 django/db/models/base.py | 2080 | 2185| 733 | 39116 | 136734 | 
| 116 | 27 django/db/models/base.py | 1381 | 1396| 142 | 39258 | 136734 | 
| 117 | 27 django/forms/models.py | 1514 | 1527| 176 | 39434 | 136734 | 
| 118 | 27 django/contrib/admin/sites.py | 1 | 33| 224 | 39658 | 136734 | 
| 119 | 27 django/contrib/admin/options.py | 593 | 609| 198 | 39856 | 136734 | 
| 120 | 27 django/contrib/admin/widgets.py | 520 | 553| 307 | 40163 | 136734 | 
| 121 | 27 django/contrib/auth/admin.py | 28 | 40| 130 | 40293 | 136734 | 
| 122 | 27 django/contrib/admindocs/views.py | 395 | 428| 211 | 40504 | 136734 | 
| 123 | 27 django/contrib/admin/options.py | 1906 | 1917| 149 | 40653 | 136734 | 
| 124 | 27 django/forms/utils.py | 141 | 198| 348 | 41001 | 136734 | 
| 125 | 27 django/contrib/admin/options.py | 742 | 759| 128 | 41129 | 136734 | 
| 126 | 28 django/views/generic/__init__.py | 1 | 40| 204 | 41333 | 136939 | 
| 127 | 29 django/contrib/admin/exceptions.py | 1 | 14| 0 | 41333 | 137006 | 
| 128 | 29 django/contrib/admin/sites.py | 443 | 459| 136 | 41469 | 137006 | 
| 129 | 30 django/core/checks/messages.py | 25 | 38| 154 | 41623 | 137586 | 
| 130 | 30 django/contrib/admin/options.py | 887 | 900| 117 | 41740 | 137586 | 
| 131 | 30 django/contrib/admin/options.py | 1104 | 1123| 131 | 41871 | 137586 | 
| 132 | 30 django/contrib/admin/options.py | 1031 | 1052| 221 | 42092 | 137586 | 
| 133 | 31 django/db/models/enums.py | 34 | 56| 183 | 42275 | 138199 | 
| 134 | 31 django/contrib/admindocs/views.py | 65 | 99| 297 | 42572 | 138199 | 
| 135 | 31 django/contrib/admin/filters.py | 594 | 629| 328 | 42900 | 138199 | 
| 136 | 31 django/contrib/admin/filters.py | 648 | 690| 371 | 43271 | 138199 | 
| 137 | 31 django/contrib/admin/helpers.py | 496 | 524| 240 | 43511 | 138199 | 
| 138 | 32 django/contrib/gis/admin/__init__.py | 1 | 30| 130 | 43641 | 138329 | 
| 139 | **32 django/contrib/admin/checks.py** | 1233 | 1244| 116 | 43757 | 138329 | 
| 140 | 32 django/contrib/admin/options.py | 2306 | 2363| 476 | 44233 | 138329 | 
| 141 | 32 django/contrib/admin/views/main.py | 344 | 374| 270 | 44503 | 138329 | 
| 142 | **32 django/contrib/admin/checks.py** | 825 | 838| 117 | 44620 | 138329 | 
| 143 | 33 django/core/checks/security/base.py | 1 | 79| 691 | 45311 | 140518 | 
| 144 | 33 django/contrib/admin/options.py | 611 | 627| 173 | 45484 | 140518 | 
| 145 | 33 django/forms/models.py | 1587 | 1602| 131 | 45615 | 140518 | 
| 146 | 33 django/contrib/admin/views/main.py | 67 | 173| 866 | 46481 | 140518 | 
| 147 | 33 django/contrib/admindocs/views.py | 102 | 138| 301 | 46782 | 140518 | 
| 148 | 33 django/db/migrations/questioner.py | 90 | 107| 163 | 46945 | 140518 | 
| 149 | 33 django/db/models/base.py | 1815 | 1837| 175 | 47120 | 140518 | 
| 150 | 33 django/contrib/admin/widgets.py | 171 | 204| 244 | 47364 | 140518 | 
| 151 | 33 django/contrib/admin/filters.py | 535 | 552| 175 | 47539 | 140518 | 
| 152 | 33 django/contrib/admin/options.py | 1919 | 1933| 132 | 47671 | 140518 | 
| 153 | 33 django/contrib/admin/widgets.py | 555 | 588| 201 | 47872 | 140518 | 
| 154 | 34 django/views/defaults.py | 1 | 26| 151 | 48023 | 141508 | 
| 155 | **34 django/contrib/admin/checks.py** | 266 | 282| 138 | 48161 | 141508 | 
| 156 | 34 django/contrib/admin/options.py | 919 | 963| 314 | 48475 | 141508 | 
| 157 | 34 django/contrib/contenttypes/admin.py | 1 | 88| 585 | 49060 | 141508 | 
| 158 | 34 django/contrib/admindocs/views.py | 298 | 392| 640 | 49700 | 141508 | 
| 159 | 34 django/contrib/admin/options.py | 554 | 576| 241 | 49941 | 141508 | 
| 160 | 34 django/contrib/admin/utils.py | 140 | 178| 310 | 50251 | 141508 | 
| 161 | 34 django/contrib/auth/admin.py | 151 | 216| 477 | 50728 | 141508 | 
| 162 | **34 django/contrib/admin/checks.py** | 521 | 537| 144 | 50872 | 141508 | 
| 163 | 35 django/template/response.py | 1 | 51| 398 | 51270 | 142621 | 
| 164 | 35 django/views/defaults.py | 124 | 150| 197 | 51467 | 142621 | 
| 165 | 36 django/db/models/fields/related.py | 1463 | 1592| 984 | 52451 | 157325 | 
| 166 | 37 django/db/models/__init__.py | 1 | 116| 682 | 53133 | 158007 | 
| 167 | 38 django/contrib/contenttypes/fields.py | 455 | 472| 123 | 53256 | 163835 | 
| 168 | 38 django/contrib/admin/utils.py | 285 | 308| 174 | 53430 | 163835 | 
| 169 | 38 django/contrib/admin/filters.py | 692 | 711| 170 | 53600 | 163835 | 
| 170 | 38 django/contrib/admin/models.py | 48 | 85| 250 | 53850 | 163835 | 
| 171 | 39 django/forms/widgets.py | 721 | 739| 153 | 54003 | 172101 | 
| 172 | 39 django/contrib/admindocs/views.py | 1 | 36| 252 | 54255 | 172101 | 
| 173 | 40 django/forms/boundfield.py | 49 | 83| 225 | 54480 | 174623 | 
| 174 | **40 django/contrib/admin/checks.py** | 505 | 519| 122 | 54602 | 174623 | 
| 175 | **40 django/contrib/admin/checks.py** | 610 | 628| 164 | 54766 | 174623 | 
| 176 | 40 django/db/migrations/questioner.py | 217 | 247| 252 | 55018 | 174623 | 
| 177 | **40 django/contrib/admin/checks.py** | 348 | 367| 146 | 55164 | 174623 | 
| 178 | 40 django/forms/widgets.py | 664 | 685| 187 | 55351 | 174623 | 
| 179 | 40 django/contrib/admin/widgets.py | 297 | 343| 439 | 55790 | 174623 | 
| 180 | 40 django/contrib/admin/filters.py | 354 | 385| 299 | 56089 | 174623 | 
| 181 | 40 django/contrib/admin/sites.py | 360 | 381| 182 | 56271 | 174623 | 
| 182 | 41 django/core/handlers/exception.py | 63 | 158| 605 | 56876 | 175753 | 


### Hint

```
I think I will make a bug fix later if required
Thanks bakdolot 👍 There's a slight difference between a model instance's attributes and the model class' meta's fields. Meta stores the reverse relationship as choice, where as this would be setup & named according to whatever the related_name is declared as.
fyi potential quick fix, this will cause it to start raising E108 errors. this is just a demo of where to look. One possibility we could abandon using get_field() and refer to _meta.fields instead? 🤔… though that would mean the E109 check below this would no longer work. django/contrib/admin/checks.py a b from django.core.exceptions import FieldDoesNotExist 99from django.db import models 1010from django.db.models.constants import LOOKUP_SEP 1111from django.db.models.expressions import Combinable 12from django.db.models.fields.reverse_related import ManyToOneRel 1213from django.forms.models import BaseModelForm, BaseModelFormSet, _get_foreign_key 1314from django.template import engines 1415from django.template.backends.django import DjangoTemplates … … class ModelAdminChecks(BaseModelAdminChecks): 897898 return [] 898899 try: 899900 field = obj.model._meta.get_field(item) 901 if isinstance(field, ManyToOneRel): 902 raise FieldDoesNotExist 900903 except FieldDoesNotExist: 901904 try: 902905 field = getattr(obj.model, item)
This is related to the recent work merged for ticket:34481.
@nessita yup I recognised bakdolot's username from that patch :D
Oh no they recognized me :D I apologize very much. I noticed this bug only after merge when I decided to check again By the way, I also noticed two bugs related to this
I checked most of the fields and found these fields that are not working correctly class QuestionAdmin(admin.ModelAdmin): list_display = ["choice", "choice_set", "somem2m", "SomeM2M_question+", "somem2m_set", "__module__", "__doc__", "objects"] Also for reproduce see ​github
Replying to Baha Sdtbekov: I checked most of the fields and found these fields that are not working correctly class QuestionAdmin(admin.ModelAdmin): list_display = ["choice", "choice_set", "somem2m", "SomeM2M_question+", "somem2m_set", "__module__", "__doc__", "objects"] Also for reproduce see ​github System checks are helpers that in this case should highlight potentially reasonable but unsupported options. IMO they don't have to catch all obviously wrong values that you can find in __dir__.
Yup agreed with felixx if they're putting __doc__ in there then they probably need to go back and do a Python tutorial :) As for choice_set & somem2m – I thought that's what you fixed up in the other patch with E109.
```

## Patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -916,9 +916,10 @@ def _check_list_display_item(self, obj, item, label):
                         id="admin.E108",
                     )
                 ]
-        if isinstance(field, models.ManyToManyField) or (
-            getattr(field, "rel", None) and field.rel.field.many_to_one
-        ):
+        if (
+            getattr(field, "is_relation", False)
+            and (field.many_to_many or field.one_to_many)
+        ) or (getattr(field, "rel", None) and field.rel.field.many_to_one):
             return [
                 checks.Error(
                     f"The value of '{label}' must not be a many-to-many field or a "

```

## Test Patch

```diff
diff --git a/tests/modeladmin/test_checks.py b/tests/modeladmin/test_checks.py
--- a/tests/modeladmin/test_checks.py
+++ b/tests/modeladmin/test_checks.py
@@ -554,6 +554,30 @@ class TestModelAdmin(ModelAdmin):
             "admin.E109",
         )
 
+    def test_invalid_related_field(self):
+        class TestModelAdmin(ModelAdmin):
+            list_display = ["song"]
+
+        self.assertIsInvalid(
+            TestModelAdmin,
+            Band,
+            "The value of 'list_display[0]' must not be a many-to-many field or a "
+            "reverse foreign key.",
+            "admin.E109",
+        )
+
+    def test_invalid_m2m_related_name(self):
+        class TestModelAdmin(ModelAdmin):
+            list_display = ["featured"]
+
+        self.assertIsInvalid(
+            TestModelAdmin,
+            Band,
+            "The value of 'list_display[0]' must not be a many-to-many field or a "
+            "reverse foreign key.",
+            "admin.E109",
+        )
+
     def test_valid_case(self):
         @admin.display
         def a_callable(obj):

```


## Code snippets

### 1 - django/contrib/admin/options.py:

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
### 2 - django/contrib/admin/checks.py:

Start line: 1093, End line: 1145

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
### 3 - django/contrib/admin/options.py:

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
### 4 - django/contrib/admin/options.py:

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
### 5 - django/contrib/admin/options.py:

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
### 6 - django/contrib/admin/filters.py:

Start line: 24, End line: 87

```python
class ListFilter:
    title = None  # Human-readable title to appear in the right sidebar.
    template = "admin/filter.html"

    def __init__(self, request, params, model, model_admin):
        self.request = request
        # This dictionary will eventually contain the request's query string
        # parameters actually used by this filter.
        self.used_parameters = {}
        if self.title is None:
            raise ImproperlyConfigured(
                "The list filter '%s' does not specify a 'title'."
                % self.__class__.__name__
            )

    def has_output(self):
        """
        Return True if some choices would be output for this filter.
        """
        raise NotImplementedError(
            "subclasses of ListFilter must provide a has_output() method"
        )

    def choices(self, changelist):
        """
        Return choices ready to be output in the template.

        `changelist` is the ChangeList to be displayed.
        """
        raise NotImplementedError(
            "subclasses of ListFilter must provide a choices() method"
        )

    def queryset(self, request, queryset):
        """
        Return the filtered queryset.
        """
        raise NotImplementedError(
            "subclasses of ListFilter must provide a queryset() method"
        )

    def expected_parameters(self):
        """
        Return the list of parameter names that are expected from the
        request's query string and that will be used by this filter.
        """
        raise NotImplementedError(
            "subclasses of ListFilter must provide an expected_parameters() method"
        )


class FacetsMixin:
    def get_facet_counts(self, pk_attname, filtered_qs):
        raise NotImplementedError(
            "subclasses of FacetsMixin must provide a get_facet_counts() method."
        )

    def get_facet_queryset(self, changelist):
        filtered_qs = changelist.get_queryset(
            self.request, exclude_parameters=self.expected_parameters()
        )
        return filtered_qs.aggregate(
            **self.get_facet_counts(changelist.pk_attname, filtered_qs)
        )
```
### 7 - django/contrib/admin/checks.py:

Start line: 957, End line: 981

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
### 8 - django/contrib/admin/options.py:

Start line: 1278, End line: 1340

```python
class ModelAdmin(BaseModelAdmin):

    def render_change_form(
        self, request, context, add=False, change=False, form_url="", obj=None
    ):
        app_label = self.opts.app_label
        preserved_filters = self.get_preserved_filters(request)
        form_url = add_preserved_filters(
            {"preserved_filters": preserved_filters, "opts": self.opts}, form_url
        )
        view_on_site_url = self.get_view_on_site_url(obj)
        has_editable_inline_admin_formsets = False
        for inline in context["inline_admin_formsets"]:
            if (
                inline.has_add_permission
                or inline.has_change_permission
                or inline.has_delete_permission
            ):
                has_editable_inline_admin_formsets = True
                break
        context.update(
            {
                "add": add,
                "change": change,
                "has_view_permission": self.has_view_permission(request, obj),
                "has_add_permission": self.has_add_permission(request),
                "has_change_permission": self.has_change_permission(request, obj),
                "has_delete_permission": self.has_delete_permission(request, obj),
                "has_editable_inline_admin_formsets": (
                    has_editable_inline_admin_formsets
                ),
                "has_file_field": context["adminform"].form.is_multipart()
                or any(
                    admin_formset.formset.is_multipart()
                    for admin_formset in context["inline_admin_formsets"]
                ),
                "has_absolute_url": view_on_site_url is not None,
                "absolute_url": view_on_site_url,
                "form_url": form_url,
                "opts": self.opts,
                "content_type_id": get_content_type_for_model(self.model).pk,
                "save_as": self.save_as,
                "save_on_top": self.save_on_top,
                "to_field_var": TO_FIELD_VAR,
                "is_popup_var": IS_POPUP_VAR,
                "app_label": app_label,
            }
        )
        if add and self.add_form_template is not None:
            form_template = self.add_form_template
        else:
            form_template = self.change_form_template

        request.current_app = self.admin_site.name

        return TemplateResponse(
            request,
            form_template
            or [
                "admin/%s/%s/change_form.html" % (app_label, self.opts.model_name),
                "admin/%s/change_form.html" % app_label,
                "admin/change_form.html",
            ],
            context,
        )
```
### 9 - django/contrib/admin/templatetags/admin_list.py:

Start line: 459, End line: 531

```python
@register.tag(name="date_hierarchy")
def date_hierarchy_tag(parser, token):
    return InclusionAdminNode(
        parser,
        token,
        func=date_hierarchy,
        template_name="date_hierarchy.html",
        takes_context=False,
    )


def search_form(cl):
    """
    Display a search form for searching the list.
    """
    return {
        "cl": cl,
        "show_result_count": cl.result_count != cl.full_result_count,
        "search_var": SEARCH_VAR,
        "is_popup_var": IS_POPUP_VAR,
        "is_facets_var": IS_FACETS_VAR,
    }


@register.tag(name="search_form")
def search_form_tag(parser, token):
    return InclusionAdminNode(
        parser,
        token,
        func=search_form,
        template_name="search_form.html",
        takes_context=False,
    )


@register.simple_tag
def admin_list_filter(cl, spec):
    tpl = get_template(spec.template)
    return tpl.render(
        {
            "title": spec.title,
            "choices": list(spec.choices(cl)),
            "spec": spec,
        }
    )


def admin_actions(context):
    """
    Track the number of times the action field has been rendered on the page,
    so we know which value to use.
    """
    context["action_index"] = context.get("action_index", -1) + 1
    return context


@register.tag(name="admin_actions")
def admin_actions_tag(parser, token):
    return InclusionAdminNode(
        parser, token, func=admin_actions, template_name="actions.html"
    )


@register.tag(name="change_list_object_tools")
def change_list_object_tools_tag(parser, token):
    """Display the row of change list object tools."""
    return InclusionAdminNode(
        parser,
        token,
        func=lambda context: context,
        template_name="change_list_object_tools.html",
    )
```
### 10 - django/contrib/admin/checks.py:

Start line: 893, End line: 930

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
        if isinstance(field, models.ManyToManyField) or (
            getattr(field, "rel", None) and field.rel.field.many_to_one
        ):
            return [
                checks.Error(
                    f"The value of '{label}' must not be a many-to-many field or a "
                    f"reverse foreign key.",
                    obj=obj.__class__,
                    id="admin.E109",
                )
            ]
        return []
```
### 20 - django/contrib/admin/checks.py:

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
### 28 - django/contrib/admin/checks.py:

Start line: 983, End line: 1040

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
### 34 - django/contrib/admin/checks.py:

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
### 35 - django/contrib/admin/checks.py:

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
### 39 - django/contrib/admin/checks.py:

Start line: 1042, End line: 1073

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
### 43 - django/contrib/admin/checks.py:

Start line: 1324, End line: 1353

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
### 51 - django/contrib/admin/checks.py:

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
### 59 - django/contrib/admin/checks.py:

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
### 66 - django/contrib/admin/checks.py:

Start line: 1277, End line: 1321

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
### 70 - django/contrib/admin/checks.py:

Start line: 932, End line: 955

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
### 72 - django/contrib/admin/checks.py:

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
### 74 - django/contrib/admin/checks.py:

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
### 75 - django/contrib/admin/checks.py:

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
### 83 - django/contrib/admin/checks.py:

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
### 84 - django/contrib/admin/checks.py:

Start line: 1075, End line: 1091

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_editable(self, obj):
        """Check that list_editable is a sequence of editable fields from
        list_display without first element."""

        if not isinstance(obj.list_editable, (list, tuple)):
            return must_be(
                "a list or tuple", option="list_editable", obj=obj, id="admin.E120"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_list_editable_item(
                        obj, item, "list_editable[%d]" % index
                    )
                    for index, item in enumerate(obj.list_editable)
                )
            )
```
### 87 - django/contrib/admin/checks.py:

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
### 88 - django/contrib/admin/checks.py:

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
### 89 - django/contrib/admin/checks.py:

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
### 93 - django/contrib/admin/checks.py:

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
### 98 - django/contrib/admin/checks.py:

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
### 100 - django/contrib/admin/checks.py:

Start line: 1147, End line: 1183

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
### 102 - django/contrib/admin/checks.py:

Start line: 1246, End line: 1275

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
### 111 - django/contrib/admin/checks.py:

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
### 114 - django/contrib/admin/checks.py:

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
### 139 - django/contrib/admin/checks.py:

Start line: 1233, End line: 1244

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
### 142 - django/contrib/admin/checks.py:

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
### 155 - django/contrib/admin/checks.py:

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
### 162 - django/contrib/admin/checks.py:

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
### 174 - django/contrib/admin/checks.py:

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
### 175 - django/contrib/admin/checks.py:

Start line: 610, End line: 628

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields(self, obj):
        """Check that `prepopulated_fields` is a dictionary containing allowed
        field types."""
        if not isinstance(obj.prepopulated_fields, dict):
            return must_be(
                "a dictionary", option="prepopulated_fields", obj=obj, id="admin.E026"
            )
        else:
            return list(
                chain.from_iterable(
                    self._check_prepopulated_fields_key(
                        obj, field_name, "prepopulated_fields"
                    )
                    + self._check_prepopulated_fields_value(
                        obj, val, 'prepopulated_fields["%s"]' % field_name
                    )
                    for field_name, val in obj.prepopulated_fields.items()
                )
            )
```
### 177 - django/contrib/admin/checks.py:

Start line: 348, End line: 367

```python
class BaseModelAdminChecks:

    def _check_fieldsets(self, obj):
        """Check that fieldsets is properly formatted and doesn't contain
        duplicates."""

        if obj.fieldsets is None:
            return []
        elif not isinstance(obj.fieldsets, (list, tuple)):
            return must_be(
                "a list or tuple", option="fieldsets", obj=obj, id="admin.E007"
            )
        else:
            seen_fields = []
            return list(
                chain.from_iterable(
                    self._check_fieldsets_item(
                        obj, fieldset, "fieldsets[%d]" % index, seen_fields
                    )
                    for index, fieldset in enumerate(obj.fieldsets)
                )
            )
```
