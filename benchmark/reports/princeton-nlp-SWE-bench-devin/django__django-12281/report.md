# django__django-12281

| **django/django** | `e2d9d66a22f9004c0349f6aa9f8762fa558bdee8` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 26564 |
| **Any found context length** | 281 |
| **Avg pos** | 91.0 |
| **Min pos** | 1 |
| **Max pos** | 90 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -1,3 +1,4 @@
+import collections
 from itertools import chain
 
 from django.apps import apps
@@ -985,15 +986,20 @@ def _check_action_permission_methods(self, obj):
 
     def _check_actions_uniqueness(self, obj):
         """Check that every action has a unique __name__."""
-        names = [name for _, name, _ in obj._get_base_actions()]
-        if len(names) != len(set(names)):
-            return [checks.Error(
-                '__name__ attributes of actions defined in %s must be '
-                'unique.' % obj.__class__,
-                obj=obj.__class__,
-                id='admin.E130',
-            )]
-        return []
+        errors = []
+        names = collections.Counter(name for _, name, _ in obj._get_base_actions())
+        for name, count in names.items():
+            if count > 1:
+                errors.append(checks.Error(
+                    '__name__ attributes of actions defined in %s must be '
+                    'unique. Name %r is not unique.' % (
+                        obj.__class__.__name__,
+                        name,
+                    ),
+                    obj=obj.__class__,
+                    id='admin.E130',
+                ))
+        return errors
 
 
 class InlineModelAdminChecks(BaseModelAdminChecks):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/checks.py | 1 | 1 | 90 | 1 | 26564
| django/contrib/admin/checks.py | 988 | 996 | 1 | 1 | 281


## Problem Statement

```
admin.E130 (duplicate __name__ attributes of actions) should specify which were duplicated.
Description
	
The fact that the __name__ is used is somewhat an implementation detail, and there's no guarantee the user has enough of an understanding of python to know what that attribute is, let alone how to fix it.
This just came up on IRC because a user had defined actions = [delete_selected] where delete_selected was a reference to their own callable, but shares the name of the base one (and by specifying the actions = they were assuming that they were wholesale replacing the actions list, where that may not be true for site-wide actions) so errored ... but they only had define a list of len(...) == 1 so how can there be a duplicate (is their thought process)?
The error message should specify those names that occur 2> (rather than just check len(...) vs len(set(...))), and ought ideally to explain where the duplicate comes from (ie: AdminSite-wide).
Related ticket about E130: #30311 (+ those it references) but is about the replacement strategy rather than the error message itself.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admin/checks.py** | 960 | 996| 281 | 281 | 9016 | 
| 2 | 2 django/contrib/admin/actions.py | 1 | 80| 609 | 890 | 9625 | 
| 3 | 3 django/contrib/admin/options.py | 880 | 901| 221 | 1111 | 28112 | 
| 4 | 3 django/contrib/admin/options.py | 903 | 931| 233 | 1344 | 28112 | 
| 5 | 3 django/contrib/admin/options.py | 1337 | 1402| 581 | 1925 | 28112 | 
| 6 | 4 django/db/models/base.py | 1471 | 1493| 171 | 2096 | 43428 | 
| 7 | 5 django/db/migrations/exceptions.py | 1 | 55| 250 | 2346 | 43679 | 
| 8 | 6 django/contrib/admin/sites.py | 1 | 29| 175 | 2521 | 47870 | 
| 9 | **6 django/contrib/admin/checks.py** | 353 | 369| 134 | 2655 | 47870 | 
| 10 | 6 django/contrib/admin/sites.py | 134 | 194| 397 | 3052 | 47870 | 
| 11 | 6 django/db/models/base.py | 1446 | 1469| 176 | 3228 | 47870 | 
| 12 | **6 django/contrib/admin/checks.py** | 230 | 260| 229 | 3457 | 47870 | 
| 13 | 6 django/contrib/admin/options.py | 864 | 878| 125 | 3582 | 47870 | 
| 14 | 7 django/contrib/admin/helpers.py | 1 | 30| 198 | 3780 | 51063 | 
| 15 | 8 django/utils/deprecation.py | 30 | 70| 336 | 4116 | 51741 | 
| 16 | 9 django/contrib/auth/__init__.py | 196 | 218| 166 | 4282 | 53313 | 
| 17 | 9 django/db/models/base.py | 1495 | 1527| 231 | 4513 | 53313 | 
| 18 | **9 django/contrib/admin/checks.py** | 879 | 927| 416 | 4929 | 53313 | 
| 19 | **9 django/contrib/admin/checks.py** | 203 | 213| 127 | 5056 | 53313 | 
| 20 | 9 django/db/models/base.py | 1160 | 1188| 213 | 5269 | 53313 | 
| 21 | 9 django/db/models/base.py | 1389 | 1444| 491 | 5760 | 53313 | 
| 22 | 10 django/core/checks/urls.py | 30 | 50| 165 | 5925 | 54014 | 
| 23 | 11 django/contrib/auth/checks.py | 97 | 167| 525 | 6450 | 55187 | 
| 24 | 11 django/db/models/base.py | 1762 | 1833| 565 | 7015 | 55187 | 
| 25 | 12 django/db/migrations/operations/models.py | 1 | 38| 238 | 7253 | 61883 | 
| 26 | 12 django/db/models/base.py | 1069 | 1112| 404 | 7657 | 61883 | 
| 27 | **12 django/contrib/admin/checks.py** | 620 | 639| 183 | 7840 | 61883 | 
| 28 | 13 django/core/management/templates.py | 210 | 241| 236 | 8076 | 64564 | 
| 29 | 14 django/forms/models.py | 752 | 773| 194 | 8270 | 76166 | 
| 30 | **14 django/contrib/admin/checks.py** | 1013 | 1040| 204 | 8474 | 76166 | 
| 31 | 14 django/contrib/auth/checks.py | 1 | 94| 646 | 9120 | 76166 | 
| 32 | 14 django/db/models/base.py | 1251 | 1280| 242 | 9362 | 76166 | 
| 33 | **14 django/contrib/admin/checks.py** | 129 | 146| 155 | 9517 | 76166 | 
| 34 | 15 django/contrib/sites/migrations/0002_alter_domain_unique.py | 1 | 21| 0 | 9517 | 76263 | 
| 35 | 16 django/contrib/admin/migrations/0003_logentry_add_action_flag_choices.py | 1 | 21| 111 | 9628 | 76374 | 
| 36 | **16 django/contrib/admin/checks.py** | 751 | 766| 182 | 9810 | 76374 | 
| 37 | 17 django/db/backends/base/schema.py | 1106 | 1136| 240 | 10050 | 87689 | 
| 38 | 18 django/contrib/admindocs/utils.py | 86 | 112| 175 | 10225 | 89595 | 
| 39 | 19 django/db/models/fields/related.py | 190 | 254| 673 | 10898 | 103107 | 
| 40 | 20 django/core/checks/templates.py | 1 | 36| 259 | 11157 | 103367 | 
| 41 | 21 django/db/migrations/autodetector.py | 465 | 506| 418 | 11575 | 115102 | 
| 42 | 22 django/contrib/sites/managers.py | 1 | 61| 385 | 11960 | 115487 | 
| 43 | **22 django/contrib/admin/checks.py** | 768 | 789| 190 | 12150 | 115487 | 
| 44 | 22 django/contrib/admindocs/utils.py | 142 | 177| 444 | 12594 | 115487 | 
| 45 | **22 django/contrib/admin/checks.py** | 1087 | 1117| 188 | 12782 | 115487 | 
| 46 | **22 django/contrib/admin/checks.py** | 215 | 228| 161 | 12943 | 115487 | 
| 47 | 22 django/contrib/admindocs/utils.py | 115 | 139| 185 | 13128 | 115487 | 
| 48 | 23 django/contrib/contenttypes/checks.py | 24 | 42| 125 | 13253 | 115736 | 
| 49 | 24 django/core/checks/security/base.py | 1 | 83| 732 | 13985 | 117517 | 
| 50 | 24 django/contrib/admindocs/utils.py | 180 | 229| 548 | 14533 | 117517 | 
| 51 | 24 django/db/models/base.py | 1556 | 1581| 183 | 14716 | 117517 | 
| 52 | 25 django/contrib/admin/templatetags/admin_list.py | 431 | 489| 343 | 15059 | 121346 | 
| 53 | **25 django/contrib/admin/checks.py** | 583 | 594| 128 | 15187 | 121346 | 
| 54 | 26 django/core/checks/model_checks.py | 178 | 211| 332 | 15519 | 123133 | 
| 55 | **26 django/contrib/admin/checks.py** | 262 | 275| 135 | 15654 | 123133 | 
| 56 | 26 django/db/migrations/operations/models.py | 414 | 435| 178 | 15832 | 123133 | 
| 57 | 26 django/core/checks/model_checks.py | 155 | 176| 263 | 16095 | 123133 | 
| 58 | **26 django/contrib/admin/checks.py** | 148 | 158| 123 | 16218 | 123133 | 
| 59 | 26 django/db/migrations/operations/models.py | 396 | 412| 182 | 16400 | 123133 | 
| 60 | **26 django/contrib/admin/checks.py** | 1042 | 1084| 343 | 16743 | 123133 | 
| 61 | 26 django/db/models/fields/related.py | 1202 | 1313| 939 | 17682 | 123133 | 
| 62 | 27 django/contrib/admin/utils.py | 287 | 305| 175 | 17857 | 127250 | 
| 63 | 28 django/core/management/commands/inspectdb.py | 175 | 229| 478 | 18335 | 129860 | 
| 64 | **28 django/contrib/admin/checks.py** | 718 | 749| 229 | 18564 | 129860 | 
| 65 | 28 django/contrib/admin/utils.py | 121 | 156| 303 | 18867 | 129860 | 
| 66 | **28 django/contrib/admin/checks.py** | 641 | 668| 232 | 19099 | 129860 | 
| 67 | **28 django/contrib/admin/checks.py** | 843 | 865| 217 | 19316 | 129860 | 
| 68 | 28 django/db/models/base.py | 1612 | 1660| 348 | 19664 | 129860 | 
| 69 | **28 django/contrib/admin/checks.py** | 706 | 716| 115 | 19779 | 129860 | 
| 70 | 29 django/db/backends/mysql/base.py | 287 | 325| 404 | 20183 | 132983 | 
| 71 | 30 django/core/management/commands/makemigrations.py | 60 | 146| 788 | 20971 | 135732 | 
| 72 | 31 django/db/models/deletion.py | 1 | 76| 566 | 21537 | 139391 | 
| 73 | 31 django/contrib/admin/sites.py | 512 | 546| 293 | 21830 | 139391 | 
| 74 | 31 django/contrib/admin/options.py | 1738 | 1819| 744 | 22574 | 139391 | 
| 75 | **31 django/contrib/admin/checks.py** | 160 | 201| 325 | 22899 | 139391 | 
| 76 | 31 django/db/migrations/operations/models.py | 304 | 343| 406 | 23305 | 139391 | 
| 77 | 32 django/db/migrations/questioner.py | 187 | 205| 237 | 23542 | 141465 | 
| 78 | 32 django/db/migrations/operations/models.py | 345 | 394| 493 | 24035 | 141465 | 
| 79 | **32 django/contrib/admin/checks.py** | 424 | 445| 191 | 24226 | 141465 | 
| 80 | 33 django/contrib/admindocs/views.py | 1 | 30| 223 | 24449 | 144773 | 
| 81 | 33 django/core/checks/model_checks.py | 112 | 127| 176 | 24625 | 144773 | 
| 82 | 33 django/utils/deprecation.py | 1 | 27| 181 | 24806 | 144773 | 
| 83 | 34 django/db/models/options.py | 1 | 36| 301 | 25107 | 151857 | 
| 84 | **34 django/contrib/admin/checks.py** | 325 | 351| 221 | 25328 | 151857 | 
| 85 | 34 django/contrib/admin/sites.py | 70 | 84| 129 | 25457 | 151857 | 
| 86 | 34 django/db/backends/base/schema.py | 1080 | 1104| 193 | 25650 | 151857 | 
| 87 | 34 django/contrib/admin/utils.py | 159 | 185| 239 | 25889 | 151857 | 
| 88 | 35 django/contrib/admin/models.py | 23 | 36| 111 | 26000 | 152980 | 
| 89 | 36 django/db/models/fields/__init__.py | 208 | 242| 235 | 26235 | 170492 | 
| **-> 90 <-** | **36 django/contrib/admin/checks.py** | 1 | 54| 329 | 26564 | 170492 | 
| 91 | 36 django/core/checks/model_checks.py | 129 | 153| 268 | 26832 | 170492 | 
| 92 | 37 django/db/backends/sqlite3/base.py | 301 | 385| 829 | 27661 | 176256 | 
| 93 | 37 django/core/management/templates.py | 119 | 182| 563 | 28224 | 176256 | 
| 94 | 37 django/core/checks/security/base.py | 85 | 180| 710 | 28934 | 176256 | 
| 95 | 38 django/db/backends/sqlite3/operations.py | 163 | 188| 190 | 29124 | 179147 | 
| 96 | **38 django/contrib/admin/checks.py** | 521 | 544| 230 | 29354 | 179147 | 
| 97 | 39 django/core/management/commands/squashmigrations.py | 202 | 215| 112 | 29466 | 181018 | 
| 98 | 39 django/db/models/base.py | 1282 | 1307| 184 | 29650 | 181018 | 
| 99 | 39 django/contrib/admindocs/utils.py | 1 | 25| 151 | 29801 | 181018 | 
| 100 | 39 django/db/models/base.py | 404 | 503| 856 | 30657 | 181018 | 
| 101 | 39 django/db/migrations/autodetector.py | 1123 | 1144| 231 | 30888 | 181018 | 
| 102 | **39 django/contrib/admin/checks.py** | 596 | 617| 162 | 31050 | 181018 | 
| 103 | 39 django/contrib/admin/options.py | 1821 | 1889| 584 | 31634 | 181018 | 
| 104 | 39 django/db/models/base.py | 1583 | 1610| 238 | 31872 | 181018 | 
| 105 | 39 django/core/checks/model_checks.py | 1 | 86| 667 | 32539 | 181018 | 
| 106 | 39 django/db/models/options.py | 149 | 208| 587 | 33126 | 181018 | 
| 107 | 40 django/contrib/auth/migrations/0010_alter_group_name_max_length.py | 1 | 17| 0 | 33126 | 181095 | 
| 108 | 40 django/db/migrations/autodetector.py | 796 | 845| 570 | 33696 | 181095 | 
| 109 | **40 django/contrib/admin/checks.py** | 546 | 581| 303 | 33999 | 181095 | 
| 110 | 40 django/contrib/admin/models.py | 39 | 72| 241 | 34240 | 181095 | 
| 111 | 40 django/db/backends/base/schema.py | 626 | 698| 792 | 35032 | 181095 | 
| 112 | 41 django/core/serializers/xml_serializer.py | 393 | 421| 233 | 35265 | 184489 | 
| 113 | 41 django/contrib/admin/sites.py | 32 | 68| 298 | 35563 | 184489 | 
| 114 | 41 django/db/models/fields/related.py | 255 | 282| 269 | 35832 | 184489 | 
| 115 | 42 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 36447 | 185514 | 
| 116 | 43 django/db/backends/oracle/schema.py | 125 | 173| 419 | 36866 | 187270 | 
| 117 | **43 django/contrib/admin/checks.py** | 371 | 397| 281 | 37147 | 187270 | 
| 118 | **43 django/contrib/admin/checks.py** | 929 | 958| 243 | 37390 | 187270 | 
| 119 | **43 django/contrib/admin/checks.py** | 487 | 507| 200 | 37590 | 187270 | 
| 120 | 43 django/contrib/admin/options.py | 844 | 862| 187 | 37777 | 187270 | 
| 121 | 44 django/contrib/sites/admin.py | 1 | 9| 0 | 37777 | 187316 | 
| 122 | 45 django/contrib/auth/management/commands/createsuperuser.py | 81 | 202| 1155 | 38932 | 189376 | 
| 123 | 46 django/db/backends/oracle/creation.py | 130 | 165| 399 | 39331 | 193271 | 
| 124 | 46 django/contrib/admin/options.py | 99 | 129| 223 | 39554 | 193271 | 


### Hint

```
Agreed, we can add names of duplicated actions to this message.
Hey! I am new to django contribution and I want to solve this issue. I want to know that error message of duplicates should write after the error of unique name in the same function ?
```

## Patch

```diff
diff --git a/django/contrib/admin/checks.py b/django/contrib/admin/checks.py
--- a/django/contrib/admin/checks.py
+++ b/django/contrib/admin/checks.py
@@ -1,3 +1,4 @@
+import collections
 from itertools import chain
 
 from django.apps import apps
@@ -985,15 +986,20 @@ def _check_action_permission_methods(self, obj):
 
     def _check_actions_uniqueness(self, obj):
         """Check that every action has a unique __name__."""
-        names = [name for _, name, _ in obj._get_base_actions()]
-        if len(names) != len(set(names)):
-            return [checks.Error(
-                '__name__ attributes of actions defined in %s must be '
-                'unique.' % obj.__class__,
-                obj=obj.__class__,
-                id='admin.E130',
-            )]
-        return []
+        errors = []
+        names = collections.Counter(name for _, name, _ in obj._get_base_actions())
+        for name, count in names.items():
+            if count > 1:
+                errors.append(checks.Error(
+                    '__name__ attributes of actions defined in %s must be '
+                    'unique. Name %r is not unique.' % (
+                        obj.__class__.__name__,
+                        name,
+                    ),
+                    obj=obj.__class__,
+                    id='admin.E130',
+                ))
+        return errors
 
 
 class InlineModelAdminChecks(BaseModelAdminChecks):

```

## Test Patch

```diff
diff --git a/tests/modeladmin/test_checks.py b/tests/modeladmin/test_checks.py
--- a/tests/modeladmin/test_checks.py
+++ b/tests/modeladmin/test_checks.py
@@ -1441,9 +1441,8 @@ class BandAdmin(ModelAdmin):
 
         self.assertIsInvalid(
             BandAdmin, Band,
-            "__name__ attributes of actions defined in "
-            "<class 'modeladmin.test_checks.ActionsCheckTests."
-            "test_actions_not_unique.<locals>.BandAdmin'> must be unique.",
+            "__name__ attributes of actions defined in BandAdmin must be "
+            "unique. Name 'action' is not unique.",
             id='admin.E130',
         )
 

```


## Code snippets

### 1 - django/contrib/admin/checks.py:

Start line: 960, End line: 996

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_action_permission_methods(self, obj):
        """
        Actions with an allowed_permission attribute require the ModelAdmin to
        implement a has_<perm>_permission() method for each permission.
        """
        actions = obj._get_base_actions()
        errors = []
        for func, name, _ in actions:
            if not hasattr(func, 'allowed_permissions'):
                continue
            for permission in func.allowed_permissions:
                method_name = 'has_%s_permission' % permission
                if not hasattr(obj, method_name):
                    errors.append(
                        checks.Error(
                            '%s must define a %s() method for the %s action.' % (
                                obj.__class__.__name__,
                                method_name,
                                func.__name__,
                            ),
                            obj=obj.__class__,
                            id='admin.E129',
                        )
                    )
        return errors

    def _check_actions_uniqueness(self, obj):
        """Check that every action has a unique __name__."""
        names = [name for _, name, _ in obj._get_base_actions()]
        if len(names) != len(set(names)):
            return [checks.Error(
                '__name__ attributes of actions defined in %s must be '
                'unique.' % obj.__class__,
                obj=obj.__class__,
                id='admin.E130',
            )]
        return []
```
### 2 - django/contrib/admin/actions.py:

Start line: 1, End line: 80

```python
"""
Built-in, globally-available admin actions.
"""

from django.contrib import messages
from django.contrib.admin import helpers
from django.contrib.admin.utils import model_ngettext
from django.core.exceptions import PermissionDenied
from django.template.response import TemplateResponse
from django.utils.translation import gettext as _, gettext_lazy


def delete_selected(modeladmin, request, queryset):
    """
    Default action which deletes the selected objects.

    This action first displays a confirmation page which shows all the
    deletable objects, or, if the user has no permission one of the related
    childs (foreignkeys), a "permission denied" message.

    Next, it deletes all selected objects and redirects back to the change list.
    """
    opts = modeladmin.model._meta
    app_label = opts.app_label

    # Populate deletable_objects, a data structure of all related objects that
    # will also be deleted.
    deletable_objects, model_count, perms_needed, protected = modeladmin.get_deleted_objects(queryset, request)

    # The user has already confirmed the deletion.
    # Do the deletion and return None to display the change list view again.
    if request.POST.get('post') and not protected:
        if perms_needed:
            raise PermissionDenied
        n = queryset.count()
        if n:
            for obj in queryset:
                obj_display = str(obj)
                modeladmin.log_deletion(request, obj, obj_display)
            modeladmin.delete_queryset(request, queryset)
            modeladmin.message_user(request, _("Successfully deleted %(count)d %(items)s.") % {
                "count": n, "items": model_ngettext(modeladmin.opts, n)
            }, messages.SUCCESS)
        # Return None to display the change list page again.
        return None

    objects_name = model_ngettext(queryset)

    if perms_needed or protected:
        title = _("Cannot delete %(name)s") % {"name": objects_name}
    else:
        title = _("Are you sure?")

    context = {
        **modeladmin.admin_site.each_context(request),
        'title': title,
        'objects_name': str(objects_name),
        'deletable_objects': [deletable_objects],
        'model_count': dict(model_count).items(),
        'queryset': queryset,
        'perms_lacking': perms_needed,
        'protected': protected,
        'opts': opts,
        'action_checkbox_name': helpers.ACTION_CHECKBOX_NAME,
        'media': modeladmin.media,
    }

    request.current_app = modeladmin.admin_site.name

    # Display the confirmation page
    return TemplateResponse(request, modeladmin.delete_selected_confirmation_template or [
        "admin/%s/%s/delete_selected_confirmation.html" % (app_label, opts.model_name),
        "admin/%s/delete_selected_confirmation.html" % app_label,
        "admin/delete_selected_confirmation.html"
    ], context)


delete_selected.allowed_permissions = ('delete',)
delete_selected.short_description = gettext_lazy("Delete selected %(verbose_name_plural)s")
```
### 3 - django/contrib/admin/options.py:

Start line: 880, End line: 901

```python
class ModelAdmin(BaseModelAdmin):

    def get_actions(self, request):
        """
        Return a dictionary mapping the names of all actions for this
        ModelAdmin to a tuple of (callable, name, description) for each action.
        """
        # If self.actions is set to None that means actions are disabled on
        # this page.
        if self.actions is None or IS_POPUP_VAR in request.GET:
            return {}
        actions = self._filter_actions_by_permissions(request, self._get_base_actions())
        return {name: (func, name, desc) for func, name, desc in actions}

    def get_action_choices(self, request, default_choices=BLANK_CHOICE_DASH):
        """
        Return a list of choices for use in a form object.  Each choice is a
        tuple (name, description).
        """
        choices = [] + default_choices
        for func, name, description in self.get_actions(request).values():
            choice = (name, description % model_format_dict(self.opts))
            choices.append(choice)
        return choices
```
### 4 - django/contrib/admin/options.py:

Start line: 903, End line: 931

```python
class ModelAdmin(BaseModelAdmin):

    def get_action(self, action):
        """
        Return a given action from a parameter, which can either be a callable,
        or the name of a method on the ModelAdmin.  Return is a tuple of
        (callable, name, description).
        """
        # If the action is a callable, just use it.
        if callable(action):
            func = action
            action = action.__name__

        # Next, look for a method. Grab it off self.__class__ to get an unbound
        # method instead of a bound one; this ensures that the calling
        # conventions are the same for functions and methods.
        elif hasattr(self.__class__, action):
            func = getattr(self.__class__, action)

        # Finally, look for a named method on the admin site
        else:
            try:
                func = self.admin_site.get_action(action)
            except KeyError:
                return None

        if hasattr(func, 'short_description'):
            description = func.short_description
        else:
            description = capfirst(action.replace('_', ' '))
        return func, action, description
```
### 5 - django/contrib/admin/options.py:

Start line: 1337, End line: 1402

```python
class ModelAdmin(BaseModelAdmin):

    def response_action(self, request, queryset):
        """
        Handle an admin action. This is called if a request is POSTed to the
        changelist; it returns an HttpResponse if the action was handled, and
        None otherwise.
        """

        # There can be multiple action forms on the page (at the top
        # and bottom of the change list, for example). Get the action
        # whose button was pushed.
        try:
            action_index = int(request.POST.get('index', 0))
        except ValueError:
            action_index = 0

        # Construct the action form.
        data = request.POST.copy()
        data.pop(helpers.ACTION_CHECKBOX_NAME, None)
        data.pop("index", None)

        # Use the action whose button was pushed
        try:
            data.update({'action': data.getlist('action')[action_index]})
        except IndexError:
            # If we didn't get an action from the chosen form that's invalid
            # POST data, so by deleting action it'll fail the validation check
            # below. So no need to do anything here
            pass

        action_form = self.action_form(data, auto_id=None)
        action_form.fields['action'].choices = self.get_action_choices(request)

        # If the form's valid we can handle the action.
        if action_form.is_valid():
            action = action_form.cleaned_data['action']
            select_across = action_form.cleaned_data['select_across']
            func = self.get_actions(request)[action][0]

            # Get the list of selected PKs. If nothing's selected, we can't
            # perform an action on it, so bail. Except we want to perform
            # the action explicitly on all objects.
            selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)
            if not selected and not select_across:
                # Reminder that something needs to be selected or nothing will happen
                msg = _("Items must be selected in order to perform "
                        "actions on them. No items have been changed.")
                self.message_user(request, msg, messages.WARNING)
                return None

            if not select_across:
                # Perform the action only on the selected objects
                queryset = queryset.filter(pk__in=selected)

            response = func(self, request, queryset)

            # Actions may return an HttpResponse-like object, which will be
            # used as the response from the POST. If not, we'll be a good
            # little HTTP citizen and redirect back to the changelist page.
            if isinstance(response, HttpResponseBase):
                return response
            else:
                return HttpResponseRedirect(request.get_full_path())
        else:
            msg = _("No action selected.")
            self.message_user(request, msg, messages.WARNING)
            return None
```
### 6 - django/db/models/base.py:

Start line: 1471, End line: 1493

```python
class Model(metaclass=ModelBase):

    @classmethod
    def _check_model_name_db_lookup_clashes(cls):
        errors = []
        model_name = cls.__name__
        if model_name.startswith('_') or model_name.endswith('_'):
            errors.append(
                checks.Error(
                    "The model name '%s' cannot start or end with an underscore "
                    "as it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E023'
                )
            )
        elif LOOKUP_SEP in model_name:
            errors.append(
                checks.Error(
                    "The model name '%s' cannot contain double underscores as "
                    "it collides with the query lookup syntax." % model_name,
                    obj=cls,
                    id='models.E024'
                )
            )
        return errors
```
### 7 - django/db/migrations/exceptions.py:

Start line: 1, End line: 55

```python
from django.db.utils import DatabaseError


class AmbiguityError(Exception):
    """More than one migration matches a name prefix."""
    pass


class BadMigrationError(Exception):
    """There's a bad migration (unreadable/bad format/etc.)."""
    pass


class CircularDependencyError(Exception):
    """There's an impossible-to-resolve circular dependency."""
    pass


class InconsistentMigrationHistory(Exception):
    """An applied migration has some of its dependencies not applied."""
    pass


class InvalidBasesError(ValueError):
    """A model's base classes can't be resolved."""
    pass


class IrreversibleError(RuntimeError):
    """An irreversible migration is about to be reversed."""
    pass


class NodeNotFoundError(LookupError):
    """An attempt on a node is made that is not available in the graph."""

    def __init__(self, message, node, origin=None):
        self.message = message
        self.origin = origin
        self.node = node

    def __str__(self):
        return self.message

    def __repr__(self):
        return "NodeNotFoundError(%r)" % (self.node,)


class MigrationSchemaMissing(DatabaseError):
    pass


class InvalidMigrationPlan(ValueError):
    pass
```
### 8 - django/contrib/admin/sites.py:

Start line: 1, End line: 29

```python
import re
from functools import update_wrapper
from weakref import WeakSet

from django.apps import apps
from django.contrib.admin import ModelAdmin, actions
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.exceptions import ImproperlyConfigured
from django.db.models.base import ModelBase
from django.http import Http404, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import NoReverseMatch, reverse
from django.utils.functional import LazyObject
from django.utils.module_loading import import_string
from django.utils.text import capfirst
from django.utils.translation import gettext as _, gettext_lazy
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.i18n import JavaScriptCatalog

all_sites = WeakSet()


class AlreadyRegistered(Exception):
    pass


class NotRegistered(Exception):
    pass
```
### 9 - django/contrib/admin/checks.py:

Start line: 353, End line: 369

```python
class BaseModelAdminChecks:

    def _check_exclude(self, obj):
        """ Check that exclude is a sequence without duplicates. """

        if obj.exclude is None:  # default value is None
            return []
        elif not isinstance(obj.exclude, (list, tuple)):
            return must_be('a list or tuple', option='exclude', obj=obj, id='admin.E014')
        elif len(obj.exclude) > len(set(obj.exclude)):
            return [
                checks.Error(
                    "The value of 'exclude' contains duplicate field(s).",
                    obj=obj.__class__,
                    id='admin.E015',
                )
            ]
        else:
            return []
```
### 10 - django/contrib/admin/sites.py:

Start line: 134, End line: 194

```python
class AdminSite:

    def unregister(self, model_or_iterable):
        """
        Unregister the given model(s).

        If a model isn't already registered, raise NotRegistered.
        """
        if isinstance(model_or_iterable, ModelBase):
            model_or_iterable = [model_or_iterable]
        for model in model_or_iterable:
            if model not in self._registry:
                raise NotRegistered('The model %s is not registered' % model.__name__)
            del self._registry[model]

    def is_registered(self, model):
        """
        Check if a model class is registered with this `AdminSite`.
        """
        return model in self._registry

    def add_action(self, action, name=None):
        """
        Register an action to be available globally.
        """
        name = name or action.__name__
        self._actions[name] = action
        self._global_actions[name] = action

    def disable_action(self, name):
        """
        Disable a globally-registered action. Raise KeyError for invalid names.
        """
        del self._actions[name]

    def get_action(self, name):
        """
        Explicitly get a registered global action whether it's enabled or
        not. Raise KeyError for invalid names.
        """
        return self._global_actions[name]

    @property
    def actions(self):
        """
        Get all the enabled actions as an iterable of (name, func).
        """
        return self._actions.items()

    @property
    def empty_value_display(self):
        return self._empty_value_display

    @empty_value_display.setter
    def empty_value_display(self, empty_value_display):
        self._empty_value_display = empty_value_display

    def has_permission(self, request):
        """
        Return True if the given HttpRequest has permission to view
        *at least one* page in the admin site.
        """
        return request.user.is_active and request.user.is_staff
```
### 12 - django/contrib/admin/checks.py:

Start line: 230, End line: 260

```python
class BaseModelAdminChecks:

    def _check_fields(self, obj):
        """ Check that `fields` only refer to existing fields, doesn't contain
        duplicates. Check if at most one of `fields` and `fieldsets` is defined.
        """

        if obj.fields is None:
            return []
        elif not isinstance(obj.fields, (list, tuple)):
            return must_be('a list or tuple', option='fields', obj=obj, id='admin.E004')
        elif obj.fieldsets:
            return [
                checks.Error(
                    "Both 'fieldsets' and 'fields' are specified.",
                    obj=obj.__class__,
                    id='admin.E005',
                )
            ]
        fields = flatten(obj.fields)
        if len(fields) != len(set(fields)):
            return [
                checks.Error(
                    "The value of 'fields' contains duplicate field(s).",
                    obj=obj.__class__,
                    id='admin.E006',
                )
            ]

        return list(chain.from_iterable(
            self._check_field_spec(obj, field_name, 'fields')
            for field_name in obj.fields
        ))
```
### 18 - django/contrib/admin/checks.py:

Start line: 879, End line: 927

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_editable_item(self, obj, field_name, label):
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E121')
        else:
            if field_name not in obj.list_display:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not "
                        "contained in 'list_display'." % (label, field_name),
                        obj=obj.__class__,
                        id='admin.E122',
                    )
                ]
            elif obj.list_display_links and field_name in obj.list_display_links:
                return [
                    checks.Error(
                        "The value of '%s' cannot be in both 'list_editable' and 'list_display_links'." % field_name,
                        obj=obj.__class__,
                        id='admin.E123',
                    )
                ]
            # If list_display[0] is in list_editable, check that
            # list_display_links is set. See #22792 and #26229 for use cases.
            elif (obj.list_display[0] == field_name and not obj.list_display_links and
                    obj.list_display_links is not None):
                return [
                    checks.Error(
                        "The value of '%s' refers to the first field in 'list_display' ('%s'), "
                        "which cannot be used unless 'list_display_links' is set." % (
                            label, obj.list_display[0]
                        ),
                        obj=obj.__class__,
                        id='admin.E124',
                    )
                ]
            elif not field.editable:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not editable through the admin." % (
                            label, field_name
                        ),
                        obj=obj.__class__,
                        id='admin.E125',
                    )
                ]
            else:
                return []
```
### 19 - django/contrib/admin/checks.py:

Start line: 203, End line: 213

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields(self, obj):
        """ Check that `raw_id_fields` only contains field names that are listed
        on the model. """

        if not isinstance(obj.raw_id_fields, (list, tuple)):
            return must_be('a list or tuple', option='raw_id_fields', obj=obj, id='admin.E001')
        else:
            return list(chain.from_iterable(
                self._check_raw_id_fields_item(obj, field_name, 'raw_id_fields[%d]' % index)
                for index, field_name in enumerate(obj.raw_id_fields)
            ))
```
### 27 - django/contrib/admin/checks.py:

Start line: 620, End line: 639

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
### 30 - django/contrib/admin/checks.py:

Start line: 1013, End line: 1040

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
                    "to the parent model '%s.%s'." % (
                        fk.name, parent_model._meta.app_label, parent_model._meta.object_name
                    ),
                    obj=obj.__class__,
                    id='admin.E201',
                )
            ]
        else:
            return []
```
### 33 - django/contrib/admin/checks.py:

Start line: 129, End line: 146

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
### 36 - django/contrib/admin/checks.py:

Start line: 751, End line: 766

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links(self, obj):
        """ Check that list_display_links is a unique subset of list_display.
        """
        from django.contrib.admin.options import ModelAdmin

        if obj.list_display_links is None:
            return []
        elif not isinstance(obj.list_display_links, (list, tuple)):
            return must_be('a list, a tuple, or None', option='list_display_links', obj=obj, id='admin.E110')
        # Check only if ModelAdmin.get_list_display() isn't overridden.
        elif obj.get_list_display.__func__ is ModelAdmin.get_list_display:
            return list(chain.from_iterable(
                self._check_list_display_links_item(obj, field_name, "list_display_links[%d]" % index)
                for index, field_name in enumerate(obj.list_display_links)
            ))
        return []
```
### 43 - django/contrib/admin/checks.py:

Start line: 768, End line: 789

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display_links_item(self, obj, field_name, label):
        if field_name not in obj.list_display:
            return [
                checks.Error(
                    "The value of '%s' refers to '%s', which is not defined in 'list_display'." % (
                        label, field_name
                    ),
                    obj=obj.__class__,
                    id='admin.E111',
                )
            ]
        else:
            return []

    def _check_list_filter(self, obj):
        if not isinstance(obj.list_filter, (list, tuple)):
            return must_be('a list or tuple', option='list_filter', obj=obj, id='admin.E112')
        else:
            return list(chain.from_iterable(
                self._check_list_filter_item(obj, item, "list_filter[%d]" % index)
                for index, item in enumerate(obj.list_filter)
            ))
```
### 45 - django/contrib/admin/checks.py:

Start line: 1087, End line: 1117

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
            "The value of '%s' refers to '%s', which is not an attribute of '%s.%s'." % (
                option, field, obj.model._meta.app_label, obj.model._meta.object_name
            ),
            obj=obj.__class__,
            id=id,
        ),
    ]
```
### 46 - django/contrib/admin/checks.py:

Start line: 215, End line: 228

```python
class BaseModelAdminChecks:

    def _check_raw_id_fields_item(self, obj, field_name, label):
        """ Check an item of `raw_id_fields`, i.e. check that field named
        `field_name` exists in model `model` and is a ForeignKey or a
        ManyToManyField. """

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E002')
        else:
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be('a foreign key or a many-to-many field', option=label, obj=obj, id='admin.E003')
            else:
                return []
```
### 53 - django/contrib/admin/checks.py:

Start line: 583, End line: 594

```python
class BaseModelAdminChecks:

    def _check_readonly_fields(self, obj):
        """ Check that readonly_fields refers to proper attribute or field. """

        if obj.readonly_fields == ():
            return []
        elif not isinstance(obj.readonly_fields, (list, tuple)):
            return must_be('a list or tuple', option='readonly_fields', obj=obj, id='admin.E034')
        else:
            return list(chain.from_iterable(
                self._check_readonly_fields_item(obj, field_name, "readonly_fields[%d]" % index)
                for index, field_name in enumerate(obj.readonly_fields)
            ))
```
### 55 - django/contrib/admin/checks.py:

Start line: 262, End line: 275

```python
class BaseModelAdminChecks:

    def _check_fieldsets(self, obj):
        """ Check that fieldsets is properly formatted and doesn't contain
        duplicates. """

        if obj.fieldsets is None:
            return []
        elif not isinstance(obj.fieldsets, (list, tuple)):
            return must_be('a list or tuple', option='fieldsets', obj=obj, id='admin.E007')
        else:
            seen_fields = []
            return list(chain.from_iterable(
                self._check_fieldsets_item(obj, fieldset, 'fieldsets[%d]' % index, seen_fields)
                for index, fieldset in enumerate(obj.fieldsets)
            ))
```
### 58 - django/contrib/admin/checks.py:

Start line: 148, End line: 158

```python
class BaseModelAdminChecks:

    def _check_autocomplete_fields(self, obj):
        """
        Check that `autocomplete_fields` is a list or tuple of model fields.
        """
        if not isinstance(obj.autocomplete_fields, (list, tuple)):
            return must_be('a list or tuple', option='autocomplete_fields', obj=obj, id='admin.E036')
        else:
            return list(chain.from_iterable([
                self._check_autocomplete_fields_item(obj, field_name, 'autocomplete_fields[%d]' % index)
                for index, field_name in enumerate(obj.autocomplete_fields)
            ]))
```
### 60 - django/contrib/admin/checks.py:

Start line: 1042, End line: 1084

```python
class InlineModelAdminChecks(BaseModelAdminChecks):

    def _check_relation(self, obj, parent_model):
        try:
            _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)
        except ValueError as e:
            return [checks.Error(e.args[0], obj=obj.__class__, id='admin.E202')]
        else:
            return []

    def _check_extra(self, obj):
        """ Check that extra is an integer. """

        if not isinstance(obj.extra, int):
            return must_be('an integer', option='extra', obj=obj, id='admin.E203')
        else:
            return []

    def _check_max_num(self, obj):
        """ Check that max_num is an integer. """

        if obj.max_num is None:
            return []
        elif not isinstance(obj.max_num, int):
            return must_be('an integer', option='max_num', obj=obj, id='admin.E204')
        else:
            return []

    def _check_min_num(self, obj):
        """ Check that min_num is an integer. """

        if obj.min_num is None:
            return []
        elif not isinstance(obj.min_num, int):
            return must_be('an integer', option='min_num', obj=obj, id='admin.E205')
        else:
            return []

    def _check_formset(self, obj):
        """ Check formset is a subclass of BaseModelFormSet. """

        if not _issubclass(obj.formset, BaseModelFormSet):
            return must_inherit_from(parent='BaseModelFormSet', option='formset', obj=obj, id='admin.E206')
        else:
            return []
```
### 64 - django/contrib/admin/checks.py:

Start line: 718, End line: 749

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
                        "method on '%s.%s'." % (
                            label, item, obj.__class__.__name__,
                            obj.model._meta.app_label, obj.model._meta.object_name,
                        ),
                        obj=obj.__class__,
                        id='admin.E108',
                    )
                ]
        if isinstance(field, models.ManyToManyField):
            return [
                checks.Error(
                    "The value of '%s' must not be a ManyToManyField." % label,
                    obj=obj.__class__,
                    id='admin.E109',
                )
            ]
        return []
```
### 66 - django/contrib/admin/checks.py:

Start line: 641, End line: 668

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_save_as(self, obj):
        """ Check save_as is a boolean. """

        if not isinstance(obj.save_as, bool):
            return must_be('a boolean', option='save_as',
                           obj=obj, id='admin.E101')
        else:
            return []

    def _check_save_on_top(self, obj):
        """ Check save_on_top is a boolean. """

        if not isinstance(obj.save_on_top, bool):
            return must_be('a boolean', option='save_on_top',
                           obj=obj, id='admin.E102')
        else:
            return []

    def _check_inlines(self, obj):
        """ Check all inline model admin classes. """

        if not isinstance(obj.inlines, (list, tuple)):
            return must_be('a list or tuple', option='inlines', obj=obj, id='admin.E103')
        else:
            return list(chain.from_iterable(
                self._check_inlines_item(obj, item, "inlines[%d]" % index)
                for index, item in enumerate(obj.inlines)
            ))
```
### 67 - django/contrib/admin/checks.py:

Start line: 843, End line: 865

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_select_related(self, obj):
        """ Check that list_select_related is a boolean, a list or a tuple. """

        if not isinstance(obj.list_select_related, (bool, list, tuple)):
            return must_be('a boolean, tuple or list', option='list_select_related', obj=obj, id='admin.E117')
        else:
            return []

    def _check_list_per_page(self, obj):
        """ Check that list_per_page is an integer. """

        if not isinstance(obj.list_per_page, int):
            return must_be('an integer', option='list_per_page', obj=obj, id='admin.E118')
        else:
            return []

    def _check_list_max_show_all(self, obj):
        """ Check that list_max_show_all is an integer. """

        if not isinstance(obj.list_max_show_all, int):
            return must_be('an integer', option='list_max_show_all', obj=obj, id='admin.E119')
        else:
            return []
```
### 69 - django/contrib/admin/checks.py:

Start line: 706, End line: 716

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_list_display(self, obj):
        """ Check that list_display only contains fields or usable attributes.
        """

        if not isinstance(obj.list_display, (list, tuple)):
            return must_be('a list or tuple', option='list_display', obj=obj, id='admin.E107')
        else:
            return list(chain.from_iterable(
                self._check_list_display_item(obj, item, "list_display[%d]" % index)
                for index, item in enumerate(obj.list_display)
            ))
```
### 75 - django/contrib/admin/checks.py:

Start line: 160, End line: 201

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
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E037')
        else:
            if not field.many_to_many and not isinstance(field, models.ForeignKey):
                return must_be(
                    'a foreign key or a many-to-many field',
                    option=label, obj=obj, id='admin.E038'
                )
            related_admin = obj.admin_site._registry.get(field.remote_field.model)
            if related_admin is None:
                return [
                    checks.Error(
                        'An admin for model "%s" has to be registered '
                        'to be referenced by %s.autocomplete_fields.' % (
                            field.remote_field.model.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id='admin.E039',
                    )
                ]
            elif not related_admin.search_fields:
                return [
                    checks.Error(
                        '%s must define "search_fields", because it\'s '
                        'referenced by %s.autocomplete_fields.' % (
                            related_admin.__class__.__name__,
                            type(obj).__name__,
                        ),
                        obj=obj.__class__,
                        id='admin.E040',
                    )
                ]
            return []
```
### 79 - django/contrib/admin/checks.py:

Start line: 424, End line: 445

```python
class BaseModelAdminChecks:

    def _check_radio_fields_key(self, obj, field_name, label):
        """ Check that a key of `radio_fields` dictionary is name of existing
        field and that the field is a ForeignKey or has `choices` defined. """

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E022')
        else:
            if not (isinstance(field, models.ForeignKey) or field.choices):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not an "
                        "instance of ForeignKey, and does not have a 'choices' definition." % (
                            label, field_name
                        ),
                        obj=obj.__class__,
                        id='admin.E023',
                    )
                ]
            else:
                return []
```
### 84 - django/contrib/admin/checks.py:

Start line: 325, End line: 351

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
                if (isinstance(field, models.ManyToManyField) and
                        not field.remote_field.through._meta.auto_created):
                    return [
                        checks.Error(
                            "The value of '%s' cannot include the ManyToManyField '%s', "
                            "because that field manually specifies a relationship model."
                            % (label, field_name),
                            obj=obj.__class__,
                            id='admin.E013',
                        )
                    ]
                else:
                    return []
```
### 90 - django/contrib/admin/checks.py:

Start line: 1, End line: 54

```python
from itertools import chain

from django.apps import apps
from django.conf import settings
from django.contrib.admin.utils import (
    NotRelationField, flatten, get_fields_from_path,
)
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import Combinable, F, OrderBy
from django.forms.models import (
    BaseModelForm, BaseModelFormSet, _get_foreign_key,
)
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils.module_loading import import_string


def _issubclass(cls, classinfo):
    """
    issubclass() variant that doesn't raise an exception if cls isn't a
    class.
    """
    try:
        return issubclass(cls, classinfo)
    except TypeError:
        return False


def _contains_subclass(class_path, candidate_paths):
    """
    Return whether or not a dotted class path (or a subclass of that class) is
    found in a list of candidate paths.
    """
    cls = import_string(class_path)
    for path in candidate_paths:
        try:
            candidate_cls = import_string(path)
        except ImportError:
            # ImportErrors are raised elsewhere.
            continue
        if _issubclass(candidate_cls, cls):
            return True
    return False


def check_admin_app(app_configs, **kwargs):
    from django.contrib.admin.sites import all_sites
    errors = []
    for site in all_sites:
        errors.extend(site.check(app_configs))
    return errors
```
### 96 - django/contrib/admin/checks.py:

Start line: 521, End line: 544

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_value_item(self, obj, field_name, label):
        """ For `prepopulated_fields` equal to {"slug": ("title",)},
        `field_name` is "title". """

        try:
            obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E030')
        else:
            return []

    def _check_ordering(self, obj):
        """ Check that ordering refers to existing fields or is random. """

        # ordering = None
        if obj.ordering is None:  # The default value is None
            return []
        elif not isinstance(obj.ordering, (list, tuple)):
            return must_be('a list or tuple', option='ordering', obj=obj, id='admin.E031')
        else:
            return list(chain.from_iterable(
                self._check_ordering_item(obj, field_name, 'ordering[%d]' % index)
                for index, field_name in enumerate(obj.ordering)
            ))
```
### 102 - django/contrib/admin/checks.py:

Start line: 596, End line: 617

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
                        "The value of '%s' is not a callable, an attribute of '%s', or an attribute of '%s.%s'." % (
                            label, obj.__class__.__name__, obj.model._meta.app_label, obj.model._meta.object_name
                        ),
                        obj=obj.__class__,
                        id='admin.E035',
                    )
                ]
            else:
                return []
```
### 109 - django/contrib/admin/checks.py:

Start line: 546, End line: 581

```python
class BaseModelAdminChecks:

    def _check_ordering_item(self, obj, field_name, label):
        """ Check that `ordering` refers to existing fields. """
        if isinstance(field_name, (Combinable, OrderBy)):
            if not isinstance(field_name, OrderBy):
                field_name = field_name.asc()
            if isinstance(field_name.expression, F):
                field_name = field_name.expression.name
            else:
                return []
        if field_name == '?' and len(obj.ordering) != 1:
            return [
                checks.Error(
                    "The value of 'ordering' has the random ordering marker '?', "
                    "but contains other fields as well.",
                    hint='Either remove the "?", or remove the other fields.',
                    obj=obj.__class__,
                    id='admin.E032',
                )
            ]
        elif field_name == '?':
            return []
        elif LOOKUP_SEP in field_name:
            # Skip ordering in the format field1__field2 (FIXME: checking
            # this format would be nice, but it's a little fiddly).
            return []
        else:
            if field_name.startswith('-'):
                field_name = field_name[1:]
            if field_name == 'pk':
                return []
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E033')
            else:
                return []
```
### 117 - django/contrib/admin/checks.py:

Start line: 371, End line: 397

```python
class BaseModelAdminChecks:

    def _check_form(self, obj):
        """ Check that form subclasses BaseModelForm. """
        if not _issubclass(obj.form, BaseModelForm):
            return must_inherit_from(parent='BaseModelForm', option='form',
                                     obj=obj, id='admin.E016')
        else:
            return []

    def _check_filter_vertical(self, obj):
        """ Check that filter_vertical is a sequence of field names. """
        if not isinstance(obj.filter_vertical, (list, tuple)):
            return must_be('a list or tuple', option='filter_vertical', obj=obj, id='admin.E017')
        else:
            return list(chain.from_iterable(
                self._check_filter_item(obj, field_name, "filter_vertical[%d]" % index)
                for index, field_name in enumerate(obj.filter_vertical)
            ))

    def _check_filter_horizontal(self, obj):
        """ Check that filter_horizontal is a sequence of field names. """
        if not isinstance(obj.filter_horizontal, (list, tuple)):
            return must_be('a list or tuple', option='filter_horizontal', obj=obj, id='admin.E018')
        else:
            return list(chain.from_iterable(
                self._check_filter_item(obj, field_name, "filter_horizontal[%d]" % index)
                for index, field_name in enumerate(obj.filter_horizontal)
            ))
```
### 118 - django/contrib/admin/checks.py:

Start line: 929, End line: 958

```python
class ModelAdminChecks(BaseModelAdminChecks):

    def _check_search_fields(self, obj):
        """ Check search_fields is a sequence. """

        if not isinstance(obj.search_fields, (list, tuple)):
            return must_be('a list or tuple', option='search_fields', obj=obj, id='admin.E126')
        else:
            return []

    def _check_date_hierarchy(self, obj):
        """ Check that date_hierarchy refers to DateField or DateTimeField. """

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
                        id='admin.E127',
                    )
                ]
            else:
                if not isinstance(field, (models.DateField, models.DateTimeField)):
                    return must_be('a DateField or DateTimeField', option='date_hierarchy', obj=obj, id='admin.E128')
                else:
                    return []
```
### 119 - django/contrib/admin/checks.py:

Start line: 487, End line: 507

```python
class BaseModelAdminChecks:

    def _check_prepopulated_fields_key(self, obj, field_name, label):
        """ Check a key of `prepopulated_fields` dictionary, i.e. check that it
        is a name of existing field and the field is one of the allowed types.
        """

        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E027')
        else:
            if isinstance(field, (models.DateTimeField, models.ForeignKey, models.ManyToManyField)):
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which must not be a DateTimeField, "
                        "a ForeignKey, a OneToOneField, or a ManyToManyField." % (label, field_name),
                        obj=obj.__class__,
                        id='admin.E028',
                    )
                ]
            else:
                return []
```
