# django__django-11727

| **django/django** | `44077985f58be02214a11ffde35776fed3c960e1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 343 |
| **Any found context length** | 343 |
| **Avg pos** | 2.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/admin/templatetags/admin_modify.py b/django/contrib/admin/templatetags/admin_modify.py
--- a/django/contrib/admin/templatetags/admin_modify.py
+++ b/django/contrib/admin/templatetags/admin_modify.py
@@ -54,12 +54,20 @@ def submit_row(context):
     is_popup = context['is_popup']
     save_as = context['save_as']
     show_save = context.get('show_save', True)
+    show_save_and_add_another = context.get('show_save_and_add_another', True)
     show_save_and_continue = context.get('show_save_and_continue', True)
     has_add_permission = context['has_add_permission']
     has_change_permission = context['has_change_permission']
     has_view_permission = context['has_view_permission']
     has_editable_inline_admin_formsets = context['has_editable_inline_admin_formsets']
     can_save = (has_change_permission and change) or (has_add_permission and add) or has_editable_inline_admin_formsets
+    can_save_and_add_another = (
+        has_add_permission and
+        not is_popup and
+        (not save_as or add) and
+        can_save and
+        show_save_and_add_another
+    )
     can_save_and_continue = not is_popup and can_save and has_view_permission and show_save_and_continue
     can_change = has_change_permission or has_editable_inline_admin_formsets
     ctx = Context(context)
@@ -70,10 +78,7 @@ def submit_row(context):
             change and context.get('show_delete', True)
         ),
         'show_save_as_new': not is_popup and has_change_permission and change and save_as,
-        'show_save_and_add_another': (
-            has_add_permission and not is_popup and
-            (not save_as or add) and can_save
-        ),
+        'show_save_and_add_another': can_save_and_add_another,
         'show_save_and_continue': can_save_and_continue,
         'show_save': show_save and can_save,
         'show_close': not(show_save and can_save)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/admin/templatetags/admin_modify.py | 57 | 57 | 1 | 1 | 343
| django/contrib/admin/templatetags/admin_modify.py | 73 | 76 | 1 | 1 | 343


## Problem Statement

```
Allow hiding the "Save and Add Another" button with a `show_save_and_add_another` context variable
Description
	
To provide better adjustability, to introduce new context var - show_save_and_add_another.
E.g. if I want to hide button "Save and add another", I can just modify extra_context - write False to the variable.
For other buttons - "Save" and "Save and continue editing", this already works exactly in this manner.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/admin/templatetags/admin_modify.py** | 48 | 81| 343 | 343 | 920 | 
| 2 | 2 django/contrib/admin/options.py | 1597 | 1621| 279 | 622 | 19286 | 
| 3 | 3 django/template/defaulttags.py | 1442 | 1475| 246 | 868 | 30326 | 
| 4 | 4 django/template/context.py | 135 | 169| 288 | 1156 | 32215 | 
| 5 | 4 django/contrib/admin/options.py | 1517 | 1596| 719 | 1875 | 32215 | 
| 6 | 4 django/contrib/admin/options.py | 1310 | 1335| 232 | 2107 | 32215 | 
| 7 | 5 django/forms/widgets.py | 337 | 371| 278 | 2385 | 40281 | 
| 8 | 6 django/contrib/auth/forms.py | 23 | 41| 161 | 2546 | 43327 | 
| 9 | 6 django/template/context.py | 27 | 132| 663 | 3209 | 43327 | 
| 10 | 7 django/contrib/admindocs/views.py | 249 | 314| 585 | 3794 | 46637 | 
| 11 | 8 django/template/base.py | 726 | 789| 531 | 4325 | 54498 | 
| 12 | 9 django/db/migrations/questioner.py | 207 | 224| 171 | 4496 | 56572 | 
| 13 | 9 django/contrib/admin/options.py | 1111 | 1156| 482 | 4978 | 56572 | 
| 14 | 10 django/template/context_processors.py | 35 | 49| 126 | 5104 | 57061 | 
| 15 | 10 django/contrib/admin/options.py | 1725 | 1806| 744 | 5848 | 57061 | 
| 16 | 10 django/forms/widgets.py | 811 | 840| 265 | 6113 | 57061 | 
| 17 | 11 django/contrib/admin/widgets.py | 200 | 226| 216 | 6329 | 60927 | 
| 18 | 11 django/contrib/admindocs/views.py | 182 | 248| 584 | 6913 | 60927 | 
| 19 | 11 django/contrib/admin/options.py | 1996 | 2017| 221 | 7134 | 60927 | 
| 20 | 12 django/views/generic/edit.py | 1 | 67| 479 | 7613 | 62643 | 
| 21 | 12 django/template/base.py | 791 | 813| 190 | 7803 | 62643 | 
| 22 | 12 django/contrib/admin/options.py | 1808 | 1876| 584 | 8387 | 62643 | 
| 23 | 12 django/contrib/admin/widgets.py | 278 | 313| 382 | 8769 | 62643 | 
| 24 | 12 django/contrib/admin/widgets.py | 229 | 276| 423 | 9192 | 62643 | 
| 25 | 13 django/forms/models.py | 444 | 471| 227 | 9419 | 74138 | 
| 26 | 14 django/db/models/base.py | 744 | 793| 456 | 9875 | 89328 | 
| 27 | 14 django/template/context.py | 172 | 212| 296 | 10171 | 89328 | 
| 28 | 15 docs/_ext/djangodocs.py | 276 | 377| 846 | 11017 | 92401 | 
| 29 | 15 django/template/context.py | 1 | 24| 128 | 11145 | 92401 | 
| 30 | 16 django/views/generic/base.py | 1 | 27| 148 | 11293 | 94004 | 
| 31 | 16 django/template/context.py | 235 | 262| 199 | 11492 | 94004 | 
| 32 | 16 django/template/context_processors.py | 52 | 82| 143 | 11635 | 94004 | 
| 33 | 17 django/contrib/admin/checks.py | 641 | 668| 232 | 11867 | 103020 | 
| 34 | 18 django/contrib/admin/helpers.py | 192 | 220| 276 | 12143 | 106213 | 
| 35 | 18 django/forms/widgets.py | 639 | 668| 238 | 12381 | 106213 | 
| 36 | 18 django/forms/widgets.py | 426 | 438| 122 | 12503 | 106213 | 
| 37 | 18 django/template/defaulttags.py | 249 | 259| 140 | 12643 | 106213 | 
| 38 | 19 django/views/decorators/debug.py | 1 | 38| 218 | 12861 | 106686 | 
| 39 | 20 django/views/debug.py | 193 | 241| 462 | 13323 | 110904 | 
| 40 | 20 django/views/debug.py | 125 | 152| 248 | 13571 | 110904 | 
| 41 | 20 django/contrib/admin/widgets.py | 125 | 164| 349 | 13920 | 110904 | 
| 42 | 21 django/contrib/auth/admin.py | 101 | 126| 286 | 14206 | 112630 | 
| 43 | 21 django/forms/widgets.py | 791 | 809| 153 | 14359 | 112630 | 
| 44 | 21 django/db/migrations/questioner.py | 227 | 240| 123 | 14482 | 112630 | 
| 45 | 21 django/template/base.py | 815 | 879| 535 | 15017 | 112630 | 
| 46 | 21 django/forms/models.py | 815 | 856| 449 | 15466 | 112630 | 
| 47 | 22 django/template/loader_tags.py | 272 | 318| 392 | 15858 | 115181 | 
| 48 | 23 django/contrib/postgres/forms/array.py | 133 | 165| 265 | 16123 | 116666 | 
| 49 | 23 django/template/defaulttags.py | 630 | 677| 331 | 16454 | 116666 | 
| 50 | 24 django/contrib/admin/sites.py | 291 | 312| 175 | 16629 | 120857 | 
| 51 | 24 django/contrib/admindocs/views.py | 155 | 179| 234 | 16863 | 120857 | 
| 52 | 25 django/template/library.py | 201 | 234| 304 | 17167 | 123395 | 
| 53 | 25 django/db/migrations/questioner.py | 143 | 160| 183 | 17350 | 123395 | 
| 54 | 25 django/forms/widgets.py | 972 | 1018| 445 | 17795 | 123395 | 
| 55 | 26 django/db/models/options.py | 262 | 294| 343 | 18138 | 130494 | 
| 56 | 27 django/contrib/auth/views.py | 330 | 362| 239 | 18377 | 133158 | 
| 57 | 28 django/contrib/auth/context_processors.py | 24 | 64| 247 | 18624 | 133572 | 
| 58 | 29 django/contrib/auth/hashers.py | 149 | 184| 245 | 18869 | 138359 | 
| 59 | 29 django/contrib/admin/options.py | 2135 | 2170| 315 | 19184 | 138359 | 
| 60 | 29 django/forms/models.py | 1337 | 1364| 209 | 19393 | 138359 | 
| 61 | 29 django/contrib/admin/options.py | 1653 | 1724| 653 | 20046 | 138359 | 
| 62 | 29 django/contrib/admin/options.py | 1445 | 1464| 133 | 20179 | 138359 | 
| 63 | 29 django/template/loader_tags.py | 126 | 150| 232 | 20411 | 138359 | 
| 64 | 29 docs/_ext/djangodocs.py | 172 | 205| 286 | 20697 | 138359 | 
| 65 | 29 django/contrib/admin/options.py | 1878 | 1915| 330 | 21027 | 138359 | 
| 66 | 29 django/views/generic/edit.py | 70 | 101| 269 | 21296 | 138359 | 
| 67 | 30 django/db/models/fields/reverse_related.py | 132 | 150| 172 | 21468 | 140473 | 
| 68 | 30 django/forms/widgets.py | 671 | 702| 261 | 21729 | 140473 | 
| 69 | 30 django/contrib/admindocs/views.py | 317 | 347| 208 | 21937 | 140473 | 
| 70 | 30 django/contrib/admin/options.py | 1963 | 1994| 249 | 22186 | 140473 | 
| 71 | 30 django/template/loader_tags.py | 153 | 188| 292 | 22478 | 140473 | 
| 72 | 31 django/db/models/fields/related_descriptors.py | 1082 | 1109| 338 | 22816 | 150805 | 
| 73 | 32 django/contrib/admin/__init__.py | 1 | 30| 286 | 23102 | 151091 | 
| 74 | 32 django/template/loader_tags.py | 1 | 38| 182 | 23284 | 151091 | 
| 75 | 32 django/forms/widgets.py | 912 | 921| 107 | 23391 | 151091 | 
| 76 | 32 django/contrib/admindocs/views.py | 135 | 153| 187 | 23578 | 151091 | 
| 77 | 33 django/contrib/auth/management/commands/createsuperuser.py | 204 | 228| 204 | 23782 | 153151 | 
| 78 | **33 django/contrib/admin/templatetags/admin_modify.py** | 1 | 45| 372 | 24154 | 153151 | 
| 79 | 33 django/contrib/admin/options.py | 542 | 585| 297 | 24451 | 153151 | 
| 80 | 33 django/forms/widgets.py | 298 | 334| 204 | 24655 | 153151 | 
| 81 | 33 django/contrib/auth/views.py | 247 | 284| 348 | 25003 | 153151 | 
| 82 | 34 django/contrib/syndication/views.py | 96 | 121| 180 | 25183 | 154877 | 
| 83 | 35 django/template/defaultfilters.py | 639 | 667| 212 | 25395 | 160942 | 
| 84 | 35 docs/_ext/djangodocs.py | 26 | 70| 385 | 25780 | 160942 | 
| 85 | 36 django/db/models/sql/query.py | 1954 | 1984| 260 | 26040 | 182476 | 
| 86 | 36 django/db/models/options.py | 363 | 385| 164 | 26204 | 182476 | 
| 87 | 36 django/template/context.py | 215 | 233| 185 | 26389 | 182476 | 
| 88 | 36 django/db/migrations/questioner.py | 109 | 141| 290 | 26679 | 182476 | 
| 89 | 37 django/db/models/expressions.py | 859 | 879| 143 | 26822 | 192753 | 
| 90 | 37 django/forms/models.py | 647 | 677| 217 | 27039 | 192753 | 
| 91 | 37 django/contrib/admin/sites.py | 331 | 351| 182 | 27221 | 192753 | 
| 92 | 38 django/db/migrations/operations/models.py | 625 | 677| 342 | 27563 | 199449 | 
| 93 | 38 django/db/migrations/operations/models.py | 488 | 515| 181 | 27744 | 199449 | 
| 94 | 38 django/contrib/admin/helpers.py | 1 | 30| 198 | 27942 | 199449 | 


### Hint

```
​PR
The options for Save and Save and continue were originally added to ensure correct behaviour when hitting Save New and getting a validation error. See ​0894643e4. They weren't really for direct usage... ... however, maybe it's reasonable. Tentatively Accept to see how the patch turns out, and what others think. This isn't really documented (like much around extending the admin). Not sure where it would go if we were to add something.
A ​Stackoverflow question suggests several alternatives including overriding the template tag or adding some CSS. I guess I'm okay with the patch (even if undocumented) but it needs a test.
Replying to Tim Graham: A ​Stackoverflow question suggests several alternatives including overriding the template tag or adding some CSS. I guess I'm okay with the patch (even if undocumented) but it needs a test. The test is the only thing missing? I would like to work on it
Hey Tim, Is this still available to work, if yes - I would like to add unit tests to it ?
```

## Patch

```diff
diff --git a/django/contrib/admin/templatetags/admin_modify.py b/django/contrib/admin/templatetags/admin_modify.py
--- a/django/contrib/admin/templatetags/admin_modify.py
+++ b/django/contrib/admin/templatetags/admin_modify.py
@@ -54,12 +54,20 @@ def submit_row(context):
     is_popup = context['is_popup']
     save_as = context['save_as']
     show_save = context.get('show_save', True)
+    show_save_and_add_another = context.get('show_save_and_add_another', True)
     show_save_and_continue = context.get('show_save_and_continue', True)
     has_add_permission = context['has_add_permission']
     has_change_permission = context['has_change_permission']
     has_view_permission = context['has_view_permission']
     has_editable_inline_admin_formsets = context['has_editable_inline_admin_formsets']
     can_save = (has_change_permission and change) or (has_add_permission and add) or has_editable_inline_admin_formsets
+    can_save_and_add_another = (
+        has_add_permission and
+        not is_popup and
+        (not save_as or add) and
+        can_save and
+        show_save_and_add_another
+    )
     can_save_and_continue = not is_popup and can_save and has_view_permission and show_save_and_continue
     can_change = has_change_permission or has_editable_inline_admin_formsets
     ctx = Context(context)
@@ -70,10 +78,7 @@ def submit_row(context):
             change and context.get('show_delete', True)
         ),
         'show_save_as_new': not is_popup and has_change_permission and change and save_as,
-        'show_save_and_add_another': (
-            has_add_permission and not is_popup and
-            (not save_as or add) and can_save
-        ),
+        'show_save_and_add_another': can_save_and_add_another,
         'show_save_and_continue': can_save_and_continue,
         'show_save': show_save and can_save,
         'show_close': not(show_save and can_save)

```

## Test Patch

```diff
diff --git a/tests/admin_views/test_templatetags.py b/tests/admin_views/test_templatetags.py
--- a/tests/admin_views/test_templatetags.py
+++ b/tests/admin_views/test_templatetags.py
@@ -29,6 +29,25 @@ def test_submit_row(self):
         self.assertIs(template_context['extra'], True)
         self.assertIs(template_context['show_save'], True)
 
+    def test_override_show_save_and_add_another(self):
+        request = self.request_factory.get(
+            reverse('admin:auth_user_change', args=[self.superuser.pk]),
+        )
+        request.user = self.superuser
+        admin = UserAdmin(User, site)
+        for extra_context, expected_flag in (
+            ({}, True),  # Default.
+            ({'show_save_and_add_another': False}, False),
+        ):
+            with self.subTest(show_save_and_add_another=expected_flag):
+                response = admin.change_view(
+                    request,
+                    str(self.superuser.pk),
+                    extra_context=extra_context,
+                )
+                template_context = submit_row(response.context_data)
+                self.assertIs(template_context['show_save_and_add_another'], expected_flag)
+
     def test_override_change_form_template_tags(self):
         """
         admin_modify template tags follow the standard search pattern

```


## Code snippets

### 1 - django/contrib/admin/templatetags/admin_modify.py:

Start line: 48, End line: 81

```python
def submit_row(context):
    """
    Display the row of buttons for delete and save.
    """
    add = context['add']
    change = context['change']
    is_popup = context['is_popup']
    save_as = context['save_as']
    show_save = context.get('show_save', True)
    show_save_and_continue = context.get('show_save_and_continue', True)
    has_add_permission = context['has_add_permission']
    has_change_permission = context['has_change_permission']
    has_view_permission = context['has_view_permission']
    has_editable_inline_admin_formsets = context['has_editable_inline_admin_formsets']
    can_save = (has_change_permission and change) or (has_add_permission and add) or has_editable_inline_admin_formsets
    can_save_and_continue = not is_popup and can_save and has_view_permission and show_save_and_continue
    can_change = has_change_permission or has_editable_inline_admin_formsets
    ctx = Context(context)
    ctx.update({
        'can_change': can_change,
        'show_delete_link': (
            not is_popup and context['has_delete_permission'] and
            change and context.get('show_delete', True)
        ),
        'show_save_as_new': not is_popup and has_change_permission and change and save_as,
        'show_save_and_add_another': (
            has_add_permission and not is_popup and
            (not save_as or add) and can_save
        ),
        'show_save_and_continue': can_save_and_continue,
        'show_save': show_save and can_save,
        'show_close': not(show_save and can_save)
    })
    return ctx
```
### 2 - django/contrib/admin/options.py:

Start line: 1597, End line: 1621

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        # ... other code
        context = {
            **self.admin_site.each_context(request),
            'title': title % opts.verbose_name,
            'adminform': adminForm,
            'object_id': object_id,
            'original': obj,
            'is_popup': IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,
            'to_field': to_field,
            'media': media,
            'inline_admin_formsets': inline_formsets,
            'errors': helpers.AdminErrorList(form, formsets),
            'preserved_filters': self.get_preserved_filters(request),
        }

        # Hide the "Save" and "Save and continue" buttons if "Save as New" was
        # previously chosen to prevent the interface from getting confusing.
        if request.method == 'POST' and not form_validated and "_saveasnew" in request.POST:
            context['show_save'] = False
            context['show_save_and_continue'] = False
            # Use the change template instead of the add template.
            add = False

        context.update(extra_context or {})

        return self.render_change_form(request, context, add=add, change=not add, obj=obj, form_url=form_url)
```
### 3 - django/template/defaulttags.py:

Start line: 1442, End line: 1475

```python
@register.tag('with')
def do_with(parser, token):
    """
    Add one or more values to the context (inside of this block) for caching
    and easy access.

    For example::

        {% with total=person.some_sql_method %}
            {{ total }} object{{ total|pluralize }}
        {% endwith %}

    Multiple values can be added to the context::

        {% with foo=1 bar=2 %}
            ...
        {% endwith %}

    The legacy format of ``{% with person.some_sql_method as total %}`` is
    still accepted.
    """
    bits = token.split_contents()
    remaining_bits = bits[1:]
    extra_context = token_kwargs(remaining_bits, parser, support_legacy=True)
    if not extra_context:
        raise TemplateSyntaxError("%r expected at least one variable "
                                  "assignment" % bits[0])
    if remaining_bits:
        raise TemplateSyntaxError("%r received an invalid token: %r" %
                                  (bits[0], remaining_bits[0]))
    nodelist = parser.parse(('endwith',))
    parser.delete_first_token()
    return WithNode(None, None, nodelist, extra_context=extra_context)
```
### 4 - django/template/context.py:

Start line: 135, End line: 169

```python
class Context(BaseContext):
    "A stack container for variable context"
    def __init__(self, dict_=None, autoescape=True, use_l10n=None, use_tz=None):
        self.autoescape = autoescape
        self.use_l10n = use_l10n
        self.use_tz = use_tz
        self.template_name = "unknown"
        self.render_context = RenderContext()
        # Set to the original template -- as opposed to extended or included
        # templates -- during rendering, see bind_template.
        self.template = None
        super().__init__(dict_)

    @contextmanager
    def bind_template(self, template):
        if self.template is not None:
            raise RuntimeError("Context is already bound to a template")
        self.template = template
        try:
            yield
        finally:
            self.template = None

    def __copy__(self):
        duplicate = super().__copy__()
        duplicate.render_context = copy(self.render_context)
        return duplicate

    def update(self, other_dict):
        "Push other_dict to the stack of dictionaries in the Context"
        if not hasattr(other_dict, '__getitem__'):
            raise TypeError('other_dict must be a mapping (dictionary-like) object.')
        if isinstance(other_dict, BaseContext):
            other_dict = other_dict.dicts[1:].pop()
        return ContextDict(self, other_dict)
```
### 5 - django/contrib/admin/options.py:

Start line: 1517, End line: 1596

```python
class ModelAdmin(BaseModelAdmin):

    def _changeform_view(self, request, object_id, form_url, extra_context):
        to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))
        if to_field and not self.to_field_allowed(request, to_field):
            raise DisallowedModelAdminToField("The field %s cannot be referenced." % to_field)

        model = self.model
        opts = model._meta

        if request.method == 'POST' and '_saveasnew' in request.POST:
            object_id = None

        add = object_id is None

        if add:
            if not self.has_add_permission(request):
                raise PermissionDenied
            obj = None

        else:
            obj = self.get_object(request, unquote(object_id), to_field)

            if not self.has_view_or_change_permission(request, obj):
                raise PermissionDenied

            if obj is None:
                return self._get_obj_does_not_exist_redirect(request, opts, object_id)

        ModelForm = self.get_form(request, obj, change=not add)
        if request.method == 'POST':
            form = ModelForm(request.POST, request.FILES, instance=obj)
            form_validated = form.is_valid()
            if form_validated:
                new_object = self.save_form(request, form, change=not add)
            else:
                new_object = form.instance
            formsets, inline_instances = self._create_formsets(request, new_object, change=not add)
            if all_valid(formsets) and form_validated:
                self.save_model(request, new_object, form, not add)
                self.save_related(request, form, formsets, not add)
                change_message = self.construct_change_message(request, form, formsets, add)
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
                formsets, inline_instances = self._create_formsets(request, form.instance, change=False)
            else:
                form = ModelForm(instance=obj)
                formsets, inline_instances = self._create_formsets(request, obj, change=True)

        if not add and not self.has_change_permission(request, obj):
            readonly_fields = flatten_fieldsets(self.get_fieldsets(request, obj))
        else:
            readonly_fields = self.get_readonly_fields(request, obj)
        adminForm = helpers.AdminForm(
            form,
            list(self.get_fieldsets(request, obj)),
            # Clear prepopulated fields on a view-only form to avoid a crash.
            self.get_prepopulated_fields(request, obj) if add or self.has_change_permission(request, obj) else {},
            readonly_fields,
            model_admin=self)
        media = self.media + adminForm.media

        inline_formsets = self.get_inline_formsets(request, formsets, inline_instances, obj)
        for inline_formset in inline_formsets:
            media = media + inline_formset.media

        if add:
            title = _('Add %s')
        elif self.has_change_permission(request, obj):
            title = _('Change %s')
        else:
            title = _('View %s')
        # ... other code
```
### 6 - django/contrib/admin/options.py:

Start line: 1310, End line: 1335

```python
class ModelAdmin(BaseModelAdmin):

    def _response_post_save(self, request, obj):
        opts = self.model._meta
        if self.has_view_or_change_permission(request):
            post_url = reverse('admin:%s_%s_changelist' %
                               (opts.app_label, opts.model_name),
                               current_app=self.admin_site.name)
            preserved_filters = self.get_preserved_filters(request)
            post_url = add_preserved_filters({'preserved_filters': preserved_filters, 'opts': opts}, post_url)
        else:
            post_url = reverse('admin:index',
                               current_app=self.admin_site.name)
        return HttpResponseRedirect(post_url)

    def response_post_save_add(self, request, obj):
        """
        Figure out where to redirect after the 'Save' button has been pressed
        when adding a new object.
        """
        return self._response_post_save(request, obj)

    def response_post_save_change(self, request, obj):
        """
        Figure out where to redirect after the 'Save' button has been pressed
        when editing an existing object.
        """
        return self._response_post_save(request, obj)
```
### 7 - django/forms/widgets.py:

Start line: 337, End line: 371

```python
class MultipleHiddenInput(HiddenInput):
    """
    Handle <input type="hidden"> for fields that have a list
    of values.
    """
    template_name = 'django/forms/widgets/multiple_hidden.html'

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        final_attrs = context['widget']['attrs']
        id_ = context['widget']['attrs'].get('id')

        subwidgets = []
        for index, value_ in enumerate(context['widget']['value']):
            widget_attrs = final_attrs.copy()
            if id_:
                # An ID attribute was given. Add a numeric index as a suffix
                # so that the inputs don't all have the same ID attribute.
                widget_attrs['id'] = '%s_%s' % (id_, index)
            widget = HiddenInput()
            widget.is_required = self.is_required
            subwidgets.append(widget.get_context(name, value_, widget_attrs)['widget'])

        context['widget']['subwidgets'] = subwidgets
        return context

    def value_from_datadict(self, data, files, name):
        try:
            getter = data.getlist
        except AttributeError:
            getter = data.get
        return getter(name)

    def format_value(self, value):
        return [] if value is None else value
```
### 8 - django/contrib/auth/forms.py:

Start line: 23, End line: 41

```python
class ReadOnlyPasswordHashWidget(forms.Widget):
    template_name = 'auth/widgets/read_only_password_hash.html'
    read_only = True

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        summary = []
        if not value or value.startswith(UNUSABLE_PASSWORD_PREFIX):
            summary.append({'label': gettext("No password set.")})
        else:
            try:
                hasher = identify_hasher(value)
            except ValueError:
                summary.append({'label': gettext("Invalid password format or unknown hashing algorithm.")})
            else:
                for key, value_ in hasher.safe_summary(value).items():
                    summary.append({'label': gettext(key), 'value': value_})
        context['summary'] = summary
        return context
```
### 9 - django/template/context.py:

Start line: 27, End line: 132

```python
class BaseContext:
    def __init__(self, dict_=None):
        self._reset_dicts(dict_)

    def _reset_dicts(self, value=None):
        builtins = {'True': True, 'False': False, 'None': None}
        self.dicts = [builtins]
        if value is not None:
            self.dicts.append(value)

    def __copy__(self):
        duplicate = copy(super())
        duplicate.dicts = self.dicts[:]
        return duplicate

    def __repr__(self):
        return repr(self.dicts)

    def __iter__(self):
        return reversed(self.dicts)

    def push(self, *args, **kwargs):
        dicts = []
        for d in args:
            if isinstance(d, BaseContext):
                dicts += d.dicts[1:]
            else:
                dicts.append(d)
        return ContextDict(self, *dicts, **kwargs)

    def pop(self):
        if len(self.dicts) == 1:
            raise ContextPopException
        return self.dicts.pop()

    def __setitem__(self, key, value):
        "Set a variable in the current context"
        self.dicts[-1][key] = value

    def set_upward(self, key, value):
        """
        Set a variable in one of the higher contexts if it exists there,
        otherwise in the current context.
        """
        context = self.dicts[-1]
        for d in reversed(self.dicts):
            if key in d:
                context = d
                break
        context[key] = value

    def __getitem__(self, key):
        "Get a variable's value, starting at the current context and going upward"
        for d in reversed(self.dicts):
            if key in d:
                return d[key]
        raise KeyError(key)

    def __delitem__(self, key):
        "Delete a variable from the current context"
        del self.dicts[-1][key]

    def __contains__(self, key):
        return any(key in d for d in self.dicts)

    def get(self, key, otherwise=None):
        for d in reversed(self.dicts):
            if key in d:
                return d[key]
        return otherwise

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
        return default

    def new(self, values=None):
        """
        Return a new context with the same properties, but with only the
        values given in 'values' stored.
        """
        new_context = copy(self)
        new_context._reset_dicts(values)
        return new_context

    def flatten(self):
        """
        Return self.dicts as one dictionary.
        """
        flat = {}
        for d in self.dicts:
            flat.update(d)
        return flat

    def __eq__(self, other):
        """
        Compare two contexts by comparing theirs 'dicts' attributes.
        """
        return (
            isinstance(other, BaseContext) and
            # because dictionaries can be put in different order
            # we have to flatten them like in templates
            self.flatten() == other.flatten()
        )
```
### 10 - django/contrib/admindocs/views.py:

Start line: 249, End line: 314

```python
class ModelDetailView(BaseAdminDocsView):

    def get_context_data(self, **kwargs):
        # ... other code
        for func_name, func in model.__dict__.items():
            if inspect.isfunction(func) or isinstance(func, property):
                try:
                    for exclude in MODEL_METHODS_EXCLUDE:
                        if func_name.startswith(exclude):
                            raise StopIteration
                except StopIteration:
                    continue
                verbose = func.__doc__
                verbose = verbose and (
                    utils.parse_rst(utils.trim_docstring(verbose), 'model', _('model:') + opts.model_name)
                )
                # Show properties and methods without arguments as fields.
                # Otherwise, show as a 'method with arguments'.
                if isinstance(func, property):
                    fields.append({
                        'name': func_name,
                        'data_type': get_return_data_type(func_name),
                        'verbose': verbose or ''
                    })
                elif method_has_no_args(func) and not func_accepts_kwargs(func) and not func_accepts_var_args(func):
                    fields.append({
                        'name': func_name,
                        'data_type': get_return_data_type(func_name),
                        'verbose': verbose or '',
                    })
                else:
                    arguments = get_func_full_args(func)
                    # Join arguments with ', ' and in case of default value,
                    # join it with '='. Use repr() so that strings will be
                    # correctly displayed.
                    print_arguments = ', '.join([
                        '='.join([arg_el[0], *map(repr, arg_el[1:])])
                        for arg_el in arguments
                    ])
                    methods.append({
                        'name': func_name,
                        'arguments': print_arguments,
                        'verbose': verbose or '',
                    })

        # Gather related objects
        for rel in opts.related_objects:
            verbose = _("related `%(app_label)s.%(object_name)s` objects") % {
                'app_label': rel.related_model._meta.app_label,
                'object_name': rel.related_model._meta.object_name,
            }
            accessor = rel.get_accessor_name()
            fields.append({
                'name': "%s.all" % accessor,
                'data_type': 'List',
                'verbose': utils.parse_rst(_("all %s") % verbose, 'model', _('model:') + opts.model_name),
            })
            fields.append({
                'name': "%s.count" % accessor,
                'data_type': 'Integer',
                'verbose': utils.parse_rst(_("number of %s") % verbose, 'model', _('model:') + opts.model_name),
            })
        return super().get_context_data(**{
            **kwargs,
            'name': '%s.%s' % (opts.app_label, opts.object_name),
            'summary': title,
            'description': body,
            'fields': fields,
            'methods': methods,
        })
```
### 78 - django/contrib/admin/templatetags/admin_modify.py:

Start line: 1, End line: 45

```python
import json

from django import template
from django.template.context import Context

from .base import InclusionAdminNode

register = template.Library()


def prepopulated_fields_js(context):
    """
    Create a list of prepopulated_fields that should render Javascript for
    the prepopulated fields for both the admin form and inlines.
    """
    prepopulated_fields = []
    if 'adminform' in context:
        prepopulated_fields.extend(context['adminform'].prepopulated_fields)
    if 'inline_admin_formsets' in context:
        for inline_admin_formset in context['inline_admin_formsets']:
            for inline_admin_form in inline_admin_formset:
                if inline_admin_form.original is None:
                    prepopulated_fields.extend(inline_admin_form.prepopulated_fields)

    prepopulated_fields_json = []
    for field in prepopulated_fields:
        prepopulated_fields_json.append({
            "id": "#%s" % field["field"].auto_id,
            "name": field["field"].name,
            "dependency_ids": ["#%s" % dependency.auto_id for dependency in field["dependencies"]],
            "dependency_list": [dependency.name for dependency in field["dependencies"]],
            "maxLength": field["field"].field.max_length or 50,
            "allowUnicode": getattr(field["field"].field, "allow_unicode", False)
        })

    context.update({
        'prepopulated_fields': prepopulated_fields,
        'prepopulated_fields_json': json.dumps(prepopulated_fields_json),
    })
    return context


@register.tag(name='prepopulated_fields_js')
def prepopulated_fields_js_tag(parser, token):
    return InclusionAdminNode(parser, token, func=prepopulated_fields_js, template_name="prepopulated_fields_js.html")
```
