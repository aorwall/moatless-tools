# django__django-15061

| **django/django** | `2c01ebb4be5d53cbf6450f356c10e436025d6d07` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3059 |
| **Any found context length** | 3059 |
| **Avg pos** | 11.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -849,9 +849,7 @@ def get_context(self, name, value, attrs):
         return context
 
     def id_for_label(self, id_):
-        if id_:
-            id_ += '_0'
-        return id_
+        return ''
 
     def value_from_datadict(self, data, files, name):
         return [

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/widgets.py | 852 | 854 | 11 | 2 | 3059


## Problem Statement

```
Remove "for = ..." from MultiWidget's <label>.
Description
	
The instance from Raw MultiWidget class generate id_for_label like f'{id_}0'
It has not sense.
For example ChoiceWidget has self.add_id_index and I can decide it myself, how I will see label_id - with or without index.
I think, it is better to remove completely id_for_label method from MultiWidget Class.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/contrib/admin/widgets.py | 195 | 221| 216 | 216 | 3885 | 
| 2 | 1 django/contrib/admin/widgets.py | 120 | 159| 325 | 541 | 3885 | 
| 3 | **2 django/forms/widgets.py** | 744 | 773| 219 | 760 | 11995 | 
| 4 | **2 django/forms/widgets.py** | 640 | 669| 238 | 998 | 11995 | 
| 5 | **2 django/forms/widgets.py** | 820 | 849| 269 | 1267 | 11995 | 
| 6 | **2 django/forms/widgets.py** | 621 | 638| 179 | 1446 | 11995 | 
| 7 | 2 django/contrib/admin/widgets.py | 161 | 192| 243 | 1689 | 11995 | 
| 8 | **2 django/forms/widgets.py** | 199 | 281| 617 | 2306 | 11995 | 
| 9 | 3 django/contrib/auth/forms.py | 33 | 54| 175 | 2481 | 15121 | 
| 10 | 4 django/forms/boundfield.py | 257 | 296| 263 | 2744 | 17390 | 
| **-> 11 <-** | **4 django/forms/widgets.py** | 851 | 894| 315 | 3059 | 17390 | 
| 12 | **4 django/forms/widgets.py** | 672 | 703| 261 | 3320 | 17390 | 
| 13 | **4 django/forms/widgets.py** | 342 | 376| 278 | 3598 | 17390 | 
| 14 | **4 django/forms/widgets.py** | 793 | 818| 214 | 3812 | 17390 | 
| 15 | 4 django/contrib/admin/widgets.py | 273 | 308| 382 | 4194 | 17390 | 
| 16 | **4 django/forms/widgets.py** | 1045 | 1064| 144 | 4338 | 17390 | 
| 17 | 4 django/forms/boundfield.py | 35 | 50| 149 | 4487 | 17390 | 
| 18 | 5 django/contrib/postgres/forms/array.py | 105 | 131| 219 | 4706 | 18984 | 
| 19 | **5 django/forms/widgets.py** | 402 | 432| 193 | 4899 | 18984 | 
| 20 | **5 django/forms/widgets.py** | 936 | 982| 402 | 5301 | 18984 | 
| 21 | **5 django/forms/widgets.py** | 551 | 585| 277 | 5578 | 18984 | 
| 22 | **5 django/forms/widgets.py** | 776 | 790| 139 | 5717 | 18984 | 
| 23 | 5 django/contrib/admin/widgets.py | 224 | 271| 423 | 6140 | 18984 | 
| 24 | **5 django/forms/widgets.py** | 448 | 466| 195 | 6335 | 18984 | 
| 25 | **5 django/forms/widgets.py** | 706 | 741| 237 | 6572 | 18984 | 
| 26 | **5 django/forms/widgets.py** | 1 | 41| 290 | 6862 | 18984 | 
| 27 | **5 django/forms/widgets.py** | 984 | 1018| 391 | 7253 | 18984 | 
| 28 | **5 django/forms/widgets.py** | 587 | 619| 233 | 7486 | 18984 | 
| 29 | 5 django/forms/boundfield.py | 233 | 254| 220 | 7706 | 18984 | 
| 30 | **5 django/forms/widgets.py** | 1066 | 1089| 250 | 7956 | 18984 | 
| 31 | 5 django/contrib/postgres/forms/array.py | 133 | 165| 265 | 8221 | 18984 | 
| 32 | **5 django/forms/widgets.py** | 434 | 446| 122 | 8343 | 18984 | 
| 33 | 5 django/forms/boundfield.py | 1 | 33| 233 | 8576 | 18984 | 
| 34 | 5 django/forms/boundfield.py | 148 | 185| 386 | 8962 | 18984 | 
| 35 | 5 django/contrib/admin/widgets.py | 1 | 46| 330 | 9292 | 18984 | 
| 36 | 5 django/forms/boundfield.py | 79 | 97| 174 | 9466 | 18984 | 
| 37 | **5 django/forms/widgets.py** | 379 | 399| 138 | 9604 | 18984 | 
| 38 | **5 django/forms/widgets.py** | 897 | 921| 172 | 9776 | 18984 | 
| 39 | **5 django/forms/widgets.py** | 512 | 548| 331 | 10107 | 18984 | 
| 40 | **5 django/forms/widgets.py** | 924 | 933| 107 | 10214 | 18984 | 
| 41 | 6 django/contrib/gis/forms/widgets.py | 44 | 74| 291 | 10505 | 19859 | 
| 42 | 6 django/contrib/auth/forms.py | 57 | 75| 124 | 10629 | 19859 | 
| 43 | 6 django/contrib/admin/widgets.py | 347 | 373| 328 | 10957 | 19859 | 
| 44 | 7 django/db/models/lookups.py | 640 | 652| 124 | 11081 | 25198 | 
| 45 | 8 django/forms/fields.py | 47 | 128| 773 | 11854 | 34619 | 
| 46 | 9 django/forms/models.py | 1347 | 1362| 131 | 11985 | 46537 | 
| 47 | **9 django/forms/widgets.py** | 284 | 300| 127 | 12112 | 46537 | 
| 48 | 9 django/contrib/admin/widgets.py | 326 | 344| 168 | 12280 | 46537 | 
| 49 | 10 django/db/backends/sqlite3/schema.py | 332 | 348| 173 | 12453 | 50740 | 
| 50 | 11 django/contrib/admin/utils.py | 370 | 409| 350 | 12803 | 54893 | 
| 51 | 12 django/db/migrations/autodetector.py | 918 | 937| 184 | 12987 | 67005 | 
| 52 | 13 django/db/backends/mysql/schema.py | 124 | 140| 205 | 13192 | 68579 | 
| 53 | 13 django/forms/models.py | 1401 | 1428| 209 | 13401 | 68579 | 
| 54 | 13 django/contrib/admin/widgets.py | 73 | 89| 145 | 13546 | 68579 | 
| 55 | 14 django/template/defaulttags.py | 160 | 221| 532 | 14078 | 79231 | 
| 56 | 15 django/db/backends/base/schema.py | 548 | 576| 289 | 14367 | 92113 | 
| 57 | 15 django/contrib/admin/widgets.py | 49 | 70| 168 | 14535 | 92113 | 
| 58 | 16 django/db/models/options.py | 205 | 246| 321 | 14856 | 99461 | 
| 59 | 17 django/contrib/admin/helpers.py | 221 | 254| 318 | 15174 | 102934 | 
| 60 | **17 django/forms/widgets.py** | 1020 | 1043| 233 | 15407 | 102934 | 
| 61 | **17 django/forms/widgets.py** | 303 | 339| 204 | 15611 | 102934 | 
| 62 | 18 django/db/migrations/operations/fields.py | 142 | 181| 344 | 15955 | 105427 | 
| 63 | 18 django/contrib/admin/widgets.py | 376 | 394| 144 | 16099 | 105427 | 
| 64 | 18 django/contrib/admin/widgets.py | 451 | 478| 191 | 16290 | 105427 | 
| 65 | **18 django/forms/widgets.py** | 469 | 509| 275 | 16565 | 105427 | 
| 66 | 18 django/forms/boundfield.py | 187 | 231| 352 | 16917 | 105427 | 
| 67 | 18 django/contrib/gis/forms/widgets.py | 77 | 119| 304 | 17221 | 105427 | 
| 68 | 19 django/contrib/gis/admin/options.py | 1 | 32| 230 | 17451 | 106894 | 
| 69 | **19 django/forms/widgets.py** | 160 | 196| 228 | 17679 | 106894 | 
| 70 | 20 django/contrib/gis/admin/widgets.py | 81 | 118| 344 | 18023 | 107858 | 
| 71 | 20 django/forms/models.py | 1161 | 1206| 323 | 18346 | 107858 | 
| 72 | 21 django/db/migrations/operations/models.py | 773 | 813| 344 | 18690 | 114346 | 
| 73 | 22 django/db/models/base.py | 1440 | 1495| 491 | 19181 | 131674 | 
| 74 | 23 django/utils/html.py | 136 | 162| 149 | 19330 | 134882 | 
| 75 | 23 django/db/models/base.py | 949 | 966| 181 | 19511 | 134882 | 
| 76 | 24 django/contrib/admin/options.py | 1642 | 1667| 291 | 19802 | 153566 | 
| 77 | 24 django/contrib/gis/forms/widgets.py | 1 | 42| 285 | 20087 | 153566 | 
| 78 | 24 django/contrib/admin/utils.py | 412 | 441| 200 | 20287 | 153566 | 
| 79 | 24 django/contrib/admin/options.py | 1472 | 1491| 133 | 20420 | 153566 | 
| 80 | 24 django/contrib/admin/options.py | 2109 | 2161| 451 | 20871 | 153566 | 
| 81 | 24 django/contrib/admin/helpers.py | 131 | 157| 220 | 21091 | 153566 | 
| 82 | 24 django/contrib/admin/widgets.py | 311 | 323| 114 | 21205 | 153566 | 
| 83 | 24 django/contrib/admin/options.py | 1551 | 1641| 770 | 21975 | 153566 | 
| 84 | 24 django/forms/fields.py | 864 | 903| 298 | 22273 | 153566 | 
| 85 | 24 django/contrib/gis/admin/options.py | 87 | 98| 139 | 22412 | 153566 | 
| 86 | 24 django/db/migrations/autodetector.py | 808 | 872| 676 | 23088 | 153566 | 
| 87 | 24 django/template/defaulttags.py | 1201 | 1229| 164 | 23252 | 153566 | 
| 88 | 25 django/forms/forms.py | 211 | 293| 685 | 23937 | 157589 | 
| 89 | 26 django/contrib/admin/templatetags/admin_modify.py | 48 | 86| 391 | 24328 | 158560 | 
| 90 | 26 django/forms/fields.py | 775 | 834| 416 | 24744 | 158560 | 
| 91 | 26 django/contrib/postgres/forms/array.py | 168 | 195| 226 | 24970 | 158560 | 
| 92 | 26 django/db/migrations/operations/models.py | 850 | 886| 331 | 25301 | 158560 | 
| 93 | 27 django/db/models/enums.py | 59 | 92| 191 | 25492 | 159172 | 
| 94 | 27 django/forms/models.py | 389 | 417| 240 | 25732 | 159172 | 
| 95 | 27 django/contrib/admin/options.py | 243 | 283| 376 | 26108 | 159172 | 
| 96 | 28 django/db/models/fields/related.py | 1249 | 1279| 172 | 26280 | 173330 | 
| 97 | 28 django/contrib/admin/options.py | 131 | 186| 604 | 26884 | 173330 | 
| 98 | 28 django/forms/forms.py | 408 | 422| 135 | 27019 | 173330 | 
| 99 | 29 django/db/models/fields/__init__.py | 367 | 393| 199 | 27218 | 191498 | 
| 100 | 30 django/contrib/admin/filters.py | 399 | 421| 211 | 27429 | 195628 | 
| 101 | 30 django/db/models/enums.py | 34 | 56| 183 | 27612 | 195628 | 
| 102 | 30 django/forms/models.py | 1209 | 1274| 543 | 28155 | 195628 | 
| 103 | 30 django/forms/models.py | 1328 | 1345| 152 | 28307 | 195628 | 
| 104 | **30 django/forms/widgets.py** | 114 | 157| 367 | 28674 | 195628 | 
| 105 | 30 django/contrib/gis/admin/options.py | 115 | 171| 563 | 29237 | 195628 | 
| 106 | 30 django/contrib/admin/filters.py | 280 | 304| 217 | 29454 | 195628 | 
| 107 | 30 django/db/migrations/operations/models.py | 393 | 414| 170 | 29624 | 195628 | 
| 108 | 30 django/forms/fields.py | 130 | 173| 290 | 29914 | 195628 | 
| 109 | 30 django/contrib/admin/widgets.py | 422 | 449| 296 | 30210 | 195628 | 
| 110 | 30 django/contrib/admin/options.py | 1 | 96| 756 | 30966 | 195628 | 
| 111 | 30 django/db/models/fields/__init__.py | 2372 | 2422| 339 | 31305 | 195628 | 


### Hint

```
I agree that we should remove for from MultiWidget's <label> but not because "It has not sense" but to improve accessibility when using a screen reader, see also #32338. It should be enough to return an empty string: def id_for_label(self, id_): return ''
​PR
```

## Patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -849,9 +849,7 @@ def get_context(self, name, value, attrs):
         return context
 
     def id_for_label(self, id_):
-        if id_:
-            id_ += '_0'
-        return id_
+        return ''
 
     def value_from_datadict(self, data, files, name):
         return [

```

## Test Patch

```diff
diff --git a/tests/forms_tests/field_tests/test_multivaluefield.py b/tests/forms_tests/field_tests/test_multivaluefield.py
--- a/tests/forms_tests/field_tests/test_multivaluefield.py
+++ b/tests/forms_tests/field_tests/test_multivaluefield.py
@@ -141,7 +141,7 @@ def test_form_as_table(self):
         self.assertHTMLEqual(
             form.as_table(),
             """
-            <tr><th><label for="id_field1_0">Field1:</label></th>
+            <tr><th><label>Field1:</label></th>
             <td><input type="text" name="field1_0" id="id_field1_0" required>
             <select multiple name="field1_1" id="id_field1_1" required>
             <option value="J">John</option>
@@ -164,7 +164,7 @@ def test_form_as_table_data(self):
         self.assertHTMLEqual(
             form.as_table(),
             """
-            <tr><th><label for="id_field1_0">Field1:</label></th>
+            <tr><th><label>Field1:</label></th>
             <td><input type="text" name="field1_0" value="some text" id="id_field1_0" required>
             <select multiple name="field1_1" id="id_field1_1" required>
             <option value="J" selected>John</option>
diff --git a/tests/forms_tests/field_tests/test_splitdatetimefield.py b/tests/forms_tests/field_tests/test_splitdatetimefield.py
--- a/tests/forms_tests/field_tests/test_splitdatetimefield.py
+++ b/tests/forms_tests/field_tests/test_splitdatetimefield.py
@@ -1,7 +1,7 @@
 import datetime
 
 from django.core.exceptions import ValidationError
-from django.forms import SplitDateTimeField
+from django.forms import Form, SplitDateTimeField
 from django.forms.widgets import SplitDateTimeWidget
 from django.test import SimpleTestCase
 
@@ -60,3 +60,16 @@ def test_splitdatetimefield_changed(self):
         self.assertTrue(f.has_changed(datetime.datetime(2008, 5, 6, 12, 40, 00), ['2008-05-06', '12:40:00']))
         self.assertFalse(f.has_changed(datetime.datetime(2008, 5, 6, 12, 40, 00), ['06/05/2008', '12:40']))
         self.assertTrue(f.has_changed(datetime.datetime(2008, 5, 6, 12, 40, 00), ['06/05/2008', '12:41']))
+
+    def test_form_as_table(self):
+        class TestForm(Form):
+            datetime = SplitDateTimeField()
+
+        f = TestForm()
+        self.assertHTMLEqual(
+            f.as_table(),
+            '<tr><th><label>Datetime:</label></th><td>'
+            '<input type="text" name="datetime_0" required id="id_datetime_0">'
+            '<input type="text" name="datetime_1" required id="id_datetime_1">'
+            '</td></tr>',
+        )
diff --git a/tests/postgres_tests/test_ranges.py b/tests/postgres_tests/test_ranges.py
--- a/tests/postgres_tests/test_ranges.py
+++ b/tests/postgres_tests/test_ranges.py
@@ -665,7 +665,7 @@ class SplitForm(forms.Form):
         self.assertHTMLEqual(str(form), '''
             <tr>
                 <th>
-                <label for="id_field_0">Field:</label>
+                <label>Field:</label>
                 </th>
                 <td>
                     <input id="id_field_0_0" name="field_0_0" type="text">
@@ -700,7 +700,7 @@ class DateTimeRangeForm(forms.Form):
             form.as_table(),
             """
             <tr><th>
-            <label for="id_datetime_field_0">Datetime field:</label>
+            <label>Datetime field:</label>
             </th><td>
             <input type="text" name="datetime_field_0" id="id_datetime_field_0">
             <input type="text" name="datetime_field_1" id="id_datetime_field_1">
@@ -717,7 +717,7 @@ class DateTimeRangeForm(forms.Form):
             form.as_table(),
             """
             <tr><th>
-            <label for="id_datetime_field_0">Datetime field:</label>
+            <label>Datetime field:</label>
             </th><td>
             <input type="text" name="datetime_field_0"
             value="2010-01-01 11:13:00" id="id_datetime_field_0">
@@ -754,7 +754,7 @@ class RangeForm(forms.Form):
 
         self.assertHTMLEqual(str(RangeForm()), '''
         <tr>
-            <th><label for="id_ints_0">Ints:</label></th>
+            <th><label>Ints:</label></th>
             <td>
                 <input id="id_ints_0" name="ints_0" type="number">
                 <input id="id_ints_1" name="ints_1" type="number">

```


## Code snippets

### 1 - django/contrib/admin/widgets.py:

Start line: 195, End line: 221

```python
class ManyToManyRawIdWidget(ForeignKeyRawIdWidget):
    """
    A Widget for displaying ManyToMany ids in the "raw_id" interface rather than
    in a <select multiple> box.
    """
    template_name = 'admin/widgets/many_to_many_raw_id.html'

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.rel.model in self.admin_site._registry:
            # The related object is registered with the same AdminSite
            context['widget']['attrs']['class'] = 'vManyToManyRawIdAdminField'
        return context

    def url_parameters(self):
        return self.base_url_parameters()

    def label_and_url_for_value(self, value):
        return '', ''

    def value_from_datadict(self, data, files, name):
        value = data.get(name)
        if value:
            return value.split(',')

    def format_value(self, value):
        return ','.join(str(v) for v in value) if value else ''
```
### 2 - django/contrib/admin/widgets.py:

Start line: 120, End line: 159

```python
class ForeignKeyRawIdWidget(forms.TextInput):
    """
    A Widget for displaying ForeignKeys in the "raw_id" interface rather than
    in a <select> box.
    """
    template_name = 'admin/widgets/foreign_key_raw_id.html'

    def __init__(self, rel, admin_site, attrs=None, using=None):
        self.rel = rel
        self.admin_site = admin_site
        self.db = using
        super().__init__(attrs)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        rel_to = self.rel.model
        if rel_to in self.admin_site._registry:
            # The related object is registered with the same AdminSite
            related_url = reverse(
                'admin:%s_%s_changelist' % (
                    rel_to._meta.app_label,
                    rel_to._meta.model_name,
                ),
                current_app=self.admin_site.name,
            )

            params = self.url_parameters()
            if params:
                related_url += '?' + urlencode(params)
            context['related_url'] = related_url
            context['link_title'] = _('Lookup')
            # The JavaScript code looks for this class.
            context['widget']['attrs'].setdefault('class', 'vForeignKeyRawIdAdminField')
        else:
            context['related_url'] = None
        if context['widget']['value']:
            context['link_label'], context['link_url'] = self.label_and_url_for_value(value)
        else:
            context['link_label'] = None
        return context
```
### 3 - django/forms/widgets.py:

Start line: 744, End line: 773

```python
class SelectMultiple(Select):
    allow_multiple_selected = True

    def value_from_datadict(self, data, files, name):
        try:
            getter = data.getlist
        except AttributeError:
            getter = data.get
        return getter(name)

    def value_omitted_from_data(self, data, files, name):
        # An unselected <select multiple> doesn't appear in POST data, so it's
        # never known if the value is actually omitted.
        return False


class RadioSelect(ChoiceWidget):
    input_type = 'radio'
    template_name = 'django/forms/widgets/radio.html'
    option_template_name = 'django/forms/widgets/radio_option.html'

    def id_for_label(self, id_, index=None):
        """
        Don't include for="field_0" in <label> to improve accessibility when
        using a screen reader, in addition clicking such a label would toggle
        the first input.
        """
        if index is None:
            return ''
        return super().id_for_label(id_, index)
```
### 4 - django/forms/widgets.py:

Start line: 640, End line: 669

```python
class ChoiceWidget(Widget):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context['widget']['optgroups'] = self.optgroups(name, context['widget']['value'], attrs)
        return context

    def id_for_label(self, id_, index='0'):
        """
        Use an incremented id for each option where the main widget
        references the zero index.
        """
        if id_ and self.add_id_index:
            id_ = '%s_%s' % (id_, index)
        return id_

    def value_from_datadict(self, data, files, name):
        getter = data.get
        if self.allow_multiple_selected:
            try:
                getter = data.getlist
            except AttributeError:
                pass
        return getter(name)

    def format_value(self, value):
        """Return selected values as a list."""
        if value is None and self.allow_multiple_selected:
            return []
        if not isinstance(value, (tuple, list)):
            value = [value]
        return [str(v) if v is not None else '' for v in value]
```
### 5 - django/forms/widgets.py:

Start line: 820, End line: 849

```python
class MultiWidget(Widget):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.is_localized:
            for widget in self.widgets:
                widget.is_localized = self.is_localized
        # value is a list of values, each corresponding to a widget
        # in self.widgets.
        if not isinstance(value, list):
            value = self.decompress(value)

        final_attrs = context['widget']['attrs']
        input_type = final_attrs.pop('type', None)
        id_ = final_attrs.get('id')
        subwidgets = []
        for i, (widget_name, widget) in enumerate(zip(self.widgets_names, self.widgets)):
            if input_type is not None:
                widget.input_type = input_type
            widget_name = name + widget_name
            try:
                widget_value = value[i]
            except IndexError:
                widget_value = None
            if id_:
                widget_attrs = final_attrs.copy()
                widget_attrs['id'] = '%s_%s' % (id_, i)
            else:
                widget_attrs = final_attrs
            subwidgets.append(widget.get_context(widget_name, widget_value, widget_attrs)['widget'])
        context['widget']['subwidgets'] = subwidgets
        return context
```
### 6 - django/forms/widgets.py:

Start line: 621, End line: 638

```python
class ChoiceWidget(Widget):

    def create_option(self, name, value, label, selected, index, subindex=None, attrs=None):
        index = str(index) if subindex is None else "%s_%s" % (index, subindex)
        option_attrs = self.build_attrs(self.attrs, attrs) if self.option_inherits_attrs else {}
        if selected:
            option_attrs.update(self.checked_attribute)
        if 'id' in option_attrs:
            option_attrs['id'] = self.id_for_label(option_attrs['id'], index)
        return {
            'name': name,
            'value': value,
            'label': label,
            'selected': selected,
            'index': index,
            'attrs': option_attrs,
            'type': self.input_type,
            'template_name': self.option_template_name,
            'wrap_label': True,
        }
```
### 7 - django/contrib/admin/widgets.py:

Start line: 161, End line: 192

```python
class ForeignKeyRawIdWidget(forms.TextInput):

    def base_url_parameters(self):
        limit_choices_to = self.rel.limit_choices_to
        if callable(limit_choices_to):
            limit_choices_to = limit_choices_to()
        return url_params_from_lookup_dict(limit_choices_to)

    def url_parameters(self):
        from django.contrib.admin.views.main import TO_FIELD_VAR
        params = self.base_url_parameters()
        params.update({TO_FIELD_VAR: self.rel.get_related_field().name})
        return params

    def label_and_url_for_value(self, value):
        key = self.rel.get_related_field().name
        try:
            obj = self.rel.model._default_manager.using(self.db).get(**{key: value})
        except (ValueError, self.rel.model.DoesNotExist, ValidationError):
            return '', ''

        try:
            url = reverse(
                '%s:%s_%s_change' % (
                    self.admin_site.name,
                    obj._meta.app_label,
                    obj._meta.object_name.lower(),
                ),
                args=(obj.pk,)
            )
        except NoReverseMatch:
            url = ''  # Admin not registered for target model.

        return Truncator(obj).words(14), url
```
### 8 - django/forms/widgets.py:

Start line: 199, End line: 281

```python
class Widget(metaclass=MediaDefiningClass):
    needs_multipart_form = False  # Determines does this widget need multipart form
    is_localized = False
    is_required = False
    supports_microseconds = True

    def __init__(self, attrs=None):
        self.attrs = {} if attrs is None else attrs.copy()

    def __deepcopy__(self, memo):
        obj = copy.copy(self)
        obj.attrs = self.attrs.copy()
        memo[id(self)] = obj
        return obj

    @property
    def is_hidden(self):
        return self.input_type == 'hidden' if hasattr(self, 'input_type') else False

    def subwidgets(self, name, value, attrs=None):
        context = self.get_context(name, value, attrs)
        yield context['widget']

    def format_value(self, value):
        """
        Return a value as it should appear when rendered in a template.
        """
        if value == '' or value is None:
            return None
        if self.is_localized:
            return formats.localize_input(value)
        return str(value)

    def get_context(self, name, value, attrs):
        return {
            'widget': {
                'name': name,
                'is_hidden': self.is_hidden,
                'required': self.is_required,
                'value': self.format_value(value),
                'attrs': self.build_attrs(self.attrs, attrs),
                'template_name': self.template_name,
            },
        }

    def render(self, name, value, attrs=None, renderer=None):
        """Render the widget as an HTML string."""
        context = self.get_context(name, value, attrs)
        return self._render(self.template_name, context, renderer)

    def _render(self, template_name, context, renderer=None):
        if renderer is None:
            renderer = get_default_renderer()
        return mark_safe(renderer.render(template_name, context))

    def build_attrs(self, base_attrs, extra_attrs=None):
        """Build an attribute dictionary."""
        return {**base_attrs, **(extra_attrs or {})}

    def value_from_datadict(self, data, files, name):
        """
        Given a dictionary of data and this widget's name, return the value
        of this widget or None if it's not provided.
        """
        return data.get(name)

    def value_omitted_from_data(self, data, files, name):
        return name not in data

    def id_for_label(self, id_):
        """
        Return the HTML ID attribute of this Widget for use by a <label>,
        given the ID of the field. Return None if no ID is available.

        This hook is necessary because some widgets have multiple HTML
        elements and, thus, multiple IDs. In that case, this method should
        return an ID value that corresponds to the first ID in the widget's
        tags.
        """
        return id_

    def use_required_attribute(self, initial):
        return not self.is_hidden
```
### 9 - django/contrib/auth/forms.py:

Start line: 33, End line: 54

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

    def id_for_label(self, id_):
        return None
```
### 10 - django/forms/boundfield.py:

Start line: 257, End line: 296

```python
@html_safe
class BoundWidget:
    """
    A container class used for iterating over widgets. This is useful for
    widgets that have choices. For example, the following can be used in a
    template:

    {% for radio in myform.beatles %}
      <label for="{{ radio.id_for_label }}">
        {{ radio.choice_label }}
        <span class="radio">{{ radio.tag }}</span>
      </label>
    {% endfor %}
    """
    def __init__(self, parent_widget, data, renderer):
        self.parent_widget = parent_widget
        self.data = data
        self.renderer = renderer

    def __str__(self):
        return self.tag(wrap_label=True)

    def tag(self, wrap_label=False):
        context = {'widget': {**self.data, 'wrap_label': wrap_label}}
        return self.parent_widget._render(self.template_name, context, self.renderer)

    @property
    def template_name(self):
        if 'template_name' in self.data:
            return self.data['template_name']
        return self.parent_widget.template_name

    @property
    def id_for_label(self):
        return self.data['attrs'].get('id')

    @property
    def choice_label(self):
        return self.data['label']
```
### 11 - django/forms/widgets.py:

Start line: 851, End line: 894

```python
class MultiWidget(Widget):

    def id_for_label(self, id_):
        if id_:
            id_ += '_0'
        return id_

    def value_from_datadict(self, data, files, name):
        return [
            widget.value_from_datadict(data, files, name + widget_name)
            for widget_name, widget in zip(self.widgets_names, self.widgets)
        ]

    def value_omitted_from_data(self, data, files, name):
        return all(
            widget.value_omitted_from_data(data, files, name + widget_name)
            for widget_name, widget in zip(self.widgets_names, self.widgets)
        )

    def decompress(self, value):
        """
        Return a list of decompressed values for the given compressed value.
        The given value can be assumed to be valid, but not necessarily
        non-empty.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    def _get_media(self):
        """
        Media for a multiwidget is the combination of all media of the
        subwidgets.
        """
        media = Media()
        for w in self.widgets:
            media = media + w.media
        return media
    media = property(_get_media)

    def __deepcopy__(self, memo):
        obj = super().__deepcopy__(memo)
        obj.widgets = copy.deepcopy(self.widgets)
        return obj

    @property
    def needs_multipart_form(self):
        return any(w.needs_multipart_form for w in self.widgets)
```
### 12 - django/forms/widgets.py:

Start line: 672, End line: 703

```python
class Select(ChoiceWidget):
    input_type = 'select'
    template_name = 'django/forms/widgets/select.html'
    option_template_name = 'django/forms/widgets/select_option.html'
    add_id_index = False
    checked_attribute = {'selected': True}
    option_inherits_attrs = False

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        if self.allow_multiple_selected:
            context['widget']['attrs']['multiple'] = True
        return context

    @staticmethod
    def _choice_has_empty_value(choice):
        """Return True if the choice's value is empty string or None."""
        value, _ = choice
        return value is None or value == ''

    def use_required_attribute(self, initial):
        """
        Don't render 'required' if the first <option> has a value, as that's
        invalid HTML.
        """
        use_required_attribute = super().use_required_attribute(initial)
        # 'required' is always okay for <select multiple>.
        if self.allow_multiple_selected:
            return use_required_attribute

        first_choice = next(iter(self.choices), None)
        return use_required_attribute and first_choice is not None and self._choice_has_empty_value(first_choice)
```
### 13 - django/forms/widgets.py:

Start line: 342, End line: 376

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
### 14 - django/forms/widgets.py:

Start line: 793, End line: 818

```python
class MultiWidget(Widget):
    """
    A widget that is composed of multiple widgets.

    In addition to the values added by Widget.get_context(), this widget
    adds a list of subwidgets to the context as widget['subwidgets'].
    These can be looped over and rendered like normal widgets.

    You'll probably want to use this class with MultiValueField.
    """
    template_name = 'django/forms/widgets/multiwidget.html'

    def __init__(self, widgets, attrs=None):
        if isinstance(widgets, dict):
            self.widgets_names = [
                ('_%s' % name) if name else '' for name in widgets
            ]
            widgets = widgets.values()
        else:
            self.widgets_names = ['_%s' % i for i in range(len(widgets))]
        self.widgets = [w() if isinstance(w, type) else w for w in widgets]
        super().__init__(attrs)

    @property
    def is_hidden(self):
        return all(w.is_hidden for w in self.widgets)
```
### 16 - django/forms/widgets.py:

Start line: 1045, End line: 1064

```python
class SelectDateWidget(Widget):

    @staticmethod
    def _parse_date_fmt():
        fmt = get_format('DATE_FORMAT')
        escaped = False
        for char in fmt:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char in 'Yy':
                yield 'year'
            elif char in 'bEFMmNn':
                yield 'month'
            elif char in 'dj':
                yield 'day'

    def id_for_label(self, id_):
        for first_select in self._parse_date_fmt():
            return '%s_%s' % (id_, first_select)
        return '%s_month' % id_
```
### 19 - django/forms/widgets.py:

Start line: 402, End line: 432

```python
class ClearableFileInput(FileInput):
    clear_checkbox_label = _('Clear')
    initial_text = _('Currently')
    input_text = _('Change')
    template_name = 'django/forms/widgets/clearable_file_input.html'

    def clear_checkbox_name(self, name):
        """
        Given the name of the file input, return the name of the clear checkbox
        input.
        """
        return name + '-clear'

    def clear_checkbox_id(self, name):
        """
        Given the name of the clear checkbox input, return the HTML id for it.
        """
        return name + '_id'

    def is_initial(self, value):
        """
        Return whether value is considered to be initial value.
        """
        return bool(value and getattr(value, 'url', False))

    def format_value(self, value):
        """
        Return the file object if it has a defined url attribute.
        """
        if self.is_initial(value):
            return value
```
### 20 - django/forms/widgets.py:

Start line: 936, End line: 982

```python
class SelectDateWidget(Widget):
    """
    A widget that splits date input into three <select> boxes.

    This also serves as an example of a Widget that has more than one HTML
    element and hence implements value_from_datadict.
    """
    none_value = ('', '---')
    month_field = '%s_month'
    day_field = '%s_day'
    year_field = '%s_year'
    template_name = 'django/forms/widgets/select_date.html'
    input_type = 'select'
    select_widget = Select
    date_re = _lazy_re_compile(r'(\d{4}|0)-(\d\d?)-(\d\d?)$')

    def __init__(self, attrs=None, years=None, months=None, empty_label=None):
        self.attrs = attrs or {}

        # Optional list or tuple of years to use in the "year" select box.
        if years:
            self.years = years
        else:
            this_year = datetime.date.today().year
            self.years = range(this_year, this_year + 10)

        # Optional dict of months to use in the "month" select box.
        if months:
            self.months = months
        else:
            self.months = MONTHS

        # Optional string, list, or tuple to use as empty_label.
        if isinstance(empty_label, (list, tuple)):
            if not len(empty_label) == 3:
                raise ValueError('empty_label list/tuple must have 3 elements.')

            self.year_none_value = ('', empty_label[0])
            self.month_none_value = ('', empty_label[1])
            self.day_none_value = ('', empty_label[2])
        else:
            if empty_label is not None:
                self.none_value = ('', empty_label)

            self.year_none_value = self.none_value
            self.month_none_value = self.none_value
            self.day_none_value = self.none_value
```
### 21 - django/forms/widgets.py:

Start line: 551, End line: 585

```python
class ChoiceWidget(Widget):
    allow_multiple_selected = False
    input_type = None
    template_name = None
    option_template_name = None
    add_id_index = True
    checked_attribute = {'checked': True}
    option_inherits_attrs = True

    def __init__(self, attrs=None, choices=()):
        super().__init__(attrs)
        # choices can be any iterable, but we may need to render this widget
        # multiple times. Thus, collapse it into a list so it can be consumed
        # more than once.
        self.choices = list(choices)

    def __deepcopy__(self, memo):
        obj = copy.copy(self)
        obj.attrs = self.attrs.copy()
        obj.choices = copy.copy(self.choices)
        memo[id(self)] = obj
        return obj

    def subwidgets(self, name, value, attrs=None):
        """
        Yield all "subwidgets" of this widget. Used to enable iterating
        options from a BoundField for choice widgets.
        """
        value = self.format_value(value)
        yield from self.options(name, value, attrs)

    def options(self, name, value, attrs=None):
        """Yield a flat list of options for this widgets."""
        for group in self.optgroups(name, value, attrs):
            yield from group[1]
```
### 22 - django/forms/widgets.py:

Start line: 776, End line: 790

```python
class CheckboxSelectMultiple(RadioSelect):
    allow_multiple_selected = True
    input_type = 'checkbox'
    template_name = 'django/forms/widgets/checkbox_select.html'
    option_template_name = 'django/forms/widgets/checkbox_option.html'

    def use_required_attribute(self, initial):
        # Don't use the 'required' attribute because browser validation would
        # require all checkboxes to be checked instead of at least one.
        return False

    def value_omitted_from_data(self, data, files, name):
        # HTML checkboxes don't appear in POST data if not checked, so it's
        # never known if the value is actually omitted.
        return False
```
### 24 - django/forms/widgets.py:

Start line: 448, End line: 466

```python
class ClearableFileInput(FileInput):

    def value_from_datadict(self, data, files, name):
        upload = super().value_from_datadict(data, files, name)
        if not self.is_required and CheckboxInput().value_from_datadict(
                data, files, self.clear_checkbox_name(name)):

            if upload:
                # If the user contradicts themselves (uploads a new file AND
                # checks the "clear" checkbox), we return a unique marker
                # object that FileField will turn into a ValidationError.
                return FILE_INPUT_CONTRADICTION
            # False signals to clear any existing value, as opposed to just None
            return False
        return upload

    def value_omitted_from_data(self, data, files, name):
        return (
            super().value_omitted_from_data(data, files, name) and
            self.clear_checkbox_name(name) not in data
        )
```
### 25 - django/forms/widgets.py:

Start line: 706, End line: 741

```python
class NullBooleanSelect(Select):
    """
    A Select Widget intended to be used with NullBooleanField.
    """
    def __init__(self, attrs=None):
        choices = (
            ('unknown', _('Unknown')),
            ('true', _('Yes')),
            ('false', _('No')),
        )
        super().__init__(attrs, choices)

    def format_value(self, value):
        try:
            return {
                True: 'true', False: 'false',
                'true': 'true', 'false': 'false',
                # For backwards compatibility with Django < 2.2.
                '2': 'true', '3': 'false',
            }[value]
        except KeyError:
            return 'unknown'

    def value_from_datadict(self, data, files, name):
        value = data.get(name)
        return {
            True: True,
            'True': True,
            'False': False,
            False: False,
            'true': True,
            'false': False,
            # For backwards compatibility with Django < 2.2.
            '2': True,
            '3': False,
        }.get(value)
```
### 26 - django/forms/widgets.py:

Start line: 1, End line: 41

```python
"""
HTML Widget classes
"""

import copy
import datetime
import warnings
from collections import defaultdict
from itertools import chain

from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import formats
from django.utils.datastructures import OrderedSet
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
from django.utils.topological_sort import (
    CyclicDependencyError, stable_topological_sort,
)
from django.utils.translation import gettext_lazy as _

from .renderers import get_default_renderer

__all__ = (
    'Media', 'MediaDefiningClass', 'Widget', 'TextInput', 'NumberInput',
    'EmailInput', 'URLInput', 'PasswordInput', 'HiddenInput',
    'MultipleHiddenInput', 'FileInput', 'ClearableFileInput', 'Textarea',
    'DateInput', 'DateTimeInput', 'TimeInput', 'CheckboxInput', 'Select',
    'NullBooleanSelect', 'SelectMultiple', 'RadioSelect',
    'CheckboxSelectMultiple', 'MultiWidget', 'SplitDateTimeWidget',
    'SplitHiddenDateTimeWidget', 'SelectDateWidget',
)

MEDIA_TYPES = ('css', 'js')


class MediaOrderConflictWarning(RuntimeWarning):
    pass
```
### 27 - django/forms/widgets.py:

Start line: 984, End line: 1018

```python
class SelectDateWidget(Widget):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        date_context = {}
        year_choices = [(i, str(i)) for i in self.years]
        if not self.is_required:
            year_choices.insert(0, self.year_none_value)
        year_name = self.year_field % name
        date_context['year'] = self.select_widget(attrs, choices=year_choices).get_context(
            name=year_name,
            value=context['widget']['value']['year'],
            attrs={**context['widget']['attrs'], 'id': 'id_%s' % year_name},
        )
        month_choices = list(self.months.items())
        if not self.is_required:
            month_choices.insert(0, self.month_none_value)
        month_name = self.month_field % name
        date_context['month'] = self.select_widget(attrs, choices=month_choices).get_context(
            name=month_name,
            value=context['widget']['value']['month'],
            attrs={**context['widget']['attrs'], 'id': 'id_%s' % month_name},
        )
        day_choices = [(i, i) for i in range(1, 32)]
        if not self.is_required:
            day_choices.insert(0, self.day_none_value)
        day_name = self.day_field % name
        date_context['day'] = self.select_widget(attrs, choices=day_choices,).get_context(
            name=day_name,
            value=context['widget']['value']['day'],
            attrs={**context['widget']['attrs'], 'id': 'id_%s' % day_name},
        )
        subwidgets = []
        for field in self._parse_date_fmt():
            subwidgets.append(date_context[field]['widget'])
        context['widget']['subwidgets'] = subwidgets
        return context
```
### 28 - django/forms/widgets.py:

Start line: 587, End line: 619

```python
class ChoiceWidget(Widget):

    def optgroups(self, name, value, attrs=None):
        """Return a list of optgroups for this widget."""
        groups = []
        has_selected = False

        for index, (option_value, option_label) in enumerate(self.choices):
            if option_value is None:
                option_value = ''

            subgroup = []
            if isinstance(option_label, (list, tuple)):
                group_name = option_value
                subindex = 0
                choices = option_label
            else:
                group_name = None
                subindex = None
                choices = [(option_value, option_label)]
            groups.append((group_name, subgroup, index))

            for subvalue, sublabel in choices:
                selected = (
                    (not has_selected or self.allow_multiple_selected) and
                    str(subvalue) in value
                )
                has_selected |= selected
                subgroup.append(self.create_option(
                    name, subvalue, sublabel, selected, index,
                    subindex=subindex, attrs=attrs,
                ))
                if subindex is not None:
                    subindex += 1
        return groups
```
### 30 - django/forms/widgets.py:

Start line: 1066, End line: 1089

```python
class SelectDateWidget(Widget):

    def value_from_datadict(self, data, files, name):
        y = data.get(self.year_field % name)
        m = data.get(self.month_field % name)
        d = data.get(self.day_field % name)
        if y == m == d == '':
            return None
        if y is not None and m is not None and d is not None:
            input_format = get_format('DATE_INPUT_FORMATS')[0]
            input_format = formats.sanitize_strftime_format(input_format)
            try:
                date_value = datetime.date(int(y), int(m), int(d))
            except ValueError:
                # Return pseudo-ISO dates with zeros for any unselected values,
                # e.g. '2017-0-23'.
                return '%s-%s-%s' % (y or 0, m or 0, d or 0)
            return date_value.strftime(input_format)
        return data.get(name)

    def value_omitted_from_data(self, data, files, name):
        return not any(
            ('{}_{}'.format(name, interval) in data)
            for interval in ('year', 'month', 'day')
        )
```
### 32 - django/forms/widgets.py:

Start line: 434, End line: 446

```python
class ClearableFileInput(FileInput):

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        checkbox_name = self.clear_checkbox_name(name)
        checkbox_id = self.clear_checkbox_id(checkbox_name)
        context['widget'].update({
            'checkbox_name': checkbox_name,
            'checkbox_id': checkbox_id,
            'is_initial': self.is_initial(value),
            'input_text': self.input_text,
            'initial_text': self.initial_text,
            'clear_checkbox_label': self.clear_checkbox_label,
        })
        return context
```
### 37 - django/forms/widgets.py:

Start line: 379, End line: 399

```python
class FileInput(Input):
    input_type = 'file'
    needs_multipart_form = True
    template_name = 'django/forms/widgets/file.html'

    def format_value(self, value):
        """File input never renders a value."""
        return

    def value_from_datadict(self, data, files, name):
        "File widgets take data from FILES, not POST"
        return files.get(name)

    def value_omitted_from_data(self, data, files, name):
        return name not in files

    def use_required_attribute(self, initial):
        return super().use_required_attribute(initial) and not initial


FILE_INPUT_CONTRADICTION = object()
```
### 38 - django/forms/widgets.py:

Start line: 897, End line: 921

```python
class SplitDateTimeWidget(MultiWidget):
    """
    A widget that splits datetime input into two <input type="text"> boxes.
    """
    supports_microseconds = False
    template_name = 'django/forms/widgets/splitdatetime.html'

    def __init__(self, attrs=None, date_format=None, time_format=None, date_attrs=None, time_attrs=None):
        widgets = (
            DateInput(
                attrs=attrs if date_attrs is None else date_attrs,
                format=date_format,
            ),
            TimeInput(
                attrs=attrs if time_attrs is None else time_attrs,
                format=time_format,
            ),
        )
        super().__init__(widgets)

    def decompress(self, value):
        if value:
            value = to_current_timezone(value)
            return [value.date(), value.time()]
        return [None, None]
```
### 39 - django/forms/widgets.py:

Start line: 512, End line: 548

```python
class CheckboxInput(Input):
    input_type = 'checkbox'
    template_name = 'django/forms/widgets/checkbox.html'

    def __init__(self, attrs=None, check_test=None):
        super().__init__(attrs)
        # check_test is a callable that takes a value and returns True
        # if the checkbox should be checked for that value.
        self.check_test = boolean_check if check_test is None else check_test

    def format_value(self, value):
        """Only return the 'value' attribute if value isn't empty."""
        if value is True or value is False or value is None or value == '':
            return
        return str(value)

    def get_context(self, name, value, attrs):
        if self.check_test(value):
            attrs = {**(attrs or {}), 'checked': True}
        return super().get_context(name, value, attrs)

    def value_from_datadict(self, data, files, name):
        if name not in data:
            # A missing value means False because HTML form submission does not
            # send results for unselected checkboxes.
            return False
        value = data.get(name)
        # Translate true and false strings to boolean values.
        values = {'true': True, 'false': False}
        if isinstance(value, str):
            value = values.get(value.lower(), value)
        return bool(value)

    def value_omitted_from_data(self, data, files, name):
        # HTML checkboxes don't appear in POST data if not checked, so it's
        # never known if the value is actually omitted.
        return False
```
### 40 - django/forms/widgets.py:

Start line: 924, End line: 933

```python
class SplitHiddenDateTimeWidget(SplitDateTimeWidget):
    """
    A widget that splits datetime input into two <input type="hidden"> inputs.
    """
    template_name = 'django/forms/widgets/splithiddendatetime.html'

    def __init__(self, attrs=None, date_format=None, time_format=None, date_attrs=None, time_attrs=None):
        super().__init__(attrs, date_format, time_format, date_attrs, time_attrs)
        for widget in self.widgets:
            widget.input_type = 'hidden'
```
### 47 - django/forms/widgets.py:

Start line: 284, End line: 300

```python
class Input(Widget):
    """
    Base class for all <input> widgets.
    """
    input_type = None  # Subclasses must define this.
    template_name = 'django/forms/widgets/input.html'

    def __init__(self, attrs=None):
        if attrs is not None:
            attrs = attrs.copy()
            self.input_type = attrs.pop('type', self.input_type)
        super().__init__(attrs)

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context['widget']['type'] = self.input_type
        return context
```
### 60 - django/forms/widgets.py:

Start line: 1020, End line: 1043

```python
class SelectDateWidget(Widget):

    def format_value(self, value):
        """
        Return a dict containing the year, month, and day of the current value.
        Use dict instead of a datetime to allow invalid dates such as February
        31 to display correctly.
        """
        year, month, day = None, None, None
        if isinstance(value, (datetime.date, datetime.datetime)):
            year, month, day = value.year, value.month, value.day
        elif isinstance(value, str):
            match = self.date_re.match(value)
            if match:
                # Convert any zeros in the date to empty strings to match the
                # empty option value.
                year, month, day = [int(val) or '' for val in match.groups()]
            else:
                input_format = get_format('DATE_INPUT_FORMATS')[0]
                try:
                    d = datetime.datetime.strptime(value, input_format)
                except ValueError:
                    pass
                else:
                    year, month, day = d.year, d.month, d.day
        return {'year': year, 'month': month, 'day': day}
```
### 61 - django/forms/widgets.py:

Start line: 303, End line: 339

```python
class TextInput(Input):
    input_type = 'text'
    template_name = 'django/forms/widgets/text.html'


class NumberInput(Input):
    input_type = 'number'
    template_name = 'django/forms/widgets/number.html'


class EmailInput(Input):
    input_type = 'email'
    template_name = 'django/forms/widgets/email.html'


class URLInput(Input):
    input_type = 'url'
    template_name = 'django/forms/widgets/url.html'


class PasswordInput(Input):
    input_type = 'password'
    template_name = 'django/forms/widgets/password.html'

    def __init__(self, attrs=None, render_value=False):
        super().__init__(attrs)
        self.render_value = render_value

    def get_context(self, name, value, attrs):
        if not self.render_value:
            value = None
        return super().get_context(name, value, attrs)


class HiddenInput(Input):
    input_type = 'hidden'
    template_name = 'django/forms/widgets/hidden.html'
```
### 65 - django/forms/widgets.py:

Start line: 469, End line: 509

```python
class Textarea(Widget):
    template_name = 'django/forms/widgets/textarea.html'

    def __init__(self, attrs=None):
        # Use slightly better defaults than HTML's 20x2 box
        default_attrs = {'cols': '40', 'rows': '10'}
        if attrs:
            default_attrs.update(attrs)
        super().__init__(default_attrs)


class DateTimeBaseInput(TextInput):
    format_key = ''
    supports_microseconds = False

    def __init__(self, attrs=None, format=None):
        super().__init__(attrs)
        self.format = format or None

    def format_value(self, value):
        return formats.localize_input(value, self.format or formats.get_format(self.format_key)[0])


class DateInput(DateTimeBaseInput):
    format_key = 'DATE_INPUT_FORMATS'
    template_name = 'django/forms/widgets/date.html'


class DateTimeInput(DateTimeBaseInput):
    format_key = 'DATETIME_INPUT_FORMATS'
    template_name = 'django/forms/widgets/datetime.html'


class TimeInput(DateTimeBaseInput):
    format_key = 'TIME_INPUT_FORMATS'
    template_name = 'django/forms/widgets/time.html'


# Defined at module level so that CheckboxInput is picklable (#17976)
def boolean_check(v):
    return not (v is False or v is None or v == '')
```
### 69 - django/forms/widgets.py:

Start line: 160, End line: 196

```python
def media_property(cls):
    def _media(self):
        # Get the media property of the superclass, if it exists
        sup_cls = super(cls, self)
        try:
            base = sup_cls.media
        except AttributeError:
            base = Media()

        # Get the media definition for this class
        definition = getattr(cls, 'Media', None)
        if definition:
            extend = getattr(definition, 'extend', True)
            if extend:
                if extend is True:
                    m = base
                else:
                    m = Media()
                    for medium in extend:
                        m = m + base[medium]
                return m + Media(definition)
            return Media(definition)
        return base
    return property(_media)


class MediaDefiningClass(type):
    """
    Metaclass for classes that can have media definitions.
    """
    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)

        if 'media' not in attrs:
            new_class.media = media_property(new_class)

        return new_class
```
### 104 - django/forms/widgets.py:

Start line: 114, End line: 157

```python
@html_safe
class Media:

    @staticmethod
    def merge(*lists):
        """
        Merge lists while trying to keep the relative order of the elements.
        Warn if the lists have the same elements in a different relative order.

        For static assets it can be important to have them included in the DOM
        in a certain order. In JavaScript you may not be able to reference a
        global or in CSS you might want to override a style.
        """
        dependency_graph = defaultdict(set)
        all_items = OrderedSet()
        for list_ in filter(None, lists):
            head = list_[0]
            # The first items depend on nothing but have to be part of the
            # dependency graph to be included in the result.
            dependency_graph.setdefault(head, set())
            for item in list_:
                all_items.add(item)
                # No self dependencies
                if head != item:
                    dependency_graph[item].add(head)
                head = item
        try:
            return stable_topological_sort(all_items, dependency_graph)
        except CyclicDependencyError:
            warnings.warn(
                'Detected duplicate Media files in an opposite order: {}'.format(
                    ', '.join(repr(list_) for list_ in lists)
                ), MediaOrderConflictWarning,
            )
            return list(all_items)

    def __add__(self, other):
        combined = Media()
        combined._css_lists = self._css_lists[:]
        combined._js_lists = self._js_lists[:]
        for item in other._css_lists:
            if item and item not in self._css_lists:
                combined._css_lists.append(item)
        for item in other._js_lists:
            if item and item not in self._js_lists:
                combined._js_lists.append(item)
        return combined
```
