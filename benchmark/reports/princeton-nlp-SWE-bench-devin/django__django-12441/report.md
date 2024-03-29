# django__django-12441

| **django/django** | `da4923ea87124102aae4455e947ce24599c0365b` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 634 |
| **Any found context length** | 634 |
| **Avg pos** | 5.5 |
| **Min pos** | 1 |
| **Max pos** | 10 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/forms/forms.py b/django/forms/forms.py
--- a/django/forms/forms.py
+++ b/django/forms/forms.py
@@ -191,7 +191,8 @@ def add_initial_prefix(self, field_name):
 
     def _html_output(self, normal_row, error_row, row_ender, help_text_html, errors_on_separate_row):
         "Output HTML. Used by as_table(), as_ul(), as_p()."
-        top_errors = self.non_field_errors()  # Errors that should be displayed above all fields.
+        # Errors that should be displayed above all fields.
+        top_errors = self.non_field_errors().copy()
         output, hidden_fields = [], []
 
         for name, field in self.fields.items():
diff --git a/django/forms/utils.py b/django/forms/utils.py
--- a/django/forms/utils.py
+++ b/django/forms/utils.py
@@ -92,6 +92,11 @@ def __init__(self, initlist=None, error_class=None):
     def as_data(self):
         return ValidationError(self.data).error_list
 
+    def copy(self):
+        copy = super().copy()
+        copy.error_class = self.error_class
+        return copy
+
     def get_json_data(self, escape_html=False):
         errors = []
         for error in self.as_data():

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/forms/forms.py | 194 | 194 | 1 | 1 | 634
| django/forms/utils.py | 95 | 95 | 10 | 2 | 2591


## Problem Statement

```
Calling a form method _html_output modifies the self._errors dict for NON_FIELD_ERRORS if there are hidden field with errors
Description
	
Each time the _html_output method of a form is called, it appends the errors of the hidden field errors to the NON_FIELD_ERRORS (all) entry.
This happen for example when the form methods as_p() as_table() as_ul() are called multiple time, or any other method that themselves call one of them.
For example, a test form with an hidden input field that add errors during the clean call.
Python 3.6.5 (default, Apr 25 2018, 14:26:36)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.
In [1]: import django
In [2]: django.__version__
Out[2]: '2.1.7'
In [3]: from django import forms
 ...:
In [4]: class TestForm(forms.Form):
 ...:	 hidden_input = forms.CharField(widget=forms.HiddenInput)
 ...:
 ...:	 def clean(self):
 ...:		 self.add_error(None, 'Form error')
 ...:		 self.add_error('hidden_input', 'Hidden input error')
 ...:
In [5]: test_form = TestForm({})
In [6]: test_form.errors
Out[6]:
{'hidden_input': ['This field is required.', 'Hidden input error'],
 '__all__': ['Form error']}
In [7]: print(test_form.as_table())
<tr><td colspan="2"><ul class="errorlist nonfield"><li>Form error</li><li>(Hidden field hidden_input) This field is required.</li><li>(Hidden field hidden_input) Hidden input error</li></ul><input type="hidden" name="hidden_input" id="id_hidden_input"></td></tr>
In [8]: test_form.errors
Out[8]:
{'hidden_input': ['This field is required.', 'Hidden input error'],
 '__all__': ['Form error', '(Hidden field hidden_input) This field is required.', '(Hidden field hidden_input) Hidden input error']}
In [9]: print(test_form.as_table())
<tr><td colspan="2"><ul class="errorlist nonfield"><li>Form error</li><li>(Hidden field hidden_input) This field is required.</li><li>(Hidden field hidden_input) Hidden input error</li><li>(Hidden field hidden_input) This field is required.</li><li>(Hidden field hidden_input) Hidden input error</li></ul><input type="hidden" name="hidden_input" id="id_hidden_input"></td></tr>
In [10]: test_form.errors
Out[10]:
{'hidden_input': ['This field is required.', 'Hidden input error'],
 '__all__': ['Form error', '(Hidden field hidden_input) This field is required.', '(Hidden field hidden_input) Hidden input error', '(Hidden field hidden_input) This field is required.', '(Hidden field hidden_input) Hidden input error']}
In [11]: test_form.non_field_errors()
Out[11]: ['Form error', '(Hidden field hidden_input) This field is required.', '(Hidden field hidden_input) Hidden input error', '(Hidden field hidden_input) This field is required.', '(Hidden field hidden_input) Hidden input error']
This bug affects probably also version 2.2.
A simple fix would be to use a copy of the error list before adding the hidden field errors in the file django/forms/forms.py:
--- forms.py	2019-03-17 18:59:04.000000000 +0100
+++ forms_fixed.py	2019-03-17 19:00:08.000000000 +0100
@@ -194,7 +194,7 @@
	 def _html_output(self, normal_row, error_row, row_ender, help_text_html, errors_on_separate_row):
		 "Output HTML. Used by as_table(), as_ul(), as_p()."
-		top_errors = self.non_field_errors() # Errors that should be displayed above all fields.
+		top_errors = self.non_field_errors().copy() # Errors that should be displayed above all fields.
		 output, hidden_fields = [], []
		 for name, field in self.fields.items():

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/forms/forms.py** | 192 | 267| 634 | 634 | 4012 | 
| 2 | **1 django/forms/forms.py** | 289 | 305| 165 | 799 | 4012 | 
| 3 | **1 django/forms/forms.py** | 354 | 375| 172 | 971 | 4012 | 
| 4 | **2 django/forms/utils.py** | 44 | 76| 241 | 1212 | 5244 | 
| 5 | **2 django/forms/forms.py** | 307 | 352| 438 | 1650 | 5244 | 
| 6 | **2 django/forms/forms.py** | 269 | 277| 132 | 1782 | 5244 | 
| 7 | **2 django/forms/forms.py** | 377 | 397| 210 | 1992 | 5244 | 
| 8 | **2 django/forms/forms.py** | 279 | 287| 113 | 2105 | 5244 | 
| 9 | **2 django/forms/utils.py** | 139 | 146| 132 | 2237 | 5244 | 
| **-> 10 <-** | **2 django/forms/utils.py** | 79 | 137| 354 | 2591 | 5244 | 
| 11 | 3 django/forms/formsets.py | 266 | 298| 241 | 2832 | 9130 | 
| 12 | **3 django/forms/forms.py** | 133 | 150| 142 | 2974 | 9130 | 
| 13 | **3 django/forms/forms.py** | 168 | 190| 182 | 3156 | 9130 | 
| 14 | 3 django/forms/formsets.py | 318 | 358| 390 | 3546 | 9130 | 
| 15 | 4 django/contrib/admin/helpers.py | 385 | 408| 195 | 3741 | 12321 | 
| 16 | **4 django/forms/forms.py** | 428 | 450| 195 | 3936 | 12321 | 
| 17 | **4 django/forms/forms.py** | 399 | 426| 190 | 4126 | 12321 | 
| 18 | 4 django/forms/formsets.py | 1 | 25| 203 | 4329 | 12321 | 
| 19 | 5 django/forms/boundfield.py | 52 | 77| 180 | 4509 | 14442 | 
| 20 | 6 django/forms/widgets.py | 337 | 371| 278 | 4787 | 22448 | 
| 21 | 7 django/forms/models.py | 350 | 380| 233 | 5020 | 34075 | 
| 22 | 7 django/contrib/admin/helpers.py | 33 | 67| 230 | 5250 | 34075 | 
| 23 | 7 django/forms/models.py | 752 | 773| 194 | 5444 | 34075 | 
| 24 | 7 django/contrib/admin/helpers.py | 92 | 120| 249 | 5693 | 34075 | 
| 25 | 7 django/forms/widgets.py | 1 | 42| 299 | 5992 | 34075 | 
| 26 | **7 django/forms/forms.py** | 53 | 109| 525 | 6517 | 34075 | 
| 27 | **7 django/forms/forms.py** | 1 | 19| 119 | 6636 | 34075 | 
| 28 | 7 django/forms/widgets.py | 298 | 334| 204 | 6840 | 34075 | 
| 29 | 7 django/forms/models.py | 1 | 28| 218 | 7058 | 34075 | 
| 30 | 7 django/contrib/admin/helpers.py | 123 | 149| 220 | 7278 | 34075 | 
| 31 | 8 django/forms/fields.py | 45 | 126| 773 | 8051 | 43088 | 
| 32 | 8 django/forms/models.py | 952 | 985| 367 | 8418 | 43088 | 
| 33 | 8 django/forms/boundfield.py | 1 | 33| 239 | 8657 | 43088 | 
| 34 | 9 django/contrib/admin/options.py | 1 | 95| 756 | 9413 | 61576 | 
| 35 | 9 django/forms/boundfield.py | 99 | 130| 267 | 9680 | 61576 | 
| 36 | **9 django/forms/forms.py** | 452 | 499| 382 | 10062 | 61576 | 
| 37 | 9 django/forms/formsets.py | 360 | 391| 261 | 10323 | 61576 | 
| 38 | 9 django/contrib/admin/options.py | 2069 | 2121| 451 | 10774 | 61576 | 
| 39 | 10 django/contrib/admin/templatetags/admin_list.py | 301 | 353| 350 | 11124 | 65405 | 
| 40 | 10 django/forms/models.py | 382 | 410| 240 | 11364 | 65405 | 
| 41 | 10 django/forms/models.py | 679 | 750| 732 | 12096 | 65405 | 
| 42 | 10 django/forms/widgets.py | 194 | 276| 622 | 12718 | 65405 | 
| 43 | 11 django/template/backends/jinja2.py | 91 | 126| 248 | 12966 | 66227 | 
| 44 | 11 django/forms/fields.py | 128 | 171| 290 | 13256 | 66227 | 
| 45 | 12 django/contrib/admin/views/main.py | 1 | 45| 324 | 13580 | 70484 | 
| 46 | 12 django/contrib/admin/helpers.py | 306 | 332| 171 | 13751 | 70484 | 
| 47 | 12 django/contrib/admin/options.py | 1523 | 1609| 760 | 14511 | 70484 | 
| 48 | 13 django/utils/html.py | 352 | 379| 212 | 14723 | 73586 | 
| 49 | 14 docs/_ext/djangodocs.py | 108 | 169| 526 | 15249 | 76631 | 
| 50 | 14 django/forms/widgets.py | 443 | 461| 195 | 15444 | 76631 | 
| 51 | 14 django/contrib/admin/helpers.py | 366 | 382| 138 | 15582 | 76631 | 
| 52 | 14 django/forms/fields.py | 579 | 604| 243 | 15825 | 76631 | 
| 53 | 15 django/contrib/postgres/forms/jsonb.py | 1 | 63| 345 | 16170 | 76976 | 
| 54 | 15 django/contrib/admin/helpers.py | 247 | 270| 235 | 16405 | 76976 | 
| 55 | 16 django/contrib/admin/checks.py | 881 | 929| 416 | 16821 | 85983 | 
| 56 | 17 django/db/models/fields/__init__.py | 364 | 390| 199 | 17020 | 103551 | 
| 57 | 18 django/contrib/admin/templatetags/admin_modify.py | 89 | 117| 203 | 17223 | 104519 | 
| 58 | **18 django/forms/forms.py** | 111 | 131| 177 | 17400 | 104519 | 
| 59 | 18 django/contrib/admin/templatetags/admin_list.py | 214 | 298| 809 | 18209 | 104519 | 
| 60 | 18 django/forms/widgets.py | 464 | 504| 275 | 18484 | 104519 | 
| 61 | 18 django/forms/models.py | 309 | 348| 387 | 18871 | 104519 | 
| 62 | 19 django/contrib/flatpages/forms.py | 52 | 70| 143 | 19014 | 105002 | 
| 63 | 19 django/forms/fields.py | 859 | 898| 298 | 19312 | 105002 | 
| 64 | 19 django/forms/fields.py | 540 | 556| 180 | 19492 | 105002 | 
| 65 | 20 django/core/exceptions.py | 1 | 96| 405 | 19897 | 106057 | 
| 66 | 21 django/views/debug.py | 322 | 351| 267 | 20164 | 110402 | 
| 67 | 21 django/views/debug.py | 387 | 455| 575 | 20739 | 110402 | 
| 68 | 22 django/contrib/admin/utils.py | 368 | 402| 305 | 21044 | 114519 | 
| 69 | 22 django/contrib/admin/options.py | 1465 | 1488| 319 | 21363 | 114519 | 
| 70 | 22 django/utils/html.py | 179 | 197| 161 | 21524 | 114519 | 
| 71 | 22 django/contrib/admin/options.py | 1110 | 1155| 482 | 22006 | 114519 | 
| 72 | 23 django/template/response.py | 1 | 43| 383 | 22389 | 115598 | 
| 73 | 24 django/db/models/base.py | 1229 | 1252| 172 | 22561 | 130970 | 
| 74 | 24 django/forms/widgets.py | 374 | 394| 138 | 22699 | 130970 | 
| 75 | 24 django/forms/models.py | 412 | 442| 243 | 22942 | 130970 | 
| 76 | 24 django/contrib/flatpages/forms.py | 1 | 50| 346 | 23288 | 130970 | 
| 77 | 24 django/contrib/admin/helpers.py | 1 | 30| 196 | 23484 | 130970 | 
| 78 | 24 django/forms/formsets.py | 228 | 264| 420 | 23904 | 130970 | 
| 79 | 24 django/forms/models.py | 207 | 276| 616 | 24520 | 130970 | 
| 80 | 24 django/contrib/admin/helpers.py | 70 | 89| 167 | 24687 | 130970 | 
| 81 | 24 django/utils/html.py | 92 | 115| 200 | 24887 | 130970 | 
| 82 | 24 django/forms/widgets.py | 507 | 543| 331 | 25218 | 130970 | 
| 83 | 24 django/contrib/admin/templatetags/admin_list.py | 105 | 194| 788 | 26006 | 130970 | 
| 84 | 24 django/db/models/base.py | 1254 | 1284| 255 | 26261 | 130970 | 
| 85 | 25 django/core/checks/messages.py | 53 | 76| 161 | 26422 | 131543 | 
| 86 | 25 django/contrib/admin/options.py | 1610 | 1634| 279 | 26701 | 131543 | 
| 87 | 25 django/contrib/admin/options.py | 1740 | 1821| 744 | 27445 | 131543 | 
| 88 | 26 django/contrib/contenttypes/admin.py | 1 | 80| 615 | 28060 | 132568 | 
| 89 | 26 django/forms/formsets.py | 393 | 431| 359 | 28419 | 132568 | 
| 90 | 26 django/contrib/admin/templatetags/admin_modify.py | 48 | 86| 391 | 28810 | 132568 | 
| 91 | 26 django/utils/html.py | 1 | 75| 614 | 29424 | 132568 | 
| 92 | 26 django/forms/widgets.py | 910 | 919| 107 | 29531 | 132568 | 
| 93 | 27 django/contrib/admin/widgets.py | 316 | 328| 114 | 29645 | 136432 | 
| 94 | 27 django/forms/models.py | 194 | 204| 131 | 29776 | 136432 | 
| 95 | 28 django/core/checks/templates.py | 1 | 36| 259 | 30035 | 136692 | 
| 96 | 28 docs/_ext/djangodocs.py | 234 | 270| 427 | 30462 | 136692 | 
| 97 | 28 django/forms/formsets.py | 105 | 122| 194 | 30656 | 136692 | 
| 98 | 29 django/template/backends/django.py | 79 | 111| 225 | 30881 | 137548 | 
| 99 | 29 django/contrib/admin/checks.py | 372 | 398| 281 | 31162 | 137548 | 
| 100 | 29 django/forms/boundfield.py | 79 | 97| 174 | 31336 | 137548 | 
| 101 | 29 django/db/models/fields/__init__.py | 338 | 362| 184 | 31520 | 137548 | 
| 102 | 30 django/views/csrf.py | 15 | 100| 835 | 32355 | 139092 | 
| 103 | 30 django/core/exceptions.py | 99 | 194| 649 | 33004 | 139092 | 
| 104 | 30 django/forms/widgets.py | 429 | 441| 122 | 33126 | 139092 | 
| 105 | 30 django/forms/widgets.py | 397 | 427| 193 | 33319 | 139092 | 
| 106 | 30 django/contrib/admin/helpers.py | 355 | 364| 134 | 33453 | 139092 | 
| 107 | 30 django/forms/fields.py | 770 | 829| 416 | 33869 | 139092 | 
| 108 | 30 django/db/models/fields/__init__.py | 244 | 306| 448 | 34317 | 139092 | 
| 109 | 30 django/forms/formsets.py | 45 | 84| 274 | 34591 | 139092 | 
| 110 | 30 django/forms/formsets.py | 28 | 42| 195 | 34786 | 139092 | 
| 111 | 30 django/forms/fields.py | 371 | 389| 134 | 34920 | 139092 | 
| 112 | 31 django/core/checks/security/base.py | 1 | 83| 732 | 35652 | 140873 | 
| 113 | 31 django/db/models/base.py | 1616 | 1664| 348 | 36000 | 140873 | 
| 114 | 32 docs/conf.py | 102 | 206| 900 | 36900 | 143911 | 
| 115 | 33 django/views/generic/edit.py | 1 | 67| 479 | 37379 | 145627 | 
| 116 | 33 django/forms/fields.py | 1133 | 1166| 293 | 37672 | 145627 | 
| 117 | 33 django/forms/models.py | 279 | 307| 288 | 37960 | 145627 | 
| 118 | 34 django/conf/locale/hr/formats.py | 5 | 43| 742 | 38702 | 146414 | 
| 119 | 35 django/forms/renderers.py | 1 | 71| 398 | 39100 | 146812 | 
| 120 | 35 django/forms/fields.py | 937 | 958| 152 | 39252 | 146812 | 
| 121 | 36 django/db/models/fields/related.py | 1191 | 1222| 180 | 39432 | 160496 | 
| 122 | 36 django/forms/fields.py | 173 | 205| 280 | 39712 | 160496 | 
| 123 | 37 django/utils/autoreload.py | 48 | 76| 156 | 39868 | 165213 | 
| 124 | **37 django/forms/forms.py** | 152 | 166| 125 | 39993 | 165213 | 
| 125 | 37 django/db/models/fields/related.py | 255 | 282| 269 | 40262 | 165213 | 
| 126 | 37 django/forms/formsets.py | 300 | 316| 171 | 40433 | 165213 | 
| 127 | 37 django/forms/formsets.py | 210 | 226| 186 | 40619 | 165213 | 
| 128 | 37 django/contrib/admin/templatetags/admin_list.py | 431 | 489| 343 | 40962 | 165213 | 
| 129 | 38 django/db/models/fields/mixins.py | 31 | 57| 173 | 41135 | 165556 | 
| 130 | 39 django/contrib/auth/tokens.py | 1 | 56| 365 | 41500 | 166368 | 
| 131 | 39 django/contrib/admin/checks.py | 770 | 791| 190 | 41690 | 166368 | 
| 132 | 39 django/db/models/fields/related.py | 1224 | 1341| 967 | 42657 | 166368 | 
| 133 | 39 django/forms/formsets.py | 179 | 208| 217 | 42874 | 166368 | 
| 134 | 39 django/db/models/fields/related.py | 127 | 154| 202 | 43076 | 166368 | 
| 135 | 39 django/forms/fields.py | 558 | 577| 171 | 43247 | 166368 | 
| 136 | 39 django/forms/models.py | 918 | 950| 353 | 43600 | 166368 | 
| 137 | 39 django/contrib/admin/options.py | 1932 | 1975| 403 | 44003 | 166368 | 
| 138 | 39 django/views/debug.py | 185 | 233| 467 | 44470 | 166368 | 
| 139 | 39 django/views/debug.py | 171 | 183| 143 | 44613 | 166368 | 
| 140 | 39 django/contrib/admin/helpers.py | 335 | 353| 181 | 44794 | 166368 | 
| 141 | 40 django/core/checks/model_checks.py | 1 | 86| 667 | 45461 | 168155 | 
| 142 | 40 django/contrib/admin/checks.py | 720 | 751| 219 | 45680 | 168155 | 
| 143 | 40 django/views/debug.py | 236 | 320| 761 | 46441 | 168155 | 
| 144 | 40 django/forms/boundfield.py | 35 | 50| 149 | 46590 | 168155 | 
| 145 | 40 django/core/checks/messages.py | 26 | 50| 259 | 46849 | 168155 | 
| 146 | 41 django/conf/locale/sl/formats.py | 5 | 43| 708 | 47557 | 168908 | 
| 147 | 42 django/db/models/options.py | 746 | 828| 827 | 48384 | 175930 | 
| 148 | 43 django/template/base.py | 1 | 94| 779 | 49163 | 183811 | 
| 149 | 44 django/conf/locale/mk/formats.py | 5 | 39| 594 | 49757 | 184450 | 
| 150 | 45 django/utils/formats.py | 1 | 57| 377 | 50134 | 186542 | 
| 151 | 45 django/forms/fields.py | 1089 | 1130| 353 | 50487 | 186542 | 
| 152 | 45 django/contrib/admin/helpers.py | 192 | 220| 276 | 50763 | 186542 | 
| 153 | 45 django/views/debug.py | 75 | 102| 218 | 50981 | 186542 | 
| 154 | 45 django/db/models/fields/related.py | 1 | 34| 244 | 51225 | 186542 | 
| 155 | 45 django/contrib/admin/templatetags/admin_list.py | 1 | 26| 170 | 51395 | 186542 | 
| 156 | 45 django/forms/fields.py | 392 | 413| 144 | 51539 | 186542 | 
| 157 | 45 django/contrib/admin/checks.py | 597 | 619| 155 | 51694 | 186542 | 
| 158 | 46 django/conf/locale/el/formats.py | 5 | 33| 455 | 52149 | 187042 | 
| 159 | 47 django/forms/__init__.py | 1 | 12| 0 | 52149 | 187132 | 
| 160 | 47 django/forms/fields.py | 666 | 705| 293 | 52442 | 187132 | 
| 161 | 47 django/contrib/admin/options.py | 2011 | 2032| 221 | 52663 | 187132 | 
| 162 | 47 django/forms/widgets.py | 763 | 786| 204 | 52867 | 187132 | 
| 163 | 47 django/db/models/base.py | 1163 | 1191| 213 | 53080 | 187132 | 
| 164 | 47 django/forms/boundfield.py | 169 | 228| 503 | 53583 | 187132 | 
| 165 | 48 django/conf/locale/lt/formats.py | 5 | 44| 676 | 54259 | 187853 | 
| 166 | 48 django/forms/fields.py | 1 | 42| 350 | 54609 | 187853 | 
| 167 | 48 django/forms/fields.py | 208 | 239| 274 | 54883 | 187853 | 
| 168 | 49 django/conf/locale/nb/formats.py | 5 | 37| 593 | 55476 | 188491 | 
| 169 | 49 django/contrib/contenttypes/admin.py | 83 | 130| 410 | 55886 | 188491 | 
| 170 | 50 django/conf/locale/en_GB/formats.py | 5 | 37| 655 | 56541 | 189191 | 
| 171 | 51 django/conf/locale/cy/formats.py | 5 | 33| 529 | 57070 | 189765 | 
| 172 | 52 django/views/i18n.py | 88 | 191| 711 | 57781 | 192312 | 
| 173 | 53 django/conf/locale/en/formats.py | 5 | 38| 610 | 58391 | 192967 | 
| 174 | 54 django/conf/locale/id/formats.py | 5 | 47| 670 | 59061 | 193682 | 
| 175 | 55 django/conf/locale/en_AU/formats.py | 5 | 37| 655 | 59716 | 194382 | 
| 176 | 56 django/conf/locale/hi/formats.py | 5 | 22| 125 | 59841 | 194551 | 
| 177 | 57 django/contrib/admindocs/utils.py | 1 | 25| 151 | 59992 | 196457 | 
| 178 | 57 django/db/models/fields/__init__.py | 1582 | 1603| 183 | 60175 | 196457 | 
| 179 | 58 django/contrib/postgres/forms/array.py | 105 | 131| 219 | 60394 | 198051 | 
| 180 | 58 django/contrib/admin/checks.py | 1020 | 1047| 194 | 60588 | 198051 | 
| 181 | 58 django/contrib/postgres/forms/array.py | 197 | 235| 271 | 60859 | 198051 | 
| 182 | 59 django/conf/locale/sr/formats.py | 5 | 40| 726 | 61585 | 198822 | 
| 183 | 59 django/forms/fields.py | 444 | 477| 225 | 61810 | 198822 | 
| 185 | 60 django/db/models/fields/__init__.py | 1056 | 1085| 218 | 62754 | 199593 | 


### Hint

```
I didn't reproduce but the report and the suggested patch make sense, accepting on that basis. Are you interested in submitting a Github pull request incorporating your patch and a regression test?
```

## Patch

```diff
diff --git a/django/forms/forms.py b/django/forms/forms.py
--- a/django/forms/forms.py
+++ b/django/forms/forms.py
@@ -191,7 +191,8 @@ def add_initial_prefix(self, field_name):
 
     def _html_output(self, normal_row, error_row, row_ender, help_text_html, errors_on_separate_row):
         "Output HTML. Used by as_table(), as_ul(), as_p()."
-        top_errors = self.non_field_errors()  # Errors that should be displayed above all fields.
+        # Errors that should be displayed above all fields.
+        top_errors = self.non_field_errors().copy()
         output, hidden_fields = [], []
 
         for name, field in self.fields.items():
diff --git a/django/forms/utils.py b/django/forms/utils.py
--- a/django/forms/utils.py
+++ b/django/forms/utils.py
@@ -92,6 +92,11 @@ def __init__(self, initlist=None, error_class=None):
     def as_data(self):
         return ValidationError(self.data).error_list
 
+    def copy(self):
+        copy = super().copy()
+        copy.error_class = self.error_class
+        return copy
+
     def get_json_data(self, escape_html=False):
         errors = []
         for error in self.as_data():

```

## Test Patch

```diff
diff --git a/tests/forms_tests/tests/test_forms.py b/tests/forms_tests/tests/test_forms.py
--- a/tests/forms_tests/tests/test_forms.py
+++ b/tests/forms_tests/tests/test_forms.py
@@ -1245,6 +1245,22 @@ def clean(self):
         self.assertTrue(f.has_error(NON_FIELD_ERRORS, 'password_mismatch'))
         self.assertFalse(f.has_error(NON_FIELD_ERRORS, 'anything'))
 
+    def test_html_output_with_hidden_input_field_errors(self):
+        class TestForm(Form):
+            hidden_input = CharField(widget=HiddenInput)
+
+            def clean(self):
+                self.add_error(None, 'Form error')
+
+        f = TestForm(data={})
+        error_dict = {
+            'hidden_input': ['This field is required.'],
+            '__all__': ['Form error'],
+        }
+        self.assertEqual(f.errors, error_dict)
+        f.as_table()
+        self.assertEqual(f.errors, error_dict)
+
     def test_dynamic_construction(self):
         # It's possible to construct a Form dynamically by adding to the self.fields
         # dictionary in __init__(). Don't forget to call Form.__init__() within the

```


## Code snippets

### 1 - django/forms/forms.py:

Start line: 192, End line: 267

```python
@html_safe
class BaseForm:

    def _html_output(self, normal_row, error_row, row_ender, help_text_html, errors_on_separate_row):
        "Output HTML. Used by as_table(), as_ul(), as_p()."
        top_errors = self.non_field_errors()  # Errors that should be displayed above all fields.
        output, hidden_fields = [], []

        for name, field in self.fields.items():
            html_class_attr = ''
            bf = self[name]
            bf_errors = self.error_class(bf.errors)
            if bf.is_hidden:
                if bf_errors:
                    top_errors.extend(
                        [_('(Hidden field %(name)s) %(error)s') % {'name': name, 'error': str(e)}
                         for e in bf_errors])
                hidden_fields.append(str(bf))
            else:
                # Create a 'class="..."' attribute if the row should have any
                # CSS classes applied.
                css_classes = bf.css_classes()
                if css_classes:
                    html_class_attr = ' class="%s"' % css_classes

                if errors_on_separate_row and bf_errors:
                    output.append(error_row % str(bf_errors))

                if bf.label:
                    label = conditional_escape(bf.label)
                    label = bf.label_tag(label) or ''
                else:
                    label = ''

                if field.help_text:
                    help_text = help_text_html % field.help_text
                else:
                    help_text = ''

                output.append(normal_row % {
                    'errors': bf_errors,
                    'label': label,
                    'field': bf,
                    'help_text': help_text,
                    'html_class_attr': html_class_attr,
                    'css_classes': css_classes,
                    'field_name': bf.html_name,
                })

        if top_errors:
            output.insert(0, error_row % top_errors)

        if hidden_fields:  # Insert any hidden fields in the last row.
            str_hidden = ''.join(hidden_fields)
            if output:
                last_row = output[-1]
                # Chop off the trailing row_ender (e.g. '</td></tr>') and
                # insert the hidden fields.
                if not last_row.endswith(row_ender):
                    # This can happen in the as_p() case (and possibly others
                    # that users write): if there are only top errors, we may
                    # not be able to conscript the last row for our purposes,
                    # so insert a new, empty row.
                    last_row = (normal_row % {
                        'errors': '',
                        'label': '',
                        'field': '',
                        'help_text': '',
                        'html_class_attr': html_class_attr,
                        'css_classes': '',
                        'field_name': '',
                    })
                    output.append(last_row)
                output[-1] = last_row[:-len(row_ender)] + str_hidden + row_ender
            else:
                # If there aren't any rows in the output, just append the
                # hidden fields.
                output.append(str_hidden)
        return mark_safe('\n'.join(output))
```
### 2 - django/forms/forms.py:

Start line: 289, End line: 305

```python
@html_safe
class BaseForm:

    def as_p(self):
        "Return this form rendered as HTML <p>s."
        return self._html_output(
            normal_row='<p%(html_class_attr)s>%(label)s %(field)s%(help_text)s</p>',
            error_row='%s',
            row_ender='</p>',
            help_text_html=' <span class="helptext">%s</span>',
            errors_on_separate_row=True,
        )

    def non_field_errors(self):
        """
        Return an ErrorList of errors that aren't associated with a particular
        field -- i.e., from Form.clean(). Return an empty ErrorList if there
        are none.
        """
        return self.errors.get(NON_FIELD_ERRORS, self.error_class(error_class='nonfield'))
```
### 3 - django/forms/forms.py:

Start line: 354, End line: 375

```python
@html_safe
class BaseForm:

    def has_error(self, field, code=None):
        return field in self.errors and (
            code is None or
            any(error.code == code for error in self.errors.as_data()[field])
        )

    def full_clean(self):
        """
        Clean all of self.data and populate self._errors and self.cleaned_data.
        """
        self._errors = ErrorDict()
        if not self.is_bound:  # Stop further processing.
            return
        self.cleaned_data = {}
        # If the form is permitted to be empty, and none of the form data has
        # changed from the initial data, short circuit any validation.
        if self.empty_permitted and not self.has_changed():
            return

        self._clean_fields()
        self._clean_form()
        self._post_clean()
```
### 4 - django/forms/utils.py:

Start line: 44, End line: 76

```python
@html_safe
class ErrorDict(dict):
    """
    A collection of errors that knows how to display itself in various formats.

    The dictionary keys are the field names, and the values are the errors.
    """
    def as_data(self):
        return {f: e.as_data() for f, e in self.items()}

    def get_json_data(self, escape_html=False):
        return {f: e.get_json_data(escape_html) for f, e in self.items()}

    def as_json(self, escape_html=False):
        return json.dumps(self.get_json_data(escape_html))

    def as_ul(self):
        if not self:
            return ''
        return format_html(
            '<ul class="errorlist">{}</ul>',
            format_html_join('', '<li>{}{}</li>', self.items())
        )

    def as_text(self):
        output = []
        for field, errors in self.items():
            output.append('* %s' % field)
            output.append('\n'.join('  * %s' % e for e in errors))
        return '\n'.join(output)

    def __str__(self):
        return self.as_ul()
```
### 5 - django/forms/forms.py:

Start line: 307, End line: 352

```python
@html_safe
class BaseForm:

    def add_error(self, field, error):
        """
        Update the content of `self._errors`.

        The `field` argument is the name of the field to which the errors
        should be added. If it's None, treat the errors as NON_FIELD_ERRORS.

        The `error` argument can be a single error, a list of errors, or a
        dictionary that maps field names to lists of errors. An "error" can be
        either a simple string or an instance of ValidationError with its
        message attribute set and a "list or dictionary" can be an actual
        `list` or `dict` or an instance of ValidationError with its
        `error_list` or `error_dict` attribute set.

        If `error` is a dictionary, the `field` argument *must* be None and
        errors will be added to the fields that correspond to the keys of the
        dictionary.
        """
        if not isinstance(error, ValidationError):
            # Normalize to ValidationError and let its constructor
            # do the hard work of making sense of the input.
            error = ValidationError(error)

        if hasattr(error, 'error_dict'):
            if field is not None:
                raise TypeError(
                    "The argument `field` must be `None` when the `error` "
                    "argument contains errors for multiple fields."
                )
            else:
                error = error.error_dict
        else:
            error = {field or NON_FIELD_ERRORS: error.error_list}

        for field, error_list in error.items():
            if field not in self.errors:
                if field != NON_FIELD_ERRORS and field not in self.fields:
                    raise ValueError(
                        "'%s' has no field named '%s'." % (self.__class__.__name__, field))
                if field == NON_FIELD_ERRORS:
                    self._errors[field] = self.error_class(error_class='nonfield')
                else:
                    self._errors[field] = self.error_class()
            self._errors[field].extend(error_list)
            if field in self.cleaned_data:
                del self.cleaned_data[field]
```
### 6 - django/forms/forms.py:

Start line: 269, End line: 277

```python
@html_safe
class BaseForm:

    def as_table(self):
        "Return this form rendered as HTML <tr>s -- excluding the <table></table>."
        return self._html_output(
            normal_row='<tr%(html_class_attr)s><th>%(label)s</th><td>%(errors)s%(field)s%(help_text)s</td></tr>',
            error_row='<tr><td colspan="2">%s</td></tr>',
            row_ender='</td></tr>',
            help_text_html='<br><span class="helptext">%s</span>',
            errors_on_separate_row=False,
        )
```
### 7 - django/forms/forms.py:

Start line: 377, End line: 397

```python
@html_safe
class BaseForm:

    def _clean_fields(self):
        for name, field in self.fields.items():
            # value_from_datadict() gets the data from the data dictionaries.
            # Each widget type knows how to retrieve its own data, because some
            # widgets split data over several HTML fields.
            if field.disabled:
                value = self.get_initial_for_field(field, name)
            else:
                value = field.widget.value_from_datadict(self.data, self.files, self.add_prefix(name))
            try:
                if isinstance(field, FileField):
                    initial = self.get_initial_for_field(field, name)
                    value = field.clean(value, initial)
                else:
                    value = field.clean(value)
                self.cleaned_data[name] = value
                if hasattr(self, 'clean_%s' % name):
                    value = getattr(self, 'clean_%s' % name)()
                    self.cleaned_data[name] = value
            except ValidationError as e:
                self.add_error(name, e)
```
### 8 - django/forms/forms.py:

Start line: 279, End line: 287

```python
@html_safe
class BaseForm:

    def as_ul(self):
        "Return this form rendered as HTML <li>s -- excluding the <ul></ul>."
        return self._html_output(
            normal_row='<li%(html_class_attr)s>%(errors)s%(label)s %(field)s%(help_text)s</li>',
            error_row='<li>%s</li>',
            row_ender='</li>',
            help_text_html=' <span class="helptext">%s</span>',
            errors_on_separate_row=False,
        )
```
### 9 - django/forms/utils.py:

Start line: 139, End line: 146

```python
@html_safe
class ErrorList(UserList, list):

    def __reduce_ex__(self, *args, **kwargs):
        # The `list` reduce function returns an iterator as the fourth element
        # that is normally used for repopulating. Since we only inherit from
        # `list` for `isinstance` backward compatibility (Refs #17413) we
        # nullify this iterator as it would otherwise result in duplicate
        # entries. (Refs #23594)
        info = super(UserList, self).__reduce_ex__(*args, **kwargs)
        return info[:3] + (None, None)
```
### 10 - django/forms/utils.py:

Start line: 79, End line: 137

```python
@html_safe
class ErrorList(UserList, list):
    """
    A collection of errors that knows how to display itself in various formats.
    """
    def __init__(self, initlist=None, error_class=None):
        super().__init__(initlist)

        if error_class is None:
            self.error_class = 'errorlist'
        else:
            self.error_class = 'errorlist {}'.format(error_class)

    def as_data(self):
        return ValidationError(self.data).error_list

    def get_json_data(self, escape_html=False):
        errors = []
        for error in self.as_data():
            message = next(iter(error))
            errors.append({
                'message': escape(message) if escape_html else message,
                'code': error.code or '',
            })
        return errors

    def as_json(self, escape_html=False):
        return json.dumps(self.get_json_data(escape_html))

    def as_ul(self):
        if not self.data:
            return ''

        return format_html(
            '<ul class="{}">{}</ul>',
            self.error_class,
            format_html_join('', '<li>{}</li>', ((e,) for e in self))
        )

    def as_text(self):
        return '\n'.join('* %s' % e for e in self)

    def __str__(self):
        return self.as_ul()

    def __repr__(self):
        return repr(list(self))

    def __contains__(self, item):
        return item in list(self)

    def __eq__(self, other):
        return list(self) == other

    def __getitem__(self, i):
        error = self.data[i]
        if isinstance(error, ValidationError):
            return next(iter(error))
        return error
```
### 12 - django/forms/forms.py:

Start line: 133, End line: 150

```python
@html_safe
class BaseForm:

    def __str__(self):
        return self.as_table()

    def __repr__(self):
        if self._errors is None:
            is_valid = "Unknown"
        else:
            is_valid = self.is_bound and not self._errors
        return '<%(cls)s bound=%(bound)s, valid=%(valid)s, fields=(%(fields)s)>' % {
            'cls': self.__class__.__name__,
            'bound': self.is_bound,
            'valid': is_valid,
            'fields': ';'.join(self.fields),
        }

    def __iter__(self):
        for name in self.fields:
            yield self[name]
```
### 13 - django/forms/forms.py:

Start line: 168, End line: 190

```python
@html_safe
class BaseForm:

    @property
    def errors(self):
        """Return an ErrorDict for the data provided for the form."""
        if self._errors is None:
            self.full_clean()
        return self._errors

    def is_valid(self):
        """Return True if the form has no errors, or False otherwise."""
        return self.is_bound and not self.errors

    def add_prefix(self, field_name):
        """
        Return the field name with a prefix appended, if this Form has a
        prefix set.

        Subclasses may wish to override.
        """
        return '%s-%s' % (self.prefix, field_name) if self.prefix else field_name

    def add_initial_prefix(self, field_name):
        """Add an 'initial' prefix for checking dynamic initial values."""
        return 'initial-%s' % self.add_prefix(field_name)
```
### 16 - django/forms/forms.py:

Start line: 428, End line: 450

```python
@html_safe
class BaseForm:

    @cached_property
    def changed_data(self):
        data = []
        for name, field in self.fields.items():
            prefixed_name = self.add_prefix(name)
            data_value = field.widget.value_from_datadict(self.data, self.files, prefixed_name)
            if not field.show_hidden_initial:
                # Use the BoundField's initial as this is the value passed to
                # the widget.
                initial_value = self[name].initial
            else:
                initial_prefixed_name = self.add_initial_prefix(name)
                hidden_widget = field.hidden_widget()
                try:
                    initial_value = field.to_python(hidden_widget.value_from_datadict(
                        self.data, self.files, initial_prefixed_name))
                except ValidationError:
                    # Always assume data has changed if validation fails.
                    data.append(name)
                    continue
            if field.has_changed(initial_value, data_value):
                data.append(name)
        return data
```
### 17 - django/forms/forms.py:

Start line: 399, End line: 426

```python
@html_safe
class BaseForm:

    def _clean_form(self):
        try:
            cleaned_data = self.clean()
        except ValidationError as e:
            self.add_error(None, e)
        else:
            if cleaned_data is not None:
                self.cleaned_data = cleaned_data

    def _post_clean(self):
        """
        An internal hook for performing additional cleaning after form cleaning
        is complete. Used for model validation in model forms.
        """
        pass

    def clean(self):
        """
        Hook for doing any extra form-wide cleaning after Field.clean() has been
        called on every field. Any ValidationError raised by this method will
        not be associated with a particular field; it will have a special-case
        association with the field named '__all__'.
        """
        return self.cleaned_data

    def has_changed(self):
        """Return True if data differs from initial."""
        return bool(self.changed_data)
```
### 26 - django/forms/forms.py:

Start line: 53, End line: 109

```python
@html_safe
class BaseForm:
    """
    The main implementation of all the Form logic. Note that this class is
    different than Form. See the comments by the Form class for more info. Any
    improvements to the form API should be made to this class, not to the Form
    class.
    """
    default_renderer = None
    field_order = None
    prefix = None
    use_required_attribute = True

    def __init__(self, data=None, files=None, auto_id='id_%s', prefix=None,
                 initial=None, error_class=ErrorList, label_suffix=None,
                 empty_permitted=False, field_order=None, use_required_attribute=None, renderer=None):
        self.is_bound = data is not None or files is not None
        self.data = MultiValueDict() if data is None else data
        self.files = MultiValueDict() if files is None else files
        self.auto_id = auto_id
        if prefix is not None:
            self.prefix = prefix
        self.initial = initial or {}
        self.error_class = error_class
        # Translators: This is the default suffix added to form field labels
        self.label_suffix = label_suffix if label_suffix is not None else _(':')
        self.empty_permitted = empty_permitted
        self._errors = None  # Stores the errors after clean() has been called.

        # The base_fields class attribute is the *class-wide* definition of
        # fields. Because a particular *instance* of the class might want to
        # alter self.fields, we create self.fields here by copying base_fields.
        # Instances should always modify self.fields; they should not modify
        # self.base_fields.
        self.fields = copy.deepcopy(self.base_fields)
        self._bound_fields_cache = {}
        self.order_fields(self.field_order if field_order is None else field_order)

        if use_required_attribute is not None:
            self.use_required_attribute = use_required_attribute

        if self.empty_permitted and self.use_required_attribute:
            raise ValueError(
                'The empty_permitted and use_required_attribute arguments may '
                'not both be True.'
            )

        # Initialize form renderer. Use a global default if not specified
        # either as an argument or as self.default_renderer.
        if renderer is None:
            if self.default_renderer is None:
                renderer = get_default_renderer()
            else:
                renderer = self.default_renderer
                if isinstance(self.default_renderer, type):
                    renderer = renderer()
        self.renderer = renderer
```
### 27 - django/forms/forms.py:

Start line: 1, End line: 19

```python
"""
Form classes
"""

import copy

from django.core.exceptions import NON_FIELD_ERRORS, ValidationError
from django.forms.fields import Field, FileField
from django.forms.utils import ErrorDict, ErrorList
from django.forms.widgets import Media, MediaDefiningClass
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.html import conditional_escape, html_safe
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _

from .renderers import get_default_renderer

__all__ = ('BaseForm', 'Form')
```
### 36 - django/forms/forms.py:

Start line: 452, End line: 499

```python
@html_safe
class BaseForm:

    @property
    def media(self):
        """Return all media required to render the widgets on this form."""
        media = Media()
        for field in self.fields.values():
            media = media + field.widget.media
        return media

    def is_multipart(self):
        """
        Return True if the form needs to be multipart-encoded, i.e. it has
        FileInput, or False otherwise.
        """
        return any(field.widget.needs_multipart_form for field in self.fields.values())

    def hidden_fields(self):
        """
        Return a list of all the BoundField objects that are hidden fields.
        Useful for manual form layout in templates.
        """
        return [field for field in self if field.is_hidden]

    def visible_fields(self):
        """
        Return a list of BoundField objects that aren't hidden fields.
        The opposite of the hidden_fields() method.
        """
        return [field for field in self if not field.is_hidden]

    def get_initial_for_field(self, field, field_name):
        """
        Return initial data for field on form. Use initial data from the form
        or the field, in that order. Evaluate callable values.
        """
        value = self.initial.get(field_name, field.initial)
        if callable(value):
            value = value()
        return value


class Form(BaseForm, metaclass=DeclarativeFieldsMetaclass):
    "A collection of Fields, plus their associated data."
    # This is a separate class from BaseForm in order to abstract the way
    # self.fields is specified. This class (Form) is the one that does the
    # fancy metaclass stuff purely for the semantic sugar -- it allows one
    # to define a form using declarative syntax.
    # BaseForm itself has no way of designating self.fields.
```
### 58 - django/forms/forms.py:

Start line: 111, End line: 131

```python
@html_safe
class BaseForm:

    def order_fields(self, field_order):
        """
        Rearrange the fields according to field_order.

        field_order is a list of field names specifying the order. Append fields
        not included in the list in the default order for backward compatibility
        with subclasses not overriding field_order. If field_order is None,
        keep all fields in the order defined in the class. Ignore unknown
        fields in field_order to allow disabling fields in form subclasses
        without redefining ordering.
        """
        if field_order is None:
            return
        fields = {}
        for key in field_order:
            try:
                fields[key] = self.fields.pop(key)
            except KeyError:  # ignore unknown fields
                pass
        fields.update(self.fields)  # add remaining fields in original order
        self.fields = fields
```
### 124 - django/forms/forms.py:

Start line: 152, End line: 166

```python
@html_safe
class BaseForm:

    def __getitem__(self, name):
        """Return a BoundField with the given name."""
        try:
            field = self.fields[name]
        except KeyError:
            raise KeyError(
                "Key '%s' not found in '%s'. Choices are: %s." % (
                    name,
                    self.__class__.__name__,
                    ', '.join(sorted(self.fields)),
                )
            )
        if name not in self._bound_fields_cache:
            self._bound_fields_cache[name] = field.get_bound_field(self, name)
        return self._bound_fields_cache[name]
```
