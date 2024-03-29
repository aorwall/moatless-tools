# django__django-13741

| **django/django** | `d746f28949c009251a8741ba03d156964050717f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 162 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py
--- a/django/contrib/auth/forms.py
+++ b/django/contrib/auth/forms.py
@@ -56,16 +56,9 @@ class ReadOnlyPasswordHashField(forms.Field):
 
     def __init__(self, *args, **kwargs):
         kwargs.setdefault("required", False)
+        kwargs.setdefault('disabled', True)
         super().__init__(*args, **kwargs)
 
-    def bound_data(self, data, initial):
-        # Always return initial because the widget doesn't
-        # render an input field.
-        return initial
-
-    def has_changed(self, initial, data):
-        return False
-
 
 class UsernameField(forms.CharField):
     def to_python(self, value):
@@ -163,12 +156,6 @@ def __init__(self, *args, **kwargs):
         if user_permissions:
             user_permissions.queryset = user_permissions.queryset.select_related('content_type')
 
-    def clean_password(self):
-        # Regardless of what the user provides, return the initial value.
-        # This is done here, rather than on the field, because the
-        # field does not have access to the initial value
-        return self.initial.get('password')
-
 
 class AuthenticationForm(forms.Form):
     """

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/contrib/auth/forms.py | 59 | 66 | 1 | 1 | 162
| django/contrib/auth/forms.py | 166 | 171 | - | 1 | -


## Problem Statement

```
Set disabled prop on ReadOnlyPasswordHashField
Description
	
Currently the django.contrib.auth.forms.UserChangeForm defines a clean_password method that returns the initial password value to prevent (accidental) changes to the password value. It is also documented that custom forms for the User model need to define this method: ​https://docs.djangoproject.com/en/3.1/topics/auth/customizing/#a-full-example
A while ago the forms.Field base class gained the ​disabled argument to:
[disable] a form field using the disabled HTML attribute so that it won’t be editable by users. Even if a user tampers with the field’s value submitted to the server, it will be ignored in favor of the value from the form’s initial data.
It seems to me that this property could be set to True be default on the ReadOnlyPasswordHashField used to display the password hash. This way the clean_password is no longer necessary and the potential pitfall when using the ReadOnlyPasswordHashField without implementing clean_password is removed.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/contrib/auth/forms.py** | 54 | 79| 162 | 162 | 3206 | 
| 2 | **1 django/contrib/auth/forms.py** | 33 | 51| 161 | 323 | 3206 | 
| 3 | **1 django/contrib/auth/forms.py** | 142 | 170| 235 | 558 | 3206 | 
| 4 | **1 django/contrib/auth/forms.py** | 329 | 370| 289 | 847 | 3206 | 
| 5 | 2 django/contrib/admin/helpers.py | 204 | 237| 318 | 1165 | 6537 | 
| 6 | **2 django/contrib/auth/forms.py** | 373 | 400| 189 | 1354 | 6537 | 
| 7 | 3 django/contrib/auth/views.py | 330 | 362| 239 | 1593 | 9201 | 
| 8 | **3 django/contrib/auth/forms.py** | 403 | 454| 356 | 1949 | 9201 | 
| 9 | 4 django/contrib/auth/admin.py | 128 | 189| 465 | 2414 | 10927 | 
| 10 | **4 django/contrib/auth/forms.py** | 223 | 248| 173 | 2587 | 10927 | 
| 11 | 4 django/contrib/auth/views.py | 247 | 284| 348 | 2935 | 10927 | 
| 12 | 5 django/contrib/auth/hashers.py | 1 | 27| 187 | 3122 | 16065 | 
| 13 | 5 django/contrib/auth/hashers.py | 64 | 82| 186 | 3308 | 16065 | 
| 14 | 5 django/contrib/auth/hashers.py | 30 | 61| 246 | 3554 | 16065 | 
| 15 | 5 django/contrib/auth/hashers.py | 441 | 479| 331 | 3885 | 16065 | 
| 16 | 6 django/forms/forms.py | 427 | 449| 195 | 4080 | 20070 | 
| 17 | 6 django/contrib/auth/hashers.py | 233 | 244| 150 | 4230 | 20070 | 
| 18 | 7 django/contrib/admin/sites.py | 309 | 324| 158 | 4388 | 24219 | 
| 19 | 8 django/contrib/auth/password_validation.py | 54 | 88| 277 | 4665 | 25703 | 
| 20 | 8 django/contrib/auth/hashers.py | 359 | 408| 426 | 5091 | 25703 | 
| 21 | 9 django/forms/fields.py | 574 | 599| 243 | 5334 | 35054 | 
| 22 | 9 django/contrib/auth/views.py | 208 | 222| 133 | 5467 | 35054 | 
| 23 | 9 django/forms/fields.py | 175 | 207| 280 | 5747 | 35054 | 
| 24 | 9 django/forms/forms.py | 376 | 396| 210 | 5957 | 35054 | 
| 25 | 9 django/contrib/auth/views.py | 286 | 327| 314 | 6271 | 35054 | 
| 26 | 9 django/contrib/auth/views.py | 224 | 244| 163 | 6434 | 35054 | 
| 27 | **9 django/contrib/auth/forms.py** | 293 | 326| 288 | 6722 | 35054 | 
| 28 | 9 django/contrib/auth/hashers.py | 85 | 107| 167 | 6889 | 35054 | 
| 29 | 10 django/contrib/admin/options.py | 2089 | 2141| 451 | 7340 | 53647 | 
| 30 | 11 django/contrib/auth/base_user.py | 48 | 153| 694 | 8034 | 54646 | 
| 31 | 11 django/contrib/auth/admin.py | 40 | 99| 504 | 8538 | 54646 | 
| 32 | 11 django/contrib/auth/password_validation.py | 1 | 32| 206 | 8744 | 54646 | 
| 33 | 11 django/contrib/admin/helpers.py | 124 | 150| 220 | 8964 | 54646 | 
| 34 | 11 django/contrib/auth/password_validation.py | 135 | 157| 197 | 9161 | 54646 | 
| 35 | 11 django/contrib/auth/hashers.py | 247 | 299| 436 | 9597 | 54646 | 
| 36 | 11 django/contrib/auth/hashers.py | 342 | 357| 138 | 9735 | 54646 | 
| 37 | 11 django/contrib/admin/helpers.py | 35 | 69| 230 | 9965 | 54646 | 
| 38 | 12 django/contrib/auth/management/commands/changepassword.py | 34 | 76| 333 | 10298 | 55180 | 
| 39 | 12 django/contrib/auth/password_validation.py | 91 | 115| 189 | 10487 | 55180 | 
| 40 | 12 django/contrib/admin/options.py | 1541 | 1627| 760 | 11247 | 55180 | 
| 41 | 13 django/contrib/admin/forms.py | 1 | 31| 185 | 11432 | 55365 | 
| 42 | 13 django/contrib/auth/hashers.py | 154 | 189| 245 | 11677 | 55365 | 
| 43 | **13 django/contrib/auth/forms.py** | 82 | 139| 413 | 12090 | 55365 | 
| 44 | 13 django/contrib/auth/hashers.py | 429 | 439| 127 | 12217 | 55365 | 
| 45 | 13 django/contrib/auth/hashers.py | 499 | 534| 250 | 12467 | 55365 | 
| 46 | 13 django/contrib/auth/hashers.py | 482 | 496| 126 | 12593 | 55365 | 
| 47 | 13 django/contrib/admin/helpers.py | 93 | 121| 249 | 12842 | 55365 | 
| 48 | 14 django/db/models/fields/__init__.py | 983 | 999| 176 | 13018 | 73811 | 
| 49 | 14 django/contrib/auth/hashers.py | 575 | 614| 257 | 13275 | 73811 | 
| 50 | 14 django/contrib/auth/hashers.py | 617 | 658| 283 | 13558 | 73811 | 
| 51 | 14 django/contrib/auth/hashers.py | 661 | 706| 314 | 13872 | 73811 | 
| 52 | 14 django/forms/fields.py | 47 | 128| 773 | 14645 | 73811 | 
| 53 | 14 django/contrib/auth/hashers.py | 537 | 572| 252 | 14897 | 73811 | 
| 54 | 14 django/contrib/admin/options.py | 1628 | 1653| 291 | 15188 | 73811 | 
| 55 | 14 django/contrib/auth/password_validation.py | 118 | 133| 154 | 15342 | 73811 | 
| 56 | 15 django/forms/formsets.py | 1 | 25| 204 | 15546 | 77942 | 
| 57 | **15 django/contrib/auth/forms.py** | 173 | 221| 385 | 15931 | 77942 | 
| 58 | 15 django/contrib/auth/admin.py | 1 | 22| 188 | 16119 | 77942 | 
| 59 | 15 django/contrib/auth/hashers.py | 411 | 427| 127 | 16246 | 77942 | 
| 60 | 15 django/contrib/auth/hashers.py | 302 | 340| 300 | 16546 | 77942 | 
| 61 | 15 django/forms/fields.py | 130 | 173| 290 | 16836 | 77942 | 
| 62 | **15 django/contrib/auth/forms.py** | 251 | 273| 196 | 17032 | 77942 | 
| 63 | 15 django/forms/fields.py | 1084 | 1125| 353 | 17385 | 77942 | 
| 64 | 16 django/contrib/auth/middleware.py | 84 | 109| 192 | 17577 | 78936 | 
| 65 | 17 django/forms/boundfield.py | 1 | 34| 242 | 17819 | 81092 | 
| 66 | 17 django/contrib/auth/password_validation.py | 160 | 204| 351 | 18170 | 81092 | 
| 67 | 17 django/forms/forms.py | 398 | 425| 190 | 18360 | 81092 | 
| 68 | 18 django/forms/models.py | 318 | 357| 387 | 18747 | 92938 | 
| 69 | 18 django/contrib/admin/helpers.py | 383 | 399| 138 | 18885 | 92938 | 
| 70 | 19 django/contrib/auth/models.py | 149 | 163| 164 | 19049 | 96210 | 
| 71 | 19 django/contrib/admin/helpers.py | 372 | 381| 134 | 19183 | 96210 | 
| 72 | 19 django/contrib/admin/options.py | 431 | 474| 350 | 19533 | 96210 | 
| 73 | 19 django/contrib/auth/hashers.py | 191 | 231| 315 | 19848 | 96210 | 
| 74 | 19 django/contrib/admin/options.py | 1 | 97| 767 | 20615 | 96210 | 
| 75 | 19 django/contrib/admin/options.py | 1128 | 1173| 482 | 21097 | 96210 | 
| 76 | 19 django/contrib/admin/helpers.py | 72 | 90| 152 | 21249 | 96210 | 
| 77 | **19 django/contrib/auth/forms.py** | 275 | 291| 146 | 21395 | 96210 | 
| 78 | 19 django/contrib/auth/management/commands/changepassword.py | 1 | 32| 205 | 21600 | 96210 | 
| 79 | 20 django/contrib/auth/backends.py | 51 | 65| 143 | 21743 | 97972 | 
| 80 | 20 django/forms/forms.py | 353 | 374| 172 | 21915 | 97972 | 
| 81 | 20 django/db/models/fields/__init__.py | 1637 | 1658| 183 | 22098 | 97972 | 
| 82 | 20 django/forms/models.py | 391 | 419| 240 | 22338 | 97972 | 
| 83 | 20 django/contrib/auth/password_validation.py | 35 | 51| 114 | 22452 | 97972 | 
| 84 | 20 django/contrib/admin/helpers.py | 264 | 287| 235 | 22687 | 97972 | 
| 85 | 20 django/contrib/admin/helpers.py | 289 | 321| 282 | 22969 | 97972 | 
| 86 | 21 django/contrib/auth/tokens.py | 87 | 118| 307 | 23276 | 98965 | 
| 87 | 22 django/contrib/auth/management/commands/createsuperuser.py | 204 | 228| 204 | 23480 | 101028 | 
| 88 | 22 django/contrib/admin/options.py | 1953 | 1996| 403 | 23883 | 101028 | 
| 89 | 22 django/forms/fields.py | 535 | 551| 180 | 24063 | 101028 | 
| 90 | 23 django/db/models/fields/mixins.py | 31 | 57| 173 | 24236 | 101371 | 
| 91 | 23 django/db/models/fields/__init__.py | 1985 | 2012| 234 | 24470 | 101371 | 
| 92 | 24 django/forms/widgets.py | 448 | 466| 195 | 24665 | 109479 | 
| 93 | 24 django/db/models/fields/__init__.py | 1077 | 1092| 173 | 24838 | 109479 | 
| 94 | 24 django/forms/formsets.py | 385 | 417| 282 | 25120 | 109479 | 
| 95 | 24 django/forms/fields.py | 730 | 762| 241 | 25361 | 109479 | 
| 96 | 24 django/contrib/auth/hashers.py | 110 | 130| 139 | 25500 | 109479 | 
| 97 | 24 django/forms/models.py | 1270 | 1300| 242 | 25742 | 109479 | 
| 98 | 25 django/db/migrations/questioner.py | 162 | 185| 246 | 25988 | 111552 | 
| 99 | 26 django/contrib/auth/mixins.py | 107 | 129| 146 | 26134 | 112416 | 
| 100 | 27 django/contrib/admin/checks.py | 600 | 611| 128 | 26262 | 121553 | 
| 101 | 27 django/db/models/fields/__init__.py | 2451 | 2500| 311 | 26573 | 121553 | 
| 102 | 27 django/db/models/fields/__init__.py | 544 | 568| 225 | 26798 | 121553 | 
| 103 | 27 django/forms/models.py | 359 | 389| 233 | 27031 | 121553 | 
| 104 | 27 django/forms/models.py | 824 | 865| 449 | 27480 | 121553 | 
| 105 | 27 django/contrib/auth/tokens.py | 30 | 70| 319 | 27799 | 121553 | 
| 106 | 27 django/forms/forms.py | 166 | 188| 182 | 27981 | 121553 | 
| 107 | 27 django/contrib/admin/options.py | 1483 | 1506| 319 | 28300 | 121553 | 
| 108 | 27 django/forms/widgets.py | 303 | 339| 204 | 28504 | 121553 | 
| 109 | 27 django/contrib/admin/sites.py | 326 | 346| 182 | 28686 | 121553 | 
| 110 | 27 django/contrib/auth/admin.py | 101 | 126| 286 | 28972 | 121553 | 
| 111 | 27 django/forms/fields.py | 1219 | 1272| 355 | 29327 | 121553 | 
| 112 | 28 django/core/management/base.py | 120 | 155| 241 | 29568 | 126190 | 
| 113 | 28 django/forms/widgets.py | 768 | 791| 203 | 29771 | 126190 | 
| 114 | 28 django/forms/models.py | 1383 | 1410| 209 | 29980 | 126190 | 
| 115 | 28 django/forms/models.py | 203 | 213| 131 | 30111 | 126190 | 
| 116 | 28 django/contrib/admin/options.py | 132 | 187| 604 | 30715 | 126190 | 
| 117 | 28 django/contrib/auth/tokens.py | 72 | 85| 176 | 30891 | 126190 | 
| 118 | 28 django/contrib/admin/checks.py | 613 | 635| 155 | 31046 | 126190 | 
| 119 | 28 django/forms/formsets.py | 330 | 383| 456 | 31502 | 126190 | 
| 120 | 28 django/forms/models.py | 929 | 961| 353 | 31855 | 126190 | 
| 121 | 28 django/forms/forms.py | 109 | 129| 177 | 32032 | 126190 | 
| 122 | 28 django/db/models/fields/__init__.py | 2311 | 2331| 163 | 32195 | 126190 | 
| 123 | 28 django/contrib/auth/hashers.py | 133 | 151| 189 | 32384 | 126190 | 
| 124 | 28 django/forms/forms.py | 150 | 164| 125 | 32509 | 126190 | 
| 125 | 28 django/forms/fields.py | 1179 | 1216| 199 | 32708 | 126190 | 
| 126 | 29 django/db/models/options.py | 524 | 552| 231 | 32939 | 133296 | 
| 127 | 29 django/forms/models.py | 688 | 759| 732 | 33671 | 133296 | 
| 128 | 30 django/db/migrations/operations/fields.py | 85 | 95| 124 | 33795 | 136394 | 
| 129 | 31 django/db/models/fields/reverse_related.py | 160 | 178| 172 | 33967 | 138716 | 
| 130 | 31 django/forms/fields.py | 439 | 472| 225 | 34192 | 138716 | 
| 131 | 32 django/contrib/admin/views/main.py | 1 | 45| 324 | 34516 | 143112 | 
| 132 | 32 django/forms/fields.py | 703 | 727| 226 | 34742 | 143112 | 
| 133 | 33 django/db/models/fields/related.py | 984 | 995| 128 | 34870 | 156988 | 
| 134 | 33 django/db/models/fields/related.py | 1202 | 1233| 180 | 35050 | 156988 | 
| 135 | 33 django/contrib/auth/views.py | 189 | 205| 122 | 35172 | 156988 | 
| 136 | 34 django/contrib/postgres/forms/ranges.py | 31 | 78| 325 | 35497 | 157665 | 
| 137 | 34 django/contrib/auth/middleware.py | 112 | 123| 107 | 35604 | 157665 | 
| 138 | 34 django/contrib/auth/middleware.py | 46 | 82| 360 | 35964 | 157665 | 
| 139 | 34 django/forms/models.py | 288 | 316| 288 | 36252 | 157665 | 
| 140 | 34 django/db/models/fields/__init__.py | 2115 | 2148| 228 | 36480 | 157665 | 
| 141 | 34 django/forms/fields.py | 210 | 241| 274 | 36754 | 157665 | 
| 142 | 34 django/forms/forms.py | 190 | 266| 636 | 37390 | 157665 | 
| 143 | 34 django/forms/fields.py | 475 | 500| 174 | 37564 | 157665 | 
| 144 | 34 django/db/models/fields/reverse_related.py | 141 | 158| 161 | 37725 | 157665 | 
| 145 | 34 django/contrib/auth/views.py | 1 | 37| 278 | 38003 | 157665 | 
| 146 | 35 django/contrib/postgres/forms/hstore.py | 1 | 59| 339 | 38342 | 158004 | 
| 147 | 35 django/contrib/admin/options.py | 377 | 429| 504 | 38846 | 158004 | 
| 148 | 35 django/db/models/fields/__init__.py | 2082 | 2112| 252 | 39098 | 158004 | 
| 149 | 35 django/forms/models.py | 1329 | 1344| 131 | 39229 | 158004 | 
| 150 | 35 django/db/models/fields/__init__.py | 1111 | 1140| 218 | 39447 | 158004 | 
| 151 | 35 django/db/models/fields/__init__.py | 2398 | 2448| 339 | 39786 | 158004 | 
| 152 | 35 django/forms/widgets.py | 434 | 446| 122 | 39908 | 158004 | 
| 153 | 35 django/contrib/auth/models.py | 389 | 470| 484 | 40392 | 158004 | 
| 154 | 35 django/db/models/fields/related.py | 841 | 862| 169 | 40561 | 158004 | 
| 155 | 35 django/contrib/admin/options.py | 1462 | 1481| 133 | 40694 | 158004 | 
| 156 | 35 django/forms/fields.py | 387 | 408| 144 | 40838 | 158004 | 
| 157 | 35 django/db/migrations/operations/fields.py | 216 | 234| 185 | 41023 | 158004 | 
| 158 | 36 django/contrib/postgres/forms/array.py | 62 | 102| 248 | 41271 | 159598 | 
| 159 | 36 django/db/models/fields/__init__.py | 1704 | 1722| 134 | 41405 | 159598 | 
| 160 | 36 django/db/models/fields/__init__.py | 1661 | 1686| 206 | 41611 | 159598 | 
| 161 | 36 django/forms/fields.py | 854 | 893| 298 | 41909 | 159598 | 
| 162 | 36 django/db/models/fields/__init__.py | 956 | 981| 209 | 42118 | 159598 | 
| 163 | 36 django/contrib/admin/options.py | 500 | 513| 165 | 42283 | 159598 | 
| 164 | 36 django/forms/fields.py | 827 | 851| 177 | 42460 | 159598 | 
| 165 | 36 django/db/models/fields/__init__.py | 367 | 393| 199 | 42659 | 159598 | 
| 166 | 36 django/forms/models.py | 603 | 635| 270 | 42929 | 159598 | 
| 167 | 36 django/db/models/fields/related.py | 127 | 154| 201 | 43130 | 159598 | 
| 168 | 36 django/forms/fields.py | 765 | 824| 416 | 43546 | 159598 | 
| 169 | 36 django/forms/models.py | 1 | 29| 234 | 43780 | 159598 | 
| 170 | 37 django/db/backends/base/schema.py | 283 | 304| 173 | 43953 | 171944 | 
| 171 | 37 django/contrib/auth/base_user.py | 1 | 45| 304 | 44257 | 171944 | 
| 172 | 37 django/db/models/fields/__init__.py | 1688 | 1702| 144 | 44401 | 171944 | 
| 173 | 37 django/contrib/auth/models.py | 232 | 285| 352 | 44753 | 171944 | 
| 174 | 37 django/contrib/admin/options.py | 1508 | 1539| 295 | 45048 | 171944 | 
| 175 | 38 django/contrib/staticfiles/storage.py | 79 | 110| 343 | 45391 | 175471 | 
| 176 | 38 django/forms/fields.py | 553 | 572| 171 | 45562 | 175471 | 
| 177 | 38 django/forms/forms.py | 306 | 351| 438 | 46000 | 175471 | 
| 178 | 38 django/forms/forms.py | 131 | 148| 142 | 46142 | 175471 | 
| 179 | 38 django/db/models/fields/__init__.py | 1161 | 1199| 293 | 46435 | 175471 | 
| 180 | 38 django/db/models/fields/__init__.py | 1301 | 1342| 332 | 46767 | 175471 | 
| 181 | 38 django/contrib/admin/options.py | 476 | 498| 241 | 47008 | 175471 | 
| 182 | 38 django/contrib/auth/models.py | 35 | 79| 402 | 47410 | 175471 | 
| 183 | **38 django/contrib/auth/forms.py** | 1 | 30| 228 | 47638 | 175471 | 
| 184 | 38 django/contrib/auth/backends.py | 163 | 181| 146 | 47784 | 175471 | 
| 185 | 38 django/forms/widgets.py | 402 | 432| 193 | 47977 | 175471 | 
| 186 | 38 django/contrib/staticfiles/storage.py | 341 | 362| 230 | 48207 | 175471 | 
| 187 | 39 django/conf/__init__.py | 147 | 164| 149 | 48356 | 177658 | 
| 188 | 39 django/conf/__init__.py | 229 | 281| 407 | 48763 | 177658 | 
| 189 | 39 django/db/models/fields/related.py | 255 | 282| 269 | 49032 | 177658 | 
| 190 | 39 django/contrib/auth/models.py | 165 | 189| 192 | 49224 | 177658 | 
| 191 | 39 django/forms/fields.py | 1164 | 1176| 113 | 49337 | 177658 | 
| 192 | 39 django/contrib/staticfiles/storage.py | 251 | 321| 575 | 49912 | 177658 | 
| 193 | 40 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 50049 | 177795 | 
| 194 | 41 django/contrib/admin/widgets.py | 161 | 192| 243 | 50292 | 181589 | 
| 195 | 41 django/db/models/options.py | 256 | 288| 331 | 50623 | 181589 | 
| 196 | 41 django/forms/models.py | 784 | 822| 314 | 50937 | 181589 | 
| 197 | 41 django/contrib/admin/options.py | 1116 | 1126| 125 | 51062 | 181589 | 
| 198 | 41 django/forms/forms.py | 51 | 107| 525 | 51587 | 181589 | 


### Hint

```
Sounds good. Would you like to provide a patch?
Replying to Mariusz Felisiak: Sounds good. Would you like to provide a patch? I don't have the time to do a proper patch (with doc changes and additional tests). But I marked it as "Easy pickings" to entice others that are trying to get into contribution to Django ;-)
I'd like to work on this as my first contribution to Django :) I will provide a patch as soon as possible.
```

## Patch

```diff
diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py
--- a/django/contrib/auth/forms.py
+++ b/django/contrib/auth/forms.py
@@ -56,16 +56,9 @@ class ReadOnlyPasswordHashField(forms.Field):
 
     def __init__(self, *args, **kwargs):
         kwargs.setdefault("required", False)
+        kwargs.setdefault('disabled', True)
         super().__init__(*args, **kwargs)
 
-    def bound_data(self, data, initial):
-        # Always return initial because the widget doesn't
-        # render an input field.
-        return initial
-
-    def has_changed(self, initial, data):
-        return False
-
 
 class UsernameField(forms.CharField):
     def to_python(self, value):
@@ -163,12 +156,6 @@ def __init__(self, *args, **kwargs):
         if user_permissions:
             user_permissions.queryset = user_permissions.queryset.select_related('content_type')
 
-    def clean_password(self):
-        # Regardless of what the user provides, return the initial value.
-        # This is done here, rather than on the field, because the
-        # field does not have access to the initial value
-        return self.initial.get('password')
-
 
 class AuthenticationForm(forms.Form):
     """

```

## Test Patch

```diff
diff --git a/tests/auth_tests/test_forms.py b/tests/auth_tests/test_forms.py
--- a/tests/auth_tests/test_forms.py
+++ b/tests/auth_tests/test_forms.py
@@ -1022,6 +1022,7 @@ def test_render(self):
 
     def test_readonly_field_has_changed(self):
         field = ReadOnlyPasswordHashField()
+        self.assertIs(field.disabled, True)
         self.assertFalse(field.has_changed('aaa', 'bbb'))
 
 

```


## Code snippets

### 1 - django/contrib/auth/forms.py:

Start line: 54, End line: 79

```python
class ReadOnlyPasswordHashField(forms.Field):
    widget = ReadOnlyPasswordHashWidget

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("required", False)
        super().__init__(*args, **kwargs)

    def bound_data(self, data, initial):
        # Always return initial because the widget doesn't
        # render an input field.
        return initial

    def has_changed(self, initial, data):
        return False


class UsernameField(forms.CharField):
    def to_python(self, value):
        return unicodedata.normalize('NFKC', super().to_python(value))

    def widget_attrs(self, widget):
        return {
            **super().widget_attrs(widget),
            'autocapitalize': 'none',
            'autocomplete': 'username',
        }
```
### 2 - django/contrib/auth/forms.py:

Start line: 33, End line: 51

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
### 3 - django/contrib/auth/forms.py:

Start line: 142, End line: 170

```python
class UserChangeForm(forms.ModelForm):
    password = ReadOnlyPasswordHashField(
        label=_("Password"),
        help_text=_(
            'Raw passwords are not stored, so there is no way to see this '
            'user’s password, but you can change the password using '
            '<a href="{}">this form</a>.'
        ),
    )

    class Meta:
        model = User
        fields = '__all__'
        field_classes = {'username': UsernameField}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        password = self.fields.get('password')
        if password:
            password.help_text = password.help_text.format('../password/')
        user_permissions = self.fields.get('user_permissions')
        if user_permissions:
            user_permissions.queryset = user_permissions.queryset.select_related('content_type')

    def clean_password(self):
        # Regardless of what the user provides, return the initial value.
        # This is done here, rather than on the field, because the
        # field does not have access to the initial value
        return self.initial.get('password')
```
### 4 - django/contrib/auth/forms.py:

Start line: 329, End line: 370

```python
class SetPasswordForm(forms.Form):
    """
    A form that lets a user change set their password without entering the old
    password
    """
    error_messages = {
        'password_mismatch': _('The two password fields didn’t match.'),
    }
    new_password1 = forms.CharField(
        label=_("New password"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
        help_text=password_validation.password_validators_help_text_html(),
    )
    new_password2 = forms.CharField(
        label=_("New password confirmation"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
    )

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_new_password2(self):
        password1 = self.cleaned_data.get('new_password1')
        password2 = self.cleaned_data.get('new_password2')
        if password1 and password2:
            if password1 != password2:
                raise ValidationError(
                    self.error_messages['password_mismatch'],
                    code='password_mismatch',
                )
        password_validation.validate_password(password2, self.user)
        return password2

    def save(self, commit=True):
        password = self.cleaned_data["new_password1"]
        self.user.set_password(password)
        if commit:
            self.user.save()
        return self.user
```
### 5 - django/contrib/admin/helpers.py:

Start line: 204, End line: 237

```python
class AdminReadonlyField:

    def contents(self):
        from django.contrib.admin.templatetags.admin_list import _boolean_icon
        field, obj, model_admin = self.field['field'], self.form.instance, self.model_admin
        try:
            f, attr, value = lookup_field(field, obj, model_admin)
        except (AttributeError, ValueError, ObjectDoesNotExist):
            result_repr = self.empty_value_display
        else:
            if field in self.form.fields:
                widget = self.form[field].field.widget
                # This isn't elegant but suffices for contrib.auth's
                # ReadOnlyPasswordHashWidget.
                if getattr(widget, 'read_only', False):
                    return widget.render(field, value)
            if f is None:
                if getattr(attr, 'boolean', False):
                    result_repr = _boolean_icon(value)
                else:
                    if hasattr(value, "__html__"):
                        result_repr = value
                    else:
                        result_repr = linebreaksbr(value)
            else:
                if isinstance(f.remote_field, ManyToManyRel) and value is not None:
                    result_repr = ", ".join(map(str, value.all()))
                elif (
                    isinstance(f.remote_field, (ForeignObjectRel, OneToOneField)) and
                    value is not None
                ):
                    result_repr = self.get_admin_url(f.remote_field, value)
                else:
                    result_repr = display_for_field(value, f, self.empty_value_display)
                result_repr = linebreaksbr(result_repr)
        return conditional_escape(result_repr)
```
### 6 - django/contrib/auth/forms.py:

Start line: 373, End line: 400

```python
class PasswordChangeForm(SetPasswordForm):
    """
    A form that lets a user change their password by entering their old
    password.
    """
    error_messages = {
        **SetPasswordForm.error_messages,
        'password_incorrect': _("Your old password was entered incorrectly. Please enter it again."),
    }
    old_password = forms.CharField(
        label=_("Old password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'current-password', 'autofocus': True}),
    )

    field_order = ['old_password', 'new_password1', 'new_password2']

    def clean_old_password(self):
        """
        Validate that the old_password field is correct.
        """
        old_password = self.cleaned_data["old_password"]
        if not self.user.check_password(old_password):
            raise ValidationError(
                self.error_messages['password_incorrect'],
                code='password_incorrect',
            )
        return old_password
```
### 7 - django/contrib/auth/views.py:

Start line: 330, End line: 362

```python
class PasswordChangeView(PasswordContextMixin, FormView):
    form_class = PasswordChangeForm
    success_url = reverse_lazy('password_change_done')
    template_name = 'registration/password_change_form.html'
    title = _('Password change')

    @method_decorator(sensitive_post_parameters())
    @method_decorator(csrf_protect)
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def form_valid(self, form):
        form.save()
        # Updating the password logs out all other sessions for the user
        # except the current one.
        update_session_auth_hash(self.request, form.user)
        return super().form_valid(form)


class PasswordChangeDoneView(PasswordContextMixin, TemplateView):
    template_name = 'registration/password_change_done.html'
    title = _('Password change successful')

    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
```
### 8 - django/contrib/auth/forms.py:

Start line: 403, End line: 454

```python
class AdminPasswordChangeForm(forms.Form):
    """
    A form used to change the password of a user in the admin interface.
    """
    error_messages = {
        'password_mismatch': _('The two password fields didn’t match.'),
    }
    required_css_class = 'required'
    password1 = forms.CharField(
        label=_("Password"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password', 'autofocus': True}),
        strip=False,
        help_text=password_validation.password_validators_help_text_html(),
    )
    password2 = forms.CharField(
        label=_("Password (again)"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
        help_text=_("Enter the same password as before, for verification."),
    )

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise ValidationError(
                self.error_messages['password_mismatch'],
                code='password_mismatch',
            )
        password_validation.validate_password(password2, self.user)
        return password2

    def save(self, commit=True):
        """Save the new password."""
        password = self.cleaned_data["password1"]
        self.user.set_password(password)
        if commit:
            self.user.save()
        return self.user

    @property
    def changed_data(self):
        data = super().changed_data
        for name in self.fields:
            if name not in data:
                return []
        return ['password']
```
### 9 - django/contrib/auth/admin.py:

Start line: 128, End line: 189

```python
@admin.register(User)
class UserAdmin(admin.ModelAdmin):

    @sensitive_post_parameters_m
    def user_change_password(self, request, id, form_url=''):
        user = self.get_object(request, unquote(id))
        if not self.has_change_permission(request, user):
            raise PermissionDenied
        if user is None:
            raise Http404(_('%(name)s object with primary key %(key)r does not exist.') % {
                'name': self.model._meta.verbose_name,
                'key': escape(id),
            })
        if request.method == 'POST':
            form = self.change_password_form(user, request.POST)
            if form.is_valid():
                form.save()
                change_message = self.construct_change_message(request, form, None)
                self.log_change(request, user, change_message)
                msg = gettext('Password changed successfully.')
                messages.success(request, msg)
                update_session_auth_hash(request, form.user)
                return HttpResponseRedirect(
                    reverse(
                        '%s:%s_%s_change' % (
                            self.admin_site.name,
                            user._meta.app_label,
                            user._meta.model_name,
                        ),
                        args=(user.pk,),
                    )
                )
        else:
            form = self.change_password_form(user)

        fieldsets = [(None, {'fields': list(form.base_fields)})]
        adminForm = admin.helpers.AdminForm(form, fieldsets, {})

        context = {
            'title': _('Change password: %s') % escape(user.get_username()),
            'adminForm': adminForm,
            'form_url': form_url,
            'form': form,
            'is_popup': (IS_POPUP_VAR in request.POST or
                         IS_POPUP_VAR in request.GET),
            'add': True,
            'change': False,
            'has_delete_permission': False,
            'has_change_permission': True,
            'has_absolute_url': False,
            'opts': self.model._meta,
            'original': user,
            'save_as': False,
            'show_save': True,
            **self.admin_site.each_context(request),
        }

        request.current_app = self.admin_site.name

        return TemplateResponse(
            request,
            self.change_user_password_template or
            'admin/auth/user/change_password.html',
            context,
        )
```
### 10 - django/contrib/auth/forms.py:

Start line: 223, End line: 248

```python
class AuthenticationForm(forms.Form):

    def confirm_login_allowed(self, user):
        """
        Controls whether the given User may log in. This is a policy setting,
        independent of end-user authentication. This default behavior is to
        allow login by active users, and reject login by inactive users.

        If the given user cannot log in, this method should raise a
        ``ValidationError``.

        If the given user may log in, this method should return None.
        """
        if not user.is_active:
            raise ValidationError(
                self.error_messages['inactive'],
                code='inactive',
            )

    def get_user(self):
        return self.user_cache

    def get_invalid_login_error(self):
        return ValidationError(
            self.error_messages['invalid_login'],
            code='invalid_login',
            params={'username': self.username_field.verbose_name},
        )
```
### 27 - django/contrib/auth/forms.py:

Start line: 293, End line: 326

```python
class PasswordResetForm(forms.Form):

    def save(self, domain_override=None,
             subject_template_name='registration/password_reset_subject.txt',
             email_template_name='registration/password_reset_email.html',
             use_https=False, token_generator=default_token_generator,
             from_email=None, request=None, html_email_template_name=None,
             extra_email_context=None):
        """
        Generate a one-use only link for resetting password and send it to the
        user.
        """
        email = self.cleaned_data["email"]
        if not domain_override:
            current_site = get_current_site(request)
            site_name = current_site.name
            domain = current_site.domain
        else:
            site_name = domain = domain_override
        email_field_name = UserModel.get_email_field_name()
        for user in self.get_users(email):
            user_email = getattr(user, email_field_name)
            context = {
                'email': user_email,
                'domain': domain,
                'site_name': site_name,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'user': user,
                'token': token_generator.make_token(user),
                'protocol': 'https' if use_https else 'http',
                **(extra_email_context or {}),
            }
            self.send_mail(
                subject_template_name, email_template_name, context, from_email,
                user_email, html_email_template_name=html_email_template_name,
            )
```
### 43 - django/contrib/auth/forms.py:

Start line: 82, End line: 139

```python
class UserCreationForm(forms.ModelForm):
    """
    A form that creates a user, with no privileges, from the given username and
    password.
    """
    error_messages = {
        'password_mismatch': _('The two password fields didn’t match.'),
    }
    password1 = forms.CharField(
        label=_("Password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        help_text=password_validation.password_validators_help_text_html(),
    )
    password2 = forms.CharField(
        label=_("Password confirmation"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
        help_text=_("Enter the same password as before, for verification."),
    )

    class Meta:
        model = User
        fields = ("username",)
        field_classes = {'username': UsernameField}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._meta.model.USERNAME_FIELD in self.fields:
            self.fields[self._meta.model.USERNAME_FIELD].widget.attrs['autofocus'] = True

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise ValidationError(
                self.error_messages['password_mismatch'],
                code='password_mismatch',
            )
        return password2

    def _post_clean(self):
        super()._post_clean()
        # Validate the password after self.instance is updated with form data
        # by super().
        password = self.cleaned_data.get('password2')
        if password:
            try:
                password_validation.validate_password(password, self.instance)
            except ValidationError as error:
                self.add_error('password2', error)

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user
```
### 57 - django/contrib/auth/forms.py:

Start line: 173, End line: 221

```python
class AuthenticationForm(forms.Form):
    """
    Base class for authenticating users. Extend this to get a form that accepts
    username/password logins.
    """
    username = UsernameField(widget=forms.TextInput(attrs={'autofocus': True}))
    password = forms.CharField(
        label=_("Password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'current-password'}),
    )

    error_messages = {
        'invalid_login': _(
            "Please enter a correct %(username)s and password. Note that both "
            "fields may be case-sensitive."
        ),
        'inactive': _("This account is inactive."),
    }

    def __init__(self, request=None, *args, **kwargs):
        """
        The 'request' parameter is set for custom auth use by subclasses.
        The form data comes in via the standard 'data' kwarg.
        """
        self.request = request
        self.user_cache = None
        super().__init__(*args, **kwargs)

        # Set the max length and label for the "username" field.
        self.username_field = UserModel._meta.get_field(UserModel.USERNAME_FIELD)
        username_max_length = self.username_field.max_length or 254
        self.fields['username'].max_length = username_max_length
        self.fields['username'].widget.attrs['maxlength'] = username_max_length
        if self.fields['username'].label is None:
            self.fields['username'].label = capfirst(self.username_field.verbose_name)

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username is not None and password:
            self.user_cache = authenticate(self.request, username=username, password=password)
            if self.user_cache is None:
                raise self.get_invalid_login_error()
            else:
                self.confirm_login_allowed(self.user_cache)

        return self.cleaned_data
```
### 62 - django/contrib/auth/forms.py:

Start line: 251, End line: 273

```python
class PasswordResetForm(forms.Form):
    email = forms.EmailField(
        label=_("Email"),
        max_length=254,
        widget=forms.EmailInput(attrs={'autocomplete': 'email'})
    )

    def send_mail(self, subject_template_name, email_template_name,
                  context, from_email, to_email, html_email_template_name=None):
        """
        Send a django.core.mail.EmailMultiAlternatives to `to_email`.
        """
        subject = loader.render_to_string(subject_template_name, context)
        # Email subject *must not* contain newlines
        subject = ''.join(subject.splitlines())
        body = loader.render_to_string(email_template_name, context)

        email_message = EmailMultiAlternatives(subject, body, from_email, [to_email])
        if html_email_template_name is not None:
            html_email = loader.render_to_string(html_email_template_name, context)
            email_message.attach_alternative(html_email, 'text/html')

        email_message.send()
```
### 77 - django/contrib/auth/forms.py:

Start line: 275, End line: 291

```python
class PasswordResetForm(forms.Form):

    def get_users(self, email):
        """Given an email, return matching user(s) who should receive a reset.

        This allows subclasses to more easily customize the default policies
        that prevent inactive users and users with unusable passwords from
        resetting their password.
        """
        email_field_name = UserModel.get_email_field_name()
        active_users = UserModel._default_manager.filter(**{
            '%s__iexact' % email_field_name: email,
            'is_active': True,
        })
        return (
            u for u in active_users
            if u.has_usable_password() and
            _unicode_ci_compare(email, getattr(u, email_field_name))
        )
```
### 183 - django/contrib/auth/forms.py:

Start line: 1, End line: 30

```python
import unicodedata

from django import forms
from django.contrib.auth import (
    authenticate, get_user_model, password_validation,
)
from django.contrib.auth.hashers import (
    UNUSABLE_PASSWORD_PREFIX, identify_hasher,
)
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ValidationError
from django.core.mail import EmailMultiAlternatives
from django.template import loader
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.utils.text import capfirst
from django.utils.translation import gettext, gettext_lazy as _

UserModel = get_user_model()


def _unicode_ci_compare(s1, s2):
    """
    Perform case-insensitive comparison of two identifiers, using the
    recommended algorithm from Unicode Technical Report 36, section
    2.11.2(B)(2).
    """
    return unicodedata.normalize('NFKC', s1).casefold() == unicodedata.normalize('NFKC', s2).casefold()
```
