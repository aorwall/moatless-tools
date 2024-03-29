# django__django-13347

| **django/django** | `b9be11d44265308863e4e8cfb458cd3605091452` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 267 |
| **Any found context length** | 267 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/views/debug.py b/django/views/debug.py
--- a/django/views/debug.py
+++ b/django/views/debug.py
@@ -91,18 +91,19 @@ def cleanse_setting(self, key, value):
         value is a dictionary, recursively cleanse the keys in that dictionary.
         """
         try:
-            if self.hidden_settings.search(key):
-                cleansed = self.cleansed_substitute
-            elif isinstance(value, dict):
-                cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
-            elif isinstance(value, list):
-                cleansed = [self.cleanse_setting('', v) for v in value]
-            elif isinstance(value, tuple):
-                cleansed = tuple([self.cleanse_setting('', v) for v in value])
-            else:
-                cleansed = value
+            is_sensitive = self.hidden_settings.search(key)
         except TypeError:
-            # If the key isn't regex-able, just return as-is.
+            is_sensitive = False
+
+        if is_sensitive:
+            cleansed = self.cleansed_substitute
+        elif isinstance(value, dict):
+            cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
+        elif isinstance(value, list):
+            cleansed = [self.cleanse_setting('', v) for v in value]
+        elif isinstance(value, tuple):
+            cleansed = tuple([self.cleanse_setting('', v) for v in value])
+        else:
             cleansed = value
 
         if callable(cleansed):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/views/debug.py | 94 | 105 | 1 | 1 | 267


## Problem Statement

```
SafeExceptionReporterFilter does not recurse into dictionaries with non-string keys
Description
	
SafeExceptionReporterFilter has provisions for recursively cleaning settings by descending into lists / tuples / dictionaries - which is great! However, recursing on dictionaries only works if the keys of the dictionary are strings.
For instance it will fail to sanitize the following example:
SOME_SETTING = {1: {'login': 'cooper', 'password': 'secret'}}
The reason for this is that cleanse_setting starts by trying to apply a the hidden_settings regex to the key before attempting to recurse into the value:
​https://github.com/django/django/blob/0b0658111cba538b91072b9a133fd5545f3f46d1/django/views/debug.py#L94

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/views/debug.py** | 80 | 111| 267 | 267 | 4462 | 
| 2 | **1 django/views/debug.py** | 180 | 192| 143 | 410 | 4462 | 
| 3 | **1 django/views/debug.py** | 113 | 139| 216 | 626 | 4462 | 
| 4 | **1 django/views/debug.py** | 141 | 153| 148 | 774 | 4462 | 
| 5 | **1 django/views/debug.py** | 64 | 77| 115 | 889 | 4462 | 
| 6 | **1 django/views/debug.py** | 155 | 178| 177 | 1066 | 4462 | 
| 7 | **1 django/views/debug.py** | 194 | 242| 467 | 1533 | 4462 | 
| 8 | **1 django/views/debug.py** | 396 | 470| 631 | 2164 | 4462 | 
| 9 | **1 django/views/debug.py** | 1 | 47| 296 | 2460 | 4462 | 
| 10 | **1 django/views/debug.py** | 245 | 329| 761 | 3221 | 4462 | 
| 11 | 2 django/core/checks/security/base.py | 1 | 84| 743 | 3964 | 6354 | 
| 12 | 3 django/core/checks/templates.py | 1 | 36| 259 | 4223 | 6614 | 
| 13 | 4 django/core/checks/translation.py | 1 | 65| 445 | 4668 | 7059 | 
| 14 | 5 django/conf/__init__.py | 167 | 226| 546 | 5214 | 9246 | 
| 15 | 6 django/conf/global_settings.py | 151 | 266| 859 | 6073 | 15006 | 
| 16 | 7 django/contrib/auth/__init__.py | 1 | 58| 393 | 6466 | 16605 | 
| 17 | 7 django/core/checks/security/base.py | 189 | 220| 223 | 6689 | 16605 | 
| 18 | 8 django/contrib/auth/password_validation.py | 1 | 32| 206 | 6895 | 18091 | 
| 19 | 8 django/conf/global_settings.py | 628 | 652| 180 | 7075 | 18091 | 
| 20 | 8 django/conf/__init__.py | 1 | 46| 298 | 7373 | 18091 | 
| 21 | 9 django/core/checks/messages.py | 53 | 76| 161 | 7534 | 18664 | 
| 22 | 9 django/core/checks/security/base.py | 223 | 244| 184 | 7718 | 18664 | 
| 23 | 10 django/core/checks/urls.py | 1 | 27| 142 | 7860 | 19365 | 
| 24 | 11 django/core/checks/security/csrf.py | 1 | 41| 299 | 8159 | 19664 | 
| 25 | 12 django/contrib/staticfiles/finders.py | 70 | 93| 202 | 8361 | 21705 | 
| 26 | 13 django/core/checks/model_checks.py | 1 | 86| 665 | 9026 | 23490 | 
| 27 | 13 django/conf/global_settings.py | 492 | 627| 812 | 9838 | 23490 | 
| 28 | 14 django/contrib/auth/hashers.py | 85 | 107| 167 | 10005 | 28628 | 
| 29 | 15 django/utils/regex_helper.py | 1 | 38| 250 | 10255 | 31269 | 
| 30 | 16 django/core/checks/__init__.py | 1 | 26| 270 | 10525 | 31539 | 
| 31 | 17 django/db/models/fields/related.py | 1 | 34| 246 | 10771 | 45415 | 
| 32 | 18 django/template/defaultfilters.py | 31 | 53| 191 | 10962 | 51501 | 
| 33 | 19 django/template/base.py | 572 | 607| 361 | 11323 | 59379 | 
| 34 | 20 django/views/csrf.py | 1 | 13| 132 | 11455 | 60923 | 
| 35 | 20 django/template/defaultfilters.py | 308 | 322| 111 | 11566 | 60923 | 
| 36 | 21 django/utils/log.py | 137 | 159| 125 | 11691 | 62565 | 
| 37 | 22 django/db/migrations/serializer.py | 108 | 118| 114 | 11805 | 65236 | 
| 38 | 23 django/db/utils.py | 1 | 49| 154 | 11959 | 67382 | 
| 39 | 23 django/core/checks/urls.py | 71 | 111| 264 | 12223 | 67382 | 
| 40 | 23 django/template/defaultfilters.py | 1 | 28| 207 | 12430 | 67382 | 
| 41 | 24 django/core/handlers/exception.py | 54 | 115| 499 | 12929 | 68386 | 
| 42 | 25 django/db/models/options.py | 1 | 34| 285 | 13214 | 75492 | 
| 43 | 26 django/urls/exceptions.py | 1 | 10| 0 | 13214 | 75517 | 
| 44 | 27 django/views/decorators/debug.py | 1 | 44| 274 | 13488 | 76106 | 
| 45 | 28 django/contrib/admin/options.py | 377 | 429| 504 | 13992 | 94675 | 
| 46 | 28 django/core/checks/security/base.py | 86 | 186| 742 | 14734 | 94675 | 
| 47 | **28 django/views/debug.py** | 362 | 394| 328 | 15062 | 94675 | 
| 48 | 28 django/template/base.py | 668 | 703| 272 | 15334 | 94675 | 
| 49 | 28 django/conf/__init__.py | 72 | 94| 221 | 15555 | 94675 | 
| 50 | 29 django/db/models/base.py | 1 | 50| 328 | 15883 | 111319 | 
| 51 | 29 django/core/checks/model_checks.py | 178 | 211| 332 | 16215 | 111319 | 
| 52 | 30 django/views/defaults.py | 1 | 24| 149 | 16364 | 112361 | 
| 53 | 30 django/template/defaultfilters.py | 424 | 459| 233 | 16597 | 112361 | 
| 54 | 30 django/conf/global_settings.py | 1 | 50| 366 | 16963 | 112361 | 
| 55 | 31 django/conf/urls/__init__.py | 1 | 23| 152 | 17115 | 112513 | 
| 56 | 31 django/db/migrations/serializer.py | 249 | 270| 166 | 17281 | 112513 | 
| 57 | 32 django/urls/__init__.py | 1 | 24| 239 | 17520 | 112752 | 
| 58 | 32 django/db/models/base.py | 1237 | 1260| 172 | 17692 | 112752 | 
| 59 | 32 django/core/checks/messages.py | 1 | 24| 156 | 17848 | 112752 | 
| 60 | 33 django/contrib/sessions/exceptions.py | 1 | 12| 0 | 17848 | 112803 | 
| 61 | 34 django/core/checks/security/sessions.py | 1 | 98| 572 | 18420 | 113376 | 
| 62 | 35 django/utils/autoreload.py | 48 | 76| 156 | 18576 | 118248 | 
| 63 | 36 django/http/__init__.py | 1 | 22| 197 | 18773 | 118445 | 
| 64 | 37 django/contrib/staticfiles/utils.py | 42 | 64| 205 | 18978 | 118894 | 
| 65 | 38 django/contrib/contenttypes/checks.py | 1 | 21| 122 | 19100 | 119143 | 
| 66 | 38 django/views/decorators/debug.py | 77 | 92| 132 | 19232 | 119143 | 
| 67 | 38 django/core/handlers/exception.py | 118 | 143| 167 | 19399 | 119143 | 
| 68 | 39 django/db/models/deletion.py | 1 | 76| 566 | 19965 | 122969 | 
| 69 | 40 django/contrib/auth/checks.py | 1 | 99| 694 | 20659 | 124441 | 
| 70 | 41 django/db/models/__init__.py | 1 | 53| 619 | 21278 | 125060 | 
| 71 | 41 django/template/base.py | 199 | 275| 503 | 21781 | 125060 | 
| 72 | 42 django/contrib/admin/exceptions.py | 1 | 12| 0 | 21781 | 125127 | 
| 73 | 42 django/conf/__init__.py | 229 | 281| 407 | 22188 | 125127 | 
| 74 | 42 django/template/defaultfilters.py | 325 | 409| 499 | 22687 | 125127 | 
| 75 | 42 django/conf/global_settings.py | 267 | 349| 800 | 23487 | 125127 | 
| 76 | 43 django/contrib/admin/checks.py | 370 | 386| 134 | 23621 | 134264 | 
| 77 | 43 django/template/defaultfilters.py | 56 | 91| 203 | 23824 | 134264 | 
| 78 | 43 django/views/csrf.py | 15 | 100| 835 | 24659 | 134264 | 
| 79 | 44 django/contrib/admin/views/main.py | 123 | 212| 861 | 25520 | 138660 | 
| 80 | 45 django/utils/translation/__init__.py | 1 | 37| 297 | 25817 | 141000 | 
| 81 | 45 django/conf/global_settings.py | 401 | 491| 794 | 26611 | 141000 | 
| 82 | 45 django/contrib/admin/checks.py | 1035 | 1062| 194 | 26805 | 141000 | 
| 83 | 45 django/conf/__init__.py | 96 | 110| 129 | 26934 | 141000 | 
| 84 | 45 django/contrib/auth/hashers.py | 1 | 27| 187 | 27121 | 141000 | 
| 85 | 46 django/shortcuts.py | 81 | 99| 200 | 27321 | 142098 | 
| 86 | 46 django/utils/regex_helper.py | 78 | 190| 962 | 28283 | 142098 | 
| 87 | 46 django/contrib/staticfiles/finders.py | 1 | 17| 110 | 28393 | 142098 | 
| 88 | 47 django/core/validators.py | 19 | 61| 336 | 28729 | 146644 | 
| 89 | 48 django/utils/formats.py | 237 | 260| 202 | 28931 | 148750 | 
| 90 | 48 django/utils/log.py | 1 | 75| 484 | 29415 | 148750 | 
| 91 | 49 django/core/exceptions.py | 102 | 214| 770 | 30185 | 149939 | 
| 92 | 50 django/utils/safestring.py | 40 | 64| 159 | 30344 | 150325 | 
| 93 | 50 django/core/validators.py | 1 | 16| 127 | 30471 | 150325 | 
| 94 | 51 django/http/response.py | 1 | 26| 157 | 30628 | 154694 | 
| 95 | 51 django/utils/autoreload.py | 1 | 45| 217 | 30845 | 154694 | 
| 96 | 52 django/utils/html.py | 352 | 379| 212 | 31057 | 157796 | 
| 97 | 52 django/db/migrations/serializer.py | 235 | 246| 133 | 31190 | 157796 | 
| 98 | 53 django/urls/resolvers.py | 145 | 196| 400 | 31590 | 163321 | 
| 99 | 53 django/contrib/auth/checks.py | 102 | 208| 776 | 32366 | 163321 | 
| 100 | 53 django/views/defaults.py | 100 | 119| 149 | 32515 | 163321 | 
| 101 | 54 django/core/checks/caches.py | 1 | 17| 0 | 32515 | 163421 | 
| 102 | 54 django/conf/__init__.py | 147 | 164| 149 | 32664 | 163421 | 
| 103 | 54 django/conf/global_settings.py | 350 | 400| 785 | 33449 | 163421 | 
| 104 | 55 django/utils/translation/trans_real.py | 1 | 58| 503 | 33952 | 167766 | 
| 105 | 56 django/core/serializers/xml_serializer.py | 386 | 402| 126 | 34078 | 171278 | 
| 106 | 57 django/forms/utils.py | 44 | 76| 241 | 34319 | 172535 | 
| 107 | 57 django/core/checks/messages.py | 26 | 50| 259 | 34578 | 172535 | 
| 108 | 57 django/utils/formats.py | 1 | 57| 377 | 34955 | 172535 | 
| 109 | 57 django/urls/resolvers.py | 271 | 287| 160 | 35115 | 172535 | 
| 110 | 58 django/contrib/sites/models.py | 1 | 22| 130 | 35245 | 173323 | 
| 111 | 58 django/core/validators.py | 328 | 358| 227 | 35472 | 173323 | 
| 112 | 58 django/forms/utils.py | 144 | 151| 132 | 35604 | 173323 | 
| 113 | 59 django/template/context.py | 1 | 24| 128 | 35732 | 175204 | 
| 114 | 60 django/db/__init__.py | 1 | 18| 141 | 35873 | 175597 | 
| 115 | 60 django/template/base.py | 1 | 94| 779 | 36652 | 175597 | 
| 116 | 60 django/core/checks/model_checks.py | 155 | 176| 263 | 36915 | 175597 | 
| 117 | 61 docs/_ext/djangodocs.py | 26 | 71| 398 | 37313 | 178753 | 
| 118 | 61 django/utils/log.py | 78 | 134| 463 | 37776 | 178753 | 
| 119 | 62 django/contrib/admin/sites.py | 1 | 29| 175 | 37951 | 182963 | 
| 120 | 62 django/template/defaultfilters.py | 492 | 585| 527 | 38478 | 182963 | 
| 121 | 62 django/template/defaultfilters.py | 412 | 421| 111 | 38589 | 182963 | 
| 122 | 63 django/forms/formsets.py | 1 | 25| 203 | 38792 | 186931 | 
| 123 | 64 django/contrib/auth/views.py | 1 | 37| 278 | 39070 | 189595 | 
| 124 | 65 django/db/backends/mysql/validation.py | 1 | 31| 239 | 39309 | 190115 | 
| 125 | 66 django/db/backends/sqlite3/introspection.py | 23 | 54| 296 | 39605 | 193964 | 
| 126 | 66 django/core/serializers/xml_serializer.py | 349 | 383| 327 | 39932 | 193964 | 
| 127 | 67 django/utils/http.py | 418 | 480| 318 | 40250 | 198135 | 
| 128 | 67 django/utils/regex_helper.py | 41 | 76| 375 | 40625 | 198135 | 
| 129 | 67 django/contrib/admin/checks.py | 809 | 858| 443 | 41068 | 198135 | 


## Patch

```diff
diff --git a/django/views/debug.py b/django/views/debug.py
--- a/django/views/debug.py
+++ b/django/views/debug.py
@@ -91,18 +91,19 @@ def cleanse_setting(self, key, value):
         value is a dictionary, recursively cleanse the keys in that dictionary.
         """
         try:
-            if self.hidden_settings.search(key):
-                cleansed = self.cleansed_substitute
-            elif isinstance(value, dict):
-                cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
-            elif isinstance(value, list):
-                cleansed = [self.cleanse_setting('', v) for v in value]
-            elif isinstance(value, tuple):
-                cleansed = tuple([self.cleanse_setting('', v) for v in value])
-            else:
-                cleansed = value
+            is_sensitive = self.hidden_settings.search(key)
         except TypeError:
-            # If the key isn't regex-able, just return as-is.
+            is_sensitive = False
+
+        if is_sensitive:
+            cleansed = self.cleansed_substitute
+        elif isinstance(value, dict):
+            cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
+        elif isinstance(value, list):
+            cleansed = [self.cleanse_setting('', v) for v in value]
+        elif isinstance(value, tuple):
+            cleansed = tuple([self.cleanse_setting('', v) for v in value])
+        else:
             cleansed = value
 
         if callable(cleansed):

```

## Test Patch

```diff
diff --git a/tests/view_tests/tests/test_debug.py b/tests/view_tests/tests/test_debug.py
--- a/tests/view_tests/tests/test_debug.py
+++ b/tests/view_tests/tests/test_debug.py
@@ -1274,6 +1274,19 @@ def test_cleanse_setting_recurses_in_dictionary(self):
             {'login': 'cooper', 'password': reporter_filter.cleansed_substitute},
         )
 
+    def test_cleanse_setting_recurses_in_dictionary_with_non_string_key(self):
+        reporter_filter = SafeExceptionReporterFilter()
+        initial = {('localhost', 8000): {'login': 'cooper', 'password': 'secret'}}
+        self.assertEqual(
+            reporter_filter.cleanse_setting('SETTING_NAME', initial),
+            {
+                ('localhost', 8000): {
+                    'login': 'cooper',
+                    'password': reporter_filter.cleansed_substitute,
+                },
+            },
+        )
+
     def test_cleanse_setting_recurses_in_list_tuples(self):
         reporter_filter = SafeExceptionReporterFilter()
         initial = [

```


## Code snippets

### 1 - django/views/debug.py:

Start line: 80, End line: 111

```python
class SafeExceptionReporterFilter:
    """
    Use annotations made by the sensitive_post_parameters and
    sensitive_variables decorators to filter out sensitive information.
    """
    cleansed_substitute = '********************'
    hidden_settings = _lazy_re_compile('API|TOKEN|KEY|SECRET|PASS|SIGNATURE', flags=re.I)

    def cleanse_setting(self, key, value):
        """
        Cleanse an individual setting key/value of sensitive content. If the
        value is a dictionary, recursively cleanse the keys in that dictionary.
        """
        try:
            if self.hidden_settings.search(key):
                cleansed = self.cleansed_substitute
            elif isinstance(value, dict):
                cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
            elif isinstance(value, list):
                cleansed = [self.cleanse_setting('', v) for v in value]
            elif isinstance(value, tuple):
                cleansed = tuple([self.cleanse_setting('', v) for v in value])
            else:
                cleansed = value
        except TypeError:
            # If the key isn't regex-able, just return as-is.
            cleansed = value

        if callable(cleansed):
            cleansed = CallableSettingWrapper(cleansed)

        return cleansed
```
### 2 - django/views/debug.py:

Start line: 180, End line: 192

```python
class SafeExceptionReporterFilter:

    def cleanse_special_types(self, request, value):
        try:
            # If value is lazy or a complex object of another kind, this check
            # might raise an exception. isinstance checks that lazy
            # MultiValueDicts will have a return value.
            is_multivalue_dict = isinstance(value, MultiValueDict)
        except Exception as e:
            return '{!r} while evaluating {!r}'.format(e, value)

        if is_multivalue_dict:
            # Cleanse MultiValueDicts (request.POST is the one we usually care about)
            value = self.get_cleansed_multivaluedict(request, value)
        return value
```
### 3 - django/views/debug.py:

Start line: 113, End line: 139

```python
class SafeExceptionReporterFilter:

    def get_safe_settings(self):
        """
        Return a dictionary of the settings module with values of sensitive
        settings replaced with stars (*********).
        """
        settings_dict = {}
        for k in dir(settings):
            if k.isupper():
                settings_dict[k] = self.cleanse_setting(k, getattr(settings, k))
        return settings_dict

    def get_safe_request_meta(self, request):
        """
        Return a dictionary of request.META with sensitive values redacted.
        """
        if not hasattr(request, 'META'):
            return {}
        return {k: self.cleanse_setting(k, v) for k, v in request.META.items()}

    def is_active(self, request):
        """
        This filter is to add safety in production environments (i.e. DEBUG
        is False). If DEBUG is True then your site is not safe anyway.
        This hook is provided as a convenience to easily activate or
        deactivate the filter on a per request basis.
        """
        return settings.DEBUG is False
```
### 4 - django/views/debug.py:

Start line: 141, End line: 153

```python
class SafeExceptionReporterFilter:

    def get_cleansed_multivaluedict(self, request, multivaluedict):
        """
        Replace the keys in a MultiValueDict marked as sensitive with stars.
        This mitigates leaking sensitive POST parameters if something like
        request.POST['nonexistent_key'] throws an exception (#21098).
        """
        sensitive_post_parameters = getattr(request, 'sensitive_post_parameters', [])
        if self.is_active(request) and sensitive_post_parameters:
            multivaluedict = multivaluedict.copy()
            for param in sensitive_post_parameters:
                if param in multivaluedict:
                    multivaluedict[param] = self.cleansed_substitute
        return multivaluedict
```
### 5 - django/views/debug.py:

Start line: 64, End line: 77

```python
@functools.lru_cache()
def get_default_exception_reporter_filter():
    # Instantiate the default filter for the first time and cache it.
    return import_string(settings.DEFAULT_EXCEPTION_REPORTER_FILTER)()


def get_exception_reporter_filter(request):
    default_filter = get_default_exception_reporter_filter()
    return getattr(request, 'exception_reporter_filter', default_filter)


def get_exception_reporter_class(request):
    default_exception_reporter_class = import_string(settings.DEFAULT_EXCEPTION_REPORTER)
    return getattr(request, 'exception_reporter_class', default_exception_reporter_class)
```
### 6 - django/views/debug.py:

Start line: 155, End line: 178

```python
class SafeExceptionReporterFilter:

    def get_post_parameters(self, request):
        """
        Replace the values of POST parameters marked as sensitive with
        stars (*********).
        """
        if request is None:
            return {}
        else:
            sensitive_post_parameters = getattr(request, 'sensitive_post_parameters', [])
            if self.is_active(request) and sensitive_post_parameters:
                cleansed = request.POST.copy()
                if sensitive_post_parameters == '__ALL__':
                    # Cleanse all parameters.
                    for k in cleansed:
                        cleansed[k] = self.cleansed_substitute
                    return cleansed
                else:
                    # Cleanse only the specified parameters.
                    for param in sensitive_post_parameters:
                        if param in cleansed:
                            cleansed[param] = self.cleansed_substitute
                    return cleansed
            else:
                return request.POST
```
### 7 - django/views/debug.py:

Start line: 194, End line: 242

```python
class SafeExceptionReporterFilter:

    def get_traceback_frame_variables(self, request, tb_frame):
        """
        Replace the values of variables marked as sensitive with
        stars (*********).
        """
        # Loop through the frame's callers to see if the sensitive_variables
        # decorator was used.
        current_frame = tb_frame.f_back
        sensitive_variables = None
        while current_frame is not None:
            if (current_frame.f_code.co_name == 'sensitive_variables_wrapper' and
                    'sensitive_variables_wrapper' in current_frame.f_locals):
                # The sensitive_variables decorator was used, so we take note
                # of the sensitive variables' names.
                wrapper = current_frame.f_locals['sensitive_variables_wrapper']
                sensitive_variables = getattr(wrapper, 'sensitive_variables', None)
                break
            current_frame = current_frame.f_back

        cleansed = {}
        if self.is_active(request) and sensitive_variables:
            if sensitive_variables == '__ALL__':
                # Cleanse all variables
                for name in tb_frame.f_locals:
                    cleansed[name] = self.cleansed_substitute
            else:
                # Cleanse specified variables
                for name, value in tb_frame.f_locals.items():
                    if name in sensitive_variables:
                        value = self.cleansed_substitute
                    else:
                        value = self.cleanse_special_types(request, value)
                    cleansed[name] = value
        else:
            # Potentially cleanse the request and any MultiValueDicts if they
            # are one of the frame variables.
            for name, value in tb_frame.f_locals.items():
                cleansed[name] = self.cleanse_special_types(request, value)

        if (tb_frame.f_code.co_name == 'sensitive_variables_wrapper' and
                'sensitive_variables_wrapper' in tb_frame.f_locals):
            # For good measure, obfuscate the decorated function's arguments in
            # the sensitive_variables decorator's frame, in case the variables
            # associated with those arguments were meant to be obfuscated from
            # the decorated function's frame.
            cleansed['func_args'] = self.cleansed_substitute
            cleansed['func_kwargs'] = self.cleansed_substitute

        return cleansed.items()
```
### 8 - django/views/debug.py:

Start line: 396, End line: 470

```python
class ExceptionReporter:

    def get_traceback_frames(self):
        def explicit_or_implicit_cause(exc_value):
            explicit = getattr(exc_value, '__cause__', None)
            suppress_context = getattr(exc_value, '__suppress_context__', None)
            implicit = getattr(exc_value, '__context__', None)
            return explicit or (None if suppress_context else implicit)

        # Get the exception and all its causes
        exceptions = []
        exc_value = self.exc_value
        while exc_value:
            exceptions.append(exc_value)
            exc_value = explicit_or_implicit_cause(exc_value)
            if exc_value in exceptions:
                warnings.warn(
                    "Cycle in the exception chain detected: exception '%s' "
                    "encountered again." % exc_value,
                    ExceptionCycleWarning,
                )
                # Avoid infinite loop if there's a cyclic reference (#29393).
                break

        frames = []
        # No exceptions were supplied to ExceptionReporter
        if not exceptions:
            return frames

        # In case there's just one exception, take the traceback from self.tb
        exc_value = exceptions.pop()
        tb = self.tb if not exceptions else exc_value.__traceback__

        while tb is not None:
            # Support for __traceback_hide__ which is used by a few libraries
            # to hide internal frames.
            if tb.tb_frame.f_locals.get('__traceback_hide__'):
                tb = tb.tb_next
                continue
            filename = tb.tb_frame.f_code.co_filename
            function = tb.tb_frame.f_code.co_name
            lineno = tb.tb_lineno - 1
            loader = tb.tb_frame.f_globals.get('__loader__')
            module_name = tb.tb_frame.f_globals.get('__name__') or ''
            pre_context_lineno, pre_context, context_line, post_context = self._get_lines_from_file(
                filename, lineno, 7, loader, module_name,
            )
            if pre_context_lineno is None:
                pre_context_lineno = lineno
                pre_context = []
                context_line = '<source code not available>'
                post_context = []
            frames.append({
                'exc_cause': explicit_or_implicit_cause(exc_value),
                'exc_cause_explicit': getattr(exc_value, '__cause__', True),
                'tb': tb,
                'type': 'django' if module_name.startswith('django.') else 'user',
                'filename': filename,
                'function': function,
                'lineno': lineno + 1,
                'vars': self.filter.get_traceback_frame_variables(self.request, tb.tb_frame),
                'id': id(tb),
                'pre_context': pre_context,
                'context_line': context_line,
                'post_context': post_context,
                'pre_context_lineno': pre_context_lineno + 1,
            })

            # If the traceback for current exception is consumed, try the
            # other exception.
            if not tb.tb_next and exceptions:
                exc_value = exceptions.pop()
                tb = exc_value.__traceback__
            else:
                tb = tb.tb_next

        return frames
```
### 9 - django/views/debug.py:

Start line: 1, End line: 47

```python
import functools
import re
import sys
import types
import warnings
from pathlib import Path

from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import get_docs_version

# Minimal Django templates engine to render the error templates
# regardless of the project's TEMPLATES setting. Templates are
# read directly from the filesystem so that the error handler
# works even if the template loader is broken.
DEBUG_ENGINE = Engine(
    debug=True,
    libraries={'i18n': 'django.templatetags.i18n'},
)

CURRENT_DIR = Path(__file__).parent


class ExceptionCycleWarning(UserWarning):
    pass


class CallableSettingWrapper:
    """
    Object to wrap callable appearing in settings.
    * Not to call in the debug page (#21345).
    * Not to break the debug page if the callable forbidding to set attributes
      (#23070).
    """
    def __init__(self, callable_setting):
        self._wrapped = callable_setting

    def __repr__(self):
        return repr(self._wrapped)
```
### 10 - django/views/debug.py:

Start line: 245, End line: 329

```python
class ExceptionReporter:
    """Organize and coordinate reporting on exceptions."""
    def __init__(self, request, exc_type, exc_value, tb, is_email=False):
        self.request = request
        self.filter = get_exception_reporter_filter(self.request)
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.tb = tb
        self.is_email = is_email

        self.template_info = getattr(self.exc_value, 'template_debug', None)
        self.template_does_not_exist = False
        self.postmortem = None

    def get_traceback_data(self):
        """Return a dictionary containing traceback information."""
        if self.exc_type and issubclass(self.exc_type, TemplateDoesNotExist):
            self.template_does_not_exist = True
            self.postmortem = self.exc_value.chain or [self.exc_value]

        frames = self.get_traceback_frames()
        for i, frame in enumerate(frames):
            if 'vars' in frame:
                frame_vars = []
                for k, v in frame['vars']:
                    v = pprint(v)
                    # Trim large blobs of data
                    if len(v) > 4096:
                        v = '%s… <trimmed %d bytes string>' % (v[0:4096], len(v))
                    frame_vars.append((k, v))
                frame['vars'] = frame_vars
            frames[i] = frame

        unicode_hint = ''
        if self.exc_type and issubclass(self.exc_type, UnicodeError):
            start = getattr(self.exc_value, 'start', None)
            end = getattr(self.exc_value, 'end', None)
            if start is not None and end is not None:
                unicode_str = self.exc_value.args[1]
                unicode_hint = force_str(
                    unicode_str[max(start - 5, 0):min(end + 5, len(unicode_str))],
                    'ascii', errors='replace'
                )
        from django import get_version

        if self.request is None:
            user_str = None
        else:
            try:
                user_str = str(self.request.user)
            except Exception:
                # request.user may raise OperationalError if the database is
                # unavailable, for example.
                user_str = '[unable to retrieve the current user]'

        c = {
            'is_email': self.is_email,
            'unicode_hint': unicode_hint,
            'frames': frames,
            'request': self.request,
            'request_meta': self.filter.get_safe_request_meta(self.request),
            'user_str': user_str,
            'filtered_POST_items': list(self.filter.get_post_parameters(self.request).items()),
            'settings': self.filter.get_safe_settings(),
            'sys_executable': sys.executable,
            'sys_version_info': '%d.%d.%d' % sys.version_info[0:3],
            'server_time': timezone.now(),
            'django_version_info': get_version(),
            'sys_path': sys.path,
            'template_info': self.template_info,
            'template_does_not_exist': self.template_does_not_exist,
            'postmortem': self.postmortem,
        }
        if self.request is not None:
            c['request_GET_items'] = self.request.GET.items()
            c['request_FILES_items'] = self.request.FILES.items()
            c['request_COOKIES_items'] = self.request.COOKIES.items()
        # Check whether exception info is available
        if self.exc_type:
            c['exception_type'] = self.exc_type.__name__
        if self.exc_value:
            c['exception_value'] = str(self.exc_value)
        if frames:
            c['lastframe'] = frames[-1]
        return c
```
### 47 - django/views/debug.py:

Start line: 362, End line: 394

```python
class ExceptionReporter:

    def _get_lines_from_file(self, filename, lineno, context_lines, loader=None, module_name=None):
        """
        Return context_lines before and after lineno from file.
        Return (pre_context_lineno, pre_context, context_line, post_context).
        """
        source = self._get_source(filename, loader, module_name)
        if source is None:
            return None, [], None, []

        # If we just read the source from a file, or if the loader did not
        # apply tokenize.detect_encoding to decode the source into a
        # string, then we should do that ourselves.
        if isinstance(source[0], bytes):
            encoding = 'ascii'
            for line in source[:2]:
                # File coding may be specified. Match pattern from PEP-263
                # (https://www.python.org/dev/peps/pep-0263/)
                match = re.search(br'coding[:=]\s*([-\w.]+)', line)
                if match:
                    encoding = match[1].decode('ascii')
                    break
            source = [str(sline, encoding, 'replace') for sline in source]

        lower_bound = max(0, lineno - context_lines)
        upper_bound = lineno + context_lines

        try:
            pre_context = source[lower_bound:lineno]
            context_line = source[lineno]
            post_context = source[lineno + 1:upper_bound]
        except IndexError:
            return None, [], None, []
        return lower_bound, pre_context, context_line, post_context
```
