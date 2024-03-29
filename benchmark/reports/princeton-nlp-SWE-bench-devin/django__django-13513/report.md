# django__django-13513

| **django/django** | `6599608c4d0befdcb820ddccce55f183f247ae4f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 631 |
| **Any found context length** | 631 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/views/debug.py b/django/views/debug.py
--- a/django/views/debug.py
+++ b/django/views/debug.py
@@ -394,19 +394,19 @@ def _get_lines_from_file(self, filename, lineno, context_lines, loader=None, mod
             return None, [], None, []
         return lower_bound, pre_context, context_line, post_context
 
-    def get_traceback_frames(self):
-        def explicit_or_implicit_cause(exc_value):
-            explicit = getattr(exc_value, '__cause__', None)
-            suppress_context = getattr(exc_value, '__suppress_context__', None)
-            implicit = getattr(exc_value, '__context__', None)
-            return explicit or (None if suppress_context else implicit)
+    def _get_explicit_or_implicit_cause(self, exc_value):
+        explicit = getattr(exc_value, '__cause__', None)
+        suppress_context = getattr(exc_value, '__suppress_context__', None)
+        implicit = getattr(exc_value, '__context__', None)
+        return explicit or (None if suppress_context else implicit)
 
+    def get_traceback_frames(self):
         # Get the exception and all its causes
         exceptions = []
         exc_value = self.exc_value
         while exc_value:
             exceptions.append(exc_value)
-            exc_value = explicit_or_implicit_cause(exc_value)
+            exc_value = self._get_explicit_or_implicit_cause(exc_value)
             if exc_value in exceptions:
                 warnings.warn(
                     "Cycle in the exception chain detected: exception '%s' "
@@ -424,6 +424,17 @@ def explicit_or_implicit_cause(exc_value):
         # In case there's just one exception, take the traceback from self.tb
         exc_value = exceptions.pop()
         tb = self.tb if not exceptions else exc_value.__traceback__
+        frames.extend(self.get_exception_traceback_frames(exc_value, tb))
+        while exceptions:
+            exc_value = exceptions.pop()
+            frames.extend(
+                self.get_exception_traceback_frames(exc_value, exc_value.__traceback__),
+            )
+        return frames
+
+    def get_exception_traceback_frames(self, exc_value, tb):
+        exc_cause = self._get_explicit_or_implicit_cause(exc_value)
+        exc_cause_explicit = getattr(exc_value, '__cause__', True)
 
         while tb is not None:
             # Support for __traceback_hide__ which is used by a few libraries
@@ -444,9 +455,9 @@ def explicit_or_implicit_cause(exc_value):
                 pre_context = []
                 context_line = '<source code not available>'
                 post_context = []
-            frames.append({
-                'exc_cause': explicit_or_implicit_cause(exc_value),
-                'exc_cause_explicit': getattr(exc_value, '__cause__', True),
+            yield {
+                'exc_cause': exc_cause,
+                'exc_cause_explicit': exc_cause_explicit,
                 'tb': tb,
                 'type': 'django' if module_name.startswith('django.') else 'user',
                 'filename': filename,
@@ -458,17 +469,8 @@ def explicit_or_implicit_cause(exc_value):
                 'context_line': context_line,
                 'post_context': post_context,
                 'pre_context_lineno': pre_context_lineno + 1,
-            })
-
-            # If the traceback for current exception is consumed, try the
-            # other exception.
-            if not tb.tb_next and exceptions:
-                exc_value = exceptions.pop()
-                tb = exc_value.__traceback__
-            else:
-                tb = tb.tb_next
-
-        return frames
+            }
+            tb = tb.tb_next
 
 
 def technical_404_response(request, exception):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/views/debug.py | 397 | 409 | 1 | 1 | 631
| django/views/debug.py | 427 | 427 | 1 | 1 | 631
| django/views/debug.py | 447 | 449 | 1 | 1 | 631
| django/views/debug.py | 461 | 471 | 1 | 1 | 631


## Problem Statement

```
debug error view doesn't respect exc.__suppress_context__ (PEP 415)
Description
	
Consider the following view that raises an exception:
class TestView(View):
	def get(self, request, *args, **kwargs):
		try:
			raise RuntimeError('my error')
		except Exception as exc:
			raise ValueError('my new error') from None
Even though the raise is from None, unlike the traceback Python shows, the debug error view still shows the RuntimeError.
This is because the explicit_or_implicit_cause() function inside get_traceback_frames() doesn't respect exc.__suppress_context__, which was introduced in Python 3.3's PEP 415:
​https://github.com/django/django/blob/38a21f2d9ed4f556af934498ec6a242f6a20418a/django/views/debug.py#L392
def get_traceback_frames(self):
	def explicit_or_implicit_cause(exc_value):
		explicit = getattr(exc_value, '__cause__', None)
		implicit = getattr(exc_value, '__context__', None)
		return explicit or implicit
Instead, it should be something more like (simplifying also for Python 3):
def explicit_or_implicit_cause(exc_value):
	return (
		exc_value.__cause__ or
		(None if exc_value.__suppress_context__ else
			exc_value.__context__)
	)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 django/views/debug.py** | 397 | 471| 631 | 631 | 4482 | 
| 2 | **1 django/views/debug.py** | 1 | 47| 296 | 927 | 4482 | 
| 3 | **1 django/views/debug.py** | 363 | 395| 328 | 1255 | 4482 | 
| 4 | 2 django/core/handlers/base.py | 292 | 316| 218 | 1473 | 7072 | 
| 5 | **2 django/views/debug.py** | 181 | 193| 143 | 1616 | 7072 | 
| 6 | 3 django/views/generic/__init__.py | 1 | 23| 189 | 1805 | 7262 | 
| 7 | **3 django/views/debug.py** | 332 | 361| 267 | 2072 | 7262 | 
| 8 | **3 django/views/debug.py** | 195 | 243| 467 | 2539 | 7262 | 
| 9 | 4 django/views/decorators/debug.py | 77 | 92| 132 | 2671 | 7851 | 
| 10 | 5 django/template/backends/jinja2.py | 91 | 126| 248 | 2919 | 8673 | 
| 11 | 6 django/core/handlers/exception.py | 54 | 126| 588 | 3507 | 9768 | 
| 12 | **6 django/views/debug.py** | 80 | 112| 258 | 3765 | 9768 | 
| 13 | 7 django/template/context_processors.py | 35 | 50| 126 | 3891 | 10257 | 
| 14 | **7 django/views/debug.py** | 246 | 330| 761 | 4652 | 10257 | 
| 15 | 8 django/views/decorators/clickjacking.py | 22 | 54| 238 | 4890 | 10633 | 
| 16 | **8 django/views/debug.py** | 64 | 77| 115 | 5005 | 10633 | 
| 17 | 9 django/views/csrf.py | 101 | 155| 577 | 5582 | 12177 | 
| 18 | 9 django/views/decorators/clickjacking.py | 1 | 19| 138 | 5720 | 12177 | 
| 19 | 10 django/views/defaults.py | 122 | 149| 214 | 5934 | 13219 | 
| 20 | **10 django/views/debug.py** | 50 | 61| 135 | 6069 | 13219 | 
| 21 | 10 django/views/csrf.py | 15 | 100| 835 | 6904 | 13219 | 
| 22 | 11 django/template/backends/django.py | 79 | 111| 225 | 7129 | 14075 | 
| 23 | 12 django/contrib/admindocs/views.py | 250 | 315| 573 | 7702 | 17371 | 
| 24 | 13 django/template/context.py | 1 | 24| 128 | 7830 | 19252 | 
| 25 | 13 django/views/decorators/debug.py | 1 | 44| 274 | 8104 | 19252 | 
| 26 | 14 django/views/generic/base.py | 1 | 27| 148 | 8252 | 20853 | 
| 27 | 14 django/contrib/admindocs/views.py | 156 | 180| 234 | 8486 | 20853 | 
| 28 | 14 django/views/defaults.py | 100 | 119| 149 | 8635 | 20853 | 
| 29 | 14 django/contrib/admindocs/views.py | 1 | 30| 223 | 8858 | 20853 | 
| 30 | **14 django/views/debug.py** | 142 | 154| 148 | 9006 | 20853 | 
| 31 | **14 django/views/debug.py** | 474 | 543| 573 | 9579 | 20853 | 
| 32 | 15 django/db/utils.py | 52 | 98| 312 | 9891 | 22999 | 
| 33 | 16 django/core/checks/messages.py | 53 | 76| 161 | 10052 | 23572 | 
| 34 | 17 django/contrib/admin/options.py | 1912 | 1950| 330 | 10382 | 42159 | 
| 35 | 17 django/views/decorators/debug.py | 47 | 75| 199 | 10581 | 42159 | 
| 36 | 17 django/views/csrf.py | 1 | 13| 132 | 10713 | 42159 | 
| 37 | 18 django/views/generic/list.py | 113 | 136| 205 | 10918 | 43731 | 
| 38 | 18 django/contrib/admindocs/views.py | 136 | 154| 187 | 11105 | 43731 | 
| 39 | **18 django/views/debug.py** | 156 | 179| 177 | 11282 | 43731 | 
| 40 | 19 django/template/base.py | 199 | 275| 503 | 11785 | 51609 | 
| 41 | 19 django/contrib/admindocs/views.py | 318 | 347| 201 | 11986 | 51609 | 
| 42 | 20 django/utils/autoreload.py | 48 | 76| 156 | 12142 | 56481 | 
| 43 | 20 django/views/defaults.py | 1 | 24| 149 | 12291 | 56481 | 
| 44 | 21 django/core/checks/model_checks.py | 178 | 211| 332 | 12623 | 58266 | 
| 45 | 22 django/conf/locale/ar_DZ/formats.py | 5 | 30| 252 | 12875 | 58563 | 
| 46 | 22 django/contrib/admindocs/views.py | 118 | 133| 154 | 13029 | 58563 | 
| 47 | 23 django/contrib/auth/urls.py | 1 | 21| 224 | 13253 | 58787 | 
| 48 | 23 django/views/defaults.py | 27 | 76| 401 | 13654 | 58787 | 
| 49 | 24 django/db/backends/utils.py | 92 | 129| 297 | 13951 | 60653 | 
| 50 | 25 django/views/decorators/csrf.py | 1 | 57| 460 | 14411 | 61113 | 
| 51 | 26 django/contrib/messages/views.py | 1 | 19| 0 | 14411 | 61209 | 
| 52 | 26 django/contrib/admindocs/views.py | 183 | 249| 584 | 14995 | 61209 | 
| 53 | 27 django/views/generic/edit.py | 1 | 67| 479 | 15474 | 62925 | 
| 54 | 28 django/views/decorators/cache.py | 27 | 48| 153 | 15627 | 63288 | 
| 55 | 28 django/core/checks/model_checks.py | 155 | 176| 263 | 15890 | 63288 | 
| 56 | 28 django/core/handlers/exception.py | 129 | 154| 167 | 16057 | 63288 | 
| 57 | 28 django/template/base.py | 507 | 535| 244 | 16301 | 63288 | 
| 58 | 29 django/middleware/csrf.py | 205 | 330| 1222 | 17523 | 66174 | 
| 59 | 30 django/contrib/admin/exceptions.py | 1 | 12| 0 | 17523 | 66241 | 
| 60 | 30 django/contrib/admin/options.py | 1251 | 1324| 659 | 18182 | 66241 | 
| 61 | 30 django/views/generic/base.py | 188 | 219| 247 | 18429 | 66241 | 
| 62 | 31 django/contrib/auth/management/__init__.py | 89 | 149| 441 | 18870 | 67351 | 
| 63 | 32 django/contrib/sessions/exceptions.py | 1 | 17| 0 | 18870 | 67422 | 
| 64 | 32 django/views/generic/base.py | 30 | 46| 136 | 19006 | 67422 | 
| 65 | 32 django/core/checks/model_checks.py | 129 | 153| 268 | 19274 | 67422 | 
| 66 | 32 django/contrib/admindocs/views.py | 87 | 115| 285 | 19559 | 67422 | 
| 67 | 33 django/views/decorators/http.py | 1 | 52| 350 | 19909 | 68376 | 
| 68 | 34 django/template/response.py | 45 | 58| 120 | 20029 | 69467 | 
| 69 | 35 django/core/management/base.py | 21 | 42| 165 | 20194 | 74104 | 
| 70 | 36 django/views/i18n.py | 286 | 303| 158 | 20352 | 76641 | 
| 71 | 37 django/contrib/auth/views.py | 330 | 362| 239 | 20591 | 79305 | 
| 72 | 37 django/template/context.py | 133 | 167| 288 | 20879 | 79305 | 
| 73 | 37 django/views/defaults.py | 79 | 97| 129 | 21008 | 79305 | 
| 74 | 38 django/template/defaulttags.py | 633 | 680| 331 | 21339 | 90448 | 
| 75 | 38 django/contrib/admindocs/views.py | 33 | 53| 159 | 21498 | 90448 | 
| 76 | 38 django/contrib/auth/views.py | 286 | 327| 314 | 21812 | 90448 | 
| 77 | 39 django/db/backends/base/features.py | 322 | 345| 199 | 22011 | 93238 | 
| 78 | 40 django/db/models/sql/query.py | 703 | 738| 389 | 22400 | 115728 | 
| 79 | 40 django/template/context.py | 233 | 260| 199 | 22599 | 115728 | 
| 80 | 41 docs/_ext/djangodocs.py | 74 | 106| 255 | 22854 | 118884 | 
| 81 | 42 django/forms/utils.py | 144 | 151| 132 | 22986 | 120141 | 
| 82 | 42 django/template/base.py | 668 | 703| 272 | 23258 | 120141 | 
| 83 | 43 django/urls/utils.py | 1 | 63| 460 | 23718 | 120601 | 
| 84 | 44 django/contrib/auth/decorators.py | 1 | 35| 313 | 24031 | 121188 | 
| 85 | 44 django/contrib/auth/decorators.py | 38 | 74| 273 | 24304 | 121188 | 
| 86 | 45 django/views/__init__.py | 1 | 4| 0 | 24304 | 121203 | 
| 87 | 45 django/contrib/auth/views.py | 247 | 284| 348 | 24652 | 121203 | 
| 88 | 46 django/db/backends/oracle/base.py | 60 | 99| 332 | 24984 | 126433 | 
| 89 | 46 django/contrib/admin/options.py | 1420 | 1459| 309 | 25293 | 126433 | 
| 90 | 47 django/contrib/staticfiles/storage.py | 112 | 147| 307 | 25600 | 129960 | 
| 91 | 47 django/views/i18n.py | 209 | 219| 137 | 25737 | 129960 | 
| 92 | 47 django/contrib/admin/options.py | 1627 | 1652| 291 | 26028 | 129960 | 
| 93 | 48 django/contrib/auth/admin.py | 101 | 126| 286 | 26314 | 131686 | 
| 94 | 48 django/template/response.py | 1 | 43| 389 | 26703 | 131686 | 
| 95 | 48 django/contrib/auth/views.py | 208 | 222| 133 | 26836 | 131686 | 
| 96 | 49 django/urls/exceptions.py | 1 | 10| 0 | 26836 | 131711 | 
| 97 | 50 django/core/exceptions.py | 107 | 218| 752 | 27588 | 132900 | 
| 98 | 50 django/contrib/admin/options.py | 515 | 532| 200 | 27788 | 132900 | 
| 99 | 51 django/core/checks/security/base.py | 1 | 84| 743 | 28531 | 134792 | 
| 100 | 51 django/contrib/admin/options.py | 1686 | 1757| 653 | 29184 | 134792 | 
| 101 | 51 django/contrib/auth/views.py | 224 | 244| 163 | 29347 | 134792 | 
| 102 | 52 django/utils/log.py | 137 | 159| 125 | 29472 | 136434 | 
| 103 | 52 django/contrib/admin/options.py | 1174 | 1249| 664 | 30136 | 136434 | 
| 104 | 53 django/contrib/admindocs/middleware.py | 1 | 29| 235 | 30371 | 136670 | 
| 105 | 53 django/contrib/admindocs/views.py | 56 | 84| 285 | 30656 | 136670 | 
| 106 | 53 django/views/generic/edit.py | 70 | 101| 269 | 30925 | 136670 | 
| 107 | 53 django/core/handlers/base.py | 210 | 273| 480 | 31405 | 136670 | 
| 108 | 54 django/template/engine.py | 1 | 53| 388 | 31793 | 137980 | 
| 109 | 54 django/contrib/auth/admin.py | 191 | 206| 185 | 31978 | 137980 | 
| 110 | 54 django/core/exceptions.py | 1 | 104| 436 | 32414 | 137980 | 
| 111 | 55 django/urls/conf.py | 57 | 78| 162 | 32576 | 138589 | 
| 112 | 56 django/db/backends/oracle/features.py | 1 | 94| 736 | 33312 | 139326 | 
| 113 | 57 django/contrib/admin/views/decorators.py | 1 | 19| 135 | 33447 | 139462 | 
| 114 | 58 django/forms/models.py | 310 | 349| 387 | 33834 | 151236 | 
| 115 | 58 django/utils/autoreload.py | 79 | 95| 161 | 33995 | 151236 | 
| 116 | 59 django/db/models/fields/related.py | 1202 | 1233| 180 | 34175 | 165112 | 
| 117 | 60 django/utils/deprecation.py | 79 | 120| 343 | 34518 | 166178 | 
| 118 | 61 django/contrib/gis/views.py | 1 | 21| 155 | 34673 | 166333 | 
| 119 | 61 django/views/decorators/http.py | 77 | 122| 349 | 35022 | 166333 | 
| 120 | **61 django/views/debug.py** | 114 | 140| 216 | 35238 | 166333 | 
| 121 | 62 django/core/checks/security/csrf.py | 1 | 41| 299 | 35537 | 166632 | 
| 122 | 63 django/utils/decorators.py | 114 | 152| 316 | 35853 | 168031 | 
| 123 | 64 django/contrib/admin/checks.py | 1035 | 1062| 194 | 36047 | 177168 | 
| 124 | 65 django/contrib/auth/mixins.py | 107 | 129| 146 | 36193 | 178032 | 
| 125 | 66 django/urls/resolvers.py | 610 | 620| 120 | 36313 | 183631 | 
| 126 | 67 django/contrib/admin/sites.py | 221 | 240| 221 | 36534 | 187829 | 
| 127 | 68 django/contrib/postgres/constraints.py | 157 | 167| 132 | 36666 | 189260 | 
| 128 | 68 django/views/decorators/http.py | 55 | 76| 272 | 36938 | 189260 | 
| 129 | 68 django/views/i18n.py | 264 | 284| 189 | 37127 | 189260 | 
| 130 | 68 django/contrib/admin/options.py | 1540 | 1626| 760 | 37887 | 189260 | 
| 131 | 69 django/contrib/flatpages/views.py | 1 | 45| 399 | 38286 | 189850 | 
| 132 | 69 django/template/base.py | 816 | 881| 540 | 38826 | 189850 | 
| 133 | 70 django/db/backends/mysql/base.py | 52 | 95| 361 | 39187 | 193280 | 
| 134 | 70 django/core/checks/messages.py | 26 | 50| 259 | 39446 | 193280 | 
| 135 | 70 django/core/handlers/base.py | 158 | 208| 379 | 39825 | 193280 | 
| 136 | 71 django/middleware/common.py | 76 | 97| 227 | 40052 | 194792 | 
| 137 | 72 django/db/backends/mysql/validation.py | 1 | 31| 239 | 40291 | 195312 | 


### Hint

```
Here is a related (but different) issue about the traceback shown by the debug error view ("debug error view shows no traceback if exc.traceback is None for innermost exception"): https://code.djangoproject.com/ticket/31672
Thanks Chris. Would you like to prepare a patch?
PR: ​https://github.com/django/django/pull/13176
Fixed in f36862b69c3325da8ba6892a6057bbd9470efd70.
```

## Patch

```diff
diff --git a/django/views/debug.py b/django/views/debug.py
--- a/django/views/debug.py
+++ b/django/views/debug.py
@@ -394,19 +394,19 @@ def _get_lines_from_file(self, filename, lineno, context_lines, loader=None, mod
             return None, [], None, []
         return lower_bound, pre_context, context_line, post_context
 
-    def get_traceback_frames(self):
-        def explicit_or_implicit_cause(exc_value):
-            explicit = getattr(exc_value, '__cause__', None)
-            suppress_context = getattr(exc_value, '__suppress_context__', None)
-            implicit = getattr(exc_value, '__context__', None)
-            return explicit or (None if suppress_context else implicit)
+    def _get_explicit_or_implicit_cause(self, exc_value):
+        explicit = getattr(exc_value, '__cause__', None)
+        suppress_context = getattr(exc_value, '__suppress_context__', None)
+        implicit = getattr(exc_value, '__context__', None)
+        return explicit or (None if suppress_context else implicit)
 
+    def get_traceback_frames(self):
         # Get the exception and all its causes
         exceptions = []
         exc_value = self.exc_value
         while exc_value:
             exceptions.append(exc_value)
-            exc_value = explicit_or_implicit_cause(exc_value)
+            exc_value = self._get_explicit_or_implicit_cause(exc_value)
             if exc_value in exceptions:
                 warnings.warn(
                     "Cycle in the exception chain detected: exception '%s' "
@@ -424,6 +424,17 @@ def explicit_or_implicit_cause(exc_value):
         # In case there's just one exception, take the traceback from self.tb
         exc_value = exceptions.pop()
         tb = self.tb if not exceptions else exc_value.__traceback__
+        frames.extend(self.get_exception_traceback_frames(exc_value, tb))
+        while exceptions:
+            exc_value = exceptions.pop()
+            frames.extend(
+                self.get_exception_traceback_frames(exc_value, exc_value.__traceback__),
+            )
+        return frames
+
+    def get_exception_traceback_frames(self, exc_value, tb):
+        exc_cause = self._get_explicit_or_implicit_cause(exc_value)
+        exc_cause_explicit = getattr(exc_value, '__cause__', True)
 
         while tb is not None:
             # Support for __traceback_hide__ which is used by a few libraries
@@ -444,9 +455,9 @@ def explicit_or_implicit_cause(exc_value):
                 pre_context = []
                 context_line = '<source code not available>'
                 post_context = []
-            frames.append({
-                'exc_cause': explicit_or_implicit_cause(exc_value),
-                'exc_cause_explicit': getattr(exc_value, '__cause__', True),
+            yield {
+                'exc_cause': exc_cause,
+                'exc_cause_explicit': exc_cause_explicit,
                 'tb': tb,
                 'type': 'django' if module_name.startswith('django.') else 'user',
                 'filename': filename,
@@ -458,17 +469,8 @@ def explicit_or_implicit_cause(exc_value):
                 'context_line': context_line,
                 'post_context': post_context,
                 'pre_context_lineno': pre_context_lineno + 1,
-            })
-
-            # If the traceback for current exception is consumed, try the
-            # other exception.
-            if not tb.tb_next and exceptions:
-                exc_value = exceptions.pop()
-                tb = exc_value.__traceback__
-            else:
-                tb = tb.tb_next
-
-        return frames
+            }
+            tb = tb.tb_next
 
 
 def technical_404_response(request, exception):

```

## Test Patch

```diff
diff --git a/tests/view_tests/tests/test_debug.py b/tests/view_tests/tests/test_debug.py
--- a/tests/view_tests/tests/test_debug.py
+++ b/tests/view_tests/tests/test_debug.py
@@ -467,6 +467,34 @@ def test_suppressed_context(self):
         self.assertIn('<p>Request data not supplied</p>', html)
         self.assertNotIn('During handling of the above exception', html)
 
+    def test_innermost_exception_without_traceback(self):
+        try:
+            try:
+                raise RuntimeError('Oops')
+            except Exception as exc:
+                new_exc = RuntimeError('My context')
+                exc.__context__ = new_exc
+                raise
+        except Exception:
+            exc_type, exc_value, tb = sys.exc_info()
+
+        reporter = ExceptionReporter(None, exc_type, exc_value, tb)
+        frames = reporter.get_traceback_frames()
+        self.assertEqual(len(frames), 1)
+        html = reporter.get_traceback_html()
+        self.assertInHTML('<h1>RuntimeError</h1>', html)
+        self.assertIn('<pre class="exception_value">Oops</pre>', html)
+        self.assertIn('<th>Exception Type:</th>', html)
+        self.assertIn('<th>Exception Value:</th>', html)
+        self.assertIn('<h2>Traceback ', html)
+        self.assertIn('<h2>Request information</h2>', html)
+        self.assertIn('<p>Request data not supplied</p>', html)
+        self.assertIn(
+            'During handling of the above exception (My context), another '
+            'exception occurred',
+            html,
+        )
+
     def test_reporting_of_nested_exceptions(self):
         request = self.rf.get('/test_view/')
         try:

```


## Code snippets

### 1 - django/views/debug.py:

Start line: 397, End line: 471

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
### 2 - django/views/debug.py:

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
### 3 - django/views/debug.py:

Start line: 363, End line: 395

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
### 4 - django/core/handlers/base.py:

Start line: 292, End line: 316

```python
class BaseHandler:

    def check_response(self, response, callback, name=None):
        """
        Raise an error if the view returned None or an uncalled coroutine.
        """
        if not(response is None or asyncio.iscoroutine(response)):
            return
        if not name:
            if isinstance(callback, types.FunctionType):  # FBV
                name = 'The view %s.%s' % (callback.__module__, callback.__name__)
            else:  # CBV
                name = 'The view %s.%s.__call__' % (
                    callback.__module__,
                    callback.__class__.__name__,
                )
        if response is None:
            raise ValueError(
                "%s didn't return an HttpResponse object. It returned None "
                "instead." % name
            )
        elif asyncio.iscoroutine(response):
            raise ValueError(
                "%s didn't return an HttpResponse object. It returned an "
                "unawaited coroutine instead. You may need to add an 'await' "
                "into your view." % name
            )
```
### 5 - django/views/debug.py:

Start line: 181, End line: 193

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
### 6 - django/views/generic/__init__.py:

Start line: 1, End line: 23

```python
from django.views.generic.base import RedirectView, TemplateView, View
from django.views.generic.dates import (
    ArchiveIndexView, DateDetailView, DayArchiveView, MonthArchiveView,
    TodayArchiveView, WeekArchiveView, YearArchiveView,
)
from django.views.generic.detail import DetailView
from django.views.generic.edit import (
    CreateView, DeleteView, FormView, UpdateView,
)
from django.views.generic.list import ListView

__all__ = [
    'View', 'TemplateView', 'RedirectView', 'ArchiveIndexView',
    'YearArchiveView', 'MonthArchiveView', 'WeekArchiveView', 'DayArchiveView',
    'TodayArchiveView', 'DateDetailView', 'DetailView', 'FormView',
    'CreateView', 'UpdateView', 'DeleteView', 'ListView', 'GenericViewError',
]


class GenericViewError(Exception):
    """A problem in a generic view."""
    pass
```
### 7 - django/views/debug.py:

Start line: 332, End line: 361

```python
class ExceptionReporter:

    def get_traceback_html(self):
        """Return HTML version of debug 500 HTTP error page."""
        with Path(CURRENT_DIR, 'templates', 'technical_500.html').open(encoding='utf-8') as fh:
            t = DEBUG_ENGINE.from_string(fh.read())
        c = Context(self.get_traceback_data(), use_l10n=False)
        return t.render(c)

    def get_traceback_text(self):
        """Return plain text version of debug 500 HTTP error page."""
        with Path(CURRENT_DIR, 'templates', 'technical_500.txt').open(encoding='utf-8') as fh:
            t = DEBUG_ENGINE.from_string(fh.read())
        c = Context(self.get_traceback_data(), autoescape=False, use_l10n=False)
        return t.render(c)

    def _get_source(self, filename, loader, module_name):
        source = None
        if hasattr(loader, 'get_source'):
            try:
                source = loader.get_source(module_name)
            except ImportError:
                pass
            if source is not None:
                source = source.splitlines()
        if source is None:
            try:
                with open(filename, 'rb') as fp:
                    source = fp.read().splitlines()
            except OSError:
                pass
        return source
```
### 8 - django/views/debug.py:

Start line: 195, End line: 243

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
### 9 - django/views/decorators/debug.py:

Start line: 77, End line: 92

```python
def sensitive_post_parameters(*parameters):
    # ... other code

    def decorator(view):
        @functools.wraps(view)
        def sensitive_post_parameters_wrapper(request, *args, **kwargs):
            assert isinstance(request, HttpRequest), (
                "sensitive_post_parameters didn't receive an HttpRequest. "
                "If you are decorating a classmethod, be sure to use "
                "@method_decorator."
            )
            if parameters:
                request.sensitive_post_parameters = parameters
            else:
                request.sensitive_post_parameters = '__ALL__'
            return view(request, *args, **kwargs)
        return sensitive_post_parameters_wrapper
    return decorator
```
### 10 - django/template/backends/jinja2.py:

Start line: 91, End line: 126

```python
def get_exception_info(exception):
    """
    Format exception information for display on the debug page using the
    structure described in the template API documentation.
    """
    context_lines = 10
    lineno = exception.lineno
    source = exception.source
    if source is None:
        exception_file = Path(exception.filename)
        if exception_file.exists():
            with open(exception_file, 'r') as fp:
                source = fp.read()
    if source is not None:
        lines = list(enumerate(source.strip().split('\n'), start=1))
        during = lines[lineno - 1][1]
        total = len(lines)
        top = max(0, lineno - context_lines - 1)
        bottom = min(total, lineno + context_lines)
    else:
        during = ''
        lines = []
        total = top = bottom = 0
    return {
        'name': exception.filename,
        'message': exception.message,
        'source_lines': lines[top:bottom],
        'line': lineno,
        'before': '',
        'during': during,
        'after': '',
        'total': total,
        'top': top,
        'bottom': bottom,
    }
```
### 12 - django/views/debug.py:

Start line: 80, End line: 112

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
            is_sensitive = self.hidden_settings.search(key)
        except TypeError:
            is_sensitive = False

        if is_sensitive:
            cleansed = self.cleansed_substitute
        elif isinstance(value, dict):
            cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            cleansed = [self.cleanse_setting('', v) for v in value]
        elif isinstance(value, tuple):
            cleansed = tuple([self.cleanse_setting('', v) for v in value])
        else:
            cleansed = value

        if callable(cleansed):
            cleansed = CallableSettingWrapper(cleansed)

        return cleansed
```
### 14 - django/views/debug.py:

Start line: 246, End line: 330

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
### 16 - django/views/debug.py:

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
### 20 - django/views/debug.py:

Start line: 50, End line: 61

```python
def technical_500_response(request, exc_type, exc_value, tb, status_code=500):
    """
    Create a technical server error response. The last three arguments are
    the values returned from sys.exc_info() and friends.
    """
    reporter = get_exception_reporter_class(request)(request, exc_type, exc_value, tb)
    if request.accepts('text/html'):
        html = reporter.get_traceback_html()
        return HttpResponse(html, status=status_code, content_type='text/html')
    else:
        text = reporter.get_traceback_text()
        return HttpResponse(text, status=status_code, content_type='text/plain; charset=utf-8')
```
### 30 - django/views/debug.py:

Start line: 142, End line: 154

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
### 31 - django/views/debug.py:

Start line: 474, End line: 543

```python
def technical_404_response(request, exception):
    """Create a technical 404 error response. `exception` is the Http404."""
    try:
        error_url = exception.args[0]['path']
    except (IndexError, TypeError, KeyError):
        error_url = request.path_info[1:]  # Trim leading slash

    try:
        tried = exception.args[0]['tried']
    except (IndexError, TypeError, KeyError):
        resolved = True
        tried = request.resolver_match.tried if request.resolver_match else None
    else:
        resolved = False
        if (not tried or (                  # empty URLconf
            request.path == '/' and
            len(tried) == 1 and             # default URLconf
            len(tried[0]) == 1 and
            getattr(tried[0][0], 'app_name', '') == getattr(tried[0][0], 'namespace', '') == 'admin'
        )):
            return default_urlconf(request)

    urlconf = getattr(request, 'urlconf', settings.ROOT_URLCONF)
    if isinstance(urlconf, types.ModuleType):
        urlconf = urlconf.__name__

    caller = ''
    try:
        resolver_match = resolve(request.path)
    except Http404:
        pass
    else:
        obj = resolver_match.func

        if hasattr(obj, '__name__'):
            caller = obj.__name__
        elif hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
            caller = obj.__class__.__name__

        if hasattr(obj, '__module__'):
            module = obj.__module__
            caller = '%s.%s' % (module, caller)

    with Path(CURRENT_DIR, 'templates', 'technical_404.html').open(encoding='utf-8') as fh:
        t = DEBUG_ENGINE.from_string(fh.read())
    reporter_filter = get_default_exception_reporter_filter()
    c = Context({
        'urlconf': urlconf,
        'root_urlconf': settings.ROOT_URLCONF,
        'request_path': error_url,
        'urlpatterns': tried,
        'resolved': resolved,
        'reason': str(exception),
        'request': request,
        'settings': reporter_filter.get_safe_settings(),
        'raising_view_name': caller,
    })
    return HttpResponseNotFound(t.render(c), content_type='text/html')


def default_urlconf(request):
    """Create an empty URLconf 404 error response."""
    with Path(CURRENT_DIR, 'templates', 'default_urlconf.html').open(encoding='utf-8') as fh:
        t = DEBUG_ENGINE.from_string(fh.read())
    c = Context({
        'version': get_docs_version(),
    })

    return HttpResponse(t.render(c), content_type='text/html')
```
### 39 - django/views/debug.py:

Start line: 156, End line: 179

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
### 120 - django/views/debug.py:

Start line: 114, End line: 140

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
