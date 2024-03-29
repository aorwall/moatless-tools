# django__django-15766

| **django/django** | `be63c78760924e1335603c36babd0ad6cfaea3c4` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 766 |
| **Any found context length** | 766 |
| **Avg pos** | 35.5 |
| **Min pos** | 3 |
| **Max pos** | 34 |
| **Top file pos** | 3 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/db/backends/base/base.py b/django/db/backends/base/base.py
--- a/django/db/backends/base/base.py
+++ b/django/db/backends/base/base.py
@@ -1,6 +1,7 @@
 import _thread
 import copy
 import datetime
+import logging
 import threading
 import time
 import warnings
@@ -26,6 +27,8 @@
 NO_DB_ALIAS = "__no_db__"
 RAN_DB_VERSION_CHECK = set()
 
+logger = logging.getLogger("django.db.backends.base")
+
 
 # RemovedInDjango50Warning
 def timezone_constructor(tzname):
@@ -417,7 +420,9 @@ def savepoint_rollback(self, sid):
 
         # Remove any callbacks registered while this savepoint was active.
         self.run_on_commit = [
-            (sids, func) for (sids, func) in self.run_on_commit if sid not in sids
+            (sids, func, robust)
+            for (sids, func, robust) in self.run_on_commit
+            if sid not in sids
         ]
 
     @async_unsafe
@@ -723,12 +728,12 @@ def schema_editor(self, *args, **kwargs):
             )
         return self.SchemaEditorClass(self, *args, **kwargs)
 
-    def on_commit(self, func):
+    def on_commit(self, func, robust=False):
         if not callable(func):
             raise TypeError("on_commit()'s callback must be a callable.")
         if self.in_atomic_block:
             # Transaction in progress; save for execution on commit.
-            self.run_on_commit.append((set(self.savepoint_ids), func))
+            self.run_on_commit.append((set(self.savepoint_ids), func, robust))
         elif not self.get_autocommit():
             raise TransactionManagementError(
                 "on_commit() cannot be used in manual transaction management"
@@ -736,15 +741,36 @@ def on_commit(self, func):
         else:
             # No transaction in progress and in autocommit mode; execute
             # immediately.
-            func()
+            if robust:
+                try:
+                    func()
+                except Exception as e:
+                    logger.error(
+                        f"Error calling {func.__qualname__} in on_commit() (%s).",
+                        e,
+                        exc_info=True,
+                    )
+            else:
+                func()
 
     def run_and_clear_commit_hooks(self):
         self.validate_no_atomic_block()
         current_run_on_commit = self.run_on_commit
         self.run_on_commit = []
         while current_run_on_commit:
-            sids, func = current_run_on_commit.pop(0)
-            func()
+            _, func, robust = current_run_on_commit.pop(0)
+            if robust:
+                try:
+                    func()
+                except Exception as e:
+                    logger.error(
+                        f"Error calling {func.__qualname__} in on_commit() during "
+                        f"transaction (%s).",
+                        e,
+                        exc_info=True,
+                    )
+            else:
+                func()
 
     @contextmanager
     def execute_wrapper(self, wrapper):
diff --git a/django/db/transaction.py b/django/db/transaction.py
--- a/django/db/transaction.py
+++ b/django/db/transaction.py
@@ -125,12 +125,12 @@ def mark_for_rollback_on_error(using=None):
         raise
 
 
-def on_commit(func, using=None):
+def on_commit(func, using=None, robust=False):
     """
     Register `func` to be called when the current transaction is committed.
     If the current transaction is rolled back, `func` will not be called.
     """
-    get_connection(using).on_commit(func)
+    get_connection(using).on_commit(func, robust)
 
 
 #################################

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/db/backends/base/base.py | 4 | 4 | 34 | 4 | 14219
| django/db/backends/base/base.py | 29 | 29 | 34 | 4 | 14219
| django/db/backends/base/base.py | 420 | 420 | - | 4 | -
| django/db/backends/base/base.py | 726 | 731 | - | 4 | -
| django/db/backends/base/base.py | 739 | 747 | - | 4 | -
| django/db/transaction.py | 128 | 133 | 3 | 3 | 766


## Problem Statement

```
Supporting robust on_commit handlers.
Description
	 
		(last modified by Josh Smeaton)
	 
I recently tracked down an issue in my application where some on_commit handlers didn't execute because one of the previous handlers raised an exception. There appears to be no way to execute on_commit handlers *robustly* as you're able to do with signals [0] using send_robust.
I could sprinkle try/catches around the place, but I'd like to avoid doing so because not all functions that are used as handlers should always swallow exceptions, but could do so when run as on_commit handlers.
Targeting which handlers can be robust or not would be really useful, for example:
def update_search(user):
	# if updating search fails, it's fine, we'll bulk update later anyway
	transaction.on_commit(lambda: search.update(user), robust=True)
def trigger_background_task_one(user):
	# if this task fails, we want to crash
	transaction.on_commit(lambda: mytask.delay(user_id=user.id))
Here if search fails to update it doesn't prevent the background task from being scheduled.
I'm proposing to add a robust kwarg that defaults to False, for backward compatibility, but allows a user to tag specific handlers as such.
[0] ​https://docs.djangoproject.com/en/4.0/topics/signals/#sending-signals

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 django/dispatch/dispatcher.py | 181 | 232| 368 | 368 | 2105 | 
| 2 | 2 django/db/backends/base/features.py | 359 | 377| 173 | 541 | 5174 | 
| **-> 3 <-** | **3 django/db/transaction.py** | 100 | 133| 225 | 766 | 7484 | 
| 4 | 3 django/db/backends/base/features.py | 6 | 221| 1745 | 2511 | 7484 | 
| 5 | **4 django/db/backends/base/base.py** | 493 | 588| 604 | 3115 | 12913 | 
| 6 | **4 django/db/transaction.py** | 315 | 340| 181 | 3296 | 12913 | 
| 7 | 5 django/contrib/postgres/signals.py | 37 | 69| 264 | 3560 | 13421 | 
| 8 | **5 django/db/transaction.py** | 136 | 179| 367 | 3927 | 13421 | 
| 9 | **5 django/db/backends/base/base.py** | 458 | 491| 276 | 4203 | 13421 | 
| 10 | 5 django/db/backends/base/features.py | 222 | 357| 1171 | 5374 | 13421 | 
| 11 | 6 django/db/models/base.py | 838 | 904| 475 | 5849 | 31972 | 
| 12 | **6 django/db/transaction.py** | 223 | 312| 635 | 6484 | 31972 | 
| 13 | 7 django/db/models/deletion.py | 1 | 93| 601 | 7085 | 35954 | 
| 14 | 8 django/db/backends/sqlite3/features.py | 1 | 64| 632 | 7717 | 37201 | 
| 15 | 9 django/db/__init__.py | 1 | 62| 299 | 8016 | 37500 | 
| 16 | 10 django/db/backends/oracle/features.py | 1 | 83| 709 | 8725 | 38808 | 
| 17 | **10 django/db/transaction.py** | 181 | 221| 333 | 9058 | 38808 | 
| 18 | 11 django/db/backends/signals.py | 1 | 4| 0 | 9058 | 38819 | 
| 19 | **11 django/db/transaction.py** | 1 | 82| 442 | 9500 | 38819 | 
| 20 | 12 django/utils/autoreload.py | 585 | 608| 177 | 9677 | 43945 | 
| 21 | 13 django/db/utils.py | 138 | 194| 517 | 10194 | 45844 | 
| 22 | 14 django/core/management/sql.py | 42 | 60| 132 | 10326 | 46211 | 
| 23 | 15 django/core/checks/model_checks.py | 161 | 185| 267 | 10593 | 48021 | 
| 24 | 16 django/db/models/options.py | 1 | 57| 347 | 10940 | 55609 | 
| 25 | 17 django/core/signals.py | 1 | 7| 0 | 10940 | 55636 | 
| 26 | 17 django/db/models/deletion.py | 142 | 185| 327 | 11267 | 55636 | 
| 27 | 17 django/db/backends/sqlite3/features.py | 115 | 149| 245 | 11512 | 55636 | 
| 28 | 17 django/utils/autoreload.py | 90 | 106| 161 | 11673 | 55636 | 
| 29 | 18 django/db/backends/postgresql/features.py | 1 | 103| 792 | 12465 | 56428 | 
| 30 | 18 django/dispatch/dispatcher.py | 234 | 280| 444 | 12909 | 56428 | 
| 31 | 18 django/utils/autoreload.py | 1 | 56| 293 | 13202 | 56428 | 
| 32 | 19 django/db/backends/sqlite3/_functions.py | 40 | 84| 671 | 13873 | 60284 | 
| 33 | **19 django/db/transaction.py** | 85 | 97| 135 | 14008 | 60284 | 
| **-> 34 <-** | **19 django/db/backends/base/base.py** | 1 | 36| 211 | 14219 | 60284 | 
| 35 | 19 django/dispatch/dispatcher.py | 1 | 45| 268 | 14487 | 60284 | 
| 36 | 20 django/contrib/auth/backends.py | 187 | 250| 486 | 14973 | 62175 | 
| 37 | 21 django/db/models/fields/related_descriptors.py | 1293 | 1324| 344 | 15317 | 73471 | 
| 38 | 21 django/db/models/options.py | 172 | 241| 645 | 15962 | 73471 | 
| 39 | 22 django/core/handlers/asgi.py | 1 | 24| 116 | 16078 | 75855 | 
| 40 | 23 django/db/backends/mysql/base.py | 276 | 312| 259 | 16337 | 79367 | 
| 41 | 23 django/db/models/base.py | 1 | 64| 346 | 16683 | 79367 | 
| 42 | 24 django/db/models/__init__.py | 1 | 116| 682 | 17365 | 80049 | 
| 43 | 25 django/db/backends/mysql/features.py | 271 | 330| 439 | 17804 | 82420 | 
| 44 | 26 django/core/handlers/base.py | 343 | 374| 214 | 18018 | 85070 | 
| 45 | 27 django/core/handlers/exception.py | 161 | 186| 167 | 18185 | 86188 | 
| 46 | 28 django/contrib/admin/options.py | 1537 | 1604| 586 | 18771 | 105408 | 
| 47 | 29 django/core/management/commands/squashmigrations.py | 62 | 160| 809 | 19580 | 107448 | 
| 48 | 30 django/utils/asyncio.py | 1 | 40| 221 | 19801 | 107670 | 
| 49 | 31 django/views/debug.py | 225 | 277| 471 | 20272 | 112416 | 
| 50 | 32 django/db/backends/sqlite3/base.py | 233 | 352| 888 | 21160 | 115506 | 
| 51 | 32 django/contrib/auth/backends.py | 1 | 32| 197 | 21357 | 115506 | 
| 52 | 32 django/db/backends/sqlite3/features.py | 66 | 113| 383 | 21740 | 115506 | 
| 53 | 32 django/db/backends/mysql/features.py | 162 | 269| 728 | 22468 | 115506 | 
| 54 | 33 django/core/checks/__init__.py | 1 | 48| 327 | 22795 | 115833 | 
| 55 | 34 django/db/models/query.py | 1906 | 1971| 539 | 23334 | 136283 | 
| 56 | 34 django/db/models/query.py | 684 | 743| 479 | 23813 | 136283 | 
| 57 | 35 django/views/decorators/debug.py | 1 | 46| 273 | 24086 | 136874 | 
| 58 | 35 django/db/models/options.py | 440 | 462| 164 | 24250 | 136874 | 
| 59 | 35 django/db/backends/mysql/features.py | 86 | 160| 597 | 24847 | 136874 | 
| 60 | 36 django/db/backends/sqlite3/operations.py | 417 | 437| 148 | 24995 | 140361 | 
| 61 | 37 django/db/backends/postgresql/operations.py | 338 | 357| 150 | 25145 | 143350 | 
| 62 | 38 django/utils/log.py | 79 | 142| 475 | 25620 | 145024 | 
| 63 | 39 django/db/backends/oracle/base.py | 64 | 102| 325 | 25945 | 150173 | 
| 64 | 40 django/db/backends/utils.py | 66 | 94| 255 | 26200 | 152208 | 
| 65 | 40 django/core/handlers/asgi.py | 161 | 189| 257 | 26457 | 152208 | 
| 66 | 40 django/contrib/auth/backends.py | 167 | 185| 146 | 26603 | 152208 | 
| 67 | 41 django/contrib/auth/middleware.py | 87 | 114| 195 | 26798 | 153218 | 
| 68 | 41 django/dispatch/dispatcher.py | 283 | 306| 139 | 26937 | 153218 | 
| 69 | 41 django/core/checks/model_checks.py | 187 | 228| 345 | 27282 | 153218 | 
| 70 | 41 django/db/backends/utils.py | 1 | 46| 287 | 27569 | 153218 | 
| 71 | 42 django/core/management/commands/compilemessages.py | 1 | 27| 162 | 27731 | 154577 | 
| 72 | 42 django/utils/log.py | 145 | 168| 125 | 27856 | 154577 | 
| 73 | 42 django/utils/autoreload.py | 431 | 464| 354 | 28210 | 154577 | 
| 74 | **42 django/db/backends/base/base.py** | 617 | 649| 230 | 28440 | 154577 | 
| 75 | 43 django/core/management/base.py | 460 | 554| 665 | 29105 | 159363 | 
| 76 | 43 django/db/backends/utils.py | 97 | 144| 327 | 29432 | 159363 | 
| 77 | 43 django/core/management/commands/squashmigrations.py | 162 | 254| 766 | 30198 | 159363 | 
| 78 | 44 django/db/backends/postgresql/base.py | 304 | 325| 156 | 30354 | 162423 | 
| 79 | 44 django/utils/autoreload.py | 384 | 428| 273 | 30627 | 162423 | 
| 80 | 45 django/contrib/auth/signals.py | 1 | 6| 0 | 30627 | 162447 | 
| 81 | 45 django/db/backends/utils.py | 48 | 64| 176 | 30803 | 162447 | 
| 82 | 46 django/db/models/sql/compiler.py | 1911 | 1972| 588 | 31391 | 178593 | 
| 83 | 47 django/core/checks/compatibility/django_4_0.py | 1 | 21| 142 | 31533 | 178736 | 
| 84 | 48 django/db/migrations/autodetector.py | 1504 | 1532| 235 | 31768 | 192197 | 
| 85 | 49 django/contrib/postgres/operations.py | 1 | 37| 271 | 32039 | 194545 | 
| 86 | 49 django/contrib/auth/middleware.py | 48 | 85| 362 | 32401 | 194545 | 
| 87 | 49 django/db/backends/sqlite3/operations.py | 142 | 167| 280 | 32681 | 194545 | 
| 88 | 50 django/middleware/common.py | 153 | 179| 255 | 32936 | 196091 | 
| 89 | 50 django/db/backends/oracle/features.py | 85 | 149| 604 | 33540 | 196091 | 
| 90 | 50 django/db/models/options.py | 330 | 366| 338 | 33878 | 196091 | 


### Hint

```
Sounds reasonable. Please take into account that the current behavior is ​documented.
Josh, Would you like to prepare a patch?
I haven't got the time to put a patch together *right now* but I could do so in the near future. Consider tagging this as "easy pickings" for a budding contributor?
I've started an easy pickings thread on -developers ML and would be happy to review and guide someone to make the change: ​https://groups.google.com/g/django-developers/c/Hyqd1Rz6cFs
Good feature suggestion. The execution in captureOnCommitCallbacks would need extending too :)
​PR
Can this ticket be closed? Seems like the PR was accepted.
Replying to Shivan Sivakumaran: Can this ticket be closed? Seems like the PR was accepted. Not until the PR is merged, then it'll be closed as fixed
```

## Patch

```diff
diff --git a/django/db/backends/base/base.py b/django/db/backends/base/base.py
--- a/django/db/backends/base/base.py
+++ b/django/db/backends/base/base.py
@@ -1,6 +1,7 @@
 import _thread
 import copy
 import datetime
+import logging
 import threading
 import time
 import warnings
@@ -26,6 +27,8 @@
 NO_DB_ALIAS = "__no_db__"
 RAN_DB_VERSION_CHECK = set()
 
+logger = logging.getLogger("django.db.backends.base")
+
 
 # RemovedInDjango50Warning
 def timezone_constructor(tzname):
@@ -417,7 +420,9 @@ def savepoint_rollback(self, sid):
 
         # Remove any callbacks registered while this savepoint was active.
         self.run_on_commit = [
-            (sids, func) for (sids, func) in self.run_on_commit if sid not in sids
+            (sids, func, robust)
+            for (sids, func, robust) in self.run_on_commit
+            if sid not in sids
         ]
 
     @async_unsafe
@@ -723,12 +728,12 @@ def schema_editor(self, *args, **kwargs):
             )
         return self.SchemaEditorClass(self, *args, **kwargs)
 
-    def on_commit(self, func):
+    def on_commit(self, func, robust=False):
         if not callable(func):
             raise TypeError("on_commit()'s callback must be a callable.")
         if self.in_atomic_block:
             # Transaction in progress; save for execution on commit.
-            self.run_on_commit.append((set(self.savepoint_ids), func))
+            self.run_on_commit.append((set(self.savepoint_ids), func, robust))
         elif not self.get_autocommit():
             raise TransactionManagementError(
                 "on_commit() cannot be used in manual transaction management"
@@ -736,15 +741,36 @@ def on_commit(self, func):
         else:
             # No transaction in progress and in autocommit mode; execute
             # immediately.
-            func()
+            if robust:
+                try:
+                    func()
+                except Exception as e:
+                    logger.error(
+                        f"Error calling {func.__qualname__} in on_commit() (%s).",
+                        e,
+                        exc_info=True,
+                    )
+            else:
+                func()
 
     def run_and_clear_commit_hooks(self):
         self.validate_no_atomic_block()
         current_run_on_commit = self.run_on_commit
         self.run_on_commit = []
         while current_run_on_commit:
-            sids, func = current_run_on_commit.pop(0)
-            func()
+            _, func, robust = current_run_on_commit.pop(0)
+            if robust:
+                try:
+                    func()
+                except Exception as e:
+                    logger.error(
+                        f"Error calling {func.__qualname__} in on_commit() during "
+                        f"transaction (%s).",
+                        e,
+                        exc_info=True,
+                    )
+            else:
+                func()
 
     @contextmanager
     def execute_wrapper(self, wrapper):
diff --git a/django/db/transaction.py b/django/db/transaction.py
--- a/django/db/transaction.py
+++ b/django/db/transaction.py
@@ -125,12 +125,12 @@ def mark_for_rollback_on_error(using=None):
         raise
 
 
-def on_commit(func, using=None):
+def on_commit(func, using=None, robust=False):
     """
     Register `func` to be called when the current transaction is committed.
     If the current transaction is rolled back, `func` will not be called.
     """
-    get_connection(using).on_commit(func)
+    get_connection(using).on_commit(func, robust)
 
 
 #################################

```

## Test Patch

```diff
diff --git a/django/test/testcases.py b/django/test/testcases.py
--- a/django/test/testcases.py
+++ b/django/test/testcases.py
@@ -59,6 +59,8 @@
 from django.utils.version import PY310
 from django.views.static import serve
 
+logger = logging.getLogger("django.test")
+
 __all__ = (
     "TestCase",
     "TransactionTestCase",
@@ -1510,10 +1512,23 @@ def captureOnCommitCallbacks(cls, *, using=DEFAULT_DB_ALIAS, execute=False):
         finally:
             while True:
                 callback_count = len(connections[using].run_on_commit)
-                for _, callback in connections[using].run_on_commit[start_count:]:
+                for _, callback, robust in connections[using].run_on_commit[
+                    start_count:
+                ]:
                     callbacks.append(callback)
                     if execute:
-                        callback()
+                        if robust:
+                            try:
+                                callback()
+                            except Exception as e:
+                                logger.error(
+                                    f"Error calling {callback.__qualname__} in "
+                                    f"on_commit() (%s).",
+                                    e,
+                                    exc_info=True,
+                                )
+                        else:
+                            callback()
 
                 if callback_count == len(connections[using].run_on_commit):
                     break
diff --git a/tests/test_utils/tests.py b/tests/test_utils/tests.py
--- a/tests/test_utils/tests.py
+++ b/tests/test_utils/tests.py
@@ -2285,6 +2285,32 @@ def branch_2():
 
         self.assertEqual(callbacks, [branch_1, branch_2, leaf_3, leaf_1, leaf_2])
 
+    def test_execute_robust(self):
+        class MyException(Exception):
+            pass
+
+        def hook():
+            self.callback_called = True
+            raise MyException("robust callback")
+
+        with self.assertLogs("django.test", "ERROR") as cm:
+            with self.captureOnCommitCallbacks(execute=True) as callbacks:
+                transaction.on_commit(hook, robust=True)
+
+        self.assertEqual(len(callbacks), 1)
+        self.assertIs(self.callback_called, True)
+
+        log_record = cm.records[0]
+        self.assertEqual(
+            log_record.getMessage(),
+            "Error calling CaptureOnCommitCallbacksTests.test_execute_robust.<locals>."
+            "hook in on_commit() (robust callback).",
+        )
+        self.assertIsNotNone(log_record.exc_info)
+        raised_exception = log_record.exc_info[1]
+        self.assertIsInstance(raised_exception, MyException)
+        self.assertEqual(str(raised_exception), "robust callback")
+
 
 class DisallowedDatabaseQueriesTests(SimpleTestCase):
     def test_disallowed_database_connections(self):
diff --git a/tests/transaction_hooks/tests.py b/tests/transaction_hooks/tests.py
--- a/tests/transaction_hooks/tests.py
+++ b/tests/transaction_hooks/tests.py
@@ -43,6 +43,47 @@ def test_executes_immediately_if_no_transaction(self):
         self.do(1)
         self.assertDone([1])
 
+    def test_robust_if_no_transaction(self):
+        def robust_callback():
+            raise ForcedError("robust callback")
+
+        with self.assertLogs("django.db.backends.base", "ERROR") as cm:
+            transaction.on_commit(robust_callback, robust=True)
+            self.do(1)
+
+        self.assertDone([1])
+        log_record = cm.records[0]
+        self.assertEqual(
+            log_record.getMessage(),
+            "Error calling TestConnectionOnCommit.test_robust_if_no_transaction."
+            "<locals>.robust_callback in on_commit() (robust callback).",
+        )
+        self.assertIsNotNone(log_record.exc_info)
+        raised_exception = log_record.exc_info[1]
+        self.assertIsInstance(raised_exception, ForcedError)
+        self.assertEqual(str(raised_exception), "robust callback")
+
+    def test_robust_transaction(self):
+        def robust_callback():
+            raise ForcedError("robust callback")
+
+        with self.assertLogs("django.db.backends", "ERROR") as cm:
+            with transaction.atomic():
+                transaction.on_commit(robust_callback, robust=True)
+                self.do(1)
+
+        self.assertDone([1])
+        log_record = cm.records[0]
+        self.assertEqual(
+            log_record.getMessage(),
+            "Error calling TestConnectionOnCommit.test_robust_transaction.<locals>."
+            "robust_callback in on_commit() during transaction (robust callback).",
+        )
+        self.assertIsNotNone(log_record.exc_info)
+        raised_exception = log_record.exc_info[1]
+        self.assertIsInstance(raised_exception, ForcedError)
+        self.assertEqual(str(raised_exception), "robust callback")
+
     def test_delays_execution_until_after_transaction_commit(self):
         with transaction.atomic():
             self.do(1)

```


## Code snippets

### 1 - django/dispatch/dispatcher.py:

Start line: 181, End line: 232

```python
class Signal:

    def send_robust(self, sender, **named):
        """
        Send signal from sender to all connected receivers catching errors.

        Arguments:

            sender
                The sender of the signal. Can be any Python object (normally one
                registered with a connect if you actually want something to
                occur).

            named
                Named arguments which will be passed to receivers.

        Return a list of tuple pairs [(receiver, response), ... ].

        If any receiver raises an error (specifically any subclass of
        Exception), return the error instance as the result for that receiver.
        """
        if (
            not self.receivers
            or self.sender_receivers_cache.get(sender) is NO_RECEIVERS
        ):
            return []

        # Call each receiver with whatever arguments it can accept.
        # Return a list of tuple pairs [(receiver, response), ... ].
        responses = []
        for receiver in self._live_receivers(sender):
            try:
                response = receiver(signal=self, sender=sender, **named)
            except Exception as err:
                logger.error(
                    "Error calling %s in Signal.send_robust() (%s)",
                    receiver.__qualname__,
                    err,
                    exc_info=err,
                )
                responses.append((receiver, err))
            else:
                responses.append((receiver, response))
        return responses

    def _clear_dead_receivers(self):
        # Note: caller is assumed to hold self.lock.
        if self._dead_receivers:
            self._dead_receivers = False
            self.receivers = [
                r
                for r in self.receivers
                if not (isinstance(r[1], weakref.ReferenceType) and r[1]() is None)
            ]
```
### 2 - django/db/backends/base/features.py:

Start line: 359, End line: 377

```python
class BaseDatabaseFeatures:
    minimum_database_version =
    # ... other code

    @cached_property
    def supports_transactions(self):
        """Confirm support for transactions."""
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE TABLE ROLLBACK_TEST (X INT)")
            self.connection.set_autocommit(False)
            cursor.execute("INSERT INTO ROLLBACK_TEST (X) VALUES (8)")
            self.connection.rollback()
            self.connection.set_autocommit(True)
            cursor.execute("SELECT COUNT(X) FROM ROLLBACK_TEST")
            (count,) = cursor.fetchone()
            cursor.execute("DROP TABLE ROLLBACK_TEST")
        return count == 0

    def allows_group_by_selected_pks_on_model(self, model):
        if not self.allows_group_by_selected_pks:
            return False
        return model._meta.managed
```
### 3 - django/db/transaction.py:

Start line: 100, End line: 133

```python
@contextmanager
def mark_for_rollback_on_error(using=None):
    """
    Internal low-level utility to mark a transaction as "needs rollback" when
    an exception is raised while not enforcing the enclosed block to be in a
    transaction. This is needed by Model.save() and friends to avoid starting a
    transaction when in autocommit mode and a single query is executed.

    It's equivalent to:

        connection = get_connection(using)
        if connection.get_autocommit():
            yield
        else:
            with transaction.atomic(using=using, savepoint=False):
                yield

    but it uses low-level utilities to avoid performance overhead.
    """
    try:
        yield
    except Exception:
        connection = get_connection(using)
        if connection.in_atomic_block:
            connection.needs_rollback = True
        raise


def on_commit(func, using=None):
    """
    Register `func` to be called when the current transaction is committed.
    If the current transaction is rolled back, `func` will not be called.
    """
    get_connection(using).on_commit(func)
```
### 4 - django/db/backends/base/features.py:

Start line: 6, End line: 221

```python
class BaseDatabaseFeatures:
    # An optional tuple indicating the minimum supported database version.
    minimum_database_version = None
    gis_enabled = False
    # Oracle can't group by LOB (large object) data types.
    allows_group_by_lob = True
    allows_group_by_pk = False
    allows_group_by_selected_pks = False
    empty_fetchmany_value = []
    update_can_self_select = True

    # Does the backend distinguish between '' and None?
    interprets_empty_strings_as_nulls = False

    # Does the backend allow inserting duplicate NULL rows in a nullable
    # unique field? All core backends implement this correctly, but other
    # databases such as SQL Server do not.
    supports_nullable_unique_constraints = True

    # Does the backend allow inserting duplicate rows when a unique_together
    # constraint exists and some fields are nullable but not all of them?
    supports_partially_nullable_unique_constraints = True
    # Does the backend support initially deferrable unique constraints?
    supports_deferrable_unique_constraints = False

    can_use_chunked_reads = True
    can_return_columns_from_insert = False
    can_return_rows_from_bulk_insert = False
    has_bulk_insert = True
    uses_savepoints = True
    can_release_savepoints = False

    # If True, don't use integer foreign keys referring to, e.g., positive
    # integer primary keys.
    related_fields_match_type = False
    allow_sliced_subqueries_with_in = True
    has_select_for_update = False
    has_select_for_update_nowait = False
    has_select_for_update_skip_locked = False
    has_select_for_update_of = False
    has_select_for_no_key_update = False
    # Does the database's SELECT FOR UPDATE OF syntax require a column rather
    # than a table?
    select_for_update_of_column = False

    # Does the default test database allow multiple connections?
    # Usually an indication that the test database is in-memory
    test_db_allows_multiple_connections = True

    # Can an object be saved without an explicit primary key?
    supports_unspecified_pk = False

    # Can a fixture contain forward references? i.e., are
    # FK constraints checked at the end of transaction, or
    # at the end of each save operation?
    supports_forward_references = True

    # Does the backend truncate names properly when they are too long?
    truncates_names = False

    # Is there a REAL datatype in addition to floats/doubles?
    has_real_datatype = False
    supports_subqueries_in_group_by = True

    # Does the backend ignore unnecessary ORDER BY clauses in subqueries?
    ignores_unnecessary_order_by_in_subqueries = True

    # Is there a true datatype for uuid?
    has_native_uuid_field = False

    # Is there a true datatype for timedeltas?
    has_native_duration_field = False

    # Does the database driver supports same type temporal data subtraction
    # by returning the type used to store duration field?
    supports_temporal_subtraction = False

    # Does the __regex lookup support backreferencing and grouping?
    supports_regex_backreferencing = True

    # Can date/datetime lookups be performed using a string?
    supports_date_lookup_using_string = True

    # Can datetimes with timezones be used?
    supports_timezones = True

    # Does the database have a copy of the zoneinfo database?
    has_zoneinfo_database = True

    # When performing a GROUP BY, is an ORDER BY NULL required
    # to remove any ordering?
    requires_explicit_null_ordering_when_grouping = False

    # Does the backend order NULL values as largest or smallest?
    nulls_order_largest = False

    # Does the backend support NULLS FIRST and NULLS LAST in ORDER BY?
    supports_order_by_nulls_modifier = True

    # Does the backend orders NULLS FIRST by default?
    order_by_nulls_first = False

    # The database's limit on the number of query parameters.
    max_query_params = None

    # Can an object have an autoincrement primary key of 0?
    allows_auto_pk_0 = True

    # Do we need to NULL a ForeignKey out, or can the constraint check be
    # deferred
    can_defer_constraint_checks = False

    # Does the backend support tablespaces? Default to False because it isn't
    # in the SQL standard.
    supports_tablespaces = False

    # Does the backend reset sequences between tests?
    supports_sequence_reset = True

    # Can the backend introspect the default value of a column?
    can_introspect_default = True

    # Confirm support for introspected foreign keys
    # Every database can do this reliably, except MySQL,
    # which can't do it for MyISAM tables
    can_introspect_foreign_keys = True

    # Map fields which some backends may not be able to differentiate to the
    # field it's introspected as.
    introspected_field_types = {
        "AutoField": "AutoField",
        "BigAutoField": "BigAutoField",
        "BigIntegerField": "BigIntegerField",
        "BinaryField": "BinaryField",
        "BooleanField": "BooleanField",
        "CharField": "CharField",
        "DurationField": "DurationField",
        "GenericIPAddressField": "GenericIPAddressField",
        "IntegerField": "IntegerField",
        "PositiveBigIntegerField": "PositiveBigIntegerField",
        "PositiveIntegerField": "PositiveIntegerField",
        "PositiveSmallIntegerField": "PositiveSmallIntegerField",
        "SmallAutoField": "SmallAutoField",
        "SmallIntegerField": "SmallIntegerField",
        "TimeField": "TimeField",
    }

    # Can the backend introspect the column order (ASC/DESC) for indexes?
    supports_index_column_ordering = True

    # Does the backend support introspection of materialized views?
    can_introspect_materialized_views = False

    # Support for the DISTINCT ON clause
    can_distinct_on_fields = False

    # Does the backend prevent running SQL queries in broken transactions?
    atomic_transactions = True

    # Can we roll back DDL in a transaction?
    can_rollback_ddl = False

    # Does it support operations requiring references rename in a transaction?
    supports_atomic_references_rename = True

    # Can we issue more than one ALTER COLUMN clause in an ALTER TABLE?
    supports_combined_alters = False

    # Does it support foreign keys?
    supports_foreign_keys = True

    # Can it create foreign key constraints inline when adding columns?
    can_create_inline_fk = True

    # Can an index be renamed?
    can_rename_index = False

    # Does it automatically index foreign keys?
    indexes_foreign_keys = True

    # Does it support CHECK constraints?
    supports_column_check_constraints = True
    supports_table_check_constraints = True
    # Does the backend support introspection of CHECK constraints?
    can_introspect_check_constraints = True

    # Does the backend support 'pyformat' style ("... %(name)s ...", {'name': value})
    # parameter passing? Note this can be provided by the backend even if not
    # supported by the Python driver
    supports_paramstyle_pyformat = True

    # Does the backend require literal defaults, rather than parameterized ones?
    requires_literal_defaults = False

    # Does the backend require a connection reset after each material schema change?
    connection_persists_old_columns = False

    # What kind of error does the backend throw when accessing closed cursor?
    closed_cursor_error_class = ProgrammingError

    # Does 'a' LIKE 'A' match?
    has_case_insensitive_like = False

    # Suffix for backends that don't support "SELECT xxx;" queries.
    bare_select_suffix = ""

    # If NULL is implied on columns without needing to be explicitly specified
    implied_column_null = False

    # Does the backend support "select for update" queries with limit (and offset)?
    supports_select_for_update_with_limit = True

    # Does the backend ignore null expressions in GREATEST and LEAST queries unless
    # every expression is null?
    greatest_least_ignores_nulls = False

    # Can the backend clone databases for parallel test execution?
    # ... other code
```
### 5 - django/db/backends/base/base.py:

Start line: 493, End line: 588

```python
class BaseDatabaseWrapper:

    def get_rollback(self):
        """Get the "needs rollback" flag -- for *advanced use* only."""
        if not self.in_atomic_block:
            raise TransactionManagementError(
                "The rollback flag doesn't work outside of an 'atomic' block."
            )
        return self.needs_rollback

    def set_rollback(self, rollback):
        """
        Set or unset the "needs rollback" flag -- for *advanced use* only.
        """
        if not self.in_atomic_block:
            raise TransactionManagementError(
                "The rollback flag doesn't work outside of an 'atomic' block."
            )
        self.needs_rollback = rollback

    def validate_no_atomic_block(self):
        """Raise an error if an atomic block is active."""
        if self.in_atomic_block:
            raise TransactionManagementError(
                "This is forbidden when an 'atomic' block is active."
            )

    def validate_no_broken_transaction(self):
        if self.needs_rollback:
            raise TransactionManagementError(
                "An error occurred in the current transaction. You can't "
                "execute queries until the end of the 'atomic' block."
            )

    # ##### Foreign key constraints checks handling #####

    @contextmanager
    def constraint_checks_disabled(self):
        """
        Disable foreign key constraint checking.
        """
        disabled = self.disable_constraint_checking()
        try:
            yield
        finally:
            if disabled:
                self.enable_constraint_checking()

    def disable_constraint_checking(self):
        """
        Backends can implement as needed to temporarily disable foreign key
        constraint checking. Should return True if the constraints were
        disabled and will need to be reenabled.
        """
        return False

    def enable_constraint_checking(self):
        """
        Backends can implement as needed to re-enable foreign key constraint
        checking.
        """
        pass

    def check_constraints(self, table_names=None):
        """
        Backends can override this method if they can apply constraint
        checking (e.g. via "SET CONSTRAINTS ALL IMMEDIATE"). Should raise an
        IntegrityError if any invalid foreign key references are encountered.
        """
        pass

    # ##### Connection termination handling #####

    def is_usable(self):
        """
        Test if the database connection is usable.

        This method may assume that self.connection is not None.

        Actual implementations should take care not to raise exceptions
        as that may prevent Django from recycling unusable connections.
        """
        raise NotImplementedError(
            "subclasses of BaseDatabaseWrapper may require an is_usable() method"
        )

    def close_if_health_check_failed(self):
        """Close existing connection if it fails a health check."""
        if (
            self.connection is None
            or not self.health_check_enabled
            or self.health_check_done
        ):
            return

        if not self.is_usable():
            self.close()
        self.health_check_done = True
```
### 6 - django/db/transaction.py:

Start line: 315, End line: 340

```python
def atomic(using=None, savepoint=True, durable=False):
    # Bare decorator: @atomic -- although the first argument is called
    # `using`, it's actually the function being decorated.
    if callable(using):
        return Atomic(DEFAULT_DB_ALIAS, savepoint, durable)(using)
    # Decorator: @atomic(...) or context manager: with atomic(...): ...
    else:
        return Atomic(using, savepoint, durable)


def _non_atomic_requests(view, using):
    try:
        view._non_atomic_requests.add(using)
    except AttributeError:
        view._non_atomic_requests = {using}
    return view


def non_atomic_requests(using=None):
    if callable(using):
        return _non_atomic_requests(using, DEFAULT_DB_ALIAS)
    else:
        if using is None:
            using = DEFAULT_DB_ALIAS
        return lambda view: _non_atomic_requests(view, using)
```
### 7 - django/contrib/postgres/signals.py:

Start line: 37, End line: 69

```python
def register_type_handlers(connection, **kwargs):
    if connection.vendor != "postgresql" or connection.alias == NO_DB_ALIAS:
        return

    try:
        oids, array_oids = get_hstore_oids(connection.alias)
        register_hstore(
            connection.connection, globally=True, oid=oids, array_oid=array_oids
        )
    except ProgrammingError:
        # Hstore is not available on the database.
        #
        # If someone tries to create an hstore field it will error there.
        # This is necessary as someone may be using PSQL without extensions
        # installed but be using other features of contrib.postgres.
        #
        # This is also needed in order to create the connection in order to
        # install the hstore extension.
        pass

    try:
        citext_oids = get_citext_oids(connection.alias)
        array_type = psycopg2.extensions.new_array_type(
            citext_oids, "citext[]", psycopg2.STRING
        )
        psycopg2.extensions.register_type(array_type, None)
    except ProgrammingError:
        # citext is not available on the database.
        #
        # The same comments in the except block of the above call to
        # register_hstore() also apply here.
        pass
```
### 8 - django/db/transaction.py:

Start line: 136, End line: 179

```python
#################################
# Decorators / context managers #
#################################


class Atomic(ContextDecorator):
    """
    Guarantee the atomic execution of a given block.

    An instance can be used either as a decorator or as a context manager.

    When it's used as a decorator, __call__ wraps the execution of the
    decorated function in the instance itself, used as a context manager.

    When it's used as a context manager, __enter__ creates a transaction or a
    savepoint, depending on whether a transaction is already in progress, and
    __exit__ commits the transaction or releases the savepoint on normal exit,
    and rolls back the transaction or to the savepoint on exceptions.

    It's possible to disable the creation of savepoints if the goal is to
    ensure that some code runs within a transaction without creating overhead.

    A stack of savepoints identifiers is maintained as an attribute of the
    connection. None denotes the absence of a savepoint.

    This allows reentrancy even if the same AtomicWrapper is reused. For
    example, it's possible to define `oa = atomic('other')` and use `@oa` or
    `with oa:` multiple times.

    Since database connections are thread-local, this is thread-safe.

    An atomic block can be tagged as durable. In this case, raise a
    RuntimeError if it's nested within another atomic block. This guarantees
    that database changes in a durable block are committed to the database when
    the block exists without error.

    This is a private API.
    """

    def __init__(self, using, savepoint, durable):
        self.using = using
        self.savepoint = savepoint
        self.durable = durable
        self._from_testcase = False
```
### 9 - django/db/backends/base/base.py:

Start line: 458, End line: 491

```python
class BaseDatabaseWrapper:

    def set_autocommit(
        self, autocommit, force_begin_transaction_with_broken_autocommit=False
    ):
        """
        Enable or disable autocommit.

        The usual way to start a transaction is to turn autocommit off.
        SQLite does not properly start a transaction when disabling
        autocommit. To avoid this buggy behavior and to actually enter a new
        transaction, an explicit BEGIN is required. Using
        force_begin_transaction_with_broken_autocommit=True will issue an
        explicit BEGIN with SQLite. This option will be ignored for other
        backends.
        """
        self.validate_no_atomic_block()
        self.close_if_health_check_failed()
        self.ensure_connection()

        start_transaction_under_autocommit = (
            force_begin_transaction_with_broken_autocommit
            and not autocommit
            and hasattr(self, "_start_transaction_under_autocommit")
        )

        if start_transaction_under_autocommit:
            self._start_transaction_under_autocommit()
        else:
            self._set_autocommit(autocommit)

        self.autocommit = autocommit

        if autocommit and self.run_commit_hooks_on_set_autocommit_on:
            self.run_and_clear_commit_hooks()
            self.run_commit_hooks_on_set_autocommit_on = False
```
### 10 - django/db/backends/base/features.py:

Start line: 222, End line: 357

```python
class BaseDatabaseFeatures:
    minimum_database_version =
    # Defaults to False to allow third-party backends to opt-in.
    can_clone_databases = False

    # Does the backend consider table names with different casing to
    # be equal?
    ignores_table_name_case = False

    # Place FOR UPDATE right after FROM clause. Used on MSSQL.
    for_update_after_from = False

    # Combinatorial flags
    supports_select_union = True
    supports_select_intersection = True
    supports_select_difference = True
    supports_slicing_ordering_in_compound = False
    supports_parentheses_in_compound = True

    # Does the database support SQL 2003 FILTER (WHERE ...) in aggregate
    # expressions?
    supports_aggregate_filter_clause = False

    # Does the backend support indexing a TextField?
    supports_index_on_text_field = True

    # Does the backend support window expressions (expression OVER (...))?
    supports_over_clause = False
    supports_frame_range_fixed_distance = False
    only_supports_unbounded_with_preceding_and_following = False

    # Does the backend support CAST with precision?
    supports_cast_with_precision = True

    # How many second decimals does the database return when casting a value to
    # a type with time?
    time_cast_precision = 6

    # SQL to create a procedure for use by the Django test suite. The
    # functionality of the procedure isn't important.
    create_test_procedure_without_params_sql = None
    create_test_procedure_with_int_param_sql = None

    # SQL to create a table with a composite primary key for use by the Django
    # test suite.
    create_test_table_with_composite_primary_key = None

    # Does the backend support keyword parameters for cursor.callproc()?
    supports_callproc_kwargs = False

    # What formats does the backend EXPLAIN syntax support?
    supported_explain_formats = set()

    # Does the backend support the default parameter in lead() and lag()?
    supports_default_in_lead_lag = True

    # Does the backend support ignoring constraint or uniqueness errors during
    # INSERT?
    supports_ignore_conflicts = True
    # Does the backend support updating rows on constraint or uniqueness errors
    # during INSERT?
    supports_update_conflicts = False
    supports_update_conflicts_with_target = False

    # Does this backend require casting the results of CASE expressions used
    # in UPDATE statements to ensure the expression has the correct type?
    requires_casted_case_in_updates = False

    # Does the backend support partial indexes (CREATE INDEX ... WHERE ...)?
    supports_partial_indexes = True
    supports_functions_in_partial_indexes = True
    # Does the backend support covering indexes (CREATE INDEX ... INCLUDE ...)?
    supports_covering_indexes = False
    # Does the backend support indexes on expressions?
    supports_expression_indexes = True
    # Does the backend treat COLLATE as an indexed expression?
    collate_as_index_expression = False

    # Does the database allow more than one constraint or index on the same
    # field(s)?
    allows_multiple_constraints_on_same_fields = True

    # Does the backend support boolean expressions in SELECT and GROUP BY
    # clauses?
    supports_boolean_expr_in_select_clause = True

    # Does the backend support JSONField?
    supports_json_field = True
    # Can the backend introspect a JSONField?
    can_introspect_json_field = True
    # Does the backend support primitives in JSONField?
    supports_primitives_in_json_field = True
    # Is there a true datatype for JSON?
    has_native_json_field = False
    # Does the backend use PostgreSQL-style JSON operators like '->'?
    has_json_operators = False
    # Does the backend support __contains and __contained_by lookups for
    # a JSONField?
    supports_json_field_contains = True
    # Does value__d__contains={'f': 'g'} (without a list around the dict) match
    # {'d': [{'f': 'g'}]}?
    json_key_contains_list_matching_requires_list = False
    # Does the backend support JSONObject() database function?
    has_json_object_function = True

    # Does the backend support column collations?
    supports_collation_on_charfield = True
    supports_collation_on_textfield = True
    # Does the backend support non-deterministic collations?
    supports_non_deterministic_collations = True

    # Does the backend support the logical XOR operator?
    supports_logical_xor = False

    # Collation names for use by the Django test suite.
    test_collations = {
        "ci": None,  # Case-insensitive.
        "cs": None,  # Case-sensitive.
        "non_default": None,  # Non-default.
        "swedish_ci": None,  # Swedish case-insensitive.
    }
    # SQL template override for tests.aggregation.tests.NowUTC
    test_now_utc_template = None

    # A set of dotted paths to tests in Django's test suite that are expected
    # to fail on this database.
    django_test_expected_failures = set()
    # A map of reasons to sets of dotted paths to tests in Django's test suite
    # that should be skipped for this database.
    django_test_skips = {}

    def __init__(self, connection):
        self.connection = connection

    @cached_property
    def supports_explaining_query_execution(self):
        """Does this backend support explaining query execution?"""
        return self.connection.ops.explain_prefix is not None
    # ... other code
```
### 12 - django/db/transaction.py:

Start line: 223, End line: 312

```python
class Atomic(ContextDecorator):

    def __exit__(self, exc_type, exc_value, traceback):
        connection = get_connection(self.using)

        if connection.in_atomic_block:
            connection.atomic_blocks.pop()

        if connection.savepoint_ids:
            sid = connection.savepoint_ids.pop()
        else:
            # Prematurely unset this flag to allow using commit or rollback.
            connection.in_atomic_block = False

        try:
            if connection.closed_in_transaction:
                # The database will perform a rollback by itself.
                # Wait until we exit the outermost block.
                pass

            elif exc_type is None and not connection.needs_rollback:
                if connection.in_atomic_block:
                    # Release savepoint if there is one
                    if sid is not None:
                        try:
                            connection.savepoint_commit(sid)
                        except DatabaseError:
                            try:
                                connection.savepoint_rollback(sid)
                                # The savepoint won't be reused. Release it to
                                # minimize overhead for the database server.
                                connection.savepoint_commit(sid)
                            except Error:
                                # If rolling back to a savepoint fails, mark for
                                # rollback at a higher level and avoid shadowing
                                # the original exception.
                                connection.needs_rollback = True
                            raise
                else:
                    # Commit transaction
                    try:
                        connection.commit()
                    except DatabaseError:
                        try:
                            connection.rollback()
                        except Error:
                            # An error during rollback means that something
                            # went wrong with the connection. Drop it.
                            connection.close()
                        raise
            else:
                # This flag will be set to True again if there isn't a savepoint
                # allowing to perform the rollback at this level.
                connection.needs_rollback = False
                if connection.in_atomic_block:
                    # Roll back to savepoint if there is one, mark for rollback
                    # otherwise.
                    if sid is None:
                        connection.needs_rollback = True
                    else:
                        try:
                            connection.savepoint_rollback(sid)
                            # The savepoint won't be reused. Release it to
                            # minimize overhead for the database server.
                            connection.savepoint_commit(sid)
                        except Error:
                            # If rolling back to a savepoint fails, mark for
                            # rollback at a higher level and avoid shadowing
                            # the original exception.
                            connection.needs_rollback = True
                else:
                    # Roll back transaction
                    try:
                        connection.rollback()
                    except Error:
                        # An error during rollback means that something
                        # went wrong with the connection. Drop it.
                        connection.close()

        finally:
            # Outermost block exit when autocommit was enabled.
            if not connection.in_atomic_block:
                if connection.closed_in_transaction:
                    connection.connection = None
                else:
                    connection.set_autocommit(True)
            # Outermost block exit when autocommit was disabled.
            elif not connection.savepoint_ids and not connection.commit_on_exit:
                if connection.closed_in_transaction:
                    connection.connection = None
                else:
                    connection.in_atomic_block = False
```
### 17 - django/db/transaction.py:

Start line: 181, End line: 221

```python
class Atomic(ContextDecorator):

    def __enter__(self):
        connection = get_connection(self.using)

        if (
            self.durable
            and connection.atomic_blocks
            and not connection.atomic_blocks[-1]._from_testcase
        ):
            raise RuntimeError(
                "A durable atomic block cannot be nested within another "
                "atomic block."
            )
        if not connection.in_atomic_block:
            # Reset state when entering an outermost atomic block.
            connection.commit_on_exit = True
            connection.needs_rollback = False
            if not connection.get_autocommit():
                # Pretend we're already in an atomic block to bypass the code
                # that disables autocommit to enter a transaction, and make a
                # note to deal with this case in __exit__.
                connection.in_atomic_block = True
                connection.commit_on_exit = False

        if connection.in_atomic_block:
            # We're already in a transaction; create a savepoint, unless we
            # were told not to or we're already waiting for a rollback. The
            # second condition avoids creating useless savepoints and prevents
            # overwriting needs_rollback until the rollback is performed.
            if self.savepoint and not connection.needs_rollback:
                sid = connection.savepoint()
                connection.savepoint_ids.append(sid)
            else:
                connection.savepoint_ids.append(None)
        else:
            connection.set_autocommit(
                False, force_begin_transaction_with_broken_autocommit=True
            )
            connection.in_atomic_block = True

        if connection.in_atomic_block:
            connection.atomic_blocks.append(self)
```
### 19 - django/db/transaction.py:

Start line: 1, End line: 82

```python
from contextlib import ContextDecorator, contextmanager

from django.db import (
    DEFAULT_DB_ALIAS,
    DatabaseError,
    Error,
    ProgrammingError,
    connections,
)


class TransactionManagementError(ProgrammingError):
    """Transaction management is used improperly."""

    pass


def get_connection(using=None):
    """
    Get a database connection by name, or the default database connection
    if no name is provided. This is a private API.
    """
    if using is None:
        using = DEFAULT_DB_ALIAS
    return connections[using]


def get_autocommit(using=None):
    """Get the autocommit status of the connection."""
    return get_connection(using).get_autocommit()


def set_autocommit(autocommit, using=None):
    """Set the autocommit status of the connection."""
    return get_connection(using).set_autocommit(autocommit)


def commit(using=None):
    """Commit a transaction."""
    get_connection(using).commit()


def rollback(using=None):
    """Roll back a transaction."""
    get_connection(using).rollback()


def savepoint(using=None):
    """
    Create a savepoint (if supported and required by the backend) inside the
    current transaction. Return an identifier for the savepoint that will be
    used for the subsequent rollback or commit.
    """
    return get_connection(using).savepoint()


def savepoint_rollback(sid, using=None):
    """
    Roll back the most recent savepoint (if one exists). Do nothing if
    savepoints are not supported.
    """
    get_connection(using).savepoint_rollback(sid)


def savepoint_commit(sid, using=None):
    """
    Commit the most recent savepoint (if one exists). Do nothing if
    savepoints are not supported.
    """
    get_connection(using).savepoint_commit(sid)


def clean_savepoints(using=None):
    """
    Reset the counter used to generate unique savepoint ids in this thread.
    """
    get_connection(using).clean_savepoints()


def get_rollback(using=None):
    """Get the "needs rollback" flag -- for *advanced use* only."""
    return get_connection(using).get_rollback()
```
### 33 - django/db/transaction.py:

Start line: 85, End line: 97

```python
def set_rollback(rollback, using=None):
    """
    Set or unset the "needs rollback" flag -- for *advanced use* only.

    When `rollback` is `True`, trigger a rollback when exiting the innermost
    enclosing atomic block that has `savepoint=True` (that's the default). Use
    this to force a rollback without raising an exception.

    When `rollback` is `False`, prevent such a rollback. Use this only after
    rolling back to a known-good state! Otherwise, you break the atomic block
    and data corruption may occur.
    """
    return get_connection(using).set_rollback(rollback)
```
### 34 - django/db/backends/base/base.py:

Start line: 1, End line: 36

```python
import _thread
import copy
import datetime
import threading
import time
import warnings
from collections import deque
from contextlib import contextmanager

try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DEFAULT_DB_ALIAS, DatabaseError, NotSupportedError
from django.db.backends import utils
from django.db.backends.base.validation import BaseDatabaseValidation
from django.db.backends.signals import connection_created
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseErrorWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property

NO_DB_ALIAS = "__no_db__"
RAN_DB_VERSION_CHECK = set()


# RemovedInDjango50Warning
def timezone_constructor(tzname):
    if settings.USE_DEPRECATED_PYTZ:
        import pytz

        return pytz.timezone(tzname)
    return zoneinfo.ZoneInfo(tzname)
```
### 74 - django/db/backends/base/base.py:

Start line: 617, End line: 649

```python
class BaseDatabaseWrapper:

    # ##### Thread safety handling #####

    @property
    def allow_thread_sharing(self):
        with self._thread_sharing_lock:
            return self._thread_sharing_count > 0

    def inc_thread_sharing(self):
        with self._thread_sharing_lock:
            self._thread_sharing_count += 1

    def dec_thread_sharing(self):
        with self._thread_sharing_lock:
            if self._thread_sharing_count <= 0:
                raise RuntimeError(
                    "Cannot decrement the thread sharing count below zero."
                )
            self._thread_sharing_count -= 1

    def validate_thread_sharing(self):
        if not (self.allow_thread_sharing or self._thread_ident == _thread.get_ident()):
            raise DatabaseError(
                "DatabaseWrapper objects created in a "
                "thread can only be used in that same thread. The object "
                "with alias '%s' was created in thread id %s and this is "
                "thread id %s." % (self.alias, self._thread_ident, _thread.get_ident())
            )
```
