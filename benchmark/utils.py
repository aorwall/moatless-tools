import re


def diff_details(text: str):
    lines = text.split('\n')
    diffs = {}
    file_path = None
    for line in lines:
        if line.startswith('diff --git'):
            file_path = line.split(' ')[2][2:]  # Extract file name after 'b/'
            diffs[file_path] = {"diffs": []}
        elif line.startswith('@@'):
            # Extract the start line and size for old file from the chunk info
            match = re.search(r'\-(\d+),(\d+)', line)
            if match:
                start_line_old, size_old = match.groups()
                # Initialize tracking for the current diff chunk
                diffs[file_path]["diffs"].append({
                    "start_line_old": int(start_line_old),
                    "lines_until_first_change": 0
                })
        elif file_path and diffs[file_path]["diffs"]:
            current_diff = diffs[file_path]["diffs"][-1]

            if (line.startswith('+') or line.startswith('-')) and "lines_until_first_change" in current_diff:
                current_diff["start_line_old"] += current_diff["lines_until_first_change"]
                del current_diff["lines_until_first_change"]
            elif "lines_until_first_change" in current_diff:
                current_diff["lines_until_first_change"] += 1

            if line.startswith('-'):
                if "lines_until_last_minus" not in current_diff:
                    current_diff["lines_until_last_minus"] = 0
                else:
                    current_diff["lines_until_last_minus"] += 1

                current_diff["end_line_old"] = current_diff["start_line_old"] + current_diff["lines_until_last_minus"]
            elif not line.startswith('+') and "lines_until_last_minus" in current_diff:
                current_diff["lines_until_last_minus"] += 1

    # Final adjustments: remove temporary tracking keys
    for file_path, details in diffs.items():
        for diff in details["diffs"]:
            diff.pop("lines_until_last_minus", None)
            if "end_line_old" not in diff:
                diff["end_line_old"] = diff["start_line_old"]
    return diffs


if __name__ == "__main__":
    text = """diff --git a/django/contrib/auth/migrations/0011_update_proxy_permissions.py b/django/contrib/auth/migrations/0011_update_proxy_permissions.py
--- a/django/contrib/auth/migrations/0011_update_proxy_permissions.py
+++ b/django/contrib/auth/migrations/0011_update_proxy_permissions.py
@@ -1,5 +1,18 @@
-from django.db import migrations
+import sys
+
+from django.core.management.color import color_style
+from django.db import migrations, transaction
 from django.db.models import Q
+from django.db.utils import IntegrityError
+
+WARNING = \"""
+    A problem arose migrating proxy model permissions for {old} to {new}.
+
+      Permission(s) for {new} already existed.
+      Codenames Q: {query}
+
+    Ensure to audit ALL permissions for {old} and {new}.
+\"""
 
 
 def update_proxy_model_permissions(apps, schema_editor, reverse=False):
@@ -7,6 +20,7 @@ def update_proxy_model_permissions(apps, schema_editor, reverse=False):
     Update the content_type of proxy model permissions to use the ContentType
     of the proxy model.
     \"""
+    style = color_style()
     Permission = apps.get_model('auth', 'Permission')
     ContentType = apps.get_model('contenttypes', 'ContentType')
     for Model in apps.get_models():
@@ -24,10 +38,16 @@ def update_proxy_model_permissions(apps, schema_editor, reverse=False):
         proxy_content_type = ContentType.objects.get_for_model(Model, for_concrete_model=False)
         old_content_type = proxy_content_type if reverse else concrete_content_type
         new_content_type = concrete_content_type if reverse else proxy_content_type
-        Permission.objects.filter(
-            permissions_query,
-            content_type=old_content_type,
-        ).update(content_type=new_content_type)
+        try:
+            with transaction.atomic():
+                Permission.objects.filter(
+                    permissions_query,
+                    content_type=old_content_type,
+                ).update(content_type=new_content_type)
+        except IntegrityError:
+            old = '{}_{}'.format(old_content_type.app_label, old_content_type.model)
+            new = '{}_{}'.format(new_content_type.app_label, new_content_type.model)
+            sys.stdout.write(style.WARNING(WARNING.format(old=old, new=new, query=permissions_query)))


 def revert_proxy_model_permissions(apps, schema_editor):

"""

    import json
    diff = diff_details(text)
    print(json.dumps(diff, indent=2))