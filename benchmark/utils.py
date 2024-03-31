import logging
import os
import re
import sys
from typing import List

from llama_index.core import get_tokenizer

from moatless.retriever import CodeSnippet


logger = logging.getLogger(__name__)


def write_file(path, text):
    with open(path, 'w') as f:
        f.write(text)
        print(f"File '{path}' was saved", file=sys.stderr)


def write_json(path, name, data):
    json_str = json.dumps(data, indent=2)
    json_path = f"{path}/{name}.json"
    write_file(json_path, json_str)


def diff_file_names(text: str) -> list[str]:
    return [
        line[len("+++ b/"):]
        for line in text.split('\n')
        if line.startswith('+++')
    ]


def recall_report(row_data: dict, code_snippets: List[CodeSnippet], repo_path: str):
    report = {}
    report["instance_id"] = row_data["instance_id"]

    patch_files = row_data["patch_files"]
    no_of_patches = len(patch_files)

    found_patch_files = set()
    found_snippets = 0

    top_file_pos = None
    file_pos = 0
    min_pos = None
    sum_pos = 0

    sum_tokens = 0
    sum_file_tokens = 0

    snippet_reports = []
    file_reports = {}

    for i, snippet in enumerate(code_snippets):
        snippet_pos = i+1

        tokenizer = get_tokenizer()
        tokens = len(tokenizer(snippet.content))

        sum_tokens += tokens

        file_report = file_reports.get(snippet.file_path)
        if not file_report:
            file_pos += 1

            file_report = {
                "file_path": snippet.file_path,
                "position": file_pos
            }

            try:
                with open(f"{repo_path}/{snippet.file_path}", "r") as f:
                    file_tokens = len(tokenizer(f.read()))
                    sum_file_tokens += file_tokens
                    file_report["tokens"] = file_tokens
                    file_report["context_length"] = sum_file_tokens
            except Exception as e:
                logger.info(f"Failed to read file: {e}")
                file_report["error"] = str(e)
                file_report["snippet_id"] = snippet.id

            file_reports[snippet.file_path] = file_report

        file_hit = snippet.file_path in row_data["patch_diff_details"]

        snippet_report = {
            "position": snippet_pos,
            "id": snippet.id,
            "distance": snippet.distance,
            "file_path": snippet.file_path,
            "start_line": snippet.start_line,
            "end_line": snippet.end_line,
            "tokens": tokens,
            "context_length": sum_tokens,
        }
        snippet_reports.append(snippet_report)

        if file_hit:
            if not top_file_pos:
                top_file_pos = file_pos

            snippet_report["file_pos"] = file_pos

            found_patch_files.add(snippet.file_path)
            diffs = row_data["patch_diff_details"][snippet.file_path]
            for diff in diffs["diffs"]:
                if "file_pos" not in diff:
                    diff["file_pos"] = file_pos
                    diff["file_context_length"] = file_report["context_length"]

                if snippet.start_line and (snippet.start_line <= diff['start_line_old'] <= diff.get('end_line_old', diff['start_line_old']) <= snippet.end_line):
                    found_snippets += 1
                    diff["pos"] = snippet_pos
                    diff["context_length"] = sum_tokens

                    if "closest_snippet" in diff:
                        del diff["closest_snippet"]
                        del diff["closest_snippet_line_distance"]

                elif "pos" not in diff and snippet.start_line and snippet.end_line:
                    line_distance = min(abs(snippet.start_line - diff['start_line_old']), abs(snippet.end_line - diff.get('end_line_old', diff['start_line_old'])))

                    if "closest_snippet" not in diff or line_distance < diff["line_distance"]:
                        diff["closest_snippet_id"] = snippet.id
                        diff["closest_snippet_line_distance"] = line_distance

    report["patch_diff_details"] = row_data["patch_diff_details"]
    report["snippets"] = snippet_reports
    report["files"] = list(file_reports.values())

    return report


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