import re


def diff_details(text: str):
    lines = text.split('\n')
    diffs = {}

    file_path = None

    for line in lines:
        if line.startswith('diff --git'):
            file_path = line.split(' ')[2][2:]  # Extract file name after 'b/'
            if file_path not in diffs:
                diffs[file_path] = { "diffs": [] }

        elif line.startswith('@@'):
            match = re.search(r'\-(\d+),(\d+)\s\+(\d+),(\d+)', line)
            if match:
                start_line_old, size_old, start_line_new, size_new = match.groups()
                end_line_old = int(start_line_old) + int(size_old) - 1
                end_line_new = int(start_line_new) + int(size_new) - 1
                diffs[file_path]["diffs"].append({
                    "start_line_old": int(start_line_old),
                    "end_line_old": end_line_old,
                    "start_line_new": int(start_line_new),
                    "end_line_new": end_line_new
                })

    return diffs



if __name__ == "__main__":
    text = """diff --git a/astroid/nodes/node_classes.py b/astroid/nodes/node_classes.py
--- a/astroid/nodes/node_classes.py
+++ b/astroid/nodes/node_classes.py
@@ -2346,24 +2346,33 @@ def itered(self):
         \"""
         return [key for (key, _) in self.items]
 
-    def getitem(self, index, context=None):
+    def getitem(
+        self, index: Const | Slice, context: InferenceContext | None = None
+    ) -> NodeNG:
         \"""Get an item from this node.
 
         :param index: The node to use as a subscript index.
-        :type index: Const or Slice
 
         :raises AstroidTypeError: When the given index cannot be used as a
             subscript index, or if this node is not subscriptable.
         :raises AstroidIndexError: If the given index does not exist in the
             dictionary.
         \"""
+        # pylint: disable-next=import-outside-toplevel; circular import
+        from astroid.helpers import safe_infer
+
         for key, value in self.items:
             # TODO(cpopa): no support for overriding yet, {1:2, **{1: 3}}.
             if isinstance(key, DictUnpack):
+                inferred_value = safe_infer(value, context)
+                if not isinstance(inferred_value, Dict):
+                    continue
+
                 try:
-                    return value.getitem(index, context)
+                    return inferred_value.getitem(index, context)
                 except (AstroidTypeError, AstroidIndexError):
                     continue
+
             for inferredkey in key.infer(context):
                 if inferredkey is util.Uninferable:
                     continue

"""

    import json
    diff = diff_details(text)
    print(json.dumps(diff, indent=2))