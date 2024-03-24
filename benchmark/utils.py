import re


def set_end_line(current_diff):
    if current_diff["last_minus_line"] == 0:
        del current_diff["end_line_old"]
    else:
        current_diff["end_line_old"] = current_diff["start_line_old"] + current_diff["last_minus_line"] - 1


def diff_details(text: str):
    lines = text.split('\n')
    diffs = {}
    file_path = None
    for line in lines:
        if line.startswith('diff --git'):
            if file_path in diffs and diffs[file_path]["diffs"]:
                set_end_line(diffs[file_path]["diffs"][-1])

            file_path = line.split(' ')[2][2:]  # Extract file name after 'b/'
            diffs[file_path] = {"diffs": []}
        elif line.startswith('@@'):
            if file_path in diffs and diffs[file_path]["diffs"]:
                set_end_line(diffs[file_path]["diffs"][-1])

            # Extract the start line and size for old file from the chunk info
            match = re.search(r'\-(\d+),(\d+)', line)
            if match:
                start_line_old, size_old = match.groups()
                # Initialize tracking for the current diff chunk
                diffs[file_path]["diffs"].append({
                    "start_line_old": int(start_line_old),
                    "end_line_old": int(start_line_old) + int(size_old) - 1,  # Temp value, to be adjusted
                    "lines_before_first_minus": 0,  # Count lines before the first '-'
                    "lines_until_last_minus": 0,
                    "last_minus_line": 0  # Track the last '-' line within the chunk
                })
        elif file_path and diffs[file_path]["diffs"]:
            current_diff = diffs[file_path]["diffs"][-1]
            if line.startswith('-'):
                # If it's the first '-' line, calculate the real start_line_old
                if current_diff["last_minus_line"] == 0:
                    current_diff["start_line_old"] += current_diff["lines_before_first_minus"]
                current_diff["last_minus_line"] = current_diff["lines_until_last_minus"]
            elif not line.startswith('+'):
                # Count non-modified and non-added lines before the first '-' and after the last '-'
                if current_diff["last_minus_line"] == 0:
                    current_diff["lines_before_first_minus"] += 1
                current_diff["lines_until_last_minus"] += 1

    if file_path in diffs and diffs[file_path]["diffs"]:
        set_end_line(diffs[file_path]["diffs"][-1])

    # Final adjustments: remove temporary tracking keys
    for file_path, details in diffs.items():
        for diff in details["diffs"]:
            del diff["lines_before_first_minus"]
            del diff["lines_until_last_minus"]
            del diff["last_minus_line"]
    return diffs


if __name__ == "__main__":
    text = """diff --git a/pydicom/valuerep.py b/pydicom/valuerep.py
--- a/pydicom/valuerep.py
+++ b/pydicom/valuerep.py
@@ -1,6 +1,5 @@
 # Copyright 2008-2018 pydicom authors. See LICENSE file for details.
 \"""Special classes for DICOM value representations (VR)\"""
-from copy import deepcopy
 from decimal import Decimal
 import re

@@ -750,6 +749,25 @@ def __ne__(self, other):
     def __str__(self):
         return '='.join(self.components).__str__()

+    def __next__(self):
+        # Get next character or stop iteration
+        if self._i < self._rep_len:
+            c = self._str_rep[self._i]
+            self._i += 1
+            return c
+        else:
+            raise StopIteration
+
+    def __iter__(self):
+        # Get string rep. and length, initialize index counter
+        self._str_rep = self.__str__()
+        self._rep_len = len(self._str_rep)
+        self._i = 0
+        return self
+
+    def __contains__(self, x):
+        return x in self.__str__()
+
     def __repr__(self):
         return '='.join(self.components).__repr__()

"""

    import json
    diff = diff_details(text)
    print(json.dumps(diff, indent=2))