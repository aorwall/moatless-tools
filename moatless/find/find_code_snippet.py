import os

ignored_dirs = ["target", "node_modules", ".git", ".idea"]


def find_code_snippet_in_files(repo_dir: str, code_snippet: str):
    occurrences = []

    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if any(dir in root for dir in ignored_dirs):
                continue

            file_path = os.path.join(root, file)
            if not file_path.endswith(".java"):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, start=1):
                        if code_snippet.lower() in line.lower():
                            relative_path = os.path.relpath(file_path, repo_dir)
                            occurrences.append(
                                (
                                    relative_path,
                                    line_number,
                                    line.strip(),
                                )
                            )
            except Exception as e:
                if "invalid" not in str(e):
                    print(f"Could not read file {file_path}: {e}")

    return occurrences
