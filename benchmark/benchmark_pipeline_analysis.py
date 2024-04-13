import json

from litellm import completion

from moatless.codeblocks.parser.python import PythonParser
from moatless.utils.repo import setup_github_repo


def extract_text_including_first_separator(
    text, separator="=========================================="
):
    separator_position = text.find(separator)
    if separator_position != -1:
        return text[separator_position:]
    else:
        return text


def llm_analysis(report_dir: str, instance_id: str, repo_dir: str = "/tmp/repos"):
    result_file = f"{dir}/{instance_id}/result.json"
    with open(result_file, "r") as f:
        result = json.load(f)

    repo_path = setup_github_repo(
        repo=result["repo"], base_commit=result["base_commit"], base_dir=repo_dir
    )

    if result["patch_files"]:
        patch_file = result["patch_files"][0]  # TODO: Only suppporting one file ATM

        parser = PythonParser()

        with open(f"{repo_path}/{patch_file}", "r") as f:
            content = f.read()

        codeblock = parser.parse(content)

    text = f"""# Problem statement
{result['problem_statement']}

# Patch
```diff
{result['model_patch']}
```

# Golden patch
```diff
{result['expected_patch']}
```
"""

    tests = extract_text_including_first_separator(result["test_logs"])

    if tests:
        text += f"""# Test results
{tests}

"""

    text += """# Instructions
You're provided with a "problem statement" of a issue in a git repository and a patch that was supposed to solve the issue. The "Golden patch" is a patch that works. 
Compare the patches and identify why the patch didn't work.

Make the following classifications:

* Missing change: If a change an expected change of the file is missing in the Patch
* Wrong part of code: If the wrong section (like wrong class, function etc) of the file is changed
* Detected by test: If the change is detected by the test
"""

    message = {"role": "user", "content": text}

    llm_response = completion(
        model="gpt-4-0125-preview", temperature=0.0, messages=[message]
    )

    print(llm_response["choices"][0]["message"]["content"])
