from code_blocks.merger import CodeMerger


def verify_merge(original_file, updated_file, expected_file, language):
    with open(original_file, 'r') as f:
        original_content = f.read()

    with open(updated_file, 'r') as f:
        updated_content = f.read()

    with open(expected_file, 'r') as f:
        expected_content = f.read()

    merger = CodeMerger(language)
    merged = merger.merge(original_content, updated_content)
    assert merged == expected_content


def test_merge_java_insert_after_comment():
    verify_merge(
        "java/calculator.java",
        "java/calculator_insert1.java",
        "java/calculator_merged1.java",
        "java")


def test_merge_java_insert_before_comment():
    verify_merge(
        "java/calculator.java",
        "java/calculator_insert2.java",
        "java/calculator_merged2.java",
        "java")

def test_merge_java_replace():
    verify_merge(
        "java/calculator.java",
        "java/calculator_replace.java",
        "java/calculator_replace.java",
        "java")

def test_merge_python_insert_after_comment():
    verify_merge(
        "python/calculator.py",
        "python/calculator_insert1.py",
        "python/calculator_merged1.py",
        "python")

def test_merge_python_insert_before_comment():
    verify_merge(
        "python/calculator.py",
        "python/calculator_insert2.py",
        "python/calculator_merged2.py",
        "python")

def test_merge_python_replace():
    verify_merge(
        "python/calculator.py",
        "python/calculator_replace.py",
        "python/calculator_replace.py",
        "python")