import pytest
from ghostcoder.schema import Folder, File

def test_generate_file_tree():
    # Create a Folder and File structure
    file1 = File(path="path/to/file1", language="python", name="file1.py")
    file2 = File(path="path/to/file2", language="python", name="file2.py")
    folder1 = Folder(name="folder1", path="path/to/folder1", children=[file1, file2])
    file3 = File(path="path", language="python", name="file3.py")
    root = Folder(name="root", path="path", children=[folder1, file3])

    file_tree = root.tree_string()

    expected_file_tree = (
        "folder1/\n"
        "  file1.py\n"
        "  file2.py\n"
        "file3.py\n"
    )

    print(file_tree)

    assert file_tree == expected_file_tree
