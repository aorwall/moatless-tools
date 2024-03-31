import json

from moatless.coder.code_writer import CodeWriter, WriteCodeRequest, FileItem


def test_pytest_dev__pytest_5808():
    code_writer = CodeWriter(expect_complete_functions=False, debug_mode=True)

    with open("data/python/pytest-dev__pytest-5808/original_pastebin.py", "r") as f:
        existing_content = f.read()

    with open("data/python/pytest-dev__pytest-5808/updated_pastebin.py", "r") as f:
        updated_content = f.read()

    request = WriteCodeRequest(
        updated_files=[
            FileItem(file_path="pastebin.py", content=updated_content)
        ],
        file_context=[
            FileItem(file_path="pastebin.py", content=existing_content)
        ]
    )

    resonse = code_writer.write_code(request)

    try:
        print(json.dumps(resonse.dict(), indent=2))
    except Exception as e:
        print(e)
