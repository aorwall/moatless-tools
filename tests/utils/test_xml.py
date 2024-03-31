from moatless.utils.xml import extract_between_tags, contains_tag

message = """Some files
<file>src/_pytest/pastebin.py</file>
<file>src/foo.py</file>

Blablabla

<finished>"""


def test_extract_between_tags_from_message():
    files = extract_between_tags("file", message, strip=True)
    assert files == ["src/_pytest/pastebin.py", "src/foo.py"]


def test_message_contains_tag():
    assert contains_tag("finished", message)
    assert not contains_tag("finished", "nope")
