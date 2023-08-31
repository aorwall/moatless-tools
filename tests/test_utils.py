
from ghostcoder.utils import is_complete, extract_code_from_text


def test_is_complete():
    assert is_complete("print('Hello, World!')")
    assert is_complete("# a comment\nprint('Hello, World!')")
    assert is_complete("# TODO: Find a way to let the LLM verify the existing file against the suggested updates")
    assert is_complete("Pet existingPet = owner.getPet(petName.toLowerCase(), false);")
    assert is_complete("<!-- Important for reproducible builds. Update using e.g. ./mvnw versions:set -DnewVersion=... -->")


def test_is_complete_incomplete():
    assert not is_complete("# ... (rest of ")
    assert not is_complete("// ... (rest of ")
    assert not is_complete("// ... other imports ")
    assert not is_complete("// rest of the code...")
    assert not is_complete("-- rest of the file content remains the same")
    assert not is_complete("def test_discard_file():\n    # ... \n    # Existing code\n    # ...")


def test_is_complete_last_line():
    code = """
print('Hello, World!')
# ... rest of the code remains the same ..."""
    assert not is_complete(code)

def test_extract_code_from_text_with_lang():
    text = "lalal ```python\nprint('Hello, World!')\n``` foo"
    assert extract_code_from_text(text) == "print('Hello, World!')"

def test_extract_code_from_text():
    text = "```\nprint('Hello, World!')\n``` foo"
    assert extract_code_from_text(text) == "print('Hello, World!')"

def test_extract_code_typescript():
    text = """Step 3: Applying the changes from Step 2 in the file:
```typescript
import React from 'react';
const Message: React.FC<MessageProps> = ({ message, isLastMessage }) => {
}
export default Message;
```
"""
    assert extract_code_from_text(text) == """import React from 'react';
const Message: React.FC<MessageProps> = ({ message, isLastMessage }) => {
}
export default Message;"""
