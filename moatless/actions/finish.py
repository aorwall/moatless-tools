from typing import ClassVar, List

from litellm import Type
from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    Observation,
    FewShotExample,
)
from moatless.file_context import FileContext


class FinishArgs(ActionArguments):
    """Indicate that the task is fully completed."""

    scratch_pad: str = Field(
        ..., description="Your reasoning about why the task is complete."
    )
    finish_reason: str = Field(..., description="Explanation of completion.")

    class Config:
        title = "Finish"

    def to_prompt(self):
        return f"Finish with reason: {self.finish_reason}"

    def equals(self, other: "ActionArguments") -> bool:
        return isinstance(other, FinishArgs)


class Finish(Action):
    args_schema: ClassVar[Type[ActionArguments]] = FinishArgs

    def execute(self, args: FinishArgs, file_context: FileContext | None = None):
        return Observation(message=args.finish_reason, terminal=True)

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="""Applied the change to tests/auth_tests/test_validators.py

```diff
--- tests/auth_tests/test_validators.py
+++ tests/auth_tests/test_validators.py
@@ -238,6 +238,7 @@
             "o'connell", "عبد ال",
             "zerowidth\u200bspace", "nonbreaking\u00a0space",
             "en\u2013dash",
+            "\njoe", "\nRené", "\nUSERNAME", "أحمد\n",
         ]
         v = validators.UnicodeUsernameValidator()
         for valid in valid_usernames:
@@ -250,7 +251,7 @@

     def test_ascii_validator(self):
         valid_usernames = ['glenn', 'GLEnN', 'jean-marc']
-        invalid_usernames = ["o'connell", 'Éric', 'jean marc', "أحمد"]
+        invalid_usernames = ["o'connell", 'Éric', 'jean marc', "أحمد", "\nglenn", "\nGLEnN", "\njean-marc", "USERNAME\n"]
         v = validators.ASCIIUsernameValidator()
         for valid in valid_usernames:
             with self.subTest(valid=valid):
             
Running 45 tests in the following files:
 * tests/calculator/test_operations.py
 * tests/calculator/test_parser.py

45 passed. 0 failed. 0 errors.
""",
                action=FinishArgs(
                    scratch_pad="The implementation has been completed in two steps: First, the username validators were updated to handle leading and trailing newlines correctly. Then, the test suite was enhanced to verify this functionality. The test additions cover both Unicode and ASCII username validators with comprehensive edge cases. All tests are now passing, confirming both the implementation and test coverage are correct.",
                    finish_reason="Username validators now properly handle leading and trailing newlines in usernames, with comprehensive test coverage verifying the functionality.",
                ),
            )
        ]
