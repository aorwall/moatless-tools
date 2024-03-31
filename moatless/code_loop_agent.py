from litellm import completion
from pydantic import BaseModel, Field

from moatless.coder.prompts import CODER_SYSTEM_PROMPT


class CodeRequest(BaseModel):
    instructions: str = Field(description="Instructions on what to implement")

class LoopAgent:

    def __init__(self, coder_model: str = "gpt-4-0125-preview"):
        self._coder_model = coder_model

    def run(self, request: CodeRequest):
        # TODO: Index repository

        # TODO: Retrieve relevant files

        # TODO: Implement code

        system_message = { "content": CODER_SYSTEM_PROMPT, "role": "system"}

        response = completion(
          model=self._coder_model,
          messages=[{"content": "Hello, how are you?","role": "user"}]
        )



        # TODO: Update tesrs

        # TODO: Run tests

        # TODO: Looop
