from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.completion import schema
from moatless.completion.schema import ChatCompletionToolParam
from pydantic import Field


class CreateJsonArtifactArgs(ActionArguments):
    """
    Create a new artifact.
    """

    artifact_type: str = Field(..., description="The type of artifact to create")
    artifact_name: str = Field(..., description="The name of the artifact to create")
    data: dict = Field(..., description="The data to create the artifact with")


class CreateJsonArtifact(Action):
    """
    Create a new artifact.
    """

    args_schema = CreateJsonArtifactArgs

    action_name: str = Field(..., description="The name to use for the action in tool calls")
    description: str = Field(..., description="Description of the action")
    json_schema: dict = Field(..., description="The JSON schema for the artifact")

    def __init__(self, artifact_type: str, artifact_name: str):
        super().__init__()

    @classmethod
    def openai_schema(cls, thoughts_in_action: bool = False) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": parameters,
            },
        }
