import json

from moatless.settings import Settings

CREATE_PLAN_SYSTEM_PROMPT = """Act as an expert software developer. 

Your task is to create a plan for how to update the provided code to solve a software requirement. 
Follow the user's requirements carefully and to the letter. 

You will plan the development by creating tasks for each part of the code files that you want to update.
The task should include the file path, the span id to the code section and the instructions for the update.

Each code file will have its contents divided into spans, with each span preceded by a unique span id tag. 

If you need to do changes in different spans you must create one task for each span.

If you want to add new functions or classes you can use the action 'add' and set the id to the span where the new block will be added.

Any code that needs to be changed must be included in a task with a correct span id set and instructions provided.  

ONLY plan for changes in the provided files, do not plan for new files to be created.
"""


def json_schema():
    # TODO: Change to pydantic object
    schema = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file that should be updated.",
                        },
                        "span_id": {
                            "type": "string",
                            "description": "Id to the code span that should be updated.",
                        },
                        "action": {
                            "type": "string",
                            "description": "How to update the code. ",
                            "enum": ["add", "update"],
                        },
                        "instructions": {
                            "type": "string",
                            "descriptions": "Instructions for editing the code block",
                        },
                    },
                    "required": ["file_path", "action", "instructions"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["thoughts", "tasks"],
        "additionalProperties": False,
    }

    if Settings.coder.enable_chain_of_thought:
        schema["properties"]["thoughts"] = {
            "type": "string",
            "description": "Your thoughts on how to approach the task.",
        }

    return json.dumps(schema)


JSON_SCHEMA = """{
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "string",
            "description": "Your thoughts on how to approach the task."
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file that should be updated."
                    },
                    "span_id": {
                        "type": "string",
                        "description": "Id to the code span that should be updated."
                    },
                    "action": {
                        "type": "string",
                        "description": "If the code block should be added, removed or updated.",
                        "enum": ["add", "remove", "update"]
                    }
                    "instructions": {
                        "type": "string",
                        "descriptions": "Instructions for editing the code block"
                    },
                },
                "required": ["file_path", "action", "instructions"],
                "additionalProperties": False
            }
        }
    },
    "required": ["thoughts", "tasks"],
    "additionalProperties": False
}"""


def create_system_prompt():
    prompt = f"# Instructions:\n{CREATE_PLAN_SYSTEM_PROMPT}"

    if Settings.coder.enable_chain_of_thought:
        prompt += "\nThink step by step how you would approach the task and write out your thoughts."

    prompt += "Respond ONLY in JSON that follows the schema below:\n"
    prompt += json_schema()
    return prompt


CODER_SYSTEM_PROMPT = """Act as an expert software developer.
Follow the user's instructions carefully and to the letter. 

You're provided with a code file where not relevant code have been commented out. 

Your task is to update the provided code span following the users instructions.
You will work with a code span in the code. A code span could be a number of lines, a class, function etc. 
You should update only the provided code span.

When you update a code block you must follow the rules below:
* Fully implement all requested functionality
* DO NOT add comments in the code describing what you changed.
* Leave NO todo's, placeholders or missing pieces
* Ensure code is complete! 
* Write out ALL existing code in the provded code span!

Respond with the updated code in a code span like
```python
# ... code here
```
"""
