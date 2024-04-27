import json

from moatless.settings import Settings

CREATE_PLAN_PROMPT = """Act as an expert software developer. 
 
Your task update the code found in the file context to solve a software requirement.
Follow the user's requirements carefully and to the letter.
Do not suggest any other changes than the ones requested. 
 
Each code file has its contents divided into spans, with each span preceded by a unique span id tag. 

You can update the code by calling the available functions for each part of the code file you want to update.

You can call more than one function but only one function per span so make sure to include all instructions needed to update the span.

If you need to do changes in different spans you call the function one time for each span.

ONLY do changes in the files found in the file context, you cannot change files outside of context or create new files.
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
                        "action": {
                            "type": "string",
                            "description": "How to update the code. ",
                            "enum": ["add", "update", "remove"],
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

    if Settings.coder.use_line_numbers:
        schema["properties"]["start_line"] = {
            "type": "int",
            "description": "Start line of the code span to update",
        }
        schema["properties"]["end_line"] = {
            "type": "int",
            "description": "End line of the code span to update.",
        }
        schema["required"].append("start_line")

    if Settings.coder.use_spans:
        schema["properties"]["span_id"] = {
            "type": "string",
            "description": "Id to the code span that should be updated.",
        }
        schema["required"].append("span_id")

    return json.dumps(schema)


def create_system_prompt():
    prompt = f"# Instructions:\n{CREATE_PLAN_PROMPT}"

    if Settings.coder.enable_chain_of_thought:
        prompt += "\nThink step by step how you would approach the task and write out your thoughts."

    prompt += "Respond ONLY in JSON that follows the schema below:\n"
    prompt += json_schema()
    return prompt


CODER_SYSTEM_PROMPT = """Act as an expert software developer.
Follow the user's instructions carefully and to the letter. 

You're provided with a code file where not relevant code have been commented out. 
Your task is to update the provided code span following the users instructions. 

When you update a code block you must follow the rules below:
* Fully implement all requested functionality
* DO NOT add comments in the code describing what you changed.
* Leave NO todo's, placeholders or missing pieces
* Ensure code is complete! 
* Write out ALL existing code in the provded code span!
* KEEP the placeholder comments named "# ... other code". DO NOT remove them or implement them.   

Respond with the updated code in a code span like
```python
# ... code here
```
"""
