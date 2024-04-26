import json

from moatless.settings import Settings

CREATE_PLAN_FIRST_PART_PROMPT = """Act as an expert software developer. 

Your task is to create a plan for how to update the provided code to solve a software requirement. 
Follow the user's requirements carefully and to the letter. 

You will plan the development by creating tasks for each part of the code files that you want to update.
"""

SPAN_INSTRUCTIONS = """
The task should include the file path, the span id and the line numbers to the code section and the instructions for the update.

Each code file will have its contents divided into spans, with each span preceded by a unique span id tag. 
"""

LINE_NUMBER_INSTRUCTIONS = """
The task should include the file path, the start and end line numbers of the code span and the instructions for the update.
"""

CREATE_PLAN_LAST_PART_PROMPT = """
If you need to do changes in different spans you must create one task for each span.

If you want to add new functions or classes you can use the action 'add' and set the id to the span where the new block will be added.

Any code that needs to be changed must be included in a task with a correct span set and instructions provided.  

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
    prompt = f"# Instructions:\n{CREATE_PLAN_FIRST_PART_PROMPT}"

    if Settings.coder.use_spans:
        prompt += SPAN_INSTRUCTIONS
    else:
        prompt += LINE_NUMBER_INSTRUCTIONS

    prompt += CREATE_PLAN_LAST_PART_PROMPT

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
