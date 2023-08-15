from typing import Optional

from code_blocks.language.java import Java
from code_blocks.language.language import Language
from code_blocks.language.python import Python

_languages = {
    "java": Java(),
    "python": Python()
}

def get_language(name: str) -> Optional[Language]:
    return _languages.get(name)
