from typing import List

from ghostcoder.llm.base import LLMWrapper
from ghostcoder.schema import Message, Stats


class BaseAction(object):

    def __init__(self,
                 llm: LLMWrapper,
                 sys_prompt: str):
        self.llm = llm
        self.sys_prompt = sys_prompt
        self.instruction_mode = False

    def execute(self,
                message: Message,
                history: List[Message] = []) -> (str, Stats):
        pass
