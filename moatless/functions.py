import logging
import uuid

from pydantic import ValidationError

from moatless.coder.add_code import AddCodeAction, AddCodeBlock
from moatless.coder.remove_code import RemoveCodeAction, RemoveCode
from moatless.coder.types import FunctionResponse
from moatless.coder.update_code import UpdateCode
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class Functions:

    def __init__(self, file_context: FileContext):
        self._file_context = file_context

        self._trace_id = uuid.uuid4().hex
        self._add_action = AddCodeAction(
            file_context=self._file_context, trace_id=self._trace_id
        )
        self._remove_action = RemoveCodeAction(file_context=self._file_context)

    def run_function(self, name: str, arguments: dict, mock_response: str = None):
        try:
            if name == UpdateCode.name:
                func = UpdateCode.model_validate(arguments, strict=True)
                func._file_context = self._file_context  # FIXME
                func._mock_response = mock_response  # FIXME
                return func.call()
            elif name == AddCodeBlock.name:
                task = AddCodeBlock.model_validate(arguments, strict=True)
                return self._add_action.execute(task, mock_response=mock_response)
            elif name == RemoveCode.name:
                task = RemoveCode.model_validate(arguments, strict=True)
                return self._remove_action.execute(task)
            else:
                logger.warning(f"Unknown function: {name}")
                return FunctionResponse(error=f"Unknown function: {name}")
        except ValidationError as e:
            logger.warning(f"Failed to validate function call. Error: {e}")
            return FunctionResponse(error=str(e))
