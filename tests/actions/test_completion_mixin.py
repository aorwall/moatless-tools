import pytest
from pydantic import BaseModel
from unittest.mock import MagicMock, patch

from moatless.actions.action import Action, CompletionModelMixin
from moatless.completion.base import BaseCompletionModel
from moatless.actions.schema import ActionArguments
from moatless.workspace import Workspace


class TestCompletionArgs(ActionArguments):
    query: str


class TestCompletionModel(BaseCompletionModel):
    def _completion(self, *args, **kwargs):
        return None

    async def _async_completion(self, *args, **kwargs):
        return None
        
    async def _validate_completion(self, completion_response):
        return [], None, []


class TestActionWithMixin(Action, CompletionModelMixin):
    args_schema = TestCompletionArgs

    async def _execute(self, args, file_context=None):
        return "Success"

    def _initialize_completion_model(self):
        # This should be called automatically by the model_validator
        self.initialization_called = True


class TestCompletionMixin:
    def test_auto_initialization(self):
        """Test that the completion model is automatically initialized after creation."""
        completion_model = TestCompletionModel(
            model_id="test_model",
            model="test_model",
            temperature=0.7,
            max_tokens=100,
            timeout=60
        )
        action = TestActionWithMixin(completion_model=completion_model)
        
        # The model_validator should have called _initialize_completion_model
        assert hasattr(action, "initialization_called")
        assert action.initialization_called is True

    def test_model_dump_with_completion(self):
        """Test that model_dump includes completion model."""
        completion_model = TestCompletionModel(
            model_id="test_model",
            model="test_model",
            temperature=0.7,
            max_tokens=100,
            timeout=60
        )
        action = TestActionWithMixin(completion_model=completion_model)
        
        dumped = action.model_dump()
        assert "completion_model" in dumped
        assert dumped["completion_model"]["model_id"] == "test_model"
        
    def test_model_dump_without_completion(self):
        """Test that model_dump works without completion model."""
        action = TestActionWithMixin()
        
        dumped = action.model_dump()
        assert "completion_model" in dumped
        assert dumped["completion_model"] is None
        
    def test_model_validate_with_completion(self):
        """Test that model_validate handles completion model data."""
        data = {
            "completion_model": {
                "model_id": "test_model", 
                "model": "test_model",
                "temperature": 0.7,
                "max_tokens": 100,
                "timeout": 60,
                "completion_model_class": "tests.actions.test_completion_mixin.TestCompletionModel"
            }
        }
        
        with patch("moatless.completion.base.BaseCompletionModel.model_validate") as mock_validate:
            mock_validate.return_value = TestCompletionModel(
                model_id="test_model",
                model="test_model",
                temperature=0.7,
                max_tokens=100,
                timeout=60
            )
            action = TestActionWithMixin.model_validate(data)
            
            mock_validate.assert_called_once()
            assert action.completion_model is not None
            assert action.completion_model.model_id == "test_model"
    
    def test_initialize_with_workspace(self):
        """Test that initialize sets workspace properly."""
        completion_model = TestCompletionModel(
            model_id="test_model",
            model="test_model",
            temperature=0.7,
            max_tokens=100,
            timeout=60
        )
        action = TestActionWithMixin(completion_model=completion_model)
        
        # Reset the flag
        action.initialization_called = False
        
        # Create a mock workspace
        workspace = MagicMock(spec=Workspace)
        
        # Call initialize
        action.initialize(workspace)
        
        # Check that workspace was set
        assert action._workspace == workspace 