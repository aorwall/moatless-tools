import json
import time
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

import pytest

from moatless.completion.stats import CompletionAttempt, CompletionInvocation, MODEL_COSTS, Usage


class TestUsage:
    def test_add(self):
        """Test the __add__ method of Usage."""
        usage1 = Usage(
            completion_cost=0.01,
            completion_tokens=100,
            prompt_tokens=200,
            cache_read_tokens=30,
            cache_write_tokens=20,
        )
        usage2 = Usage(
            completion_cost=0.02,
            completion_tokens=50,
            prompt_tokens=150,
            cache_read_tokens=40,
            cache_write_tokens=25,
        )
        
        combined = usage1 + usage2
        
        assert combined.completion_cost == 0.03
        assert combined.completion_tokens == 150
        assert combined.prompt_tokens == 350
        assert combined.cache_read_tokens == 70
        assert combined.cache_write_tokens == 45
    
    def test_calculate_cost(self):
        """Test the calculate_cost static method."""
        model = "claude-3-5-sonnet-20241022"
        prompt_tokens = 1000
        completion_tokens = 200
        cache_read_tokens = 300
        
        # Calculate expected cost from MODEL_COSTS
        rates = MODEL_COSTS[model]
        non_cached_tokens = prompt_tokens - cache_read_tokens
        expected_input_cost = non_cached_tokens * rates["input"] / 1_000_000
        expected_cache_cost = cache_read_tokens * rates["cache"] / 1_000_000
        expected_output_cost = completion_tokens * rates["output"] / 1_000_000
        expected_total = expected_input_cost + expected_cache_cost + expected_output_cost
        
        cost = Usage.calculate_cost(model, prompt_tokens, completion_tokens, cache_read_tokens)
        assert cost == pytest.approx(expected_total)
    
    def test_update_from_response(self):
        """Test updating usage from a completion response."""
        # Create a mock response
        mock_response = {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200
            }
        }
        
        usage = Usage()
        usage.update_from_response(mock_response, "claude-3-5-sonnet-20241022")
        
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 50
        assert usage.completion_cost > 0  # Should have calculated cost based on model
        
    @patch("litellm.cost_calculator.completion_cost")
    def test_litellm_cost_calculation(self, mock_completion_cost):
        """Test that litellm cost calculation is used when available."""
        # Set up the mock to return a known value
        mock_completion_cost.return_value = 0.05
        
        # Create a mock response
        mock_response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        usage = Usage()
        usage.update_from_response(mock_response, "claude-3-5-sonnet-20241022")
        
        # Verify litellm's cost calculation was used
        mock_completion_cost.assert_called_once()
        assert usage.completion_cost == 0.05
        
    def test_str_representation(self):
        """Test the string representation of Usage."""
        usage = Usage(
            completion_cost=0.0123,
            completion_tokens=100,
            prompt_tokens=200,
            cache_read_tokens=50,
        )
        
        str_repr = str(usage)
        assert "cost: $0.0123" in str_repr
        assert "completion tokens: 100" in str_repr
        assert "prompt tokens: 200" in str_repr
        assert "cache read tokens: 50" in str_repr


class TestCompletionAttempt:
    def test_properties(self):
        """Test the properties of the CompletionAttempt class."""
        usage = Usage(
            completion_cost=0.01,
            completion_tokens=100,
            prompt_tokens=200,
            cache_read_tokens=30,
            cache_write_tokens=20,
        )
        
        attempt = CompletionAttempt(
            start_time=1000.0,
            end_time=2000.0,
            usage=usage,
            success=True,
            attempt_number=1
        )
        
        # Test properties
        assert attempt.completion_cost == 0.01
        assert attempt.completion_tokens == 100
        assert attempt.prompt_tokens == 200
        assert attempt.cache_read_tokens == 30
        assert attempt.cache_write_tokens == 20
    
    def test_update_from_response(self):
        """Test updating attempt from a completion response."""
        # Create a mock response object
        class MockUsage(BaseModel):
            prompt_tokens: int = 150
            completion_tokens: int = 50
            total_tokens: int = 200
        
        class MockResponse(BaseModel):
            usage: MockUsage = MockUsage()
        
        attempt = CompletionAttempt()
        attempt.update_from_response(MockResponse(), "claude-3-5-sonnet-20241022")
        
        assert attempt.usage.prompt_tokens == 150
        assert attempt.usage.completion_tokens == 50
        assert attempt.usage.completion_cost > 0
    
    def test_get_calculated_cost(self):
        """Test getting calculated cost."""
        # Create an attempt with usage data
        usage = Usage(
            completion_cost=0.0,  # Zero cost to test the calculation
            prompt_tokens=1000,
            completion_tokens=200,
            cache_read_tokens=300,
        )
        
        attempt = CompletionAttempt(usage=usage)
        
        # Test with a known model
        model = "claude-3-5-sonnet-20241022"
        
        # Calculate expected cost
        rates = MODEL_COSTS[model]
        non_cached_tokens = usage.prompt_tokens - usage.cache_read_tokens
        expected_input_cost = non_cached_tokens * rates["input"] / 1_000_000
        expected_cache_cost = usage.cache_read_tokens * rates["cache"] / 1_000_000
        expected_output_cost = usage.completion_tokens * rates["output"] / 1_000_000
        expected_total = expected_input_cost + expected_cache_cost + expected_output_cost
        
        cost = attempt.get_calculated_cost(model)
        assert cost == pytest.approx(expected_total)
        
        # Set a cost and test that it returns that value
        attempt.usage.completion_cost = 0.05
        assert attempt.get_calculated_cost(model) == 0.05
    
    def test_addition(self):
        """Test adding two attempts together."""
        usage1 = Usage(
            completion_cost=1.0,
            completion_tokens=100,
            prompt_tokens=200,
            cache_read_tokens=50,
            cache_write_tokens=30,
        )
        
        usage2 = Usage(
            completion_cost=2.0,
            completion_tokens=150,
            prompt_tokens=250,
            cache_read_tokens=70,
            cache_write_tokens=40,
        )
        
        att1 = CompletionAttempt(usage=usage1)
        att2 = CompletionAttempt(usage=usage2)
        
        combined = att1 + att2
        
        assert combined.usage.completion_cost == 3.0
        assert combined.usage.completion_tokens == 250
        assert combined.usage.prompt_tokens == 450
        assert combined.usage.cache_read_tokens == 120
        assert combined.usage.cache_write_tokens == 70
    
    def test_from_completion_response(self):
        """Test factory method for creating attempts from responses."""
        # Create a mock response object
        class MockUsage(BaseModel):
            prompt_tokens: int = 150
            completion_tokens: int = 50
            total_tokens: int = 200
        
        class MockResponse(BaseModel):
            usage: MockUsage = MockUsage()
        
        attempt = CompletionAttempt.from_completion_response(MockResponse(), "claude-3-5-sonnet-20241022")
        
        assert attempt.usage.prompt_tokens == 150
        assert attempt.usage.completion_tokens == 50
        assert attempt.usage.completion_cost > 0
    
    def test_str_representation(self):
        """Test string representation of the attempt."""
        # Successful attempt
        usage = Usage(
            completion_cost=0.0123,
            completion_tokens=100,
            prompt_tokens=200,
            cache_read_tokens=50,
        )
        
        att = CompletionAttempt(
            start_time=1000000.0,  # 1000 seconds in milliseconds
            end_time=1002000.0,    # 1002 seconds in milliseconds
            usage=usage,
            success=True,
        )
        
        str_repr = str(att)
        assert "SUCCESS" in str_repr
        assert "duration: 2.00s" in str_repr
        assert "cost: $0.0123" in str_repr
        assert "completion tokens: 100" in str_repr
        assert "prompt tokens: 200" in str_repr
        
        # Failed attempt
        att = CompletionAttempt(
            start_time=1000000.0,
            end_time=1003000.0,
            success=False,
            failure_reason="API Error",
        )
        str_repr = str(att)
        assert "FAILED: API Error" in str_repr
        assert "duration: 3.00s" in str_repr


class TestCompletionInvocation:
    def test_context_manager(self):
        """Test the context manager functionality."""
        model = "claude-3-5-sonnet-20241022"
        
        with CompletionInvocation(model=model) as invocation:
            # Do some work
            time.sleep(0.1)
            assert invocation.current_attempt is not None
            invocation.current_attempt.usage.prompt_tokens = 100
            invocation.current_attempt.usage.completion_tokens = 50
        
        # After exiting context manager
        assert invocation.start_time > 0
        assert invocation.end_time > 0
        assert invocation.end_time - invocation.start_time >= 100  # At least 100ms
        assert len(invocation.attempts) == 1
        assert invocation.attempts[0].usage.prompt_tokens == 100
        assert invocation.attempts[0].usage.completion_tokens == 50
        assert invocation.current_attempt is None  # Reset after exit
    
    def test_context_manager_with_exception(self):
        """Test context manager when an exception occurs."""
        class TestException(Exception):
            pass
        
        model = "claude-3-5-sonnet-20241022"
        
        try:
            with CompletionInvocation(model=model) as invocation:
                # Do some work
                time.sleep(0.1)
                assert invocation.current_attempt is not None
                invocation.current_attempt.usage.prompt_tokens = 100
                # Raise an exception
                raise TestException("Test error")
        except TestException:
            pass
        
        # After exception
        assert invocation.start_time > 0
        assert invocation.end_time > 0
        assert len(invocation.attempts) == 1
        assert invocation.attempts[0].success is False
        assert "Test error" in invocation.attempts[0].failure_reason
    
    def test_multiple_attempts(self):
        """Test adding multiple attempts to an invocation."""
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation(model=model)
        
        # First attempt
        with invocation as inv:
            assert invocation.current_attempt is not None
            invocation.current_attempt.usage.prompt_tokens = 100
            invocation.current_attempt.usage.completion_tokens = 50
        
        # Second attempt
        with invocation as inv:
            assert invocation.current_attempt is not None
            invocation.current_attempt.usage.prompt_tokens = 150
            invocation.current_attempt.usage.completion_tokens = 75
        
        # Check attempts
        assert len(invocation.attempts) == 2
        assert invocation.attempts[0].usage.prompt_tokens == 100
        assert invocation.attempts[0].usage.completion_tokens == 50
        assert invocation.attempts[1].usage.prompt_tokens == 150
        assert invocation.attempts[1].usage.completion_tokens == 75
    
    def test_add_attempt(self):
        """Test manually adding an attempt."""
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation(model=model)
        
        attempt = CompletionAttempt()
        attempt.usage.prompt_tokens = 100
        attempt.usage.completion_tokens = 50
        
        invocation.add_attempt(attempt)
        
        assert len(invocation.attempts) == 1
        assert invocation.attempts[0] is attempt
    
    def test_usage_property(self):
        """Test the usage property that sums all attempts."""
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation(model=model)
        
        # Add attempts with various usage stats
        attempt1 = CompletionAttempt()
        attempt1.usage.completion_cost = 0.01
        attempt1.usage.prompt_tokens = 100
        attempt1.usage.completion_tokens = 50
        invocation.add_attempt(attempt1)
        
        attempt2 = CompletionAttempt()
        attempt2.usage.completion_cost = 0.02
        attempt2.usage.prompt_tokens = 150
        attempt2.usage.completion_tokens = 75
        invocation.add_attempt(attempt2)
        
        # Get total usage
        total_usage = invocation.usage
        
        assert total_usage.completion_cost == 0.03
        assert total_usage.prompt_tokens == 250
        assert total_usage.completion_tokens == 125
    
    def test_last_attempt_property(self):
        """Test the last_attempt property."""
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation(model=model)
        
        # No attempts yet
        assert invocation.last_attempt is None
        
        # Add some attempts
        attempt1 = CompletionAttempt()
        invocation.add_attempt(attempt1)
        assert invocation.last_attempt is attempt1
        
        attempt2 = CompletionAttempt()
        invocation.add_attempt(attempt2)
        assert invocation.last_attempt is attempt2
    
    def test_duration_properties(self):
        """Test duration_ms and duration_sec properties."""
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation(
            model=model,
            start_time=1000.0,  # 1 second in ms
            end_time=3500.0,   # 3.5 seconds in ms
        )
        
        assert invocation.duration_ms == 2500.0
        assert invocation.duration_sec == 2.5
        
        # Test when times are not set
        invocation2 = CompletionInvocation(model=model)
        assert invocation2.duration_ms == 0
        assert invocation2.duration_sec == 0
    
    def test_str_representation(self):
        """Test string representation of the invocation."""
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation(
            model=model,
            start_time=1000.0,
            end_time=3500.0,
        )
        
        # Add a successful attempt
        attempt1 = CompletionAttempt(success=True)
        attempt1.usage.completion_cost = 0.01
        attempt1.usage.prompt_tokens = 100
        attempt1.usage.completion_tokens = 50
        invocation.add_attempt(attempt1)
        
        # String should show success and metrics
        str_repr = str(invocation)
        assert "SUCCESS" in str_repr
        assert model in str_repr
        assert "attempts: 1" in str_repr
        assert "duration: 2.50s" in str_repr
        
        # Add a failed attempt
        attempt2 = CompletionAttempt(success=False)
        invocation.add_attempt(attempt2)
        
        # Even with one failed attempt, if any succeed it shows success
        str_repr = str(invocation)
        assert "SUCCESS" in str_repr
        assert "attempts: 2" in str_repr
        
        # Create an invocation with only failed attempts
        invocation2 = CompletionInvocation(model=model)
        attempt3 = CompletionAttempt(success=False)
        invocation2.add_attempt(attempt3)
        
        # Should show as failed
        str_repr = str(invocation2)
        assert "FAILED" in str_repr
    
    def test_from_llm_completion(self):
        """Test factory method from_llm_completion."""
        # Create a mock response object
        class MockUsage(BaseModel):
            prompt_tokens: int = 150
            completion_tokens: int = 50
            total_tokens: int = 200
        
        class MockResponse(BaseModel):
            usage: MockUsage = MockUsage()
        
        model = "claude-3-5-sonnet-20241022"
        invocation = CompletionInvocation.from_llm_completion(MockResponse(), model)
        
        assert invocation.model == model
        assert len(invocation.attempts) == 1
        assert invocation.attempts[0].usage.prompt_tokens == 150
        assert invocation.attempts[0].usage.completion_tokens == 50
        
        # Test with provided metrics
        existing_attempt = CompletionAttempt()
        existing_attempt.usage.prompt_tokens = 200
        existing_attempt.usage.completion_tokens = 100
        
        invocation2 = CompletionInvocation.from_llm_completion(
            MockResponse(), model, completion_metrics=existing_attempt
        )
        
        assert invocation2.model == model
        assert len(invocation2.attempts) == 1
        assert invocation2.attempts[0] is existing_attempt
        
        # Test with None response
        with pytest.raises(ValueError, match="Completion response is None"):
            CompletionInvocation.from_llm_completion(None, model) 