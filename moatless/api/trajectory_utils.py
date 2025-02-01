from http.client import HTTPException
import json
import logging
from typing import Dict, Any, List
from moatless.api.schema import (
    TrajectoryDTO,
    NodeDTO,
    ActionDTO,
    ObservationDTO,
    CompletionDTO,
    ActionStepDTO,
    FileContextDTO,
    UpdatedFileDTO,
    UsageDTO,
    FileContextFileDTO,
    FileContextSpanDTO,
)
from moatless.node import Node
from moatless.file_context import FileContext
from moatless.runtime.runtime import TestStatus

logger = logging.getLogger(__name__)


def convert_moatless_node_to_api_node(node: Node, action_history: Dict[str, str]) -> NodeDTO:
    """Convert a Moatless Node to an API Node model."""
    action_steps = []
    all_warnings = []
    all_errors = []

    for step in node.action_steps:
        warnings = []
        errors = []

        # Convert action
        action = ActionDTO(
            name=step.action.name,
            shortSummary=step.action.short_summary(),
            thoughts=getattr(step.action, "thoughts", None),
            properties=step.action.model_dump(exclude={"thoughts", "name"}),
        )

        # Check for duplicate actions using the hash map
        current_action_dump = step.action.model_dump(exclude={"thoughts"})
        dump_str = str(current_action_dump)
        if dump_str in action_history:
            errors.append(f"Same action as in step {action_history[dump_str]}")
        else:
            action_history[dump_str] = node.node_id

        # Convert observation
        observation = None
        if step.observation:
            if node and node.observation and node.observation.properties.get("fail_reason"):
                errors.append(node.observation.properties["fail_reason"])

            # Add flags as warnings if they exist
            if node and node.observation and node.observation.properties.get("flags"):
                warnings.extend(node.observation.properties["flags"])

            observation = ObservationDTO(
                message=step.observation.message,
                summary=step.observation.summary,
                properties=step.observation.properties if hasattr(step.observation, "properties") else {},
                expectCorrection=step.observation.expect_correction
                if hasattr(step.observation, "expect_correction")
                else False,
            )

        # Check for Finish action specific errors/warnings
        if step.action.name == "Finish":
            if not node.file_context.has_patch():
                errors.append("finish_without_patch")
            elif not node.file_context.has_test_patch():
                warnings.append("finish_without_test_patch")

        # Convert completion
        completion = None
        if step.completion and step.completion.usage:
            usage = UsageDTO(
                completionCost=step.completion.usage.get_calculated_cost(step.completion.model),
                promptTokens=step.completion.usage.get_total_prompt_tokens(step.completion.model),
                completionTokens=step.completion.usage.completion_tokens,
                cachedTokens=step.completion.usage.cache_read_tokens,
            )
            tokens = []
            if step.completion.usage.prompt_tokens:
                tokens.append(f"{step.completion.usage.get_total_prompt_tokens(step.completion.model)}↑")
            if step.completion.usage.completion_tokens:
                tokens.append(f"{step.completion.usage.completion_tokens}↓")
            if step.completion.usage.cache_read_tokens:
                tokens.append(f"{step.completion.usage.cache_read_tokens}⚡")

            completion = CompletionDTO(
                type="action_step",
                usage=usage,
                retries=step.completion.retries,
                tokens=" ".join(tokens),
                input=json.dumps(step.completion.input, indent=2)
                if hasattr(step.completion, "input") and step.completion.input is not None
                else None,
                response=json.dumps(step.completion.response, indent=2)
                if hasattr(step.completion, "response") and step.completion.response is not None
                else None,
            )

            # Add retries warning
            if step.completion.retries and step.completion.retries > 1:
                warnings.append("retries")

        # Add step errors and warnings to the overall lists
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        action_steps.append(
            ActionStepDTO(
                thoughts=getattr(step.action, "thoughts", None),
                action=action,
                observation=observation,
                completion=completion,
                errors=errors,
                warnings=warnings,
            )
        )

    # Convert action completion
    action_completion = None
    if "build_action" in node.completions:
        completion = node.completions["build_action"]
        if completion and completion.usage:
            usage = UsageDTO(
                completionCost=completion.usage.get_calculated_cost(completion.model),
                promptTokens=completion.usage.prompt_tokens,
                completionTokens=completion.usage.completion_tokens,
                cachedTokens=completion.usage.cache_read_tokens,
            )
            tokens = []
            if completion.usage.prompt_tokens:
                tokens.append(f"{completion.usage.get_total_prompt_tokens(completion.model)}↑")
            if completion.usage.completion_tokens:
                tokens.append(f"{completion.usage.completion_tokens}↓")
            if completion.usage.cache_read_tokens:
                tokens.append(f"{completion.usage.cache_read_tokens}⚡")

            action_completion = CompletionDTO(
                type="build_action",
                usage=usage,
                retries=completion.retries,
                tokens=" ".join(tokens),
                input=json.dumps(completion.input, indent=2)
                if hasattr(completion, "input") and completion.input is not None
                else None,
                response=json.dumps(completion.response, indent=2)
                if hasattr(completion, "response") and completion.response is not None
                else None,
            )

            # Add retries warning for build action
            if completion.retries and completion.retries > 1:
                if action_steps and "retries" not in action_steps[0].warnings:
                    action_steps[0].warnings.append("retries")
                all_warnings.append("retries")

    # Get test results summary if available
    test_results_summary = None
    if node.file_context and node.file_context.test_files:
        all_test_results = [
            result.model_dump() for test_file in node.file_context.test_files for result in test_file.test_results
        ]
        test_results_summary = get_test_results_summary(all_test_results)

        # Add test-related warnings
        error_tests = sum(1 for t in all_test_results if t.get("status") == "error")
        failed_tests = sum(1 for t in all_test_results if t.get("status") == "failed")
        if failed_tests > 0:
            all_warnings.append(f"{failed_tests} tests failed")
        if error_tests > 0:
            all_warnings.append(f"{error_tests} test errors")

    return NodeDTO(
        nodeId=node.node_id,
        actionSteps=action_steps,
        assistantMessage=node.assistant_message,
        userMessage=node.user_message,
        actionCompletion=action_completion,
        fileContext=file_context_to_dto(node.file_context, node.parent.file_context if node.parent else None)
        if node.file_context
        else None,
        error=node.error,
        warnings=all_warnings,
        errors=all_errors,
        terminal=node.is_terminal(),
        allNodeErrors=all_errors,
        allNodeWarnings=all_warnings,
        testResultsSummary=test_results_summary,
    )


def file_context_to_dto(file_context: FileContext, previous_context: FileContext | None = None) -> FileContextDTO:
    """Convert FileContext to FileContextDTO."""
    if not file_context:
        return None
    error_tests = 0
    failed_tests = 0
    if file_context.test_files:
        for test_file in file_context.test_files:
            for result in test_file.test_results:
                if result.status == TestStatus.ERROR:
                    error_tests += 1
                elif result.status == TestStatus.FAILED:
                    failed_tests += 1

    warnings = []
    if failed_tests > 0 or error_tests > 0:
        if failed_tests > 0:
            warnings.append(f"{failed_tests} tests failed")
        if error_tests > 0:
            warnings.append(f"{error_tests} test errors")

    files = []
    for context_file in file_context.files:
        files.append(
            FileContextFileDTO(
                file_path=context_file.file_path,
                patch=context_file.patch,
                spans=[FileContextSpanDTO(**span.model_dump()) for span in context_file.spans],
                show_all_spans=context_file.show_all_spans,
                tokens=context_file.context_size(),
                is_new=context_file._is_new,
                was_edited=context_file.was_edited,
            )
        )

    # Get updated files by comparing with previous context
    updated_files = []
    if previous_context:
        updated_files = get_updated_files(previous_context, file_context)

    return FileContextDTO(
        summary=file_context.create_summary(),
        testResults=[result.model_dump() for test_file in file_context.test_files for result in test_file.test_results]
        if file_context.test_files
        else None,
        patch=file_context.generate_git_patch(),
        files=files,
        warnings=warnings,
        updatedFiles=updated_files,
    )


def get_updated_files(old_context: FileContext, new_context: FileContext) -> List[UpdatedFileDTO]:
    """
    Compare two FileContexts and return information about files that have been updated.
    Updates include content changes, span additions/removals, and file additions.

    Args:
        old_context: The previous FileContext to compare against
        new_context: The new FileContext

    Returns:
        List[UpdatedFileDTO]: List of updated files with their changes and status:
            - added_to_context: File is newly added to context
            - updated_context: File's spans have changed
            - modified: File's content has been modified (patch changed)
    """
    updated_files = []

    # Check files in current context
    for file_path, current_file in new_context._files.items():
        old_file = old_context._files.get(file_path)

        if old_file is None:
            # New file added
            updated_files.append(
                UpdatedFileDTO(
                    file_path=file_path,
                    status="added_to_context",
                    patch=current_file.patch,
                    tokens=current_file.context_size(),
                )
            )
        else:
            # Check for content changes
            if current_file.patch != old_file.patch:
                updated_files.append(
                    UpdatedFileDTO(
                        file_path=file_path,
                        status="modified",
                        patch=current_file.patch,
                        tokens=current_file.context_size(),
                    )
                )
                continue

            # Check for span changes
            current_spans = current_file.span_ids
            old_spans = old_file.span_ids
            if current_spans != old_spans:
                updated_files.append(
                    UpdatedFileDTO(
                        file_path=file_path,
                        status="updated_context",
                        patch=current_file.patch,
                        tokens=current_file.context_size(),
                    )
                )

    return updated_files


def calculate_token_metrics(trajectory_data: Dict[str, Any]) -> Dict[str, int]:
    """Calculate token-related metrics from trajectory data."""
    prompt_tokens = trajectory_data.get("prompt_tokens", 0)
    completion_tokens = trajectory_data.get("completion_tokens", 0)
    cached_tokens = trajectory_data.get("cached_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens

    return {
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "cachedTokens": cached_tokens,
        "totalTokens": total_tokens,
    }


def extract_action_metrics(trajectory_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract action-related metrics from trajectory data."""
    return {
        "failedActions": trajectory_data.get("failed_actions", 0),
        "duplicatedActions": trajectory_data.get("duplicated_actions", 0),
    }


def convert_nodes(trajectory_data: Dict[str, Any]) -> List[NodeDTO]:
    """Convert nodes from trajectory data to NodeDTOs."""
    nodes = []
    action_history = {}  # Track action dumps to detect duplicates

    if "root" in trajectory_data:
        root_node = Node.reconstruct(trajectory_data["root"])
    elif "nodes" in trajectory_data:
        root_node = Node.reconstruct(trajectory_data["nodes"])
    else:
        logger.error(f"No root node found in trajectory data {trajectory_data.keys()}")
        raise HTTPException(400, "No root node found in trajectory data")
    for node in root_node.get_all_nodes():
        node_dto = convert_moatless_node_to_api_node(node, action_history)
        nodes.append(node_dto)

    return nodes


def create_trajectory_dto(trajectory_data: Dict[str, Any]) -> TrajectoryDTO:
    """Create TrajectoryDTO from trajectory data loaded from a file."""
    if not trajectory_data:
        logger.warning("Empty trajectory data provided")
        return TrajectoryDTO()

    # Calculate token metrics
    token_metrics = calculate_token_metrics(trajectory_data)

    # Extract action metrics
    action_metrics = extract_action_metrics(trajectory_data)

    # Convert nodes
    nodes = convert_nodes(trajectory_data)

    logger.info(f"Loading trajectory with {len(nodes)} nodes")

    return TrajectoryDTO(
        duration=trajectory_data.get("duration"),
        error=trajectory_data.get("error"),
        iterations=trajectory_data.get("all_transitions"),
        completionCost=trajectory_data.get("total_cost"),
        flags=trajectory_data.get("flags", []),
        nodes=nodes,
        **token_metrics,
        **action_metrics,
    )


def load_trajectory_from_file(file_path: str) -> TrajectoryDTO:
    """Load trajectory data from a file and convert it to TrajectoryDTO."""
    try:
        logger.info(f"Loading trajectory from file: {file_path}")
        with open(file_path, "r") as f:
            trajectory_data = json.load(f)
            return create_trajectory_dto(trajectory_data)
    except FileNotFoundError:
        logger.warning(f"Trajectory file not found: {file_path}")
        raise HTTPException(status_code=400, detail=f"Trajectory file not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in trajectory file: {file_path}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in trajectory file: {file_path}")
    except Exception as e:
        logger.error(f"Error processing trajectory file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing trajectory file: {str(e)}")


def get_test_results_summary(test_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute summary of test results."""
    if not test_results:
        return None
    return {
        "total": len(test_results),
        "passed": sum(1 for t in test_results if t.get("status") == "passed"),
        "failed": sum(1 for t in test_results if t.get("status") == "failed"),
        "errors": sum(1 for t in test_results if t.get("status") == "error"),
        "skipped": sum(1 for t in test_results if t.get("status") == "skipped"),
    }
