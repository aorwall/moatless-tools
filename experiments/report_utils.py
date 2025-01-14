import os
from typing import List, Dict, Optional
from datetime import datetime, timezone
from moatless.benchmark.report import BenchmarkResult
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.completion.model import Completion, Usage
from moatless.file_context import FileContext
from experiments.schema import (
    EvaluationResponseDTO, EvaluationSettingsDTO, InstanceItemDTO, UsageDTO,
    InstanceResponseDTO, NodeDTO, ActionDTO, ObservationDTO, CompletionDTO, ActionStepDTO, FileContextDTO, FileContextSpanDTO, FileContextFileDTO, UpdatedFileDTO
)
from moatless.loop import AgenticLoop
from moatless.node import Node as MoatlessNode
from testbeds.schema import TestStatus
import json
import logging

logger = logging.getLogger(__name__)
    

def load_resolution_rates() -> Dict[str, float]:
    """Load resolution rates from the resolved submissions data."""
    dir = os.getcwd()
    dataset_path = os.path.join(dir, "datasets", "resolved_submissions.json")
    with open(dataset_path) as f:
        resolved_data = json.load(f)
        return {
            instance_id: len(data["resolved_submissions"]) / data["no_of_submissions"] 
            if data["no_of_submissions"] > 0 else 0.0
            for instance_id, data in resolved_data.items()
        }


def create_evaluation_response(evaluation_name: str, instance_items: List[InstanceItemDTO], first_tree: Optional[AgenticLoop] = None) -> EvaluationResponseDTO:
    """Create EvaluationResponseDTO from evaluation and instance items."""
    # Calculate totals from instance items
    
    prompt_tokens = sum(item.promptTokens or 0 for item in instance_items)
    completion_tokens = sum(item.completionTokens or 0 for item in instance_items)
    cached_tokens = sum(item.cachedTokens or 0 for item in instance_items)
    total_tokens = prompt_tokens + completion_tokens

    total_cost = sum(item.completionCost or 0 for item in instance_items)

    # Create settings from first tree if available
    settings = None
    if first_tree:
        settings = EvaluationSettingsDTO(
            model=first_tree.agent._completion.model,
            temperature=first_tree.agent._completion.temperature,
            maxIterations=first_tree.max_iterations,
            responseFormat=first_tree.agent._completion.response_format.value,
            messageHistoryFormat=first_tree.agent.message_generator.message_history_type.value,
            maxCost=first_tree.max_cost or 0.0
        )

    return EvaluationResponseDTO(
        name=evaluation_name,
        status="completed",
        isActive=False,
        settings=settings,
        startedAt=datetime.now(timezone.utc),
        totalCost=total_cost,
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        cachedTokens=cached_tokens,
        totalTokens=total_tokens,
        totalInstances=len(instance_items),
        completedInstances=sum(1 for i in instance_items if i.status == "completed"),
        errorInstances=sum(1 for i in instance_items if i.status == "error"),
        resolvedInstances=sum(1 for i in instance_items if i.resolved is True),
        failedInstances=sum(1 for i in instance_items if i.status == "failed"),
        instances=instance_items
    )

def derive_instance_status(result: BenchmarkResult) -> str:
    """Derive instance status from a BenchmarkResult."""
    return (
        "resolved" if result.resolved is True else
        "error" if result.status == "error" else
        "failed" if result.resolved is False and result.status == "completed" else 
        result.status
    )

def create_instance_dto(model: str, result: BenchmarkResult, resolution_rates: Dict[str, float], splits: List[str] = None) -> InstanceItemDTO:
    """Create InstanceItemDTO from a BenchmarkResult."""
    status = derive_instance_status(result)

    resolution_rate = resolution_rates.get(result.instance_id, None)
    if not resolution_rate:
        logger.warning(f"Resolution rate not found for instance {result.instance_id}")
    
    
    return InstanceItemDTO(
        instanceId=result.instance_id,
        status=status,
        duration=result.duration,
        resolved=result.resolved,
        error=result.error if result.error else None,
        iterations=result.all_transitions,
        completionCost=result.total_cost,
        totalTokens=result.prompt_tokens + result.completion_tokens,
        promptTokens=result.prompt_tokens,
        cachedTokens=result.cached_tokens,
        completionTokens=result.completion_tokens,
        resolutionRate=resolution_rate,
        splits=splits or [],
        failedActions=result.failed_actions,
        duplicatedActions=result.duplicated_actions,
        flags=result.flags
    )

def convert_moatless_node_to_api_node(node: MoatlessNode, action_history: Dict[str, str]) -> NodeDTO:
    """Convert a Moatless Node to an API Node model."""

    # Just to return empty node for root node
    if not node.parent:
        return NodeDTO(
            nodeId=node.node_id
        )

    # Convert action steps
    action_steps = []
    for step in node.action_steps:
        warnings = []
        errors = []

        logger.info(step.action.short_summary())
        # Convert action
        action = ActionDTO(
            name=step.action.name,
            shortSummary=step.action.short_summary(),
            thoughts=getattr(step.action, "thoughts", None),
            properties=step.action.model_dump(exclude={"thoughts", "name"})
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
            if node.observation.properties.get("fail_reason"):
                errors.append(node.observation.properties["fail_reason"])

            # Add flags as warnings if they exist
            if node.observation.properties.get("flags"):
                warnings.extend(node.observation.properties["flags"])

            observation = ObservationDTO(
                message=step.observation.message,
                summary=step.observation.summary,
                properties=step.observation.properties if hasattr(step.observation, "properties") else {},
                expectCorrection=step.observation.expect_correction if hasattr(step.observation, "expect_correction") else False
            )

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
                cachedTokens=step.completion.usage.cached_tokens
            )
            tokens = []
            if step.completion.usage.prompt_tokens:
                tokens.append(f"{step.completion.usage.get_total_prompt_tokens(step.completion.model)}↑")
            if step.completion.usage.completion_tokens:
                tokens.append(f"{step.completion.usage.completion_tokens}↓")
            if step.completion.usage.cached_tokens:
                tokens.append(f"{step.completion.usage.cached_tokens}⚡")
            
            completion = CompletionDTO(
                type="action_step",
                usage=usage,
                tokens=" ".join(tokens),
                input=json.dumps(step.completion.input, indent=2) if hasattr(step.completion, "input") and step.completion.input is not None else None,
                response=json.dumps(step.completion.response, indent=2) if hasattr(step.completion, "response") and step.completion.response is not None else None
            )

        action_steps.append(ActionStepDTO(
            thoughts=getattr(step.action, "thoughts", None),
            action=action,
            observation=observation,
            completion=completion,
            errors=errors,
            warnings=warnings
        ))

    # Convert completions
    action_completion = None
    if "build_action" in node.completions:
        completion = node.completions["build_action"]
        if completion and completion.usage:
            usage = UsageDTO(
                completionCost=calculate_completion_cost(completion),
                promptTokens=completion.usage.get_total_prompt_tokens(completion.model),
                completionTokens=completion.usage.completion_tokens,
                cachedTokens=completion.usage.cached_tokens
            )
            tokens = []
            if completion.usage.prompt_tokens:
                tokens.append(f"{completion.usage.get_total_prompt_tokens(completion.model)}↑")
            if completion.usage.completion_tokens:
                tokens.append(f"{completion.usage.completion_tokens}↓")
            if completion.usage.cached_tokens:
                tokens.append(f"{completion.usage.cached_tokens}⚡")
            
            action_completion = CompletionDTO(
                type="build_action",
                usage=usage,
                tokens=" ".join(tokens),
                input=json.dumps(completion.input, indent=2) if hasattr(completion, "input") and completion.input is not None else None,
                response=json.dumps(completion.response, indent=2) if hasattr(completion, "response") and completion.response is not None else None
            )

    # Convert file context if exists
    file_context = None
    if node.file_context:
        previous_context = None
        if node.parent:
            previous_context = node.parent.file_context
        file_context = file_context_to_dto(node.file_context, previous_context)

    # FIXME: Fulfix to set fist message on Step 1
    user_message = node.user_message
    if node.parent and not node.parent.parent:
        user_message = node.parent.user_message
    
    return NodeDTO(
        nodeId=node.node_id,
        actionSteps=action_steps,
        assistantMessage=node.assistant_message,
        userMessage=user_message,
        actionCompletion=action_completion,
        fileContext=file_context,
        error=node.error,
        terminal=node.is_terminal()
    )

def create_instance_response(
    agentic_loop: AgenticLoop,
    instance: dict,
    eval_result: Optional[dict] = None,
    resolution_rates: Dict[str, float] = None,
    splits: List[str] = None,
    result: Optional[BenchmarkResult] = None
) -> InstanceResponseDTO:
    """Create InstanceResponseDTO from a SearchTree and instance data."""
    nodes = []
    # Track action dumps to detect duplicates using a dictionary
    action_history = {}  # model_dump -> node_id mapping
    
    for moatless_node in agentic_loop.root.get_all_nodes():
        node_dto = convert_moatless_node_to_api_node(moatless_node, action_history)
        nodes.append(node_dto)

    instance_id = agentic_loop.metadata.get("instance_id")
    
    status = derive_instance_status(result) if result else "pending"

    return InstanceResponseDTO(
        nodes=nodes,
        totalNodes=len(nodes),
        instance=instance,
        evalResult=eval_result,
        status=status,
        duration=result.duration if result else None,
        completionCost=result.total_cost if result else None,
        totalTokens=result.prompt_tokens + result.completion_tokens if result else None,
        promptTokens=result.prompt_tokens if result else None,
        cachedTokens=result.cached_tokens if result else None,
        completionTokens=result.completion_tokens if result else None,
        resolved=result.resolved if result else None,
        error=result.error if result else None,
        iterations=result.all_transitions if result else None,
        resolutionRate=resolution_rates.get(instance_id) if resolution_rates else None,
        splits=splits or [],
        flags=result.flags if result else None
    )

def get_updated_files(
    old_context: FileContext, new_context: FileContext
) -> List[UpdatedFileDTO]:
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
            updated_files.append(UpdatedFileDTO(
                file_path=file_path,
                status="added_to_context",
                patch=current_file.patch,
                tokens=current_file.context_size()
            ))
        else:
            # Check for content changes
            if current_file.patch != old_file.patch:
                updated_files.append(UpdatedFileDTO(
                    file_path=file_path,
                    status="modified",
                    patch=current_file.patch,
                    tokens=current_file.context_size()
                ))
                continue

            # Check for span changes
            current_spans = current_file.span_ids
            old_spans = old_file.span_ids
            if current_spans != old_spans:
                updated_files.append(UpdatedFileDTO(
                    file_path=file_path,
                    status="updated_context",
                    patch=current_file.patch,
                    tokens=current_file.context_size()
                ))

    return updated_files


def file_context_to_dto(file_context: FileContext, previous_context: FileContext | None = None) -> FileContextDTO:
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

    """Convert FileContext to FileContextDTO."""
    files = []
    for context_file in file_context.files:
        files.append(FileContextFileDTO(
            file_path=context_file.file_path,
            patch=context_file.patch,
            spans=[FileContextSpanDTO(**span.model_dump()) for span in context_file.spans],
            show_all_spans=context_file.show_all_spans,
            tokens=context_file.context_size(),
            is_new=context_file._is_new,
            was_edited=context_file.was_edited,
        ))

    # Get updated files by comparing with previous context
    updated_files = []
    if previous_context:
        updated_files = get_updated_files(previous_context, file_context)

    return FileContextDTO(
        summary=file_context.create_summary(),
        testResults=[result.model_dump() for test_file in file_context.test_files
                    for result in test_file.test_results] if file_context.test_files else None,
        patch=file_context.generate_git_patch(),
        files=files,
        warnings=warnings,
        updatedFiles=updated_files
    )
    

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Calculate cost based on token counts and model."""
    return Usage.calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)

def calculate_completion_cost(completion: Completion) -> float:
    if not completion.usage:
        return 0.0
    return completion.usage.get_calculated_cost(completion.model)
