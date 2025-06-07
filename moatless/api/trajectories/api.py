"""API endpoints for run status and trajectory data."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Body
from moatless.flow.manager import FlowManager
from moatless.flow.schema import (
    ExecuteNodeRequest,
    StartTrajectoryRequest,
    TrajectoryListItem,
    TrajectoryResponseDTO,
)
from moatless.flow.flow import AgenticFlow
from moatless.api.dependencies import get_flow_manager, get_storage
from moatless.flow.search_tree import SearchTree
from moatless.storage.base import BaseStorage
from .schema import CreateTrajectoryRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=list[TrajectoryListItem])
async def get_trajectories(flow_manager: FlowManager = Depends(get_flow_manager)):
    """Get all trajectories."""
    try:
        return await flow_manager.list_trajectories()
    except Exception as e:
        logger.exception(f"Error getting trajectories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}", response_model=TrajectoryResponseDTO)
async def get_trajectory(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Get the status, trajectory data, and events for a specific trajectory."""
    try:
        return await flow_manager.get_trajectory(project_id, trajectory_id)
    except ValueError as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting trajectory data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/settings")
async def get_trajectory_settings(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_trajectory_settings(project_id, trajectory_id)


@router.get("/{project_id}/{trajectory_id}/logs")
async def get_trajectory_logs(
    project_id: str,
    trajectory_id: str,
    file_name: str | None = None,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Get the log files for a specific trajectory.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        file_name: Optional specific log file name to retrieve

    Returns:
        The log file contents
    """
    try:
        return await flow_manager.get_trajectory_logs(project_id, trajectory_id, file_name)
    except ValueError as e:
        logger.exception(f"Error getting trajectory logs: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting trajectory logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/events")
async def get_trajectory_events(
    project_id: str,
    trajectory_id: str,
    storage: BaseStorage = Depends(get_storage),
):
    """Get the events for a specific trajectory.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        event_bus: The event bus instance

    Returns:
        The trajectory events
    """
    key = storage.get_trajectory_path(project_id, trajectory_id)
    key = f"{key}/events.jsonl"

    if not await storage.exists(key):
        logger.info(f"No events found for key {key}")
        return []

    return await storage.read_lines(key)


@router.get("/{project_id}/{trajectory_id}/completions/{node_id}/{item_id}")
async def get_completions(
    project_id: str,
    trajectory_id: str,
    node_id: str,
    item_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_completions(project_id, trajectory_id, node_id, item_id)


@router.post("/{project_id}/{trajectory_id}/start")
async def start_trajectory(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Start a trajectory without additional parameters."""
    try:
        await flow_manager.start_trajectory(project_id, trajectory_id)
        return {"status": "success", "message": f"Started trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error starting trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error starting trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/retry-trajectory")
async def retry_trajectory(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Reset and restart a trajectory by removing all children from the root node."""
    try:
        await flow_manager.retry_trajectory(project_id, trajectory_id)
        return {"status": "success", "message": f"Retried trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error retrying trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error retrying trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/resume")
async def resume_trajectory(
    project_id: str,
    trajectory_id: str,
    request: StartTrajectoryRequest,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Resume a trajectory."""
    try:
        await flow_manager.resume_trajectory(project_id, trajectory_id, request)
        return {"status": "success", "message": f"Resumed trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error resuming trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error resuming trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/execute")
async def execute_node(
    project_id: str,
    trajectory_id: str,
    request: ExecuteNodeRequest,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Execute a run."""
    try:
        return await flow_manager.execute_node(project_id, trajectory_id, request.node_id)
    except ValueError as e:
        logger.exception(f"Error executing node: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error executing node: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/tree")
async def get_node_tree(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_node_tree(project_id, trajectory_id)


@router.get("/{project_id}/{trajectory_id}/node/{node_id}")
async def get_node(
    project_id: str,
    trajectory_id: str,
    node_id: int,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.get_node(project_id, trajectory_id, node_id)


@router.post("/{project_id}/{trajectory_id}/node/{node_id}/reset")
async def reset_node(
    project_id: str,
    trajectory_id: str,
    node_id: int,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    return await flow_manager.reset_node(project_id, trajectory_id, node_id)


@router.get("/{project_id}/{trajectory_id}/node/{node_id}/evaluation")
async def get_node_evaluation_files(
    project_id: str,
    trajectory_id: str,
    node_id: int,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Get the evaluation files for a specific node.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        node_id: The node ID

    Returns:
        A dictionary mapping file names to file contents
    """
    try:
        return await flow_manager.get_node_evaluation_files(project_id, trajectory_id, node_id)
    except ValueError as e:
        logger.exception(f"Error getting node evaluation files: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting node evaluation files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/{trajectory_id}/chat-messages")
async def get_chat_messages(
    project_id: str,
    trajectory_id: str,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Get formatted chat messages for display in the UI.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID

    Returns:
        A list of formatted chat messages
    """

    def format_action_args(action_name, action_dict):
        """Format action arguments based on action type for UI display"""
        formatted_args = None

        # Format AddCodingTasks action
        if action_name == "AddCodingTasks":
            if "tasks" in action_dict:
                formatted_args = {
                    "type": "tasklist",
                    "items": [
                        {
                            "type": "task",
                            "id": task.get("id", ""),
                            "title": task.get("title", ""),
                            "instructions": task.get("instructions", ""),
                            "priority": task.get("priority", 0),
                            "related_files": task.get("related_files", []),
                        }
                        for task in action_dict.get("tasks", [])
                    ],
                }

        # Add more action type formatters here as needed

        return formatted_args

    try:
        # Read the trajectory node from storage
        root_node = await flow_manager.read_trajectory_node(project_id, trajectory_id)

        # Extract relevant chat messages from nodes in the entire trajectory
        chat_messages = []

        # Process all nodes in the tree
        for node in root_node.get_all_nodes():
            node_messages = []

            # 1. User message (if present)
            if node.user_message:
                node_messages.append(
                    {
                        "type": "user_message",
                        "content": {"message": node.user_message},
                        "node_id": node.node_id,
                        "id": f"{node.node_id}-user_message",
                        "trajectory_id": trajectory_id,
                        "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                    }
                )

            # 2. Assistant response combines: assistant message, thoughts, and actions+observations
            assistant_content = {}

            # Assistant message
            if node.assistant_message:
                assistant_content["message"] = node.assistant_message

            # Thoughts
            if node.thoughts:
                assistant_content["thought"] = node.thoughts.text

            # Actions and observations
            if node.action_steps:
                assistant_content["actions"] = []

                for i, step in enumerate(node.action_steps):
                    if step.action:
                        # Create action details
                        action_properties = step.action.model_dump()
                        action_name = step.action.name

                        # Extract most relevant information for each action type using dict access
                        action_dict = action_properties
                        short_summary = ""
                        if action_dict.get("thought"):
                            short_summary = action_dict["thought"][:100] + (
                                "..." if len(action_dict["thought"]) > 100 else ""
                            )
                        elif action_dict.get("query"):
                            short_summary = action_dict["query"][:100] + (
                                "..." if len(action_dict["query"]) > 100 else ""
                            )
                        elif action_dict.get("command"):
                            short_summary = f"$ {action_dict['command']}"
                        elif action_dict.get("file_path"):
                            short_summary = action_dict["file_path"]
                            if action_dict.get("content"):
                                short_summary += f" (modified)"

                        # Format action arguments based on action type for UI display
                        formatted_args = format_action_args(action_name, action_dict)

                        # Create action entry with its observation
                        action_entry = {
                            "name": action_name,
                            "shortSummary": short_summary,
                            "properties": action_properties,
                            "formattedArgs": formatted_args,
                            "observation": None,
                        }

                        # Add observation if available
                        if step.observation:
                            observation_properties = (
                                step.observation.model_dump() if hasattr(step.observation, "model_dump") else {}
                            )
                            action_entry["observation"] = {
                                "message": step.observation.message
                                if hasattr(step.observation, "message")
                                else str(step.observation),
                                "summary": step.observation.summary if hasattr(step.observation, "summary") else None,
                                "properties": observation_properties,
                            }

                        assistant_content["actions"].append(action_entry)

            # Add artifact changes
            if node.artifact_changes:
                assistant_content["artifacts"] = []
                for artifact in node.artifact_changes:
                    assistant_content["artifacts"].append(artifact.model_dump())

            # Only add assistant message if there's any content
            if assistant_content:
                node_messages.append(
                    {
                        "type": "assistant_response",
                        "content": assistant_content,
                        "node_id": node.node_id,
                        "id": f"{node.node_id}-assistant_response",
                        "trajectory_id": trajectory_id,
                        "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                    }
                )

            # Add all node messages to the chat
            chat_messages.extend(node_messages)

        return {"messages": chat_messages}
    except ValueError as e:
        logger.exception(f"Error getting chat messages: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error getting chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{project_id}/{trajectory_id}/settings")
async def save_trajectory_settings(
    project_id: str,
    trajectory_id: str,
    settings: dict = Body(...),
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Save settings for a specific trajectory.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        settings: The settings data to save

    Returns:
        A success response
    """
    try:
        await flow_manager.save_trajectory_settings(project_id, trajectory_id, settings)
        return {"status": "success", "message": f"Settings saved for trajectory {trajectory_id}"}
    except ValueError as e:
        logger.exception(f"Error saving trajectory settings: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error saving trajectory settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_trajectory(
    request: CreateTrajectoryRequest,
    flow_manager: FlowManager = Depends(get_flow_manager),
):
    """Create a new trajectory with flow configuration."""
    try:
        # Prepare flow config if provided
        flow_config = None
        if request.flow_config:
            flow_config = SearchTree.model_validate(request.flow_config)

        # Create the flow
        flow = await flow_manager.create_flow(
            flow_id=request.flow_id,
            flow_config=flow_config,
            model_id=request.model_id,
            message=request.message,
            trajectory_id=request.trajectory_id,
            project_id=request.project_id,
            metadata=request.metadata,
        )

        # Return trajectory response
        return flow
    except ValueError as e:
        logger.exception(f"Error creating trajectory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error creating trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
