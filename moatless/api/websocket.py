"""WebSocket module for Moatless API."""

import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect
from moatless.events import BaseEvent

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # Maps for subscriptions
        self.project_subscriptions: Dict[str, Set[WebSocket]] = {}
        self.trajectory_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Maps each socket to its subscriptions for cleanup
        self.socket_subscriptions: Dict[WebSocket, Dict[str, Set[str]]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        try:
            logger.debug("Accepting WebSocket connection")
            await websocket.accept()
            self.active_connections.add(websocket)
            self.socket_subscriptions[websocket] = {"projects": set(), "trajectories": set()}
            logger.debug(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            raise

    async def disconnect(self, websocket: WebSocket):
        """Safely disconnect a WebSocket connection and clean up subscriptions."""
        try:
            self.active_connections.discard(websocket)

            # Clean up subscriptions
            if websocket in self.socket_subscriptions:
                for project_id in self.socket_subscriptions[websocket]["projects"]:
                    if project_id in self.project_subscriptions:
                        self.project_subscriptions[project_id].discard(websocket)
                        if not self.project_subscriptions[project_id]:
                            del self.project_subscriptions[project_id]

                for trajectory_id in self.socket_subscriptions[websocket]["trajectories"]:
                    if trajectory_id in self.trajectory_subscriptions:
                        self.trajectory_subscriptions[trajectory_id].discard(websocket)
                        if not self.trajectory_subscriptions[trajectory_id]:
                            del self.trajectory_subscriptions[trajectory_id]

                del self.socket_subscriptions[websocket]

            if not websocket.client_state.DISCONNECTED:
                await websocket.close()
            logger.debug(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")

    async def subscribe_to_project(self, websocket: WebSocket, project_id: str):
        """Subscribe a connection to a specific project."""
        if project_id not in self.project_subscriptions:
            self.project_subscriptions[project_id] = set()

        self.project_subscriptions[project_id].add(websocket)
        self.socket_subscriptions[websocket]["projects"].add(project_id)
        logger.info(f"WebSocket subscribed to project: {project_id}")

        # Acknowledge subscription
        await websocket.send_json({"type": "subscription_ack", "subscription": "project", "id": project_id})

    async def subscribe_to_trajectory(self, websocket: WebSocket, project_id: str, trajectory_id: str):
        """Subscribe a connection to a specific trajectory."""
        # TODO: Support for project_id
        if trajectory_id not in self.trajectory_subscriptions:
            self.trajectory_subscriptions[trajectory_id] = set()

        self.trajectory_subscriptions[trajectory_id].add(websocket)
        self.socket_subscriptions[websocket]["trajectories"].add(trajectory_id)
        logger.info(f"WebSocket subscribed to trajectory: {trajectory_id}")

        # Acknowledge subscription
        await websocket.send_json({"type": "subscription_ack", "subscription": "trajectory", "id": trajectory_id})

    async def unsubscribe_from_project(self, websocket: WebSocket, project_id: str):
        """Unsubscribe a connection from a specific project."""
        if project_id in self.project_subscriptions:
            self.project_subscriptions[project_id].discard(websocket)
            if not self.project_subscriptions[project_id]:
                del self.project_subscriptions[project_id]

        if websocket in self.socket_subscriptions:
            self.socket_subscriptions[websocket]["projects"].discard(project_id)

        logger.info(f"WebSocket unsubscribed from project: {project_id}")

        # Acknowledge unsubscription
        await websocket.send_json({"type": "unsubscription_ack", "subscription": "project", "id": project_id})

    async def unsubscribe_from_trajectory(self, websocket: WebSocket, project_id: str, trajectory_id: str):
        """Unsubscribe a connection from a specific trajectory."""
        if trajectory_id in self.trajectory_subscriptions:
            self.trajectory_subscriptions[trajectory_id].discard(websocket)
            if not self.trajectory_subscriptions[trajectory_id]:
                del self.trajectory_subscriptions[trajectory_id]

        if websocket in self.socket_subscriptions:
            self.socket_subscriptions[websocket]["trajectories"].discard(trajectory_id)

        logger.info(f"WebSocket unsubscribed from trajectory: {trajectory_id}")

        # Acknowledge unsubscription
        await websocket.send_json({"type": "unsubscription_ack", "subscription": "trajectory", "id": trajectory_id})

    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            logger.debug("No active connections, skipping broadcast")
            return

        logger.debug(f"Broadcasting message to {len(self.active_connections)} clients")

        connections = self.active_connections.copy()
        disconnected = set()

        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.add(connection)

        for connection in disconnected:
            await self.disconnect(connection)

    async def broadcast_event(self, event: BaseEvent):
        """Handle system events and broadcast them to subscribers or all clients."""
        logger.info(f"Broadcasting event: {event.scope}:{event.event_type}")

        event_dict = event.model_dump(mode="json")

        project_id = event_dict.get("project_id")
        trajectory_id = event_dict.get("trajectory_id")

        # Track connections that received the message via specific subscription
        notified_connections = set()

        # Only send flow and evaluation events to project subscribers
        if event_dict.get("scope") == "flow" or event_dict.get("scope") == "evaluation":
            if project_id and project_id in self.project_subscriptions:
                logger.info(f"Broadcasting event to project subscribers: {project_id}")
                for connection in self.project_subscriptions[project_id].copy():
                    # Skip if already notified via trajectory subscription
                    if connection in notified_connections:
                        continue

                    try:
                        await connection.send_text(json.dumps(event_dict))
                        notified_connections.add(connection)
                    except Exception as e:
                        logger.error(f"Failed to send message to project subscriber: {e}")
                        await self.disconnect(connection)

        if trajectory_id and trajectory_id in self.trajectory_subscriptions:
            logger.info(f"Broadcasting event to trajectory subscribers: {trajectory_id}")
            for connection in self.trajectory_subscriptions[trajectory_id].copy():
                if connection in notified_connections:
                    continue

                try:
                    await connection.send_text(json.dumps(event_dict))
                    notified_connections.add(connection)
                except Exception as e:
                    logger.error(f"Failed to send message to trajectory subscriber: {e}")
                    await self.disconnect(connection)

    async def handle_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket messages."""
        try:
            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif message_type == "subscribe" or message_type == "unsubscribe":
                subscription_type = data.get("subscription")

                if not subscription_type:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid subscription request. Missing subscription type."}
                    )
                    return

                project_id = data.get("project_id")
                if not project_id:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid subscription request. Missing project ID."}
                    )
                    return

                trajectory_id = data.get("trajectory_id")

                if subscription_type == "project":
                    if not project_id:
                        await websocket.send_json(
                            {"type": "error", "message": "Invalid subscription request. Missing project ID."}
                        )
                        return
                    await self.subscribe_to_project(websocket, project_id)

                elif subscription_type == "trajectory":
                    if not trajectory_id:
                        await websocket.send_json(
                            {"type": "error", "message": "Invalid subscription request. Missing trajectory ID."}
                        )
                        return
                    await self.subscribe_to_trajectory(websocket, project_id, trajectory_id)

                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Invalid subscription type: {subscription_type}"}
                    )

            elif message_type == "unsubscribe":
                subscription_type = data.get("subscription")

                if not subscription_type:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid unsubscription request. Missing subscription type."}
                    )
                    return

                project_id = data.get("project_id")
                if not project_id:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid unsubscription request. Missing project ID."}
                    )
                    return

                if subscription_type == "project":
                    await self.unsubscribe_from_project(websocket, project_id)

                elif subscription_type == "trajectory":
                    trajectory_id = data.get("trajectory_id")
                    if not trajectory_id:
                        await websocket.send_json(
                            {"type": "error", "message": "Invalid unsubscription request. Missing trajectory ID."}
                        )
                        return
                    await self.unsubscribe_from_trajectory(websocket, project_id, trajectory_id)

                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Invalid subscription type: {subscription_type}"}
                    )

            else:
                logger.debug(f"Received unsupported message type: {message_type}")
                await websocket.send_json({"type": "error", "message": f"Unsupported message type: {message_type}"})

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            try:
                await websocket.send_json({"type": "error", "message": "Failed to process message"})
            except:
                logger.error("Failed to send error response")


# Create a singleton instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint handler."""
    try:
        await manager.connect(websocket)

        while True:
            try:
                # Receive and process messages
                message = await websocket.receive_text()
                data = json.loads(message)
                await manager.handle_message(websocket, data)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                break
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection: {e}")
        raise
    finally:
        await manager.disconnect(websocket)


async def handle_event(event: BaseEvent):
    """Handle system events and broadcast them via WebSocket"""
    await manager.broadcast_event(event)
