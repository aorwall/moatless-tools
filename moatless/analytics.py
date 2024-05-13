import logging
import os

from moatless.session import Session
from moatless.settings import Settings

_posthog = None


logger = logging.getLogger(__name__)

def send_event(event: str, properties: dict = None, session_id: str = None):

    try:
        session_id = session_id or Session.session_id

        if Session.tags:
            properties = properties or {}
            properties["tags"] = Session.tags[0] if len(Session.tags) == 1 else Session.tags

        if Settings.analytics_file:
            with open(Settings.analytics_file, "a") as f:
                log = {
                    "session_id": session_id,
                    "event": event,
                    "tags": Session.tags,
                    "properties": properties,
                }
                f.write(f"{log}\n")

        global _posthog

        if not os.getenv("POSTHOG_API_KEY"):
            return

        if _posthog is None:
            try:
                from posthog import Posthog

                _posthog = Posthog(
                    project_api_key=os.getenv("POSTHOG_API_KEY"),
                    host="https://eu.posthog.com",
                )
            except ImportError:
                print("Posthog not installed. Skipping event tracking.")
                return

        _posthog.capture(distinct_id=session_id, event=event, properties=properties)

    except Exception as e:
        logger.exception(f"Error occurred in sending event: {e}")