import os

from moatless.settings import Settings

_posthog = None


def send_event(event: str, properties: dict = None):
    session_id = "test"  # FIXME
    if Settings.analytics_file:
        with open(Settings.analytics_file, "a") as f:
            log = {
                "session_id": session_id,
                "event": event,
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
