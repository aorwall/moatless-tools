import os
import logging

from dotenv import load_dotenv

from moatless.eventbus.local_bus import LocalEventBus
from moatless.runner.asyncio_runner import AsyncioRunner
from moatless.storage.file_storage import FileStorage


load_dotenv()


logger = logging.getLogger(__name__)

storage = FileStorage()


if os.environ.get("REDIS_URL"):
    logger.info(f"Use RQ Runner and Redis Event Bus with redis url: {os.environ.get('REDIS_URL')}")
    try:
        from moatless.eventbus.redis_bus import RedisEventBus
        from moatless.runner.rq import RQRunner

        event_bus = RedisEventBus(redis_url=os.environ.get("REDIS_URL"), storage=storage)
        runner = RQRunner(redis_url=os.environ.get("REDIS_URL"))
    except Exception as e:
        logger.error(f"Failed to initialize event bus and runner: {e}")
        raise e
else:
    logger.info("Use Local Runner and Local Event Bus")
    event_bus = LocalEventBus()
    runner = AsyncioRunner()
