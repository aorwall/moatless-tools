import logging
from dotenv import load_dotenv
import moatless.api.api
import uvicorn


def run_api():
    """Run the Moatless API server"""
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Create and run API without workspace
    api = moatless.api.api.create_api()
    logger.info("Starting API server")
    uvicorn.run(api, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_api()
