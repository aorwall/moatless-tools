import logging
import os

logger = logging.getLogger(__name__)


def save_to_json(files, filename):
    # Get the absolute path of the file
    full_path = os.path.abspath(filename)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        with open(full_path, "w") as f:
            try:
                import json5

                json5.dump(files, f, indent=2, quote_keys=True, trailing_commas=False)
            except ImportError:
                import json

                json.dump(files, f, indent=2, ensure_ascii=False)

        logger.info(f"LLM output saved to {full_path}")
    except Exception as e:
        logger.error(f"Error saving interactions to JSON: {e}")
        logger.error(f"Attempted to save to: {full_path}")
